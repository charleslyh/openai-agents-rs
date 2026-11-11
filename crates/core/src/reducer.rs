//! Pluggable context reduction strategies (provider-neutral).
//!
//! A [`ContextReducer`] turns a history into a shorter one: it drops old exchanges, shortens bulky
//! tool outputs, summarizes, or any mix. Reducers are plain building blocks; you assemble them
//! and decide where they run. A reducer that calls a model (for example a summarizing one) costs
//! a model call every time it runs, so mind where you mount it.
//!
//! Every reducer should keep a tool call together with its output, never drop pinned items (see
//! [`crate::is_pinned`]) and keep the latest exchange. [`crate::ExchangeView`] does the splitting.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::context::TokenCounter;
use crate::error::AgentsError;

/// What a reducer is given.
pub struct ReduceInput<'a> {
    /// The history to reduce, oldest first.
    pub items: &'a [Value],
    /// Tokens already used by instructions and tool schemas, which the result must leave room for.
    pub reserved_tokens: usize,
    /// Counts the tokens of one item.
    pub counter: &'a TokenCounter,
}

/// What a reduction did, for logs, traces and tests.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReduceReport {
    /// Estimated tokens of the input history.
    pub tokens_before: usize,
    /// Estimated tokens of the reduced history.
    pub tokens_after: usize,
    /// Whole exchanges that were removed or replaced by a summary.
    pub dropped_exchanges: usize,
    /// Whether a summary was written.
    pub summarized: bool,
    /// Whether the history differs from the input.
    pub changed: bool,
}

impl ReduceReport {
    /// A report for a reduction from `before` to `after`, with token counts taken by `counter`.
    pub fn measure(before: &[Value], after: &[Value], counter: &TokenCounter) -> Self {
        let count = |items: &[Value]| items.iter().map(|item| counter(item)).sum::<usize>();
        Self {
            tokens_before: count(before),
            tokens_after: count(after),
            changed: before != after,
            ..Self::default()
        }
    }

    /// Set the number of dropped exchanges.
    pub fn dropped_exchanges(mut self, exchanges: usize) -> Self {
        self.dropped_exchanges = exchanges;
        self
    }

    /// Mark that a summary was written.
    pub fn summarized(mut self, summarized: bool) -> Self {
        self.summarized = summarized;
        self
    }

    /// The report of `self` followed by `next`: before of the first, after of the last, the rest
    /// added up.
    pub fn then(&self, next: &ReduceReport) -> ReduceReport {
        ReduceReport {
            tokens_before: self.tokens_before,
            tokens_after: next.tokens_after,
            dropped_exchanges: self.dropped_exchanges + next.dropped_exchanges,
            summarized: self.summarized || next.summarized,
            changed: self.changed || next.changed,
        }
    }
}

/// The reduced history and what it took.
#[derive(Debug, Clone)]
pub struct ReduceOutcome {
    /// The new history.
    pub items: Vec<Value>,
    /// What changed.
    pub report: ReduceReport,
}

impl ReduceOutcome {
    /// An outcome for `after`, measured against the input of `input`.
    pub fn new(input: &ReduceInput<'_>, after: Vec<Value>) -> Self {
        let report = ReduceReport::measure(input.items, &after, input.counter);
        Self {
            items: after,
            report,
        }
    }

    /// An outcome that leaves the input as it is.
    pub fn unchanged(input: &ReduceInput<'_>) -> Self {
        Self::new(input, input.items.to_vec())
    }

    /// Set the report's dropped exchange count.
    pub fn dropped_exchanges(mut self, exchanges: usize) -> Self {
        self.report.dropped_exchanges = exchanges;
        self
    }

    /// Mark that a summary was written.
    pub fn summarized(mut self, summarized: bool) -> Self {
        self.report.summarized = summarized;
        self
    }
}

/// A strategy that shortens a history.
#[async_trait]
pub trait ContextReducer: Send + Sync {
    /// Short name, used in traces.
    fn name(&self) -> &str;

    /// Reduce `input.items`. Return the history unchanged when there is nothing to do.
    async fn reduce(&self, input: ReduceInput<'_>) -> Result<ReduceOutcome, AgentsError>;
}

/// Runs reducers one after another, each seeing the previous one's output.
///
/// The first error stops the chain. Every step runs inside its own `context_reduce` span.
pub struct ChainReducer {
    steps: Vec<Arc<dyn ContextReducer>>,
}

impl ChainReducer {
    /// A chain over `steps`, run in order. An empty chain changes nothing.
    pub fn new(steps: Vec<Arc<dyn ContextReducer>>) -> Self {
        Self { steps }
    }
}

#[async_trait]
impl ContextReducer for ChainReducer {
    fn name(&self) -> &str {
        "chain"
    }

    async fn reduce(&self, input: ReduceInput<'_>) -> Result<ReduceOutcome, AgentsError> {
        let mut outcome = ReduceOutcome::unchanged(&input);
        for step in &self.steps {
            let _span = crate::tracing::custom_span(&format!("context_reduce:{}", step.name()));
            let next = step
                .reduce(ReduceInput {
                    items: &outcome.items,
                    reserved_tokens: input.reserved_tokens,
                    counter: input.counter,
                })
                .await?;
            outcome = ReduceOutcome {
                report: outcome.report.then(&next.report),
                items: next.items,
            };
        }
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::default_token_counter;
    use serde_json::json;

    /// Keeps only the last `n` items.
    struct KeepLast(usize);

    #[async_trait]
    impl ContextReducer for KeepLast {
        fn name(&self) -> &str {
            "keep_last"
        }

        async fn reduce(&self, input: ReduceInput<'_>) -> Result<ReduceOutcome, AgentsError> {
            let from = input.items.len().saturating_sub(self.0);
            Ok(ReduceOutcome::new(&input, input.items[from..].to_vec()).dropped_exchanges(from))
        }
    }

    struct Fails;

    #[async_trait]
    impl ContextReducer for Fails {
        fn name(&self) -> &str {
            "fails"
        }

        async fn reduce(&self, _: ReduceInput<'_>) -> Result<ReduceOutcome, AgentsError> {
            Err(crate::error::UserError::new("boom").into())
        }
    }

    fn items(n: usize) -> Vec<Value> {
        (0..n)
            .map(|i| json!({"role": "user", "content": format!("message {i}")}))
            .collect()
    }

    #[tokio::test]
    async fn empty_chain_changes_nothing() {
        let counter = default_token_counter();
        let history = items(3);
        let outcome = ChainReducer::new(vec![])
            .reduce(ReduceInput {
                items: &history,
                reserved_tokens: 0,
                counter: &counter,
            })
            .await
            .unwrap();
        assert_eq!(outcome.items, history);
        assert!(!outcome.report.changed);
    }

    #[tokio::test]
    async fn chain_feeds_each_step_the_previous_output_and_sums_reports() {
        let counter = default_token_counter();
        let history = items(10);
        let chain = ChainReducer::new(vec![Arc::new(KeepLast(6)), Arc::new(KeepLast(3))]);
        let outcome = chain
            .reduce(ReduceInput {
                items: &history,
                reserved_tokens: 0,
                counter: &counter,
            })
            .await
            .unwrap();
        assert_eq!(outcome.items, history[7..].to_vec());
        let report = outcome.report;
        assert_eq!(report.dropped_exchanges, 4 + 3);
        assert!(report.changed);
        assert!(report.tokens_after < report.tokens_before);
    }

    #[tokio::test]
    async fn chain_stops_at_the_first_error() {
        let counter = default_token_counter();
        let history = items(2);
        let chain = ChainReducer::new(vec![Arc::new(Fails), Arc::new(KeepLast(1))]);
        let result = chain
            .reduce(ReduceInput {
                items: &history,
                reserved_tokens: 0,
                counter: &counter,
            })
            .await;
        assert!(result.is_err());
    }
}
