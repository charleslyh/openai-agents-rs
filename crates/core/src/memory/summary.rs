//! Summarizing as a [`ContextReducer`], and summarizers that build on other summarizers.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use super::compaction::Summarizer;
use crate::context::{is_summary_item, summary_item, summary_text, ExchangeView};
use crate::error::{AgentsError, ModelError};
use crate::reducer::{ContextReducer, ReduceInput, ReduceOutcome};
use crate::usage::Usage;

/// Replace all but the `keep_recent` latest exchanges by one summary item written by
/// `summarizer`. Returns the new history and how many exchanges were summarized, or `None` when
/// there are not more exchanges than `keep_recent`.
///
/// Earlier summaries are carried into the new one, and the pinned system / developer messages
/// stay in front.
pub(crate) async fn summarize_history(
    items: &[Value],
    summarizer: &dyn Summarizer,
    keep_recent: usize,
    timeout: Option<Duration>,
) -> Result<Option<(Vec<Value>, usize)>, AgentsError> {
    let ExchangeView { pinned, exchanges } = ExchangeView::split(items);
    if exchanges.len() <= keep_recent {
        return Ok(None);
    }
    let (old, recent) = exchanges.split_at(exchanges.len() - keep_recent);

    let earlier: Vec<String> = pinned.iter().filter_map(summary_text).collect();
    let previous = (!earlier.is_empty()).then(|| earlier.join("\n\n"));
    let old_items: Vec<Value> = old.iter().flatten().cloned().collect();
    let call = summarizer.summarize(previous.as_deref(), &old_items);
    let notes = match timeout {
        Some(limit) => tokio::time::timeout(limit, call).await.map_err(|_| {
            AgentsError::from(ModelError::Behavior(format!(
                "the summarizer did not answer within {limit:?}"
            )))
        })??,
        None => call.await?,
    };

    let mut compacted: Vec<Value> = pinned
        .into_iter()
        .filter(|item| !is_summary_item(item))
        .collect();
    compacted.push(summary_item(&notes));
    compacted.extend(recent.iter().flatten().cloned());
    Ok(Some((compacted, old.len())))
}

/// Summarizes old exchanges and keeps the most recent ones word for word.
///
/// Every run of this reducer that finds something to summarize costs the summarizer's model call.
/// Mounted as a model input filter that happens before each model call; on a
/// [`CompactingSession`](super::CompactingSession) it happens when the history grows past the
/// trigger.
pub struct Summarize {
    summarizer: Arc<dyn Summarizer>,
    keep_recent_exchanges: usize,
}

impl Summarize {
    /// Summarize with `summarizer`, keeping the 2 latest exchanges verbatim.
    pub fn new(summarizer: Arc<dyn Summarizer>) -> Self {
        Self {
            summarizer,
            keep_recent_exchanges: 2,
        }
    }

    /// Number of latest exchanges kept word for word (at least 1).
    pub fn keep_recent_exchanges(mut self, exchanges: usize) -> Self {
        self.keep_recent_exchanges = exchanges.max(1);
        self
    }

    /// Usage of the summaries written so far (see [`Summarizer::usage`]).
    pub fn usage(&self) -> Usage {
        self.summarizer.usage()
    }
}

#[async_trait]
impl ContextReducer for Summarize {
    fn name(&self) -> &str {
        "summarize"
    }

    async fn reduce(&self, input: ReduceInput<'_>) -> Result<ReduceOutcome, AgentsError> {
        match summarize_history(
            input.items,
            self.summarizer.as_ref(),
            self.keep_recent_exchanges,
            None,
        )
        .await?
        {
            Some((items, summarized)) => Ok(ReduceOutcome::new(&input, items)
                .dropped_exchanges(summarized)
                .summarized(true)),
            None => Ok(ReduceOutcome::unchanged(&input)),
        }
    }
}

/// A [`Summarizer`] that condenses its inner summarizer's notes again when they grow too long.
///
/// Notes that carry forward across many compactions only ever grow. When the result is longer
/// than `max_chars`, the inner summarizer is asked to condense the notes (up to `max_rounds`
/// times, stopping early once they fit or stop shrinking).
pub struct HierarchicalSummarizer {
    inner: Arc<dyn Summarizer>,
    max_chars: usize,
    max_rounds: usize,
}

impl HierarchicalSummarizer {
    /// Condense notes longer than `max_chars` characters, in at most 2 extra rounds.
    pub fn new(inner: Arc<dyn Summarizer>, max_chars: usize) -> Self {
        Self {
            inner,
            max_chars,
            max_rounds: 2,
        }
    }

    /// Most extra condensing rounds per summary.
    pub fn max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds;
        self
    }
}

#[async_trait]
impl Summarizer for HierarchicalSummarizer {
    async fn summarize(
        &self,
        previous_summary: Option<&str>,
        items: &[Value],
    ) -> Result<String, AgentsError> {
        let mut notes = self.inner.summarize(previous_summary, items).await?;
        for _ in 0..self.max_rounds {
            let length = notes.chars().count();
            if length <= self.max_chars {
                break;
            }
            let as_item = serde_json::json!({"role": "user", "content": notes});
            // A failed extra round is not fatal: the longer notes are still correct.
            let Ok(shorter) = self.inner.summarize(None, &[as_item]).await else {
                break;
            };
            if shorter.chars().count() >= length {
                break;
            }
            notes = shorter;
        }
        Ok(notes)
    }

    fn usage(&self) -> Usage {
        self.inner.usage()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use serde_json::json;

    use super::*;
    use crate::context::default_token_counter;

    /// Returns each canned answer in turn and records its inputs.
    struct Canned {
        answers: Mutex<Vec<String>>,
        calls: Mutex<Vec<(Option<String>, usize)>>,
    }

    impl Canned {
        fn new(answers: &[&str]) -> Arc<Self> {
            Arc::new(Self {
                answers: Mutex::new(answers.iter().rev().map(|a| a.to_string()).collect()),
                calls: Mutex::default(),
            })
        }
    }

    #[async_trait]
    impl Summarizer for Canned {
        async fn summarize(
            &self,
            previous: Option<&str>,
            items: &[Value],
        ) -> Result<String, AgentsError> {
            self.calls
                .lock()
                .unwrap()
                .push((previous.map(str::to_string), items.len()));
            self.answers
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| AgentsError::internal("out of answers"))
        }
    }

    fn history() -> Vec<Value> {
        let mut items = vec![json!({"role": "system", "content": "s"})];
        for i in 0..5 {
            items.push(json!({"role": "user", "content": format!("q{i}")}));
            items.push(json!({"role": "assistant", "content": format!("a{i}")}));
        }
        items
    }

    #[tokio::test]
    async fn summarize_replaces_old_exchanges_and_keeps_recent_ones() {
        let counter = default_token_counter();
        let items = history();
        let canned = Canned::new(&["notes"]);
        let outcome = Summarize::new(canned.clone())
            .keep_recent_exchanges(2)
            .reduce(ReduceInput {
                items: &items,
                reserved_tokens: 0,
                counter: &counter,
            })
            .await
            .unwrap();
        assert!(outcome.report.summarized);
        assert_eq!(outcome.report.dropped_exchanges, 3);
        assert_eq!(outcome.items.len(), 1 + 1 + 4);
        assert!(is_summary_item(&outcome.items[1]));
        assert_eq!(outcome.items[2]["content"], "q3");
        assert_eq!(canned.calls.lock().unwrap()[0], (None, 6));
    }

    #[tokio::test]
    async fn summarize_carries_the_previous_summary_forward() {
        let counter = default_token_counter();
        let mut items = history();
        items.insert(1, summary_item("old notes"));
        let canned = Canned::new(&["merged"]);
        let outcome = Summarize::new(canned.clone())
            .reduce(ReduceInput {
                items: &items,
                reserved_tokens: 0,
                counter: &counter,
            })
            .await
            .unwrap();
        assert_eq!(
            canned.calls.lock().unwrap()[0].0.as_deref(),
            Some("old notes")
        );
        assert_eq!(
            outcome.items.iter().filter(|i| is_summary_item(i)).count(),
            1
        );
    }

    #[tokio::test]
    async fn summarize_does_nothing_without_enough_exchanges() {
        let counter = default_token_counter();
        let items = history();
        let canned = Canned::new(&[]);
        let outcome = Summarize::new(canned.clone())
            .keep_recent_exchanges(5)
            .reduce(ReduceInput {
                items: &items,
                reserved_tokens: 0,
                counter: &counter,
            })
            .await
            .unwrap();
        assert!(!outcome.report.changed);
        assert!(canned.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn hierarchical_condenses_long_notes_until_they_fit() {
        let long = "x".repeat(100);
        let medium = "y".repeat(50);
        let short = "z".repeat(10);
        let canned = Canned::new(&[&long, &medium, &short]);
        let summarizer = HierarchicalSummarizer::new(canned.clone(), 20);
        let notes = summarizer.summarize(None, &[]).await.unwrap();
        assert_eq!(notes, short);
        assert_eq!(canned.calls.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn hierarchical_stops_when_notes_do_not_shrink_or_a_round_fails() {
        let long = "x".repeat(100);
        let canned = Canned::new(&[&long, &"y".repeat(120)]);
        let notes = HierarchicalSummarizer::new(canned.clone(), 20)
            .summarize(None, &[])
            .await
            .unwrap();
        assert_eq!(notes, long, "a longer answer is discarded");

        let canned = Canned::new(&[&long]);
        let notes = HierarchicalSummarizer::new(canned, 20)
            .summarize(None, &[])
            .await
            .unwrap();
        assert_eq!(notes, long, "a failed round keeps the notes");
    }

    #[tokio::test]
    async fn hierarchical_leaves_short_notes_alone() {
        let canned = Canned::new(&["short"]);
        let notes = HierarchicalSummarizer::new(canned.clone(), 20)
            .summarize(None, &[])
            .await
            .unwrap();
        assert_eq!(notes, "short");
        assert_eq!(canned.calls.lock().unwrap().len(), 1);
    }
}
