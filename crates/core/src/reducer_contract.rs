//! The behaviour every [`ContextReducer`] must have, as a test suite any reducer can run.
//!
//! A reducer outside this workspace can run it too, so a custom strategy is held to the same rules
//! as the bundled ones: it never separates a tool call from its output, never drops system messages or the summary (a summarizer may merge it into a new one),
//! keeps the latest exchange, never makes the history longer and gives the same answer when run on
//! its own output.
//!
//! Behind the `reducer-contract` feature of `opai-agents-core`, which is off by default.
//!
//! ```rust,no_run
//! use opai_agents_core::{reducer_contract, ContextReducer};
//!
//! # async fn check(reducer: &dyn ContextReducer) {
//! reducer_contract::assert_reducer_contract(reducer).await;
//! # }
//! ```

use std::collections::HashSet;

use serde_json::{json, Value};

use crate::context::{
    default_token_counter, estimate_item_tokens, is_summary_item, summary_item, TokenCounter,
};
use crate::reducer::{ContextReducer, ReduceInput, ReduceReport};

/// Run every contract check against `reducer`, panicking with the first failure.
///
/// Configure the reducer so that it has something to do on a history of eight exchanges, each with
/// a tool call whose output is about 2000 characters; the checks are only meaningful when the
/// history changes. Returns the report of that reduction so the caller can assert that it did.
pub async fn assert_reducer_contract(reducer: &dyn ContextReducer) -> ReduceReport {
    let counter = default_token_counter();
    let history = sample_history();
    let report = assert_reduction_rules(reducer, &history, &counter).await;
    assert_idempotent(reducer, &history, &counter).await;
    assert_degenerate_histories(reducer, &counter).await;
    report
}

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

fn call(id: &str) -> Value {
    json!({"type": "function_call", "call_id": id, "name": "lookup", "arguments": "{}"})
}

fn output(id: &str) -> Value {
    json!({"type": "function_call_output", "call_id": id, "output": "r".repeat(2_000)})
}

fn assistant(text: &str) -> Value {
    json!({"type": "message", "role": "assistant",
           "content": [{"type": "output_text", "text": text, "annotations": []}]})
}

fn system() -> Value {
    json!({"role": "system", "content": "be brief"})
}

fn notes() -> Value {
    summary_item("the user is planning a trip")
}

fn sample_history() -> Vec<Value> {
    let mut items = vec![system(), notes()];
    for i in 0..8 {
        items.push(user(&format!("question {i}")));
        items.push(call(&format!("c{i}")));
        items.push(output(&format!("c{i}")));
        items.push(assistant(&format!("answer {i}")));
    }
    items
}

fn input<'a>(items: &'a [Value], counter: &'a TokenCounter) -> ReduceInput<'a> {
    ReduceInput {
        items,
        reserved_tokens: 0,
        counter,
    }
}

fn ids(items: &[Value], kind: &str) -> HashSet<String> {
    items
        .iter()
        .filter(|item| item["type"] == kind)
        .filter_map(|item| item["call_id"].as_str().map(str::to_string))
        .collect()
}

async fn assert_reduction_rules(
    reducer: &dyn ContextReducer,
    history: &[Value],
    counter: &TokenCounter,
) -> ReduceReport {
    let outcome = reducer
        .reduce(input(history, counter))
        .await
        .expect("reduce a plain history");
    let kept = &outcome.items;

    assert_eq!(
        ids(kept, "function_call"),
        ids(kept, "function_call_output"),
        "a tool call and its output must stay together"
    );
    assert!(kept.contains(&system()), "the system message must stay");
    // A summarizing reducer may fold the earlier summary into a new one, but never lose it.
    assert!(
        kept.iter().any(is_summary_item),
        "the earlier summary must stay, or be replaced by a new summary"
    );
    let latest = history.len() - 4;
    assert!(
        kept.contains(&history[latest]) && kept.contains(&history[latest + 1]),
        "the latest exchange must stay"
    );
    assert!(
        kept.iter().any(|item| item["call_id"] == "c7"),
        "the latest exchange keeps its tool call and output"
    );

    let before: usize = history.iter().map(estimate_item_tokens).sum();
    let after: usize = kept.iter().map(estimate_item_tokens).sum();
    assert!(
        after <= before,
        "the history must not grow: {before} -> {after}"
    );

    let report = &outcome.report;
    assert_eq!(
        report.changed,
        kept.as_slice() != history,
        "report.changed must say whether the items changed"
    );
    assert_eq!(report.tokens_before, before, "report.tokens_before");
    assert_eq!(report.tokens_after, after, "report.tokens_after");
    outcome.report
}

async fn assert_idempotent(
    reducer: &dyn ContextReducer,
    history: &[Value],
    counter: &TokenCounter,
) {
    let once = reducer
        .reduce(input(history, counter))
        .await
        .expect("first reduce");
    let twice = reducer
        .reduce(input(&once.items, counter))
        .await
        .expect("second reduce");
    assert_eq!(
        once.items, twice.items,
        "reducing a reduced history must change nothing"
    );
    assert!(!twice.report.changed, "second reduce must report no change");
}

async fn assert_degenerate_histories(reducer: &dyn ContextReducer, counter: &TokenCounter) {
    let empty = reducer
        .reduce(input(&[], counter))
        .await
        .expect("reduce an empty history");
    assert!(empty.items.is_empty(), "an empty history stays empty");

    let no_user = vec![system(), assistant("hello"), call("x"), output("x")];
    let outcome = reducer
        .reduce(input(&no_user, counter))
        .await
        .expect("reduce a history without user messages");
    assert_eq!(
        ids(&outcome.items, "function_call"),
        ids(&outcome.items, "function_call_output"),
        "pairs stay together without user messages too"
    );
    assert!(
        outcome.items.contains(&system()),
        "pinned item kept without user messages"
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::context::{ContextWindowTrimmer, HeadTail, ToolOutputTrimmer};
    use crate::reducer::ChainReducer;

    #[tokio::test]
    async fn context_window_trimmer_meets_the_contract() {
        let reducer = ContextWindowTrimmer::new().max_exchanges(3);
        assert!(assert_reducer_contract(&reducer).await.changed);
    }

    #[tokio::test]
    async fn head_tail_meets_the_contract() {
        let reducer = HeadTail::new(1, 2);
        assert!(assert_reducer_contract(&reducer).await.changed);
    }

    #[tokio::test]
    async fn tool_output_trimmer_meets_the_contract() {
        let reducer = ToolOutputTrimmer::new()
            .recent_exchanges(2)
            .max_output_chars(100);
        assert!(assert_reducer_contract(&reducer).await.changed);
    }

    /// A selector that asks to keep nothing is still held to the rules by the skeleton.
    struct KeepNothing;

    impl crate::select::ExchangeSelector for KeepNothing {
        fn name(&self) -> &str {
            "keep_nothing"
        }

        fn select(&self, _: &crate::select::SelectionView<'_>) -> Vec<usize> {
            Vec::new()
        }
    }

    #[tokio::test]
    async fn a_custom_selector_meets_the_contract_without_effort() {
        let reducer = crate::select::DropExchanges::new(KeepNothing);
        assert!(assert_reducer_contract(&reducer).await.changed);
    }

    struct Fixed;

    #[async_trait::async_trait]
    impl crate::memory::Summarizer for Fixed {
        async fn summarize(
            &self,
            _: Option<&str>,
            _: &[Value],
        ) -> Result<String, crate::error::AgentsError> {
            Ok("short notes".into())
        }
    }

    #[tokio::test]
    async fn summarize_meets_the_contract() {
        let reducer = crate::memory::Summarize::new(Arc::new(Fixed)).keep_recent_exchanges(2);
        let report = assert_reducer_contract(&reducer).await;
        assert!(report.changed && report.summarized);
    }

    #[tokio::test]
    async fn a_chain_meets_the_contract() {
        let reducer = ChainReducer::new(vec![
            Arc::new(
                ToolOutputTrimmer::new()
                    .max_output_chars(300)
                    .preview_chars(50),
            ),
            Arc::new(HeadTail::new(1, 2)),
        ]);
        assert!(assert_reducer_contract(&reducer).await.changed);
    }

    /// A reducer that breaks a rule must fail the suite.
    struct DropsOutputs;

    #[async_trait::async_trait]
    impl ContextReducer for DropsOutputs {
        fn name(&self) -> &str {
            "drops_outputs"
        }

        async fn reduce(
            &self,
            input: ReduceInput<'_>,
        ) -> Result<crate::reducer::ReduceOutcome, crate::error::AgentsError> {
            let kept = input
                .items
                .iter()
                .filter(|item| item["type"] != "function_call_output")
                .cloned()
                .collect();
            Ok(crate::reducer::ReduceOutcome::new(&input, kept))
        }
    }

    #[tokio::test]
    #[should_panic(expected = "a tool call and its output must stay together")]
    async fn a_reducer_that_splits_a_pair_fails_the_contract() {
        assert_reducer_contract(&DropsOutputs).await;
    }
}
