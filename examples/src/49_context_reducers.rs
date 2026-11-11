//! Context reducers: build a strategy from small parts (Rust-only, offline)
//!
//! A `ContextReducer` turns a history into a shorter one and reports what it did. The ready-made
//! ones fall into three families, and `ChainReducer` runs any mix of them in order:
//!
//! * **Rewrite**: `ToolOutputTrimmer` shortens bulky tool outputs of older exchanges.
//! * **Drop**: `ContextWindowTrimmer` (token / exchange budget) and `HeadTail` (first and latest
//!   exchanges) keep whole exchanges only. Your own rule is one small `ExchangeSelector`;
//!   `DropExchanges` makes it a reducer and keeps tool calls with their outputs, pinned messages
//!   and the latest exchange for you.
//! * **Summarize**: `Summarize` turns any `Summarizer` into a reducer. It costs a model call
//!   whenever it has something to summarize, so put it last in a chain, after the free ones.
//!
//! Where a reducer runs is a separate choice:
//!
//! * `reducer_filter(reducer)`: rewrites the input of every model call, history untouched.
//! * `CompactingSession::with_reducer(store, reducer)`: rewrites the stored history once it
//!   passes a trigger.
//!
//! This example needs no key: the history is made up and the summarizer is a stub.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example context_reducers
//! ```

use std::sync::Arc;

use opai_agents::{
    async_trait, default_token_counter, reducer_filter, AgentsError, ChainReducer, ContextReducer,
    DropExchanges, ExchangeSelector, HeadTail, InMemorySession, ReduceInput, ReduceReport,
    SelectionView, Session, Summarize, Summarizer, ToolOutputTrimmer,
};
use serde_json::{json, Value};

/// A made-up conversation of six exchanges; every other one has a bulky tool output.
fn history() -> Vec<Value> {
    let mut items = vec![json!({"role": "system", "content": "Answer briefly."})];
    for i in 0..6 {
        items.push(json!({"role": "user", "content": format!("question {i}")}));
        items.push(json!({"type": "function_call", "call_id": format!("c{i}"),
                          "name": "search", "arguments": "{}"}));
        items.push(
            json!({"type": "function_call_output", "call_id": format!("c{i}"),
                          "output": "result ".repeat(if i % 2 == 0 { 300 } else { 5 })}),
        );
        items.push(json!({"role": "assistant", "content": format!("answer {i}")}));
    }
    items
}

/// Your own drop rule: keep only the cheap exchanges. The
/// skeleton still keeps the latest exchange and never separates a call from its output.
struct KeepSmallOnes {
    max_tokens: usize,
}

impl ExchangeSelector for KeepSmallOnes {
    fn name(&self) -> &str {
        "keep_small_ones"
    }

    fn select(&self, view: &SelectionView<'_>) -> Vec<usize> {
        (0..view.exchanges.len())
            .filter(|&i| view.costs[i] <= self.max_tokens)
            .collect()
    }
}

/// Stands in for `ModelSummarizer`, which would ask a model to write the notes.
struct StubSummarizer;

#[async_trait]
impl Summarizer for StubSummarizer {
    async fn summarize(
        &self,
        _previous: Option<&str>,
        items: &[Value],
    ) -> Result<String, AgentsError> {
        Ok(format!("{} earlier items were condensed.", items.len()))
    }
}

fn show(label: &str, items: &[Value], report: &ReduceReport) {
    println!(
        "{label:<22} items {:>2}  tokens {:>5} -> {:>5}  dropped exchanges {}  summarized {}",
        items.len(),
        report.tokens_before,
        report.tokens_after,
        report.dropped_exchanges,
        report.summarized,
    );
}

async fn run(
    label: &str,
    reducer: &dyn ContextReducer,
    items: &[Value],
) -> Result<Vec<Value>, AgentsError> {
    let counter = default_token_counter();
    let outcome = reducer
        .reduce(ReduceInput {
            items,
            reserved_tokens: 0,
            counter: &counter,
        })
        .await?;
    show(label, &outcome.items, &outcome.report);
    Ok(outcome.items)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let items = history();
    println!("input: {} items\n", items.len());

    // One reducer at a time.
    run("tool output trimmer", &ToolOutputTrimmer::new(), &items).await?;
    run("head 1 + tail 2", &HeadTail::new(1, 2), &items).await?;
    run(
        "your own selector",
        &DropExchanges::new(KeepSmallOnes { max_tokens: 100 }),
        &items,
    )
    .await?;
    run(
        "summarize",
        &Summarize::new(Arc::new(StubSummarizer)).keep_recent_exchanges(2),
        &items,
    )
    .await?;

    // A chain: the free steps first, so the paid one has less to read.
    let chain = ChainReducer::new(vec![
        Arc::new(ToolOutputTrimmer::new()),
        Arc::new(HeadTail::new(1, 3)),
        Arc::new(Summarize::new(Arc::new(StubSummarizer)).keep_recent_exchanges(2)),
    ]);
    println!();
    run("chain of all three", &chain, &items).await?;

    // Mounted per model call: pass this as `RunConfig.model.input_filter`. The stored history
    // is not changed.
    let _filter = reducer_filter(Arc::new(HeadTail::new(1, 3)));

    // Mounted on the stored history: once it passes the trigger, the store is rewritten.
    let store = InMemorySession::shared("reducers-demo");
    let session =
        opai_agents::CompactingSession::with_reducer(store.clone(), Arc::new(HeadTail::new(1, 2)))
            .trigger_items(10)
            .shared();
    session.add_items(items).await?;
    println!(
        "\nstored history after the session wrote it: {} items",
        store.get_items(None).await?.len()
    );
    Ok(())
}
