//! Context compaction
//!
//! Web demo checklist: "Context compaction".
//!
//! `CompactingSession` wraps a session store. Once the stored history passes a trigger, old
//! exchanges are replaced by a model-written summary; the most recent exchanges stay word for word.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example compaction
//! ```

mod common;

use std::sync::Arc;

use opai_agents::{
    Agent, CompactingSession, InMemorySession, ModelSummarizer, RunOptions, Runner, Session,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The same model serves two roles here: it answers the user and it writes the summaries.
    // Use a cheaper model for `ModelSummarizer` if you like.
    let model = common::model()?;
    let agent = Agent::new("Assistant")
        .instructions("Answer in one short sentence.")
        .model(Arc::clone(&model));

    // `CompactingSession` wraps any other session (the `store`) and shrinks it: when the history
    // grows past a limit, older exchanges are replaced by a model-written summary. The agent code
    // does not change; it just receives a different `Session`.
    let store = InMemorySession::shared("compaction-demo");

    // Kept low so the compaction shows up within a few runs.
    let session: Arc<dyn Session> =
        CompactingSession::new(store.clone(), Arc::new(ModelSummarizer::new(model)))
            .trigger_items(6)
            // The newest exchange(s) stay verbatim, so the model keeps exact wording where it matters.
            .keep_recent_exchanges(1)
            .shared();

    for input in [
        "Name one famous lighthouse.",
        "Name one famous bridge.",
        "Name one famous tower.",
        "What did we talk about?",
    ] {
        let options = RunOptions::new().session(Arc::clone(&session));
        let result = Runner::run(&agent, input, options).await?;
        println!("user> {input}");
        println!(
            "agent> {}",
            result.final_output_as_str().unwrap_or_default()
        );

        // Read the wrapped store directly to see what is really kept. The summary is stored as
        // an ordinary message that starts with `<conversation_summary>`.
        let items = store.get_items(None).await?;
        let compacted = items.iter().any(|item| {
            item.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|c| c.starts_with("<conversation_summary>"))
        });
        println!(
            "stored items: {}{}",
            items.len(),
            if compacted { " (compacted)" } else { "" }
        );
    }
    Ok(())
}
