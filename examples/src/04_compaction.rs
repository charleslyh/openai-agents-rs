//! Context compaction
//!
//! Web demo checklist: "Context compaction".
//!
//! `CompactingSession` wraps a session store. Once the stored history passes a trigger, old
//! turns are replaced by a model-written summary; the most recent turns stay word for word.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p openai-agents-examples --example compaction
//! ```

mod common;

use std::sync::Arc;

use openai_agents::{
    Agent, CompactingSession, InMemorySession, ModelSummarizer, RunOptions, Runner, Session,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = common::model()?;
    let agent = Agent::new("Assistant")
        .instructions("Answer in one short sentence.")
        .model(Arc::clone(&model));

    let store = InMemorySession::shared("compaction-demo");
    // Kept low so the compaction shows up within a few turns.
    let session: Arc<dyn Session> =
        CompactingSession::new(store.clone(), Arc::new(ModelSummarizer::new(model)))
            .trigger_items(6)
            .keep_recent_turns(1)
            .shared();

    for input in [
        "Name one famous lighthouse.",
        "Name one famous bridge.",
        "Name one famous tower.",
        "What did we talk about?",
    ] {
        let mut options = RunOptions::default();
        options.session = Some(Arc::clone(&session));
        let result = Runner::run(&agent, input, options).await?;
        println!("user> {input}");
        println!(
            "agent> {}",
            result.final_output_as_str().unwrap_or_default()
        );

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
