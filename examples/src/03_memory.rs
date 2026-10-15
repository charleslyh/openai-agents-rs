//! Multi-turn memory
//!
//! Web demo checklist: "Multi-turn memory".
//!
//! One `Session` carries the history between runs, so the agent remembers earlier turns.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p openai-agents-examples --example memory
//! ```

mod common;

use std::sync::Arc;

use openai_agents::{Agent, InMemorySession, RunOptions, Runner, Session};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Assistant")
        .instructions("Answer in one short sentence.")
        .model(common::model()?);

    let session: Arc<dyn Session> = InMemorySession::shared("memory-demo");

    for input in [
        "My favourite colour is teal. Remember that.",
        "What is my favourite colour?",
    ] {
        let mut options = RunOptions::default();
        options.session = Some(Arc::clone(&session));
        let result = Runner::run(&agent, input, options).await?;
        println!("user> {input}");
        println!(
            "agent> {}",
            result.final_output_as_str().unwrap_or_default()
        );
    }
    Ok(())
}
