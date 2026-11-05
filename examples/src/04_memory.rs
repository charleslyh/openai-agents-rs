//! Multi-turn memory
//!
//! Web demo checklist: "Multi-turn memory".
//!
//! One `Session` carries the history between runs, so the agent remembers earlier turns.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example memory
//! ```

mod common;

use std::sync::Arc;

use opai_agents::{Agent, InMemorySession, RunOptions, Runner, Session};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Assistant")
        .instructions("Answer in one short sentence.")
        .model(common::model()?);

    // A session stores the conversation. This one lives in memory, so it is gone when the program
    // exits. The agent itself stays stateless.
    let session: Arc<dyn Session> = InMemorySession::shared("memory-demo");

    for input in [
        "My favourite colour is teal. Remember that.",
        "What is my favourite colour?",
    ] {
        // Passing the same session to each run is all it takes: the run loads the earlier turns
        // into the model input and appends this turn when it finishes.
        let options = RunOptions::new().session(Arc::clone(&session));
        let result = Runner::run(&agent, input, options).await?;
        println!("user> {input}");
        println!(
            "agent> {}",
            result.final_output_as_str().unwrap_or_default()
        );
    }
    Ok(())
}
