//! Multi-agent handoff
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example handoff
//! ```

mod common;

use std::sync::Arc;

use opai_agents::{handoff, Agent, RunOptions, Runner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = common::model()?;

    // The specialist. `handoff_description` is what the *triage* model reads to decide when to
    // hand off to it, so describe the work it takes, not how it behaves.
    let billing = Agent::new("Billing")
        .handoff_description("Handles billing and invoice questions.")
        .instructions("You answer billing questions briefly and clearly.")
        .model(Arc::clone(&model));

    let triage = Agent::new("Triage")
        .instructions(
            "You are a triage agent. For billing or invoice questions, hand off to Billing. \
             Otherwise answer briefly yourself.",
        )
        .model(model)
        // `handoff(agent)` exposes the agent to the model as a tool named `transfer_to_billing`.
        // Calling it transfers the conversation: Billing answers the user directly from then on.
        // (Contrast with `agents_as_tools`, where the caller keeps control and gets a result back.)
        .handoffs(vec![handoff(billing)]);

    let input = common::example_input("What do I owe on my last invoice?");
    let result = Runner::run(&triage, input, RunOptions::default()).await?;
    // After a handoff, the agent that produced the answer is not the one you started with.
    println!("last_agent={}", result.last_agent_name);
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
