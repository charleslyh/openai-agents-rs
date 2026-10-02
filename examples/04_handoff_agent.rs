//! 04 — Multi-agent handoff
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run --example 04_handoff_agent
//! ```

#[path = "common/mod.rs"]
mod common;

use std::sync::Arc;

use openai_agents::{handoff, Agent, RunOptions, Runner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = common::model()?;

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
        .handoffs(vec![handoff(billing)]);

    let input = common::input_with_fallback(
        "User: ",
        "What do I owe on my last invoice?",
    )?;
    let result = Runner::run(&triage, input, RunOptions::default()).await?;
    println!("last_agent={}", result.last_agent_name);
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
