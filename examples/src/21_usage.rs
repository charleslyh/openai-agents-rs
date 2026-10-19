//! Usage accounting
//!
//! Web demo checklist: "Usage accounting".
//!
//! `RunResult::usage` aggregates requests and tokens across the whole run. Token counts need
//! `ModelSettings::include_usage`, which is requested here.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p openai-agents-examples --example usage
//! ```

mod common;

use openai_agents::{Agent, RunOptions, Runner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Assistant")
        .instructions("Answer briefly.")
        .model(common::model()?);

    let options = RunOptions::new().tune(|s| s.include_usage = Some(true));

    let input = "Summarise what an agent SDK does.";
    let result = Runner::run(&agent, input, options).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());

    let usage = &result.usage;
    println!(
        "usage: {} requests, {} input tokens ({} cached), {} output tokens, {} total",
        usage.requests,
        usage.input_tokens,
        usage.input_tokens_details.cached_tokens,
        usage.output_tokens,
        usage.total_tokens
    );
    Ok(())
}
