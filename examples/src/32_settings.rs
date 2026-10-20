//! Run settings
//!
//! Web demo checklist: "Run settings".
//!
//! The knobs of one run: `ModelSettings` (temperature, retry policy for transient model
//! errors) in `RunConfig`, and `RunOptions::max_turns`.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p openai-agents-examples --example settings
//! ```

mod common;

use openai_agents::{retry_policies, Agent, ModelRetrySettings, RunOptions, Runner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Assistant")
        .instructions("Answer briefly.")
        .model(common::model()?);

    let options = RunOptions::new()
        .max_turns(5)
        .temperature(0.2)
        // Up to two retries when the connection fails; never for other errors.
        .tune(|s| s.retry = Some(ModelRetrySettings::new(2, retry_policies::network_error())));

    let input = "Hello!";
    let result = Runner::run(&agent, input, options).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
