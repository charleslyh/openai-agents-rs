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

use openai_agents::{
    retry_policies, Agent, ModelRetrySettings, ModelSettings, RunConfig, RunOptions, Runner,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Assistant")
        .instructions("Answer briefly.")
        .model(common::model()?);

    let mut run_config = RunConfig::default();
    run_config.model.settings = Some(ModelSettings {
        temperature: Some(0.2),
        // Up to two retries when the connection fails; never for other errors.
        retry: Some(ModelRetrySettings::new(2, retry_policies::network_error())),
        ..Default::default()
    });
    let mut options = RunOptions::default();
    options.max_turns = Some(5);
    options.run_config = run_config;

    let input = "Hello!";
    let result = Runner::run(&agent, input, options).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
