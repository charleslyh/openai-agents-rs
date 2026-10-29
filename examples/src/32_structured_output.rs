//! Structured output (`output_type`)
//!
//! Declares an output type; the SDK advertises its JSON Schema to the model (Responses:
//! `text.format`, Chat Completions: `response_format`) and validates the reply. A reply that
//! does not match surfaces as a model behavior error, matching Python's
//! `Agent(output_type=...)`.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example structured_output
//! ```

mod common;

use std::sync::Arc;

use opai_agents::{Agent, AgentOutputSchema, RunOptions, Runner};
use serde::Deserialize;

/// The shape the model must answer with.
///
/// `JsonSchema` produces the schema; `Deserialize` lets the SDK validate the reply.
#[derive(Debug, Deserialize, opai_agents::schemars::JsonSchema)]
struct WeatherReport {
    /// City the report is about.
    city: String,
    /// Temperature in Celsius.
    celsius: f32,
    /// One-line summary.
    summary: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Weather")
        .instructions("Always answer with the requested JSON object.")
        .model(common::model()?)
        .output_type(Arc::new(AgentOutputSchema::of::<WeatherReport>()?));

    let question = "What is the weather in Paris right now? Answer in JSON.";

    let result = Runner::run(&agent, question, RunOptions::default()).await?;
    println!("raw final_output: {}", result.final_output);

    let report: WeatherReport = result.final_output_as::<WeatherReport>()?;
    println!("{}: {}°C — {}", report.city, report.celsius, report.summary);
    Ok(())
}
