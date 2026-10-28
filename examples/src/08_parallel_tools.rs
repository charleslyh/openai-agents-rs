//! Parallel tool calls
//!
//! Web demo checklist: "Parallel tool calls".
//!
//! With `parallel_tool_calls` on, the model can ask for several tools in one turn, and the
//! runner executes them concurrently (bounded by `ToolExecutionConfig`).
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example parallel_tools
//! ```

mod common;

use opai_agents::{function_tool, Agent, RunOptions, Runner, ToolExecutionConfig};

/// Get the weather for one city.
#[function_tool]
async fn get_weather(city: String) -> String {
    println!("[tool] get_weather({city})");
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    format!("The weather in {city} is sunny, 18C.")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Weather")
        .instructions(
            "Call get_weather once per city, all in the same turn, then compare the cities.",
        )
        .model(common::model()?)
        .tools(vec![get_weather()]);

    let options = RunOptions::new()
        .tune(|s| s.parallel_tool_calls = Some(true))
        .tools(|t| {
            *t = t
                .clone()
                .execution(ToolExecutionConfig::new().concurrency(4))
        });

    let input = "Compare the weather in Paris and Tokyo.";
    let result = Runner::run(&agent, input, options).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
