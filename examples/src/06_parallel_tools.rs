//! Parallel tool calls
//!
//! Web demo checklist: "Parallel tool calls".
//!
//! With `parallel_tool_calls` on, the model can ask for several tools in one turn, and the
//! runner executes them concurrently (bounded by `ToolExecutionConfig`).
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`, `EXAMPLE_INPUT`.
//!
//! ```bash
//! cargo run -p openai-agents-examples --example parallel_tools
//! ```

mod common;

use openai_agents::{
    function_tool, Agent, ModelSettings, RunConfig, RunOptions, Runner, ToolExecutionConfig,
};

#[function_tool(description = "Get the weather for one city")]
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

    let mut run_config = RunConfig::default();
    run_config.model_settings = Some(ModelSettings {
        parallel_tool_calls: Some(true),
        ..Default::default()
    });
    run_config.tool_execution = Some(ToolExecutionConfig {
        max_function_tool_concurrency: Some(4),
    });
    let mut options = RunOptions::default();
    options.run_config = run_config;

    let input =
        common::input_with_fallback("request> ", "Compare the weather in Paris and Tokyo.")?;
    let result = Runner::run(&agent, input, options).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
