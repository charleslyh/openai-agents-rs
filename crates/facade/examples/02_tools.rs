//! 02 — Function tools (core path)
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run --example 02_tools
//! ```

#[path = "common/mod.rs"]
mod common;

use openai_agents::{function_tool, Agent, RunOptions, Runner};

#[function_tool(description = "Get the current weather for a city")]
async fn get_weather(city: String) -> String {
    println!("[debug] get_weather called city={city}");
    format!("The weather in {city} is sunny with wind, 14-20C.")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Hello world")
        .instructions("You are a helpful agent. Use tools when they help answer the user.")
        .model(common::model()?)
        .tools(vec![get_weather()]);

    let result = Runner::run(
        &agent,
        "What's the weather in Tokyo?",
        RunOptions::default(),
    )
    .await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
