//! Function tools
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example tools
//! ```

mod common;

use opai_agents::{function_tool, Agent, RunOptions, Runner};

/// Get the current weather for a city.
// The doc comment is what the model reads to decide when to call the tool, so write it for the
// model. Arguments are parsed from the model's JSON into the typed parameters (`city: String`);
// the return value goes back to the model as the tool result. Async functions work the same.
#[function_tool]
async fn get_weather(city: String) -> String {
    println!("[debug] get_weather called city={city}");
    format!("The weather in {city} is sunny with wind, 14-20C.")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Hello world")
        .instructions("You are a helpful agent. Use tools when they help answer the user.")
        .model(common::model()?)
        // The macro generates a function of the same name that returns the tool.
        .tools(vec![get_weather()]);

    // One run can take several model turns: ask, call the tool, answer with its result.
    let result = Runner::run(
        &agent,
        "What's the weather in Tokyo?",
        RunOptions::default(),
    )
    .await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
