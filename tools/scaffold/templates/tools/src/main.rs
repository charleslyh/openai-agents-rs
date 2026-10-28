//! An agent with one tool, and the same tool written with a typed argument struct.
//!
//! `#[function_tool]` generates the JSON Schema from the signature, so the model sees `city` and
//! the handler receives a `String` — no `serde_json::Value` in between.

use opai_agents::prelude::*;
use opai_agents::function_tool;

/// Get the current weather for a city.
#[function_tool]
async fn get_weather(city: String) -> String {
    format!("The weather in {city} is sunny, 14-20C.")
}

/// `FunctionTool::typed` does the same without the attribute macro: the struct is the argument
/// schema, and its doc comments become the parameter descriptions the model reads. The derives are
/// pointed at the SDK's re-exports, so the project needs no `serde` or `schemars` dependency.
#[derive(opai_agents::serde::Deserialize, opai_agents::schemars::JsonSchema)]
#[serde(crate = "opai_agents::serde")]
#[schemars(crate = "opai_agents::schemars")]
struct Lookup {
    /// The city to look up.
    city: String,
}

fn get_forecast() -> FunctionTool {
    FunctionTool::typed(
        "get_forecast",
        "Get the three-day forecast for a city",
        |_ctx, Lookup { city }| async move {
            Ok(serde_json::Value::String(format!("{city}: sunny, sunny, rain.")))
        },
    )
}

#[tokio::main]
async fn main() -> Result<(), AgentsError> {
    let agent = Agent::new("Weather assistant")
        .instructions("Answer with the tools; say so when a tool has no answer.")
        .tools(vec![get_weather(), get_forecast()]);
    let input = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "What is the weather and forecast in Tokyo?".to_string());

    let result = Runner::run(&agent, input, opai_agents::zero_config_options()?).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
