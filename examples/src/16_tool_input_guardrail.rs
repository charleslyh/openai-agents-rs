//! Tool input guardrail
//!
//! Web demo checklist: "Tool input guardrail".
//!
//! `known-city` runs before the `get_weather` body and rejects cities the tool has no data
//! for. `reject_content` replaces the call with a message for the model; the run carries on.
//! (`raise_exception` would stop the run instead.)
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`, `EXAMPLE_INPUT`.
//!
//! ```bash
//! cargo run -p openai-agents-examples --example tool_input_guardrail
//! ```

mod common;

use openai_agents::{
    Agent, FunctionTool, RunOptions, Runner, ToolGuardrailFunctionOutput, ToolInputGuardrail,
};
use serde_json::{json, Value};

const KNOWN_CITIES: [&str; 4] = ["paris", "tokyo", "london", "berlin"];

fn known_city() -> ToolInputGuardrail {
    ToolInputGuardrail::new("known-city", |data| async move {
        let args: Value = serde_json::from_str(&data.context.tool_arguments).unwrap_or(Value::Null);
        let city = args
            .get("city")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        if KNOWN_CITIES.contains(&city.as_str()) {
            ToolGuardrailFunctionOutput::allow(json!({"city": city}))
        } else {
            println!("[guardrail] known-city rejected `{city}`");
            ToolGuardrailFunctionOutput::reject_content(
                format!(
                    "Blocked: unknown city. Supported cities: {}.",
                    KNOWN_CITIES.join(", ")
                ),
                json!({"city": city}),
            )
        }
    })
}

fn get_weather() -> FunctionTool {
    FunctionTool::new(
        "get_weather",
        "Get the weather for a known city",
        json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false
        }),
        |_ctx, raw| async move {
            let args: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
            let city = args.get("city").and_then(Value::as_str).unwrap_or("?");
            Ok(json!(format!("The weather in {city} is sunny, 18C.")))
        },
    )
    .with_tool_input_guardrails(vec![known_city()])
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Weather")
        .instructions("Use get_weather and report the result. Relay any tool error to the user.")
        .model(common::model()?)
        .tools(vec![get_weather()]);

    let input = common::input_with_fallback("question> ", "What is the weather in Atlantis?")?;
    let result = Runner::run(&agent, input, RunOptions::default()).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    println!(
        "tool input guardrail results: {}",
        result.tool_input_guardrail_results.len()
    );
    Ok(())
}
