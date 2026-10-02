//! 06 — Human-in-the-loop approvals
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//! Non-interactive: `HITL_AUTO=approve` or `HITL_AUTO=reject`.
//!
//! ```bash
//! cargo run --example 06_human_in_the_loop
//! HITL_AUTO=approve cargo run --example 06_human_in_the_loop
//! ```

#[path = "common/mod.rs"]
mod common;

use std::path::Path;

use openai_agents::{function_tool, Agent, FunctionTool, RunOptions, RunState, Runner};

#[function_tool(description = "Get the weather for a given city")]
async fn get_weather(city: String) -> String {
    format!("The weather in {city} is sunny")
}

fn get_temperature_tool() -> FunctionTool {
    FunctionTool::new(
        "get_temperature",
        "Get the temperature for a given city",
        serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
            "additionalProperties": false
        }),
        |_ctx, args| async move {
            let v: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
            let city = v.get("city").and_then(|c| c.as_str()).unwrap_or("?");
            Ok(serde_json::Value::String(format!(
                "The temperature in {city} is 20° Celsius"
            )))
        },
    )
    .with_needs_approval_fn(|params, _call_id| async move {
        // Dynamic approval: only Oakland (matches Python example).
        params
            .get("city")
            .and_then(|c| c.as_str())
            .map(|c| c.contains("Oakland"))
            .unwrap_or(false)
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Weather Assistant")
        .instructions(
            "You are a helpful weather assistant. \
             Answer questions about weather and temperature using the available tools.",
        )
        .model(common::model()?)
        .tools(vec![get_weather(), get_temperature_tool()]);

    let input = common::input_with_fallback(
        "Ask about weather: ",
        "What is the weather and temperature in Oakland?",
    )?;

    let mut result = Runner::run(&agent, input, RunOptions::default()).await?;
    let path = Path::new(".cache/agent_patterns/human_in_the_loop/result.json");

    while result.is_interrupted() {
        println!("\n{}", "=".repeat(80));
        println!("Run interrupted - tool approval required");
        println!("{}\n", "=".repeat(80));

        let mut state = result.to_state()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&state.to_json())?)?;
        println!("State saved to {}", path.display());

        println!("Loading state from {}", path.display());
        let stored: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        state = RunState::from_json(&agent.name, stored)?;

        for interruption in result.interruptions.clone() {
            println!("\nTool call details:");
            println!("  Agent: {}", interruption.agent_name);
            println!("  Tool: {}", interruption.tool_name);
            println!("  Arguments: {}", interruption.arguments);

            if common::confirm("\nDo you approve this tool call?")? {
                println!("✓ Approved: {}", interruption.tool_name);
                state.approve(&interruption, false);
            } else {
                println!("✗ Rejected: {}", interruption.tool_name);
                state.reject(&interruption, false, None);
            }
        }

        println!("\nResuming agent execution...");
        result = Runner::run_state(&agent, state, RunOptions::default()).await?;
    }

    println!("\n{}", "=".repeat(80));
    println!("Final Output:");
    println!("{}", "=".repeat(80));
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
