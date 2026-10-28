//! Human-in-the-loop approvals
//!
//! By default the run asks for each paused call through `RunOptions::resolve_with` and carries
//! on by itself. `HITL_MODE=persist` shows the explicit flow instead: the paused run is saved as
//! JSON, loaded back, resolved and resumed, which is what a run that outlives the process needs.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//! Non-interactive: `HITL_AUTO=approve` or `HITL_AUTO=reject`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example hitl
//! HITL_AUTO=approve cargo run -p opai-agents-examples --example hitl
//! HITL_MODE=persist HITL_AUTO=approve cargo run -p opai-agents-examples --example hitl
//! ```

mod common;

use std::path::Path;

use opai_agents::prelude::*;

/// Get the weather for a given city.
#[function_tool]
async fn get_weather(city: String) -> String {
    format!("The weather in {city} is sunny")
}

/// `FunctionTool::typed` generates the schema and parses the arguments from `#[derive]`s.
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct City {
    /// The city to look up.
    city: String,
}

fn get_temperature_tool() -> FunctionTool {
    FunctionTool::typed(
        "get_temperature",
        "Get the temperature for a given city",
        |_ctx, City { city }| async move {
            Ok(serde_json::Value::String(format!(
                "The temperature in {city} is 20° Celsius"
            )))
        },
    )
    .with_needs_resolution_fn(|params, _call_id| async move {
        // Dynamic approval: only Oakland (matches Python example). The predicate sees the raw
        // arguments, as it does before the call is made.
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

    let input = "What is the weather and temperature in Oakland?";

    let result = if std::env::var("HITL_MODE").as_deref() == Ok("persist") {
        run_with_saved_state(&agent, input).await?
    } else {
        let options = RunOptions::new().resolve_with(|call| async move {
            println!("\nTool call: {} {}", call.tool_name, call.arguments);
            match common::confirm("Do you approve this tool call?") {
                Ok(true) => ToolResolution::Run,
                _ => ToolResolution::reject("The user rejected this tool call."),
            }
        });
        Runner::run(&agent, input, options).await?
    };

    println!("\n{}", "=".repeat(80));
    println!("Final Output:");
    println!("{}", "=".repeat(80));
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}

/// The explicit flow: save the paused run, load it back, resolve each call, resume.
async fn run_with_saved_state(
    agent: &Agent,
    input: &str,
) -> Result<RunResult, Box<dyn std::error::Error>> {
    let mut result = Runner::run(agent, input, RunOptions::new()).await?;
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
        result = Runner::run_state(agent, state, RunOptions::new()).await?;
    }
    Ok(result)
}
