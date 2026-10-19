//! Tool failure recovery
//!
//! Web demo checklist: "Tool failure recovery".
//!
//! `flaky_inventory` fails on its first call. A failing tool does not end the run: the error
//! text goes back to the model, which can read it and call the tool again.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p openai-agents-examples --example tool_recovery
//! ```

mod common;

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use openai_agents::{Agent, AgentsError, FunctionTool, RunOptions, Runner};
use serde_json::json;

/// The tool's arguments (see [`FunctionTool::typed`]).
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Item {
    /// The item to count.
    item: String,
}

fn flaky_inventory() -> FunctionTool {
    let calls = Arc::new(AtomicU32::new(0));
    // `FunctionTool::typed` derives the schema from the argument type and parses the arguments.
    FunctionTool::typed(
        "flaky_inventory",
        "Check warehouse inventory for an item (may fail transiently)",
        move |_ctx, Item { item }| {
            let calls = Arc::clone(&calls);
            async move {
                let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
                println!("[tool] flaky_inventory call #{n} for `{item}`");
                if n == 1 {
                    return Err(AgentsError::tool(
                        "inventory backend unavailable, try again",
                    ));
                }
                Ok(json!("42 units in stock"))
            }
        },
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Ops")
        .instructions("Use flaky_inventory. If it fails, read the error and retry the call once.")
        .model(common::model()?)
        .tools(vec![flaky_inventory()]);

    let input = "Check warehouse inventory for widget A.";
    let result = Runner::run(&agent, input, RunOptions::default()).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
