//! Tool timeout
//!
//! Web demo checklist: "Tool timeout".
//!
//! `slow_report` outlives its 1 s limit. With `ToolTimeoutBehavior::ErrorAsResult` (the
//! default) the timeout is handed to the model as the tool result and the run goes on; with
//! `RaiseException` the run fails with `AgentsError::ToolTimeout`. Set `TOOL_TIMEOUT_RAISE=1`
//! to try the second one.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example tool_timeout
//! TOOL_TIMEOUT_RAISE=1 cargo run -p opai-agents-examples --example tool_timeout
//! ```

mod common;

use std::time::Duration;

use opai_agents::{Agent, AgentsError, FunctionTool, RunOptions, Runner, ToolTimeoutBehavior};
use serde_json::json;

fn slow_report(behavior: ToolTimeoutBehavior) -> FunctionTool {
    // `()` as the argument type: no parameters, and the generated schema says so.
    FunctionTool::typed(
        "slow_report",
        "Build the quarterly report (takes a long time)",
        |_ctx, ()| async move {
            println!("[tool] slow_report started");
            tokio::time::sleep(Duration::from_secs(30)).await;
            Ok(json!("report ready"))
        },
    )
    .with_timeout(1.0)
    .with_timeout_behavior(behavior)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let behavior = if std::env::var("TOOL_TIMEOUT_RAISE").is_ok_and(|v| v == "1") {
        ToolTimeoutBehavior::RaiseException
    } else {
        ToolTimeoutBehavior::ErrorAsResult
    };
    let agent = Agent::new("Ops")
        .instructions("Use slow_report to build reports. If it times out, say so.")
        .model(common::model()?)
        .tools(vec![slow_report(behavior)]);

    let input = "Build the quarterly report.";
    match Runner::run(&agent, input, RunOptions::default()).await {
        Ok(result) => println!("{}", result.final_output_as_str().unwrap_or_default()),
        Err(AgentsError::ToolTimeout(e)) => println!("run aborted by tool timeout: {e}"),
        Err(e) => return Err(e.into()),
    }
    Ok(())
}
