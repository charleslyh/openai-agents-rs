//! An agent whose tool call waits for a human.
//!
//! The tool is marked `needs_resolution`, so the run pauses with the call; `resolve_with` is asked
//! what to do and the run continues. Approving, rejecting and answering the call yourself are the
//! three resolutions — `ToolResolution::Run`, `ToolResolution::reject(..)` and
//! `ToolResolution::output(..)`.
//!
//! Non-interactive use: `AGENTS_AUTO=approve` or `AGENTS_AUTO=reject`.

use opai_agents::prelude::*;
use opai_agents::function_tool;

#[function_tool(description = "Refund an order; spends real money")]
async fn refund(order_id: String) -> String {
    format!("Refunded {order_id}.")
}

#[tokio::main]
async fn main() -> Result<(), AgentsError> {
    let agent = Agent::new("Refunds")
        .instructions("Refund the order the user names.")
        .tools(vec![refund().with_needs_resolution(true)]);
    let input = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Refund order 1234.".to_string());

    let auto = std::env::var("AGENTS_AUTO").unwrap_or_default();
    let options = opai_agents::zero_config_options()?.resolve_with(move |call: ToolInterruptionItem| {
        let auto = auto.clone();
        async move {
            println!("Tool `{}` wants to run with {}", call.tool_name, call.arguments);
            match auto.as_str() {
                "approve" => ToolResolution::Run,
                "reject" => ToolResolution::reject("The user rejected this call."),
                _ => {
                    println!("Approve? [y/N] ");
                    let mut line = String::new();
                    std::io::stdin().read_line(&mut line).ok();
                    if line.trim().eq_ignore_ascii_case("y") {
                        ToolResolution::Run
                    } else {
                        ToolResolution::reject("The user rejected this call.")
                    }
                }
            }
        }
    });

    let result = Runner::run(&agent, input, options).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
