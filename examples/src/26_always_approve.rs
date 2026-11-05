//! Sticky always_approve
//!
//! Approving with `always_approve=true` skips later prompts for the same tool in this run.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`, `HITL_AUTO`.
//!
//! ```bash
//! HITL_AUTO=approve cargo run -p opai-agents-examples --example always_approve
//! ```

mod common;

use opai_agents::{function_tool, Agent, RunOptions, Runner};

/// Charge a payment for an invoice id.
// `needs_approval` makes every call to this tool pause the run until a human decides. Use it
// for actions with side effects (money, deletes, e-mail).
#[function_tool(needs_approval)]
async fn pay(invoice_id: String) -> String {
    println!("[tool] pay invoice_id={invoice_id}");
    format!("paid:{invoice_id}")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("cashier")
        .instructions(
            "You must call the `pay` tool once for invoice A100 and once for invoice B200. \
             Do not skip either call. After both succeed, summarize briefly.",
        )
        .model(common::model()?)
        .tools(vec![pay()]);

    let mut result = Runner::run(
        &agent,
        "Please pay invoices A100 and B200.",
        RunOptions::default(),
    )
    .await?;

    // A run that reached an approval does not fail: it returns early with `interruptions` (the
    // pending calls) and can be resumed. Loop, because the model may ask again after resuming.
    while result.is_interrupted() {
        println!("interruptions={}", result.interruptions.len());
        // `to_state` snapshots the paused run. Decisions are recorded on the state, then the
        // state is handed back to the runner.
        let mut state = result.to_state()?;
        for item in result.interruptions.clone() {
            println!("pending tool={} args={}", item.tool_name, item.arguments);
            if common::confirm("approve (sticky for rest of run)?")? {
                // Sticky: subsequent `pay` calls in this run won't pause again.
                state.approve(&item, true);
            } else {
                // A rejection is not an error: the model is told the call was denied (with this
                // message) and can explain or choose another action.
                state.reject(&item, false, Some("payment denied"));
            }
        }

        // Resume from where it stopped; approved calls now execute.
        result = Runner::run_state(&agent, state, RunOptions::default()).await?;
    }

    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
