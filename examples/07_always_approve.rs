//! 07 — Sticky always_approve
//!
//! Approving with `always_approve=true` skips later prompts for the same tool in this run.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`, `HITL_AUTO`.
//!
//! ```bash
//! HITL_AUTO=approve cargo run --example 07_always_approve
//! ```

#[path = "common/mod.rs"]
mod common;

use openai_agents::{function_tool, Agent, RunOptions, Runner};

#[function_tool(description = "Charge a payment for an invoice id", needs_approval)]
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

    while result.is_interrupted() {
        println!("interruptions={}", result.interruptions.len());
        let mut state = result.to_state()?;
        for item in result.interruptions.clone() {
            println!(
                "pending tool={} args={}",
                item.tool_name, item.arguments
            );
            if common::confirm("approve (sticky for rest of run)?")? {
                // Sticky: subsequent `pay` calls in this run won't pause again.
                state.approve(&item, true);
            } else {
                state.reject(&item, false, Some("payment denied"));
            }
        }
        result = Runner::run_state(&agent, state, RunOptions::default()).await?;
    }

    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
