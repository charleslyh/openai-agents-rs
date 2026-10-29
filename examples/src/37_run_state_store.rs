//! A paused run that outlives the process (Rust-only)
//!
//! An approval can take hours. `hitl` shows `resolve_with` (the run waits inside one process) and
//! a hand-rolled JSON file; this example shows the supported way to park a run: a
//! `RunStateStore`. `FileRunStateStore` writes each paused run to its own file, under a key you
//! choose (an order id, a ticket number), with an optional time to live. Redis and Postgres have
//! the same trait, so a worker on another machine can pick the run up.
//!
//! The example plays two "processes" with a scripted model, so it needs no key:
//!
//! 1. The first process runs the agent. The model asks to call `refund`, which needs approval, so
//!    the run pauses. The state is saved under the key `ticket-42`, and everything in memory is
//!    dropped.
//! 2. The second process builds a brand-new agent, loads `ticket-42`, approves the call and
//!    resumes. The refund tool runs and the model gives its final answer. The state is deleted.
//!
//! Set `AGENTS_STATE_KEY` to sign the stored state with HMAC-SHA256: a state whose file was edited
//! by hand (say, to turn a rejection into an approval) then fails to load instead of resuming.
//! Without the variable the store warns and writes unsigned states.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example run_state_store
//! AGENTS_STATE_KEY=change-me cargo run -p opai-agents-examples --example run_state_store
//! ```

use std::sync::Arc;
use std::time::Duration;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{Agent, FileRunStateStore, FunctionTool, RunOptions, RunStateStore, Runner};

const KEY: &str = "ticket-42";

/// The agent both "processes" use, with the model's scripted answers.
///
/// The agent itself is never stored, only the paused run is, so the resuming side must be able to
/// build the same agent again (same name, same tools).
fn new_agent(steps: Vec<ModelStep>) -> Agent {
    Agent::new("Support")
        .instructions("Refund the order the customer names.")
        .model(Arc::new(ScriptedModel::new(steps)))
        .tools(vec![FunctionTool::constant(
            "refund",
            "Refund an order.",
            "refunded",
        )
        .with_needs_resolution(true)])
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("STATE_DIR").unwrap_or_else(|_| "/tmp/agents-run-states".into());

    // ---- Process 1: run until the approval is needed, park the run, exit. ----
    {
        let store = FileRunStateStore::from_env(&dir)?;
        let result = Runner::run(
            &new_agent(vec![ModelStep::from(ItemHelpers::function_tool_call(
                "refund",
                r#"{"order":"A-1001"}"#,
                "call-1",
            ))]),
            "Please refund order A-1001.",
            RunOptions::new(),
        )
        .await?;
        assert!(result.is_interrupted(), "the refund must wait for approval");

        for call in &result.interruptions {
            println!(
                "[process 1] paused: {} wants to call {} with {}",
                call.agent_name, call.tool_name, call.arguments
            );
        }
        store
            .save(
                KEY,
                &result.to_state()?,
                Some(Duration::from_secs(24 * 3600)),
            )
            .await?;
        println!(
            "[process 1] saved the run under `{KEY}` in {dir} (signed: {})",
            store.is_signed()
        );
        // The result, the agent and the model are dropped here. Only the file is left.
    }

    // ---- Process 2: a fresh agent, the stored run, a decision. ----
    let store = FileRunStateStore::from_env(&dir)?;
    let Some(mut state) = store.load(KEY).await? else {
        return Err(format!("no paused run under `{KEY}` (expired or already resumed)").into());
    };
    println!(
        "\n[process 2] loaded `{KEY}`: {} pending call(s)",
        state.interruptions.len()
    );

    for call in state.interruptions.clone() {
        println!("[process 2] a human approves {}", call.tool_name);
        state.approve(&call, false);
    }

    let result = Runner::run_state(
        &new_agent(vec![ModelStep::from(ItemHelpers::text_message(
            "Order A-1001 has been refunded.",
        ))]),
        state,
        RunOptions::new(),
    )
    .await?;
    store.delete(KEY).await?;
    println!(
        "[process 2] resumed and finished: {}",
        result.final_output_as_str().unwrap_or_default()
    );
    Ok(())
}
