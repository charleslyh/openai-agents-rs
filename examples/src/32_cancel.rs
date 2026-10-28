//! Stop a run
//!
//! Web demo checklist: "Stop a run".
//!
//! `RunResultStreaming::cancel` stops a streamed run. `Immediate` aborts now; `AfterTurn` lets
//! the current turn finish first. A cancelled run has no final output. `cancel_on_drop(true)`
//! also stops the run when the handle is dropped.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`, `OPENAI_API`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example cancel
//! ```

mod common;

use opai_agents::{Agent, CancelMode, RunOptions, Runner};

/// Stop after this many streamed text deltas.
const STOP_AFTER_DELTAS: usize = 25;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Storyteller")
        .instructions("You write short stories.")
        .model(common::model()?);

    let mut streamed = Runner::run_streamed(
        agent,
        "Write a 200 word story about a lighthouse.",
        RunOptions::default(),
    )
    .cancel_on_drop(true);

    let mut printer = common::stream::Printer::text_only();
    while let Some(event) = streamed.next_event().await {
        printer.feed(&event?)?;
        if printer.text_deltas == STOP_AFTER_DELTAS {
            streamed.cancel(CancelMode::Immediate);
        }
    }
    printer.finish()?;
    let deltas = printer.text_deltas;
    println!(
        "cancelled={} complete={} (after {deltas} deltas)",
        streamed.is_cancelled(),
        streamed.is_complete()
    );
    Ok(())
}
