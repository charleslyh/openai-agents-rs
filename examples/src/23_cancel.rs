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
//! cargo run -p openai-agents-examples --example cancel
//! ```

mod common;

use std::io::{self, Write};

use openai_agents::{Agent, CancelMode, RunOptions, Runner, StreamEvent};

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

    let mut deltas = 0usize;
    while let Some(event) = streamed.next_event().await {
        if let StreamEvent::RawResponse { data } = event? {
            if data.get("type").and_then(|t| t.as_str()) == Some("response.output_text.delta") {
                print!(
                    "{}",
                    data.get("delta").and_then(|d| d.as_str()).unwrap_or("")
                );
                io::stdout().flush()?;
                deltas += 1;
                if deltas == STOP_AFTER_DELTAS {
                    streamed.cancel(CancelMode::Immediate);
                }
            }
        }
    }
    println!();
    println!(
        "cancelled={} complete={} (after {deltas} deltas)",
        streamed.is_cancelled(),
        streamed.is_complete()
    );
    Ok(())
}
