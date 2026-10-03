//! 01 — Minimal Agent + Runner (core path)
//!
//! Minimal API agent example (Python: `examples/basic/hello_world.py` style).
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run --example 01_hello
//! ```

#[path = "common/mod.rs"]
mod common;

use openai_agents::{Agent, RunOptions, Runner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Assistant")
        .instructions("You only respond in haikus.")
        .model(common::model()?);

    let result = Runner::run(
        &agent,
        "Tell me about recursion in programming.",
        RunOptions::default(),
    )
    .await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
