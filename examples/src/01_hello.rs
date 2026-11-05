//! Hello world (minimal agent and runner)
//!
//! Minimal API agent example (Python: `examples/basic/hello_world.py` style).
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example hello
//! ```

mod common;

use opai_agents::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // An `Agent` is configuration only: a name, instructions (the system prompt) and a model.
    // It holds no conversation state, so one agent can serve any number of runs.
    let agent = Agent::new("Assistant")
        .instructions("You only respond in haikus.")
        .model(common::model()?);

    // `Runner::run` drives the loop (model call, tools, handoffs) until the agent produces a
    // final output. `RunOptions` carries everything per run: session, context, limits, hooks.
    let result = Runner::run(
        &agent,
        "Tell me about recursion in programming.",
        RunOptions::new(),
    )
    .await?;

    // The final output is a JSON value; `final_output_as_str` is the text view of it.
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
