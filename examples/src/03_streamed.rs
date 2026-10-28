//! Streaming run events
//!
//! Prints assistant text as token deltas (typewriter); `common::stream::Printer` does the printing. Both OpenAI APIs emit the standard
//! Responses wire events (D-011): `response.output_text.delta` for visible tokens, and
//! `response.reasoning_summary_text.delta` / `response.reasoning_text.delta` for thinking.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`, `OPENAI_API`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example streamed
//! ```

mod common;

use opai_agents::{function_tool, Agent, RunOptions, Runner};

/// Return how many jokes to tell (1-5).
#[function_tool]
fn how_many_jokes() -> i64 {
    3
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Joker")
        .instructions("First call the `how_many_jokes` tool, then tell that many short jokes.")
        .model(common::model()?)
        .tools(vec![how_many_jokes()]);

    println!("=== Run starting ===");
    let mut streamed = Runner::run_streamed(agent, "Hello", RunOptions::default());
    let mut printer = common::stream::Printer::new();
    while let Some(event) = streamed.next_event().await {
        printer.feed(&event?)?;
    }
    printer.finish()?;
    println!("=== Run complete ===");
    Ok(())
}
