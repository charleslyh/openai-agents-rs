//! Thinking / reasoning
//!
//! Web demo checklist: "Reasoning".
//!
//! Two parts: asking the model to think, and reading what it thought.
//!
//! - **Asking**: `RunOptions::reasoning(..)` sets a typed `Reasoning`. The adapter maps it per API:
//!   the Responses API gets the `reasoning` object (`effort`, `summary`), Chat Completions gets
//!   `reasoning_effort` (its only reasoning field). A model *name* resolved through a provider
//!   can carry the effort too: `Agent::new(..).model_name("o4-mini:high")`.
//! - **Reading**: thinking arrives as `response.reasoning_summary_text.delta` /
//!   `response.reasoning_text.delta` wire events, before the answer's
//!   `response.output_text.delta`. `wire_events::parse` turns them into typed `WireEvent`s. Servers
//!   that return their chain of thought (DeepSeek `reasoning_content`, ...) show up the same way.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL` (a reasoning model). Optional: `OPENAI_BASE_URL`,
//! `OPENAI_API`, `REASONING_EFFORT` (`low` | `medium` | `high`, default `high`),
//! `EXAMPLE_INPUT`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example reasoning
//! ```

mod common;

use opai_agents::{Agent, Reasoning, ReasoningEffort, ReasoningSummary, RunOptions, Runner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let effort = std::env::var("REASONING_EFFORT").unwrap_or_else(|_| "high".into());

    let agent = Agent::new("Thinker")
        .instructions("Solve the problem. Give the answer in one short sentence.")
        .model(common::model()?);

    // `summary` asks for a readable summary of the thinking (OpenAI does not return the raw
    // chain of thought). Chat Completions has no such field and ignores it.
    let options = RunOptions::new().reasoning(
        Reasoning::effort(ReasoningEffort::from(effort.as_str()))
            .with_summary(ReasoningSummary::Auto),
    );

    let input = common::example_input(
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. \
         How much does the ball cost?",
    );

    let mut streamed = Runner::run_streamed(agent, input, options);
    let mut printer = common::stream::Printer::text_only();
    while let Some(event) = streamed.next_event().await {
        printer.feed(&event?)?;
    }
    printer.finish()?;

    if printer.reasoning_chars == 0 {
        eprintln!(
            "(no reasoning text was streamed: use a reasoning model, and with the Responses API \
             an organization allowed to receive reasoning summaries)"
        );
    } else {
        eprintln!("({} characters of thinking)", printer.reasoning_chars);
    }
    Ok(())
}
