//! Thinking / reasoning
//!
//! Web demo checklist: "Reasoning".
//!
//! Two parts: asking the model to think, and reading what it thought.
//!
//! - **Asking**: with the Responses API, `ModelSettings::reasoning` is sent as the request's
//!   `reasoning` object (`effort`, `summary`). Chat Completions has no such mapping, so the
//!   effort goes through `extra_args` as `reasoning_effort`.
//! - **Reading**: thinking arrives as `response.reasoning_summary_text.delta` /
//!   `response.reasoning_text.delta` wire events, before the answer's
//!   `response.output_text.delta`. `wire_events::parse` turns them into typed `WireEvent`s. Servers
//!   that return their chain of thought (DeepSeek `reasoning_content`, ...) show up the same way.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL` (a reasoning model). Optional: `OPENAI_BASE_URL`,
//! `OPENAI_API`, `REASONING_EFFORT` (`low` | `medium` | `high`, default `medium`),
//! `EXAMPLE_INPUT`.
//!
//! ```bash
//! cargo run -p openai-agents-examples --example reasoning
//! ```

mod common;

use std::io::{self, Write};

use openai_agents::model::wire_events::{self, WireEvent};
use openai_agents::serde_json::{json, Map};
use openai_agents::{Agent, ModelSettings, RunConfig, RunOptions, Runner, StreamEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let effort = std::env::var("REASONING_EFFORT").unwrap_or_else(|_| "high".into());

    let settings = match common::ExampleApi::from_env()? {
        // Responses API: `reasoning` is the request's own object. `summary` asks for a
        // readable summary of the thinking (raw chain of thought is not returned by OpenAI).
        common::ExampleApi::Responses => ModelSettings {
            reasoning: Some(json!({"effort": effort, "summary": "auto"})),
            ..Default::default()
        },
        // Chat Completions: not mapped, so forward the provider field as-is.
        common::ExampleApi::ChatCompletions => {
            let mut extra = Map::new();
            extra.insert("reasoning_effort".into(), json!(effort));
            ModelSettings {
                extra_args: Some(extra),
                ..Default::default()
            }
        }
    };

    let agent = Agent::new("Thinker")
        .instructions("Solve the problem. Give the answer in one short sentence.")
        .model(common::model()?);

    let mut options = RunOptions::default();
    options.run_config = RunConfig {
        model_settings: Some(settings),
        ..Default::default()
    };

    let input = common::example_input(
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. \
         How much does the ball cost?",
    );

    let mut streamed = Runner::run_streamed(agent, input, options);
    let mut thinking = false;
    let mut answering = false;
    let mut reasoning_chars = 0usize;

    while let Some(ev) = streamed.next_event().await {
        let StreamEvent::RawResponse { data } = ev? else {
            continue;
        };
        match wire_events::parse(&data) {
            WireEvent::ReasoningSummaryTextDelta { delta, .. }
            | WireEvent::ReasoningTextDelta { delta, .. } => {
                if !thinking {
                    println!("-- Thinking:");
                    thinking = true;
                }
                reasoning_chars += delta.chars().count();
                print!("{delta}");
                io::stdout().flush()?;
            }
            WireEvent::OutputTextDelta { delta, .. } => {
                if !answering {
                    if thinking {
                        println!();
                    }
                    println!("-- Answer:");
                    answering = true;
                }
                print!("{delta}");
                io::stdout().flush()?;
            }
            _ => {}
        }
    }
    println!();

    if reasoning_chars == 0 {
        eprintln!(
            "(no reasoning text was streamed: use a reasoning model, and with the Responses API \
             an organization allowed to receive reasoning summaries)"
        );
    } else {
        eprintln!("({reasoning_chars} characters of thinking)");
    }
    Ok(())
}
