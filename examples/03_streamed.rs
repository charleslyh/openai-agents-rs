//! 03 — Streaming run events (core path)
//!
//! Prints assistant text as token deltas (typewriter). Both OpenAI APIs emit the standard
//! Responses wire events (D-011): `response.output_text.delta` for visible tokens, and
//! `response.reasoning_summary_text.delta` / `response.reasoning_text.delta` for thinking.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`, `OPENAI_API`.
//!
//! ```bash
//! cargo run --example 03_streamed
//! ```

#[path = "common/mod.rs"]
mod common;

use std::io::{self, Write};

use openai_agents::{
    function_tool, Agent, ItemHelpers, RunItem, RunItemStreamName, RunOptions, Runner, StreamEvent,
};

#[function_tool(description = "Return how many jokes to tell (1-5)")]
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
    let mut saw_text_delta = false;
    let mut section: Section = Section::None;

    while let Some(ev) = streamed.next_event().await {
        match ev? {
            StreamEvent::RawResponse { data } => {
                let kind = data.get("type").and_then(|t| t.as_str()).unwrap_or("");
                let delta = data.get("delta").and_then(|d| d.as_str()).unwrap_or("");
                match kind {
                    "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                        open_section(&mut section, Section::Reasoning)?;
                        print!("{delta}");
                        io::stdout().flush()?;
                    }
                    "response.output_text.delta" => {
                        open_section(&mut section, Section::Message)?;
                        print!("{delta}");
                        io::stdout().flush()?;
                        saw_text_delta = true;
                    }
                    _ => {}
                }
            }
            StreamEvent::AgentUpdated { agent_name } => {
                end_section(&mut section)?;
                println!("Agent updated: {agent_name}");
            }
            StreamEvent::RunItem { name, item } => {
                end_section(&mut section)?;
                match name {
                    RunItemStreamName::ToolCalled => {
                        let tool = item
                            .raw_item()
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("?");
                        println!("-- Tool was called: {tool}");
                    }
                    RunItemStreamName::ToolOutput => {
                        if let RunItem::ToolCallOutput(o) = &item {
                            println!("-- Tool output: {}", o.output);
                        }
                    }
                    RunItemStreamName::MessageOutputCreated => {
                        if !saw_text_delta {
                            if let RunItem::Message(m) = &item {
                                println!(
                                    "-- Message output:\n{}",
                                    ItemHelpers::text_message_output(m)
                                );
                            }
                        }
                        // Reset so a later turn can still stream deltas.
                        saw_text_delta = false;
                    }
                    _ => {}
                }
            }
        }
    }
    end_section(&mut section)?;
    println!("=== Run complete ===");
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    Reasoning,
    Message,
}

fn open_section(current: &mut Section, next: Section) -> io::Result<()> {
    if *current == next {
        return Ok(());
    }
    end_section(current)?;
    match next {
        Section::Reasoning => print!("-- Reasoning:\n"),
        Section::Message => print!("-- Message output:\n"),
        Section::None => {}
    }
    io::stdout().flush()?;
    *current = next;
    Ok(())
}

fn end_section(current: &mut Section) -> io::Result<()> {
    if *current != Section::None {
        println!();
        io::stdout().flush()?;
        *current = Section::None;
    }
    Ok(())
}
