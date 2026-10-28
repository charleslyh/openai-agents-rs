//! Printing a streamed run to the terminal, shared by the streaming examples.
//!
//! [`Printer::feed`] takes each [`StreamEvent`] and prints what a person wants to read: thinking
//! and answer text as they arrive (a typewriter), and one line for each agent change and tool
//! call. An example keeps only what is its own, such as cancelling after a number of deltas.

use std::io::{self, Write};

use opai_agents::model::wire_events::{self, WireEvent};
use opai_agents::{ItemHelpers, RunItem, RunItemStreamName, StreamEvent};

/// What is being printed at the moment, so a heading appears once where it starts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    Reasoning,
    Message,
}

/// Prints a stream of events and counts what went by.
pub struct Printer {
    section: Section,
    /// Whether the current turn's answer arrived as deltas (so it is not printed again whole).
    streamed_text: bool,
    /// Whether tool calls and agent changes get a line of their own.
    show_steps: bool,
    /// Characters of thinking seen so far.
    pub reasoning_chars: usize,
    /// Text deltas of the answer seen so far.
    pub text_deltas: usize,
}

impl Printer {
    /// Print thinking, answer text, agent changes and tool calls.
    pub fn new() -> Self {
        Self {
            section: Section::None,
            streamed_text: false,
            show_steps: true,
            reasoning_chars: 0,
            text_deltas: 0,
        }
    }

    /// Print only the thinking and the answer text.
    pub fn text_only() -> Self {
        Self {
            show_steps: false,
            ..Self::new()
        }
    }

    /// Print one event. Call it for every event the run yields.
    pub fn feed(&mut self, event: &StreamEvent) -> io::Result<()> {
        match event {
            StreamEvent::RawResponse { data } => match wire_events::parse(data) {
                WireEvent::ReasoningSummaryTextDelta { delta, .. }
                | WireEvent::ReasoningTextDelta { delta, .. } => {
                    self.open(Section::Reasoning)?;
                    self.reasoning_chars += delta.chars().count();
                    print!("{delta}");
                    io::stdout().flush()
                }
                WireEvent::OutputTextDelta { delta, .. } => {
                    self.open(Section::Message)?;
                    self.streamed_text = true;
                    self.text_deltas += 1;
                    print!("{delta}");
                    io::stdout().flush()
                }
                _ => Ok(()),
            },
            StreamEvent::AgentUpdated { agent_name } if self.show_steps => {
                self.end()?;
                println!("Agent updated: {agent_name}");
                Ok(())
            }
            StreamEvent::RunItem { name, item } if self.show_steps => self.step(name, item),
            _ => Ok(()),
        }
    }

    /// Close the line the last delta left open. Call it once the stream has ended.
    pub fn finish(&mut self) -> io::Result<()> {
        self.end()
    }

    fn step(&mut self, name: &RunItemStreamName, item: &RunItem) -> io::Result<()> {
        self.end()?;
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
                if let RunItem::ToolCallOutput(o) = item {
                    println!("-- Tool output: {}", o.output);
                }
            }
            RunItemStreamName::MessageOutputCreated => {
                // A model that does not stream deltas hands over the whole message here.
                if !self.streamed_text {
                    if let RunItem::Message(m) = item {
                        println!(
                            "-- Message output:\n{}",
                            ItemHelpers::text_message_output(m)
                        );
                    }
                }
                // A later turn can still stream.
                self.streamed_text = false;
            }
            _ => {}
        }
        Ok(())
    }

    fn open(&mut self, next: Section) -> io::Result<()> {
        if self.section == next {
            return Ok(());
        }
        self.end()?;
        match next {
            Section::Reasoning => println!("-- Reasoning:"),
            Section::Message => println!("-- Message output:"),
            Section::None => {}
        }
        self.section = next;
        io::stdout().flush()
    }

    fn end(&mut self) -> io::Result<()> {
        if self.section != Section::None {
            println!();
            self.section = Section::None;
            io::stdout().flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn delta(kind: &str, text: &str) -> StreamEvent {
        StreamEvent::RawResponse {
            data: json!({
                "type": kind, "item_id": "i", "output_index": 0, "content_index": 0,
                "summary_index": 0, "delta": text, "sequence_number": 0
            }),
        }
    }

    #[test]
    fn it_counts_thinking_characters_and_answer_deltas_separately() {
        let mut printer = Printer::text_only();
        for event in [
            delta("response.reasoning_summary_text.delta", "hm"),
            delta("response.reasoning_text.delta", "ok"),
            delta("response.output_text.delta", "a"),
            delta("response.output_text.delta", "b"),
            delta("response.refusal.delta", "ignored"),
        ] {
            printer.feed(&event).unwrap();
        }
        printer.finish().unwrap();
        assert_eq!(printer.reasoning_chars, 4);
        assert_eq!(printer.text_deltas, 2);
    }

    #[test]
    fn a_new_printer_has_counted_nothing() {
        let printer = Printer::new();
        assert_eq!((printer.reasoning_chars, printer.text_deltas), (0, 0));
    }
}
