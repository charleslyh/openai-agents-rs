//! Agents as tools (orchestration)
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! EXAMPLE_INPUT="Translate 'Hello, world!' to French and Spanish." \
//!   cargo run -p opai-agents-examples --example agents_as_tools
//! ```

mod common;

use std::sync::Arc;

use opai_agents::{Agent, AsToolConfig, ItemHelpers, RunItem, RunOptions, Runner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = common::model()?;

    let spanish = Agent::new("spanish_agent")
        .instructions("You translate the user's message to Spanish")
        .handoff_description("An english to spanish translator")
        .model(Arc::clone(&model));

    let french = Agent::new("french_agent")
        .instructions("You translate the user's message to French")
        .handoff_description("An english to french translator")
        .model(Arc::clone(&model));

    let italian = Agent::new("italian_agent")
        .instructions("You translate the user's message to Italian")
        .handoff_description("An english to italian translator")
        .model(Arc::clone(&model));

    let orchestrator = Agent::new("orchestrator_agent")
        .instructions(
            "You are a translation agent. You use the tools given to you to translate. \
             If asked for multiple translations, you call the relevant tools in order. \
             You never translate on your own, you always use the provided tools.",
        )
        .model(Arc::clone(&model))
        // `as_tool` wraps an agent as a tool. The orchestrator calls it, the translator runs a
        // full nested run on the given text, and its final output returns as the tool result.
        // The orchestrator stays in charge: it gets the result back and decides what to do next.
        .tools(vec![
            spanish.as_tool(AsToolConfig {
                // Name and description are what the orchestrator's model sees: write them as a
                // tool, not as an agent.
                name: Some("translate_to_spanish".into()),
                description: Some("Translate the user's message to Spanish".into()),
                ..Default::default()
            }),
            french.as_tool(AsToolConfig {
                name: Some("translate_to_french".into()),
                description: Some("Translate the user's message to French".into()),
                ..Default::default()
            }),
            italian.as_tool(AsToolConfig {
                name: Some("translate_to_italian".into()),
                description: Some("Translate the user's message to Italian".into()),
                ..Default::default()
            }),
        ]);

    // A second, separate agent that runs after the first: orchestration here is plain Rust code
    // (run one agent, feed its result to the next), not an SDK feature.
    let synthesizer = Agent::new("synthesizer_agent")
        .instructions(
            "You inspect translations, correct them if needed, and produce a final concatenated response.",
        )
        .model(model);

    let msg = common::example_input("Translate 'Hello, world!' to French and Spanish.");

    let orchestrator_result = Runner::run(&orchestrator, msg, RunOptions::default()).await?;

    // `new_items` is everything the run produced, in order: messages, tool calls, tool outputs.
    for item in &orchestrator_result.new_items {
        if let RunItem::Message(m) = item {
            let text = ItemHelpers::text_message_output(m);
            if !text.is_empty() {
                println!("  - Translation step: {text}");
            }
        }
    }

    // `to_input_list` ends with the orchestrator's assistant message. Some models
    // (notably DeepSeek via TokenHub) then return empty `content` for the synthesizer.
    // Append a user nudge so the synthesizer always has a fresh turn to answer.
    // `to_input_list` turns a finished run into input for the next one: the original input plus
    // every item the run produced.
    let mut synth_input = orchestrator_result.to_input_list();
    synth_input.push(serde_json::json!({
        "role": "user",
        "content": "Please produce the final concatenated response for the user now."
    }));

    let synthesizer_result = Runner::run(&synthesizer, synth_input, RunOptions::default()).await?;

    println!(
        "\n\nFinal response:\n{}",
        synthesizer_result.final_output_as_str().unwrap_or_default()
    );
    Ok(())
}
