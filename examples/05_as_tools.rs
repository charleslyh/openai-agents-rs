//! 05 — Agents as tools (orchestration)
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! EXAMPLE_INPUT="Translate 'Hello, world!' to French and Spanish." \
//!   cargo run --example 05_as_tools
//! ```

#[path = "common/mod.rs"]
mod common;

use std::sync::Arc;

use openai_agents::{Agent, AsToolConfig, ItemHelpers, RunItem, RunOptions, Runner};

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
        .tools(vec![
            spanish.as_tool(AsToolConfig {
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

    let synthesizer = Agent::new("synthesizer_agent")
        .instructions(
            "You inspect translations, correct them if needed, and produce a final concatenated response.",
        )
        .model(model);

    let msg = common::input_with_fallback(
        "Hi! What would you like translated, and to which languages? ",
        "Translate 'Hello, world!' to French and Spanish.",
    )?;

    let orchestrator_result = Runner::run(&orchestrator, msg, RunOptions::default()).await?;
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
    let mut synth_input = orchestrator_result.to_input_list();
    synth_input.push(serde_json::json!({
        "role": "user",
        "content": "Please produce the final concatenated response for the user now."
    }));

    let synthesizer_result =
        Runner::run(&synthesizer, synth_input, RunOptions::default()).await?;

    println!(
        "\n\nFinal response:\n{}",
        synthesizer_result.final_output_as_str().unwrap_or_default()
    );
    Ok(())
}
