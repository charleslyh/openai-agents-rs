//! Watching a sub-agent work: `AsToolConfig::on_stream`
//!
//! `agents_as_tools` shows an orchestrator calling translators as tools. While a translator runs,
//! the outer run only waits: the user sees nothing until the tool returns. `on_stream` closes that
//! gap. The nested agent then runs in streaming mode, and every event of the nested run (answer
//! text as it is generated, its own tool calls, its handoffs) goes to your handler while the outer
//! run waits for the tool's result. The result itself is unchanged.
//!
//! Each `AgentToolStreamEvent` says which nested agent produced it and which tool call started
//! the nested run, so one handler can serve several agent-tools, or several parallel calls of the
//! same one. The handler is awaited per event: a slow handler slows the nested run, it does not
//! queue events without bound.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example agents_as_tools_stream
//! EXAMPLE_INPUT="Translate 'Good morning' to French and Spanish." \
//!   cargo run -p opai-agents-examples --example agents_as_tools_stream
//! ```

mod common;

use std::sync::{Arc, Mutex};

use opai_agents::{Agent, AsToolConfig, RunOptions, Runner};

use common::stream::Printer;

fn translator(
    language: &str,
    model: &Arc<dyn opai_agents::Model>,
    printer: &Arc<Mutex<Printer>>,
) -> opai_agents::FunctionTool {
    let printer = Arc::clone(printer);
    let last_call = Arc::new(Mutex::new(String::new()));
    Agent::new(format!("{language}_translator"))
        .instructions(format!(
            "You translate the user's message to {language}. Reply with the translation only."
        ))
        .model(Arc::clone(model))
        .as_tool(
            AsToolConfig {
                name: Some(format!("translate_to_{}", language.to_lowercase())),
                description: Some(format!("Translate the user's message to {language}")),
                ..Default::default()
            }
            .on_stream(move |update| {
                let printer = Arc::clone(&printer);
                let last_call = Arc::clone(&last_call);
                async move {
                    let mut printer = printer.lock().unwrap();
                    let mut last = last_call.lock().unwrap();
                    // A heading where a nested run starts, then its text as it arrives.
                    if *last != update.tool_call_id {
                        *last = update.tool_call_id.clone();
                        println!("\n[{} via {}]", update.agent_name, update.tool_name);
                    }
                    let _ = printer.feed(&update.event);
                }
            }),
        )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = common::model()?;
    let printer = Arc::new(Mutex::new(Printer::text_only()));

    let orchestrator = Agent::new("orchestrator_agent")
        .instructions(
            "You are a translation agent. You use the tools given to you to translate. \
             If asked for multiple translations, you call the relevant tools in order. \
             Then list the translations. You never translate on your own.",
        )
        .model(Arc::clone(&model))
        .tools(vec![
            translator("French", &model, &printer),
            translator("Spanish", &model, &printer),
        ]);

    let input = common::example_input("Translate 'Hello, world!' to French and Spanish.");
    let result = Runner::run(&orchestrator, input, RunOptions::default()).await?;

    println!(
        "\n\nFinal response:\n{}",
        result.final_output_as_str().unwrap_or_default()
    );
    Ok(())
}
