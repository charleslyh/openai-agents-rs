//! Model middleware: rewrite what leaves the process (Rust-only)
//!
//! A `ModelMiddleware` sees each turn's model input (instructions and input items) right before
//! the request is sent, and returns the version to send. Several of them compose in the order they
//! were added, so unrelated concerns live in separate types instead of one big closure:
//!
//! * `RedactEmails` replaces e-mail addresses in the conversation with `[email]`, so personal data
//!   never reaches the provider.
//! * `AddNote` appends a line to the system instructions.
//!
//! Only the request is rewritten. The conversation the caller holds (the `RunResult` input, the
//! session) keeps the original text; the model alone sees the redacted one.
//!
//! The model is scripted so the example needs no key, and so it can print exactly what the model
//! was given.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example model_middleware
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{
    Agent, AgentsError, ModelInputData, ModelMiddleware, ModelOptions, RunConfig,
    RunContextWrapper, RunOptions, Runner,
};
use serde_json::Value;

/// Replaces every word that looks like an e-mail address.
struct RedactEmails;

fn redact(text: &str) -> String {
    text.split(' ')
        .map(|word| if word.contains('@') { "[email]" } else { word })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Redacts the `content` / `text` strings anywhere in an input item.
fn redact_value(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                match child {
                    Value::String(s) if key == "content" || key == "text" => *s = redact(s),
                    other => redact_value(other),
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(redact_value),
        _ => {}
    }
}

#[async_trait]
impl ModelMiddleware for RedactEmails {
    fn name(&self) -> &str {
        "redact_emails"
    }

    async fn prepare(
        &self,
        mut data: ModelInputData,
        _agent: &Agent,
        _context: &RunContextWrapper,
    ) -> Result<ModelInputData, AgentsError> {
        data.input.iter_mut().for_each(redact_value);
        Ok(data)
    }
}

/// Appends a fixed line to the system instructions.
struct AddNote(&'static str);

#[async_trait]
impl ModelMiddleware for AddNote {
    fn name(&self) -> &str {
        "add_note"
    }

    async fn prepare(
        &self,
        mut data: ModelInputData,
        _agent: &Agent,
        _context: &RunContextWrapper,
    ) -> Result<ModelInputData, AgentsError> {
        data.instructions = Some(match data.instructions {
            Some(existing) => format!("{existing}\n{}", self.0),
            None => self.0.to_string(),
        });
        Ok(data)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("I will write to the address you gave."),
    )]));
    let agent = Agent::new("Assistant")
        .instructions("You help users draft short e-mails.")
        .model(Arc::clone(&model) as Arc<dyn opai_agents::Model>);

    let options = RunOptions::new().config(
        RunConfig::default().with_model(
            ModelOptions::new()
                .middleware(RedactEmails)
                .middleware(AddNote("Never repeat personal data.")),
        ),
    );

    let input = "Write to alice@example.com about the invoice.";
    println!("caller input : {input}");
    let result = Runner::run(&agent, input, options).await?;
    println!(
        "agent output : {}",
        result.final_output_as_str().unwrap_or_default()
    );

    let call = model.first_call().expect("the model was called once");
    println!("\nwhat the model received:");
    println!(
        "  instructions: {}",
        call.system_instructions
            .unwrap_or_default()
            .replace('\n', " | ")
    );
    println!("  input       : {}", call.input);
    Ok(())
}
