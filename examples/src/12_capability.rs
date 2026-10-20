//! Named capabilities: a bundle of tools, instructions and guardrails under one name (X-007)
//!
//! A `Capability` contributes parts to an agent: tools, an instruction paragraph, input/output
//! guardrails, default model settings and required extension types. Mounting one is the same as
//! handing over its parts by hand - both agents below behave identically.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p openai-agents-examples --example capability
//! ```

mod common;

use std::sync::Arc;

use openai_agents::prelude::*;
use openai_agents::{Capability, CapabilityParts, FunctionTool};

/// A named bundle: two tools, a paragraph of instructions and an output guardrail.
struct MathCapability;

impl Capability for MathCapability {
    fn name(&self) -> &str {
        "math"
    }

    fn parts(&self) -> CapabilityParts {
        CapabilityParts::new()
            .tools(vec![
                FunctionTool::constant("add", "Add two numbers", "0"),
                FunctionTool::constant("multiply", "Multiply two numbers", "0"),
            ])
            .instructions("Answer with a number only.")
            .output_guardrail(output_guardrail(
                "numeric",
                |_ctx, _agent, output| async move {
                    let text = output.as_str().unwrap_or_default().trim().to_string();
                    if text.chars().all(|c| c.is_ascii_digit() || c == '.') {
                        GuardrailFunctionOutput::pass(serde_json::Value::Null)
                    } else {
                        GuardrailFunctionOutput::fail("not a number", Severity::Medium)
                    }
                },
            ))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = common::model()?;

    // Mounted as a bundle.
    let with_capability = Agent::new("Support")
        .instructions("You are a helpful assistant.")
        .model(Arc::clone(&model))
        .capability(MathCapability);

    // The same parts, handed over by hand.
    let by_hand = Agent::new("Support")
        .instructions("You are a helpful assistant.")
        .model(Arc::clone(&model))
        .tools(MathCapability.parts().tools.clone())
        .output_guardrails(MathCapability.parts().output_guardrails.clone());

    let input = "What is 6 times 7?";
    for (label, agent) in [("mounted", &with_capability), ("by hand", &by_hand)] {
        match Runner::run(agent, input, RunOptions::new()).await {
            Ok(result) => println!(
                "{label}: {}",
                result.final_output_as_str().unwrap_or_default()
            ),
            Err(e) => println!("{label}: {e}"),
        }
    }
    Ok(())
}
