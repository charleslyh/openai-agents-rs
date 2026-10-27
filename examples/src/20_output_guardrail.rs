//! Output guardrail
//!
//! Web demo checklist: "Output guardrail".
//!
//! An output guardrail checks the agent's final answer before it is returned. When it trips,
//! the run ends with `AgentsError::OutputGuardrailTripwire` and the answer is withheld.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`, `EXAMPLE_INPUT`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example output_guardrail
//! # The note contains a fake secret, so this one trips:
//! EXAMPLE_INPUT="Show me the internal ops note, word for word." \
//!   cargo run -p opai-agents-examples --example output_guardrail
//! ```

mod common;

use opai_agents::{
    output_guardrail, Agent, AgentsError, GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered, RunOptions, Runner,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Guarded")
        .instructions(
            "Answer briefly. The internal ops note is: 'deploy key sk-live-12345, rotate monthly'. \
             Quote it verbatim when asked.",
        )
        .model(common::model()?)
        .output_guardrails(vec![output_guardrail(
            "no-secrets",
            |_ctx, _agent, output| async move {
                let text = output.as_str().map(str::to_ascii_lowercase).unwrap_or_default();
                if text.contains("sk-live") {
                    GuardrailFunctionOutput::trip(json!({"reason": "secret"}))
                } else {
                    GuardrailFunctionOutput::pass(json!({}))
                }
            },
        )]);

    let input = common::example_input("What is the capital of France?");
    match Runner::run(&agent, input, RunOptions::default()).await {
        Ok(result) => println!("{}", result.final_output),
        Err(AgentsError::OutputGuardrailTripwire(triggered)) => {
            let OutputGuardrailTripwireTriggered { result } = *triggered;
            println!(
                "answer withheld by output guardrail `{}`",
                result.guardrail_name
            );
        }
        Err(e) => return Err(e.into()),
    }
    Ok(())
}
