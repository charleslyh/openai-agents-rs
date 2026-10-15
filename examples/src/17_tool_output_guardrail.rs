//! Tool output guardrail
//!
//! Web demo checklist: "Tool output guardrail".
//!
//! `pii-redactor` runs after `lookup_customer` and replaces emails and phone numbers in the
//! tool result before the model sees it.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p openai-agents-examples --example tool_output_guardrail
//! ```

mod common;

use openai_agents::{
    Agent, FunctionTool, RunOptions, Runner, ToolGuardrailFunctionOutput, ToolOutputGuardrail,
};
use serde_json::{json, Value};

fn pii_redactor() -> ToolOutputGuardrail {
    ToolOutputGuardrail::new("pii-redactor", |data| async move {
        let text = match &data.output {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let mut changed = false;
        let redacted: Vec<String> = text
            .split(' ')
            .map(|word| {
                if word.contains('@') {
                    changed = true;
                    "[email]".to_string()
                } else if word.chars().filter(char::is_ascii_digit).count() >= 7 {
                    changed = true;
                    "[phone]".to_string()
                } else {
                    word.to_string()
                }
            })
            .collect();
        if changed {
            println!("[guardrail] pii-redactor redacted the tool result");
            ToolGuardrailFunctionOutput::reject_content(
                redacted.join(" "),
                json!({"redacted": true}),
            )
        } else {
            ToolGuardrailFunctionOutput::allow(json!({"redacted": false}))
        }
    })
}

fn lookup_customer() -> FunctionTool {
    FunctionTool::new(
        "lookup_customer",
        "Look up a customer's billing record by name",
        json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": false
        }),
        |_ctx, _args| async move {
            Ok(json!(
                "Customer Alice: plan Pro, balance 0 USD, email alice@example.com, phone +14155550123"
            ))
        },
    )
    .with_tool_output_guardrails(vec![pii_redactor()])
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Billing")
        .instructions("Use lookup_customer and quote the record exactly as the tool returned it.")
        .model(common::model()?)
        .tools(vec![lookup_customer()]);

    let input = "Look up the customer record for Alice and quote it exactly.";
    let result = Runner::run(&agent, input, RunOptions::default()).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    println!(
        "tool output guardrail results: {}",
        result.tool_output_guardrail_results.len()
    );
    Ok(())
}
