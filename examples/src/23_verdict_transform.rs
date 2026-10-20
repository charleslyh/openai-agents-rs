//! Guardrail verdicts: redact instead of refuse (X-012)
//!
//! Python's guardrails can only pass or trip the run. A Rust guardrail can also return
//! `Verdict::Transform`, so the run continues with rewritten content: the model and the session
//! receive the redacted text, not the original.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example verdict_transform
//! ```

mod common;

use std::sync::Arc;

use opai_agents::prelude::*;
use opai_agents::{InMemorySession, Session};

/// Replace anything that looks like a payment card number.
fn redact(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut digits = 0usize;
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            digits += 1;
        } else if ch == ' ' || ch == '-' {
            // Runs of digits separated only by spaces or dashes count as one number.
        } else {
            digits = 0;
        }
        out.push(if digits >= 4 { '•' } else { ch });
    }
    out
}

/// An input guardrail that rewrites the user's text before the model sees it. Only guardrails
/// built with `run_in_parallel(false)` may transform: a parallel one runs while the model is
/// already reading the original.
fn redactor() -> InputGuardrail {
    input_guardrail("redact-card", |_ctx, _agent, input| async move {
        let text = match &input {
            InputLike::Text(text) => text.clone(),
            InputLike::Items(items) => serde_json::to_string(items).unwrap_or_default(),
        };
        let redacted = redact(&text);
        if redacted == text {
            GuardrailFunctionOutput::pass(serde_json::Value::Null)
        } else {
            GuardrailFunctionOutput::transform(redacted)
        }
    })
    .run_in_parallel(false)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = Arc::new(InMemorySession::new("verdict-transform"));
    let agent = Agent::new("Billing")
        .instructions("Repeat the card number back to the user.")
        .model(common::model()?)
        .input_guardrails(vec![redactor()]);

    let input = "My card is 4111 1111 1111 1111, please confirm it.";
    let result = Runner::run(&agent, input, RunOptions::new().session(session.clone())).await?;

    println!(
        "final output: {}",
        result.final_output_as_str().unwrap_or_default()
    );
    // What the session stored is redacted too: the input is only saved after the blocking
    // guardrails have run, so there was still something to replace.
    let stored = serde_json::to_string(&session.get_items(None).await?)?;
    println!("session:      {stored}");
    assert!(
        !stored.contains("4111 1111"),
        "the card number must not be stored"
    );
    Ok(())
}
