//! Zero configuration: one line to a running agent (Rust-only, X-015)
//!
//! No key yet? The offline model answers locally, so you can see the loop work before you pay for
//! anything. Then set one variable and run the same line again.
//!
//! ```bash
//! AGENTS_OFFLINE=1 cargo run -p opai-agents-examples --example zero_config
//! OPENAI_API_KEY=sk-... OPENAI_MODEL=gpt-4o cargo run -p opai-agents-examples --example zero_config
//! DEEPSEEK_API_KEY=sk-... DEEPSEEK_MODEL=deepseek-chat cargo run -p opai-agents-examples --example zero_config
//! ```
//!
//! `opai_agents::run` picks the provider from the environment: a preset whose key is set, an
//! endpoint declared with `AGENTS_PROVIDER_1_ENDPOINT`, or the offline model with
//! `AGENTS_OFFLINE=1`.

use opai_agents::prelude::*;

#[tokio::main]
async fn main() -> Result<(), AgentsError> {
    match opai_agents::run("You answer in one short sentence.", "What is Rust?").await {
        Ok(result) => println!("{}", result.final_output_as_str().unwrap_or_default()),
        Err(e) => {
            // Nothing configured is a normal way to start: say what to set.
            eprintln!("{e}");
            eprintln!(
                "\nSet one of OPENAI_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, \
                 OPENROUTER_API_KEY or ANTHROPIC_API_KEY, or declare an endpoint with \
                 AGENTS_PROVIDER_1_ENDPOINT — or run this example with AGENTS_OFFLINE=1 to see \
                 it work without any of them."
            );
        }
    }
    Ok(())
}
