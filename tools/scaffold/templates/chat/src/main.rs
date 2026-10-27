//! A minimal agent: instructions, a model from the environment, one run.
//!
//! The provider comes from the environment: set `OPENAI_API_KEY` (or `DEEPSEEK_API_KEY`,
//! `GEMINI_API_KEY`, …), or run with `AGENTS_OFFLINE=1` to see the loop work without a key.

use opai_agents::prelude::*;

#[tokio::main]
async fn main() -> Result<(), AgentsError> {
    let agent = Agent::new("Assistant").instructions("You answer in one short paragraph.");
    let input = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Tell me what you can do.".to_string());

    let result = Runner::run(&agent, input, opai_agents::zero_config_options()?).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
