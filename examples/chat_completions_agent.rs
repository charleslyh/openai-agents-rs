//! Chat Completions agent example.
//!
//! Requires `OPENAI_API_KEY`. Run with:
//! `cargo run --example chat_completions_agent --features openai`

use std::sync::Arc;

use openai_agents::{
    set_default_openai_api, Agent, DefaultOpenAiApi, OpenAIChatCompletionsModel, RunOptions,
    Runner,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_default_openai_api(DefaultOpenAiApi::ChatCompletions);
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let model = Arc::new(OpenAIChatCompletionsModel::new(
        "gpt-4.1-mini",
        api_key,
        None,
    ));
    let agent = Agent::new("Assistant")
        .instructions("You are a concise assistant.")
        .model(model);

    let result = Runner::run(&agent, "Say hello in one short sentence.", RunOptions::default()).await?;
    println!("{}", result.final_output);
    Ok(())
}
