//! Chat Completions agent example.
//!
//! Required env: `OPENAI_API_KEY`, `OPENAI_MODEL`
//! Optional: `OPENAI_BASE_URL`

use std::sync::Arc;

use openai_agents::{
    set_default_openai_api, Agent, DefaultOpenAiApi, OpenAIChatCompletionsModel, RunOptions,
    Runner,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_default_openai_api(DefaultOpenAiApi::ChatCompletions);

    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY is required")?;
    let model_name = std::env::var("OPENAI_MODEL")
        .map_err(|_| "OPENAI_MODEL is required (e.g. deepseek-v4-pro for TokenHub)")?;
    let base_url = std::env::var("OPENAI_BASE_URL").ok();

    eprintln!(
        "using model={model_name} base={}",
        base_url.as_deref().unwrap_or("(openai default)")
    );

    let model = Arc::new(OpenAIChatCompletionsModel::new(
        model_name,
        api_key,
        base_url.as_deref(),
    ));
    let agent = Agent::new("Assistant")
        .instructions("You are a concise assistant.")
        .model(model);

    let result =
        Runner::run(&agent, "Say hello in one short sentence.", RunOptions::default()).await?;
    println!("{}", result.final_output);
    Ok(())
}
