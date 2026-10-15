//! Builds the model from the environment.
//!
//! - `OPENAI_API_KEY` (required)
//! - `OPENAI_MODEL` (required)
//! - `OPENAI_BASE_URL` (optional)
//! - `OPENAI_API` = `chat_completions` (default) | `responses`

use std::sync::Arc;

use openai_agents::{
    set_default_openai_api, DefaultOpenAiApi, Model, OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
};

pub fn model() -> Result<Arc<dyn Model>, Box<dyn std::error::Error>> {
    let raw = std::env::var("OPENAI_API").unwrap_or_else(|_| "chat_completions".into());
    let responses = match raw.to_ascii_lowercase().as_str() {
        "chat" | "chat_completions" | "completions" | "chat-completions" => false,
        "responses" | "response" => true,
        other => {
            return Err(format!(
                "invalid OPENAI_API={other:?}; expected chat_completions|responses"
            )
            .into())
        }
    };
    let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY is required")?;
    let model_name = std::env::var("OPENAI_MODEL").map_err(|_| "OPENAI_MODEL is required")?;
    let base_url = std::env::var("OPENAI_BASE_URL").ok();
    eprintln!(
        "using api={} model={model_name} base={}",
        if responses {
            "responses"
        } else {
            "chat_completions"
        },
        base_url.as_deref().unwrap_or("(openai default)")
    );
    if responses {
        set_default_openai_api(DefaultOpenAiApi::Responses);
        Ok(Arc::new(OpenAIResponsesModel::new(
            model_name,
            api_key,
            base_url.as_deref(),
        )))
    } else {
        set_default_openai_api(DefaultOpenAiApi::ChatCompletions);
        Ok(Arc::new(OpenAIChatCompletionsModel::new(
            model_name,
            api_key,
            base_url.as_deref(),
        )))
    }
}
