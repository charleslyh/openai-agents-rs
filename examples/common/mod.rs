//! Shared helpers for live OpenAI examples (env + confirm prompts).
//!
//! Model selection via `OPENAI_API` (example-harness default: chat completions).
//!
//! Note: the crate's own default is the Responses API (Python-aligned); this harness overrides
//! it because most OpenAI-compatible gateways implement Chat Completions SSE first.
//! - `chat` / `chat_completions` / `completions` → Chat Completions
//! - `responses` → Responses API

#![allow(dead_code)]

use std::io::{self, Write};
use std::sync::Arc;

use openai_agents::{
    set_default_openai_api, DefaultOpenAiApi, Model, OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
};

/// Which OpenAI HTTP API to use for examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExampleApi {
    /// Chat Completions (`/v1/chat/completions`).
    ChatCompletions,
    /// Responses (`/v1/responses`).
    Responses,
}

impl ExampleApi {
    /// Parse from `OPENAI_API` env (default: chat completions).
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        let raw = std::env::var("OPENAI_API").unwrap_or_else(|_| "chat_completions".into());
        match raw.to_ascii_lowercase().as_str() {
            "chat" | "chat_completions" | "completions" | "chat-completions" => {
                Ok(Self::ChatCompletions)
            }
            "responses" | "response" => Ok(Self::Responses),
            other => Err(format!(
                "invalid OPENAI_API={other:?}; expected chat_completions|responses"
            )
            .into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::ChatCompletions => "chat_completions",
            Self::Responses => "responses",
        }
    }
}

/// Build a model from env. Default API is **chat completions**.
///
/// Env:
/// - `OPENAI_API_KEY` (required)
/// - `OPENAI_MODEL` (required)
/// - `OPENAI_BASE_URL` (optional)
/// - `OPENAI_API` = `chat_completions` (default) | `responses`
pub fn model() -> Result<Arc<dyn Model>, Box<dyn std::error::Error>> {
    let api = ExampleApi::from_env()?;
    let (api_key, model_name, base_url) = openai_env(api)?;
    match api {
        ExampleApi::ChatCompletions => {
            set_default_openai_api(DefaultOpenAiApi::ChatCompletions);
            Ok(Arc::new(OpenAIChatCompletionsModel::new(
                model_name,
                api_key,
                base_url.as_deref(),
            )))
        }
        ExampleApi::Responses => {
            set_default_openai_api(DefaultOpenAiApi::Responses);
            Ok(Arc::new(OpenAIResponsesModel::new(
                model_name,
                api_key,
                base_url.as_deref(),
            )))
        }
    }
}

fn openai_env(
    api: ExampleApi,
) -> Result<(String, String, Option<String>), Box<dyn std::error::Error>> {
    let api_key =
        std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY is required")?;
    let model_name = std::env::var("OPENAI_MODEL")
        .map_err(|_| "OPENAI_MODEL is required (e.g. deepseek-v4-pro)")?;
    let base_url = std::env::var("OPENAI_BASE_URL").ok();
    eprintln!(
        "using api={} model={model_name} base={}",
        api.as_str(),
        base_url.as_deref().unwrap_or("(openai default)")
    );
    Ok((api_key, model_name, base_url))
}

/// Yes/no confirm. `HITL_AUTO=approve|reject` skips the prompt (for CI / scripting).
pub fn confirm(question: &str) -> io::Result<bool> {
    if let Ok(auto) = std::env::var("HITL_AUTO") {
        return Ok(matches!(
            auto.to_ascii_lowercase().as_str(),
            "approve" | "y" | "yes" | "1" | "true"
        ));
    }
    print!("{question} [y/N]: ");
    io::stdout().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

/// Read a line, or use `fallback` when empty / `EXAMPLE_INPUT` is set.
pub fn input_with_fallback(prompt: &str, fallback: &str) -> io::Result<String> {
    if let Ok(v) = std::env::var("EXAMPLE_INPUT") {
        if !v.trim().is_empty() {
            eprintln!("{prompt}{v}");
            return Ok(v);
        }
    }
    print!("{prompt}");
    io::stdout().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let line = line.trim().to_string();
    if line.is_empty() {
        eprintln!("(using fallback) {fallback}");
        Ok(fallback.to_string())
    } else {
        Ok(line)
    }
}
