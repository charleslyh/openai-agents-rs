//! OpenAI model adapters built on `async-openai` config + HTTP.

mod chat_completions;
mod responses;

pub use chat_completions::OpenAIChatCompletionsModel;
pub use responses::OpenAIResponsesModel;

use async_openai::config::{Config, OpenAIConfig};

use crate::error::ModelError;
use crate::model_settings::ModelSettings;
use crate::tool::FunctionTool;
use crate::usage::Usage;

/// Shared OpenAI-compatible endpoint configuration.
#[derive(Clone)]
pub struct OpenAiEndpoint {
    /// async-openai config (api key + base URL).
    pub config: OpenAIConfig,
    /// HTTP client used for Requests.
    pub http: reqwest::Client,
    api_key: String,
}

impl OpenAiEndpoint {
    /// Build an endpoint targeting OpenAI or a mock base URL.
    pub fn new(api_key: impl Into<String>, base_url: Option<&str>) -> Self {
        let api_key = api_key.into();
        let mut config = OpenAIConfig::new().with_api_key(api_key.clone());
        if let Some(base) = base_url {
            config = config.with_api_base(base);
        }
        Self {
            config,
            http: reqwest::Client::new(),
            api_key,
        }
    }

    /// Absolute URL for a path like `/chat/completions` or `/responses`.
    pub fn url(&self, path: &str) -> String {
        let base = self.config.api_base().trim_end_matches('/');
        let path = if path.starts_with('/') {
            path.to_string()
        } else {
            format!("/{path}")
        };
        format!("{base}{path}")
    }

    /// API key string for Authorization header.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }
}

pub(crate) fn merge_usage(usage: Option<(u64, u64)>) -> Usage {
    match usage {
        Some((input, output)) => Usage::from_tokens(input, output),
        None => Usage::default(),
    }
}

pub(crate) fn apply_model_settings_chat(body: &mut serde_json::Value, settings: &ModelSettings) {
    if let Some(t) = settings.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    if let Some(p) = settings.top_p {
        body["top_p"] = serde_json::json!(p);
    }
    if let Some(m) = settings.max_tokens {
        body["max_tokens"] = serde_json::json!(m);
    }
    if let Some(tc) = &settings.tool_choice {
        body["tool_choice"] = match tc.as_str() {
            "auto" | "none" | "required" => serde_json::json!(tc),
            name => serde_json::json!({"type": "function", "function": {"name": name}}),
        };
    }
    if let Some(p) = settings.parallel_tool_calls {
        body["parallel_tool_calls"] = serde_json::json!(p);
    }
    if let Some(extra) = &settings.extra_body {
        if let (Some(base), Some(ext)) = (body.as_object_mut(), extra.as_object()) {
            for (k, v) in ext {
                base.insert(k.clone(), v.clone());
            }
        }
    }
}

pub(crate) fn tools_as_chat(tools: &[FunctionTool]) -> Vec<serde_json::Value> {
    tools.iter().map(|t| t.to_chat_tool()).collect()
}

pub(crate) fn tools_as_responses(tools: &[FunctionTool]) -> Vec<serde_json::Value> {
    tools.iter().map(|t| t.to_function_tool_param()).collect()
}

pub(crate) fn map_transport(err: impl std::fmt::Display) -> ModelError {
    ModelError::Transport(err.to_string())
}
