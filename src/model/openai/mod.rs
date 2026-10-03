//! OpenAI model adapters built on `async-openai` config + HTTP.

mod chat_completions;
mod responses;

pub use chat_completions::OpenAIChatCompletionsModel;
pub use responses::OpenAIResponsesModel;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_openai::config::{Config, OpenAIConfig};

use crate::error::{ModelError, UserError};
use crate::model::provider::ModelProvider;
use crate::model::Model;
use crate::model_settings::ModelSettings;
use crate::tool::FunctionTool;
use crate::usage::Usage;

/// Resolves OpenAI model names into [`Model`] instances (Python: `OpenAIProvider`).
///
/// The API used depends on [`crate::get_default_openai_api`]: Responses by default,
/// Chat Completions when the default was switched.
#[derive(Clone)]
pub struct OpenAIProvider {
    api_key: String,
    base_url: Option<String>,
    default_model: Option<String>,
    force_chat_completions: bool,
    cache: Arc<Mutex<HashMap<String, Arc<dyn Model>>>>,
}

impl std::fmt::Debug for OpenAIProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never render the API key.
        f.debug_struct("OpenAIProvider")
            .field("base_url", &self.base_url)
            .field("default_model", &self.default_model)
            .field("force_chat_completions", &self.force_chat_completions)
            .finish()
    }
}

impl OpenAIProvider {
    /// Build a provider from explicit settings.
    pub fn new(
        api_key: impl Into<String>,
        base_url: Option<&str>,
        default_model: Option<&str>,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.map(str::to_string),
            default_model: default_model.map(str::to_string),
            force_chat_completions: false,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Build from `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `OPENAI_MODEL`.
    ///
    /// Secrets are only read from the environment or passed explicitly; they are never logged.
    pub fn from_env() -> Result<Self, UserError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            UserError::new("OPENAI_API_KEY is not set; pass an API key to OpenAIProvider::new")
        })?;
        if api_key.trim().is_empty() {
            return Err(UserError::new("OPENAI_API_KEY is empty"));
        }
        Ok(Self::new(
            api_key,
            std::env::var("OPENAI_BASE_URL").ok().as_deref(),
            std::env::var("OPENAI_MODEL").ok().as_deref(),
        ))
    }

    /// Always resolve to Chat Completions models regardless of the global default.
    pub fn always_chat_completions(mut self) -> Self {
        self.force_chat_completions = true;
        self
    }
}

impl ModelProvider for OpenAIProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>, UserError> {
        let name = model_name
            .or(self.default_model.as_deref())
            .ok_or_else(|| {
                UserError::new(
                    "No model name provided; set `Agent.model_name`, `RunConfig.model` or \
                     `OPENAI_MODEL`",
                )
            })?
            .to_string();

        if let Ok(cache) = self.cache.lock() {
            if let Some(model) = cache.get(&name) {
                return Ok(Arc::clone(model));
            }
        }

        let use_responses = !self.force_chat_completions
            && matches!(
                crate::run::get_default_openai_api(),
                crate::run::DefaultOpenAiApi::Responses
            );
        let model: Arc<dyn Model> = if use_responses {
            Arc::new(OpenAIResponsesModel::new(
                name.clone(),
                self.api_key.clone(),
                self.base_url.as_deref(),
            ))
        } else {
            Arc::new(OpenAIChatCompletionsModel::new(
                name.clone(),
                self.api_key.clone(),
                self.base_url.as_deref(),
            ))
        };

        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(name, Arc::clone(&model));
        }
        Ok(model)
    }
}

/// Shared OpenAI-compatible endpoint configuration.
#[derive(Clone)]
pub struct OpenAiEndpoint {
    /// async-openai config (api key + base URL).
    pub config: OpenAIConfig,
    /// HTTP client used for Requests.
    pub http: reqwest::Client,
    api_key: String,
}

impl std::fmt::Debug for OpenAiEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The API key must never reach logs or traces.
        f.debug_struct("OpenAiEndpoint")
            .field("api_base", &self.config.api_base())
            .finish()
    }
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

/// Settings shared by both OpenAI APIs.
pub(crate) fn apply_model_settings_common(body: &mut serde_json::Value, settings: &ModelSettings) {
    if let Some(t) = settings.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    if let Some(p) = settings.top_p {
        body["top_p"] = serde_json::json!(p);
    }
    if let Some(f) = settings.frequency_penalty {
        body["frequency_penalty"] = serde_json::json!(f);
    }
    if let Some(f) = settings.presence_penalty {
        body["presence_penalty"] = serde_json::json!(f);
    }
    if let Some(tc) = &settings.tool_choice {
        body["tool_choice"] = tc.to_request_value();
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
    if let Some(extra) = &settings.extra_args {
        if let Some(base) = body.as_object_mut() {
            for (k, v) in extra {
                base.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    }
}

/// Chat Completions mapping (Python: `OpenAIChatCompletionsModel`).
pub(crate) fn apply_model_settings_chat(body: &mut serde_json::Value, settings: &ModelSettings) {
    apply_model_settings_common(body, settings);
    if let Some(m) = settings.max_tokens {
        body["max_tokens"] = serde_json::json!(m);
    }
    if let Some(t) = settings.top_logprobs {
        body["top_logprobs"] = serde_json::json!(t);
    }
    if let Some(s) = settings.store {
        body["store"] = serde_json::json!(s);
    }
    if let Some(meta) = &settings.metadata {
        body["metadata"] = serde_json::Value::Object(meta.clone());
    }
}

/// Responses API mapping (Python: `OpenAIResponsesModel`).
pub(crate) fn apply_model_settings_responses(
    body: &mut serde_json::Value,
    settings: &ModelSettings,
) {
    apply_model_settings_common(body, settings);
    // Responses API names this `max_output_tokens`.
    if let Some(m) = settings.max_tokens {
        body["max_output_tokens"] = serde_json::json!(m);
    }
    if let Some(t) = settings.truncation {
        body["truncation"] = serde_json::json!(match t {
            crate::model_settings::Truncation::Auto => "auto",
            crate::model_settings::Truncation::Disabled => "disabled",
        });
    }
    if let Some(r) = &settings.reasoning {
        body["reasoning"] = r.clone();
    }
    if let Some(v) = settings.verbosity {
        body["verbosity"] = serde_json::json!(match v {
            crate::model_settings::Verbosity::Low => "low",
            crate::model_settings::Verbosity::Medium => "medium",
            crate::model_settings::Verbosity::High => "high",
        });
    }
    if let Some(meta) = &settings.metadata {
        body["metadata"] = serde_json::Value::Object(meta.clone());
    }
    if let Some(s) = settings.store {
        body["store"] = serde_json::json!(s);
    }
    if let Some(t) = settings.top_logprobs {
        body["top_logprobs"] = serde_json::json!(t);
    }
    if let Some(inc) = &settings.response_include {
        body["include"] = serde_json::json!(inc);
    }
}

/// Apply `timeout` / `extra_headers` to an outgoing request.
pub(crate) fn decorate_request(
    request: reqwest::RequestBuilder,
    settings: &ModelSettings,
) -> reqwest::RequestBuilder {
    let mut request = request;
    if let Some(t) = settings.timeout {
        if t > 0.0 {
            request = request.timeout(std::time::Duration::from_secs_f32(t));
        }
    }
    if let Some(headers) = &settings.extra_headers {
        for (k, v) in headers {
            if let Some(v) = v.as_str() {
                request = request.header(k.as_str(), v);
            }
        }
    }
    request
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
