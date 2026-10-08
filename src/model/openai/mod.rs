//! OpenAI model adapters built on `async-openai` config + HTTP.

pub mod chat_convert;
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
    // `extra_args` / `extra_body` are applied by the callers, after every mapped field, so that
    // `extra_body` keeps the highest precedence (openai SDK `_merge_mappings`).
}

/// Apply `ModelSettings.extra_args`, filling only keys nothing else has set.
///
/// Python merges `extra_args` into the API call kwargs and raises `TypeError` when a key is
/// already provided (`openai_chatcompletions.py:725`, `openai_responses.py:1059`). Rust has no
/// kwargs layer, so the same collision is reported as [`ModelError::Behavior`] with Python's
/// message. `extra_body` is applied afterwards and still wins.
pub(crate) fn apply_extra_args(
    body: &mut serde_json::Value,
    settings: &ModelSettings,
    api: &str,
) -> Result<(), ModelError> {
    let Some(extra) = settings.extra_args.as_ref() else {
        return Ok(());
    };
    let mut duplicates: Vec<&str> = extra
        .keys()
        .filter(|key| body.get(key.as_str()).is_some())
        .map(String::as_str)
        .collect();
    if !duplicates.is_empty() {
        duplicates.sort_unstable();
        let keys = duplicates
            .iter()
            .map(|k| format!("'{k}'"))
            .collect::<Vec<_>>()
            .join(", ");
        let noun = if duplicates.len() == 1 {
            "keyword argument"
        } else {
            "keyword arguments"
        };
        return Err(ModelError::Behavior(format!(
            "extra_args: {api}.create() got multiple values for {noun} {keys}"
        )));
    }
    if let Some(base) = body.as_object_mut() {
        for (key, value) in extra {
            base.insert(key.clone(), value.clone());
        }
    }
    Ok(())
}

/// Apply `ModelSettings.extra_body`, overriding whatever the request already holds.
///
/// Python passes this as the OpenAI SDK's nested `extra_body` argument (`docs/models/index.md`:
/// "remains a nested `extra_body` argument"), which the SDK merges over the request body, so it
/// has the highest precedence — above both mapped settings and [`apply_extra_args`].
pub(crate) fn apply_extra_body(body: &mut serde_json::Value, settings: &ModelSettings) {
    if let (Some(base), Some(serde_json::Value::Object(extra))) =
        (body.as_object_mut(), settings.extra_body.as_ref())
    {
        for (key, value) in extra {
            base.insert(key.clone(), value.clone());
        }
    }
}

/// Chat Completions mapping (Python: `OpenAIChatCompletionsModel`).
pub(crate) fn apply_model_settings_chat(
    body: &mut serde_json::Value,
    settings: &ModelSettings,
) -> Result<(), ModelError> {
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
    apply_extra_args(body, settings, "chat.completions")?;
    apply_extra_body(body, settings);
    Ok(())
}

/// Responses API mapping (Python: `OpenAIResponsesModel`).
pub(crate) fn apply_model_settings_responses(
    body: &mut serde_json::Value,
    settings: &ModelSettings,
) -> Result<(), ModelError> {
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
    apply_extra_args(body, settings, "responses")?;
    apply_extra_body(body, settings);
    Ok(())
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

/// Classify a `reqwest` failure: connection, timeout and body-read errors are the ones a retry
/// policy may replay; an undecodable body is not (Python: `APIConnectionError` / `APITimeoutError`).
pub(crate) fn map_reqwest(err: reqwest::Error) -> ModelError {
    if err.is_timeout() {
        return ModelError::Connection(crate::error::ModelConnectionError {
            message: err.to_string(),
            is_timeout: true,
        });
    }
    if err.is_connect() || err.is_request() || err.is_body() {
        return ModelError::Connection(crate::error::ModelConnectionError {
            message: err.to_string(),
            is_timeout: false,
        });
    }
    ModelError::Transport(err.to_string())
}

/// Turn a non-success HTTP response into a [`ModelError::Status`] that keeps the status, headers
/// and body the retry layer reads (`Retry-After`, `x-should-retry`, the error code).
pub(crate) async fn status_error(label: &str, resp: reqwest::Response) -> ModelError {
    let status = resp.status();
    let headers = resp
        .headers()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|v| (name.as_str().to_ascii_lowercase(), v.to_string()))
        })
        .collect();
    let text = resp.text().await.unwrap_or_default();
    let body = serde_json::from_str::<serde_json::Value>(&text).unwrap_or_else(|_| {
        if text.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(text)
        }
    });
    ModelError::Status(crate::error::ModelStatusError {
        status_code: status.as_u16(),
        message: format!("{label} status={status} body={body}"),
        body,
        headers,
    })
}

/// Whether a response carries a plain JSON body instead of an SSE stream.
///
/// Some gateways (and wiremock) ignore `stream: true` and answer with the full object, so
/// adapters must fall back to the non-streaming parse.
pub(crate) fn is_json_response(resp: &reqwest::Response) -> bool {
    resp.headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase()
        .contains("application/json")
}

/// Incremental SSE parser shared by the Chat Completions and Responses adapters.
///
/// OpenAI streams `text/event-stream` bodies as `data: {...}` lines separated by blank lines.
/// `[DONE]` terminators and non-`data:` lines (comments, `event:`) are skipped.
pub(crate) struct SseReader {
    buffer: String,
}

impl SseReader {
    /// Start with an empty buffer.
    pub(crate) fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Append raw bytes from the HTTP byte stream.
    pub(crate) fn feed(&mut self, chunk: &[u8]) {
        self.buffer.push_str(&String::from_utf8_lossy(chunk));
    }

    /// Drain the next complete `data:` payload, if one is buffered.
    pub(crate) fn next_event(&mut self) -> Option<Result<serde_json::Value, ModelError>> {
        while let Some(pos) = self.buffer.find('\n') {
            let mut line: String = self.buffer[..pos].to_string();
            self.buffer.drain(..=pos);
            if line.ends_with('\r') {
                line.pop();
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                continue;
            }
            return Some(
                serde_json::from_str(data).map_err(|e| {
                    ModelError::Behavior(format!("invalid SSE JSON: {e}; data={data}"))
                }),
            );
        }
        None
    }
}
