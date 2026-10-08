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
/// The API used depends on [`crate::get_default_openai_api`]: Chat Completions by default,
/// Responses when the default was switched or the provider was given `.api(Responses)`.
#[derive(Clone)]
pub struct OpenAIProvider {
    api_key: String,
    base_url: Option<String>,
    default_model: Option<String>,
    api: Option<crate::run::DefaultOpenAiApi>,
    cache: Arc<Mutex<HashMap<String, Arc<dyn Model>>>>,
}

impl std::fmt::Debug for OpenAIProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never render the API key.
        f.debug_struct("OpenAIProvider")
            .field("base_url", &self.base_url)
            .field("default_model", &self.default_model)
            .field("api", &self.api)
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
            api: None,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Build from `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `OPENAI_MODEL`.
    ///
    /// The key may be missing when `OPENAI_BASE_URL` points at another server: many local and
    /// gateway endpoints need no key, and no `Authorization` header is sent for an empty one.
    /// Secrets are only read from the environment or passed explicitly; they are never logged.
    pub fn from_env() -> Result<Self, UserError> {
        let base_url = std::env::var("OPENAI_BASE_URL").ok().filter(|v| !v.trim().is_empty());
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        if api_key.trim().is_empty() && base_url.is_none() {
            return Err(UserError::new(
                "OPENAI_API_KEY is not set; set it, set OPENAI_BASE_URL for a server that needs \
                 no key, or build a provider with OpenAIProvider::new / CompatibleProvider",
            ));
        }
        Ok(Self::new(
            api_key.trim(),
            base_url.as_deref(),
            std::env::var("OPENAI_MODEL").ok().as_deref(),
        ))
    }

    /// Always resolve to Chat Completions models regardless of the global default.
    pub fn always_chat_completions(self) -> Self {
        self.api(crate::run::DefaultOpenAiApi::ChatCompletions)
    }

    /// Use this API for every model of the provider, whatever the global default says.
    pub fn api(mut self, api: crate::run::DefaultOpenAiApi) -> Self {
        self.api = Some(api);
        self
    }

    /// The API a model gets (D-I): the provider's own choice, else `set_default_openai_api`,
    /// else Chat Completions.
    fn resolve_api(&self) -> crate::run::DefaultOpenAiApi {
        self.api.or_else(crate::run::explicit_default_openai_api).unwrap_or_default()
    }
}

/// A provider for any server that speaks the OpenAI Chat Completions or Responses protocol:
/// third-party hosts, gateways and local servers (vLLM, Ollama, llama.cpp, LiteLLM, ...).
///
/// Unlike [`OpenAIProvider`] it needs a base URL, takes the API key as optional, and defaults to
/// Chat Completions, which is what nearly every compatible server implements.
///
/// ```no_run
/// use std::sync::Arc;
/// use openai_agents::{CompatibleProvider, DefaultOpenAiApi, MultiProvider};
///
/// let router = MultiProvider::new()
///     .register("local", Arc::new(CompatibleProvider::new("http://localhost:8000/v1")))
///     .register(
///         "gateway",
///         Arc::new(
///             CompatibleProvider::new("https://gateway.example.com/v1")
///                 .api_key("secret")
///                 .api(DefaultOpenAiApi::Responses),
///         ),
///     );
/// // Agents then name models as "local/qwen2.5-7b" or "gateway/my-model".
/// ```
#[derive(Clone, Debug)]
pub struct CompatibleProvider {
    inner: OpenAIProvider,
}

impl CompatibleProvider {
    /// A provider for the server at `base_url` (for example `http://localhost:8000/v1`).
    pub fn new(base_url: impl AsRef<str>) -> Self {
        Self {
            inner: OpenAIProvider::new("", Some(base_url.as_ref()), None)
                .api(crate::run::DefaultOpenAiApi::ChatCompletions),
        }
    }

    /// Send `Authorization: Bearer <key>`; without a key no such header is sent.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.inner.api_key = key.into();
        self
    }

    /// The protocol to speak (default: Chat Completions).
    pub fn api(mut self, api: crate::run::DefaultOpenAiApi) -> Self {
        self.inner.api = Some(api);
        self
    }

    /// The model used when a name is not given.
    pub fn default_model(mut self, name: impl Into<String>) -> Self {
        self.inner.default_model = Some(name.into());
        self
    }
}

impl ModelProvider for CompatibleProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>, UserError> {
        self.inner.get_model(model_name)
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

        let use_responses = matches!(self.resolve_api(), crate::run::DefaultOpenAiApi::Responses);
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

    /// `Authorization: Bearer <key>`, or no header at all for an empty key (servers that need none).
    pub(crate) fn auth_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        if !self.api_key.is_empty() {
            if let Ok(mut value) =
                reqwest::header::HeaderValue::from_str(&format!("Bearer {}", self.api_key))
            {
                value.set_sensitive(true);
                headers.insert(reqwest::header::AUTHORIZATION, value);
            }
        }
        headers
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
    if let Some(query) = &settings.extra_query {
        let pairs = query_pairs(query);
        if !pairs.is_empty() {
            request = request.query(&pairs);
        }
    }
    request
}

/// Flatten `ModelSettings.extra_query` into `key=value` pairs: scalars as text, arrays as
/// repeated keys, `null` skipped, objects as compact JSON.
fn query_pairs(query: &serde_json::Map<String, serde_json::Value>) -> Vec<(String, String)> {
    use serde_json::Value;
    fn text(value: &Value) -> Option<String> {
        match value {
            Value::Null => None,
            Value::String(s) => Some(s.clone()),
            Value::Bool(b) => Some(b.to_string()),
            Value::Number(n) => Some(n.to_string()),
            other => Some(other.to_string()),
        }
    }
    let mut pairs = Vec::new();
    for (key, value) in query {
        match value {
            Value::Array(items) => {
                pairs.extend(items.iter().filter_map(text).map(|v| (key.clone(), v)));
            }
            other => pairs.extend(text(other).map(|v| (key.clone(), v))),
        }
    }
    pairs
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::DefaultOpenAiApi::{ChatCompletions, Responses};

    /// D-I: Chat Completions unless the provider (or `set_default_openai_api`) says otherwise,
    /// whatever the host. (These cases never touch the process-global setting.)
    #[test]
    fn api_defaults_to_chat_completions() {
        for base in [None, Some("https://api.openai.com/v1"), Some("http://localhost:8000/v1")] {
            assert_eq!(OpenAIProvider::new("k", base, None).resolve_api(), ChatCompletions, "{base:?}");
        }
        assert_eq!(
            OpenAIProvider::new("k", None, None).api(Responses).resolve_api(),
            Responses
        );
        assert_eq!(OpenAIProvider::new("k", None, None).always_chat_completions().resolve_api(), ChatCompletions);
    }
}
