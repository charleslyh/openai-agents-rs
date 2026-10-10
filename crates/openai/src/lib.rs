//! OpenAI and OpenAI-compatible model adapters for the Rust OpenAI Agents SDK.
//!
//! Chat Completions and Responses models over HTTP (`async-openai` config plus `reqwest`), the
//! providers that turn model names into them, and the conversion between Responses items and
//! Chat Completions messages. The core crate knows none of this: it only defines the `Model`
//! and `ModelProvider` contracts these types implement.

#![deny(missing_docs)]

mod chat_completions;
pub mod chat_convert;
mod default_api;
mod provider;
mod request;
mod responses;
pub mod retry_advice;
mod transport;

pub use chat_completions::OpenAIChatCompletionsModel;
pub use default_api::{get_default_openai_api, set_default_openai_api, DefaultOpenAiApi};
pub use provider::{CompatibleProvider, OpenAIProvider};
pub use responses::OpenAIResponsesModel;

pub(crate) use request::{
    apply_model_settings_chat, apply_model_settings_responses, decorate_request, tools_as_chat,
    tools_as_responses,
};
pub(crate) use transport::{is_json_response, map_reqwest, map_transport, status_error, SseReader};

use async_openai::config::{Config, OpenAIConfig};

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
