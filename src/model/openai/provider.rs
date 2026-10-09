//! `OpenAIProvider` and `CompatibleProvider`: name -> model resolution.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::UserError;
use crate::model::provider::ModelProvider;
use crate::model::Model;

use super::{OpenAIChatCompletionsModel, OpenAIResponsesModel};

/// Resolves OpenAI model names into [`Model`] instances (Python: `OpenAIProvider`).
///
/// The API used depends on [`crate::get_default_openai_api`]: Chat Completions by default,
/// Responses when the default was switched or the provider was given `.api(Responses)`.
#[derive(Clone)]
pub struct OpenAIProvider {
    api_key: String,
    base_url: Option<String>,
    default_model: Option<String>,
    api: Option<crate::model::DefaultOpenAiApi>,
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
        let base_url = std::env::var("OPENAI_BASE_URL")
            .ok()
            .filter(|v| !v.trim().is_empty());
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
        self.api(crate::model::DefaultOpenAiApi::ChatCompletions)
    }

    /// Use this API for every model of the provider, whatever the global default says.
    pub fn api(mut self, api: crate::model::DefaultOpenAiApi) -> Self {
        self.api = Some(api);
        self
    }

    /// The API a model gets (D-I): the provider's own choice, else `set_default_openai_api`,
    /// else Chat Completions.
    fn resolve_api(&self) -> crate::model::DefaultOpenAiApi {
        self.api
            .or_else(crate::model::explicit_default_openai_api)
            .unwrap_or_default()
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
                .api(crate::model::DefaultOpenAiApi::ChatCompletions),
        }
    }

    /// Send `Authorization: Bearer <key>`; without a key no such header is sent.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.inner.api_key = key.into();
        self
    }

    /// The protocol to speak (default: Chat Completions).
    pub fn api(mut self, api: crate::model::DefaultOpenAiApi) -> Self {
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

        let use_responses = matches!(
            self.resolve_api(),
            crate::model::DefaultOpenAiApi::Responses
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DefaultOpenAiApi::{ChatCompletions, Responses};

    /// D-I: Chat Completions unless the provider (or `set_default_openai_api`) says otherwise,
    /// whatever the host. (These cases never touch the process-global setting.)
    #[test]
    fn api_defaults_to_chat_completions() {
        for base in [
            None,
            Some("https://api.openai.com/v1"),
            Some("http://localhost:8000/v1"),
        ] {
            assert_eq!(
                OpenAIProvider::new("k", base, None).resolve_api(),
                ChatCompletions,
                "{base:?}"
            );
        }
        assert_eq!(
            OpenAIProvider::new("k", None, None)
                .api(Responses)
                .resolve_api(),
            Responses
        );
        assert_eq!(
            OpenAIProvider::new("k", None, None)
                .always_chat_completions()
                .resolve_api(),
            ChatCompletions
        );
    }
}
