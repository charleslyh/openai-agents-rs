//! A preset turned into a provider: [`Provider`].
//!
//! [`Provider`] is a builder over one entry of the preset table (or over a URL you supply). It
//! reads the key from the environment when the preset names one, and only builds once there is
//! somewhere to send the request. Keys never appear in `Debug`, errors or traces.

use std::sync::Arc;

use openai_agents_core::error::UserError;
use openai_agents_core::model::provider::ModelProvider;
use openai_agents_openai::{CompatibleProvider, DefaultOpenAiApi};

use super::spec::{preset, Capabilities, ProviderApi, ProviderSpec, LAST_CHECKED};

/// A provider built from a preset or from a URL, ready to be registered under a prefix.
///
/// ```no_run
/// use openai_agents_providers::Provider;
///
/// // DEEPSEEK_API_KEY in the environment, endpoint and protocol from the preset table.
/// let provider = Provider::preset("deepseek")?.build()?;
/// # Ok::<(), openai_agents_core::UserError>(())
/// ```
pub struct Provider {
    name: String,
    spec: Option<&'static ProviderSpec>,
    base_url: Option<String>,
    api_key: Option<String>,
    api: Option<ProviderApi>,
    default_model: Option<String>,
}

impl std::fmt::Debug for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never render the key.
        f.debug_struct("Provider")
            .field("name", &self.name)
            .field("base_url", &self.base_url)
            .field("api", &self.api)
            .field("has_key", &self.api_key.is_some())
            .finish()
    }
}

impl Provider {
    /// Start from a preset of the table; its key variable, endpoint, protocol and capabilities
    /// become the defaults.
    ///
    /// The key is read when [`Self::build`] runs, so a preset can be constructed before the
    /// environment is set.
    pub fn preset(name: &str) -> Result<Self, UserError> {
        let spec = preset(name).ok_or_else(|| {
            UserError::new(format!(
                "unknown provider preset `{name}`; known: {} (table last checked {LAST_CHECKED})",
                super::spec::PRESETS
                    .iter()
                    .map(|p| p.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
        Ok(Self {
            name: name.to_string(),
            spec: Some(spec),
            base_url: Some(spec.base_url.to_string()),
            api_key: None,
            api: Some(spec.api),
            default_model: spec.default_model.map(str::to_string),
        })
    }

    /// A provider for a server of your own: no preset, and no capabilities claimed beyond the
    /// protocol basics.
    pub fn custom(name: impl Into<String>, base_url: impl AsRef<str>) -> Result<Self, UserError> {
        let name = name.into();
        let base_url = validate_endpoint(base_url.as_ref())?.to_string();
        Ok(Self {
            name,
            spec: None,
            base_url: Some(base_url),
            api_key: None,
            api: Some(ProviderApi::ChatCompletions),
            default_model: None,
        })
    }

    /// Override the endpoint (a gateway in front of the provider, a regional URL, ...).
    pub fn base_url(mut self, base_url: impl AsRef<str>) -> Result<Self, UserError> {
        self.base_url = Some(validate_endpoint(base_url.as_ref())?.to_string());
        Ok(self)
    }

    /// Set the key explicitly instead of reading it from the environment.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Speak a different protocol than the preset's default.
    pub fn api(mut self, api: ProviderApi) -> Self {
        self.api = Some(api);
        self
    }

    /// The model to resolve when a run names none.
    pub fn default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// What the provider supports: the preset's claim, or the basics for a custom endpoint.
    pub fn capabilities(&self) -> Capabilities {
        match self.spec {
            Some(spec) => spec.capabilities,
            None => Capabilities::chat(),
        }
    }

    /// The model this provider answers with when a run names none.
    pub fn default_model_name(&self) -> Option<&str> {
        self.default_model.as_deref()
    }

    /// The name this provider is registered under.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Build the provider.
    ///
    /// Fails when the preset names a key variable that is unset, or when no endpoint is known.
    pub fn build(self) -> Result<Arc<dyn ModelProvider>, UserError> {
        let base_url = self
            .base_url
            .ok_or_else(|| UserError::new(format!("provider `{}` has no endpoint", self.name)))?;
        let key = match self.api_key {
            Some(key) => key,
            None => match self.spec.and_then(|s| s.key_env) {
                Some(var) => std::env::var(var).map_err(|_| {
                    UserError::new(format!(
                        "provider `{}` needs `{var}`; set it, or pass a key with \
                         `Provider::api_key(..)`",
                        self.name
                    ))
                })?,
                None => String::new(),
            },
        };
        let mut provider = CompatibleProvider::new(&base_url);
        if !key.trim().is_empty() {
            provider = provider.api_key(key);
        }
        provider = provider.api(match self.api.unwrap_or_default() {
            ProviderApi::ChatCompletions => DefaultOpenAiApi::ChatCompletions,
            ProviderApi::Responses => DefaultOpenAiApi::Responses,
        });
        if let Some(model) = self.default_model {
            provider = provider.default_model(model);
        }
        tracing::debug!(provider = %self.name, %base_url, "built a provider from a preset");
        Ok(Arc::new(provider))
    }
}

/// Reject an endpoint that is not a loopback `http` URL and not `https` (Rust-only, X-013).
///
/// Plain `http` to anything but the local machine sends the API key — and the whole prompt — in
/// clear text over the network, so it needs `AGENTS_PROVIDER_INSECURE_HTTP=1`. Loopback is
/// allowed: that is what a local runtime (Ollama, vLLM, llama.cpp) serves.
pub fn validate_endpoint(url: &str) -> Result<&str, UserError> {
    let rest = url
        .strip_prefix("https://")
        .map(|_| None)
        .or_else(|| url.strip_prefix("http://").map(Some))
        .ok_or_else(|| {
            UserError::new(format!(
                "endpoint `{url}` must start with `https://` (or `http://` for a loopback address)"
            ))
        })?;
    let Some(host) = rest else {
        return Ok(url);
    };
    let host = host.split(['/', '?', '#']).next().unwrap_or(host);
    let host = host.rsplit_once('@').map_or(host, |(_, h)| h);
    // An IPv6 literal carries its own colons: only a port after `]` splits the host off.
    let host = match host.find(']') {
        Some(end) => &host[..=end],
        None => host.split_once(':').map_or(host, |(h, _)| h),
    };
    if is_loopback(host) {
        return Ok(url);
    }
    let allowed = std::env::var("AGENTS_PROVIDER_INSECURE_HTTP")
        .ok()
        .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true"));
    if allowed {
        tracing::warn!(%url, "sending an API key over plain http; AGENTS_PROVIDER_INSECURE_HTTP is set");
        return Ok(url);
    }
    Err(UserError::new(format!(
        "endpoint `{url}` is plain http to a non-loopback host; use https, or set \
         AGENTS_PROVIDER_INSECURE_HTTP=1 if the network is trusted"
    )))
}

fn is_loopback(host: &str) -> bool {
    let host = host.trim_matches(['[', ']']);
    host == "localhost" || host == "::1" || host.starts_with("127.") || host.ends_with(".localhost")
}
