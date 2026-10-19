//! Discovering providers from the environment and routing names to them.
//!
//! [`ProviderRegistry::from_env`] registers every preset whose key variable is set, plus the
//! endpoints declared with `AGENTS_PROVIDER_{N}_ENDPOINT`. [`ProviderRouter`] then resolves
//! `prefix/model` names, and falls back to inference for names that carry no prefix but clearly
//! belong to one provider (`claude-sonnet-x`, `deepseek-chat`).

use std::sync::Arc;

use async_trait::async_trait;
use openai_agents_core::error::UserError;
use openai_agents_core::model::provider::ModelProvider;
use openai_agents_core::model::Model;

use super::catalog::advise;
use super::spec::{infer_preset, Capabilities, ProviderApi, PRESETS};
use super::{OfflineProvider, Provider, LAST_CHECKED};

/// How many `AGENTS_PROVIDER_{N}_...` endpoints are scanned.
const CUSTOM_SLOTS: usize = 9;

/// One registered provider: its prefix, what it supports, and the provider itself.
#[derive(Clone)]
pub struct Entry {
    /// The prefix (and the name in [`Provider`] terms).
    pub name: String,
    /// What the provider supports.
    pub capabilities: Capabilities,
    /// The provider that resolves names under this prefix.
    pub provider: Arc<dyn ModelProvider>,
    /// The model this provider answers with when a run names none.
    pub default_model: Option<String>,
}

impl std::fmt::Debug for Entry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Entry")
            .field("name", &self.name)
            .field(
                "capabilities",
                &self.capabilities.supported().collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// The providers available to a process, keyed by prefix.
///
/// ```no_run
/// use openai_agents_providers::ProviderRegistry;
///
/// let registry = ProviderRegistry::from_env()?;
/// println!("{}", registry.names().collect::<Vec<_>>().join(", "));
/// let provider = registry.into_provider()?; // Arc<dyn ModelProvider>
/// # Ok::<(), openai_agents_core::UserError>(())
/// ```
#[derive(Debug, Default)]
pub struct ProviderRegistry {
    entries: Vec<Entry>,
    default: Option<String>,
}

impl ProviderRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            default: None,
        }
    }

    /// Register every preset whose key variable is set, then the endpoints declared with
    /// `AGENTS_PROVIDER_{N}_ENDPOINT` / `_API_KEY` / `_NAME` / `_DEFAULT_MODEL`.
    ///
    /// A preset that needs no key (Ollama) is registered only when its `enable_env` is set
    /// (`OLLAMA_HOST`): a run with nothing configured must fail with advice, not guess at a
    /// localhost port. A custom endpoint that is not `https` and not loopback fails unless
    /// `AGENTS_PROVIDER_INSECURE_HTTP=1`.
    pub fn from_env() -> Result<Self, UserError> {
        // Offline wins over everything: a run that must not reach the network must not reach it,
        // even when a key happens to be set.
        if env_flag("AGENTS_OFFLINE") {
            let provider: Arc<dyn ModelProvider> = Arc::new(OfflineProvider);
            let mut registry = Self::new();
            // Everything: the offline model answers locally and sends nothing, so no request
            // field can be beyond it.
            registry.push("offline", Capabilities::full(), provider, None);
            return registry.with_default("offline");
        }
        let mut registry = Self::new();
        for spec in PRESETS {
            // A local server needs no key, so it is registered only when you say where it is.
            if let Some(var) = spec.enable_env {
                if env_or_none(var).is_none() {
                    continue;
                }
            }
            let Ok(built) = Provider::preset(spec.name).and_then(|p| p.build()) else {
                continue; // no key: not available, and not an error
            };
            let capabilities = spec.capabilities;
            registry.push(
                spec.name,
                capabilities,
                built,
                spec.default_model.map(str::to_string),
            );
            // OpenAI also answers the protocol-picking prefixes the SDK has always had.
            if spec.name == "openai" {
                for (prefix, api) in [
                    ("openai_chat_completions", ProviderApi::ChatCompletions),
                    ("openai_responses", ProviderApi::Responses),
                ] {
                    if let Ok(built) = Provider::preset("openai")
                        .map(|p| p.api(api))
                        .and_then(|p| p.build())
                    {
                        registry.push(
                            prefix,
                            capabilities,
                            built,
                            spec.default_model.map(str::to_string),
                        );
                    }
                }
            }
        }
        for slot in 1..=CUSTOM_SLOTS {
            let Some(endpoint) = env_or_none(&format!("AGENTS_PROVIDER_{slot}_ENDPOINT")) else {
                continue;
            };
            let name = env_or_none(&format!("AGENTS_PROVIDER_{slot}_NAME"))
                .unwrap_or_else(|| format!("provider_{slot}"));
            let mut provider = Provider::custom(&name, &endpoint)?;
            if let Some(key) = env_or_none(&format!("AGENTS_PROVIDER_{slot}_API_KEY")) {
                provider = provider.api_key(key);
            }
            if let Some(model) = env_or_none(&format!("AGENTS_PROVIDER_{slot}_DEFAULT_MODEL")) {
                provider = provider.default_model(model);
            }
            let capabilities = provider.capabilities();
            let default_model = provider.default_model_name().map(str::to_string);
            let built = provider.build()?;
            registry.push(&name, capabilities, built, default_model);
        }
        Ok(registry)
    }

    /// Register a provider you built yourself under `name`.
    pub fn register(
        mut self,
        name: impl Into<String>,
        provider: Arc<dyn ModelProvider>,
        capabilities: Capabilities,
    ) -> Self {
        self.push(&name.into(), capabilities, provider, None);
        self
    }

    fn push(
        &mut self,
        name: &str,
        capabilities: Capabilities,
        provider: Arc<dyn ModelProvider>,
        default_model: Option<String>,
    ) {
        self.entries.retain(|e| e.name != name);
        self.entries.push(Entry {
            name: name.to_string(),
            capabilities,
            provider,
            default_model,
        });
    }

    /// The model the default provider answers with, when it has an obvious one.
    pub fn default_model(&self) -> Option<&str> {
        let name = self.default.as_deref()?;
        self.get(name)?.default_model.as_deref()
    }

    /// The prefix used when a name carries none.
    pub fn default(&self) -> Option<&str> {
        self.default.as_deref()
    }

    /// Choose the prefix used when a name carries none; it must be registered.
    pub fn with_default(mut self, name: impl Into<String>) -> Result<Self, UserError> {
        let name = name.into();
        if !self.entries.iter().any(|e| e.name == name) {
            return Err(UserError::new(format!(
                "cannot default to `{name}`: it is not registered (registered: {})",
                self.names().collect::<Vec<_>>().join(", ")
            )));
        }
        self.default = Some(name);
        Ok(self)
    }

    /// The registered prefixes, in registration order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.name.as_str())
    }

    /// Whether nothing was registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entry registered under `name`.
    pub fn get(&self, name: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// The entries, in registration order.
    pub fn entries(&self) -> impl Iterator<Item = &Entry> {
        self.entries.iter()
    }

    /// Which registered provider owns `model_name`, by its model prefixes.
    ///
    /// Only brands that name their own models are inferred (see [`infer_preset`]); `llama3` or
    /// `qwen2.5` are served by too many places to guess.
    pub fn infer(&self, model_name: &str) -> Option<&Entry> {
        let preset = infer_preset(model_name)?;
        self.entries.iter().find(|e| e.name == preset.name)
    }

    /// Build the router that resolves names through this registry.
    ///
    /// Without a default the router picks one: `openai` when it is registered, otherwise the
    /// first registered provider. An empty registry fails with a message that says what to set.
    pub fn into_provider(mut self) -> Result<Arc<dyn ModelProvider>, UserError> {
        if self.entries.is_empty() {
            return Err(UserError::new(format!(
                "no model provider is configured. Set one of {}, or declare an endpoint with \
                 AGENTS_PROVIDER_1_ENDPOINT (presets last checked {LAST_CHECKED})",
                PRESETS
                    .iter()
                    .filter_map(|p| p.key_env)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        if self.default.is_none() {
            let fallback = self
                .entries
                .iter()
                .find(|e| e.name == "openai")
                .or_else(|| self.entries.first())
                .map(|e| e.name.clone());
            self.default = fallback;
        }
        Ok(Arc::new(ProviderRouter {
            entries: self.entries,
            default: self.default.unwrap_or_default(),
        }))
    }
}

/// Resolves `prefix/model` (and inferred unprefixed names) through a registry.
///
/// Implements [`ModelProvider`], so it drops in anywhere a provider does.
pub struct ProviderRouter {
    entries: Vec<Entry>,
    default: String,
}

impl std::fmt::Debug for ProviderRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderRouter")
            .field(
                "prefixes",
                &self.entries.iter().map(|e| &e.name).collect::<Vec<_>>(),
            )
            .field("default", &self.default)
            .finish()
    }
}

#[async_trait]
impl ModelProvider for ProviderRouter {
    async fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>, UserError> {
        let Some(name) = model_name else {
            let entry = self
                .entries
                .iter()
                .find(|e| e.name == self.default)
                .ok_or_else(|| self.unknown("<default>"))?;
            return entry.provider.get_model(None).await;
        };
        match name.split_once('/') {
            Some((prefix, rest)) if !prefix.is_empty() && !rest.is_empty() => {
                let entry = self
                    .entries
                    .iter()
                    .find(|e| e.name == prefix)
                    .ok_or_else(|| self.unknown(prefix))?;
                note_model(rest);
                entry.provider.get_model(Some(rest)).await
            }
            _ => {
                let inferred = infer_preset(name)
                    .and_then(|preset| self.entries.iter().find(|e| e.name == preset.name));
                let entry = inferred
                    .or_else(|| self.entries.iter().find(|e| e.name == self.default))
                    .ok_or_else(|| self.unknown(&self.default))?;
                note_model(name);
                entry.provider.get_model(Some(name)).await
            }
        }
    }
}

impl ProviderRouter {
    fn unknown(&self, prefix: &str) -> UserError {
        UserError::new(format!(
            "no model provider is registered for `{prefix}`; registered: {}",
            self.entries
                .iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }
}

/// Say something about a model id the catalogue recognises: retired, superseded, or a near miss.
/// Never fails a run — the catalogue is a hint.
fn note_model(model_name: &str) {
    if let Some(advice) = advise(model_name) {
        tracing::warn!(model = model_name, "{advice}");
    }
}

/// Whether `var` is set to something truthy.
fn env_flag(var: &str) -> bool {
    env_or_none(var)
        .is_some_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}

fn env_or_none(var: &str) -> Option<String> {
    std::env::var(var)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}
