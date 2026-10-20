//! Model providers (Python: `agents.models.interface.ModelProvider`, `MultiProvider`).
//!
//! A provider turns a model *name* into a [`Model`] instance, so agents can be declared with
//! `model_name("gpt-4o")` instead of a pre-built model object.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::error::UserError;
use crate::model::Model;

/// Resolves model names into model instances (Python: `ModelProvider`).
///
/// `get_model` is async (Python's is sync, X-008) so a provider can fetch credentials, refresh a
/// token or look a deployment up before it builds the model. The runner calls it once per turn
/// that resolves a name; cache inside the provider (or wrap it in [`CachedProvider`]) when that
/// work is expensive.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Return the model for `model_name`, or the provider default when it is `None`.
    async fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>, UserError>;
}

/// Routes `prefix/model` names to a registered provider (Python: `MultiProvider`).
///
/// Names without a prefix use the `openai` provider by default, mirroring Python.
pub struct MultiProvider {
    default_prefix: String,
    providers: HashMap<String, Arc<dyn ModelProvider>>,
}

impl std::fmt::Debug for MultiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiProvider")
            .field("default_prefix", &self.default_prefix)
            .field("providers", &self.providers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl Default for MultiProvider {
    fn default() -> Self {
        Self {
            default_prefix: "openai".to_string(),
            providers: HashMap::new(),
        }
    }
}

impl MultiProvider {
    /// Create an empty router using the `openai` prefix as the default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a provider under `prefix`.
    ///
    /// A router starts empty; names without a prefix go to the provider registered under the
    /// default prefix (`openai`, as in Python).
    pub fn register(mut self, prefix: impl Into<String>, provider: Arc<dyn ModelProvider>) -> Self {
        self.providers.insert(prefix.into(), provider);
        self
    }

    /// Register a provider on an existing router.
    pub fn add(&mut self, prefix: impl Into<String>, provider: Arc<dyn ModelProvider>) {
        self.providers.insert(prefix.into(), provider);
    }

    /// Split `prefix/model`, falling back to the default prefix.
    pub fn split<'a>(&'a self, model_name: &'a str) -> (&'a str, &'a str) {
        match model_name.split_once('/') {
            Some((prefix, rest)) if !prefix.is_empty() && !rest.is_empty() => (prefix, rest),
            _ => (self.default_prefix.as_str(), model_name),
        }
    }
}

#[async_trait]
impl ModelProvider for MultiProvider {
    async fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>, UserError> {
        let Some(name) = model_name else {
            return Err(UserError::new(
                "MultiProvider requires a model name; set `Agent.model_name` or `RunConfig.model`",
            ));
        };
        let (prefix, rest) = self.split(name);
        let provider = self.providers.get(prefix).ok_or_else(|| {
            UserError::new(format!(
                "Unknown model provider `{prefix}` for `{name}`. Register it with \
                 `MultiProvider::register(\"{prefix}\", ..)`"
            ))
        })?;
        provider.get_model(Some(rest)).await
    }
}

/// A provider that always fails with an actionable message.
///
/// Used when no provider is configured, so callers learn why name resolution is unavailable
/// instead of hitting an opaque "no model" error.
#[derive(Debug, Default)]
pub struct MissingProvider;

#[async_trait]
impl ModelProvider for MissingProvider {
    async fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>, UserError> {
        Err(UserError::new(format!(
            "No ModelProvider can resolve `{}`. Set `RunConfig.model.provider`, or install a \
             default with `Runtime::set_model_provider` / `set_model_provider_factory` (the \
             `opai-agents` crate installs an OpenAI one on `Runtime::global()` when its \
             `openai` feature is on and `OPENAI_API_KEY` or `OPENAI_BASE_URL` is set).",
            model_name.unwrap_or("<default>")
        )))
    }
}

/// Cache wrapper so repeated resolution of the same name returns one instance.
pub struct CachedProvider<P: ModelProvider> {
    inner: Arc<P>,
    cache: Mutex<HashMap<String, Arc<dyn Model>>>,
}

impl<P: ModelProvider> std::fmt::Debug for CachedProvider<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let keys: Vec<String> = self
            .cache
            .lock()
            .map(|c| c.keys().cloned().collect())
            .unwrap_or_default();
        f.debug_struct("CachedProvider")
            .field("cached", &keys)
            .finish()
    }
}

impl<P: ModelProvider> CachedProvider<P> {
    /// Wrap `inner` with a per-name cache.
    pub fn new(inner: Arc<P>) -> Self {
        Self {
            inner,
            cache: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl<P: ModelProvider> ModelProvider for CachedProvider<P> {
    async fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>, UserError> {
        let key = model_name.unwrap_or("<default>").to_string();
        if let Ok(cache) = self.cache.lock() {
            if let Some(model) = cache.get(&key) {
                return Ok(Arc::clone(model));
            }
        }
        // The lock is not held across the await; two concurrent misses may both build a model,
        // and the first one stored wins so every caller ends up sharing one instance.
        let model = self.inner.get_model(model_name).await?;
        match self.cache.lock() {
            Ok(mut cache) => Ok(Arc::clone(cache.entry(key).or_insert(model))),
            Err(_) => Ok(model),
        }
    }
}
