//! Model providers (Python: `agents.models.interface.ModelProvider`, `MultiProvider`).
//!
//! A provider turns a model *name* into a [`Model`] instance, so agents can be declared with
//! `model_name("gpt-4o")` instead of a pre-built model object.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::UserError;
use crate::model::Model;

/// Resolves model names into model instances (Python: `ModelProvider`).
pub trait ModelProvider: Send + Sync {
    /// Return the model for `model_name`, or the provider default when it is `None`.
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>, UserError>;
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
    /// Built-in prefixes are `openai` (Responses or Chat Completions, depending on
    /// [`crate::set_default_openai_api`]) and `openai_chat_completions`.
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

impl ModelProvider for MultiProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>, UserError> {
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
        provider.get_model(Some(rest))
    }
}

/// A provider that always fails with an actionable message.
///
/// Used when no provider is configured, so callers learn why name resolution is unavailable
/// instead of hitting an opaque "no model" error.
#[derive(Debug, Default)]
pub struct MissingProvider;

impl ModelProvider for MissingProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>, UserError> {
        Err(UserError::new(format!(
            "No ModelProvider can resolve `{}`. Enable the `openai` feature for the default \
             OpenAI provider, or set `RunConfig.model_provider`.",
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

impl<P: ModelProvider> ModelProvider for CachedProvider<P> {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>, UserError> {
        let key = model_name.unwrap_or("<default>").to_string();
        if let Ok(cache) = self.cache.lock() {
            if let Some(model) = cache.get(&key) {
                return Ok(Arc::clone(model));
            }
        }
        let model = self.inner.get_model(model_name)?;
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(key, Arc::clone(&model));
        }
        Ok(model)
    }
}
