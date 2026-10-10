//! Resolving named agents and the model to call.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::agent::Agent;
use crate::error::{AgentsError, UserError};
use crate::model::{MissingProvider, Model, ModelProvider, ModelRef};

use super::config::RunConfig;

/// Record `root` and every agent it owns through `handoff(..)`, by name. Late-bound handoffs
/// are stand-ins and are skipped; the first agent seen under a name is kept.
pub(crate) fn collect_agents(root: &Agent, into: &mut HashMap<String, Agent>) {
    if into.contains_key(&root.name) {
        return;
    }
    into.insert(root.name.clone(), root.clone());
    for h in root.handoffs.iter().filter(|h| !h.late_bound) {
        collect_agents(&h.agent, into);
    }
}

pub(crate) fn resolve_agent_by_name(root: &Agent, name: &str) -> Result<Agent, AgentsError> {
    if root.name == name {
        return Ok(root.clone());
    }
    let mut stack = vec![root.clone()];
    while let Some(agent) = stack.pop() {
        for h in agent.handoffs.iter().filter(|h| !h.late_bound) {
            if h.agent.name == name {
                return Ok((*h.agent).clone());
            }
            stack.push((*h.agent).clone());
        }
    }
    Err(UserError::new(format!("Agent `{name}` not found in starting agent graph")).into())
}

type ProviderFactory = Box<dyn Fn() -> Arc<dyn ModelProvider> + Send + Sync>;

static DEFAULT_PROVIDER_FACTORY: OnceLock<ProviderFactory> = OnceLock::new();
static DEFAULT_MODEL_PROVIDER: OnceLock<Arc<dyn ModelProvider>> = OnceLock::new();
static MISSING_PROVIDER: OnceLock<Arc<dyn ModelProvider>> = OnceLock::new();

/// Register how to build the process-wide default [`ModelProvider`], used when
/// `RunConfig.model_provider` is not set (Python: the default `MultiProvider` behind `RunConfig`).
///
/// Core knows no concrete provider. A crate that does (the `openai-agents` crate with its
/// `openai` feature) registers a factory here, and core calls it once, the first time a model
/// name has to be resolved without an explicit provider. Registering a factory rather than a
/// provider keeps that moment lazy: configuration such as environment variables is read when it
/// is first needed, not when the factory is registered.
///
/// The default can be registered once per process. Returns `false`, leaving the earlier one in
/// place, if there already is one. Choose a provider for a single run with
/// `RunConfig.model_provider` instead.
pub fn set_default_model_provider_factory(
    factory: impl Fn() -> Arc<dyn ModelProvider> + Send + Sync + 'static,
) -> bool {
    DEFAULT_PROVIDER_FACTORY.set(Box::new(factory)).is_ok()
}

/// The default provider: the one the registered factory builds, or, when none is registered, one
/// that fails with an actionable message.
pub fn default_provider() -> &'static Arc<dyn ModelProvider> {
    if let Some(provider) = DEFAULT_MODEL_PROVIDER.get() {
        return provider;
    }
    match DEFAULT_PROVIDER_FACTORY.get() {
        Some(factory) => DEFAULT_MODEL_PROVIDER.get_or_init(|| factory()),
        None => MISSING_PROVIDER.get_or_init(|| Arc::new(MissingProvider)),
    }
}

/// Resolve the model for a turn.
///
/// Order matches Python: `RunConfig.model` (instance or name) → `Agent.model` →
/// `Agent.model_name`, with names resolved through the configured provider.
pub(crate) fn resolve_model(
    agent: &Agent,
    run_config: &RunConfig,
) -> Result<Arc<dyn Model>, AgentsError> {
    let provider: &Arc<dyn ModelProvider> = match &run_config.model_provider {
        Some(p) => p,
        None => default_provider(),
    };

    if let Some(ModelRef::Instance(m)) = &run_config.model {
        return Ok(Arc::clone(m));
    }
    if let Some(ModelRef::Name(name)) = &run_config.model {
        return Ok(provider.get_model(Some(name))?);
    }
    if let Some(m) = &agent.model {
        return Ok(Arc::clone(m));
    }
    if let Some(name) = &agent.model_name {
        return Ok(provider.get_model(Some(name))?);
    }
    Err(UserError::new(
        "No model configured. Bind a Model (e.g. ScriptedModel) via `Agent::model`, set \
         `Agent::model_name` / `RunConfig.model` to a name resolvable by a ModelProvider, or \
         install a default with `set_default_model_provider_factory` (the `openai-agents` crate does so \
         for OpenAI when its `openai` feature is on and OPENAI_API_KEY is set).",
    )
    .into())
}
