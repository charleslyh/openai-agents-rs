//! Resolving named agents and the model to call.

use std::collections::HashMap;
use std::sync::Arc;

use crate::agent::Agent;
use crate::error::{AgentsError, UserError};
use crate::model::{Model, ModelProvider, ModelRef};
use crate::model_settings::{ModelSettings, Reasoning, ReasoningEffort};
use crate::runtime::Runtime;

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

/// Register how to build [`Runtime::global()`]'s default [`ModelProvider`], used when
/// `RunConfig.model.provider` is not set (Python: the default `MultiProvider` behind `RunConfig`).
///
/// Core knows no concrete provider. A crate that does (the `openai-agents` crate with its
/// `openai` feature) registers a factory here, and core calls it once, the first time a model
/// name has to be resolved without an explicit provider. Registering a factory rather than a
/// provider keeps that moment lazy: configuration such as environment variables is read when it
/// is first needed, not when the factory is registered.
///
/// Thin wrapper over [`Runtime::global().set_model_provider_factory`](Runtime::set_model_provider_factory);
/// prefer building an explicit [`Runtime`] in new code. The default can be registered once per
/// process. Returns `false`, leaving the earlier one in place, if there already is one. Choose a
/// provider for a single run with `RunConfig.model.provider` instead.
pub fn set_default_model_provider_factory(
    factory: impl Fn() -> Arc<dyn ModelProvider> + Send + Sync + 'static,
) -> bool {
    Runtime::global().set_model_provider_factory(factory)
}

/// The default provider: [`Runtime::global()`]'s provider.
pub fn default_provider() -> Arc<dyn ModelProvider> {
    Runtime::global().provider()
}

/// The model for a turn and the reasoning effort its name asked for, if any.
pub(crate) struct ResolvedModel {
    pub(crate) model: Arc<dyn Model>,
    /// From a `:<effort>` model-name suffix (see [`ReasoningEffort::split_model_name`]).
    pub(crate) name_effort: Option<ReasoningEffort>,
}

impl ResolvedModel {
    fn named(provider: &dyn ModelProvider, name: &str) -> Result<Self, AgentsError> {
        let (base, name_effort) = ReasoningEffort::split_model_name(name);
        Ok(Self {
            model: provider.get_model(Some(base))?,
            name_effort,
        })
    }

    fn instance(model: &Arc<dyn Model>) -> Self {
        Self {
            model: Arc::clone(model),
            name_effort: None,
        }
    }

    /// Fill in the suffix effort unless the settings already choose one.
    pub(crate) fn apply_to(&self, settings: &mut ModelSettings) {
        let Some(effort) = &self.name_effort else {
            return;
        };
        let reasoning = settings.reasoning.get_or_insert_with(Reasoning::default);
        if reasoning.effort.is_none() {
            reasoning.effort = Some(effort.clone());
        }
    }
}

/// Resolve the model for a turn.
///
/// Order matches Python: `RunConfig.model` (instance or name) → `Agent.model` →
/// `Agent.model_name`, with names resolved through the configured provider. A name ending in a
/// known `:<effort>` is resolved without the suffix (Rust-only, X-006).
pub(crate) fn resolve_model(
    agent: &Agent,
    run_config: &RunConfig,
) -> Result<ResolvedModel, AgentsError> {
    let provider: Arc<dyn ModelProvider> = match &run_config.model.provider {
        Some(p) => Arc::clone(p),
        None => default_provider(),
    };

    if let Some(ModelRef::Instance(m)) = &run_config.model.model_ref {
        return Ok(ResolvedModel::instance(m));
    }
    if let Some(ModelRef::Name(name)) = &run_config.model.model_ref {
        return ResolvedModel::named(&*provider, name);
    }
    if let Some(m) = &agent.model {
        return Ok(ResolvedModel::instance(m));
    }
    if let Some(name) = &agent.model_name {
        return ResolvedModel::named(&*provider, name);
    }
    Err(UserError::new(
        "No model configured. Bind a Model (e.g. ScriptedModel) via `Agent::model`, set \
         `Agent::model_name` / `RunConfig.model` to a name resolvable by a ModelProvider, or \
         install a default with `set_default_model_provider_factory` (the `openai-agents` crate does so \
         for OpenAI when its `openai` feature is on and OPENAI_API_KEY is set).",
    )
    .into())
}
