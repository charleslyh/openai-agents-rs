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
/// are stand-ins and are skipped; the first agent seen under a name is kept. Only `Arc`s are
/// cloned, never an agent.
pub(crate) fn collect_agents(root: &Arc<Agent>, into: &mut HashMap<String, Arc<Agent>>) {
    if into.contains_key(&root.name) {
        return;
    }
    into.insert(root.name.clone(), Arc::clone(root));
    for h in root.handoffs.iter().filter(|h| !h.late_bound) {
        collect_agents(&h.agent, into);
    }
}

/// A stable fingerprint of the agent graph reachable from `root` (Rust-only, X-011).
///
/// Covers each agent's name, its own and capability tools (by name) and its handoffs (by tool
/// name), sorted, so the order agents and tools were added in does not matter. Dynamic parts
/// (MCP tools, `is_enabled`) are left out. FNV-1a 64 is enough: this detects resuming a state
/// with the wrong agents, it is not an integrity check.
pub(crate) fn agent_graph_fingerprint(root: &Arc<Agent>) -> String {
    let mut agents = HashMap::new();
    collect_agents(root, &mut agents);
    let mut names: Vec<&String> = agents.keys().collect();
    names.sort();

    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |field: &str| {
        // Length-prefixed, so ("ab", "c") and ("a", "bc") differ.
        for byte in (field.len() as u64)
            .to_le_bytes()
            .iter()
            .chain(field.as_bytes())
        {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for name in names {
        let agent = &agents[name];
        feed(name);
        let mut tools: Vec<&str> = agent.local_tools().map(|t| t.name.as_str()).collect();
        tools.sort_unstable();
        feed(&tools.len().to_string());
        tools.into_iter().for_each(&mut feed);
        let mut handoffs: Vec<&str> = agent
            .handoffs
            .iter()
            .map(|h| h.tool_name.as_str())
            .collect();
        handoffs.sort_unstable();
        feed(&handoffs.len().to_string());
        handoffs.into_iter().for_each(&mut feed);
    }
    format!("fnv1a64:{hash:016x}")
}

pub(crate) fn resolve_agent_by_name(
    root: &Arc<Agent>,
    name: &str,
) -> Result<Arc<Agent>, AgentsError> {
    if root.name == name {
        return Ok(Arc::clone(root));
    }
    let mut stack = vec![Arc::clone(root)];
    while let Some(agent) = stack.pop() {
        for h in agent.handoffs.iter().filter(|h| !h.late_bound) {
            if h.agent.name == name {
                return Ok(Arc::clone(&h.agent));
            }
            stack.push(Arc::clone(&h.agent));
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

/// The default provider: that of the runtime in scope (a run's own, else [`Runtime::global()`]).
pub fn default_provider() -> Arc<dyn ModelProvider> {
    Runtime::current().provider()
}

/// The model for a turn and the reasoning effort its name asked for, if any.
pub(crate) struct ResolvedModel {
    pub(crate) model: Arc<dyn Model>,
    /// From a `:<effort>` model-name suffix (see [`ReasoningEffort::split_model_name`]).
    pub(crate) name_effort: Option<ReasoningEffort>,
}

impl ResolvedModel {
    async fn named(provider: &dyn ModelProvider, name: &str) -> Result<Self, AgentsError> {
        let (base, name_effort) = ReasoningEffort::split_model_name(name);
        Ok(Self {
            model: provider.get_model(Some(base)).await?,
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
pub(crate) async fn resolve_model(
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
        return ResolvedModel::named(&*provider, name).await;
    }
    if let Some(m) = &agent.model {
        return Ok(ResolvedModel::instance(m));
    }
    if let Some(name) = &agent.model_name {
        return ResolvedModel::named(&*provider, name).await;
    }
    Err(UserError::new(
        "No model configured. Bind a Model (e.g. ScriptedModel) via `Agent::model`, set \
         `Agent::model_name` / `RunConfig.model` to a name resolvable by a ModelProvider, or \
         install a default with `set_default_model_provider_factory` (the `openai-agents` crate does so \
         for OpenAI when its `openai` feature is on and OPENAI_API_KEY is set).",
    )
    .into())
}
