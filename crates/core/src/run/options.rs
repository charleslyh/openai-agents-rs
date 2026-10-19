//! The topic groups that make up [`RunConfig`](super::RunConfig).
//!
//! Python's `RunConfig` is one flat dataclass. Here its fields are grouped by what they
//! configure, so a caller reaches for `config.tracing.workflow_name` or
//! `config.model.settings` instead of scanning twenty-odd sibling fields. Every group is
//! `#[non_exhaustive]`: build it with `Default::default()` and the chain methods (or assign
//! the public fields), and a field added later does not break the caller.

use std::future::Future;
use std::sync::Arc;

use serde_json::Value;

use super::config::{
    default_trace_include_sensitive_data, OutputGuardrailBlockedMessage, ToolErrorFormatter,
    ToolErrorFormatterArgs, ToolExecutionConfig, ToolNameCollisionPolicy, ToolNotFoundBehavior,
};
use crate::error::AgentsError;
use crate::handoffs::{HandoffHistoryMapper, HandoffInputFilter};
use crate::items::ReasoningItemIdPolicy;
use crate::memory::{SessionInputCallback, SessionSettings};
use crate::model::{CallModelData, CallModelInputFilter, ModelInputData, ModelProvider, ModelRef};
use crate::model_settings::ModelSettings;
use crate::tracing::TracingConfig;

/// Generates consuming setters that store `Some(value)` or the value itself.
macro_rules! setters {
    ($($(#[$meta:meta])* $name:ident($ty:ty) => $field:ident;)*) => {
        $(
            $(#[$meta])*
            #[must_use]
            pub fn $name(mut self, value: $ty) -> Self {
                self.$field = value;
                self
            }
        )*
    };
    ($($(#[$meta:meta])* $name:ident($ty:ty) => Some($field:ident);)*) => {
        $(
            $(#[$meta])*
            #[must_use]
            pub fn $name(mut self, value: $ty) -> Self {
                self.$field = Some(value);
                self
            }
        )*
    };
}

fn flag<T>(slot: &Option<T>) -> &'static str {
    if slot.is_some() {
        "set"
    } else {
        "unset"
    }
}

/// Which model to run and how to talk to it (Python: `RunConfig.model`, `model_provider`,
/// `model_settings`, `reasoning_item_id_policy`, `call_model_input_filter`).
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct ModelOptions {
    /// Override model for the whole run: an instance, or a name resolved by
    /// [`provider`](Self::provider) (Python: `RunConfig.model`).
    pub model_ref: Option<ModelRef>,
    /// Provider used to resolve string model names (Python: `RunConfig.model_provider`).
    ///
    /// When unset, names go through the default set with
    /// [`crate::set_default_model_provider_factory`].
    pub provider: Option<Arc<dyn ModelProvider>>,
    /// Settings merged over the agent's own (Python: `RunConfig.model_settings`).
    pub settings: Option<ModelSettings>,
    /// Whether reasoning item ids are kept in the input the runner builds
    /// (Python: `RunConfig.reasoning_item_id_policy`; `None` preserves them).
    pub reasoning_item_id_policy: Option<ReasoningItemIdPolicy>,
    /// Edit the model input just before each call (Python: `RunConfig.call_model_input_filter`).
    pub input_filter: Option<CallModelInputFilter>,
}

impl ModelOptions {
    /// Group with every field unset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run this model instead of each agent's own.
    #[must_use]
    pub fn model(mut self, model: impl Into<ModelRef>) -> Self {
        self.model_ref = Some(model.into());
        self
    }

    setters! {
        /// Resolve model names through `provider`.
        provider(Arc<dyn ModelProvider>) => Some(provider);
        /// Merge `settings` over the agent's model settings.
        settings(ModelSettings) => Some(settings);
        /// Keep or strip reasoning item ids.
        reasoning_item_id_policy(ReasoningItemIdPolicy) => Some(reasoning_item_id_policy);
    }

    /// Install a [`CallModelInputFilter`] from an async closure.
    #[must_use]
    pub fn input_filter<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(CallModelData) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ModelInputData, AgentsError>> + Send + 'static,
    {
        self.input_filter = Some(Arc::new(move |data| Box::pin(f(data))));
        self
    }
}

impl std::fmt::Debug for ModelOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelOptions")
            .field("model_ref", &self.model_ref)
            .field("provider", &flag(&self.provider))
            .field("settings", &self.settings)
            .field("reasoning_item_id_policy", &self.reasoning_item_id_policy)
            .field("input_filter", &flag(&self.input_filter))
            .finish()
    }
}

/// How the run is traced (Python: `RunConfig.tracing_disabled`, `workflow_name`, `trace_id`,
/// `group_id`, `trace_metadata`, `trace_include_sensitive_data`, `tracing`).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TracingOptions {
    /// Disable tracing for this run.
    pub disabled: bool,
    /// Workflow name for the root trace.
    pub workflow_name: Option<String>,
    /// Reuse this trace id instead of generating one (Python: `RunConfig.trace_id`).
    pub trace_id: Option<String>,
    /// Grouping identifier for the trace, e.g. a chat thread id (Python: `RunConfig.group_id`).
    pub group_id: Option<String>,
    /// Extra metadata attached to the trace (Python: `RunConfig.trace_metadata`).
    pub metadata: Option<Value>,
    /// Whether tool inputs/outputs and model payloads may be recorded
    /// (Python: `RunConfig.trace_include_sensitive_data`).
    pub include_sensitive_data: bool,
    /// Tracing settings for this run (Python: `RunConfig.tracing`).
    pub config: Option<TracingConfig>,
}

impl Default for TracingOptions {
    fn default() -> Self {
        Self {
            disabled: false,
            workflow_name: None,
            trace_id: None,
            group_id: None,
            metadata: None,
            include_sensitive_data: default_trace_include_sensitive_data(),
            config: None,
        }
    }
}

impl TracingOptions {
    /// Group with the defaults (tracing on, sensitive data per the environment default).
    pub fn new() -> Self {
        Self::default()
    }

    /// Name the root trace.
    #[must_use]
    pub fn workflow_name(mut self, name: impl Into<String>) -> Self {
        self.workflow_name = Some(name.into());
        self
    }

    /// Reuse `id` as the trace id.
    #[must_use]
    pub fn trace_id(mut self, id: impl Into<String>) -> Self {
        self.trace_id = Some(id.into());
        self
    }

    /// Group related traces, e.g. by chat thread.
    #[must_use]
    pub fn group_id(mut self, id: impl Into<String>) -> Self {
        self.group_id = Some(id.into());
        self
    }

    setters! {
        /// Turn tracing off (or back on) for this run.
        disabled(bool) => disabled;
        /// Allow or forbid recording tool and model payloads.
        include_sensitive_data(bool) => include_sensitive_data;
    }

    setters! {
        /// Attach extra metadata to the trace.
        metadata(Value) => Some(metadata);
        /// Tracing settings for this run.
        config(TracingConfig) => Some(config);
    }
}

/// How session history joins the run (Python: `RunConfig.session_input_callback`,
/// `session_settings`).
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct SessionOptions {
    /// Merge session history with the new input (Python: `RunConfig.session_input_callback`).
    pub input_callback: Option<SessionInputCallback>,
    /// Session read settings for this run, overlaid on the session's own
    /// (Python: `RunConfig.session_settings`).
    pub settings: Option<SessionSettings>,
}

impl SessionOptions {
    /// Group with every field unset.
    pub fn new() -> Self {
        Self::default()
    }

    setters! {
        /// Merge session history with the new input through `callback`.
        input_callback(SessionInputCallback) => Some(input_callback);
        /// Overlay `settings` on the session's own read settings.
        settings(SessionSettings) => Some(settings);
    }
}

impl std::fmt::Debug for SessionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionOptions")
            .field("input_callback", &flag(&self.input_callback))
            .field("settings", &self.settings)
            .finish()
    }
}

/// What a handoff passes on (Python: `RunConfig.nest_handoff_history`,
/// `handoff_history_mapper`, `handoff_input_filter`).
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct HandoffOptions {
    /// Compact earlier history into summary messages when handing off
    /// (Python: `RunConfig.nest_handoff_history`, opt-in, default false).
    pub nest_history: bool,
    /// Custom mapper for the nested history (Python: `RunConfig.handoff_history_mapper`);
    /// only used when nesting is on.
    pub history_mapper: Option<HandoffHistoryMapper>,
    /// Default filter for every handoff without its own (Python: `RunConfig.handoff_input_filter`).
    pub input_filter: Option<HandoffInputFilter>,
}

impl HandoffOptions {
    /// Group with every field unset.
    pub fn new() -> Self {
        Self::default()
    }

    setters! {
        /// Compact earlier history into summary messages when handing off.
        nest_history(bool) => nest_history;
    }

    setters! {
        /// Map the nested history with `mapper`.
        history_mapper(HandoffHistoryMapper) => Some(history_mapper);
        /// Filter the input of every handoff that has no filter of its own.
        input_filter(HandoffInputFilter) => Some(input_filter);
    }
}

impl std::fmt::Debug for HandoffOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandoffOptions")
            .field("nest_history", &self.nest_history)
            .field("history_mapper", &flag(&self.history_mapper))
            .field("input_filter", &flag(&self.input_filter))
            .finish()
    }
}

/// How tool calls are executed and reported (Python: `RunConfig.tool_not_found_behavior`,
/// `tool_execution`, `tool_name_collision_policy`, `tool_error_formatter`,
/// `output_guardrail_blocked_message`).
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct ToolOptions {
    /// Behavior when the model calls an unknown tool (Python: `RunConfig.tool_not_found_behavior`).
    pub not_found_behavior: ToolNotFoundBehavior,
    /// Execution settings for local function tools (Python: `RunConfig.tool_execution`).
    pub execution: Option<ToolExecutionConfig>,
    /// Collision handling for tool and handoff names (Python: `tool_name_collision_policy`).
    pub name_collision_policy: ToolNameCollisionPolicy,
    /// Customize approval-rejection and tool-not-found messages
    /// (Python: `RunConfig.tool_error_formatter`).
    pub error_formatter: Option<ToolErrorFormatter>,
    /// Placeholder shown instead of a tool output that an output guardrail rejected
    /// (Python: `RunConfig.output_guardrail_blocked_message`).
    pub guardrail_blocked_message: Option<OutputGuardrailBlockedMessage>,
}

impl ToolOptions {
    /// Group with the defaults.
    pub fn new() -> Self {
        Self::default()
    }

    setters! {
        /// Choose what happens when the model calls an unknown tool.
        not_found_behavior(ToolNotFoundBehavior) => not_found_behavior;
        /// Choose how tool and handoff name collisions are handled.
        name_collision_policy(ToolNameCollisionPolicy) => name_collision_policy;
    }

    setters! {
        /// Limit how local function tools run.
        execution(ToolExecutionConfig) => Some(execution);
        /// Replace the placeholder for a tool output an output guardrail rejected.
        guardrail_blocked_message(OutputGuardrailBlockedMessage) => Some(guardrail_blocked_message);
    }

    /// Install a [`ToolErrorFormatter`] from an async closure.
    #[must_use]
    pub fn error_formatter<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(ToolErrorFormatterArgs) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<String>> + Send + 'static,
    {
        self.error_formatter = Some(Arc::new(move |args| Box::pin(f(args))));
        self
    }
}

impl std::fmt::Debug for ToolOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolOptions")
            .field("not_found_behavior", &self.not_found_behavior)
            .field("execution", &self.execution)
            .field("name_collision_policy", &self.name_collision_policy)
            .field("error_formatter", &flag(&self.error_formatter))
            .field("guardrail_blocked_message", &self.guardrail_blocked_message)
            .finish()
    }
}
