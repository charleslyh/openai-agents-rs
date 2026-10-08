//! Agent handoffs (Python: `agents.handoffs` subset).

use std::future::Future;
use std::sync::Arc;
use std::pin::Pin;

use serde_json::Value;

use crate::agent::Agent;
use crate::error::AgentsError;
use crate::run_context::RunContextWrapper;
use crate::tool::ToolEnabled;

pub mod history;

pub use history::{
    default_handoff_history_mapper, get_conversation_history_wrappers, nest_handoff_history,
    reset_conversation_history_wrappers, set_conversation_history_wrappers, HandoffHistoryMapper,
};

/// Conversation handed to the next agent, as seen by an input filter
/// (Python: `HandoffInputData`, flattened to Responses input items).
///
/// The next agent receives `input_history`, then `pre_handoff_items`, then `new_items`.
#[derive(Debug, Clone)]
pub struct HandoffInputData {
    /// The input given to `Runner::run`.
    pub input_history: Vec<Value>,
    /// Items produced before the turn in which the handoff happened.
    pub pre_handoff_items: Vec<Value>,
    /// Items of the handoff turn: model output, tool outputs and the handoff output.
    pub new_items: Vec<Value>,
    /// The run context when the handoff was invoked.
    pub run_context: RunContextWrapper,
}

/// Filters the conversation passed to the next agent (Python: `HandoffInputFilter`).
pub type HandoffInputFilter = Arc<
    dyn Fn(HandoffInputData) -> Pin<Box<dyn Future<Output = Result<HandoffInputData, AgentsError>> + Send>>
        + Send
        + Sync,
>;

/// Callback run when a handoff is invoked (Python: `on_handoff`).
///
/// The second argument is the parsed tool-call arguments when the handoff declares an input
/// schema, otherwise `None`.
pub type OnHandoff = Arc<
    dyn Fn(RunContextWrapper, Option<Value>) -> Pin<Box<dyn Future<Output = Result<(), AgentsError>> + Send>>
        + Send
        + Sync,
>;

/// Build a [`HandoffInputFilter`] from an async closure.
pub fn handoff_input_filter<F, Fut>(f: F) -> HandoffInputFilter
where
    F: Fn(HandoffInputData) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<HandoffInputData, AgentsError>> + Send + 'static,
{
    Arc::new(move |data| Box::pin(f(data)))
}

/// A handoff to another agent, exposed to the model as a tool (Python: `Handoff`).
#[derive(Clone)]
pub struct Handoff {
    /// Target agent.
    pub agent: Arc<Agent>,
    /// Tool name shown to the LLM.
    pub tool_name: String,
    /// Tool description shown to the LLM.
    pub tool_description: String,
    /// Filters the history forwarded to the target agent (Python: `Handoff.input_filter`).
    pub input_filter: Option<HandoffInputFilter>,
    /// Callback run when the handoff is invoked (Python: `on_handoff`).
    pub on_handoff: Option<OnHandoff>,
    /// JSON schema of the handoff tool arguments (Python: `input_type` -> `input_json_schema`).
    ///
    /// `None` means the tool takes no arguments.
    pub input_json_schema: Option<Value>,
    /// Whether the handoff is offered to the model (Python: `Handoff.is_enabled`).
    pub is_enabled: ToolEnabled,
    /// Per-handoff override of `RunConfig.nest_handoff_history`
    /// (Python: `Handoff.nest_handoff_history`).
    pub nest_handoff_history: Option<bool>,
}

impl std::fmt::Debug for Handoff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handoff")
            .field("tool_name", &self.tool_name)
            .field("tool_description", &self.tool_description)
            .field("agent_name", &self.agent.name)
            .field("has_input_filter", &self.input_filter.is_some())
            .field("has_on_handoff", &self.on_handoff.is_some())
            .field("input_json_schema", &self.input_json_schema)
            .finish()
    }
}

impl Handoff {
    /// Attach an input filter (Python: `handoff(agent, input_filter=...)`).
    pub fn with_input_filter(mut self, filter: HandoffInputFilter) -> Self {
        self.input_filter = Some(filter);
        self
    }

    /// Override `RunConfig.nest_handoff_history` for this handoff only.
    pub fn with_nest_handoff_history(mut self, nest: bool) -> Self {
        self.nest_handoff_history = Some(nest);
        self
    }

    /// Enable or disable the handoff statically or per turn (Python: `handoff(is_enabled=...)`).
    pub fn with_is_enabled(mut self, enabled: impl Into<ToolEnabled>) -> Self {
        self.is_enabled = enabled.into();
        self
    }

    /// Run `f(context)` when the handoff is invoked (Python: `on_handoff` without `input_type`).
    pub fn with_on_handoff<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContextWrapper) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), AgentsError>> + Send + 'static,
    {
        self.on_handoff = Some(Arc::new(move |ctx, _input| Box::pin(f(ctx))));
        self
    }

    /// Run `f(context, input)` with the model-provided arguments (Python: `on_handoff` with
    /// `input_type`). `schema` is advertised as the handoff tool's parameters, after being made
    /// strict, and the call arguments must parse as JSON.
    pub fn with_on_handoff_input<F, Fut>(mut self, schema: Value, f: F) -> Result<Self, AgentsError>
    where
        F: Fn(RunContextWrapper, Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), AgentsError>> + Send + 'static,
    {
        self.input_json_schema = Some(crate::strict_schema::ensure_strict_json_schema(&schema)?);
        self.on_handoff = Some(Arc::new(move |ctx, input| {
            let input = input.unwrap_or(Value::Null);
            Box::pin(f(ctx, input))
        }));
        Ok(self)
    }

    /// Default tool name (Python: `Handoff.default_tool_name`).
    pub fn default_tool_name(agent_name: &str) -> String {
        transform_string_function_style(&format!("transfer_to_{agent_name}"))
    }

    /// Default tool description.
    pub fn default_tool_description(agent_name: &str, handoff_description: Option<&str>) -> String {
        let extra = handoff_description.unwrap_or("");
        format!("Handoff to the {agent_name} agent to handle the request. {extra}")
    }
}

/// Create a handoff to `agent` (Python: `handoff(agent)`).
pub fn handoff(agent: Agent) -> Handoff {
    let tool_name = Handoff::default_tool_name(&agent.name);
    let tool_description =
        Handoff::default_tool_description(&agent.name, agent.handoff_description.as_deref());
    Handoff {
        agent: Arc::new(agent),
        tool_name,
        tool_description,
        input_filter: None,
        on_handoff: None,
        input_json_schema: None,
        is_enabled: ToolEnabled::Fixed(true),
        nest_handoff_history: None,
    }
}

/// Create a handoff with overrides.
pub fn handoff_with(
    agent: Agent,
    tool_name_override: Option<&str>,
    tool_description_override: Option<&str>,
) -> Handoff {
    let tool_name = tool_name_override
        .map(str::to_string)
        .unwrap_or_else(|| Handoff::default_tool_name(&agent.name));
    let tool_description = tool_description_override
        .map(str::to_string)
        .unwrap_or_else(|| {
            Handoff::default_tool_description(&agent.name, agent.handoff_description.as_deref())
        });
    Handoff {
        agent: Arc::new(agent),
        tool_name,
        tool_description,
        input_filter: None,
        on_handoff: None,
        input_json_schema: None,
        is_enabled: ToolEnabled::Fixed(true),
        nest_handoff_history: None,
    }
}

/// Python `_transforms.transform_string_function_style` (simplified).
pub fn transform_string_function_style(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_underscore = false;
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    out.trim_matches('_').to_string()
}
