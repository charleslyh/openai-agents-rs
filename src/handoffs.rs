//! Agent handoffs (Python: `agents.handoffs` subset).

use std::sync::Arc;

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::agent::Agent;
use crate::error::AgentsError;
use crate::run_context::RunContextWrapper;

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
}

impl std::fmt::Debug for Handoff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handoff")
            .field("tool_name", &self.tool_name)
            .field("tool_description", &self.tool_description)
            .field("agent_name", &self.agent.name)
            .field("has_input_filter", &self.input_filter.is_some())
            .finish()
    }
}

impl Handoff {
    /// Attach an input filter (Python: `handoff(agent, input_filter=...)`).
    pub fn with_input_filter(mut self, filter: HandoffInputFilter) -> Self {
        self.input_filter = Some(filter);
        self
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
