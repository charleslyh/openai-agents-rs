//! Agent handoffs (Python: `agents.handoffs` subset).

use std::sync::Arc;

use crate::agent::Agent;

/// A handoff to another agent, exposed to the model as a tool (Python: `Handoff`).
#[derive(Clone)]
pub struct Handoff {
    /// Target agent.
    pub agent: Arc<Agent>,
    /// Tool name shown to the LLM.
    pub tool_name: String,
    /// Tool description shown to the LLM.
    pub tool_description: String,
}

impl std::fmt::Debug for Handoff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handoff")
            .field("tool_name", &self.tool_name)
            .field("tool_description", &self.tool_description)
            .field("agent_name", &self.agent.name)
            .finish()
    }
}

impl Handoff {
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
