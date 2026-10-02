//! Agent definition (Python: `agents.agent.Agent` subset).

use std::sync::Arc;

use crate::handoffs::Handoff;
use crate::model::Model;
use crate::model_settings::{get_default_model_settings, ModelSettings};
use crate::tool::FunctionTool;

/// How tool results affect the run loop (Python: `tool_use_behavior`).
#[derive(Debug, Clone)]
pub enum ToolUseBehavior {
    /// Feed tool results back to the LLM (default).
    RunLlmAgain,
    /// Stop after the first tool and use its output as final_output.
    StopOnFirstTool,
    /// Stop when any of the named tools is called.
    StopAtTools {
        /// Tool names that finalize the run.
        stop_at_tool_names: Vec<String>,
    },
}

impl Default for ToolUseBehavior {
    fn default() -> Self {
        Self::RunLlmAgain
    }
}

/// Options for [`Agent::as_tool`] (Python: `Agent.as_tool` kwargs subset).
#[derive(Debug, Clone, Default)]
pub struct AsToolConfig {
    /// Override tool name (default: sanitized agent name).
    pub name: Option<String>,
    /// Tool description (default: handoff_description or agent name).
    pub description: Option<String>,
    /// Whether the agent-tool itself needs approval before the nested run starts.
    pub needs_approval: bool,
    /// Max turns for the nested run.
    pub max_turns: Option<usize>,
}

/// An agent configuration (Python: `Agent`).
#[derive(Clone)]
pub struct Agent {
    /// Display / identity name.
    pub name: String,
    /// System instructions.
    pub instructions: Option<String>,
    /// Description used when this agent is a handoff target.
    pub handoff_description: Option<String>,
    /// Function tools.
    pub tools: Vec<FunctionTool>,
    /// Handoffs to other agents.
    pub handoffs: Vec<Handoff>,
    /// Bound model instance (preferred for tests / custom providers).
    pub model: Option<Arc<dyn Model>>,
    /// Model name resolved via a provider.
    pub model_name: Option<String>,
    /// Model settings.
    pub model_settings: ModelSettings,
    /// Tool use behavior.
    pub tool_use_behavior: ToolUseBehavior,
    /// Reset tool_choice after a tool turn (Python default: true).
    pub reset_tool_choice: bool,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("name", &self.name)
            .field("instructions", &self.instructions)
            .field("handoff_description", &self.handoff_description)
            .field("tools", &self.tools)
            .field("handoffs", &self.handoffs)
            .field("model_name", &self.model_name)
            .field("model_settings", &self.model_settings)
            .field("tool_use_behavior", &self.tool_use_behavior)
            .field("reset_tool_choice", &self.reset_tool_choice)
            .field("model_bound", &self.model.is_some())
            .finish()
    }
}

impl Agent {
    /// Create an agent with a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            instructions: None,
            handoff_description: None,
            tools: Vec::new(),
            handoffs: Vec::new(),
            model: None,
            model_name: None,
            model_settings: get_default_model_settings(),
            tool_use_behavior: ToolUseBehavior::default(),
            reset_tool_choice: true,
        }
    }

    /// Set instructions.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Set handoff description.
    pub fn handoff_description(mut self, description: impl Into<String>) -> Self {
        self.handoff_description = Some(description.into());
        self
    }

    /// Set tools.
    pub fn tools(mut self, tools: Vec<FunctionTool>) -> Self {
        self.tools = tools;
        self
    }

    /// Set handoffs.
    pub fn handoffs(mut self, handoffs: Vec<Handoff>) -> Self {
        self.handoffs = handoffs;
        self
    }

    /// Bind a model instance.
    pub fn model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Set a model name (resolved by OpenAI provider when no instance is bound).
    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
        self
    }

    /// Set model settings.
    pub fn model_settings(mut self, settings: ModelSettings) -> Self {
        self.model_settings = settings;
        self
    }

    /// Set tool use behavior.
    pub fn tool_use_behavior(mut self, behavior: ToolUseBehavior) -> Self {
        self.tool_use_behavior = behavior;
        self
    }

    /// Enabled tools for this run (filters `is_enabled`).
    pub fn enabled_tools(&self) -> Vec<FunctionTool> {
        self.tools.iter().filter(|t| t.is_enabled).cloned().collect()
    }

    /// Expose this agent as a function tool (Python: `Agent.as_tool`).
    ///
    /// Nested tool approvals surface on the outer run's `interruptions`; approve/reject on the
    /// outer [`crate::RunState`], then resume with [`crate::Runner::run_state`].
    pub fn as_tool(&self, config: AsToolConfig) -> FunctionTool {
        crate::run::build_agent_as_tool(self, config)
    }
}
