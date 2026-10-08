//! Agent definition (Python: `agents.agent.Agent` subset).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::agent_output::AgentOutputSchemaBase;
use crate::guardrail::{InputGuardrail, OutputGuardrail};
use crate::handoffs::Handoff;
use crate::lifecycle::AgentHooks;
use crate::model::Model;
use crate::model_settings::{get_default_model_settings, ModelSettings};
use crate::run_context::RunContextWrapper;
use crate::tool::FunctionTool;

/// System instructions for an agent (Python: `Agent.instructions`).
#[derive(Clone)]
pub enum Instructions {
    /// A fixed system prompt.
    Static(String),
    /// Generated per run from the context and the agent
    /// (Python: a callable `(context, agent) -> str`).
    Dynamic(DynamicInstructions),
}

impl std::fmt::Debug for Instructions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static(s) => f.debug_tuple("Static").field(s).finish(),
            Self::Dynamic(_) => f.write_str("Dynamic(..)"),
        }
    }
}

impl From<&str> for Instructions {
    fn from(value: &str) -> Self {
        Self::Static(value.to_string())
    }
}

impl From<String> for Instructions {
    fn from(value: String) -> Self {
        Self::Static(value)
    }
}

/// Body of a dynamic instruction generator: `(context, agent) -> String`.
pub type DynamicInstructions = Arc<
    dyn Fn(RunContextWrapper, Arc<Agent>) -> Pin<Box<dyn Future<Output = String> + Send>>
        + Send
        + Sync,
>;

/// A completed function tool call handed to a custom [`ToolUseBehavior`]
/// (Python: `FunctionToolResult`, subset).
#[derive(Debug, Clone)]
pub struct FunctionToolResult {
    /// Name of the tool that ran.
    pub tool_name: String,
    /// Call id from the model.
    pub call_id: String,
    /// The tool output.
    pub output: serde_json::Value,
}

/// Decision of a custom tool-use function (Python: `ToolsToFinalOutputResult`).
#[derive(Debug, Clone)]
pub struct ToolsToFinalOutputResult {
    /// Whether the run ends with `final_output`.
    pub is_final_output: bool,
    /// The final output when `is_final_output` is set.
    pub final_output: Option<serde_json::Value>,
}

impl ToolsToFinalOutputResult {
    /// Keep going: feed the tool results back to the model.
    pub fn run_llm_again() -> Self {
        Self { is_final_output: false, final_output: None }
    }

    /// End the run with `output`.
    pub fn final_output(output: serde_json::Value) -> Self {
        Self { is_final_output: true, final_output: Some(output) }
    }
}

/// Custom rule deciding whether tool results are final
/// (Python: `ToolsToFinalOutputFunction`, synchronous form).
pub type ToolsToFinalOutputFn =
    Arc<dyn Fn(&RunContextWrapper, &[FunctionToolResult]) -> ToolsToFinalOutputResult + Send + Sync>;

/// How tool results affect the run loop (Python: `tool_use_behavior`).
#[derive(Clone)]
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
    /// Decide with a custom function over the turn's tool results.
    Custom(ToolsToFinalOutputFn),
}

impl std::fmt::Debug for ToolUseBehavior {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RunLlmAgain => f.write_str("RunLlmAgain"),
            Self::StopOnFirstTool => f.write_str("StopOnFirstTool"),
            Self::StopAtTools { stop_at_tool_names } => f
                .debug_struct("StopAtTools")
                .field("stop_at_tool_names", stop_at_tool_names)
                .finish(),
            Self::Custom(_) => f.write_str("Custom(..)"),
        }
    }
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
    /// System instructions (static or generated per run).
    pub instructions: Option<Instructions>,
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
    /// Structured output schema (Python: `Agent.output_type`).
    ///
    /// When set and not plain text, the model is constrained to the schema and the run's
    /// `final_output` is the validated JSON value.
    pub output_type: Option<Arc<dyn AgentOutputSchemaBase>>,
    /// Checks run against the run input (Python: `Agent.input_guardrails`).
    pub input_guardrails: Vec<InputGuardrail>,
    /// Checks run against the final output (Python: `Agent.output_guardrails`).
    pub output_guardrails: Vec<OutputGuardrail>,
    /// Lifecycle hooks scoped to this agent (Python: `Agent.hooks`).
    pub hooks: Option<Arc<dyn AgentHooks>>,
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
            output_type: None,
            input_guardrails: Vec::new(),
            output_guardrails: Vec::new(),
            hooks: None,
        }
    }

    /// Set instructions.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(Instructions::Static(instructions.into()));
        self
    }

    /// Generate instructions from the run context (Python: a callable `instructions`).
    pub fn dynamic_instructions<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContextWrapper, Arc<Agent>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        let f = Arc::new(f);
        self.instructions = Some(Instructions::Dynamic(Arc::new(move |ctx, agent| {
            let f = Arc::clone(&f);
            Box::pin(async move { f(ctx, agent).await })
        })));
        self
    }

    /// Resolve the system prompt for a run.
    pub async fn resolve_instructions(&self, context: &RunContextWrapper) -> Option<String> {
        match &self.instructions {
            Some(Instructions::Static(s)) => Some(s.clone()),
            Some(Instructions::Dynamic(f)) => {
                Some(f(context.clone(), Arc::new(self.clone())).await)
            }
            None => None,
        }
    }

    /// Set input guardrails.
    pub fn input_guardrails(mut self, guardrails: Vec<InputGuardrail>) -> Self {
        self.input_guardrails = guardrails;
        self
    }

    /// Set output guardrails.
    pub fn output_guardrails(mut self, guardrails: Vec<OutputGuardrail>) -> Self {
        self.output_guardrails = guardrails;
        self
    }

    /// Attach lifecycle hooks for this agent.
    pub fn hooks(mut self, hooks: Arc<dyn AgentHooks>) -> Self {
        self.hooks = Some(hooks);
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

    /// Declare a structured output schema (Python: `Agent.output_type`).
    pub fn output_type(mut self, schema: Arc<dyn AgentOutputSchemaBase>) -> Self {
        self.output_type = Some(schema);
        self
    }

    /// Structured output schema for this agent, when declared.
    pub fn output_schema(&self) -> Option<&dyn AgentOutputSchemaBase> {
        self.output_type.as_deref()
    }

    /// Enabled tools for this turn (evaluates each tool's `is_enabled`).
    pub async fn enabled_tools(&self, context: &RunContextWrapper) -> Vec<FunctionTool> {
        let mut out = Vec::new();
        for tool in &self.tools {
            if tool.is_enabled.resolve(context, self).await {
                out.push(tool.clone());
            }
        }
        out
    }

    /// Enabled handoffs for this turn (evaluates each handoff's `is_enabled`).
    pub async fn enabled_handoffs(&self, context: &RunContextWrapper) -> Vec<Handoff> {
        let mut out = Vec::new();
        for h in &self.handoffs {
            if h.is_enabled.resolve(context, self).await {
                out.push(h.clone());
            }
        }
        out
    }

    /// Expose this agent as a function tool (Python: `Agent.as_tool`).
    ///
    /// Nested tool approvals surface on the outer run's `interruptions`; approve/reject on the
    /// outer [`crate::RunState`], then resume with [`crate::Runner::run_state`].
    pub fn as_tool(&self, config: AsToolConfig) -> FunctionTool {
        crate::run::build_agent_as_tool(self, config)
    }
}
