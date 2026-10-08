//! OpenAI Agents SDK for Rust — port of `openai-agents` Python v0.23.1.
//!
//! The public surface mirrors `agents/__init__.py` for the supported subset; see
//! `docs/COMPAT.md` for the support matrix and `docs/DEVIATIONS.md` for recorded differences.

#![deny(missing_docs)]

pub mod agent;
pub mod agent_output;
pub mod error;
pub mod guardrail;
pub mod handoffs;
pub mod items;
pub mod lifecycle;
pub mod model;
pub mod model_settings;
pub mod result;
pub mod run;
pub mod run_context;
pub mod run_state;
pub mod stream_events;
pub mod strict_schema;
pub mod testing;
pub mod tool;
pub mod tracing;
pub mod usage;

pub use agent::{
    Agent, AsToolConfig, FunctionToolResult, ToolUseBehavior, ToolsToFinalOutputFn,
    ToolsToFinalOutputResult,
};
pub use agent_output::{
    output_schema, AgentOutputSchema, AgentOutputSchemaBase, CustomOutputSchema,
};
pub use error::{
    AgentsError, InputGuardrailTripwireTriggered, MaxTurnsExceeded, ModelError,
    OutputGuardrailTripwireTriggered, UserError,
};
pub use guardrail::{
    input_guardrail, output_guardrail, GuardrailFunctionOutput, InputGuardrail,
    InputGuardrailResult, OutputGuardrail, OutputGuardrailResult,
};
pub use handoffs::{handoff, handoff_with, Handoff};
pub use items::{
    HandoffCallItem, HandoffOutputItem, InputLike, ItemHelpers, MessageOutputItem, ModelResponse,
    ReasoningItem, ResponseInputItem, ResponseOutputItem, RunItem, ToolApprovalItem, ToolCallItem,
    ToolCallOutputItem,
};
pub use lifecycle::{AgentHooks, RunHooks};
pub use model::{
    default_model_provider, MissingProvider, Model, ModelProvider, ModelRef, ModelRequest,
    ModelTracing, MultiProvider,
};
pub use model_settings::{ModelSettings, ToolChoice, Truncation, Verbosity};
pub use result::{RunResult, RunResultStreaming, StreamingSnapshot};
pub use run::{
    default_trace_include_sensitive_data, set_default_openai_api, DefaultOpenAiApi, RunConfig,
    RunOptions, Runner, ToolNotFoundBehavior, DEFAULT_MAX_TURNS,
};
pub use run_context::{ContextValue, RunContextWrapper};
pub use run_state::{
    ApprovalDecision, ApprovalStore, RunState, StickyDecision, RUN_STATE_SCHEMA_VERSION,
    SUPPORTED_RUN_STATE_SCHEMAS,
};
/// Re-exported so `#[function_tool]`-generated code does not require a direct dependency.
pub use schemars;
/// Re-exported so `#[function_tool]`-generated code does not require a direct dependency.
pub use serde;
pub use stream_events::{RunItemStreamName, StreamEvent};
pub use tool::{
    FunctionTool, NeedsApproval, ToolContext, ToolFailureHandling, ToolResult,
    DEFAULT_APPROVAL_REJECTION_MESSAGE, DEFAULT_TOOL_ERROR_MESSAGE,
};
pub use usage::Usage;

/// Attribute macro: turn a function into a [`FunctionTool`] constructor (Python: `@function_tool`).
pub use openai_agents_macros::function_tool;

#[cfg(feature = "openai")]
pub use model::openai::{OpenAIChatCompletionsModel, OpenAIProvider, OpenAIResponsesModel};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
