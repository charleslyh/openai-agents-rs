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
pub mod context;
pub mod mcp;
pub mod memory;
pub mod model;
pub mod model_settings;
pub(crate) mod pyjson;
pub mod result;
pub mod retry;
pub mod run;
pub mod run_context;
pub mod run_state;
pub mod stream_events;
pub mod strict_schema;
pub mod testing;
pub mod tool;
pub mod tool_guardrails;
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
    AgentsError, InputGuardrailTripwireTriggered, MaxTurnsExceeded, ModelError, ModelRefusalError,
    ModelConnectionError, ModelStatusError, ModelTimeoutError,
    OutputGuardrailTripwireTriggered, ToolInputGuardrailTripwireTriggered,
    ToolOutputGuardrailTripwireTriggered, ToolTimeoutError, UserError,
};
pub use guardrail::{
    input_guardrail, output_guardrail, GuardrailFunctionOutput, InputGuardrail,
    InputGuardrailResult, OutputGuardrail, OutputGuardrailResult,
};
pub use handoffs::{
    default_handoff_history_mapper, get_conversation_history_wrappers, handoff,
    handoff_input_filter, handoff_with, nest_handoff_history, reset_conversation_history_wrappers,
    set_conversation_history_wrappers, Handoff, HandoffHistoryMapper, HandoffInputData,
    HandoffInputFilter, OnHandoff,
};
pub use items::{
    HandoffCallItem, HandoffOutputItem, InputLike, ItemHelpers, MessageOutputItem, ModelResponse,
    ReasoningItem, ResponseInputItem, ResponseOutputItem, RunItem, ToolApprovalItem, ToolCallItem,
    ToolCallOutputItem,
};
pub use lifecycle::{AgentHooks, RunHooks};
#[cfg(feature = "mcp")]
pub use mcp::{StdioParams, StreamableHttpParams};
pub use mcp::{
    mcp_function_tools, render_tool_result, McpCallToolResult, McpClient, McpConfig, McpError,
    McpServer, McpTool, RequireApproval, ToolFilter, ToolFilterContext,
};
pub use memory::{
    CompactingSession, InMemorySession, ModelSummarizer, Session, SessionInputCallback,
    SessionSettings, Summarizer, DEFAULT_TRIGGER_TOKENS,
};
#[cfg(feature = "sqlite")]
pub use memory::SqliteSession;
pub use model::{
    default_model_provider, MissingProvider, Model, ModelProvider, ModelRef, ModelRequest,
    ModelTracing, MultiProvider,
};
pub use model_settings::{ModelSettings, ToolChoice, Truncation, Verbosity};
pub use result::{CancelMode, RunResult, RunResultStreaming, StreamingSnapshot};
pub use run::{
    CallModelData, CallModelInputFilter, ModelInputData, OutputGuardrailBlockedMessage,
    OutputGuardrailBlockedMessageArgs, OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT, ToolErrorFormatter, ToolErrorFormatterArgs,
    ToolErrorKind, ToolExecutionConfig, ToolNameCollisionPolicy, RunErrorData, RunErrorHandler, RunErrorHandlerInput,
    RunErrorHandlerResult, RunErrorHandlers, RunHandledError,
    default_trace_include_sensitive_data, set_default_openai_api, DefaultOpenAiApi, RunConfig,
    RunOptions, Runner, ToolNotFoundBehavior, DEFAULT_MAX_TURNS,
};
pub use run_context::{ContextValue, RunContextWrapper};
pub use tracing::TracingConfig;
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
    FunctionTool, IsEnabledFn, NeedsApproval, ToolContext, ToolEnabled, ToolFailureHandling, ToolResult, ToolTimeoutBehavior,
    DEFAULT_APPROVAL_REJECTION_MESSAGE, DEFAULT_TOOL_ERROR_MESSAGE,
};
pub use tool_guardrails::{
    ToolGuardrailBehavior, ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolInputGuardrailData,
    ToolInputGuardrailResult, ToolOutputGuardrail, ToolOutputGuardrailData,
    ToolOutputGuardrailResult,
};
pub use context::{
    chain_input_filters, default_token_counter, estimate_item_tokens, input_filter,
    ContextWindowTrimmer, TokenCounter, ToolOutputTrimmer,
};
pub use items::ReasoningItemIdPolicy;
pub use retry::{
    retry_policies, ModelRetryAdvice, ModelRetryAdviceRequest, ModelRetryBackoffSettings,
    ModelRetryNormalizedError, ModelRetrySettings, ReplaySafety, RetryDecision, RetryPolicy,
    RetryPolicyContext,
};
pub use usage::{InputTokensDetails, OutputTokensDetails, RequestUsage, Usage};

/// Attribute macro: turn a function into a [`FunctionTool`] constructor (Python: `@function_tool`).
pub use openai_agents_macros::function_tool;

#[cfg(feature = "openai")]
pub use model::openai::{
    CompatibleProvider, OpenAIChatCompletionsModel, OpenAIProvider, OpenAIResponsesModel,
};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
