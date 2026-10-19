//! OpenAI Agents SDK for Rust — port of `openai-agents` Python v0.23.1.
//!
//! The public surface mirrors `agents/__init__.py` for the supported subset; see
//! `docs/COMPAT.md` for the support matrix and `docs/DEVIATIONS.md` for recorded differences.

#![deny(missing_docs)]

pub mod agent;
pub mod agent_output;
pub mod context;
pub mod error;
pub mod extensions;
pub mod guardrail;
pub mod handoffs;
pub mod items;
pub mod lifecycle;
pub mod mcp;
pub mod memory;
pub mod model;
pub mod model_middleware;
pub mod model_settings;
pub mod prelude;
pub mod result;
pub mod retry;
pub mod run;
pub mod run_context;
pub mod run_state;
pub mod runtime;
pub mod stream_events;
pub mod strict_schema;
pub(crate) mod support;
pub mod tool;
pub mod tool_guardrails;
pub mod tool_middleware;
pub mod tracing;
pub mod usage;
pub mod utf8_stream;

pub use agent::{
    Agent, FunctionToolResult, ToolUseBehavior, ToolsToFinalOutputFn, ToolsToFinalOutputResult,
};
pub use agent::{Capability, CapabilityParts, MountedCapability, NamedCapability};
pub use agent_output::{
    output_schema, AgentOutputSchema, AgentOutputSchemaBase, CustomOutputSchema,
};
/// Re-exported for implementing the async extension traits ([`Model`], [`ModelProvider`],
/// [`Tool`], [`Session`], ...) without a direct `async-trait` dependency.
pub use async_trait::async_trait;
pub use context::{
    chain_input_filters, default_token_counter, estimate_item_tokens, input_filter,
    ContextWindowTrimmer, TokenCounter, ToolOutputTrimmer,
};
pub use error::{
    AgentsError, InputGuardrailTripwireTriggered, MaxTurnsExceeded, ModelConnectionError,
    ModelError, ModelRefusalError, ModelStatusError, ModelTimeoutError,
    OutputGuardrailTripwireTriggered, ToolInputGuardrailTripwireTriggered,
    ToolOutputGuardrailTripwireTriggered, ToolTimeoutError, UserError,
};
pub use extensions::{Extensions, TypeRequirement};
pub use guardrail::{
    input_guardrail, output_guardrail, GuardrailFunctionOutput, InputGuardrail,
    InputGuardrailResult, OutputGuardrail, OutputGuardrailResult, Severity, Verdict,
};
pub use handoffs::{
    default_handoff_history_mapper, get_conversation_history_wrappers, handoff,
    handoff_input_filter, handoff_to_name, handoff_with, nest_handoff_history,
    reset_conversation_history_wrappers, set_conversation_history_wrappers, Handoff,
    HandoffHistoryMapper, HandoffInputData, HandoffInputFilter, OnHandoff,
};
pub use items::ReasoningItemIdPolicy;
pub use items::{
    HandoffCallItem, HandoffOutputItem, InputLike, ItemHelpers, MessageOutputItem, ModelResponse,
    ReasoningItem, ResponseInputItem, ResponseOutputItem, RunItem, ToolCallItem,
    ToolCallOutputItem, ToolInterruptionItem,
};
pub use lifecycle::{AgentHooks, RunHooks};
pub use mcp::{
    mcp_function_tools, render_tool_result, McpCallToolResult, McpConfig, McpServer, McpTool,
};
pub use memory::{
    CompactingSession, InMemorySession, ModelSummarizer, Session, SessionInputCallback,
    SessionSettings, Summarizer, DEFAULT_TRIGGER_TOKENS,
};
pub use model::{
    CallModelData, CallModelInputFilter, MissingProvider, Model, ModelInputData, ModelProvider,
    ModelRef, ModelRequest, ModelTracing, MultiProvider,
};
pub use model_middleware::ModelMiddleware;
pub use model_settings::{
    ModelSettings, Reasoning, ReasoningEffort, ReasoningSummary, ToolChoice, Truncation, Verbosity,
};
pub use result::{CancelMode, RunResult, RunResultStreaming, StreamingSnapshot};
pub use retry::{
    retry_policies, ModelRetryAdvice, ModelRetryAdviceRequest, ModelRetryBackoffSettings,
    ModelRetryNormalizedError, ModelRetrySettings, ReplaySafety, RetryDecision, RetryPolicy,
    RetryPolicyContext,
};
pub use run::{default_provider, set_default_model_provider_factory};
pub use run::{
    default_trace_include_sensitive_data, AsToolConfig, HandoffOptions, ModelOptions,
    OutputGuardrailBlockedMessage, OutputGuardrailBlockedMessageArgs, RunConfig, RunErrorData,
    RunErrorHandler, RunErrorHandlerInput, RunErrorHandlerResult, RunErrorHandlers,
    RunHandledError, RunOptions, Runner, SessionOptions, ToolErrorFormatter,
    ToolErrorFormatterArgs, ToolErrorKind, ToolExecutionConfig, ToolNameCollisionPolicy,
    ToolNotFoundBehavior, ToolOptions, ToolResolver, TracingOptions, DEFAULT_MAX_TURNS,
    OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT,
};
pub use run_context::{ContextValue, RunContextWrapper};
pub use run_state::{
    ResolutionStore, RunState, StickyDecision, ToolResolution, RUN_STATE_SCHEMA_VERSION,
    SUPPORTED_RUN_STATE_SCHEMAS,
};
pub use runtime::{Runtime, RuntimeBuilder};
/// Re-exported so `#[function_tool]`-generated code does not require a direct dependency.
pub use schemars;
/// Re-exported so `#[function_tool]`-generated code does not require a direct dependency.
pub use serde;
/// Re-exported so `#[function_tool]`-generated code does not require a direct dependency.
pub use serde_json;
pub use stream_events::{RunItemStreamName, StreamEvent};
pub use tool::{
    FunctionTool, IsEnabledFn, NeedsResolution, Tool, ToolContext, ToolEnabled,
    ToolFailureHandling, ToolOutput, ToolPolicy, ToolSpec, ToolTimeoutBehavior,
    DEFAULT_APPROVAL_REJECTION_MESSAGE, DEFAULT_TOOL_ERROR_MESSAGE,
};
pub use tool_guardrails::{
    ToolGuardrailBehavior, ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolInputGuardrailData,
    ToolInputGuardrailResult, ToolOutputGuardrail, ToolOutputGuardrailData,
    ToolOutputGuardrailResult,
};
pub use tool_middleware::{NextTool, ToolMiddleware};
pub use tracing::TracingConfig;
pub use usage::{InputTokensDetails, OutputTokensDetails, RequestUsage, Usage};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
