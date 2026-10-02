//! OpenAI Agents SDK for Rust — Phase-1 port of `openai-agents` Python v0.23.1.
//!
//! Public surface mirrors `agents/__init__.py` for the supported subset.

#![deny(missing_docs)]

pub mod agent;
pub mod error;
pub mod handoffs;
pub mod items;
pub mod model;
pub mod model_settings;
pub mod result;
pub mod run;
pub mod run_state;
pub mod stream_events;
pub mod testing;
pub mod tool;
pub mod tracing;
pub mod usage;

pub use agent::{Agent, AsToolConfig, ToolUseBehavior};
pub use error::{AgentsError, MaxTurnsExceeded, ModelError, UserError};
pub use handoffs::{handoff, handoff_with, Handoff};
pub use items::{
    InputLike, ItemHelpers, MessageOutputItem, ModelResponse, ResponseInputItem, ResponseOutputItem,
    RunItem, ToolApprovalItem, ToolCallItem, ToolCallOutputItem,
};
pub use model::{Model, ModelRequest, ModelTracing};
pub use model_settings::ModelSettings;
pub use result::{RunResult, RunResultStreaming, StreamingSnapshot};
pub use run::{
    set_default_openai_api, DefaultOpenAiApi, RunConfig, RunOptions, Runner, DEFAULT_MAX_TURNS,
};
pub use run_state::{
    ApprovalDecision, ApprovalStore, RunState, StickyDecision, RUN_STATE_SCHEMA_VERSION,
};
pub use stream_events::{RunItemStreamName, StreamEvent};
pub use tool::{
    FunctionTool, NeedsApproval, ToolContext, ToolResult, DEFAULT_APPROVAL_REJECTION_MESSAGE,
};
pub use usage::Usage;

/// Attribute macro: turn a function into a [`FunctionTool`] constructor (Python: `@function_tool`).
pub use openai_agents_macros::function_tool;

#[cfg(feature = "openai")]
pub use model::openai::{OpenAIChatCompletionsModel, OpenAIResponsesModel};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
