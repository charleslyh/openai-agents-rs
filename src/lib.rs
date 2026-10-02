//! OpenAI Agents SDK for Rust — Phase-1 port of `openai-agents` Python v0.23.1.
//!
//! Public surface mirrors `agents/__init__.py` for the supported subset.

#![deny(missing_docs)]

pub mod agent;
pub mod error;
pub mod items;
pub mod model;
pub mod model_settings;
pub mod result;
pub mod run;
pub mod testing;
pub mod tool;
pub mod tracing;
pub mod usage;

pub use agent::{Agent, ToolUseBehavior};
pub use error::{AgentsError, MaxTurnsExceeded, ModelError, UserError};
pub use items::{
    InputLike, ItemHelpers, MessageOutputItem, ModelResponse, ResponseInputItem, ResponseOutputItem,
    RunItem, ToolCallItem, ToolCallOutputItem,
};
pub use model::{Model, ModelRequest, ModelTracing};
pub use model_settings::ModelSettings;
pub use result::RunResult;
pub use run::{
    set_default_openai_api, DefaultOpenAiApi, RunConfig, RunOptions, Runner, DEFAULT_MAX_TURNS,
};
pub use tool::{FunctionTool, ToolContext};
pub use usage::Usage;

#[cfg(feature = "openai")]
pub use model::openai::{OpenAIChatCompletionsModel, OpenAIResponsesModel};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
