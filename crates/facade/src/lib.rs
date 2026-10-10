//! OpenAI Agents SDK for Rust — port of `openai-agents` Python v0.23.1.
//!
//! This crate is the single entry point: it re-exports everything from `openai-agents-core` under
//! the paths the SDK has always had, and adds the `#[function_tool]` macro. The Cargo features
//! (`openai`, `mcp`, `sqlite`) pull in the crate that implements them. Core knows no concrete model
//! provider; with the `openai` feature, [`Runner`] installs the OpenAI one as the process default.
//!
//! The public surface mirrors `agents/__init__.py` for the supported subset; see
//! `docs/COMPAT.md` for the support matrix and `docs/DEVIATIONS.md` for recorded differences.

#![deny(missing_docs)]

mod runner;

pub use openai_agents_core::*;

/// The core runner plus the default OpenAI model provider; shadows `openai_agents_core::Runner`.
pub use runner::Runner;

/// Deterministic test doubles (Python: `agents.testing`).
pub use openai_agents_testing as testing;

/// OpenAI and OpenAI-compatible model adapters (Chat Completions and Responses).
#[cfg(feature = "openai")]
pub use openai_agents_openai as openai;

#[cfg(feature = "openai")]
pub use openai_agents_openai::{
    get_default_openai_api, set_default_openai_api, CompatibleProvider, DefaultOpenAiApi,
    OpenAIChatCompletionsModel, OpenAIProvider, OpenAIResponsesModel,
};

#[cfg(feature = "openai")]
pub use runner::default_model_provider;

/// MCP client: stdio and streamable HTTP transports, tool filters, approval policy.
#[cfg(feature = "mcp")]
pub use openai_agents_mcp::{
    McpClient, McpError, McpServerManager, RequireApproval, StdioParams, StreamableHttpParams,
    ToolFilter, ToolFilterContext,
};

/// SQLite-backed [`Session`] (Python: `SQLiteSession`).
#[cfg(feature = "sqlite")]
pub use openai_agents_sqlite::SqliteSession;

/// Attribute macro: turn a function into a [`FunctionTool`] constructor (Python: `@function_tool`).
pub use openai_agents_macros::function_tool;

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
