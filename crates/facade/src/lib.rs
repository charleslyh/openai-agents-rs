//! OpenAI Agents SDK for Rust — port of `openai-agents` Python v0.23.1.
//!
//! This crate is the single entry point: it re-exports everything from `openai-agents-core` under
//! the paths the SDK has always had, and adds the `#[function_tool]` macro. The Cargo features
//! (`openai`, `mcp`, `sqlite`, `providers`) pull in the crate that implements them. Core knows no concrete model
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

/// Provider presets, discovery and the capability matrix.
#[cfg(feature = "providers")]
pub use openai_agents_providers as providers;

/// Attribute macro: turn a function into a [`FunctionTool`] constructor (Python: `@function_tool`).
pub use openai_agents_macros::function_tool;

/// Run one agent end to end, taking the provider from the environment (Rust-only, X-015).
///
/// The shortest thing that works: no `Agent`, no `ModelProvider`, no `RunOptions`. The provider is
/// whatever the environment configures — a preset whose key is set, an endpoint declared with
/// `AGENTS_PROVIDER_1_ENDPOINT`, or the offline model when `AGENTS_OFFLINE=1` — so the same line
/// works before a key exists and after.
///
/// ```no_run
/// # async fn go() -> Result<(), openai_agents_core::AgentsError> {
/// let result = openai_agents::run("You answer in one sentence.", "What is Rust?").await?;
/// println!("{}", result.final_output_as_str().unwrap_or_default());
/// # Ok(())
/// # }
/// ```
///
/// With the `providers` feature off, no provider is installed and the run falls back to whatever
/// [`Runtime`] has: with the `openai` feature, OpenAI from
/// `OPENAI_API_KEY`.
pub async fn run(
    instructions: impl Into<String>,
    input: impl Into<String>,
) -> Result<RunResult, AgentsError> {
    let agent = Agent::new("Assistant").instructions(instructions.into());
    let options = zero_config_options()?;
    Runner::run(&agent, input.into(), options).await
}

/// The model a run with no model named should use: `AGENTS_MODEL`, or `OPENAI_MODEL`, or what the
/// discovered provider suggests, or [`FALLBACK_MODEL`].
pub fn model_name_from_env() -> Option<String> {
    for var in ["AGENTS_MODEL", "OPENAI_MODEL"] {
        if let Some(name) = std::env::var(var)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
        {
            return Some(name);
        }
    }
    None
}

/// The model used when neither the environment nor the provider names one.
pub const FALLBACK_MODEL: &str = "gpt-4o-mini";

/// `RunOptions` for a run that named no provider and no model: both come from the environment.
///
/// Errors when nothing at all is configured: the message says which variables to set.
pub fn zero_config_options() -> Result<RunOptions, AgentsError> {
    #[cfg(feature = "providers")]
    {
        let registry = openai_agents_providers::ProviderRegistry::from_env()?;
        let model = model_name_from_env()
            .or_else(|| registry.default_model().map(str::to_string))
            .unwrap_or_else(|| FALLBACK_MODEL.to_string());
        let provider = registry.into_provider()?;
        Ok(RunOptions::new().provider(Some(provider)).model(model))
    }
    #[cfg(not(feature = "providers"))]
    {
        let model = model_name_from_env().unwrap_or_else(|| FALLBACK_MODEL.to_string());
        Ok(RunOptions::new().model(model))
    }
}

/// The types most programs need, for a single glob import: `use openai_agents::prelude::*;`.
///
/// Same set as [`openai_agents_core::prelude`], plus the `#[function_tool]` macro, with this
/// crate's [`Runner`] (which installs the default OpenAI provider) in place of the
/// core one.
///
/// ```no_run
/// use openai_agents::prelude::*;
///
/// #[tokio::main]
/// async fn main() -> Result<(), AgentsError> {
///     let agent = Agent::new("Assistant")
///         .instructions("You only respond in haikus.")
///         .model_name(std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".into()));
///     let result = Runner::run(&agent, "Tell me about recursion.", RunOptions::new()).await?;
///     println!("{}", result.final_output);
///     Ok(())
/// }
/// ```
pub mod prelude {
    pub use openai_agents_core::prelude::*;

    pub use crate::function_tool;
    pub use crate::Runner;
}

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
