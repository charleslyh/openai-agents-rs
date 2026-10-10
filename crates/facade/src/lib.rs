//! OpenAI Agents SDK for Rust — port of `openai-agents` Python v0.23.1.
//!
//! This crate is the single entry point: it re-exports everything from `openai-agents-core` under
//! the paths the SDK has always had, and adds the `#[function_tool]` macro. The Cargo features
//! (`openai`, `mcp`, `sqlite`) are forwarded to the crate that implements them.
//!
//! The public surface mirrors `agents/__init__.py` for the supported subset; see
//! `docs/COMPAT.md` for the support matrix and `docs/DEVIATIONS.md` for recorded differences.

#![deny(missing_docs)]

pub use openai_agents_core::*;

/// Attribute macro: turn a function into a [`FunctionTool`] constructor (Python: `@function_tool`).
pub use openai_agents_macros::function_tool;

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
