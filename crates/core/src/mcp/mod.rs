//! The MCP contract the core needs (Python: `agents.mcp`).
//!
//! An MCP server offers tools over a standard protocol, so the same tool works with any model:
//! the SDK lists the server's tools, turns each into a function tool the model can call, and
//! forwards the call. Give servers to an agent with [`Agent::mcp_servers`](crate::Agent::mcp_servers).
//!
//! This module holds only what an [`Agent`](crate::Agent) depends on: the [`McpServer`] trait, the
//! data types it exchanges, and [`mcp_function_tools`], which turns a server's tools into function
//! tools. The client with its stdio and streamable HTTP transports, tool filters, approval
//! policies and the server manager live in the `openai-agents-mcp` crate (the `mcp` feature of
//! `openai-agents`). Anything that implements [`McpServer`] works, including an in-process source.

mod server;
mod tools;
mod types;

pub use server::McpServer;
pub use tools::{mcp_function_tools, McpConfig};
pub use types::{render_tool_result, McpCallToolResult, McpTool};
