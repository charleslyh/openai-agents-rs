//! Model Context Protocol client for the Rust OpenAI Agents SDK (Python: `agents.mcp`).
//!
//! An MCP server offers tools over a standard protocol, so the same tool works with any model:
//! the SDK lists the server's tools, turns each into a function tool the model can call, and
//! forwards the call. Give servers to an agent with `Agent::mcp_servers`.
//!
//! The contract an agent depends on (`McpServer`, `McpTool`, `McpCallToolResult`, `McpConfig`,
//! `mcp_function_tools`) lives in `opai-agents-core`. This crate is one implementation of it,
//! [`McpClient`], plus the tool filters, approval policy and [`McpServerManager`] that go with it.
//!
//! ```ignore
//! let server = McpClient::stdio(StdioParams::new("npx").args(["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]))
//!     .shared();
//! let agent = Agent::new("files").mcp_servers(vec![server.clone()]);
//! let result = Runner::run(&agent, "list the files", RunOptions::default()).await?;
//! server.cleanup().await?;
//! ```
//!
//! Supported: stdio and streamable HTTP transports, tool listing with pagination and optional
//! caching, static and dynamic tool filters, approval policies, `structuredContent`, and
//! strict-schema conversion. A server connects on first use (`connect()` does it eagerly) and the
//! child process of a stdio server is killed when the client is dropped.
//!
//! Not ported: the legacy SSE transport, prompts and resources, tool name prefixing
//! (`include_server_in_tool_names`), `tool_meta_resolver`, per-server retries and tool guardrails.
//! The OpenAI-hosted MCP tool (`HostedMCPTool`) is out of scope.

#![deny(missing_docs)]

mod client;
mod filter;
mod http;
mod manager;
mod rpc;
mod stdio;

#[cfg(test)]
mod tests;

pub use client::McpClient;
pub use filter::{RequireApproval, ToolFilter, ToolFilterContext};
pub use http::StreamableHttpParams;
pub use manager::McpServerManager;
pub use rpc::McpError;
pub use stdio::StdioParams;
