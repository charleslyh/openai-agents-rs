//! Model Context Protocol client (Python: `agents.mcp`).
//!
//! An MCP server offers tools over a standard protocol, so the same tool works with any model:
//! the SDK lists the server's tools, turns each into a function tool the model can call, and
//! forwards the call. Give servers to an agent with [`Agent::mcp_servers`](crate::Agent::mcp_servers).
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

mod client;
mod filter;
mod rpc;
mod server;
mod tools;
mod types;

#[cfg(feature = "mcp")]
mod http;
#[cfg(feature = "mcp")]
mod stdio;

mod manager;
pub use manager::McpServerManager;

pub use client::McpClient;
pub use filter::{McpConfig, RequireApproval, ToolFilter, ToolFilterContext};
pub use rpc::McpError;
pub use server::McpServer;
pub use tools::mcp_function_tools;
pub use types::{render_tool_result, McpCallToolResult, McpTool};

#[cfg(feature = "mcp")]
pub use http::StreamableHttpParams;
#[cfg(feature = "mcp")]
pub use stdio::StdioParams;

pub(crate) use types::MAX_TOOL_PAGES;

#[cfg(test)]
mod tests;
