//! The server abstraction: what an agent needs from a tool source.

use async_trait::async_trait;
use serde_json::Value;

use crate::agent::Agent;
use crate::error::AgentsError;
use crate::run_context::RunContextWrapper;
use crate::tool::NeedsApproval;

use super::{McpCallToolResult, McpTool};

// ---------------------------------------------------------------------------------------------
// The server abstraction

/// A source of tools (Python: `MCPServer`).
///
/// `McpClient` implements it for real servers; implement it yourself for an in-process source.
#[async_trait]
pub trait McpServer: std::fmt::Debug + Send + Sync {
    /// Name used in messages and traces.
    fn name(&self) -> &str;

    /// Connect now. Optional: the first use connects on its own.
    async fn connect(&self) -> Result<(), AgentsError>;

    /// Disconnect and release resources. A later call reconnects.
    async fn cleanup(&self) -> Result<(), AgentsError>;

    /// The tools this agent may use, after filtering.
    async fn list_tools(
        &self,
        context: &RunContextWrapper,
        agent: &Agent,
    ) -> Result<Vec<McpTool>, AgentsError>;

    /// Call a tool.
    async fn call_tool(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> Result<McpCallToolResult, AgentsError>;

    /// Forget a cached tool list.
    fn invalidate_tools_cache(&self) {}

    /// Whether calls of `tool` need approval.
    fn needs_approval_for(&self, _tool: &McpTool) -> NeedsApproval {
        NeedsApproval::Fixed(false)
    }

    /// Send `structuredContent` instead of the content blocks when a result has it.
    fn use_structured_content(&self) -> bool {
        false
    }
}
