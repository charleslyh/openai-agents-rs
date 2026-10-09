//! Tool filtering and approval policy (Python: `ToolFilter`, `require_approval`).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::agent::Agent;
use crate::run_context::RunContextWrapper;
use crate::tool::NeedsApproval;

use super::McpTool;

// ---------------------------------------------------------------------------------------------
// Tool filtering and approval

/// What a dynamic tool filter is told (Python: `ToolFilterContext`).
#[derive(Clone)]
pub struct ToolFilterContext {
    /// The run context.
    pub run_context: RunContextWrapper,
    /// The agent listing the tools.
    pub agent: Arc<Agent>,
    /// Name of the server.
    pub server_name: String,
}

type DynamicFilter = Arc<
    dyn Fn(ToolFilterContext, McpTool) -> Pin<Box<dyn Future<Output = bool> + Send>> + Send + Sync,
>;

/// Which of a server's tools the agent may use (Python: `ToolFilter`).
#[derive(Clone)]
pub enum ToolFilter {
    /// Fixed lists. `allowed` is applied first, then `blocked`.
    Static {
        /// If set, only these tools pass.
        allowed: Option<Vec<String>>,
        /// These tools never pass.
        blocked: Option<Vec<String>>,
    },
    /// Decide per tool, per listing.
    Dynamic(DynamicFilter),
}

impl std::fmt::Debug for ToolFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static { allowed, blocked } => f
                .debug_struct("Static")
                .field("allowed", allowed)
                .field("blocked", blocked)
                .finish(),
            Self::Dynamic(_) => f.write_str("Dynamic(..)"),
        }
    }
}

impl ToolFilter {
    /// Only these tools.
    pub fn allow<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Static {
            allowed: Some(names.into_iter().map(Into::into).collect()),
            blocked: None,
        }
    }

    /// Everything except these tools.
    pub fn block<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Static {
            allowed: None,
            blocked: Some(names.into_iter().map(Into::into).collect()),
        }
    }

    /// Decide with an async closure.
    pub fn dynamic<F, Fut>(f: F) -> Self
    where
        F: Fn(ToolFilterContext, McpTool) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        Self::Dynamic(Arc::new(move |ctx, tool| Box::pin(f(ctx, tool))))
    }
}

/// Which tool calls need human approval (Python: `require_approval`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum RequireApproval {
    /// No call needs approval.
    #[default]
    Never,
    /// Every call needs approval.
    Always,
    /// Per tool; a tool in neither list needs no approval.
    PerTool {
        /// Tools that always need approval.
        always: Vec<String>,
        /// Tools that never do.
        never: Vec<String>,
    },
}

impl RequireApproval {
    pub(crate) fn for_tool(&self, name: &str) -> NeedsApproval {
        NeedsApproval::Fixed(match self {
            Self::Never => false,
            Self::Always => true,
            Self::PerTool { always, .. } => always.iter().any(|n| n == name),
        })
    }
}

/// How an agent turns MCP tools into function tools (Python: `MCPConfig`).
#[derive(Debug, Clone, Copy, Default)]
pub struct McpConfig {
    /// Rewrite each tool's input schema into OpenAI's strict form. A schema that cannot be made
    /// strict is used as it is.
    pub convert_schemas_to_strict: bool,
}
