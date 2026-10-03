//! Function tools (Python: `agents.tool.FunctionTool` Phase-1 subset).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::error::AgentsError;
use crate::items::ToolApprovalItem;
use crate::run_context::RunContextWrapper;
use crate::run_state::RunState;

/// Default rejection text when a tool call is rejected (Python: `DEFAULT_APPROVAL_REJECTION_MESSAGE`).
pub const DEFAULT_APPROVAL_REJECTION_MESSAGE: &str = "Tool execution was not approved.";

/// Context passed to tool invocations (Python: `ToolContext` subset).
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Tool name being invoked.
    pub tool_name: String,
    /// Call id from the model.
    pub tool_call_id: String,
    /// Raw JSON arguments string from the model.
    pub tool_arguments: String,
    /// Run context shared with guardrails and hooks.
    pub run_context: RunContextWrapper,
}

impl ToolContext {
    /// Build a tool context from the run context.
    pub fn new(
        tool_name: impl Into<String>,
        tool_call_id: impl Into<String>,
        tool_arguments: impl Into<String>,
        run_context: RunContextWrapper,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            tool_call_id: tool_call_id.into(),
            tool_arguments: tool_arguments.into(),
            run_context,
        }
    }

    /// Convenience accessor for the user context.
    pub fn context<T: std::any::Any + Send + Sync>(&self) -> Option<&T> {
        self.run_context.context::<T>()
    }
}

/// Result of invoking a function tool (Python: `FunctionToolResult` subset).
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Tool output when the call completed (absent when interrupted).
    pub output: Option<Value>,
    /// Nested / bubbled approval interruptions.
    pub interruptions: Vec<ToolApprovalItem>,
    /// Nested agent run state when `Agent.as_tool` paused for HITL.
    pub nested_state: Option<Box<RunState>>,
}

impl ToolResult {
    /// Successful tool output.
    pub fn output(value: Value) -> Self {
        Self {
            output: Some(value),
            interruptions: Vec::new(),
            nested_state: None,
        }
    }

    /// Paused for nested approvals.
    pub fn interrupted(interruptions: Vec<ToolApprovalItem>, nested_state: RunState) -> Self {
        Self {
            output: None,
            interruptions,
            nested_state: Some(Box::new(nested_state)),
        }
    }
}

/// Async tool invoker signature.
pub type ToolInvoker = Arc<
    dyn Fn(ToolContext, String) -> Pin<Box<dyn Future<Output = Result<ToolResult, AgentsError>> + Send>>
        + Send
        + Sync,
>;

/// Dynamic approval policy (Python: `needs_approval` callable).
///
/// Arguments: parsed JSON params object, tool call id → whether approval is required.
pub type NeedsApprovalFn = Arc<
    dyn Fn(Value, String) -> Pin<Box<dyn Future<Output = bool> + Send>> + Send + Sync,
>;

/// When a function tool requires human approval (Python: `FunctionTool.needs_approval`).
#[derive(Clone)]
pub enum NeedsApproval {
    /// Fixed always / never.
    Fixed(bool),
    /// Per-call policy.
    Dynamic(NeedsApprovalFn),
}

impl Default for NeedsApproval {
    fn default() -> Self {
        Self::Fixed(false)
    }
}

impl From<bool> for NeedsApproval {
    fn from(value: bool) -> Self {
        Self::Fixed(value)
    }
}

impl NeedsApproval {
    /// Evaluate whether this call needs approval.
    pub async fn requires(&self, arguments_json: &str, call_id: &str) -> bool {
        match self {
            Self::Fixed(v) => *v,
            Self::Dynamic(f) => {
                let params = serde_json::from_str::<Value>(arguments_json)
                    .unwrap_or(Value::Object(Default::default()));
                f(params, call_id.to_string()).await
            }
        }
    }
}

/// A function tool exposed to the model (Python: `FunctionTool`).
#[derive(Clone)]
pub struct FunctionTool {
    /// Tool name shown to the LLM.
    pub name: String,
    /// Tool description shown to the LLM.
    pub description: String,
    /// JSON Schema for parameters.
    pub params_json_schema: Value,
    /// Invocation callback.
    pub on_invoke_tool: ToolInvoker,
    /// Whether the schema is strict.
    pub strict_json_schema: bool,
    /// Whether the tool is enabled.
    pub is_enabled: bool,
    /// Whether / when the tool pauses for human approval before invoke.
    pub needs_approval: NeedsApproval,
}

impl std::fmt::Debug for FunctionTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("params_json_schema", &self.params_json_schema)
            .field("strict_json_schema", &self.strict_json_schema)
            .field("is_enabled", &self.is_enabled)
            .field(
                "needs_approval",
                &matches!(self.needs_approval, NeedsApproval::Fixed(true)),
            )
            .finish()
    }
}

impl FunctionTool {
    /// Create a tool with an async invoker that returns a JSON [`Value`].
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        params_json_schema: Value,
        on_invoke: F,
    ) -> Self
    where
        F: Fn(ToolContext, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, AgentsError>> + Send + 'static,
    {
        let on_invoke = Arc::new(on_invoke);
        Self::new_with_result(name, description, params_json_schema, move |ctx, args| {
            let on_invoke = Arc::clone(&on_invoke);
            async move { Ok(ToolResult::output(on_invoke(ctx, args).await?)) }
        })
    }

    /// Create a tool whose invoker can return nested interruptions (`Agent.as_tool`).
    pub fn new_with_result<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        params_json_schema: Value,
        on_invoke: F,
    ) -> Self
    where
        F: Fn(ToolContext, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolResult, AgentsError>> + Send + 'static,
    {
        let on_invoke = Arc::new(on_invoke);
        Self {
            name: name.into(),
            description: description.into(),
            params_json_schema,
            on_invoke_tool: Arc::new(move |ctx, args| {
                let on_invoke = Arc::clone(&on_invoke);
                Box::pin(async move { on_invoke(ctx, args).await })
            }),
            strict_json_schema: true,
            is_enabled: true,
            needs_approval: NeedsApproval::Fixed(false),
        }
    }

    /// Require approval before every invoke (Python: `needs_approval=True`).
    pub fn with_needs_approval(mut self, needs: impl Into<NeedsApproval>) -> Self {
        self.needs_approval = needs.into();
        self
    }

    /// Dynamic approval policy from an async callback.
    pub fn with_needs_approval_fn<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(Value, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        let f = Arc::new(f);
        self.needs_approval = NeedsApproval::Dynamic(Arc::new(move |params, call_id| {
            let f = Arc::clone(&f);
            Box::pin(async move { f(params, call_id).await })
        }));
        self
    }

    /// Create a tool that returns a constant JSON/string value (handy for tests).
    pub fn constant(
        name: impl Into<String>,
        description: impl Into<String>,
        return_value: impl Into<String>,
    ) -> Self {
        let return_value = return_value.into();
        Self::new(
            name,
            description,
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            move |_ctx, _args| {
                let return_value = return_value.clone();
                async move { Ok(Value::String(return_value)) }
            },
        )
    }

    /// Convert to a Responses/Chat tool definition JSON fragment.
    pub fn to_function_tool_param(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.params_json_schema,
            "strict": self.strict_json_schema
        })
    }

    /// Chat Completions tools array entry.
    pub fn to_chat_tool(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params_json_schema,
                "strict": self.strict_json_schema
            }
        })
    }
}
