//! Function tools (Python: `agents.tool.FunctionTool` Phase-1 subset).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::error::AgentsError;

/// Context passed to tool invocations (Python: `ToolContext` subset).
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Tool name being invoked.
    pub tool_name: String,
    /// Call id from the model.
    pub tool_call_id: String,
    /// Raw JSON arguments string from the model.
    pub tool_arguments: String,
}

/// Async tool invoker signature.
pub type ToolInvoker = Arc<
    dyn Fn(ToolContext, String) -> Pin<Box<dyn Future<Output = Result<Value, AgentsError>> + Send>>
        + Send
        + Sync,
>;

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
}

impl std::fmt::Debug for FunctionTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("params_json_schema", &self.params_json_schema)
            .field("strict_json_schema", &self.strict_json_schema)
            .field("is_enabled", &self.is_enabled)
            .finish()
    }
}

impl FunctionTool {
    /// Create a tool with an async invoker.
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
        }
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
