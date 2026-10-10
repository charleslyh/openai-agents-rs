//! MCP data types and result rendering.

use serde_json::Value;

/// A tool a server offers (Python: `mcp.types.Tool`).
#[derive(Debug, Clone, PartialEq)]
pub struct McpTool {
    /// Tool name, unique within the server.
    pub name: String,
    /// What the tool does.
    pub description: Option<String>,
    /// Human readable title.
    pub title: Option<String>,
    /// JSON Schema of the arguments.
    pub input_schema: Value,
}

/// Answer of a tool call (Python: `mcp.types.CallToolResult`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct McpCallToolResult {
    /// Content blocks (`text`, `image`, ...), as the server sent them.
    pub content: Vec<Value>,
    /// Structured result, when the tool declares an output schema.
    pub structured_content: Option<Value>,
    /// The server flags the call as failed. The content then explains why and is shown to the
    /// model like any other result.
    pub is_error: bool,
}

/// The text the model sees for a tool result.
///
/// Text blocks are joined with newlines. With `use_structured_content` and a successful call, the
/// structured result is sent instead, as JSON. Other blocks (images, resources) cannot travel in a
/// Chat Completions tool message, so an image becomes a short note and anything else its JSON.
/// (Python returns the content blocks as a list of tool output parts; a plain string works with
/// every provider.)
pub fn render_tool_result(result: &McpCallToolResult, use_structured_content: bool) -> String {
    if use_structured_content && !result.is_error {
        if let Some(structured) = &result.structured_content {
            return structured.to_string();
        }
    }
    result
        .content
        .iter()
        .map(|item| match item.get("type").and_then(Value::as_str) {
            Some("text") => item
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            Some("image") => format!(
                "[image: {}]",
                item.get("mimeType")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown type")
            ),
            _ => item.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}
