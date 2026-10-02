//! Run / model items in OpenAI Responses JSON shape (Python: `agents.items`).

use serde_json::{json, Value};

use crate::usage::Usage;

/// Responses API input item (`ResponseInputItemParam` as JSON).
pub type ResponseInputItem = Value;
/// Responses API output item (`ResponseOutputItem` as JSON).
pub type ResponseOutputItem = Value;

/// Provider-neutral model response (Python: `ModelResponse`).
#[derive(Debug, Clone)]
pub struct ModelResponse {
    /// Output items (messages, function calls, …).
    pub output: Vec<ResponseOutputItem>,
    /// Usage for this call.
    pub usage: Usage,
    /// Response id (Responses API).
    pub response_id: Option<String>,
    /// Transport request id when available.
    pub request_id: Option<String>,
}

impl ModelResponse {
    /// Convert output items into input items for the next model turn.
    pub fn to_input_items(&self) -> Vec<ResponseInputItem> {
        self.output.clone()
    }
}

/// An item produced during a run (Python: `RunItem` union, Phase-1 subset).
#[derive(Debug, Clone)]
pub enum RunItem {
    /// Assistant message.
    Message(MessageOutputItem),
    /// Tool / function call request from the model.
    ToolCall(ToolCallItem),
    /// Tool call result sent back to the model.
    ToolCallOutput(ToolCallOutputItem),
    /// Pending human approval for a tool call (not sent to the model as input).
    ToolApproval(ToolApprovalItem),
}

impl RunItem {
    /// Originating agent name.
    pub fn agent_name(&self) -> &str {
        match self {
            Self::Message(i) => &i.agent_name,
            Self::ToolCall(i) => &i.agent_name,
            Self::ToolCallOutput(i) => &i.agent_name,
            Self::ToolApproval(i) => &i.agent_name,
        }
    }

    /// Underlying Responses-format JSON.
    ///
    /// For [`Self::ToolApproval`], returns the pending function_call raw item (not a model input).
    pub fn raw_item(&self) -> &Value {
        match self {
            Self::Message(i) => &i.raw_item,
            Self::ToolCall(i) => &i.raw_item,
            Self::ToolCallOutput(i) => &i.raw_item,
            Self::ToolApproval(i) => &i.raw_item,
        }
    }

    /// Whether this item may be forwarded as model input (Python skips `ToolApprovalItem`).
    pub fn is_model_input(&self) -> bool {
        !matches!(self, Self::ToolApproval(_))
    }
}

/// Message output item.
#[derive(Debug, Clone)]
pub struct MessageOutputItem {
    /// Agent that produced the item.
    pub agent_name: String,
    /// Raw Responses message object.
    pub raw_item: ResponseOutputItem,
}

/// Function/tool call item.
#[derive(Debug, Clone)]
pub struct ToolCallItem {
    /// Agent that produced the item.
    pub agent_name: String,
    /// Raw function_call object.
    pub raw_item: ResponseOutputItem,
}

/// Function/tool call output item.
#[derive(Debug, Clone)]
pub struct ToolCallOutputItem {
    /// Agent that produced the item.
    pub agent_name: String,
    /// Raw function_call_output object.
    pub raw_item: ResponseInputItem,
    /// Structured / string tool output.
    pub output: Value,
}

/// Pending tool-approval interruption (Python: `ToolApprovalItem`).
#[derive(Debug, Clone)]
pub struct ToolApprovalItem {
    /// Agent that requested the tool call.
    pub agent_name: String,
    /// Tool name.
    pub tool_name: String,
    /// Model call id.
    pub call_id: String,
    /// Raw JSON arguments string.
    pub arguments: String,
    /// Underlying function_call object.
    pub raw_item: ResponseOutputItem,
}

impl ToolApprovalItem {
    /// Convenience alias for tool name (Python: `ToolApprovalItem.name`).
    pub fn name(&self) -> &str {
        &self.tool_name
    }
}

/// Helpers for building and inspecting items (Python: `ItemHelpers`).
pub struct ItemHelpers;

impl ItemHelpers {
    /// Normalize string or list input into a list of input items.
    pub fn input_to_new_input_list(input: &InputLike) -> Vec<ResponseInputItem> {
        match input {
            InputLike::Text(s) => vec![json!({"role": "user", "content": s})],
            InputLike::Items(items) => items.clone(),
        }
    }

    /// Extract concatenated assistant text from message run items.
    pub fn text_message_outputs(items: &[RunItem]) -> String {
        items
            .iter()
            .filter_map(|i| match i {
                RunItem::Message(m) => Some(Self::text_message_output(m)),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Extract text from a single message item.
    pub fn text_message_output(message: &MessageOutputItem) -> String {
        extract_message_text(&message.raw_item).unwrap_or_default()
    }

    /// Build a standard assistant text message output item (test helper shape).
    pub fn text_message(content: impl Into<String>) -> ResponseOutputItem {
        let content = content.into();
        json!({
            "id": "1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": content,
                "annotations": [],
                "logprobs": []
            }]
        })
    }

    /// Build a function_call output item (test helper shape).
    pub fn function_tool_call(
        name: impl Into<String>,
        arguments: impl Into<String>,
        call_id: impl Into<String>,
    ) -> ResponseOutputItem {
        json!({
            "id": "1",
            "type": "function_call",
            "name": name.into(),
            "arguments": arguments.into(),
            "call_id": call_id.into()
        })
    }

    /// Build a function_call_output input item.
    pub fn function_call_output(
        call_id: impl Into<String>,
        output: impl Into<String>,
    ) -> ResponseInputItem {
        json!({
            "type": "function_call_output",
            "call_id": call_id.into(),
            "output": output.into()
        })
    }
}

/// Input accepted by `Runner::run` (string or Responses input list).
#[derive(Debug, Clone)]
pub enum InputLike {
    /// Plain user text.
    Text(String),
    /// Pre-built Responses input items.
    Items(Vec<ResponseInputItem>),
}

impl From<&str> for InputLike {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for InputLike {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Vec<ResponseInputItem>> for InputLike {
    fn from(value: Vec<ResponseInputItem>) -> Self {
        Self::Items(value)
    }
}

/// Extract assistant text from a message-shaped output item.
pub fn extract_message_text(item: &Value) -> Option<String> {
    if item.get("type").and_then(|t| t.as_str()) != Some("message") {
        return None;
    }
    let content = item.get("content")?.as_array()?;
    let mut out = String::new();
    for part in content {
        if part.get("type").and_then(|t| t.as_str()) == Some("output_text") {
            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                out.push_str(text);
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// True if the item is a function_call.
pub fn is_function_call(item: &Value) -> bool {
    item.get("type").and_then(|t| t.as_str()) == Some("function_call")
}

/// Parse function call fields.
pub fn function_call_parts(item: &Value) -> Option<(String, String, String)> {
    if !is_function_call(item) {
        return None;
    }
    let name = item.get("name")?.as_str()?.to_string();
    let arguments = item
        .get("arguments")
        .and_then(|a| a.as_str())
        .unwrap_or("")
        .to_string();
    let call_id = item
        .get("call_id")
        .and_then(|c| c.as_str())
        .unwrap_or("call")
        .to_string();
    Some((name, arguments, call_id))
}
