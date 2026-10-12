//! JSON encoding of the pieces of a [`super::RunState`].

use serde_json::{json, Value};

use crate::error::{AgentsError, ToolGuardrailBehavior, ToolGuardrailFunctionOutput, UserError};
use crate::items::{
    HandoffCallItem, HandoffOutputItem, InputLike, MessageOutputItem, ModelResponse, ReasoningItem,
    RunItem, ToolCallItem, ToolCallOutputItem, ToolInterruptionItem,
};

pub(super) fn serialize_input(input: &InputLike) -> Value {
    match input {
        InputLike::Text(s) => json!({"type": "text", "text": s}),
        InputLike::Items(items) => json!({"type": "items", "items": items}),
    }
}

pub(super) fn deserialize_input(value: &Value) -> Result<InputLike, AgentsError> {
    match value.get("type").and_then(|v| v.as_str()) {
        Some("text") => Ok(InputLike::Text(
            value
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        )),
        Some("items") => Ok(InputLike::Items(
            value
                .get("items")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default(),
        )),
        _ => Err(UserError::new("invalid RunState input").into()),
    }
}

pub(super) fn serialize_model_response(r: &ModelResponse) -> Value {
    json!({
        "output": r.output,
        "usage": r.usage,
        "response_id": r.response_id,
        "request_id": r.request_id,
        "raw_usage": r.raw_usage,
    })
}

pub(super) fn deserialize_model_response(value: &Value) -> Result<ModelResponse, AgentsError> {
    Ok(ModelResponse {
        output: value
            .get("output")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default(),
        usage: serde_json::from_value(value.get("usage").cloned().unwrap_or(json!({})))
            .unwrap_or_default(),
        response_id: value
            .get("response_id")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        request_id: value
            .get("request_id")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        raw_usage: value.get("raw_usage").filter(|v| !v.is_null()).cloned(),
    })
}

/// `(guardrail_name, output)` of each entry of a stored guardrail result list.
pub(super) fn guardrail_results<'a>(
    value: &'a Value,
    key: &str,
) -> impl Iterator<Item = (String, Value)> + 'a {
    value
        .get(key)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            Some((
                entry.get("guardrail_name")?.as_str()?.to_string(),
                entry.get("output").cloned().unwrap_or(Value::Null),
            ))
        })
}

pub(super) fn serialize_tool_guardrail_output(output: &ToolGuardrailFunctionOutput) -> Value {
    let behavior = match &output.behavior {
        ToolGuardrailBehavior::Allow => json!({"type": "allow"}),
        ToolGuardrailBehavior::RejectContent { message } => {
            json!({"type": "reject_content", "message": message})
        }
        ToolGuardrailBehavior::RaiseException => json!({"type": "raise_exception"}),
    };
    json!({"output_info": output.output_info, "behavior": behavior})
}

pub(super) fn deserialize_tool_guardrail_output(value: &Value) -> ToolGuardrailFunctionOutput {
    let behavior = value.get("behavior").unwrap_or(&Value::Null);
    ToolGuardrailFunctionOutput {
        output_info: value.get("output_info").cloned().unwrap_or(Value::Null),
        behavior: match behavior.get("type").and_then(Value::as_str) {
            Some("reject_content") => ToolGuardrailBehavior::RejectContent {
                message: behavior
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            },
            Some("raise_exception") => ToolGuardrailBehavior::RaiseException,
            _ => ToolGuardrailBehavior::Allow,
        },
    }
}

pub(super) fn serialize_interruption_item(item: &ToolInterruptionItem) -> Value {
    json!({
        "type": "tool_interruption_item",
        "agent_name": item.agent_name,
        "tool_name": item.tool_name,
        "call_id": item.call_id,
        "arguments": item.arguments,
        "raw_item": item.raw_item,
    })
}

pub(super) fn deserialize_interruption_item(
    value: &Value,
) -> Result<ToolInterruptionItem, AgentsError> {
    Ok(ToolInterruptionItem {
        agent_name: value
            .get("agent_name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        tool_name: value
            .get("tool_name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        call_id: value
            .get("call_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        arguments: value
            .get("arguments")
            .and_then(|v| v.as_str())
            .unwrap_or("{}")
            .to_string(),
        raw_item: value.get("raw_item").cloned().unwrap_or(Value::Null),
    })
}

pub(super) fn serialize_run_item(item: &RunItem) -> Value {
    match item {
        RunItem::Message(m) => json!({
            "kind": "message",
            "agent_name": m.agent_name,
            "raw_item": m.raw_item,
        }),
        RunItem::ToolCall(t) => json!({
            "kind": "tool_call",
            "agent_name": t.agent_name,
            "raw_item": t.raw_item,
        }),
        RunItem::ToolCallOutput(o) => json!({
            "kind": "tool_call_output",
            "agent_name": o.agent_name,
            "raw_item": o.raw_item,
            "output": o.output,
        }),
        RunItem::ToolInterruption(a) => json!({
            "kind": "tool_interruption",
            "item": serialize_interruption_item(a),
        }),
        RunItem::HandoffCall(h) => json!({
            "kind": "handoff_call",
            "agent_name": h.agent_name,
            "raw_item": h.raw_item,
        }),
        RunItem::HandoffOutput(h) => json!({
            "kind": "handoff_output",
            "agent_name": h.agent_name,
            "raw_item": h.raw_item,
            "source_agent_name": h.source_agent_name,
            "target_agent_name": h.target_agent_name,
        }),
        RunItem::Reasoning(r) => json!({
            "kind": "reasoning",
            "agent_name": r.agent_name,
            "raw_item": r.raw_item,
        }),
    }
}

pub(super) fn agent_name_field(value: &Value) -> String {
    value
        .get("agent_name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

pub(super) fn raw_item_field(value: &Value) -> Value {
    value.get("raw_item").cloned().unwrap_or(Value::Null)
}

pub(super) fn deserialize_run_item(value: &Value) -> Result<RunItem, AgentsError> {
    match value.get("kind").and_then(|v| v.as_str()) {
        Some("message") => Ok(RunItem::Message(MessageOutputItem {
            agent_name: value
                .get("agent_name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            raw_item: value.get("raw_item").cloned().unwrap_or(Value::Null),
        })),
        Some("tool_call") => Ok(RunItem::ToolCall(ToolCallItem {
            agent_name: value
                .get("agent_name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            raw_item: value.get("raw_item").cloned().unwrap_or(Value::Null),
        })),
        Some("tool_call_output") => Ok(RunItem::ToolCallOutput(ToolCallOutputItem {
            agent_name: value
                .get("agent_name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            raw_item: value.get("raw_item").cloned().unwrap_or(Value::Null),
            output: value.get("output").cloned().unwrap_or(Value::Null),
        })),
        Some("tool_interruption" | "tool_approval") => {
            Ok(RunItem::ToolInterruption(deserialize_interruption_item(
                value
                    .get("item")
                    .ok_or_else(|| UserError::new("tool_interruption missing item"))?,
            )?))
        }
        Some("handoff_call") => Ok(RunItem::HandoffCall(HandoffCallItem {
            agent_name: agent_name_field(value),
            raw_item: raw_item_field(value),
        })),
        Some("handoff_output") => Ok(RunItem::HandoffOutput(HandoffOutputItem {
            agent_name: agent_name_field(value),
            raw_item: raw_item_field(value),
            source_agent_name: value
                .get("source_agent_name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            target_agent_name: value
                .get("target_agent_name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        })),
        Some("reasoning") => Ok(RunItem::Reasoning(ReasoningItem {
            agent_name: agent_name_field(value),
            raw_item: raw_item_field(value),
        })),
        other => Err(UserError::new(format!("unknown RunItem kind: {other:?}")).into()),
    }
}
