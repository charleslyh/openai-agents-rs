//! Recording tool results as run items and reading them back.

use serde_json::Value;

use crate::agent::Agent;
use crate::items::{ItemHelpers, RunItem, ToolCallOutputItem};
use crate::stream_events::{RunItemStreamName, StreamEvent};

use crate::run::agent_loop::{emit, EventTx};

/// Shape a tool result into the run's final output.
///
/// Python (`_maybe_finalize_from_tool_results`) coerces the tool output to `str` unless the
/// agent declared a non-plain-text `output_type`.
pub(crate) fn finalize_tool_output(agent: &Agent, output: &Value) -> Value {
    match agent.output_type.as_deref() {
        Some(schema) if !schema.is_plain_text() => output.clone(),
        _ => Value::String(value_to_tool_string(output)),
    }
}

/// Append (and stream) a tool output item unless one already exists for the call id.
pub(crate) async fn push_tool_output(
    events: &Option<EventTx>,
    generated_items: &mut Vec<RunItem>,
    agent_name: &str,
    call_id: &str,
    output: &Value,
) {
    if has_tool_output(generated_items, call_id) {
        return;
    }
    let out_item = tool_output_item(agent_name, call_id, output);
    emit(
        events,
        StreamEvent::RunItem {
            name: RunItemStreamName::ToolOutput,
            item: out_item.clone(),
        },
    )
    .await;
    generated_items.push(out_item);
}

pub(crate) fn has_tool_output(items: &[RunItem], call_id: &str) -> bool {
    items.iter().any(|gi| {
        matches!(
            gi,
            RunItem::ToolCallOutput(o)
                if o.raw_item.get("call_id").and_then(|c| c.as_str()) == Some(call_id)
        )
    })
}

pub(crate) fn existing_tool_output(items: &[RunItem], call_id: &str) -> Option<Value> {
    items.iter().find_map(|gi| match gi {
        RunItem::ToolCallOutput(o)
            if o.raw_item.get("call_id").and_then(|c| c.as_str()) == Some(call_id) =>
        {
            Some(o.output.clone())
        }
        _ => None,
    })
}

pub(crate) fn tool_output_item(agent_name: &str, call_id: &str, output: &Value) -> RunItem {
    let raw = ItemHelpers::function_call_output(call_id, value_to_tool_string(output));
    RunItem::ToolCallOutput(ToolCallOutputItem {
        agent_name: agent_name.to_string(),
        raw_item: raw,
        output: output.clone(),
    })
}

pub(crate) fn value_to_tool_string(output: &Value) -> String {
    match output {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}
