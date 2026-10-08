//! Python-flavoured JSON text, for strings the model reads.
//!
//! Python's `json.dumps(..., ensure_ascii=False)` separates items with `", "` and keys from
//! values with `": "`, and pydantic dumps keep model field order. Text built from these values
//! (handoff transfer messages, nested history summaries) is part of the model input, so it is
//! rendered the way the Python SDK renders it.

use serde_json::Value;

/// `json.dumps(value, ensure_ascii=False)`.
pub(crate) fn dumps(value: &Value) -> String {
    let mut out = String::new();
    write_value(value, &mut out);
    out
}

fn write_value(value: &Value, out: &mut String) {
    match value {
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push_str(": ");
                write_value(item, out);
            }
            out.push('}');
        }
        other => out.push_str(&other.to_string()),
    }
}

/// Field order of the pydantic models behind common Responses items, keyed by `type`.
fn field_order(item_type: &str) -> Option<&'static [&'static str]> {
    Some(match item_type {
        "message" => &["id", "content", "role", "status", "type"],
        "output_text" => &["annotations", "text", "type", "logprobs"],
        "refusal" => &["refusal", "type"],
        "function_call" => &[
            "arguments", "call_id", "name", "type", "id", "namespace", "status",
        ],
        "function_call_output" => &["call_id", "output", "type", "id", "status"],
        "reasoning" => &["id", "summary", "type", "content", "encrypted_content", "status"],
        "summary_text" | "reasoning_text" => &["text", "type"],
        _ => return None,
    })
}

/// Reorder the keys of known item types to the Python SDK's field order, recursively.
///
/// Unknown types and unknown keys keep their original relative order (known keys first).
pub(crate) fn python_field_order(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(python_field_order).collect()),
        Value::Object(map) => {
            let order = map
                .get("type")
                .and_then(Value::as_str)
                .and_then(field_order)
                .unwrap_or(&[]);
            let mut out = serde_json::Map::new();
            for key in order {
                if let Some(v) = map.get(*key) {
                    out.insert((*key).to_string(), python_field_order(v));
                }
            }
            for (key, v) in map {
                if !out.contains_key(key) {
                    out.insert(key.clone(), python_field_order(v));
                }
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}
