//! Chat Completions reply `message` -> Responses output items.

use serde_json::{json, Value};

use openai_agents_core::model::wire_events::FAKE_RESPONSES_ID;

use super::{str_field, REASONING_FIELD_KEY};

/// Text of a reply `content` field. Most servers send a string; some send a list of parts.
fn reply_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// `url_citation` annotations of a reply message (Python: `convert_url_citations`).
fn url_citations(message: &Value) -> Vec<Value> {
    message
        .get("annotations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|a| str_field(a, "type") == Some("url_citation"))
        .filter_map(|a| {
            let c = a.get("url_citation")?;
            Some(json!({
                "type": "url_citation",
                "start_index": c.get("start_index")?.as_i64()?,
                "end_index": c.get("end_index")?.as_i64()?,
                "url": str_field(c, "url")?,
                "title": str_field(c, "title")?,
            }))
        })
        .collect()
}

/// Arguments of a tool call as the text the runner parses: servers send a JSON string, a few send
/// the JSON object itself.
fn tool_call_arguments(call: &Value) -> String {
    match call.pointer("/function/arguments") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Turn a Chat Completions reply `message` into Responses output items
/// (Python: `message_to_output_items`): a reasoning item when the server sent its chain of
/// thought, then one assistant message with the text and any refusal, then one `function_call`
/// per tool call.
///
/// Differences from Python, for servers that deviate from the OpenAI wire format: `content` may be
/// a list of parts, tool call `arguments` may be a JSON object, and a tool call without an `id`
/// gets a generated one (the runner pairs outputs by call id).
pub fn chat_message_to_output_items(message: &Value) -> Vec<Value> {
    chat_message_to_output_items_with(message, None)
}

/// [`chat_message_to_output_items`] that also records where the items came from
/// (Python: the `provider_data` argument, `{"model": ..., "response_id": ...}` for a model call).
///
/// The origin is what later lets a reply's chain of thought be sent back only to the model that
/// wrote it. Besides that, a Claude `thinking_blocks` list is kept on the reasoning item, and a
/// Gemini `thought_signature` (`tool_calls[].extra_content.google`) on its function call.
pub fn chat_message_to_output_items_with(
    message: &Value,
    provider_data: Option<&Value>,
) -> Vec<Value> {
    let mut items = Vec::new();
    let origin = provider_data
        .and_then(Value::as_object)
        .filter(|d| !d.is_empty());

    let reasoning_content = str_field(message, "reasoning_content").filter(|t| !t.is_empty());
    let thinking_blocks: Vec<Value> = message
        .get("thinking_blocks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|b| b.is_object())
        .cloned()
        .collect();
    // The structured fields win when a server sends more than one.
    let reasoning_field = str_field(message, "reasoning")
        .filter(|t| !t.is_empty())
        .filter(|_| reasoning_content.is_none() && thinking_blocks.is_empty());
    if reasoning_content.is_some() || reasoning_field.is_some() || !thinking_blocks.is_empty() {
        let mut reasoning = json!({
            "id": FAKE_RESPONSES_ID,
            "summary": reasoning_content
                .map(|text| vec![json!({"text": text, "type": "summary_text"})])
                .unwrap_or_default(),
            "type": "reasoning",
        });
        if let Some(text) = reasoning_field {
            reasoning["content"] = json!([{"text": text, "type": "reasoning_text"}]);
        }
        let mut data = origin.cloned().unwrap_or_default();
        if reasoning_field.is_some() {
            data.insert(REASONING_FIELD_KEY.into(), "reasoning".into());
        }
        if !thinking_blocks.is_empty() {
            // The normalized fields cannot hold empty or redacted blocks, so the provider's own
            // list is the source of truth for replay and the fields below are derived from it.
            let mut texts = Vec::new();
            let mut signatures = Vec::new();
            for block in &thinking_blocks {
                if let Some(text) = str_field(block, "thinking").filter(|t| !t.is_empty()) {
                    texts.push(json!({"text": text, "type": "reasoning_text"}));
                }
                if let Some(signature) = str_field(block, "signature").filter(|s| !s.is_empty()) {
                    signatures.push(signature);
                }
            }
            reasoning["content"] = Value::Array(texts);
            if !signatures.is_empty() {
                reasoning["encrypted_content"] = signatures.join("\n").into();
            }
            data.insert("thinking_blocks".into(), Value::Array(thinking_blocks));
        }
        if !data.is_empty() {
            reasoning["provider_data"] = Value::Object(data);
        }
        items.push(reasoning);
    }

    let mut content = Vec::new();
    let text = reply_text(message.get("content"));
    if !text.is_empty() {
        content.push(json!({
            "annotations": url_citations(message),
            "text": text,
            "type": "output_text",
            "logprobs": [],
        }));
    }
    if let Some(refusal) = str_field(message, "refusal").filter(|r| !r.is_empty()) {
        content.push(json!({"refusal": refusal, "type": "refusal"}));
    }
    if !content.is_empty() {
        let mut message_item = json!({
            "id": FAKE_RESPONSES_ID,
            "content": content,
            "role": "assistant",
            "type": "message",
            "status": "completed",
        });
        if let Some(origin) = origin {
            message_item["provider_data"] = Value::Object(origin.clone());
        }
        items.push(message_item);
    }

    for call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        // Servers that omit `type` still mean a function call; other kinds are not supported.
        if !matches!(str_field(call, "type"), None | Some("function")) {
            continue;
        }
        let id = str_field(call, "id")
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple()));
        let mut item = json!({
            "id": FAKE_RESPONSES_ID,
            "call_id": id,
            "arguments": tool_call_arguments(call),
            "name": call.pointer("/function/name").and_then(Value::as_str).unwrap_or_default(),
            "type": "function_call",
        });
        let mut data = origin.cloned().unwrap_or_default();
        if let Some(signature) = call
            .pointer("/extra_content/google/thought_signature")
            .filter(|s| s.as_str().is_some_and(|s| !s.is_empty()))
        {
            data.insert("thought_signature".into(), signature.clone());
        }
        if !data.is_empty() {
            item["provider_data"] = Value::Object(data);
        }
        items.push(item);
    }
    items
}
