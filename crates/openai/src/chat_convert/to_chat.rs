//! Run input items -> Chat Completions `messages`.

use serde_json::{json, Map, Value};

use openai_agents_core::error::ModelError;

use super::{
    default_should_replay_reasoning, str_field, unsupported, ChatConvertOptions,
    OMITTED_TOOL_OUTPUT_PLACEHOLDER, REASONING_FIELD_KEY,
};

fn text_part(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

/// Content of a user message: a string stays a string, a part list becomes Chat Completions
/// content parts (Python: `extract_all_content`).
fn extract_all_content(content: &Value) -> Result<Value, ModelError> {
    let parts = match content {
        Value::String(_) => return Ok(content.clone()),
        Value::Array(parts) => parts,
        other => return Err(unsupported(format!("Unknown content: {other}"))),
    };
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        let kind = str_field(part, "type");
        match kind {
            // `output_text` is accepted too so an assistant message without an id still works.
            Some("input_text" | "text" | "output_text") => {
                let text = str_field(part, "text").ok_or_else(|| {
                    unsupported(format!("Only text content is supported here, got: {part}"))
                })?;
                out.push(text_part(text));
            }
            Some("input_image") => {
                let url = str_field(part, "image_url")
                    .filter(|u| !u.is_empty())
                    .ok_or_else(|| {
                        unsupported(format!(
                            "Only image URLs are supported for input_image {part}"
                        ))
                    })?;
                let detail = str_field(part, "detail").unwrap_or("auto");
                out.push(json!({"type": "image_url", "image_url": {"url": url, "detail": detail}}));
            }
            Some("image_url") => {
                let url = part
                    .pointer("/image_url/url")
                    .and_then(Value::as_str)
                    .filter(|u| !u.is_empty())
                    .ok_or_else(|| {
                        unsupported(format!(
                            "Only image URLs are supported for image_url {part}"
                        ))
                    })?;
                let detail = part
                    .pointer("/image_url/detail")
                    .and_then(Value::as_str)
                    .unwrap_or("auto");
                out.push(json!({"type": "image_url", "image_url": {"url": url, "detail": detail}}));
            }
            Some("input_audio") => {
                let audio = part.get("input_audio").filter(|a| a.is_object());
                let (data, format) = audio
                    .and_then(|a| Some((str_field(a, "data")?, str_field(a, "format")?)))
                    .filter(|(d, f)| !d.is_empty() && !f.is_empty())
                    .ok_or_else(|| {
                        unsupported(format!("input_audio requires both data and format {part}"))
                    })?;
                out.push(
                    json!({"type": "input_audio", "input_audio": {"data": data, "format": format}}),
                );
            }
            Some("input_file") => {
                let mut file = Map::new();
                if let Some(data) = str_field(part, "file_data").filter(|d| !d.is_empty()) {
                    file.insert("file_data".into(), data.into());
                } else if let Some(id) = str_field(part, "file_id").filter(|d| !d.is_empty()) {
                    file.insert("file_id".into(), id.into());
                } else {
                    return Err(unsupported(format!(
                        "Only file_data or file_id is supported for input_file {part}"
                    )));
                }
                if let Some(name) = str_field(part, "filename").filter(|n| !n.is_empty()) {
                    file.insert("filename".into(), name.into());
                }
                out.push(json!({"type": "file", "file": file}));
            }
            _ => return Err(unsupported(format!("Unknown content: {part}"))),
        }
    }
    Ok(Value::Array(out))
}

/// Like [`extract_all_content`] but keeps only text parts, for roles that cannot carry anything
/// else (Python: `extract_text_content`).
fn extract_text_content(content: &Value) -> Result<Value, ModelError> {
    match extract_all_content(content)? {
        Value::Array(parts) => Ok(Value::Array(
            parts
                .into_iter()
                .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
                .collect(),
        )),
        text => Ok(text),
    }
}

/// Whether `item` is an "easy" message: only `role`, `content` and optionally `type: message` /
/// `phase` (Python: `maybe_easy_input_message`).
fn is_easy_message(item: &Value) -> bool {
    let Some(map) = item.as_object() else {
        return false;
    };
    map.contains_key("content")
        && map.contains_key("role")
        && map
            .keys()
            .all(|k| matches!(k.as_str(), "content" | "role" | "type" | "phase"))
        && map.get("type").is_none_or(|t| t == "message")
        && map
            .get("phase")
            .is_none_or(|p| p.is_null() || p == "commentary" || p == "final_answer")
        && matches!(
            str_field(item, "role"),
            Some("user" | "assistant" | "system" | "developer")
        )
}

fn has_tool_calls(message: &Map<String, Value>) -> bool {
    message
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|c| !c.is_empty())
}

/// The assistant message under construction and the reasoning waiting to be attached to it.
#[derive(Default)]
struct MessageBuilder {
    result: Vec<Value>,
    current: Option<Map<String, Value>>,
    /// Chain of thought to send as `reasoning_content` (DeepSeek).
    pending_reasoning: Option<String>,
    /// Chain of thought to send as the `reasoning` field (servers that returned it that way).
    pending_reasoning_field: Option<String>,
    /// Claude thinking blocks, and whether they are the provider's own complete list (`true`)
    /// or were rebuilt from the normalized reasoning item.
    pending_thinking: Option<(Vec<Value>, bool)>,
}

impl MessageBuilder {
    fn flush(&mut self, clear_pending_reasoning: bool) {
        if let Some(mut message) = self.current.take() {
            // The API rejects an empty `tool_calls` array.
            if !has_tool_calls(&message) {
                message.remove("tool_calls");
                // Stale reasoning must not leak into a later assistant message.
                self.pending_reasoning = None;
                self.pending_reasoning_field = None;
            }
            self.result.push(Value::Object(message));
        }
        if clear_pending_reasoning {
            self.clear_pending();
        }
    }

    fn clear_pending(&mut self) {
        self.pending_reasoning = None;
        self.pending_reasoning_field = None;
        self.pending_thinking = None;
    }

    fn apply_pending_reasoning(&mut self, message: &mut Map<String, Value>) {
        self.apply_pending_thinking(message);
        if let Some(reasoning) = self.pending_reasoning.take() {
            message.insert("reasoning_content".into(), reasoning.into());
        }
        if let Some(reasoning) = self.pending_reasoning_field.take() {
            message.insert("reasoning".into(), reasoning.into());
        }
    }

    /// Python: `apply_pending_thinking_blocks`. The provider's own list goes in as
    /// `thinking_blocks`; a list rebuilt from a stored item leads the message `content`.
    fn apply_pending_thinking(&mut self, message: &mut Map<String, Value>) {
        let Some((blocks, native)) = self.pending_thinking.take().filter(|(b, _)| !b.is_empty())
        else {
            return;
        };
        if native {
            message.insert("thinking_blocks".into(), Value::Array(blocks));
            return;
        }
        let mut parts = match message.remove("content") {
            Some(Value::String(text)) => vec![text_part(&text)],
            Some(Value::Array(parts)) => parts,
            _ => Vec::new(),
        };
        let mut content = blocks;
        content.append(&mut parts);
        message.insert("content".into(), Value::Array(content));
    }

    /// The open assistant message, started when there is none.
    fn assistant(&mut self) -> &mut Map<String, Value> {
        let mut message = self.current.take().unwrap_or_else(|| {
            let mut m = Map::new();
            m.insert("role".into(), "assistant".into());
            m.insert("content".into(), Value::Null);
            m.insert("tool_calls".into(), json!([]));
            m
        });
        self.apply_pending_reasoning(&mut message);
        self.current.insert(message)
    }
}

/// Turn the run's input items into Chat Completions `messages` (Python: `items_to_messages`).
///
/// Consecutive function calls and the assistant text around them share one assistant message,
/// which is what Chat Completions servers expect.
pub fn items_to_chat_messages(
    items: &[Value],
    options: &ChatConvertOptions,
) -> Result<Vec<Value>, ModelError> {
    let mut builder = MessageBuilder::default();
    for item in items {
        let kind = str_field(item, "type");
        let role = str_field(item, "role");
        if is_easy_message(item)
            || (kind == Some("message") && matches!(role, Some("user" | "system" | "developer")))
        {
            builder.flush(true);
            let content = item.get("content").unwrap_or(&Value::Null);
            let message = match role {
                Some("user") => json!({"role": "user", "content": extract_all_content(content)?}),
                Some(role @ ("system" | "developer" | "assistant")) => {
                    json!({"role": role, "content": extract_text_content(content)?})
                }
                other => {
                    return Err(unsupported(format!(
                        "Unexpected role in message: {other:?}"
                    )))
                }
            };
            builder.result.push(message);
        } else if kind == Some("message")
            && role == Some("assistant")
            && item.get("id").is_some()
            && item.get("content").is_some()
        {
            assistant_output_message(&mut builder, item)?;
        } else if kind == Some("function_call") {
            let call_id = str_field(item, "call_id")
                .ok_or_else(|| unsupported(format!("function_call without call_id: {item}")))?;
            let arguments = str_field(item, "arguments")
                .filter(|a| !a.is_empty())
                .unwrap_or("{}");
            let name = str_field(item, "name").unwrap_or_default();
            let mut call = json!({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            });
            // Gemini wants its thought signature back in Google's `extra_content` format.
            let signature = item
                .pointer("/provider_data/thought_signature")
                .filter(|s| s.as_str().is_some_and(|s| !s.is_empty()));
            if let (Some(signature), true) =
                (signature, options.model.to_lowercase().contains("gemini"))
            {
                call["extra_content"] = json!({"google": {"thought_signature": signature}});
            }
            if let Some(calls) = builder
                .assistant()
                .get_mut("tool_calls")
                .and_then(Value::as_array_mut)
            {
                calls.push(call);
            }
        } else if kind == Some("function_call_output") {
            let call_id = str_field(item, "call_id").ok_or_else(|| {
                unsupported(
                    "Unpaired function outputs are supported by Responses but cannot be converted \
                     to Chat Completions tool messages. Use a Responses model to preserve this input.",
                )
            })?;
            builder.flush(true);
            let content = tool_output_content(item.get("output").unwrap_or(&Value::Null))?;
            builder
                .result
                .push(json!({"role": "tool", "tool_call_id": call_id, "content": content}));
        } else if kind == Some("reasoning") {
            builder.clear_pending();
            restore_reasoning(&mut builder, item, options);
            let replay = match &options.replay_reasoning {
                Some(replay) => replay(&options.model, item),
                None => default_should_replay_reasoning(&options.model, item),
            };
            if replay {
                let texts: Vec<&str> = item
                    .get("summary")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|s| str_field(s, "text"))
                    .filter(|t| !t.is_empty())
                    .collect();
                if !texts.is_empty() {
                    builder.pending_reasoning = Some(texts.join("\n"));
                }
            }
        } else if kind == Some("item_reference") {
            return Err(unsupported(format!(
                "Encountered an item_reference, which is not supported: {item}"
            )));
        } else if kind == Some("compaction") {
            return Err(unsupported(
                "Compaction items are not supported for chat completions. \
                 Please use the Responses API to handle compaction.",
            ));
        } else {
            return Err(unsupported(format!(
                "Unhandled item type or structure: {item}"
            )));
        }
    }
    builder.flush(true);
    Ok(builder.result)
}

/// What an earlier reasoning item contributes to the next assistant message besides
/// `reasoning_content`: the `reasoning` field of the server that sent it that way, and Claude
/// thinking blocks (Python: the `maybe_reasoning_message` branch of `items_to_messages`).
fn restore_reasoning(builder: &mut MessageBuilder, item: &Value, options: &ChatConvertOptions) {
    let model = options.model.as_str();
    let provider_data = item.get("provider_data").and_then(Value::as_object);
    let item_model = provider_data
        .and_then(|d| d.get("model"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let reasoning_field = provider_data
        .and_then(|d| d.get(REASONING_FIELD_KEY))
        .and_then(Value::as_str);
    let content: Vec<&Value> = item
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .collect();

    if reasoning_field == Some("reasoning") && !model.is_empty() && model == item_model {
        let texts: Vec<&str> = content
            .iter()
            .filter(|c| str_field(c, "type") == Some("reasoning_text"))
            .filter_map(|c| str_field(c, "text"))
            .filter(|t| !t.is_empty())
            .collect();
        if !texts.is_empty() {
            builder.pending_reasoning_field = Some(texts.join("\n"));
        }
    }

    let lowered = model.to_lowercase();
    let is_claude = lowered.contains("claude") || lowered.contains("anthropic");
    // Only blocks Claude itself produced (or that predate provider tracking) may go back.
    let origin_unknown = !provider_data.is_some_and(|d| d.keys().any(|k| k != "thinking_blocks"));
    if !(is_claude && options.preserve_thinking_blocks && (model == item_model || origin_unknown)) {
        return;
    }
    let complete = provider_data
        .and_then(|d| d.get("thinking_blocks"))
        .and_then(Value::as_array)
        .filter(|blocks| !blocks.is_empty() && blocks.iter().all(Value::is_object));
    if let Some(blocks) = complete {
        builder.pending_reasoning_field = None;
        builder.pending_thinking = Some((blocks.clone(), true));
    } else if !content.is_empty() && reasoning_field != Some("reasoning") {
        let mut signatures: std::collections::VecDeque<&str> = str_field(item, "encrypted_content")
            .filter(|s| !s.is_empty())
            .map(|s| s.split('\n').collect())
            .unwrap_or_default();
        let blocks: Vec<Value> = content
            .iter()
            .filter(|c| str_field(c, "type") == Some("reasoning_text"))
            .map(|c| {
                let mut block =
                    json!({"type": "thinking", "thinking": str_field(c, "text").unwrap_or_default()});
                if let Some(signature) = signatures.pop_front() {
                    block["signature"] = signature.into();
                }
                block
            })
            .collect();
        builder.pending_thinking = Some((blocks, false));
    }
}

/// An assistant `message` item. It joins the open assistant message when that one already holds
/// this turn's tool calls (a streamed turn can list its calls before its text); otherwise it
/// starts a new one.
fn assistant_output_message(builder: &mut MessageBuilder, item: &Value) -> Result<(), ModelError> {
    let mut texts: Vec<&str> = Vec::new();
    let mut refusal: Option<&str> = None;
    for part in item
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match str_field(part, "type") {
            Some("output_text") => texts.push(str_field(part, "text").unwrap_or_default()),
            Some("refusal") => refusal = str_field(part, "refusal"),
            Some("output_audio") => {
                return Err(unsupported(format!(
                    "Only audio IDs are supported for chat completions, but got: {part}"
                )))
            }
            _ => {
                return Err(unsupported(format!(
                    "Unknown content type in ResponseOutputMessage: {part}"
                )))
            }
        }
    }
    let combined = (!texts.is_empty()).then(|| texts.join("\n"));

    let mergeable = builder.current.as_ref().is_some_and(|m| {
        has_tool_calls(m)
            && !m.contains_key("refusal")
            && matches!(
                m.get("content"),
                None | Some(Value::Null) | Some(Value::Array(_))
            )
    });
    if mergeable {
        let mut message = builder.current.take().unwrap_or_default();
        if let Some(text) = combined {
            let content = match message.remove("content") {
                Some(Value::Array(mut parts)) => {
                    parts.push(text_part(&text));
                    Value::Array(parts)
                }
                _ => Value::String(text),
            };
            message.insert("content".into(), content);
        }
        if let Some(refusal) = refusal {
            message.insert("refusal".into(), refusal.into());
        }
        builder.apply_pending_reasoning(&mut message);
        builder.current = Some(message);
    } else {
        // A reasoning item can precede the message and the tool calls of one turn, so keep it.
        builder.flush(false);
        let mut message = Map::new();
        message.insert("role".into(), "assistant".into());
        if let Some(refusal) = refusal {
            message.insert("refusal".into(), refusal.into());
        }
        if let Some(text) = combined {
            message.insert("content".into(), text.into());
        }
        message.insert("tool_calls".into(), json!([]));
        builder.apply_pending_reasoning(&mut message);
        builder.current = Some(message);
    }
    Ok(())
}

/// `content` of a `tool` message: text only, since Chat Completions has no non-text tool results.
fn tool_output_content(output: &Value) -> Result<Value, ModelError> {
    match output {
        Value::String(_) => Ok(output.clone()),
        Value::Array(_) => {
            let parts = match extract_all_content(output)? {
                Value::Array(parts) => parts,
                _ => Vec::new(),
            };
            let text: Vec<Value> = parts
                .into_iter()
                .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
                .collect();
            if text.is_empty() {
                ::tracing::warn!(
                    "Chat Completions tool outputs cannot be empty or contain only non-text content; \
                     replacing the tool output with a placeholder"
                );
                Ok(Value::String(OMITTED_TOOL_OUTPUT_PLACEHOLDER.into()))
            } else {
                Ok(Value::Array(text))
            }
        }
        // A bare JSON value: send it as the text it would print as.
        Value::Null => Ok(Value::String(String::new())),
        other => Ok(Value::String(other.to_string())),
    }
}
