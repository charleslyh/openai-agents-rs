//! Nested handoff history (Python: `agents.handoffs.history`).
//!
//! With `nest_handoff_history` the next agent does not see the raw transcript. The earlier
//! turns are folded into assistant messages that hold a numbered summary between
//! `<CONVERSATION HISTORY>` markers, while user-visible messages of the handoff turn stay as
//! real items in their original position.

use std::sync::Mutex;

use serde_json::{json, Map, Value};

use super::HandoffInputData;
use crate::pyjson;

const DEFAULT_START: &str = "<CONVERSATION HISTORY>";
const DEFAULT_END: &str = "</CONVERSATION HISTORY>";
const PREAMBLE: &str =
    "For context, here is the conversation so far between the user and the previous agent:";
const LEGACY_PREAMBLE: &str = "For context, here is the conversation so far:";

/// Item types that the summary already covers, so they are not forwarded verbatim.
const SUMMARY_ONLY_TYPES: [&str; 3] = ["function_call", "function_call_output", "reasoning"];

/// SDK-only keys removed before an item is summarized or replayed.
const INTERNAL_KEYS: [&str; 3] = ["_agents_tool_description", "_agents_tool_title", "created_by"];

static WRAPPERS: Mutex<Option<(String, String)>> = Mutex::new(None);

/// Maps the flattened transcript to the history the next agent receives
/// (Python: `HandoffHistoryMapper`). Synchronous, like Python's.
pub type HandoffHistoryMapper = std::sync::Arc<dyn Fn(Vec<Value>) -> Vec<Value> + Send + Sync>;

/// Override the markers around the generated summary; `None` leaves a side unchanged
/// (Python: `set_conversation_history_wrappers`). Process-global, like Python's.
pub fn set_conversation_history_wrappers(start: Option<&str>, end: Option<&str>) {
    let (cur_start, cur_end) = get_conversation_history_wrappers();
    *WRAPPERS.lock().expect("wrappers") = Some((
        start.map(str::to_string).unwrap_or(cur_start),
        end.map(str::to_string).unwrap_or(cur_end),
    ));
}

/// Restore the default `<CONVERSATION HISTORY>` markers.
pub fn reset_conversation_history_wrappers() {
    *WRAPPERS.lock().expect("wrappers") = None;
}

/// The current `(start, end)` markers.
pub fn get_conversation_history_wrappers() -> (String, String) {
    WRAPPERS
        .lock()
        .expect("wrappers")
        .clone()
        .unwrap_or_else(|| (DEFAULT_START.to_string(), DEFAULT_END.to_string()))
}

/// Fold the earlier transcript into summary messages (Python: `nest_handoff_history`).
///
/// The returned data carries the exact model input in `input_history`; `pre_handoff_items` and
/// `new_items` are empty. Nesting does not redact anything: tool arguments and outputs remain in
/// the summary text, so sanitize the input first when that matters.
pub fn nest_handoff_history(
    data: HandoffInputData,
    history_mapper: Option<&HandoffHistoryMapper>,
) -> HandoffInputData {
    let flattened: Vec<Value> = flatten_nested_history(&data.input_history)
        .into_iter()
        .map(|item| strip_internal_metadata(&item))
        .collect();

    let pre = data
        .pre_handoff_items
        .iter()
        .map(|item| (item.clone(), should_forward_pre_item(item)));
    let new = data
        .new_items
        .iter()
        .map(|item| (item.clone(), should_forward_new_item(item)));
    let items: Vec<(Value, bool)> = pre.chain(new).collect();

    let history = match history_mapper {
        Some(mapper) => {
            let mut transcript = flattened;
            transcript.extend(items.into_iter().map(|(item, _)| item));
            mapper(transcript)
        }
        None => build_ordered_default_history(flattened, items),
    };

    HandoffInputData {
        input_history: history,
        pre_handoff_items: Vec::new(),
        new_items: Vec::new(),
        run_context: data.run_context,
    }
}

/// One assistant message summarizing the transcript
/// (Python: `default_handoff_history_mapper`).
pub fn default_handoff_history_mapper(transcript: Vec<Value>) -> Vec<Value> {
    vec![build_summary_message(&transcript)]
}

fn build_ordered_default_history(flattened: Vec<Value>, items: Vec<(Value, bool)>) -> Vec<Value> {
    let mut history: Vec<Value> = Vec::new();
    let mut pending = flattened;
    for (item, forward_verbatim) in items {
        if !forward_verbatim {
            pending.push(item);
            continue;
        }
        if !pending.is_empty() || history.is_empty() {
            history.extend(default_handoff_history_mapper(std::mem::take(&mut pending)));
        }
        history.push(item);
    }
    if !pending.is_empty() || history.is_empty() {
        history.extend(default_handoff_history_mapper(pending));
    }
    history
}

fn build_summary_message(transcript: &[Value]) -> Value {
    let lines: Vec<String> = if transcript.is_empty() {
        vec!["(no previous turns recorded)".to_string()]
    } else {
        transcript
            .iter()
            .enumerate()
            .map(|(i, item)| format!("{}. {}", i + 1, format_transcript_item(item)))
            .collect()
    };
    let (start, end) = get_conversation_history_wrappers();
    let mut content = vec![PREAMBLE.to_string(), start];
    content.extend(lines);
    content.push(end);
    json!({"role": "assistant", "content": content.join("\n")})
}

fn strip_internal_metadata(item: &Value) -> Value {
    let mut item = item.clone();
    if let Some(map) = item.as_object_mut() {
        for key in INTERNAL_KEYS {
            map.remove(key);
        }
    }
    item
}

fn format_transcript_item(item: &Value) -> String {
    let item = strip_internal_metadata(item);
    if item.get("role").is_some_and(Value::is_string) {
        match item.get("content") {
            None | Some(Value::Null) => return format_legacy(&item),
            Some(Value::String(text)) if !text.contains('\n') && !text.contains('\r') => {
                return format_legacy(&item)
            }
            _ => {}
        }
    }
    format_json(&item)
}

fn format_json(item: &Value) -> String {
    let mut payload = pyjson::python_field_order(item);
    if let Some(map) = payload.as_object_mut() {
        map.remove("provider_data");
    }
    pyjson::dumps(&payload)
}

fn format_legacy(item: &Value) -> String {
    if let Some(role) = item.get("role").and_then(Value::as_str) {
        let prefix = match item.get("name").and_then(Value::as_str) {
            Some(name) if !name.is_empty() => format!("{role} ({name})"),
            _ => role.to_string(),
        };
        return format!("{prefix}: {}", stringify_content(item.get("content")));
    }
    format_json(item)
}

fn stringify_content(content: Option<&Value>) -> String {
    match content {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(other) => pyjson::dumps(other),
    }
}

// ---- flattening of earlier summaries -------------------------------------------------------

fn flatten_nested_history(items: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for item in items {
        match extract_nested_transcript(item) {
            Some(parsed) => out.extend(parsed),
            None => out.push(item.clone()),
        }
    }
    out
}

fn extract_nested_transcript(item: &Value) -> Option<Vec<Value>> {
    if item.get("role").and_then(Value::as_str) != Some("assistant") {
        return None;
    }
    let content = item.get("content")?.as_str()?;
    let (start, end) = get_conversation_history_wrappers();
    let (preamble, wrapped) = content.split_once('\n')?;
    if preamble != PREAMBLE && preamble != LEGACY_PREAMBLE {
        return None;
    }
    let start_wrapper = format!("{start}\n");
    let end_wrapper = format!("\n{end}");
    let body = wrapped
        .strip_prefix(start_wrapper.as_str())?
        .strip_suffix(end_wrapper.as_str())?;
    Some(
        split_summary_records(body)
            .iter()
            .filter_map(|line| parse_summary_line(line))
            .collect(),
    )
}

fn starts_numbered_record(line: &str) -> bool {
    let stripped = line.trim_start();
    match stripped.find('.') {
        Some(dot) => dot > 0 && stripped[..dot].chars().all(|c| c.is_ascii_digit()),
        None => false,
    }
}

fn split_summary_records(body: &str) -> Vec<String> {
    let mut records = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut current_is_numbered = false;
    for raw in body.lines() {
        if raw.trim().is_empty() {
            continue;
        }
        let numbered = starts_numbered_record(raw);
        if current.is_empty() {
            current = vec![raw.trim().to_string()];
            current_is_numbered = numbered;
            continue;
        }
        if numbered || !current_is_numbered {
            records.push(current.join("\n"));
            current = vec![raw.trim().to_string()];
            current_is_numbered = numbered;
            continue;
        }
        current.push(raw.trim_end().to_string());
    }
    if !current.is_empty() {
        records.push(current.join("\n"));
    }
    records
}

fn strip_line_number(line: &str) -> &str {
    match line.find('.') {
        Some(dot) if dot > 0 && line[..dot].chars().all(|c| c.is_ascii_digit()) => {
            line[dot + 1..].trim_start()
        }
        _ => line,
    }
}

fn parse_summary_line(line: &str) -> Option<Value> {
    let stripped = line.trim();
    if stripped.is_empty() {
        return None;
    }
    let stripped = strip_line_number(stripped);
    if let Some(parsed) = parse_summary_json(stripped) {
        return Some(parsed);
    }
    let Some((role_part, remainder)) = stripped.split_once(':') else {
        if !is_bare_role_record(stripped) {
            return None;
        }
        let (role, name) = split_role_and_name(stripped);
        let mut recovered = Map::new();
        recovered.insert("role".into(), Value::String(role));
        if let Some(name) = name {
            recovered.insert("name".into(), Value::String(name));
        }
        recovered.insert("content".into(), Value::String(String::new()));
        return Some(Value::Object(recovered));
    };
    let role_text = role_part.trim();
    if role_text.is_empty() {
        return None;
    }
    let (role, name) = split_role_and_name(role_text);
    let content = remainder.trim();
    if !content.is_empty() {
        if let Some(typed) = parse_legacy_typed_item(&role, content) {
            return Some(typed);
        }
    }
    let mut item = Map::new();
    item.insert("role".into(), Value::String(role));
    if let Some(name) = name {
        item.insert("name".into(), Value::String(name));
    }
    item.insert("content".into(), Value::String(content.to_string()));
    Some(Value::Object(item))
}

fn parse_summary_json(text: &str) -> Option<Value> {
    let mut parsed: Value = serde_json::from_str(text).ok()?;
    let map = parsed.as_object_mut()?;
    map.remove("provider_data");
    Some(strip_internal_metadata(&parsed))
}

fn parse_legacy_typed_item(item_type: &str, content: &str) -> Option<Value> {
    if matches!(item_type, "assistant" | "user" | "system" | "developer") {
        return None;
    }
    let mut parsed: Value = serde_json::from_str(content).ok()?;
    let map = parsed.as_object_mut()?;
    map.remove("provider_data");
    map.insert("type".into(), Value::String(item_type.to_string()));
    Some(strip_internal_metadata(&parsed))
}

fn is_bare_role_record(text: &str) -> bool {
    let mut candidate = text;
    if candidate.ends_with(')') {
        if let Some(open) = candidate.rfind('(') {
            candidate = candidate[..open].trim();
        }
    }
    matches!(candidate, "user" | "assistant" | "system" | "developer")
}

fn split_role_and_name(role_text: &str) -> (String, Option<String>) {
    if role_text.ends_with(')') {
        if let Some(open) = role_text.rfind('(') {
            let name = role_text[open + 1..role_text.len() - 1].trim();
            let role = role_text[..open].trim();
            if !name.is_empty() {
                let role = if role.is_empty() { "developer" } else { role };
                return (role.to_string(), Some(name.to_string()));
            }
        }
    }
    let role = if role_text.is_empty() {
        "developer"
    } else {
        role_text
    };
    (role.to_string(), None)
}

// ---- which items stay verbatim -------------------------------------------------------------

fn is_programmatic(item: &Value) -> bool {
    matches!(
        item.get("type").and_then(Value::as_str),
        Some("program" | "program_output")
    ) || item
        .get("caller")
        .and_then(|c| c.get("type"))
        .and_then(Value::as_str)
        == Some("program")
}

fn is_summary_only(item: &Value) -> bool {
    item.get("type")
        .and_then(Value::as_str)
        .is_some_and(|t| SUMMARY_ONLY_TYPES.contains(&t))
}

/// Earlier turns are summarized, except non-assistant items that are not tool bookkeeping.
fn should_forward_pre_item(item: &Value) -> bool {
    if is_programmatic(item) {
        return false;
    }
    if item.get("role").and_then(Value::as_str) == Some("assistant") {
        return false;
    }
    !is_summary_only(item)
}

/// The handoff turn keeps every item with a role; tool bookkeeping is summarized.
fn should_forward_new_item(item: &Value) -> bool {
    if is_programmatic(item) {
        return false;
    }
    if item
        .get("role")
        .and_then(Value::as_str)
        .is_some_and(|r| !r.is_empty())
    {
        return true;
    }
    !is_summary_only(item)
}
