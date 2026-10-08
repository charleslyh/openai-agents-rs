//! Cleaning of the model input built from a session (Python: `drop_orphan_function_calls`,
//! `deduplicate_input_items_preferring_latest` in `run_internal/items.py`).
//!
//! Stored history can hold a tool call whose output was never saved (a crash between the two, a
//! rejected approval, a `limit` that cut the pair in half) or the same item twice. Providers
//! reject both, so the model input is cleaned before use. Only function-call items are handled:
//! the other item kinds Python lists are OpenAI-hosted tools this SDK does not have.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::model::wire_events::FAKE_RESPONSES_ID;

const CALL: &str = "function_call";
const OUTPUT: &str = "function_call_output";

fn item_type(item: &Value) -> Option<&str> {
    item.get("type").and_then(Value::as_str)
}

fn call_id(item: &Value) -> Option<&str> {
    item.get("call_id").and_then(Value::as_str)
}

/// Drop function calls without an output, and the reasoning items that led up to them.
///
/// `prunable[i]` says whether item `i` may be dropped: only stored history is, since a caller who
/// passes a call on purpose gets it as given. With `prune_outputs`, history outputs whose call is
/// no longer present are dropped too (a `limit` can cut a pair in half).
pub(crate) fn drop_orphan_function_calls(
    items: Vec<Value>,
    prunable: &[bool],
    prune_outputs: bool,
) -> Vec<Value> {
    let completed: HashSet<&str> = items
        .iter()
        .filter(|i| item_type(i) == Some(OUTPUT))
        .filter_map(call_id)
        .collect();

    let mut dropped: HashSet<usize> = HashSet::new();
    for (index, item) in items.iter().enumerate() {
        if !prunable.get(index).copied().unwrap_or(false) || item_type(item) != Some(CALL) {
            continue;
        }
        if !call_id(item).is_some_and(|id| completed.contains(id)) {
            dropped.insert(index);
        }
    }
    let triggers = dropped.clone();

    if prune_outputs {
        let available: HashSet<&str> = items
            .iter()
            .enumerate()
            .filter(|(i, item)| !dropped.contains(i) && item_type(item) == Some(CALL))
            .filter_map(|(_, item)| call_id(item))
            .collect();
        for (index, item) in items.iter().enumerate() {
            if prunable.get(index).copied().unwrap_or(false)
                && item_type(item) == Some(OUTPUT)
                && call_id(item).is_some_and(|id| !available.contains(id))
            {
                dropped.insert(index);
            }
        }
    }
    if dropped.is_empty() {
        return items;
    }

    // A reasoning item belongs to the next item the model produced; when that one was dropped
    // the reasoning item dangles and the Responses API rejects it.
    let mut dangling: HashSet<usize> = HashSet::new();
    for index in (0..items.len()).rev() {
        if dropped.contains(&index) || item_type(&items[index]) != Some("reasoning") {
            continue;
        }
        for next in index + 1..items.len() {
            if dangling.contains(&next) || item_type(&items[next]) == Some("reasoning") {
                continue;
            }
            if triggers.contains(&next) {
                dangling.insert(index);
            }
            break;
        }
    }
    items
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !dropped.contains(i) && !dangling.contains(i))
        .map(|(_, item)| item)
        .collect()
}

/// Stable identity of an item that carries an id, `None` for messages and anonymous items.
fn dedupe_key(item: &Value) -> Option<String> {
    let role = item.get("role");
    let kind = item_type(item).or_else(|| role.and_then(Value::as_str));
    if role.is_some() || kind == Some("message") {
        return None;
    }
    let kind = kind.unwrap_or("");
    if let Some(id) = call_id(item) {
        if kind == CALL || kind == OUTPUT {
            return Some(format!("call_id:{kind}:{id}"));
        }
    }
    match item.get("id").and_then(Value::as_str) {
        Some(id) if id != FAKE_RESPONSES_ID => Some(format!("id:{kind}:{id}")),
        _ => call_id(item).map(|id| format!("call_id:{kind}:{id}")),
    }
}

/// Keep one item per identity, with the latest content.
///
/// Calls and reasoning stay where they first appeared, so they cannot move behind their
/// outputs; every other identified item moves to its latest position, so a replacement does not
/// jump earlier in the conversation.
pub(crate) fn deduplicate_input_items_preferring_latest(items: Vec<Value>) -> Vec<Value> {
    let mut latest: HashMap<String, usize> = HashMap::new();
    let mut anchor: HashMap<String, usize> = HashMap::new();
    for (index, item) in items.iter().enumerate() {
        let Some(key) = dedupe_key(item) else { continue };
        latest.insert(key.clone(), index);
        let anchors_first = matches!(item_type(item), Some(CALL | "reasoning"));
        if !anchor.contains_key(&key) || !anchors_first {
            anchor.insert(key, index);
        }
    }
    let values: Vec<Value> = items.clone();
    items
        .into_iter()
        .enumerate()
        .filter_map(|(index, item)| match dedupe_key(&item) {
            None => Some(item),
            Some(key) if anchor[&key] == index => Some(values[latest[&key]].clone()),
            Some(_) => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn call(id: &str) -> Value {
        json!({"type": "function_call", "call_id": id, "name": "t", "arguments": "{}"})
    }
    fn out(id: &str) -> Value {
        json!({"type": "function_call_output", "call_id": id, "output": "x"})
    }
    fn user(t: &str) -> Value {
        json!({"role": "user", "content": t})
    }
    fn reasoning(id: &str) -> Value {
        json!({"type": "reasoning", "id": id, "summary": []})
    }

    #[test]
    fn orphan_calls_and_their_reasoning_are_dropped_from_history_only() {
        let items = vec![user("a"), reasoning("r1"), call("c1"), reasoning("r2"), call("c2"), out("c2"), call("c3")];
        let history = [true, true, true, true, true, true, false];
        let kept = drop_orphan_function_calls(items.clone(), &history, false);
        assert_eq!(kept, vec![user("a"), reasoning("r2"), call("c2"), out("c2"), call("c3")]);
        // Nothing is prunable: everything stays.
        assert_eq!(drop_orphan_function_calls(items.clone(), &[false; 7], false), items);
    }

    #[test]
    fn outputs_whose_call_was_cut_off_are_dropped_when_asked() {
        let items = vec![out("c1"), user("a"), call("c2"), out("c2")];
        let all = [true; 4];
        assert_eq!(drop_orphan_function_calls(items.clone(), &all, false), items);
        assert_eq!(
            drop_orphan_function_calls(items, &all, true),
            vec![user("a"), call("c2"), out("c2")]
        );
    }

    #[test]
    fn duplicates_keep_one_item_with_the_latest_content() {
        let mut newer = out("c1");
        newer["output"] = json!("newer");
        let items = vec![user("hi"), call("c1"), out("c1"), user("hi"), call("c1"), newer.clone()];
        assert_eq!(
            deduplicate_input_items_preferring_latest(items),
            vec![user("hi"), call("c1"), user("hi"), newer],
            "messages are never deduplicated; the call keeps its first slot, the output its last"
        );
        let two_reasoning = vec![reasoning("r1"), call("c1"), reasoning("r1"), out("c1")];
        assert_eq!(
            deduplicate_input_items_preferring_latest(two_reasoning),
            vec![reasoning("r1"), call("c1"), out("c1")]
        );
    }
}
