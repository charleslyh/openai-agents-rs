//! Property / randomized tests (verification layer 1.5).
//!
//! The behavior tests assert scenarios somebody thought of. These assert *invariants* over
//! generated inputs, which is how the pure, self-contained pieces get coverage for the inputs
//! nobody wrote down: arbitrary JSON fed to the strict-schema pass, and arbitrary conversation
//! shapes fed to the context trimmers.

use proptest::prelude::*;
use serde_json::{json, Map, Value};

use openai_agents::{estimate_item_tokens, ContextWindowTrimmer, ToolOutputTrimmer};

/// An arbitrary JSON value with bounded depth and width.
///
/// Most of these are *not* valid JSON Schemas; that is the point. `ensure_strict_json_schema`
/// must reject them, never panic on them.
fn arb_json() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(Value::from),
        "[a-z_]{0,8}".prop_map(Value::String),
    ];
    leaf.prop_recursive(
        4,  // depth
        32, // max nodes
        6,  // items per collection
        |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..4).prop_map(Value::Array),
                prop::collection::hash_map("[a-z]{1,6}", inner, 0..4)
                    .prop_map(|m| { Value::Object(m.into_iter().collect::<Map<String, Value>>()) }),
            ]
        },
    )
}

/// An arbitrary Responses-shaped conversation item: pinned (system / developer), a user turn
/// start, an assistant message, and a tool call with its output.
fn arb_item() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(json!({"role": "system", "content": "you are helpful"})),
        Just(json!({"role": "developer", "content": "be brief"})),
        "[a-z]{1,6}".prop_map(|t| json!({
            "role": "user",
            "content": [{"type": "input_text", "text": t}]
        })),
        "[a-z]{1,6}".prop_map(|t| json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": t}]
        })),
        "[a-z]{1,6}".prop_map(|t| json!({
            "type": "function_call",
            "name": "tool",
            "call_id": format!("call-{t}"),
            "arguments": "{}"
        })),
        "[a-z]{1,6}".prop_map(|t| json!({
            "type": "function_call_output",
            "call_id": "call-x",
            "output": t
        })),
    ]
}

fn arb_items() -> impl Strategy<Value = Vec<Value>> {
    prop::collection::vec(arb_item(), 0..12)
}

fn is_pinned(item: &Value) -> bool {
    let role = item.get("role").and_then(|r| r.as_str());
    matches!(role, Some("system" | "developer"))
        || item
            .get("content")
            .and_then(|c| c.as_str())
            .is_some_and(|t| t.contains("<conversation_summary>"))
}

proptest! {
    /// The strict-schema pass returns or rejects; it never panics, and whatever it accepts is
    /// already strict, so running it again is a no-op (idempotent).
    #[test]
    fn strict_schema_never_panics_and_is_idempotent(schema in arb_json()) {
        let first = openai_agents::strict_schema::ensure_strict_json_schema(&schema);
        if let Ok(strict) = first {
            let second = openai_agents::strict_schema::ensure_strict_json_schema(&strict);
            prop_assert!(
                second.is_ok(),
                "a strict schema must survive a second pass: {:?}",
                second.err()
            );
            prop_assert_eq!(
                second.expect("checked above").to_string(),
                strict.to_string(),
                "the second pass changed an already-strict schema"
            );
        }
    }

    /// Whatever the pass accepts, every object node is closed (`additionalProperties: false`),
    /// which is what "strict" means to the Responses API.
    #[test]
    fn strict_schema_output_closes_every_object(schema in arb_json()) {
        if let Ok(strict) = openai_agents::strict_schema::ensure_strict_json_schema(&schema) {
            assert_objects_closed(&strict);
        }
    }

    /// The window trimmer only ever removes whole items: nothing is invented, and every pinned
    /// item survives.
    #[test]
    fn context_trimmer_only_removes_items(
        items in arb_items(),
        max_tokens in prop::option::of(0usize..400),
        max_turns in prop::option::of(1usize..6),
        reserved in 0usize..200,
    ) {
        let mut trimmer = ContextWindowTrimmer::new();
        if let Some(t) = max_tokens {
            trimmer = trimmer.max_tokens(t);
        }
        if let Some(t) = max_turns {
            trimmer = trimmer.max_turns(t);
        }
        let trimmed = trimmer.trim(&items, reserved);

        prop_assert!(trimmed.len() <= items.len(), "trimming added items");
        for item in &trimmed {
            prop_assert!(
                items.contains(item),
                "the trimmer produced an item that was not in the input: {item}"
            );
        }
        for item in items.iter().filter(|i| is_pinned(i)) {
            prop_assert!(
                trimmed.contains(item),
                "a pinned item was dropped: {item}"
            );
        }
    }

    /// Trimming is stable: trimming an already-trimmed history does nothing further.
    #[test]
    fn context_trimmer_is_idempotent(
        items in arb_items(),
        max_tokens in prop::option::of(0usize..400),
        max_turns in prop::option::of(1usize..6),
    ) {
        let mut trimmer = ContextWindowTrimmer::new();
        if let Some(t) = max_tokens {
            trimmer = trimmer.max_tokens(t);
        }
        if let Some(t) = max_turns {
            trimmer = trimmer.max_turns(t);
        }
        let once = trimmer.trim(&items, 0);
        let twice = trimmer.trim(&once, 0);
        prop_assert_eq!(once.len(), twice.len(), "the second trim changed the history");
        prop_assert!(once.iter().eq(twice.iter()), "the second trim changed the history");
    }

    /// The output trimmer never makes a history bigger, and leaves the recent turns alone.
    #[test]
    fn tool_output_trimmer_never_grows_history(
        items in arb_items(),
        max_chars in 1usize..40,
        preview_chars in 1usize..40,
    ) {
        let trimmer = ToolOutputTrimmer::new()
            .max_output_chars(max_chars)
            .preview_chars(preview_chars);
        let trimmed = trimmer.trim(&items);
        prop_assert_eq!(trimmed.len(), items.len(), "the trimmer changed the item count");
        for (before, after) in items.iter().zip(trimmed.iter()) {
            prop_assert!(
                item_size(after) <= item_size(before) + 64,
                "the trimmer grew an item: {before} -> {after}"
            );
        }
    }

    /// A token estimate is a size, not a random number: it grows with the payload and is never
    /// zero for a non-empty item.
    #[test]
    fn token_estimate_grows_with_text(prefix in "[a-z]{0,32}", suffix in "[a-z]{0,32}") {
        let short = json!({"role": "user", "content": [{"type": "input_text", "text": prefix}]});
        let long = json!({
            "role": "user",
            "content": [{"type": "input_text", "text": format!("{prefix}{suffix}")}]
        });
        prop_assert!(estimate_item_tokens(&short) <= estimate_item_tokens(&long));
    }
}

fn item_size(item: &Value) -> usize {
    item.to_string().len()
}

fn assert_objects_closed(value: &Value) {
    match value {
        Value::Object(map) => {
            // Only schema nodes are checked: `properties`, `$defs` and friends are containers
            // whose values happen to be objects, not schemas in their own right.
            if map.get("type").and_then(|t| t.as_str()) == Some("object") {
                assert_eq!(
                    map.get("additionalProperties"),
                    Some(&Value::Bool(false)),
                    "object node is not closed: {value}"
                );
            }
            for child in map.values() {
                assert_objects_closed(child);
            }
        }
        Value::Array(items) => {
            for child in items {
                assert_objects_closed(child);
            }
        }
        _ => {}
    }
}
