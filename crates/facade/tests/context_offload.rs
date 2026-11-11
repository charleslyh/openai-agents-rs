//! `OffloadToolOutputs` with the read tool: an old tool output leaves the history and the model
//! can still get it back.

mod common;

use std::sync::Arc;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{
    offload_capability, reference_for, Agent, BlobStore, CompactingSession, InMemoryBlobStore,
    InMemorySession, OffloadToolOutputs, RunOptions, Runner, Session,
};
use serde_json::{json, Value};

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

#[tokio::test]
async fn an_offloaded_output_can_be_read_back_through_the_capability() {
    let secret = format!(
        "{} the answer is 42 {}",
        "x".repeat(3_000),
        "y".repeat(3_000)
    );
    let blobs = InMemoryBlobStore::shared();

    // A stored history in which an old tool output is bulky.
    let store = InMemorySession::shared("chat-1");
    store
        .add_items(vec![
            user("look it up"),
            json!({"type": "function_call", "call_id": "c1", "name": "search", "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "c1", "output": secret}),
            user("and another"),
            user("and one more"),
        ])
        .await
        .unwrap();

    let session = CompactingSession::with_reducer(
        store.clone(),
        Arc::new(OffloadToolOutputs::new(blobs.clone(), "chat-1").recent_exchanges(2)),
    )
    .trigger_items(1)
    .shared();
    // A write after the old items are in place triggers the reducer.
    session.add_items(vec![user("one last")]).await.unwrap();

    let stored = store.get_items(None).await.unwrap();
    let note = stored[2]["output"].as_str().unwrap();
    assert!(note.starts_with("[Offloaded:"), "{note}");
    assert!(!stored
        .iter()
        .any(|i| i.to_string().contains("the answer is 42")));
    let key = reference_for(&secret);
    assert!(note.contains(&key));
    assert_eq!(
        blobs.get("chat-1", &key).await.unwrap().as_deref(),
        Some(secret.as_str())
    );

    // The model reads the middle of the original back.
    let args = json!({"ref": key, "offset": 2_990, "length": 40}).to_string();
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(
            "read_offloaded",
            &args,
            "r1",
        )),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("a")
        .model(model.clone())
        .capability(offload_capability(blobs.clone(), "chat-1"));
    Runner::run(
        &agent,
        "what was the answer?",
        RunOptions::new().session(session),
    )
    .await
    .expect("run");

    let second = model.calls()[1].input.to_string();
    assert!(second.contains("the answer is 42"), "{second}");
    let first = &model.calls()[0];
    assert!(first.tool_names.iter().any(|n| n == "read_offloaded"));
    assert!(first
        .system_instructions
        .as_deref()
        .unwrap_or_default()
        .contains("[Offloaded:"));
}

#[tokio::test]
async fn the_read_tool_of_one_namespace_cannot_read_another() {
    let blobs = InMemoryBlobStore::shared();
    let text = "private ".repeat(500);
    let key = reference_for(&text);
    blobs.put("alice", &key, &text).await.unwrap();

    let args = json!({"ref": key}).to_string();
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(
            "read_offloaded",
            &args,
            "r1",
        )),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("a")
        .model(model.clone())
        .capability(offload_capability(blobs, "bob"));
    Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    let second = model.calls()[1].input.to_string();
    assert!(!second.contains("private"));
    assert!(second.contains("No stored output"));
}
