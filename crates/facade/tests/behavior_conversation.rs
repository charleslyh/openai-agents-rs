//! A run's content reaches whatever is attached to the conversation hub, next to its spans.
#![cfg(feature = "observability")]

use std::sync::{Arc, Mutex};

use opai_agents::observability::{ConversationEvent, ConversationHub};
use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{Agent, FunctionTool, RunOptions, Runner};

fn lookup_tool() -> FunctionTool {
    FunctionTool::new(
        "lookup",
        "look a city up",
        serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false
        }),
        |_ctx, _args| async move { Ok(serde_json::json!("1200 orders")) },
    )
}

fn scripted_agent() -> Agent {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(
            "lookup",
            r#"{"city":"Berlin"}"#,
            "c1",
        )),
        ModelStep::from(ItemHelpers::text_message("Berlin: 1200 orders")),
    ]));
    Agent::new("Ops")
        .instructions("Use lookup.")
        .model(model)
        .tools(vec![lookup_tool()])
}

fn attach(hub: &ConversationHub) -> Arc<Mutex<Vec<ConversationEvent>>> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    hub.attach(Arc::new(move |e: &ConversationEvent| {
        sink.lock().unwrap().push(e.clone());
    }));
    seen
}

#[tokio::test]
async fn prompts_replies_and_tool_calls_reach_an_attached_sink() {
    let hub = ConversationHub::new();
    let seen = attach(&hub);

    let options = RunOptions::new().hooks(hub.recorder("run-a"));
    Runner::run(&scripted_agent(), "metrics for Berlin?", options)
        .await
        .expect("run");

    let events = seen.lock().unwrap().clone();
    let kinds: Vec<_> = events.iter().map(|e| e.kind).collect();
    assert_eq!(
        kinds,
        [
            "agent_start",
            "llm_start",
            "llm_end",
            "tool_start",
            "tool_end",
            "llm_start",
            "llm_end",
            "agent_end"
        ],
        "{events:#?}"
    );
    assert!(events
        .iter()
        .all(|e| e.run_id == "run-a" && e.agent == "Ops"));

    let first_prompt = &events[1].data;
    assert_eq!(first_prompt["system_prompt"], "Use lookup.");
    assert!(first_prompt["input"]
        .to_string()
        .contains("metrics for Berlin?"));

    assert_eq!(events[3].data["tool"], "lookup");
    assert_eq!(events[3].data["arguments"]["city"], "Berlin");
    assert_eq!(events[4].data["result"], "1200 orders");
    assert!(events[6].data["output"]
        .to_string()
        .contains("Berlin: 1200 orders"));
}

#[tokio::test]
async fn metadata_only_runs_leave_no_text_behind() {
    let hub = ConversationHub::new().metadata_only();
    let seen = attach(&hub);

    let options = RunOptions::new().hooks(hub.recorder("run-b"));
    Runner::run(&scripted_agent(), "metrics for Berlin?", options)
        .await
        .expect("run");

    let wire: Vec<_> = seen.lock().unwrap().iter().map(|e| e.to_json()).collect();
    let wire = serde_json::to_string(&wire).unwrap();
    for secret in ["Berlin", "1200 orders", "Use lookup."] {
        assert!(!wire.contains(secret), "{secret} leaked: {wire}");
    }
    assert!(wire.contains("lookup"), "tool names are metadata");
}
