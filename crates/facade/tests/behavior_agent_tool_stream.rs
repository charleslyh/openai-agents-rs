//! `AsToolConfig::on_stream`: the nested run's events reach a handler, and the tool behaves the
//! same as without one (Python: `Agent.as_tool(on_stream=...)`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{
    handoff, Agent, AgentToolStreamEvent, AsToolConfig, FunctionTool, RunItem, RunOptions,
    RunState, Runner, StreamEvent,
};

type Seen = Arc<Mutex<Vec<AgentToolStreamEvent>>>;

fn recording(seen: &Seen) -> AsToolConfig {
    let seen = Arc::clone(seen);
    AsToolConfig {
        name: Some("call_helper".into()),
        description: Some("Call helper".into()),
        ..Default::default()
    }
    .on_stream(move |event| {
        let seen = Arc::clone(&seen);
        async move { seen.lock().unwrap().push(event) }
    })
}

fn call_helper(id: &str) -> ModelStep {
    ModelStep::from(ItemHelpers::function_tool_call(
        "call_helper",
        r#"{"input":"hi"}"#,
        id,
    ))
}

fn deltas(seen: &Seen) -> String {
    seen.lock()
        .unwrap()
        .iter()
        .filter_map(|e| match &e.event {
            StreamEvent::RawResponse { data } if data["type"] == "response.output_text.delta" => {
                data["delta"].as_str().map(str::to_string)
            }
            _ => None,
        })
        .collect()
}

/// The handler sees the nested run's events in order, labelled with the tool call; the outer run
/// ends as it would without a handler.
#[tokio::test]
async fn nested_events_reach_the_handler_and_the_result_is_unchanged() {
    let seen: Seen = Arc::default();
    let helper = Agent::new("helper").model(Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("nested answer"),
    )])));
    let outer = Agent::new("boss")
        .model(Arc::new(ScriptedModel::new([
            call_helper("outer-1"),
            ModelStep::from(ItemHelpers::text_message("outer done")),
        ])))
        .tools(vec![helper.as_tool(recording(&seen))]);

    let result = Runner::run(&outer, "go", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("outer done"));

    // The tool output the outer model saw is the nested final output.
    assert!(result.new_items.iter().any(|item| matches!(
        item,
        RunItem::ToolCallOutput(o) if o.output == serde_json::json!("nested answer")
    )));

    let events = seen.lock().unwrap();
    assert!(matches!(
        events.first().map(|e| &e.event),
        Some(StreamEvent::AgentUpdated { agent_name }) if agent_name == "helper"
    ));
    assert!(events
        .iter()
        .all(|e| e.tool_call_id == "outer-1" && e.tool_name == "call_helper"));
    assert!(events.iter().all(|e| e.agent_name == "helper"));
    drop(events);
    assert_eq!(deltas(&seen), "nested answer");
}

/// A handoff inside the nested run changes `agent_name` from the next event on.
#[tokio::test]
async fn agent_name_follows_a_nested_handoff() {
    let seen: Seen = Arc::default();
    let specialist =
        Agent::new("specialist").model(Arc::new(ScriptedModel::new([ModelStep::from(
            ItemHelpers::text_message("from specialist"),
        )])));
    let to_specialist = handoff(specialist);
    let helper = Agent::new("helper")
        .model(Arc::new(ScriptedModel::new([ModelStep::from(
            ItemHelpers::function_tool_call(to_specialist.tool_name.clone(), "{}", "h1"),
        )])))
        .handoffs(vec![to_specialist]);
    let outer = Agent::new("boss")
        .model(Arc::new(ScriptedModel::new([
            call_helper("outer-1"),
            ModelStep::from(ItemHelpers::text_message("done")),
        ])))
        .tools(vec![helper.as_tool(recording(&seen))]);

    Runner::run(&outer, "go", RunOptions::default())
        .await
        .expect("run");

    let events = seen.lock().unwrap();
    let switch = events
        .iter()
        .position(|e| {
            matches!(&e.event, StreamEvent::AgentUpdated { agent_name } if agent_name == "specialist")
        })
        .expect("the handoff is reported");
    assert!(events[..switch].iter().all(|e| e.agent_name == "helper"));
    assert!(events[switch..]
        .iter()
        .all(|e| e.agent_name == "specialist"));
}

/// A nested approval still bubbles up to the outer run, and resuming it streams the rest.
#[tokio::test]
async fn nested_approval_bubbles_up_and_resumes_with_streaming() {
    let seen: Seen = Arc::default();
    let ran = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&ran);
    let secret = FunctionTool::new(
        "secret",
        "secret",
        serde_json::json!({"type":"object","properties":{},"additionalProperties":false}),
        move |_ctx, _args| {
            let counter = Arc::clone(&counter);
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(serde_json::Value::String("secret-ok".into()))
            }
        },
    )
    .with_needs_resolution(true);
    let helper = Agent::new("helper")
        .model(Arc::new(ScriptedModel::new([
            ModelStep::from(ItemHelpers::function_tool_call("secret", "{}", "n1")),
            ModelStep::from(ItemHelpers::text_message("nested done")),
        ])))
        .tools(vec![secret]);
    let outer = Agent::new("boss")
        .model(Arc::new(ScriptedModel::new([
            call_helper("outer-1"),
            ModelStep::from(ItemHelpers::text_message("outer done")),
        ])))
        .tools(vec![helper.as_tool(recording(&seen))]);

    let paused = Runner::run(&outer, "go", RunOptions::default())
        .await
        .expect("run");
    assert!(paused.is_interrupted());
    assert_eq!(paused.interruptions[0].tool_name, "secret");
    assert_eq!(paused.interruptions[0].agent_name, "helper");
    assert_eq!(ran.load(Ordering::SeqCst), 0);
    assert!(!seen.lock().unwrap().is_empty(), "the first leg streamed");
    seen.lock().unwrap().clear();

    let mut state = RunState::from_string(&outer.name, &paused.to_state().unwrap().to_string())
        .expect("round trip");
    state.approve(&paused.interruptions[0], false);
    let result = Runner::run_state(&outer, state, RunOptions::default())
        .await
        .expect("resume");

    assert_eq!(result.final_output_as_str(), Some("outer done"));
    assert_eq!(ran.load(Ordering::SeqCst), 1);
    assert_eq!(deltas(&seen), "nested done", "the resumed leg streamed too");
}

/// A failing nested run fails the tool the way it does without a handler.
#[tokio::test]
async fn a_failing_nested_run_fails_the_tool() {
    let seen: Seen = Arc::default();
    // No steps: the scripted model errors on its first call.
    let helper = Agent::new("helper").model(Arc::new(ScriptedModel::new([])));
    let outer = Agent::new("boss")
        .model(Arc::new(ScriptedModel::new([call_helper("outer-1")])))
        .tools(vec![helper.as_tool(recording(&seen))]);
    let streamed = Runner::run(&outer, "go", RunOptions::default()).await;

    let helper = Agent::new("helper").model(Arc::new(ScriptedModel::new([])));
    let outer = Agent::new("boss")
        .model(Arc::new(ScriptedModel::new([call_helper("outer-1")])))
        .tools(vec![helper.as_tool(AsToolConfig {
            name: Some("call_helper".into()),
            ..Default::default()
        })]);
    let plain = Runner::run(&outer, "go", RunOptions::default()).await;

    assert_eq!(
        streamed.is_err(),
        plain.is_err(),
        "streaming does not change whether a nested failure is fatal"
    );
}

/// The handler can also be set on a streamed outer run: events of both levels interleave.
#[tokio::test]
async fn works_under_a_streamed_outer_run() {
    let seen: Seen = Arc::default();
    let helper = Agent::new("helper").model(Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("nested answer"),
    )])));
    let outer = Agent::new("boss")
        .model(Arc::new(ScriptedModel::new([
            call_helper("outer-1"),
            ModelStep::from(ItemHelpers::text_message("outer done")),
        ])))
        .tools(vec![helper.as_tool(recording(&seen))]);

    let mut streamed = Runner::run_streamed(outer, "go", RunOptions::default());
    streamed.collect_events().await.expect("events");
    assert_eq!(
        streamed.final_output(),
        Some(serde_json::json!("outer done"))
    );
    assert_eq!(deltas(&seen), "nested answer");
}
