//! Streaming runner tests.

use std::sync::Arc;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{
    Agent, FunctionTool, RunItemStreamName, RunOptions, Runner, StreamEvent, ToolUseBehavior,
};

#[tokio::test]
async fn streamed_plain_text_emits_agent_raw_and_message() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("hello"),
    )]));
    let agent = Agent::new("assistant").model(model);
    let mut streamed = Runner::run_streamed(agent, "hi", RunOptions::default());
    let events = streamed.collect_events().await.expect("events");

    assert!(matches!(
        &events[0],
        StreamEvent::AgentUpdated { agent_name } if agent_name == "assistant"
    ));
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::RawResponse { .. })));
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::RunItem {
            name: RunItemStreamName::MessageOutputCreated,
            ..
        }
    )));
    assert!(streamed.is_complete());
    assert_eq!(
        streamed
            .final_output()
            .and_then(|v| v.as_str().map(str::to_string)),
        Some("hello".into())
    );
}

#[tokio::test]
async fn streamed_tool_then_text_emits_tool_events() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("assistant")
        .model(model)
        .tools(vec![FunctionTool::constant("echo", "echo", "ok")]);
    let mut streamed = Runner::run_streamed(agent, "go", RunOptions::default());
    let events = streamed.collect_events().await.expect("events");

    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::RunItem {
            name: RunItemStreamName::ToolCalled,
            ..
        }
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::RunItem {
            name: RunItemStreamName::ToolOutput,
            ..
        }
    )));
    assert_eq!(
        streamed
            .final_output()
            .and_then(|v| v.as_str().map(str::to_string)),
        Some("done".into())
    );
}

/// Every turn ends with exactly one standard `response.completed` wire event (D-011).
#[tokio::test]
async fn streamed_emits_one_completed_event_per_turn() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("assistant")
        .model(model)
        .tools(vec![FunctionTool::constant("echo", "echo", "ok")]);
    let mut streamed = Runner::run_streamed(agent, "go", RunOptions::default());
    let events = streamed.collect_events().await.expect("events");

    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(
                e,
                StreamEvent::RawResponse { data } if data["type"] == "response.completed"
            ))
            .count(),
        2
    );
    assert_eq!(
        streamed
            .final_output()
            .and_then(|v| v.as_str().map(str::to_string)),
        Some("done".into())
    );
}

#[tokio::test]
async fn streamed_stop_on_first_tool() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call("echo", "{}", "c1"),
    )]));
    let agent = Agent::new("assistant")
        .model(model)
        .tools(vec![FunctionTool::constant("echo", "echo", "final")])
        .tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
    let mut streamed = Runner::run_streamed(agent, "go", RunOptions::default());
    let _ = streamed.collect_events().await.expect("events");
    assert_eq!(
        streamed
            .final_output()
            .and_then(|v| v.as_str().map(str::to_string)),
        Some("final".into())
    );
}

/// Streamed HITL keeps a resumable `RunState` (drain first, then `to_state`).
#[tokio::test]
async fn streamed_hitl_to_state_then_run_state_streamed() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("delete_file", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("deleted")),
    ]));
    let agent = Agent::new("assistant")
        .model(model)
        .tools(vec![FunctionTool::constant(
            "delete_file",
            "delete",
            "deleted",
        )
        .with_needs_resolution(true)]);

    let mut streamed = Runner::run_streamed(agent.clone(), "go", RunOptions::default());
    let _ = streamed.collect_events().await.expect("events");
    assert!(streamed.is_interrupted());
    assert!(streamed.final_output().is_none());

    let mut state = streamed.to_state().expect("to_state");
    let item = state.interruptions[0].clone();
    state.approve(&item, false);

    let mut resumed = Runner::run_state_streamed(agent, state, RunOptions::default())
        .expect("run_state_streamed");
    let _ = resumed.collect_events().await.expect("resume events");
    assert!(!resumed.is_interrupted());
    assert_eq!(
        resumed
            .final_output()
            .and_then(|v| v.as_str().map(str::to_string)),
        Some("deleted".into())
    );
}
