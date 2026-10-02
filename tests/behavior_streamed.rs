//! Streaming runner tests.

use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
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
        streamed.final_output().and_then(|v| v.as_str().map(str::to_string)),
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
        streamed.final_output().and_then(|v| v.as_str().map(str::to_string)),
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
        streamed.final_output().and_then(|v| v.as_str().map(str::to_string)),
        Some("final".into())
    );
}
