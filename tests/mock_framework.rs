//! Framework tests: MockResponses / MockCompletions drive the agent loop.
#![cfg(feature = "testing")]

use std::sync::Arc;
use std::time::Duration;

use openai_agents::testing::{
    ConcurrentProbe, MockCompletions, MockResponses, MockToolCall,
};
use openai_agents::{
    Agent, FunctionTool, RunItemStreamName, RunOptions, Runner, StreamEvent, ToolUseBehavior,
};

fn weather() -> FunctionTool {
    FunctionTool::new(
        "get_weather",
        "weather",
        serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }),
        |ctx, args| async move {
            let v: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
            let city = v
                .get("city")
                .and_then(|c| c.as_str())
                .unwrap_or(&ctx.tool_arguments);
            Ok(serde_json::Value::String(format!("sunny in {city}")))
        },
    )
}

#[tokio::test]
async fn responses_multi_turn_tool_loop() {
    let mock = MockResponses::start().await;
    mock.enqueue_tool_calls([MockToolCall::new(
        "get_weather",
        r#"{"city":"Paris"}"#,
        "call-w",
    )]);
    mock.enqueue_text("It is sunny in Paris.");

    let agent = Agent::new("assistant")
        .instructions("use tools")
        .model(Arc::new(mock.model("mock-model")))
        .tools(vec![weather()]);

    let result = Runner::run(&agent, "weather in Paris?", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("It is sunny in Paris."));
    assert_eq!(result.raw_responses.len(), 2);

    let bodies = mock.request_bodies().await;
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[0]["model"], "mock-model");
    assert!(bodies[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t["name"] == "get_weather"));
    let input1 = bodies[1]["input"].as_array().expect("turn-2 input");
    assert!(input1.iter().any(|i| i["type"] == "function_call_output"
        && i["call_id"] == "call-w"
        && i["output"].as_str().unwrap().contains("Paris")));
}

#[tokio::test]
async fn completions_multi_turn_tool_loop() {
    let mock = MockCompletions::start().await;
    mock.enqueue_tool_calls([MockToolCall::new(
        "get_weather",
        r#"{"city":"Tokyo"}"#,
        "call-w",
    )]);
    mock.enqueue_text("Tokyo is sunny.");

    let agent = Agent::new("assistant")
        .model(Arc::new(mock.model("mock-chat")))
        .tools(vec![weather()]);

    let result = Runner::run(&agent, "weather?", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("Tokyo is sunny."));
    assert_eq!(result.raw_responses.len(), 2);

    let bodies = mock.request_bodies().await;
    assert_eq!(bodies.len(), 2);
    assert!(bodies[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t["function"]["name"] == "get_weather"));
    let msgs = bodies[1]["messages"].as_array().expect("messages");
    assert!(msgs.iter().any(|m| m["role"] == "tool"
        && m["tool_call_id"] == "call-w"
        && m["content"].as_str().unwrap().contains("Tokyo")));
}

#[tokio::test]
async fn responses_parallel_mock_tools() {
    let mock = MockResponses::start().await;
    mock.enqueue_tool_calls([
        MockToolCall::new("slow", "{}", "c-slow"),
        MockToolCall::new("fast", "{}", "c-fast"),
    ]);
    mock.enqueue_text("both done");

    let probe = ConcurrentProbe::new();
    let agent = Agent::new("assistant")
        .model(Arc::new(mock.model("mock")))
        .tools(vec![
            probe.delayed_tool("slow", Duration::from_millis(80), "slow-ok"),
            probe.delayed_tool("fast", Duration::from_millis(10), "fast-ok"),
        ]);

    let started = std::time::Instant::now();
    let result = Runner::run(&agent, "run both", RunOptions::default())
        .await
        .expect("run");
    let elapsed = started.elapsed();

    assert_eq!(result.final_output_as_str(), Some("both done"));
    assert!(
        probe.max_inflight() >= 2,
        "expected overlapping tools, max_inflight={}",
        probe.max_inflight()
    );
    assert!(
        elapsed < Duration::from_millis(150),
        "expected parallel tools, elapsed={elapsed:?}"
    );
    // Completion order: fast finishes first.
    assert_eq!(probe.completed_names()[0], "fast");
}

#[tokio::test]
async fn completions_parallel_mock_tools() {
    let mock = MockCompletions::start().await;
    mock.enqueue_tool_calls([
        MockToolCall::new("slow", "{}", "c-slow"),
        MockToolCall::new("fast", "{}", "c-fast"),
    ]);
    mock.enqueue_text("ok");

    let probe = ConcurrentProbe::new();
    let agent = Agent::new("assistant")
        .model(Arc::new(mock.model("mock")))
        .tools(vec![
            probe.delayed_tool("slow", Duration::from_millis(80), "s"),
            probe.delayed_tool("fast", Duration::from_millis(10), "f"),
        ]);

    let result = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("ok"));
    assert!(probe.max_inflight() >= 2);
}

#[tokio::test]
async fn responses_streamed_tool_loop_events() {
    let mock = MockResponses::start().await;
    mock.enqueue_tool_calls([MockToolCall::new("get_weather", r#"{"city":"Oslo"}"#, "c1")]);
    mock.enqueue_text("Oslo is sunny.");

    let agent = Agent::new("streamer")
        .model(Arc::new(mock.model("mock")))
        .tools(vec![weather()]);

    let mut streamed = Runner::run_streamed(agent, "weather?", RunOptions::default());
    let events = streamed.collect_events().await.expect("events");

    assert!(matches!(
        &events[0],
        StreamEvent::AgentUpdated { agent_name } if agent_name == "streamer"
    ));
    let names: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::RunItem { name, .. } => Some(*name),
            _ => None,
        })
        .collect();
    assert!(names.contains(&RunItemStreamName::ToolCalled));
    assert!(names.contains(&RunItemStreamName::ToolOutput));
    assert!(names.contains(&RunItemStreamName::MessageOutputCreated));
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, StreamEvent::RawResponse { .. }))
            .count(),
        2
    );
    assert_eq!(
        streamed
            .final_output()
            .and_then(|v| v.as_str().map(str::to_string)),
        Some("Oslo is sunny.".into())
    );
}

#[tokio::test]
async fn completions_streamed_stop_on_first_tool() {
    let mock = MockCompletions::start().await;
    mock.enqueue_tool_calls([MockToolCall::new("get_weather", r#"{"city":"X"}"#, "c1")]);

    let agent = Agent::new("assistant")
        .model(Arc::new(mock.model("mock")))
        .tools(vec![weather()])
        .tool_use_behavior(ToolUseBehavior::StopOnFirstTool);

    let mut streamed = Runner::run_streamed(agent, "go", RunOptions::default());
    let events = streamed.collect_events().await.expect("events");
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::RunItem {
            name: RunItemStreamName::ToolOutput,
            ..
        }
    )));
    let out = streamed.final_output().expect("final");
    assert!(out.as_str().unwrap().contains("sunny"));
}

#[tokio::test]
async fn responses_three_turn_loop_then_text() {
    let mock = MockResponses::start().await;
    mock.enqueue_tool_calls([MockToolCall::new(
        "get_weather",
        r#"{"city":"A"}"#,
        "c1",
    )]);
    mock.enqueue_tool_calls([MockToolCall::new(
        "get_weather",
        r#"{"city":"B"}"#,
        "c2",
    )]);
    mock.enqueue_text("A then B");

    let agent = Agent::new("assistant")
        .model(Arc::new(mock.model("mock")))
        .tools(vec![weather()]);

    let result = Runner::run(&agent, "two cities", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.raw_responses.len(), 3);
    assert_eq!(result.final_output_as_str(), Some("A then B"));
    assert_eq!(mock.request_bodies().await.len(), 3);
}
