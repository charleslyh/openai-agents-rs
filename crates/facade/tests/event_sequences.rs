//! Exact event sequences of streamed runs.
//!
//! Each test names one way a tool call can end and pins the complete, ordered list of semantic
//! events a consumer sees: which events appear, in what order, and with which output. Raw model
//! events are left out because their count depends on the model, not on the run loop.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{
    Agent, CancellationToken, FunctionTool, RunConfig, RunItem, RunOptions, Runner, StreamEvent,
    ToolExecutionConfig, ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolOutputGuardrail,
    ToolUseBehavior,
};
use serde_json::{json, Value};

fn empty_schema() -> Value {
    json!({"type": "object", "properties": {}, "additionalProperties": false})
}

/// One short label per semantic event; raw model events are skipped.
fn labels(events: &[StreamEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::RawResponse { .. } => None,
            StreamEvent::AgentUpdated { agent_name } => Some(format!("agent:{agent_name}")),
            StreamEvent::RunItem { name, item } => Some(match item {
                RunItem::ToolCallOutput(o) => format!(
                    "{}:{}",
                    name.as_str(),
                    o.raw_item["output"].as_str().unwrap_or("<non-text>")
                ),
                RunItem::Message(_) => format!("{}:message", name.as_str()),
                _ => name.as_str().to_string(),
            }),
            _ => Some("<unknown>".to_string()),
        })
        .collect()
}

fn call_then_text(name: &str) -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(name, "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]))
}

async fn collect(agent: &Agent, options: RunOptions) -> (Vec<String>, bool) {
    let mut streamed = Runner::run_streamed(agent.clone(), "go", options);
    let events = tokio::time::timeout(Duration::from_secs(5), streamed.collect_events())
        .await
        .expect("the stream must end")
        .expect("a stream that ends is not an error");
    (labels(&events), streamed.is_cancelled())
}

#[tokio::test]
async fn a_successful_call_is_called_then_output_then_the_final_message() {
    let agent = Agent::new("a")
        .model(call_then_text("echo"))
        .tools(vec![FunctionTool::constant("echo", "echo", "ok")]);

    let (events, cancelled) = collect(&agent, RunOptions::default()).await;

    assert_eq!(
        events,
        [
            "agent:a",
            "tool_called",
            "tool_output:ok",
            "message_output_created:message"
        ]
    );
    assert!(!cancelled);
}

#[tokio::test]
async fn a_rejecting_input_guardrail_skips_the_body_and_answers_with_its_message() {
    let body_runs = Arc::new(AtomicUsize::new(0));
    let tool = {
        let runs = Arc::clone(&body_runs);
        FunctionTool::new("t", "t", empty_schema(), move |_ctx, _args| {
            let runs = Arc::clone(&runs);
            async move {
                runs.fetch_add(1, Ordering::SeqCst);
                Ok(json!("body"))
            }
        })
        .with_tool_input_guardrails(vec![ToolInputGuardrail::new("g", |_data| async {
            ToolGuardrailFunctionOutput::reject_content("blocked", json!(null))
        })])
    };
    let agent = Agent::new("a").model(call_then_text("t")).tools(vec![tool]);

    let (events, _) = collect(&agent, RunOptions::default()).await;

    assert_eq!(
        events,
        [
            "agent:a",
            "tool_called",
            "tool_output:blocked",
            "message_output_created:message"
        ]
    );
    assert_eq!(body_runs.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn an_output_guardrail_rewrites_the_output_the_consumer_sees() {
    let tool = FunctionTool::constant("t", "t", "secret-123").with_tool_output_guardrails(vec![
        ToolOutputGuardrail::new("redact", |_data| async {
            ToolGuardrailFunctionOutput::reject_content("[redacted]", json!(null))
        }),
    ]);
    let agent = Agent::new("a").model(call_then_text("t")).tools(vec![tool]);

    let (events, _) = collect(&agent, RunOptions::default()).await;

    assert_eq!(
        events,
        [
            "agent:a",
            "tool_called",
            "tool_output:[redacted]",
            "message_output_created:message"
        ]
    );
}

#[tokio::test]
async fn stopping_on_the_first_tool_ends_the_stream_after_its_output() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("never asked for")),
    ]));
    let agent = Agent::new("a")
        .model(model.clone())
        .tools(vec![FunctionTool::constant("echo", "echo", "ok")])
        .tool_use_behavior(ToolUseBehavior::StopOnFirstTool);

    let (events, _) = collect(&agent, RunOptions::default()).await;

    assert_eq!(events, ["agent:a", "tool_called", "tool_output:ok"]);
    assert_eq!(model.calls().len(), 1, "the model is not asked again");
}

#[tokio::test]
async fn a_call_waiting_for_approval_pauses_the_stream_without_an_output() {
    let body_runs = Arc::new(AtomicUsize::new(0));
    let tool = {
        let runs = Arc::clone(&body_runs);
        FunctionTool::new("danger", "needs a person", empty_schema(), move |_c, _a| {
            let runs = Arc::clone(&runs);
            async move {
                runs.fetch_add(1, Ordering::SeqCst);
                Ok(json!("ran"))
            }
        })
        .with_needs_resolution(true)
    };
    let model = call_then_text("danger");
    let agent = Agent::new("a").model(model.clone()).tools(vec![tool]);

    let mut streamed = Runner::run_streamed(agent, "go", RunOptions::default());
    let events = tokio::time::timeout(Duration::from_secs(5), streamed.collect_events())
        .await
        .expect("the stream must end")
        .expect("a paused stream is not an error");

    // The pause is reported through the result, not as a stream event.
    assert_eq!(labels(&events), ["agent:a", "tool_called"]);
    assert!(streamed.is_interrupted());
    assert!(!streamed.is_cancelled());
    assert_eq!(body_runs.load(Ordering::SeqCst), 0);
    assert_eq!(model.calls().len(), 1, "the model waits for the answer");
}

#[tokio::test]
async fn cancelling_after_a_call_started_still_reports_its_output() {
    let token = CancellationToken::new();
    let tool = FunctionTool::new("slow", "slow", empty_schema(), |ctx, _args| async move {
        ctx.cancellation().cancelled().await;
        Ok(json!("partial"))
    });
    let model = call_then_text("slow");
    let agent = Agent::new("a").model(model.clone()).tools(vec![tool]);
    let canceller = {
        let token = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            token.cancel();
        })
    };

    let (events, cancelled) = collect(&agent, RunOptions::new().cancellation_token(token)).await;
    canceller.await.unwrap();

    assert_eq!(events, ["agent:a", "tool_called", "tool_output:partial"]);
    assert!(cancelled);
    assert_eq!(model.calls().len(), 1, "no turn starts after the cancel");
}

#[tokio::test]
async fn a_call_that_never_started_gets_a_cancelled_output_in_order() {
    let token = CancellationToken::new();
    let tool = |name: &'static str| {
        let token = token.clone();
        FunctionTool::new(name, name, empty_schema(), move |_ctx, _args| {
            let token = token.clone();
            async move {
                token.cancel();
                Ok(json!(name))
            }
        })
    };
    let model = Arc::new(ScriptedModel::new([ModelStep::from(vec![
        ItemHelpers::function_tool_call("a", "{}", "c1"),
        ItemHelpers::function_tool_call("b", "{}", "c2"),
    ])]));
    let agent = Agent::new("x")
        .model(model)
        .tools(vec![tool("a"), tool("b")]);
    let mut config = RunConfig::default();
    config.tools.execution = Some(ToolExecutionConfig {
        max_function_tool_concurrency: Some(1),
    });
    let mut options = RunOptions::new().cancellation_token(token);
    options.run_config = config;

    let (events, cancelled) = collect(&agent, options).await;

    assert!(cancelled);
    assert_eq!(&events[..3], ["agent:x", "tool_called", "tool_called"]);
    let outputs = &events[3..];
    assert_eq!(outputs.len(), 2, "every call has an output: {events:?}");
    assert_eq!(outputs[0], "tool_output:a");
    assert!(outputs[1].to_lowercase().contains("cancel"), "{events:?}");
}
