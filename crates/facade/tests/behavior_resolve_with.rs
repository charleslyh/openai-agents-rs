//! Rust-only (X-009): `RunOptions::resolve_with` resolves paused tool calls in-process, with the
//! same outcome as the explicit `RunState` approve / reject / provide_output loop.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{
    Agent, AgentsError, AsToolConfig, FunctionTool, RunOptions, Runner, ToolInterruptionItem,
    ToolResolution,
};

fn counting_tool(name: &str, counter: &Arc<AtomicUsize>) -> FunctionTool {
    let counter = Arc::clone(counter);
    FunctionTool::new(
        name,
        name,
        serde_json::json!({"type":"object","properties":{},"additionalProperties":false}),
        move |_ctx, _args| {
            let counter = Arc::clone(&counter);
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(serde_json::Value::String("ran".into()))
            }
        },
    )
    .with_needs_resolution(true)
}

/// The `function_call_output` the model received for `call_id` on its last call.
fn output_sent_for(model: &ScriptedModel, call_id: &str) -> Option<String> {
    let input = model.last_call()?.input;
    input.as_array()?.iter().find_map(|item| {
        (item["type"] == "function_call_output" && item["call_id"] == call_id)
            .then(|| item["output"].as_str().map(str::to_owned))
            .flatten()
    })
}

#[tokio::test]
async fn approving_through_the_callback_runs_the_tool_and_finishes() {
    let runs = Arc::new(AtomicUsize::new(0));
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("deploy", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("deployed")),
    ]));
    let agent = Agent::new("ops")
        .model(model.clone())
        .tools(vec![counting_tool("deploy", &runs)]);

    let asked: Arc<Mutex<Vec<ToolInterruptionItem>>> = Arc::default();
    let seen = Arc::clone(&asked);
    let options = RunOptions::new().resolve_with(move |call| {
        seen.lock().unwrap().push(call);
        async { ToolResolution::Run }
    });
    let result = Runner::run(&agent, "ship it", options).await.expect("run");

    assert!(!result.is_interrupted());
    assert_eq!(result.final_output_as_str(), Some("deployed"));
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    let asked = asked.lock().unwrap();
    assert_eq!(asked.len(), 1);
    assert_eq!(
        (asked[0].agent_name.as_str(), asked[0].tool_name.as_str()),
        ("ops", "deploy")
    );
    assert_eq!(asked[0].call_id, "c1");
    model.assert_complete();
}

#[tokio::test]
async fn reject_and_output_reach_the_model_without_running_the_tool() {
    let runs = Arc::new(AtomicUsize::new(0));
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(vec![
            ItemHelpers::function_tool_call("delete", "{}", "c1"),
            ItemHelpers::function_tool_call("render", "{}", "c2"),
        ]),
        ModelStep::from(ItemHelpers::text_message("ok")),
    ]));
    let agent = Agent::new("a").model(model.clone()).tools(vec![
        counting_tool("delete", &runs),
        counting_tool("render", &runs),
    ]);

    let options = RunOptions::new().resolve_with(|call| async move {
        match call.tool_name.as_str() {
            "delete" => ToolResolution::reject("not allowed"),
            _ => ToolResolution::output("rendered elsewhere"),
        }
    });
    let result = Runner::run(&agent, "go", options).await.expect("run");

    assert_eq!(result.final_output_as_str(), Some("ok"));
    assert_eq!(runs.load(Ordering::SeqCst), 0, "neither tool body ran");
    assert_eq!(
        output_sent_for(&model, "c1").as_deref(),
        Some("not allowed")
    );
    assert_eq!(
        output_sent_for(&model, "c2").as_deref(),
        Some("rendered elsewhere")
    );
}

/// Pauses in later turns are resolved too, and a nested `as_tool` pause is resolved on the
/// nested run like the explicit flow does.
#[tokio::test]
async fn later_and_nested_pauses_are_resolved_as_well() {
    let nested_runs = Arc::new(AtomicUsize::new(0));
    let nested_model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("secret", "{}", "n1")),
        ModelStep::from(ItemHelpers::text_message("nested-done")),
    ]));
    let helper = Agent::new("helper")
        .model(nested_model.clone())
        .tools(vec![counting_tool("secret", &nested_runs)]);
    let call_helper = helper.as_tool(AsToolConfig {
        name: Some("call_helper".into()),
        description: Some("Call helper".into()),
        needs_resolution: false,
        max_turns: Some(5),
    });

    let outer_runs = Arc::new(AtomicUsize::new(0));
    let outer_model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(
            "call_helper",
            r#"{"input":"x"}"#,
            "o1",
        )),
        ModelStep::from(ItemHelpers::function_tool_call("deploy", "{}", "o2")),
        ModelStep::from(ItemHelpers::text_message("all done")),
    ]));
    let outer = Agent::new("boss")
        .model(outer_model.clone())
        .tools(vec![call_helper, counting_tool("deploy", &outer_runs)]);

    let asked = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&asked);
    let options = RunOptions::new().resolve_with(move |_call| {
        counter.fetch_add(1, Ordering::SeqCst);
        async { ToolResolution::Run }
    });
    let result = Runner::run(&outer, "go", options).await.expect("run");

    assert_eq!(result.final_output_as_str(), Some("all done"));
    assert_eq!(asked.load(Ordering::SeqCst), 2);
    assert_eq!(nested_runs.load(Ordering::SeqCst), 1);
    assert_eq!(outer_runs.load(Ordering::SeqCst), 1);
    nested_model.assert_complete();
    outer_model.assert_complete();
}

/// A streamed run cannot resolve in-process; it fails instead of silently ignoring the resolver.
#[tokio::test]
async fn a_streamed_run_with_a_resolver_fails_closed() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("never"),
    )]));
    let agent = Agent::new("a").model(model.clone());
    let options = RunOptions::new().resolve_with(|_call| async { ToolResolution::Run });
    let mut stream = Runner::run_streamed(agent, "go", options);
    let error = stream
        .collect_events()
        .await
        .expect_err("the stream reports an error");
    assert!(matches!(error, AgentsError::User(_)), "{error:?}");
    assert!(error.to_string().contains("resolve_with"), "{error}");
    assert!(model.last_call().is_none(), "the model is never called");
}
