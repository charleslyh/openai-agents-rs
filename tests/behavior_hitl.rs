//! Human-in-the-loop / tool approval behavior tests.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    Agent, FunctionTool, RunOptions, Runner, DEFAULT_APPROVAL_REJECTION_MESSAGE,
};

#[tokio::test]
async fn approve_then_resume_invokes_tool() {
    let invokes = Arc::new(AtomicUsize::new(0));
    let invokes_cb = Arc::clone(&invokes);
    let tool = FunctionTool::new(
        "delete_file",
        "delete a file",
        serde_json::json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"],
            "additionalProperties": false
        }),
        move |_ctx, args| {
            let invokes_cb = Arc::clone(&invokes_cb);
            async move {
                invokes_cb.fetch_add(1, Ordering::SeqCst);
                Ok(serde_json::Value::String(format!("deleted:{args}")))
            }
        },
    )
    .with_needs_approval(true);

    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(
            "delete_file",
            r#"{"path":"/tmp/x"}"#,
            "call-1",
        )),
        ModelStep::from(ItemHelpers::text_message("ok")),
    ]));
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![tool]);

    let result = Runner::run(&agent, "delete", RunOptions::default())
        .await
        .expect("run");
    assert!(result.is_interrupted());
    assert_eq!(result.interruptions.len(), 1);
    assert_eq!(result.interruptions[0].tool_name, "delete_file");
    assert_eq!(result.final_output, serde_json::Value::Null);
    assert_eq!(invokes.load(Ordering::SeqCst), 0);
    assert_eq!(model.calls().len(), 1);

    let mut state = result.to_state().expect("to_state");
    state.approve(&result.interruptions[0], false);
    let result = Runner::run_state(&agent, state, RunOptions::default())
        .await
        .expect("resume");
    assert!(!result.is_interrupted());
    assert_eq!(result.final_output_as_str(), Some("ok"));
    assert_eq!(invokes.load(Ordering::SeqCst), 1);
    assert_eq!(model.calls().len(), 2);
    model.assert_complete();
}

#[tokio::test]
async fn reject_then_resume_skips_tool_body() {
    let invokes = Arc::new(AtomicUsize::new(0));
    let invokes_cb = Arc::clone(&invokes);
    let tool = FunctionTool::new(
        "danger",
        "danger",
        serde_json::json!({"type":"object","properties":{},"additionalProperties":false}),
        move |_ctx, _args| {
            let invokes_cb = Arc::clone(&invokes_cb);
            async move {
                invokes_cb.fetch_add(1, Ordering::SeqCst);
                Ok(serde_json::Value::String("ran".into()))
            }
        },
    )
    .with_needs_approval(true);

    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("danger", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("denied-path")),
    ]));
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![tool]);

    let result = Runner::run(&agent, "do it", RunOptions::default())
        .await
        .expect("run");
    let mut state = result.to_state().expect("state");
    state.reject(&result.interruptions[0], false, None);
    let result = Runner::run_state(&agent, state, RunOptions::default())
        .await
        .expect("resume");
    assert_eq!(invokes.load(Ordering::SeqCst), 0);
    assert_eq!(result.final_output_as_str(), Some("denied-path"));
    assert_eq!(model.calls().len(), 2);
    assert!(
        model.calls()[1]
            .input
            .to_string()
            .contains(DEFAULT_APPROVAL_REJECTION_MESSAGE),
        "expected default rejection in second model input: {}",
        model.calls()[1].input
    );
    model.assert_complete();
}

#[tokio::test]
async fn dynamic_needs_approval_only_for_matching_args() {
    let tool = FunctionTool::new(
        "get_temperature",
        "temp",
        serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
            "additionalProperties": false
        }),
        |_ctx, args| async move { Ok(serde_json::Value::String(format!("temp:{args}"))) },
    )
    .with_needs_approval_fn(|params, _call_id| async move {
        params
            .get("city")
            .and_then(|c| c.as_str())
            .map(|c| c.contains("Oakland"))
            .unwrap_or(false)
    });

    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(
            "get_temperature",
            r#"{"city":"SF"}"#,
            "c-sf",
        )),
        ModelStep::from(ItemHelpers::text_message("warm")),
    ]));
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![tool]);
    let result = Runner::run(&agent, "sf", RunOptions::default())
        .await
        .expect("run");
    assert!(!result.is_interrupted());
    assert_eq!(result.final_output_as_str(), Some("warm"));
    model.assert_complete();
}

#[tokio::test]
async fn custom_rejection_message() {
    let tool = FunctionTool::constant("x", "x", "nope").with_needs_approval(true);
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("x", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![tool]);
    let result = Runner::run(&agent, "x", RunOptions::default())
        .await
        .expect("run");
    let mut state = result.to_state().expect("state");
    state.reject(&result.interruptions[0], false, Some("User denied"));
    let _ = Runner::run_state(&agent, state, RunOptions::default())
        .await
        .expect("resume");
    assert!(model.calls()[1]
        .input
        .to_string()
        .contains("User denied"));
}

#[tokio::test]
async fn always_approve_applies_to_later_calls() {
    let invokes = Arc::new(AtomicUsize::new(0));
    let invokes_cb = Arc::clone(&invokes);
    let tool = FunctionTool::new(
        "pay",
        "pay",
        serde_json::json!({"type":"object","properties":{},"additionalProperties":false}),
        move |_ctx, _args| {
            let invokes_cb = Arc::clone(&invokes_cb);
            async move {
                invokes_cb.fetch_add(1, Ordering::SeqCst);
                Ok(serde_json::Value::String("paid".into()))
            }
        },
    )
    .with_needs_approval(true);

    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("pay", "{}", "c1")),
        ModelStep::from(ItemHelpers::function_tool_call("pay", "{}", "c2")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![tool]);

    let result = Runner::run(&agent, "pay", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.interruptions.len(), 1);
    let mut state = result.to_state().expect("state");
    state.approve(&result.interruptions[0], true);
    let result = Runner::run_state(&agent, state, RunOptions::default())
        .await
        .expect("resume1");
    assert!(!result.is_interrupted());
    assert_eq!(result.final_output_as_str(), Some("done"));
    assert_eq!(invokes.load(Ordering::SeqCst), 2);
    model.assert_complete();
}

#[tokio::test]
async fn run_state_json_round_trip() {
    let tool = FunctionTool::constant("x", "x", "ok").with_needs_approval(true);
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("x", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("after")),
    ]));
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![tool]);
    let result = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    let mut state = result.to_state().expect("state");
    state.approve(&result.interruptions[0], false);
    let json = state.to_json();
    assert_eq!(
        json.get("$schemaVersion").and_then(|v| v.as_str()),
        Some(openai_agents::RUN_STATE_SCHEMA_VERSION)
    );
    let restored = openai_agents::RunState::from_json(&agent.name, json).expect("from_json");
    let result = Runner::run_state(&agent, restored, RunOptions::default())
        .await
        .expect("resume");
    assert_eq!(result.final_output_as_str(), Some("after"));
    model.assert_complete();
}

#[tokio::test]
async fn agent_as_tool_nested_approval_bubbles_to_outer() {
    let nested_invokes = Arc::new(AtomicUsize::new(0));
    let nested_cb = Arc::clone(&nested_invokes);
    let nested_tool = FunctionTool::new(
        "secret",
        "secret",
        serde_json::json!({"type":"object","properties":{},"additionalProperties":false}),
        move |_ctx, _args| {
            let nested_cb = Arc::clone(&nested_cb);
            async move {
                nested_cb.fetch_add(1, Ordering::SeqCst);
                Ok(serde_json::Value::String("secret-ok".into()))
            }
        },
    )
    .with_needs_approval(true);

    let nested_model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("secret", "{}", "nested-1")),
        ModelStep::from(ItemHelpers::text_message("nested-done")),
    ]));
    let nested_agent = Agent::new("helper")
        .model(nested_model.clone())
        .tools(vec![nested_tool]);

    let agent_tool = nested_agent.as_tool(openai_agents::AsToolConfig {
        name: Some("call_helper".into()),
        description: Some("Call helper".into()),
        needs_approval: false,
        max_turns: Some(5),
    });

    let outer_model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(
            "call_helper",
            r#"{"input":"do secret"}"#,
            "outer-1",
        )),
        ModelStep::from(ItemHelpers::text_message("outer-done")),
    ]));
    let outer = Agent::new("boss")
        .model(outer_model.clone())
        .tools(vec![agent_tool]);

    let result = Runner::run(&outer, "delegate", RunOptions::default())
        .await
        .expect("outer run");
    assert!(result.is_interrupted());
    assert_eq!(result.interruptions.len(), 1);
    assert_eq!(result.interruptions[0].tool_name, "secret");
    assert_eq!(result.interruptions[0].agent_name, "helper");
    assert_eq!(nested_invokes.load(Ordering::SeqCst), 0);

    let state = result.to_state().expect("state");
    let mut state = openai_agents::RunState::from_string(&outer.name, &state.to_string())
        .expect("from_string");
    state.approve(&result.interruptions[0], false);

    let result = Runner::run_state(&outer, state, RunOptions::default())
        .await
        .expect("resume");
    assert!(!result.is_interrupted());
    assert_eq!(result.final_output_as_str(), Some("outer-done"));
    assert_eq!(nested_invokes.load(Ordering::SeqCst), 1);
    nested_model.assert_complete();
    outer_model.assert_complete();
}
