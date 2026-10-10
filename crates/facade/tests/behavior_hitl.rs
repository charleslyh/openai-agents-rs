//! Human-in-the-loop / tool approval behavior tests.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    Agent, FunctionTool, RunOptions, RunState, Runner, DEFAULT_APPROVAL_REJECTION_MESSAGE,
    RUN_STATE_SCHEMA_VERSION, SUPPORTED_RUN_STATE_SCHEMAS,
};

/// D-C: Python executes the calls that do not need approval before pausing, so a mixed batch
/// must report the pending approval *and* the sibling tool output.
#[tokio::test]
async fn mixed_batch_executes_unapproved_tools_before_pausing() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(vec![
            ItemHelpers::function_tool_call("safe_read", "{}", "c1"),
            ItemHelpers::function_tool_call("delete_file", "{}", "c2"),
        ]),
        ModelStep::from(ItemHelpers::text_message("resumed")),
    ]));
    let agent = Agent::new("assistant").model(model).tools(vec![
        FunctionTool::constant("safe_read", "read", "read-ok"),
        FunctionTool::constant("delete_file", "delete", "deleted").with_needs_approval(true),
    ]);

    let result = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    assert!(result.is_interrupted());
    assert_eq!(result.interruptions.len(), 1);
    assert_eq!(result.interruptions[0].tool_name, "delete_file");
    // The sibling tool still ran in the same turn.
    assert!(result.new_items.iter().any(|i| matches!(
        i,
        openai_agents::RunItem::ToolCallOutput(o)
            if o.raw_item.get("call_id").and_then(|c| c.as_str()) == Some("c1")
    )));
}

/// RunState schema v2 is current, and v1 payloads remain readable.
#[tokio::test]
async fn run_state_json_accepts_previous_schema_version() {
    assert_eq!(
        SUPPORTED_RUN_STATE_SCHEMAS.last().copied(),
        Some(RUN_STATE_SCHEMA_VERSION)
    );

    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("delete_file", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("assistant")
        .model(model)
        .tools(vec![FunctionTool::constant(
            "delete_file",
            "delete",
            "deleted",
        )
        .with_needs_approval(true)]);
    let result = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");

    let mut value = result.to_state().expect("to_state").to_json();
    assert_eq!(value["$schemaVersion"], RUN_STATE_SCHEMA_VERSION);
    value["$schemaVersion"] = serde_json::json!("openai-agents-rust/1");
    let restored =
        RunState::from_json("assistant", value.clone()).expect("v1 payload must still load");
    assert_eq!(restored.interruptions.len(), 1);

    // Ids emitted before the `openai-agents-rust` → `openai-agents-rs` rename stay loadable.
    value["$schemaVersion"] = serde_json::json!("openai-agents-rust/2");
    let restored =
        RunState::from_json("assistant", value.clone()).expect("legacy v2 payload must still load");
    assert_eq!(restored.interruptions.len(), 1);

    value["$schemaVersion"] = serde_json::json!("openai-agents-rs/99");
    assert!(RunState::from_json("assistant", value.clone()).is_err());

    // A `/2` payload is the previous shape and stays loadable.
    value["$schemaVersion"] = serde_json::json!("openai-agents-rs/2");
    let restored = RunState::from_json("assistant", value).expect("v2 payload must still load");
    assert_eq!(restored.interruptions.len(), 1);
}

/// D-042: an unlimited run pauses as `max_turns: null` and resumes unlimited. A `/2` snapshot
/// always carried a number, which loads as `Some(n)`.
#[tokio::test]
async fn unlimited_run_state_round_trip_keeps_max_turns_null() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("delete_file", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("assistant")
        .model(model)
        .tools(vec![FunctionTool::constant(
            "delete_file",
            "delete",
            "deleted",
        )
        .with_needs_approval(true)]);
    let mut opts = RunOptions::default();
    opts.max_turns = None;
    let paused = Runner::run(&agent, "go", opts).await.expect("run");
    assert!(paused.is_interrupted());

    let json = paused.to_state().expect("to_state").to_json();
    assert_eq!(json["$schemaVersion"], RUN_STATE_SCHEMA_VERSION);
    assert!(
        json["max_turns"].is_null(),
        "max_turns: {:?}",
        json["max_turns"]
    );

    let restored = RunState::from_json("assistant", json).expect("restore");
    assert_eq!(restored.max_turns, None);

    // A `/2`-era number still means "that many turns".
    let mut legacy = paused.to_state().expect("to_state").to_json();
    legacy["$schemaVersion"] = serde_json::json!("openai-agents-rs/2");
    legacy["max_turns"] = serde_json::json!(4);
    let restored = RunState::from_json("assistant", legacy).expect("restore");
    assert_eq!(restored.max_turns, Some(4));
}

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
    assert!(model.calls()[1].input.to_string().contains("User denied"));
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
    let mut state =
        openai_agents::RunState::from_string(&outer.name, &state.to_string()).expect("from_string");
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

/// D-027: a run paused for approval has saved the turns before the paused one; resuming with the
/// same session saves the rest exactly once. The final history equals Python's (checked there:
/// user, call, output, call, output, message); Python also stores the pending call while paused,
/// which this SDK does not.
#[tokio::test]
async fn paused_run_saves_each_item_once_across_resume() {
    use openai_agents::{InMemorySession, Session};
    let kinds = |items: Vec<serde_json::Value>| -> Vec<String> {
        items
            .iter()
            .map(|i| {
                i["type"]
                    .as_str()
                    .or(i["role"].as_str())
                    .unwrap()
                    .to_string()
            })
            .collect()
    };
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c1")),
        ModelStep::from(ItemHelpers::function_tool_call("del", "{}", "c2")),
        ModelStep::from(ItemHelpers::text_message("ok")),
    ]));
    let agent = Agent::new("a").model(model).tools(vec![
        FunctionTool::constant("echo", "e", "x"),
        FunctionTool::constant("del", "d", "deleted").with_needs_approval(true),
    ]);
    let session = InMemorySession::shared("conv");
    let options = || {
        let mut options = RunOptions::default();
        options.session = Some(session.clone());
        options
    };

    let paused = Runner::run(&agent, "go", options()).await.expect("run");
    assert!(paused.is_interrupted());
    assert_eq!(
        kinds(session.get_items(None).await.unwrap()),
        ["user", "function_call", "function_call_output"],
        "the finished turn is saved, the paused one is not"
    );

    // The state survives a JSON round trip, including how much was already saved.
    let json = paused.to_state().expect("state").to_json();
    let mut state = openai_agents::RunState::from_json("a", json).expect("from_json");
    state.approve(&paused.interruptions[0], false);
    let resumed = Runner::run_state(&agent, state, options())
        .await
        .expect("resume");
    assert_eq!(resumed.final_output_as_str(), Some("ok"));
    assert_eq!(
        kinds(session.get_items(None).await.unwrap()),
        [
            "user",
            "function_call",
            "function_call_output",
            "function_call",
            "function_call_output",
            "message"
        ]
    );
}

/// D-029: guardrail results gathered before a pause are part of the resumed run's result, also
/// after the state went through JSON.
#[tokio::test]
async fn guardrail_results_survive_a_pause_and_resume() {
    use openai_agents::{
        GuardrailFunctionOutput, InputGuardrail, ToolGuardrailFunctionOutput, ToolOutputGuardrail,
    };
    let seen = ToolOutputGuardrail::new("seen", |_data| async {
        ToolGuardrailFunctionOutput::allow(serde_json::json!({"ok": true}))
    });
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c1")),
        ModelStep::from(ItemHelpers::function_tool_call("del", "{}", "c2")),
        ModelStep::from(ItemHelpers::text_message("ok")),
    ]));
    let agent = Agent::new("a")
        .model(model)
        .input_guardrails(vec![InputGuardrail::new("in", |_c, _a, _i| async {
            GuardrailFunctionOutput::pass(serde_json::json!("clean"))
        })])
        .tools(vec![
            FunctionTool::constant("echo", "e", "x").with_tool_output_guardrails(vec![seen]),
            FunctionTool::constant("del", "d", "deleted").with_needs_approval(true),
        ]);

    let paused = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    assert!(paused.is_interrupted());
    assert_eq!(
        paused.input_guardrail_results.len(),
        1,
        "available on the paused result too"
    );
    assert_eq!(paused.tool_output_guardrail_results.len(), 1);

    let json = paused.to_state().expect("state").to_json();
    let mut state = openai_agents::RunState::from_json("a", json).expect("from_json");
    state.approve(&paused.interruptions[0], false);
    let resumed = Runner::run_state(&agent, state, RunOptions::default())
        .await
        .expect("resume");
    assert_eq!(resumed.input_guardrail_results.len(), 1);
    assert_eq!(
        resumed.input_guardrail_results[0].output.output_info,
        serde_json::json!("clean")
    );
    assert_eq!(resumed.tool_output_guardrail_results.len(), 1);
    assert_eq!(
        resumed.tool_output_guardrail_results[0].guardrail_name,
        "seen"
    );
    assert_eq!(
        resumed.tool_output_guardrail_results[0].output.output_info,
        serde_json::json!({"ok": true})
    );
}
