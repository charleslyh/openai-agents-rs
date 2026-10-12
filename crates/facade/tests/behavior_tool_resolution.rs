//! D-048: a paused tool call resolved with an output produced outside the run.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    Agent, AsToolConfig, FunctionTool, RunItem, RunOptions, RunState, Runner, ToolResolution,
};
use serde_json::{json, Value};

/// A tool that counts its invocations and pauses when `mode` is `"later"`.
fn deferrable(name: &str, invokes: &Arc<AtomicUsize>) -> FunctionTool {
    let invokes = Arc::clone(invokes);
    FunctionTool::new(
        name,
        "render",
        json!({
            "type": "object",
            "properties": { "mode": { "type": "string" } },
            "required": ["mode"],
            "additionalProperties": false
        }),
        move |_ctx, _args| {
            let invokes = Arc::clone(&invokes);
            async move {
                invokes.fetch_add(1, Ordering::SeqCst);
                Ok(Value::String("ran-in-run".into()))
            }
        },
    )
    .with_needs_resolution_fn(|params, _call_id| async move { params["mode"] == "later" })
}

fn output_sent_to_model(model: &ScriptedModel, call_id: &str) -> Option<Value> {
    let input = model.last_call()?.input;
    input.as_array()?.iter().find_map(|item| {
        (item["type"] == "function_call_output" && item["call_id"] == call_id)
            .then(|| item["output"].clone())
    })
}

#[tokio::test]
async fn provided_output_resumes_without_running_the_tool() {
    let invokes = Arc::new(AtomicUsize::new(0));
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(vec![
            ItemHelpers::function_tool_call("render", r#"{"mode":"now"}"#, "c0"),
            ItemHelpers::function_tool_call("render", r#"{"mode":"later"}"#, "c1"),
        ]),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![deferrable("render", &invokes)]);

    let paused = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    assert!(paused.is_interrupted());
    assert_eq!(paused.interruptions.len(), 1);
    assert_eq!(paused.interruptions[0].call_id, "c1");
    assert_eq!(invokes.load(Ordering::SeqCst), 1, "the `now` call ran");

    // The resolution survives a JSON round trip, as a host that resumes later needs.
    let mut state = paused.to_state().expect("state");
    state.provide_output(&paused.interruptions[0], json!("rendered elsewhere"));
    let state = RunState::from_string("assistant", &state.to_string()).expect("restore");
    assert_eq!(
        state.resolution_status("assistant", "render", "c1"),
        Some(ToolResolution::Output {
            output: json!("rendered elsewhere")
        })
    );

    let result = Runner::run_state(&agent, state, RunOptions::default())
        .await
        .expect("resume");
    assert_eq!(result.final_output_as_str(), Some("done"));
    assert_eq!(
        invokes.load(Ordering::SeqCst),
        1,
        "the paused call never ran"
    );
    assert_eq!(
        output_sent_to_model(&model, "c1"),
        Some(json!("rendered elsewhere"))
    );
    assert!(result.new_items.iter().any(|i| matches!(
        i,
        RunItem::ToolCallOutput(o)
            if o.raw_item["call_id"] == "c1" && o.output == json!("rendered elsewhere")
    )));
    model.assert_complete();
}

#[tokio::test]
async fn approvals_and_outputs_resolve_one_paused_batch() {
    let invokes = Arc::new(AtomicUsize::new(0));
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(vec![
            ItemHelpers::function_tool_call("render", r#"{"mode":"later"}"#, "c1"),
            ItemHelpers::function_tool_call("delete", "{}", "c2"),
        ]),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("assistant").model(model.clone()).tools(vec![
        deferrable("render", &invokes),
        FunctionTool::constant("delete", "d", "deleted").with_needs_approval(true),
    ]);

    let paused = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(paused.interruptions.len(), 2);
    let mut state = paused.to_state().expect("state");
    for item in &paused.interruptions {
        match item.call_id.as_str() {
            "c1" => state.provide_output(item, json!("rendered")),
            _ => state.approve(item, false),
        }
    }

    let result = Runner::run_state(&agent, state, RunOptions::default())
        .await
        .expect("resume");
    assert_eq!(result.final_output_as_str(), Some("done"));
    assert_eq!(invokes.load(Ordering::SeqCst), 0);
    assert_eq!(output_sent_to_model(&model, "c1"), Some(json!("rendered")));
    assert_eq!(output_sent_to_model(&model, "c2"), Some(json!("deleted")));
}

#[tokio::test]
async fn partly_resolved_batch_pauses_again_for_the_rest() {
    let invokes = Arc::new(AtomicUsize::new(0));
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(vec![
            ItemHelpers::function_tool_call("render", r#"{"mode":"later"}"#, "c1"),
            ItemHelpers::function_tool_call("render", r#"{"mode":"later"}"#, "c2"),
        ]),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![deferrable("render", &invokes)]);

    let paused = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    let mut state = paused.to_state().expect("state");
    state.provide_output(&paused.interruptions[0], json!("first"));
    let paused = Runner::run_state(&agent, state, RunOptions::default())
        .await
        .expect("resume");
    assert_eq!(paused.interruptions.len(), 1);
    assert_eq!(paused.interruptions[0].call_id, "c2");

    let mut state = paused.to_state().expect("state");
    state.provide_output(&paused.interruptions[0], json!("second"));
    let result = Runner::run_state(&agent, state, RunOptions::default())
        .await
        .expect("resume");
    assert_eq!(result.final_output_as_str(), Some("done"));
    assert_eq!(invokes.load(Ordering::SeqCst), 0);
    assert_eq!(output_sent_to_model(&model, "c1"), Some(json!("first")));
    assert_eq!(output_sent_to_model(&model, "c2"), Some(json!("second")));
}

#[tokio::test]
async fn nested_agent_tool_call_takes_a_provided_output() {
    let invokes = Arc::new(AtomicUsize::new(0));
    let nested_model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(
            "render",
            r#"{"mode":"later"}"#,
            "nested-1",
        )),
        ModelStep::from(ItemHelpers::text_message("nested-done")),
    ]));
    let helper = Agent::new("helper")
        .model(nested_model.clone())
        .tools(vec![deferrable("render", &invokes)]);
    let outer_model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(
            "call_helper",
            r#"{"input":"render it"}"#,
            "outer-1",
        )),
        ModelStep::from(ItemHelpers::text_message("outer-done")),
    ]));
    let outer = Agent::new("boss")
        .model(outer_model.clone())
        .tools(vec![helper.as_tool(AsToolConfig {
            name: Some("call_helper".into()),
            description: None,
            needs_resolution: false,
            max_turns: Some(5),
        })]);

    let paused = Runner::run(&outer, "delegate", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(paused.interruptions[0].agent_name, "helper");
    let mut state = paused.to_state().expect("state");
    state.provide_output(&paused.interruptions[0], json!("rendered"));

    let result = Runner::run_state(&outer, state, RunOptions::default())
        .await
        .expect("resume");
    assert_eq!(result.final_output_as_str(), Some("outer-done"));
    assert_eq!(invokes.load(Ordering::SeqCst), 0);
    assert_eq!(
        output_sent_to_model(&nested_model, "nested-1"),
        Some(json!("rendered"))
    );
    nested_model.assert_complete();
    outer_model.assert_complete();
}

/// `/3` snapshots kept the resolutions under `approvals`; they still load and resume.
#[tokio::test]
async fn schema_3_approvals_still_resume() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("delete", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![
            FunctionTool::constant("delete", "d", "deleted").with_needs_approval(true)
        ]);
    let paused = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    let mut state = paused.to_state().expect("state");
    state.approve(&paused.interruptions[0], false);

    let mut legacy = state.to_json();
    let resolutions = legacy
        .as_object_mut()
        .unwrap()
        .remove("resolutions")
        .unwrap();
    legacy["approvals"] = resolutions;
    legacy["$schemaVersion"] = json!("openai-agents-rs/3");
    for item in legacy["generated_items"].as_array_mut().unwrap() {
        if item["kind"] == "tool_interruption" {
            item["kind"] = json!("tool_approval");
        }
    }
    let state = RunState::from_json("assistant", legacy).expect("load /3");

    let result = Runner::run_state(&agent, state, RunOptions::default())
        .await
        .expect("resume");
    assert_eq!(result.final_output_as_str(), Some("done"));
    assert_eq!(output_sent_to_model(&model, "c1"), Some(json!("deleted")));
}
