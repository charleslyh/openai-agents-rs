//! Rust-only (X-010): a resolution given through `RunState` is bound to the agent, tool and
//! arguments of the paused call it answered.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{Agent, FunctionTool, RunOptions, RunState, Runner};
use serde_json::Value;

fn deploy_tool(runs: &Arc<AtomicUsize>) -> FunctionTool {
    let runs = Arc::clone(runs);
    FunctionTool::new(
        "deploy",
        "deploy",
        serde_json::json!({
            "type": "object",
            "properties": {"env": {"type": "string"}},
            "required": ["env"],
            "additionalProperties": false
        }),
        move |_ctx, _args| {
            let runs = Arc::clone(&runs);
            async move {
                runs.fetch_add(1, Ordering::SeqCst);
                Ok(Value::String("deployed".into()))
            }
        },
    )
    .with_needs_resolution(true)
}

fn agent(runs: &Arc<AtomicUsize>, model: Arc<ScriptedModel>) -> Agent {
    Agent::new("ops")
        .model(model)
        .tools(vec![deploy_tool(runs)])
}

fn script() -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(
            "deploy",
            r#"{"env":"staging"}"#,
            "c1",
        )),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]))
}

/// Rewrite the `arguments` of every object with `call_id == id` under `value`.
fn rewrite_arguments(value: &mut Value, id: &str, arguments: &str) {
    match value {
        Value::Object(map) => {
            if map.get("call_id").and_then(Value::as_str) == Some(id)
                && map.contains_key("arguments")
            {
                map.insert("arguments".into(), Value::String(arguments.into()));
            }
            for child in map.values_mut() {
                rewrite_arguments(child, id, arguments);
            }
        }
        Value::Array(items) => items
            .iter_mut()
            .for_each(|child| rewrite_arguments(child, id, arguments)),
        _ => {}
    }
}

async fn paused_and_approved(runs: &Arc<AtomicUsize>) -> (Agent, Value) {
    let agent = agent(runs, script());
    let result = Runner::run(&agent, "ship", RunOptions::new())
        .await
        .expect("run");
    assert!(result.is_interrupted());
    let mut state = result.to_state().expect("state");
    state.approve(&result.interruptions[0], false);
    let json = state.to_json();
    (agent, json)
}

#[tokio::test]
async fn an_approval_runs_the_call_it_was_given_for() {
    let runs = Arc::new(AtomicUsize::new(0));
    let (agent, json) = paused_and_approved(&runs).await;
    assert_eq!(
        json["resolutions"]["by_call"]["c1"]["bound"]["arguments"],
        r#"{"env":"staging"}"#
    );
    let state = RunState::from_json("ops", json).expect("load");
    let result = Runner::run_state(&agent, state, RunOptions::new())
        .await
        .expect("resume");
    assert_eq!(result.final_output_as_str(), Some("done"));
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

/// Arguments edited after the decision (in a stored state, by a middleman, ...) do not inherit
/// the approval: the call pauses again, showing the new arguments, and the tool does not run.
#[tokio::test]
async fn edited_arguments_pause_again_instead_of_running() {
    let runs = Arc::new(AtomicUsize::new(0));
    let (agent, mut json) = paused_and_approved(&runs).await;
    rewrite_arguments(&mut json["pending_response"], "c1", r#"{"env":"prod"}"#);
    rewrite_arguments(&mut json["interruptions"], "c1", r#"{"env":"prod"}"#);

    let state = RunState::from_json("ops", json).expect("load");
    let result = Runner::run_state(&agent, state, RunOptions::new())
        .await
        .expect("resume");
    assert!(
        result.is_interrupted(),
        "the edited call needs a new decision"
    );
    assert_eq!(result.interruptions[0].arguments, r#"{"env":"prod"}"#);
    assert_eq!(runs.load(Ordering::SeqCst), 0);
}

/// States written before bindings existed (no `bound`) keep working.
#[tokio::test]
async fn an_unbound_resolution_from_an_older_state_still_applies() {
    let runs = Arc::new(AtomicUsize::new(0));
    let (agent, mut json) = paused_and_approved(&runs).await;
    json["resolutions"]["by_call"]["c1"]
        .as_object_mut()
        .expect("entry")
        .remove("bound");
    let state = RunState::from_json("ops", json).expect("load");
    let result = Runner::run_state(&agent, state, RunOptions::new())
        .await
        .expect("resume");
    assert_eq!(result.final_output_as_str(), Some("done"));
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_malformed_binding_is_rejected_on_load() {
    let runs = Arc::new(AtomicUsize::new(0));
    let (_agent, mut json) = paused_and_approved(&runs).await;
    json["resolutions"]["by_call"]["c1"]["bound"] = serde_json::json!({"agent": "ops"});
    let err = RunState::from_json("ops", json).expect_err("must fail");
    assert!(err.to_string().contains("bound"), "{err}");
}
