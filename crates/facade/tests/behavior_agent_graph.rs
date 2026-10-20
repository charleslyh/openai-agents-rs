//! Rust-only (X-011): resuming checks that the agent graph is the one the run paused with.

use std::sync::Arc;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{Agent, AgentsError, FunctionTool, RunOptions, RunState, Runner};

fn model() -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("delete", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]))
}

fn tools(extra: bool) -> Vec<FunctionTool> {
    let mut tools = vec![
        FunctionTool::constant("read", "read", "r"),
        FunctionTool::constant("delete", "delete", "d").with_needs_resolution(true),
    ];
    if extra {
        tools.push(FunctionTool::constant("format_disk", "format", "f"));
    }
    tools
}

/// Pause with `tools(false)` and return the approved state as JSON.
async fn paused_state_json(model: Arc<ScriptedModel>) -> serde_json::Value {
    let agent = Agent::new("ops").model(model).tools(tools(false));
    let result = Runner::run(&agent, "go", RunOptions::new())
        .await
        .expect("run");
    let mut state = result.to_state().expect("state");
    state.approve(&result.interruptions[0], false);
    state.to_json()
}

#[tokio::test]
async fn the_same_graph_rebuilt_in_another_order_resumes() {
    let model = model();
    let json = paused_state_json(model.clone()).await;
    assert!(json["agent_graph"]
        .as_str()
        .is_some_and(|f| f.starts_with("fnv1a64:")));

    let mut reordered = tools(false);
    reordered.reverse();
    let rebuilt = Agent::new("ops").model(model).tools(reordered);
    let state = RunState::from_json("ops", json).expect("load");
    let result = Runner::run_state(&rebuilt, state, RunOptions::new())
        .await
        .expect("same graph resumes");
    assert_eq!(result.final_output_as_str(), Some("done"));
}

#[tokio::test]
async fn a_changed_graph_is_refused_unless_allowed() {
    let model = model();
    let json = paused_state_json(model.clone()).await;
    let changed = Agent::new("ops").model(model).tools(tools(true));

    let state = RunState::from_json("ops", json.clone()).expect("load");
    let err = Runner::run_state(&changed, state, RunOptions::new())
        .await
        .expect_err("a different graph must not resume silently");
    assert!(matches!(err, AgentsError::User(_)), "{err:?}");
    assert!(
        err.to_string().contains("allow_agent_graph_change"),
        "{err}"
    );

    let state = RunState::from_json("ops", json.clone()).expect("load");
    assert!(Runner::run_state_streamed(changed.clone(), state, RunOptions::new()).is_err());

    let mut state = RunState::from_json("ops", json).expect("load");
    state.allow_agent_graph_change();
    assert!(state.agent_graph().is_none());
    let result = Runner::run_state(&changed, state, RunOptions::new())
        .await
        .expect("allowed");
    assert_eq!(result.final_output_as_str(), Some("done"));
}

#[tokio::test]
async fn a_state_without_a_fingerprint_resumes_as_before() {
    let model = model();
    let mut json = paused_state_json(model.clone()).await;
    json.as_object_mut().expect("object").remove("agent_graph");
    let changed = Agent::new("ops").model(model).tools(tools(true));
    let state = RunState::from_json("ops", json).expect("load");
    let result = Runner::run_state(&changed, state, RunOptions::new())
        .await
        .expect("no fingerprint, no check");
    assert_eq!(result.final_output_as_str(), Some("done"));
}
