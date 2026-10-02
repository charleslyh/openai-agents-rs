//! ScriptedModel behavior tests (verification layer 1).

use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    Agent, FunctionTool, MaxTurnsExceeded, RunOptions, Runner, ToolUseBehavior, DEFAULT_MAX_TURNS,
};

#[tokio::test]
async fn plain_text_final_output() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("hello"),
    )]));
    let agent = Agent::new("assistant").model(model.clone());
    let result = Runner::run(&agent, "hi", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("hello"));
    assert_eq!(result.last_agent_name, "assistant");
    assert_eq!(result.raw_responses.len(), 1);
    model.assert_complete();
    assert_eq!(model.calls().len(), 1);
}

#[tokio::test]
async fn tool_then_text() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "call-1")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let tool = FunctionTool::constant("echo", "echo tool", "tool-ok");
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![tool]);
    let result = Runner::run(&agent, "use tool", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("done"));
    assert_eq!(result.raw_responses.len(), 2);
    assert!(result
        .new_items
        .iter()
        .any(|i| matches!(i, openai_agents::RunItem::ToolCallOutput(_))));
    model.assert_complete();
    assert_eq!(model.calls().len(), 2);
    assert!(model.calls()[0].tool_names.iter().any(|n| n == "echo"));
}

#[tokio::test]
async fn stop_on_first_tool() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call("echo", "{}", "call-1"),
    )]));
    let tool = FunctionTool::constant("echo", "echo tool", "tool-final");
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![tool])
        .tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
    let result = Runner::run(&agent, "use tool", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("tool-final"));
    assert_eq!(result.raw_responses.len(), 1);
    model.assert_complete();
}

#[tokio::test]
async fn max_turns_exceeded() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c1")),
        ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c2")),
        ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c3")),
    ]));
    let tool = FunctionTool::constant("echo", "echo", "ok");
    let agent = Agent::new("assistant")
        .model(model)
        .tools(vec![tool]);
    let mut opts = RunOptions::default();
    opts.max_turns = Some(2);
    let err = Runner::run(&agent, "loop", opts).await.expect_err("should exceed");
    match err {
        openai_agents::AgentsError::MaxTurns(MaxTurnsExceeded { max_turns }) => {
            assert_eq!(max_turns, 2);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[tokio::test]
async fn default_max_turns_constant() {
    assert_eq!(DEFAULT_MAX_TURNS, 10);
}

#[tokio::test]
async fn stop_at_named_tool() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call("finish", "{}", "call-1"),
    )]));
    let tool = FunctionTool::constant("finish", "finish", "stopped");
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![tool])
        .tool_use_behavior(ToolUseBehavior::StopAtTools {
            stop_at_tool_names: vec!["finish".into()],
        });
    let result = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("stopped"));
    model.assert_complete();
}

#[tokio::test]
async fn to_input_list_includes_new_items() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("bye"),
    )]));
    let agent = Agent::new("assistant").model(model);
    let result = Runner::run(&agent, "hi", RunOptions::default())
        .await
        .expect("run");
    let list = result.to_input_list();
    assert!(list.len() >= 2);
}
