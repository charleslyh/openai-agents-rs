//! ScriptedModel behavior tests (verification layer 1).

use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    handoff, Agent, AgentsError, FunctionTool, MaxTurnsExceeded, ModelError, ModelSettings,
    RunItem, RunOptions, Runner, ToolChoice, ToolUseBehavior, DEFAULT_MAX_TURNS,
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

// --- Phase-2 parity fixes ---------------------------------------------------

/// B1: `last_agent` must be the agent that actually finished, not the starting agent.
#[tokio::test]
async fn last_agent_is_the_agent_that_finished() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(
            "transfer_to_spanish",
            "{}",
            "c1",
        )),
        ModelStep::from(ItemHelpers::text_message("hola")),
    ]));
    let spanish = Agent::new("Spanish").model(model.clone());
    let triage = Agent::new("Triage")
        .model(model.clone())
        .handoffs(vec![handoff(spanish)]);
    let result = Runner::run(&triage, "hi", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.last_agent().name, "Spanish");
    assert_eq!(result.last_agent_name, "Spanish");
}

/// B4: `reset_tool_choice` clears `tool_choice` from the turn *after* the agent used tools.
#[tokio::test]
async fn tool_choice_is_reset_on_the_following_turn() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![FunctionTool::constant("echo", "echo", "ok")])
        .model_settings(ModelSettings {
            tool_choice: Some("required".into()),
            ..Default::default()
        });
    let result = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("done"));

    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        calls[0].model_settings.tool_choice,
        Some(ToolChoice::Required)
    );
    assert_eq!(calls[1].model_settings.tool_choice, None);
}

/// B4 (regression): the reset must also apply after a turn that ended early via
/// `stop_on_first_tool` / `StopAtTools`, which Python handles at turn start.
#[tokio::test]
async fn reset_tool_choice_defaults_to_true() {
    let agent = Agent::new("assistant");
    assert!(agent.reset_tool_choice);
}

/// B5: `stop_on_first_tool` still records the output of every tool in the batch.
#[tokio::test]
async fn stop_on_first_tool_records_every_tool_output() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(vec![
        ItemHelpers::function_tool_call("first", "{}", "c1"),
        ItemHelpers::function_tool_call("second", "{}", "c2"),
    ])]));
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![
            FunctionTool::constant("first", "first", "out-1"),
            FunctionTool::constant("second", "second", "out-2"),
        ])
        .tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
    let result = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");

    assert_eq!(result.final_output_as_str(), Some("out-1"));
    let outputs: Vec<String> = result
        .new_items
        .iter()
        .filter_map(|i| match i {
            RunItem::ToolCallOutput(o) => o.output.as_str().map(str::to_string),
            _ => None,
        })
        .collect();
    assert_eq!(outputs, vec!["out-1".to_string(), "out-2".to_string()]);
}

/// B6: a `function_call` without a `call_id` is a model behavior error, not a silent default.
#[tokio::test]
async fn function_call_without_call_id_is_a_behavior_error() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(serde_json::json!({
        "type": "function_call",
        "name": "echo",
        "arguments": "{}"
    }))]));
    let agent = Agent::new("assistant")
        .model(model)
        .tools(vec![FunctionTool::constant("echo", "echo", "ok")]);
    let err = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect_err("malformed function_call must fail");
    match err {
        AgentsError::Model(ModelError::Behavior(msg)) => {
            assert!(msg.contains("call_id"), "unexpected message: {msg}");
        }
        other => panic!("unexpected error: {other}"),
    }
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
