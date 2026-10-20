//! Rust-only: a tool's declared `Tool::requires` types are checked before the first model turn
//! (see `docs/DEVIATIONS.md` X-003 and `docs/refactor/09-d1-context-generics.md`).

use std::sync::Arc;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{Agent, AgentsError, FunctionTool, RunOptions, Runner};

struct Db(&'static str);

fn tool_needing_db() -> FunctionTool {
    FunctionTool::new(
        "lookup",
        "reads from the database",
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }),
        |ctx, _args| async move {
            let db: &Db = ctx.require::<Db>()?;
            Ok(serde_json::json!(db.0))
        },
    )
    .require::<Db>()
}

#[tokio::test]
async fn a_declared_but_missing_extension_fails_before_the_first_turn() {
    let model = Arc::new(ScriptedModel::new([]));
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![tool_needing_db()]);

    let err = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect_err("must fail before the model is called");
    let message = err.to_string();
    assert!(matches!(err, AgentsError::User(_)), "{message}");
    assert!(message.contains("lookup"), "{message}");
    assert!(message.contains("assistant"), "{message}");
    assert!(message.contains("Db"), "{message}");
    assert!(message.contains("RunOptions::with"), "{message}");
    // Nothing reached the model: the check runs before the first turn.
    model.assert_complete();
}

#[tokio::test]
async fn a_declared_extension_is_found_and_the_tool_runs() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("lookup", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("assistant")
        .model(model.clone())
        .tools(vec![tool_needing_db()]);

    let result = Runner::run(&agent, "go", RunOptions::default().with(Db("main")))
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("done"));
}

#[tokio::test]
async fn requirements_are_checked_on_handoff_targets_too() {
    let handoff_model = Arc::new(ScriptedModel::new([]));
    let target = Agent::new("specialist")
        .model(handoff_model)
        .tools(vec![tool_needing_db()]);
    let agent = Agent::new("assistant").handoffs(vec![opai_agents::handoff(target)]);

    let err = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect_err("must fail");
    let message = err.to_string();
    assert!(message.contains("specialist"), "{message}");
    assert!(message.contains("lookup"), "{message}");
}

#[tokio::test]
async fn a_tool_without_declared_requirements_is_unaffected() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("ping", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("pong")),
    ]));
    let agent = Agent::new("assistant")
        .model(model)
        .tools(vec![FunctionTool::constant("ping", "d", "pong")]);

    let result = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("pong"));
}
