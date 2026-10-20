//! Rust-only (X-021): a deferred capability stays out of the prompt until the model loads it.

use std::sync::Arc;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{
    Agent, AgentsError, CapabilityParts, FunctionTool, GuardrailFunctionOutput, ModelError,
    RunOptions, RunState, Runner, LOAD_CAPABILITY_TOOL,
};
use serde_json::json;

fn db() -> CapabilityParts {
    CapabilityParts::new()
        .tool(FunctionTool::constant("query", "Run SQL", "rows"))
        .instructions("Use `query` for SQL.")
        .deferred("run SQL queries")
}

fn agent(model: Arc<ScriptedModel>) -> Agent {
    Agent::new("ops")
        .model(model)
        .instructions("Be brief.")
        .tools(vec![FunctionTool::constant("ping", "Ping", "pong")])
        .capability(db().named("db"))
}

fn load(name: &str, call_id: &str) -> opai_agents::ResponseOutputItem {
    ItemHelpers::function_tool_call(
        LOAD_CAPABILITY_TOOL,
        json!({"name": name}).to_string(),
        call_id,
    )
}

fn text(s: &str) -> ModelStep {
    ModelStep::from(ItemHelpers::text_message(s))
}

#[tokio::test]
async fn an_unloaded_capability_costs_one_catalog_line() {
    let model = Arc::new(ScriptedModel::new([text("ok")]));
    Runner::run(&agent(model.clone()), "go", RunOptions::default())
        .await
        .expect("run");

    let call = model.last_call().expect("called");
    assert_eq!(call.tool_names, ["ping", LOAD_CAPABILITY_TOOL]);
    let instructions = call.system_instructions.expect("instructions");
    assert!(instructions.starts_with("Be brief."), "{instructions}");
    assert!(
        instructions.contains("- db: run SQL queries"),
        "{instructions}"
    );
    assert!(!instructions.contains("Use `query`"), "{instructions}");
}

#[tokio::test]
async fn loading_makes_the_tools_and_instructions_visible() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(load("db", "c1")),
        ModelStep::from(ItemHelpers::function_tool_call("query", "{}", "c2")),
        text("done"),
    ]));
    let result = Runner::run(&agent(model.clone()), "go", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("done"));

    let calls = model.calls();
    assert_eq!(calls[0].tool_names, ["ping", LOAD_CAPABILITY_TOOL]);
    // After loading: the capability's tool, its paragraph, no catalog and nothing left to load.
    assert_eq!(calls[1].tool_names, ["ping", "query"]);
    let instructions = calls[1].system_instructions.clone().expect("instructions");
    assert!(
        instructions.contains("Use `query` for SQL."),
        "{instructions}"
    );
    assert!(!instructions.contains("load on demand"), "{instructions}");
    assert!(calls[1].input.to_string().contains("Loaded `db`"));
    assert!(calls[2].input.to_string().contains("rows"));
}

#[tokio::test]
async fn a_tool_that_is_not_loaded_cannot_be_called() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call("query", "{}", "c1"),
    )]));
    let err = Runner::run(&agent(model), "go", RunOptions::default())
        .await
        .expect_err("`query` is not offered before it is loaded");
    assert!(
        matches!(&err, AgentsError::Model(ModelError::Behavior(m)) if m.contains("query")),
        "{err:?}"
    );
}

#[tokio::test]
async fn loading_twice_or_loading_a_stranger_changes_nothing() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(vec![load("db", "c1"), load("db", "c2"), load("nope", "c3")]),
        text("done"),
    ]));
    Runner::run(&agent(model.clone()), "go", RunOptions::default())
        .await
        .expect("run");

    let second = &model.calls()[1];
    assert_eq!(second.tool_names, ["ping", "query"], "no duplicate tool");
    let input = second.input.to_string();
    assert_eq!(input.matches("Loaded `db`").count(), 1, "{input}");
    assert!(input.contains("`db` is already loaded"), "{input}");
    assert!(input.contains("No loadable capability"), "{input}");
    let instructions = second.system_instructions.clone().expect("instructions");
    assert_eq!(instructions.matches("Use `query`").count(), 1);
}

#[tokio::test]
async fn loading_is_per_run() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(load("db", "c1")),
        text("one"),
        text("two"),
    ]));
    let agent = agent(model.clone());
    Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("first");
    Runner::run(&agent, "again", RunOptions::default())
        .await
        .expect("second");

    let calls = model.calls();
    assert_eq!(calls[1].tool_names, ["ping", "query"]);
    assert_eq!(calls[2].tool_names, ["ping", LOAD_CAPABILITY_TOOL]);
}

#[tokio::test]
async fn a_paused_run_remembers_what_was_loaded() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(vec![
            load("db", "c1"),
            ItemHelpers::function_tool_call("delete", "{}", "c2"),
        ]),
        text("done"),
    ]));
    let build = |model: Arc<ScriptedModel>| {
        agent(model).tools(vec![
            FunctionTool::constant("ping", "Ping", "pong"),
            FunctionTool::constant("delete", "Delete", "d").with_needs_resolution(true),
        ])
    };
    let paused = Runner::run(&build(model.clone()), "go", RunOptions::default())
        .await
        .expect("run");
    let mut state = paused.to_state().expect("paused");
    state.approve(&paused.interruptions[0], false);
    let saved = state.to_json();
    assert_eq!(
        saved["loaded_capabilities"],
        json!([{"agent": "ops", "capability": "db"}])
    );

    // A fresh process: new agent value, state read back from JSON.
    let state = RunState::from_json("ops", saved).expect("load");
    Runner::run_state(&build(model.clone()), state, RunOptions::default())
        .await
        .expect("resume");
    let resumed = model.last_call().expect("called");
    assert!(
        resumed.tool_names.contains(&"query".to_owned()),
        "{:?}",
        resumed.tool_names
    );
    assert!(!resumed
        .tool_names
        .contains(&LOAD_CAPABILITY_TOOL.to_owned()));
}

#[tokio::test]
async fn an_older_state_without_the_field_loads_with_nothing_loaded() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("delete", "{}", "c1")),
        text("done"),
    ]));
    let build = |model: Arc<ScriptedModel>| {
        agent(model).tools(vec![
            FunctionTool::constant("delete", "Delete", "d").with_needs_resolution(true)
        ])
    };
    let paused = Runner::run(&build(model.clone()), "go", RunOptions::default())
        .await
        .expect("run");
    let mut state = paused.to_state().expect("paused");
    state.approve(&paused.interruptions[0], false);
    let mut saved = state.to_json();
    saved
        .as_object_mut()
        .expect("object")
        .remove("loaded_capabilities");

    let state = RunState::from_json("ops", saved).expect("load");
    Runner::run_state(&build(model.clone()), state, RunOptions::default())
        .await
        .expect("resume");
    let resumed = model.last_call().expect("called");
    assert_eq!(resumed.tool_names, ["delete", LOAD_CAPABILITY_TOOL]);
}

#[tokio::test]
async fn guardrails_of_an_unloaded_capability_still_apply() {
    let blocker = opai_agents::input_guardrail("block", |_ctx, _agent, _input| async {
        GuardrailFunctionOutput::trip(json!({"reason": "blocked"}))
    });
    let model = Arc::new(ScriptedModel::new([text("ok")]));
    let agent = Agent::new("a").model(model).capability(
        CapabilityParts::new()
            .input_guardrail(blocker)
            .deferred("checks")
            .named("safe"),
    );
    let err = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect_err("tripwire");
    assert!(
        matches!(err, AgentsError::InputGuardrailTripwire(_)),
        "{err}"
    );
}
