//! Rust-only (X-007): a mounted `Capability` behaves exactly like handing the agent the same
//! parts directly, whatever order the builder calls come in.

use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    input_guardrail, Agent, AgentsError, CapabilityParts, FunctionTool, GuardrailFunctionOutput,
    ModelSettings, RunOptions, Runner,
};

fn scripted() -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]))
}

fn clock_parts() -> CapabilityParts {
    CapabilityParts::new()
        .tool(FunctionTool::constant("now", "Current time", "12:00"))
        .instructions("Use `now` for the time.")
        .model_settings(ModelSettings {
            temperature: Some(0.1),
            max_tokens: Some(64),
            ..Default::default()
        })
}

#[tokio::test]
async fn a_capability_equals_its_parts_given_by_hand() {
    let mounted_model = scripted();
    let mounted = Agent::new("a")
        .model(mounted_model.clone())
        .capability(clock_parts().named("clock"))
        // Set after mounting on purpose: the capability's parts are kept apart, not overwritten.
        .instructions("Be brief.")
        .tools(vec![FunctionTool::constant("ping", "Ping", "pong")])
        .model_settings(ModelSettings {
            temperature: Some(0.7),
            ..Default::default()
        });
    Runner::run(&mounted, "go", RunOptions::default())
        .await
        .expect("run");

    let manual_model = scripted();
    let manual = Agent::new("a")
        .model(manual_model.clone())
        .instructions("Be brief.\n\nUse `now` for the time.")
        .tools(vec![
            FunctionTool::constant("ping", "Ping", "pong"),
            FunctionTool::constant("now", "Current time", "12:00"),
        ])
        .model_settings(ModelSettings {
            temperature: Some(0.7),
            max_tokens: Some(64),
            ..Default::default()
        });
    Runner::run(&manual, "go", RunOptions::default())
        .await
        .expect("run");

    let a = mounted_model.last_call().expect("called");
    let b = manual_model.last_call().expect("called");
    assert_eq!(a.system_instructions, b.system_instructions);
    assert_eq!(a.tool_names, b.tool_names);
    assert_eq!(a.model_settings, b.model_settings);
    // The agent's own temperature wins over the capability default; its max_tokens shows through.
    assert_eq!(a.model_settings.temperature, Some(0.7));
    assert_eq!(a.model_settings.max_tokens, Some(64));
}

#[tokio::test]
async fn run_settings_win_over_capability_defaults() {
    let model = scripted();
    let agent = Agent::new("a")
        .model(model.clone())
        .capability(clock_parts().named("clock"));
    Runner::run(&agent, "go", RunOptions::new().temperature(0.9))
        .await
        .expect("run");
    let call = model.last_call().expect("called");
    assert_eq!(call.model_settings.temperature, Some(0.9));
    assert_eq!(call.model_settings.max_tokens, Some(64));
    assert_eq!(
        call.system_instructions.as_deref(),
        Some("Use `now` for the time.")
    );
}

#[tokio::test]
async fn capability_guardrails_run_with_the_agents_own() {
    let blocker = input_guardrail("block", |_ctx, _agent, _input| async {
        GuardrailFunctionOutput::trip(serde_json::json!({"reason": "blocked"}))
    });
    let agent = Agent::new("a").model(scripted()).capability(
        CapabilityParts::new()
            .input_guardrail(blocker)
            .named("safe-input"),
    );
    let err = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect_err("tripwire");
    assert!(
        matches!(err, AgentsError::InputGuardrailTripwire(_)),
        "{err}"
    );
}

struct Store;

#[tokio::test]
async fn a_capability_requirement_is_checked_before_the_first_turn() {
    let model = Arc::new(ScriptedModel::new([]));
    let agent = Agent::new("a")
        .model(model.clone())
        .capability(CapabilityParts::new().require::<Store>().named("memory"));
    let err = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect_err("missing Store");
    let message = err.to_string();
    assert!(message.contains("capability `memory`"), "{message}");
    assert!(message.contains("Store"), "{message}");
    model.assert_complete();

    let model = scripted();
    let agent = Agent::new("a")
        .model(model)
        .capability(CapabilityParts::new().require::<Store>().named("memory"));
    Runner::run(&agent, "go", RunOptions::new().with(Store))
        .await
        .expect("run");
}
