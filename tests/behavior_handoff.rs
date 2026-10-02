//! Handoff behavior tests.

use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    handoff, Agent, Handoff, RunItemStreamName, RunOptions, Runner, StreamEvent,
};

#[tokio::test]
async fn handoff_switches_agent_and_continues() {
    let specialist_model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("specialist done"),
    )]));
    let specialist = Agent::new("Specialist")
        .instructions("Handle specialized requests.")
        .model(specialist_model);

    let triage_model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call(
            Handoff::default_tool_name("Specialist"),
            "{}",
            "h1",
        ),
    )]));
    let triage = Agent::new("Triage")
        .instructions("Route to specialist when needed.")
        .model(triage_model)
        .handoffs(vec![handoff(specialist)]);

    let result = Runner::run(&triage, "please specialize", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("specialist done"));
    assert_eq!(result.last_agent_name, "Specialist");
    assert_eq!(result.raw_responses.len(), 2);
}

#[tokio::test]
async fn streamed_handoff_emits_events() {
    let specialist_model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let specialist = Agent::new("B").model(specialist_model);
    let triage_model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call(Handoff::default_tool_name("B"), "{}", "h1"),
    )]));
    let triage = Agent::new("A")
        .model(triage_model)
        .handoffs(vec![handoff(specialist)]);

    let mut streamed = Runner::run_streamed(triage, "go", RunOptions::default());
    let events = streamed.collect_events().await.expect("events");
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::RunItem {
            name: RunItemStreamName::HandoffRequested,
            ..
        }
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::RunItem {
            name: RunItemStreamName::HandoffOccured,
            ..
        }
    )));
    assert!(
        events
            .iter()
            .filter(|e| matches!(e, StreamEvent::AgentUpdated { .. }))
            .count()
            >= 2
    );
    assert_eq!(streamed.current_agent_name(), "B");
}

#[test]
fn default_tool_name_sanitizes() {
    assert_eq!(
        Handoff::default_tool_name("Support Agent"),
        "transfer_to_support_agent"
    );
}
