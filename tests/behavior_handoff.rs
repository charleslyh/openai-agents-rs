//! Handoff behavior tests.

use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    handoff, Agent, Handoff, HandoffOutputItem, RunItem, RunItemStreamName, RunOptions, Runner,
    StreamEvent,
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

/// D-A: handoffs produce `HandoffCallItem` / `HandoffOutputItem`, not plain tool items.
#[tokio::test]
async fn handoff_produces_handoff_run_items() {
    let specialist_model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("specialist done"),
    )]));
    let specialist = Agent::new("Specialist").model(specialist_model);
    let triage_model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call(Handoff::default_tool_name("Specialist"), "{}", "h1"),
    )]));
    let triage = Agent::new("Triage")
        .model(triage_model)
        .handoffs(vec![handoff(specialist)]);

    let result = Runner::run(&triage, "please specialize", RunOptions::default())
        .await
        .expect("run");

    let calls: Vec<&str> = result
        .new_items
        .iter()
        .filter_map(|i| match i {
            RunItem::HandoffCall(c) => Some(c.agent_name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(calls, vec!["Triage"]);

    let outputs: Vec<HandoffOutputItem> = result
        .new_items
        .iter()
        .filter_map(|i| match i {
            RunItem::HandoffOutput(o) => Some(o.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].source_agent_name, "Triage");
    assert_eq!(outputs[0].target_agent_name, "Specialist");
    // Python: `Handoff.get_transfer_message` -> `json.dumps({"assistant": <target>})`, which
    // puts a space after the colon; the model reads this text.
    assert_eq!(
        outputs[0].raw_item["output"],
        serde_json::json!(r#"{"assistant": "Specialist"}"#)
    );
    // No plain tool items for the handoff itself.
    assert!(!result
        .new_items
        .iter()
        .any(|i| matches!(i, RunItem::ToolCall(_))));
}

/// D-A: reasoning output becomes its own run item instead of being wrapped as a message.
#[tokio::test]
async fn reasoning_output_becomes_reasoning_item() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(vec![
        serde_json::json!({"type": "reasoning", "summary": []}),
        ItemHelpers::text_message("answer"),
    ])]));
    let agent = Agent::new("thinker").model(model);
    let result = Runner::run(&agent, "think", RunOptions::default())
        .await
        .expect("run");
    assert!(result
        .new_items
        .iter()
        .any(|i| matches!(i, RunItem::Reasoning(_))));
    // The reasoning item must not be reported as an assistant message.
    assert_eq!(
        result
            .new_items
            .iter()
            .filter(|i| matches!(i, RunItem::Message(_)))
            .count(),
        1
    );
}

#[test]
fn default_tool_name_sanitizes() {
    assert_eq!(
        Handoff::default_tool_name("Support Agent"),
        "transfer_to_support_agent"
    );
}
