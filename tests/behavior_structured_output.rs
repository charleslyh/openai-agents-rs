//! Structured output (`output_type`) behavior tests.

use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    Agent, AgentOutputSchema, AgentOutputSchemaBase, AgentsError, ModelError, RunOptions, Runner,
};

#[derive(Debug, Clone, PartialEq, serde::Deserialize, openai_agents::schemars::JsonSchema)]
struct Answer {
    /// Short answer text.
    text: String,
    /// Confidence between 0 and 1.
    confidence: f32,
}

fn answer_json() -> String {
    serde_json::json!({"text": "42", "confidence": 0.9}).to_string()
}

#[test]
fn schema_is_strict_and_named() {
    let schema = AgentOutputSchema::of::<Answer>().expect("schema");
    assert_eq!(schema.name(), std::any::type_name::<Answer>());
    assert!(schema.is_strict_json_schema());
    assert!(!schema.is_plain_text());
    let json = schema.json_schema().expect("json schema");
    assert_eq!(json["additionalProperties"], false);
    let required = json["required"].as_array().expect("required");
    assert!(required.iter().any(|r| r == "text"));
    assert!(required.iter().any(|r| r == "confidence"));
    assert_eq!(json["properties"]["text"]["description"], "Short answer text.");
}

/// `String` mirrors Python's `output_type=str` and stays plain text.
#[test]
fn string_output_type_is_plain_text() {
    let schema = AgentOutputSchema::of::<String>().expect("schema");
    assert!(schema.is_plain_text());
    assert!(schema.json_schema().is_err());
}

/// Types that are not JSON objects get wrapped under `response`, like Python.
#[test]
fn non_object_types_are_wrapped() {
    let schema = AgentOutputSchema::of::<Vec<String>>().expect("schema");
    let json = schema.json_schema().expect("json schema");
    assert_eq!(json["type"], "object");
    assert_eq!(json["properties"]["response"]["type"], "array");
    assert_eq!(json["required"][0], "response");
    let parsed = schema
        .validate_json(r#"{"response": ["a", "b"]}"#)
        .expect("validate");
    assert_eq!(parsed, serde_json::json!(["a", "b"]));
}

#[tokio::test]
async fn structured_output_is_validated_and_returned() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message(answer_json()),
    )]));
    let agent = Agent::new("structured")
        .model(model)
        .output_type(Arc::new(AgentOutputSchema::of::<Answer>().expect("schema")));
    let result = Runner::run(&agent, "what is the answer?", RunOptions::default())
        .await
        .expect("run");

    assert_eq!(
        result.final_output,
        serde_json::json!({"text": "42", "confidence": 0.9})
    );
    let answer: Answer = result.final_output_as::<Answer>().expect("typed output");
    assert_eq!(answer.text, "42");
}

#[tokio::test]
async fn invalid_structured_output_is_a_behavior_error() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message(r#"{"text": "42"}"#),
    )]));
    let agent = Agent::new("structured")
        .model(model)
        .output_type(Arc::new(AgentOutputSchema::of::<Answer>().expect("schema")));
    let err = Runner::run(&agent, "?", RunOptions::default())
        .await
        .expect_err("missing field must fail");
    assert!(
        matches!(err, AgentsError::Model(ModelError::Behavior(_))),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn empty_structured_output_is_a_behavior_error() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message(""),
    )]));
    let agent = Agent::new("structured")
        .model(model)
        .output_type(Arc::new(AgentOutputSchema::of::<Answer>().expect("schema")));
    let err = Runner::run(&agent, "?", RunOptions::default())
        .await
        .expect_err("empty output must fail");
    match err {
        AgentsError::Model(ModelError::Behavior(msg)) => {
            assert!(msg.contains("no final output"), "unexpected message: {msg}");
        }
        other => panic!("unexpected error: {other}"),
    }
}

/// Python coerces a tool result to `str` unless a non-plain-text `output_type` is declared.
#[tokio::test]
async fn stop_on_first_tool_stringifies_without_output_type() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call("count", "{}", "c1"),
    )]));
    let agent = Agent::new("counter")
        .model(model)
        .tools(vec![openai_agents::FunctionTool::constant(
            "count",
            "count",
            "7",
        )])
        .tool_use_behavior(openai_agents::ToolUseBehavior::StopOnFirstTool);
    let result = Runner::run(&agent, "count", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output, serde_json::json!("7"));

    // With a structured output type the raw value is preserved.
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call("count", "{}", "c1"),
    )]));
    let agent = Agent::new("counter")
        .model(model)
        .tools(vec![openai_agents::FunctionTool::new(
            "count",
            "count",
            serde_json::json!({"type": "object", "properties": {}, "additionalProperties": false}),
            |_ctx, _args| async { Ok(serde_json::json!(7)) },
        )])
        .output_type(Arc::new(AgentOutputSchema::of::<Answer>().expect("schema")))
        .tool_use_behavior(openai_agents::ToolUseBehavior::StopOnFirstTool);
    let result = Runner::run(&agent, "count", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output, serde_json::json!(7));
}
