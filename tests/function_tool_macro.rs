//! Tests for `#[function_tool]`.

use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{function_tool, Agent, RunOptions, Runner};

#[function_tool(description = "Add two integers")]
async fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[function_tool(name = "greet", description = "Greet someone")]
fn greet_person(name: String) -> String {
    format!("hello {name}")
}

#[tokio::test]
async fn macro_tool_invoked_in_runner() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(
            "add",
            r#"{"a":2,"b":3}"#,
            "c1",
        )),
        ModelStep::from(ItemHelpers::text_message("5")),
    ]));
    let agent = Agent::new("math").model(model).tools(vec![add()]);
    let result = Runner::run(&agent, "add", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("5"));
    assert!(result.new_items.iter().any(|i| {
        matches!(i, openai_agents::RunItem::ToolCallOutput(o) if o.output == serde_json::json!(5))
    }));
}

#[tokio::test]
async fn macro_name_override_and_sync_fn() {
    let tool = greet_person();
    assert_eq!(tool.name, "greet");
    assert_eq!(tool.description, "Greet someone");
    let out = (tool.on_invoke_tool)(
        openai_agents::ToolContext {
            tool_name: "greet".into(),
            tool_call_id: "c".into(),
            tool_arguments: r#"{"name":"Ada"}"#.into(),
        },
        r#"{"name":"Ada"}"#.into(),
    )
    .await
    .expect("invoke");
    assert_eq!(out, serde_json::json!("hello Ada"));
}
