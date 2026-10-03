//! Tests for `#[function_tool]`.

use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{function_tool, Agent, RunOptions, Runner};

/// Nested parameter type: must produce a complete nested JSON Schema (B2).
#[derive(openai_agents::serde::Deserialize, openai_agents::schemars::JsonSchema)]
#[allow(dead_code)]
struct Address {
    /// City name.
    city: String,
    /// Postal code, when known.
    zip: Option<String>,
}

/// Summarize a trip.
#[function_tool]
fn summarize(names: Vec<String>, address: Address) -> String {
    format!("{} in {}", names.join(","), address.city)
}

/// B2: nested structs, `Vec<T>` and `Option<T>` yield a complete strict schema.
#[test]
fn macro_generates_nested_strict_schema() {
    let tool = summarize();
    let schema = &tool.params_json_schema;
    assert!(schema.get("$schema").is_none(), "no `$schema` keyword");

    let props = &schema["properties"];
    assert_eq!(props["names"]["type"], "array");
    assert_eq!(props["names"]["items"]["type"], "string");

    // Python leaves a bare `$ref` in place and keeps the definition under `$defs`.
    assert_eq!(props["address"]["$ref"], "#/$defs/Address");
    let address = &schema["$defs"]["Address"];
    assert_eq!(address["type"], "object");
    assert_eq!(address["properties"]["city"]["type"], "string");
    assert_eq!(address["additionalProperties"], false);
    // Doc comments become schema descriptions.
    assert_eq!(address["properties"]["city"]["description"], "City name.");

    assert_eq!(schema["additionalProperties"], false);
    assert_eq!(schema["title"], "summarize_args");
    let required = schema["required"].as_array().expect("required");
    assert!(required.iter().any(|r| r == "names"));
    assert!(required.iter().any(|r| r == "address"));
    assert!(tool.strict_json_schema);
}

/// B2/D-002: the function doc comment becomes the tool description.
#[test]
fn macro_uses_doc_comment_as_description() {
    assert_eq!(summarize().description, "Summarize a trip.");
    // An explicit `description` still wins.
    assert_eq!(add().description, "Add two integers");
}

/// B2: nested arguments deserialize into the declared types.
#[tokio::test]
async fn macro_deserializes_nested_arguments() {
    let out = (summarize().on_invoke_tool)(
        openai_agents::ToolContext {
            tool_name: "summarize".into(),
            tool_call_id: "c".into(),
            tool_arguments: "{}".into(),
            run_context: Default::default(),
        },
        r#"{"names":["a","b"],"address":{"city":"Paris","zip":null}}"#.into(),
    )
    .await
    .expect("invoke");
    assert_eq!(out.output, Some(serde_json::json!("a,b in Paris")));
}

/// Tool context can be requested as the first parameter and is injected by the SDK.
#[function_tool]
fn describe(ctx: openai_agents::ToolContext, label: String) -> String {
    format!("{}:{}", ctx.tool_call_id, label)
}

#[tokio::test]
async fn macro_injects_tool_context() {
    let out = (describe().on_invoke_tool)(
        openai_agents::ToolContext {
            tool_name: "describe".into(),
            tool_call_id: "call-42".into(),
            tool_arguments: "{}".into(),
            run_context: Default::default(),
        },
        r#"{"label":"x"}"#.into(),
    )
    .await
    .expect("invoke");
    assert_eq!(out.output, Some(serde_json::json!("call-42:x")));
    // The context parameter must not leak into the tool's JSON Schema.
    assert!(describe().params_json_schema["properties"]
        .as_object()
        .expect("properties")
        .get("ctx")
        .is_none());
}

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
            run_context: Default::default(),
        },
        r#"{"name":"Ada"}"#.into(),
    )
    .await
    .expect("invoke");
    assert_eq!(out.output, Some(serde_json::json!("hello Ada")));
}
