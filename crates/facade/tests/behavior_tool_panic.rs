//! A panic inside a tool or a hook must not unwind through the whole run.

use std::sync::Arc;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{
    Agent, AgentsError, FunctionTool, RunHooks, RunItem, RunOptions, Runner, ToolContext,
};
use serde_json::{json, Value};

fn panicking_tool(name: &str) -> FunctionTool {
    FunctionTool::new(
        name,
        "always panics",
        json!({"type": "object", "properties": {}, "additionalProperties": false}),
        |_ctx, _args| async {
            if true {
                panic!("secret panic detail");
            }
            Ok::<Value, AgentsError>(Value::Null)
        },
    )
}

fn calling_model(tool: &str) -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(tool, "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("recovered")),
    ]))
}

#[tokio::test]
async fn a_panicking_tool_is_reported_to_the_model_and_the_run_continues() {
    let agent = Agent::new("a")
        .model(calling_model("boom"))
        .tools(vec![panicking_tool("boom")]);
    let result = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run must survive a tool panic");
    assert_eq!(result.final_output_as_str(), Some("recovered"));
    let outputs: Vec<String> = result
        .new_items
        .iter()
        .filter_map(|i| match i {
            RunItem::ToolCallOutput(o) => o.raw_item["output"].as_str().map(str::to_string),
            _ => None,
        })
        .collect();
    assert_eq!(outputs.len(), 1);
    assert!(!outputs[0].contains("secret panic detail"), "{outputs:?}");
}

struct PanicOnToolStart;

#[async_trait::async_trait]
impl RunHooks for PanicOnToolStart {
    async fn on_tool_start(&self, _context: ToolContext, _agent: &Agent, _tool: &FunctionTool) {
        panic!("hook exploded");
    }
}

#[tokio::test]
async fn a_panicking_hook_ends_the_run_with_an_internal_error() {
    let ok_tool = FunctionTool::new(
        "ok",
        "fine",
        json!({"type": "object", "properties": {}, "additionalProperties": false}),
        |_ctx, _args| async { Ok::<Value, AgentsError>(json!("fine")) },
    );
    let agent = Agent::new("a")
        .model(calling_model("ok"))
        .tools(vec![ok_tool]);
    let mut opts = RunOptions::default();
    opts.hooks = Some(Arc::new(PanicOnToolStart) as Arc<dyn RunHooks>);
    let err = Runner::run(&agent, "go", opts).await.unwrap_err();
    assert!(matches!(err, AgentsError::Internal { .. }), "{err}");
}
