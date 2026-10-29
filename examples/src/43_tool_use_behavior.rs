//! Deciding what happens after a tool runs, and what the model may call
//!
//! By default a tool result goes back to the model, which writes the answer: one extra model call
//! per tool turn. That is wasteful when the tool's result *is* the answer, and it is not enough
//! when a particular tool must be called. Two settings cover both:
//!
//! * **`ToolUseBehavior`** (on the agent) decides whether a tool result ends the run:
//!   `RunLlmAgain` (default), `StopOnFirstTool`, `StopAtTools { .. }` (end on specific tools) or
//!   `Custom(..)` (a function over the turn's results).
//! * **`ToolChoice`** (in `ModelSettings`) constrains the model's next call: `Auto`, `Required`,
//!   `None`, or `Function(name)` to force one tool. By default it is reset after the first tool
//!   turn (`reset_tool_choice`), because a forced tool would otherwise be called again forever.
//!
//! The model is scripted so that each case is repeatable and the number of model calls can be
//! counted; no key is needed.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example tool_use_behavior
//! ```

use std::sync::Arc;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{
    Agent, FunctionTool, Model, ModelSettings, RunOptions, Runner, ToolChoice, ToolUseBehavior,
    ToolsToFinalOutputResult,
};

/// An agent with two tools, whose model follows `steps`.
fn agent(steps: Vec<ModelStep>) -> (Agent, Arc<ScriptedModel>) {
    let model = Arc::new(ScriptedModel::new(steps));
    let agent = Agent::new("Assistant")
        .model(Arc::clone(&model) as Arc<dyn Model>)
        .tools(vec![
            FunctionTool::constant("lookup", "Look up the order status.", "shipped"),
            FunctionTool::constant("submit_answer", "Submit the final answer.", "42"),
        ]);
    (agent, model)
}

fn call(name: &str, id: &str) -> ModelStep {
    ModelStep::from(ItemHelpers::function_tool_call(name, "{}", id))
}

fn text(message: &str) -> ModelStep {
    ModelStep::from(ItemHelpers::text_message(message))
}

async fn report(title: &str, agent: &Agent, model: &ScriptedModel) {
    let result = Runner::run(agent, "Where is my order?", RunOptions::new())
        .await
        .expect("run");
    println!("{title}");
    println!("  model calls : {}", model.calls().len());
    println!("  final output: {}", result.final_output);
}

#[tokio::main]
async fn main() {
    println!("-- ToolUseBehavior --\n");

    let (a, m) = agent(vec![call("lookup", "c1"), text("Your order has shipped.")]);
    report("RunLlmAgain (default): the model writes the answer", &a, &m).await;

    let (a, m) = agent(vec![call("lookup", "c1"), text("never needed")]);
    let a = a.tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
    report("StopOnFirstTool: the tool's result is the answer", &a, &m).await;

    let (a, m) = agent(vec![
        call("lookup", "c1"),
        call("submit_answer", "c2"),
        text("never needed"),
    ]);
    let a = a.tool_use_behavior(ToolUseBehavior::StopAtTools {
        stop_at_tool_names: vec!["submit_answer".into()],
    });
    report(
        "StopAtTools [submit_answer]: `lookup` goes back to the model, `submit_answer` ends the run",
        &a,
        &m,
    )
    .await;

    let (a, m) = agent(vec![call("lookup", "c1"), text("never needed")]);
    let a = a.tool_use_behavior(ToolUseBehavior::Custom(Arc::new(
        |_ctx, results| match results.iter().find(|r| r.output == "shipped") {
            Some(r) => ToolsToFinalOutputResult::final_output(serde_json::json!({
                "tool": r.tool_name,
                "status": r.output,
            })),
            None => ToolsToFinalOutputResult::run_llm_again(),
        },
    )));
    report(
        "Custom: end only when a result says `shipped`, with an answer you build",
        &a,
        &m,
    )
    .await;

    println!("\n-- ToolChoice --\n");

    let forced = ModelSettings {
        tool_choice: Some(ToolChoice::Function("lookup".into())),
        ..Default::default()
    };

    let (a, m) = agent(vec![call("lookup", "c1"), text("Your order has shipped.")]);
    let a = a.model_settings(forced.clone());
    Runner::run(&a, "Where is my order?", RunOptions::new())
        .await
        .expect("run");
    println!("forced `lookup`, reset_tool_choice = true (default)");
    for (i, c) in m.calls().iter().enumerate() {
        println!(
            "  call {}: tool_choice = {:?}",
            i + 1,
            c.model_settings.tool_choice
        );
    }

    let (mut a, m) = agent(vec![call("lookup", "c1"), text("Your order has shipped.")]);
    a = a.model_settings(forced);
    a.reset_tool_choice = false;
    Runner::run(&a, "Where is my order?", RunOptions::new())
        .await
        .expect("run");
    println!("forced `lookup`, reset_tool_choice = false");
    for (i, c) in m.calls().iter().enumerate() {
        println!(
            "  call {}: tool_choice = {:?}",
            i + 1,
            c.model_settings.tool_choice
        );
    }
    println!("  (a real model would be forced to call `lookup` on every turn and never answer)");
}
