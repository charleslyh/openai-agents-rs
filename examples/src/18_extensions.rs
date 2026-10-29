//! Typed dependencies for tools (Rust-only)
//!
//! Python hands a run one context object. Rust runs carry a small typed map instead
//! (`Extensions`), with at most one value per type, so independent parts of an application can
//! each bring their own dependency (a database handle, a clock, the caller's id) without agreeing
//! on a single application-wide struct.
//!
//! * `RunOptions::with(value)` puts a value in the map.
//! * `ctx.context::<T>()` reads it (`Option`); `ctx.require::<T>()` reads it or fails with an error
//!   that names the tool and the type.
//! * `FunctionTool::require::<T>()` declares that a tool cannot work without one. The runner
//!   checks the declaration before the first model turn, so a missing dependency is a startup
//!   error naming the agent, the tool and the type, not a surprise in the middle of a
//!   conversation and not a billed model call.
//!
//! The example runs the same agent twice, once without the dependency and once with it. The model
//! is scripted, so no key is needed, and the number of model calls shows that the first run never
//! reached the model.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example extensions
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{Agent, FunctionTool, Model, RunOptions, Runner};
use serde_json::json;

/// A stand-in for a database handle.
struct Orders(HashMap<&'static str, &'static str>);

/// Who is calling; the tool uses it to scope the lookup.
struct CallerId(&'static str);

fn order_status_tool() -> FunctionTool {
    FunctionTool::typed(
        "order_status",
        "Look up the status of the caller's order.",
        |ctx, ()| async move {
            // Declared with `.require::<Orders>()` below, so this cannot fail in a started run.
            let orders = ctx.require::<Orders>()?;
            // Optional: a missing caller id degrades instead of failing.
            let caller = ctx.context::<CallerId>().map_or("anonymous", |c| c.0);
            let status = orders.0.get(caller).copied().unwrap_or("no order found");
            Ok(json!({"caller": caller, "status": status}))
        },
    )
    .require::<Orders>()
}

fn agent() -> (Agent, Arc<ScriptedModel>) {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("order_status", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("Your order has shipped.")),
    ]));
    let agent = Agent::new("Support")
        .model(Arc::clone(&model) as Arc<dyn Model>)
        .tools(vec![order_status_tool()]);
    (agent, model)
}

#[tokio::main]
async fn main() {
    // ---- Without the dependency: the run is refused before any model call. ----
    let (agent_without, model) = agent();
    let outcome = Runner::run(&agent_without, "Where is my order?", RunOptions::new()).await;
    println!("without `Orders`:");
    println!("  model calls: {}", model.calls().len());
    println!(
        "  result     : {}",
        outcome
            .map(|_| "ok".to_string())
            .unwrap_or_else(|e| e.to_string())
    );

    // ---- With it: one value per type, added with `.with`. ----
    let (agent_with, model) = agent();
    let options = RunOptions::new()
        .with(Orders(HashMap::from([
            ("u-7", "shipped"),
            ("u-8", "processing"),
        ])))
        .with(CallerId("u-7"));
    let result = Runner::run(&agent_with, "Where is my order?", options)
        .await
        .expect("run");
    println!("\nwith `Orders` and `CallerId`:");
    println!("  model calls: {}", model.calls().len());
    println!(
        "  answer     : {}",
        result.final_output_as_str().unwrap_or_default()
    );
}
