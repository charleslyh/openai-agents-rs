//! Run context
//!
//! Web demo checklist: "Run context".
//!
//! `RunOptions::context` carries your own value through the run. A tool downcasts it with
//! `ctx.run_context.context::<T>()`; the model never sees it.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p openai-agents-examples --example run_context
//! ```

mod common;

use std::sync::Arc;

use openai_agents::{Agent, FunctionTool, RunOptions, Runner};

/// The user context carried through the run.
struct Tenant {
    id: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tool = FunctionTool::new(
        "tenant_id",
        "Return the current tenant id from the run context.",
        serde_json::json!({"type": "object", "properties": {}, "additionalProperties": false}),
        |ctx, _args| async move {
            let id = ctx
                .run_context
                .context::<Tenant>()
                .map(|t| t.id.clone())
                .unwrap_or_else(|| "unknown".into());
            Ok(serde_json::json!(id))
        },
    );

    let agent = Agent::new("Assistant")
        .instructions("Answer briefly. Use tenant_id for questions about the current tenant.")
        .model(common::model()?)
        .tools(vec![tool]);

    let mut options = RunOptions::default();
    options.context = Some(Arc::new(Tenant { id: "acme".into() }));

    let input = "Which tenant am I?";
    let result = Runner::run(&agent, input, options).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
