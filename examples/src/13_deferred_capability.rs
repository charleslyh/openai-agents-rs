//! Loading a capability on demand (X-021)
//!
//! A capability marked `deferred` stays out of the prompt: the model sees one catalog line and a
//! framework tool, `load_capability`, and none of the capability's tools until it asks for them.
//! That keeps a large tool collection from costing context on every turn.
//!
//! The wrapper model below prints the tool names each step is offered, so the change is visible in
//! the output: first `load_capability`, then `query`.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example deferred_capability
//! ```

mod common;

use std::sync::Arc;

use async_trait::async_trait;
use opai_agents::prelude::*;
use opai_agents::{ModelError, ModelRequest, ModelResponse, NamedCapability};

/// A bundle that is only worth its context when the model actually needs SQL.
fn database() -> NamedCapability {
    CapabilityParts::new()
        .tool(FunctionTool::constant(
            "query",
            "Run one SQL statement and return its rows",
            "id | name\n1 | Ada",
        ))
        .instructions("Use `query` for anything that needs the database.")
        .deferred("run SQL against the orders database")
        .named("database")
}

/// Prints what each model call is offered, then asks the real model.
struct TraceTools {
    inner: Arc<dyn Model>,
}

#[async_trait]
impl Model for TraceTools {
    async fn get_response(&self, request: ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        let names: Vec<&str> = request.tools.iter().map(|t| t.name.as_str()).collect();
        println!("  tools offered: {names:?}");
        self.inner.get_response(request).await
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(TraceTools {
        inner: common::model()?,
    });
    let agent = Agent::new("Analyst")
        .instructions("Answer questions about the shop's data.")
        .model(Arc::clone(&model) as Arc<dyn Model>)
        .capability(database());

    println!("First run: the model sees the catalog, loads `database`, then queries.");
    let result = Runner::run(
        &agent,
        "How many customers are there? Look at the database.",
        RunOptions::new(),
    )
    .await?;
    println!("\n{}", result.final_output_as_str().unwrap_or_default());

    Ok(())
}
