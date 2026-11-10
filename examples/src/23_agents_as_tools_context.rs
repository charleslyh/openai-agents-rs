//! Agents as tools: the nested run shares the outer run's context
//!
//! A sub-agent used with `Agent::as_tool` runs a nested run. By default that run inherits the
//! outer run's user context (`RunOptions::context` / `RunOptions::with`), so the sub-agent's tools
//! can use the same dependencies (a hub client, a tenant) without any hand-written plumbing.
//! `AsToolConfig::inherit_context(false)` isolates a sub-agent; `AsToolConfig::context(..)` gives
//! it a context of its own.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example agents_as_tools_context
//! ```

mod common;

use std::sync::Arc;

use opai_agents::{Agent, AsToolConfig, FunctionTool, RunOptions, Runner};

/// A dependency of the desk: a client for some hub service. It holds a credential, so the
/// example never prints it.
struct DeskHub {
    base_url: String,
    #[allow(dead_code)]
    token: String,
}

impl DeskHub {
    /// Stand-in for an authenticated HTTP call.
    fn scene_list(&self) -> Vec<&'static str> {
        let _ = &self.base_url;
        vec!["intro", "chase", "finale"]
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = common::model()?;

    // The sub-agent's tool reads `DeskHub` from the run context. Nothing in this agent says where
    // the hub comes from: it is whatever the run it is part of was started with.
    let list_scenes = FunctionTool::typed(
        "list_scenes",
        "List the scenes of the current project from the hub.",
        |ctx, ()| async move {
            let hub = ctx.require::<DeskHub>()?;
            Ok(serde_json::json!(hub.scene_list()))
        },
    );
    let director = Agent::new("director")
        .instructions("You are the 3D director. Use list_scenes to answer questions about scenes.")
        .model(Arc::clone(&model))
        .tools(vec![list_scenes]);

    // A sub-agent that must not reach the hub: its tool asks for it and finds nothing.
    let probe = FunctionTool::typed(
        "hub_status",
        "Report whether the hub is available to you.",
        |ctx, ()| async move {
            let available = ctx.context::<DeskHub>().is_some();
            Ok(serde_json::json!(if available {
                "hub available"
            } else {
                "no hub: this agent runs without the desk's context"
            }))
        },
    );
    let researcher = Agent::new("researcher")
        .instructions("You research on your own. Call hub_status once and report what it says.")
        .model(Arc::clone(&model))
        .tools(vec![probe]);

    let desk = Agent::new("desk")
        .instructions(
            "You are the desk. Delegate scene questions to director_delegate, and ask \
             sandboxed_researcher to report on its hub access. Then summarize both answers.",
        )
        .model(model)
        .tools(vec![
            // Default: inherits the desk's context. The director's tool finds the `DeskHub`.
            director.as_tool(AsToolConfig {
                name: Some("director_delegate".into()),
                description: Some("Ask the 3D director about the project's scenes".into()),
                ..Default::default()
            }),
            // Isolated: the same context would be visible to every tool of this agent, so a
            // sub-agent that should not hold the hub's credential opts out. Use
            // `.context(other_value)` instead to hand it a different dependency.
            researcher.as_tool(
                AsToolConfig {
                    name: Some("sandboxed_researcher".into()),
                    description: Some("Research without access to the desk's services".into()),
                    ..Default::default()
                }
                .inherit_context(false),
            ),
        ]);

    // The context is supplied once, on the outer run.
    let options = RunOptions::new().context(DeskHub {
        base_url: "https://hub.example.com".into(),
        token: "not-a-real-token".into(),
    });

    let result = Runner::run(
        &desk,
        "Which scenes does the project have, and can the researcher see the hub?",
        options,
    )
    .await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
