//! Lifecycle hooks
//!
//! Web demo checklist: "Lifecycle hooks".
//!
//! `RunHooks` (set on `RunOptions`) observes the whole run; `AgentHooks` (set on one agent)
//! observes only that agent. Every method has a default no-op body, so implement what you need.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example hooks
//! ```

mod common;

use std::sync::Arc;

use opai_agents::{Agent, AgentHooks, RunContextWrapper, RunHooks, RunOptions, Runner};

// Hooks observe a run without changing it, which makes them the place for logging, metrics and
// progress display. `RunHooks` sees every agent in the run (set per run); `AgentHooks` sees
// one agent only (set on the agent). Every method has a default, so implement only what you need.
struct PrintRunHooks;

#[async_trait::async_trait]
impl RunHooks for PrintRunHooks {
    async fn on_agent_start(&self, _context: RunContextWrapper, agent: &Agent) {
        println!("[run] agent_start {}", agent.name);
    }

    // Called before each model call, with the system prompt and input about to be sent. A run with
    // tools makes several model calls, so this fires more than once.
    async fn on_llm_start(
        &self,
        _context: RunContextWrapper,
        agent: &Agent,
        _system_prompt: Option<&str>,
        _input: &[serde_json::Value],
    ) {
        println!("[run] llm_start {}", agent.name);
    }

    async fn on_agent_end(
        &self,
        _context: RunContextWrapper,
        agent: &Agent,
        output: &serde_json::Value,
    ) {
        println!("[run] agent_end {} -> {output}", agent.name);
    }
}

// Agent-level hooks are the same idea scoped to one agent: they fire only while that agent is
// active.
struct PrintAgentHooks;

#[async_trait::async_trait]
impl AgentHooks for PrintAgentHooks {
    async fn on_end(
        &self,
        _context: RunContextWrapper,
        agent: &Agent,
        _output: &serde_json::Value,
    ) {
        println!("[agent] on_end {}", agent.name);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new("Assistant")
        .instructions("Answer briefly.")
        .model(common::model()?)
        .hooks(Arc::new(PrintAgentHooks));

    // Run-level hooks go in the options, so each call can choose its own (for example one per
    // request, carrying a request id).
    let options = RunOptions::new().hooks(Arc::new(PrintRunHooks));

    let input = "Say hello.";
    let result = Runner::run(&agent, input, options).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
