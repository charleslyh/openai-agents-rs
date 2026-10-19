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
//! cargo run -p openai-agents-examples --example hooks
//! ```

mod common;

use std::sync::Arc;

use openai_agents::{Agent, AgentHooks, RunContextWrapper, RunHooks, RunOptions, Runner};

struct PrintRunHooks;

#[async_trait::async_trait]
impl RunHooks for PrintRunHooks {
    async fn on_agent_start(&self, _context: RunContextWrapper, agent: &Agent) {
        println!("[run] agent_start {}", agent.name);
    }

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

    let options = RunOptions::new().hooks(Arc::new(PrintRunHooks));

    let input = "Say hello.";
    let result = Runner::run(&agent, input, options).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
