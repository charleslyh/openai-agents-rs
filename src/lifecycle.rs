//! Lifecycle hooks (Python: `agents.lifecycle`).
//!
//! Every method has a no-op default, so implement only the events you care about.

use async_trait::async_trait;
use serde_json::Value;

use crate::agent::Agent;
use crate::items::{ModelResponse, ResponseInputItem};
use crate::run_context::RunContextWrapper;
use crate::tool::{FunctionTool, ToolContext};

/// Hooks invoked for every agent in a run (Python: `RunHooks`).
#[async_trait]
pub trait RunHooks: Send + Sync {
    /// Called before an agent is invoked; also called again after each handoff.
    async fn on_agent_start(&self, _context: RunContextWrapper, _agent: &Agent) {}

    /// Called when an agent produced its final output.
    async fn on_agent_end(&self, _context: RunContextWrapper, _agent: &Agent, _output: &Value) {}

    /// Called when a handoff occurs.
    async fn on_handoff(&self, _context: RunContextWrapper, _from_agent: &Agent, _to_agent: &Agent) {}

    /// Called immediately before a tool is invoked.
    async fn on_tool_start(&self, _context: ToolContext, _agent: &Agent, _tool: &FunctionTool) {}

    /// Called immediately after a tool returns.
    async fn on_tool_end(
        &self,
        _context: ToolContext,
        _agent: &Agent,
        _tool: &FunctionTool,
        _result: &Value,
    ) {
    }

    /// Called immediately before the agent issues an LLM call.
    async fn on_llm_start(
        &self,
        _context: RunContextWrapper,
        _agent: &Agent,
        _system_prompt: Option<&str>,
        _input_items: &[ResponseInputItem],
    ) {
    }

    /// Called immediately after the agent receives an LLM response.
    async fn on_llm_end(
        &self,
        _context: RunContextWrapper,
        _agent: &Agent,
        _response: &ModelResponse,
    ) {
    }
}

/// Hooks scoped to a single agent (Python: `AgentHooks`).
#[async_trait]
pub trait AgentHooks: Send + Sync {
    /// Called before this agent is invoked.
    async fn on_start(&self, _context: RunContextWrapper, _agent: &Agent) {}

    /// Called when this agent produced its final output.
    async fn on_end(&self, _context: RunContextWrapper, _agent: &Agent, _output: &Value) {}

    /// Called when this agent is being handed off to.
    async fn on_handoff(&self, _context: RunContextWrapper, _agent: &Agent, _source: &Agent) {}

    /// Called immediately before a tool is invoked.
    async fn on_tool_start(&self, _context: ToolContext, _agent: &Agent, _tool: &FunctionTool) {}

    /// Called immediately after a tool returns.
    async fn on_tool_end(
        &self,
        _context: ToolContext,
        _agent: &Agent,
        _tool: &FunctionTool,
        _result: &Value,
    ) {
    }

    /// Called immediately before this agent issues an LLM call.
    async fn on_llm_start(
        &self,
        _context: RunContextWrapper,
        _agent: &Agent,
        _system_prompt: Option<&str>,
        _input_items: &[ResponseInputItem],
    ) {
    }

    /// Called immediately after this agent receives an LLM response.
    async fn on_llm_end(
        &self,
        _context: RunContextWrapper,
        _agent: &Agent,
        _response: &ModelResponse,
    ) {
    }
}
