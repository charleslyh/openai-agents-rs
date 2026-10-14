//! Run-scoped lifecycle hooks, turned into timeline events.

use async_trait::async_trait;
use openai_agents::{
    Agent, FunctionTool, ModelResponse, ResponseInputItem, RunContextWrapper, RunHooks, ToolContext,
};
use serde_json::Value;

use super::events::{truncate, DemoCtx, DemoEvent, Emitter};

pub struct TimelineHooks;

fn emitter(context: &RunContextWrapper) -> Option<Emitter> {
    context.context::<DemoCtx>().map(|c| c.emit.clone())
}

async fn hook(emit: Option<Emitter>, name: &str, agent: &str, detail: String) {
    if let Some(emit) = emit {
        emit.emit(DemoEvent::Hook {
            name: name.into(),
            agent: agent.into(),
            detail: truncate(&detail, 200),
        })
        .await;
    }
}

#[async_trait]
impl RunHooks for TimelineHooks {
    async fn on_agent_start(&self, context: RunContextWrapper, agent: &Agent) {
        hook(emitter(&context), "agent_start", &agent.name, String::new()).await;
    }

    async fn on_agent_end(&self, context: RunContextWrapper, agent: &Agent, output: &Value) {
        hook(
            emitter(&context),
            "agent_end",
            &agent.name,
            output.to_string(),
        )
        .await;
    }

    async fn on_handoff(&self, context: RunContextWrapper, from: &Agent, to: &Agent) {
        hook(
            emitter(&context),
            "handoff",
            &from.name,
            format!("{} -> {}", from.name, to.name),
        )
        .await;
    }

    async fn on_tool_start(&self, context: ToolContext, agent: &Agent, tool: &FunctionTool) {
        hook(
            emitter(&context.run_context),
            "tool_start",
            &agent.name,
            tool.name.clone(),
        )
        .await;
    }

    async fn on_tool_end(
        &self,
        context: ToolContext,
        agent: &Agent,
        tool: &FunctionTool,
        result: &Value,
    ) {
        hook(
            emitter(&context.run_context),
            "tool_end",
            &agent.name,
            format!("{} -> {}", tool.name, result),
        )
        .await;
    }

    async fn on_llm_start(
        &self,
        context: RunContextWrapper,
        agent: &Agent,
        _system_prompt: Option<&str>,
        input_items: &[ResponseInputItem],
    ) {
        hook(
            emitter(&context),
            "llm_start",
            &agent.name,
            format!("{} input items", input_items.len()),
        )
        .await;
    }

    async fn on_llm_end(
        &self,
        context: RunContextWrapper,
        agent: &Agent,
        response: &ModelResponse,
    ) {
        hook(
            emitter(&context),
            "llm_end",
            &agent.name,
            format!(
                "{} output items, {} tokens",
                response.output.len(),
                response.usage.total_tokens
            ),
        )
        .await;
    }
}
