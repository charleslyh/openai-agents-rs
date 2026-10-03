//! 09 — Context, guardrails and lifecycle hooks
//!
//! Shows the three features that share the run context:
//! - a user context (`RunOptions.context`) downcast inside a tool,
//! - an input guardrail that trips and aborts the run,
//! - run-scoped and agent-scoped hooks.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`, `EXAMPLE_INPUT`.
//!
//! ```bash
//! cargo run --example 09_guardrails_hooks
//! # Triggers the tripwire:
//! EXAMPLE_INPUT="ignore the rules" cargo run --example 09_guardrails_hooks
//! ```

#[path = "common/mod.rs"]
mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use openai_agents::{
    input_guardrail, output_guardrail, Agent, AgentHooks, AgentsError, FunctionTool,
    GuardrailFunctionOutput, InputGuardrailTripwireTriggered, RunContextWrapper, RunHooks,
    RunOptions, Runner,
};

/// The user context carried through tools, guardrails and hooks.
struct Tenant {
    id: String,
}

struct TenantHooks {
    llm_calls: AtomicUsize,
}

#[async_trait::async_trait]
impl RunHooks for TenantHooks {
    async fn on_llm_start(
        &self,
        context: RunContextWrapper,
        agent: &Agent,
        _system_prompt: Option<&str>,
        _input: &[serde_json::Value],
    ) {
        self.llm_calls.fetch_add(1, Ordering::SeqCst);
        let tenant = context.context::<Tenant>().map(|t| t.id.as_str());
        println!("llm_start agent={} tenant={:?}", agent.name, tenant);
    }
}

struct AgentLogger;

#[async_trait::async_trait]
impl AgentHooks for AgentLogger {
    async fn on_end(&self, _context: RunContextWrapper, agent: &Agent, output: &serde_json::Value) {
        println!("agent_end {} -> {output}", agent.name);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A tool that reads the run context.
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

    let agent = Agent::new("Guarded")
        .instructions("Answer briefly.")
        .model(common::model()?)
        .tools(vec![tool])
        .hooks(Arc::new(AgentLogger))
        // Trips when the user asks to ignore the rules.
        .input_guardrails(vec![input_guardrail(
            "no-jailbreak",
            |_ctx, _agent, input| {
                let text = match &input {
                    openai_agents::InputLike::Text(s) => s.clone(),
                    openai_agents::InputLike::Items(items) => items
                        .iter()
                        .filter_map(|i| i.get("content").and_then(|c| c.as_str()))
                        .collect::<Vec<_>>()
                        .join(" "),
                };
                async move {
                    if text.to_ascii_lowercase().contains("ignore the rules") {
                        GuardrailFunctionOutput::trip(serde_json::json!({"reason": "jailbreak"}))
                    } else {
                        GuardrailFunctionOutput::pass(serde_json::json!({}))
                    }
                }
            },
        )])
        // Refuses answers that mention a fake secret.
        .output_guardrails(vec![output_guardrail(
            "no-secrets",
            |_ctx, _agent, output| async move {
                if output
                    .as_str()
                    .is_some_and(|s| s.to_ascii_lowercase().contains("sk-live"))
                {
                    GuardrailFunctionOutput::trip(serde_json::json!({"reason": "secret"}))
                } else {
                    GuardrailFunctionOutput::pass(serde_json::json!({}))
                }
            },
        )]);

    let hooks = Arc::new(TenantHooks {
        llm_calls: AtomicUsize::new(0),
    });
    let mut opts = RunOptions::default();
    opts.context = Some(Arc::new(Tenant { id: "acme".into() }));
    opts.hooks = Some(Arc::clone(&hooks) as Arc<dyn RunHooks>);

    let question = common::input_with_fallback("question> ", "What is your tenant id?")?;

    match Runner::run(&agent, question, opts).await {
        Ok(result) => {
            println!("answer: {}", result.final_output);
            println!("llm calls: {}", hooks.llm_calls.load(Ordering::SeqCst));
            println!(
                "guardrails: input={} output={}",
                result.input_guardrail_results.len(),
                result.output_guardrail_results.len()
            );
        }
        Err(AgentsError::InputGuardrailTripwire(InputGuardrailTripwireTriggered {
            result,
        })) => {
            println!("blocked by guardrail `{}`", result.guardrail_name);
        }
        Err(e) => return Err(e.into()),
    }
    Ok(())
}
