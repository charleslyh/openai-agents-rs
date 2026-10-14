//! The single agent network behind the chat box. Triage is the only entry point; it answers
//! simple things itself and hands off the rest.

use std::sync::Arc;

use openai_agents::{
    handoff, input_guardrail, output_guardrail, Agent, AgentOutputSchema, AsToolConfig,
    GuardrailFunctionOutput, InputGuardrail, InputLike, McpServer, Model, OutputGuardrail,
    RunErrorHandlerInput, RunErrorHandlerResult, RunErrorHandlers,
};
use serde::Deserialize;
use serde_json::{json, Value};

use super::events::{DemoCtx, DemoEvent};
use super::faults::ToolFaults;
use super::tools;

/// The structured answer of the Weather agent.
#[derive(Debug, Deserialize, openai_agents::schemars::JsonSchema)]
#[allow(dead_code)]
pub struct WeatherReport {
    /// City the report is about.
    city: String,
    /// Temperature in Celsius.
    celsius: f32,
    /// One-line summary.
    summary: String,
}

const TRIAGE_INSTRUCTIONS: &str = "\
You are the Triage agent of a demo assistant. Route every request exactly like this:
- Billing, invoice or customer-record questions: hand off to Billing.
- Weather or temperature questions: hand off to Weather.
- City metrics, slow reports or inventory questions: hand off to Ops.
- Translation requests: use translate_to_french / translate_to_spanish. Never translate yourself.
- Jokes: call how_many_jokes first, then tell that many short jokes.
- Questions about the current user or tenant: call whoami.
- Questions about the internal note: call lookup_internal_note and quote it verbatim.
- Questions about a city's facts: call mcp_city_facts. Temperature unit conversion: call mcp_convert_temperature.
- Anything else: answer briefly yourself.
When you call tools that do not depend on each other, call them in the same turn.";

fn input_text(input: &InputLike) -> String {
    match input {
        InputLike::Text(s) => s.clone(),
        InputLike::Items(items) => items
            .iter()
            .filter_map(|i| i.get("content").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn no_jailbreak() -> InputGuardrail {
    input_guardrail("no-jailbreak", |_ctx, _agent, input| {
        let text = input_text(&input).to_ascii_lowercase();
        async move {
            if text.contains("ignore the rules") || text.contains("ignore all previous") {
                GuardrailFunctionOutput::trip(json!({"reason": "jailbreak"}))
            } else {
                GuardrailFunctionOutput::pass(json!({}))
            }
        }
    })
}

fn no_secrets() -> OutputGuardrail {
    output_guardrail("no-secrets", |_ctx, _agent, output| async move {
        let text = match &output {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        if text.to_ascii_lowercase().contains("sk-live") {
            GuardrailFunctionOutput::trip(json!({"reason": "secret"}))
        } else {
            GuardrailFunctionOutput::pass(json!({}))
        }
    })
}

/// Build a fresh network for one run. Everything run-specific (faults, the faulty model, the MCP
/// decorator) is captured here, so nothing leaks into the next request.
pub fn build_network(
    model: Arc<dyn Model>,
    faults: &ToolFaults,
    mcp: Option<Arc<dyn McpServer>>,
) -> Result<Agent, openai_agents::AgentsError> {
    let billing = Agent::new("Billing")
        .handoff_description("Billing, invoices and customer records.")
        .instructions(
            "You answer billing questions. Use lookup_customer for customer records and \
             summarize the result briefly. Never invent contact details.",
        )
        .model(Arc::clone(&model))
        .tools(vec![tools::lookup_customer(faults)])
        .output_guardrails(vec![no_secrets()]);

    let weather = Agent::new("Weather")
        .handoff_description("Weather and temperature questions.")
        .instructions(
            "You are the weather agent. Use get_weather for conditions and get_temperature for \
             temperatures; call both for several cities in the same turn when asked. Always \
             answer with the requested JSON object (city, celsius, summary); for several cities \
             report the first one and mention the others in the summary.",
        )
        .model(Arc::clone(&model))
        .tools(vec![
            tools::get_weather(faults),
            tools::get_temperature(faults),
        ])
        .output_type(Arc::new(AgentOutputSchema::of::<WeatherReport>()?))
        .output_guardrails(vec![no_secrets()]);

    let ops = Agent::new("Ops")
        .handoff_description("City metrics, slow reports and inventory.")
        .instructions(
            "You are the operations agent. Use get_city_metrics for city metrics (call it for \
             several cities in the same turn when asked), slow_report for the quarterly report, \
             and flaky_inventory for inventory. If a tool fails, read the error and retry the \
             same call once. Report results briefly.",
        )
        .model(Arc::clone(&model))
        .tools(vec![
            tools::get_city_metrics(faults),
            tools::slow_report(faults),
            tools::flaky_inventory(faults),
        ])
        .output_guardrails(vec![no_secrets()]);

    let french = Agent::new("french_agent")
        .instructions(
            "You translate the user's message to French. Reply with the translation only.",
        )
        .model(Arc::clone(&model));
    let spanish = Agent::new("spanish_agent")
        .instructions(
            "You translate the user's message to Spanish. Reply with the translation only.",
        )
        .model(Arc::clone(&model));

    let triage_tools = vec![
        tools::how_many_jokes(faults),
        tools::whoami(faults),
        tools::lookup_internal_note(faults),
        french.as_tool(AsToolConfig {
            name: Some("translate_to_french".into()),
            description: Some("Translate the user's message to French".into()),
            ..Default::default()
        }),
        spanish.as_tool(AsToolConfig {
            name: Some("translate_to_spanish".into()),
            description: Some("Translate the user's message to Spanish".into()),
            ..Default::default()
        }),
    ];

    let mut triage = Agent::new("Triage")
        .instructions(TRIAGE_INSTRUCTIONS)
        .model(model)
        .tools(triage_tools)
        .handoffs(vec![handoff(billing), handoff(weather), handoff(ops)])
        .input_guardrails(vec![no_jailbreak()])
        .output_guardrails(vec![no_secrets()]);
    if let Some(server) = mcp {
        triage = triage.mcp_servers(vec![server]);
    }
    Ok(triage)
}

/// Turn the three recoverable run errors into a friendly final answer. The answer has the right
/// shape for the agent that was running (the Weather agent has a structured output type).
pub fn error_handlers() -> RunErrorHandlers {
    RunErrorHandlers::default()
        .on_max_turns(|input| {
            handled(
                input,
                "max_turns",
                "The agent ran out of turns before it could finish.",
            )
        })
        .on_model_refusal(|input| {
            handled(
                input,
                "model_refusal",
                "The model declined to answer this request.",
            )
        })
        .on_invalid_final_output(|input| {
            handled(
                input,
                "invalid_final_output",
                "The model's answer did not match the requested format.",
            )
        })
}

async fn handled(
    input: RunErrorHandlerInput,
    kind: &'static str,
    message: &'static str,
) -> Result<Option<RunErrorHandlerResult>, openai_agents::AgentsError> {
    if let Some(demo) = input.context.context::<DemoCtx>() {
        demo.emit
            .emit(DemoEvent::Handled {
                kind: kind.into(),
                message: message.into(),
            })
            .await;
    }
    let output = if input.run_data.last_agent.output_type.is_some() {
        json!({"city": "n/a", "celsius": 0.0, "summary": message})
    } else {
        Value::String(message.to_string())
    };
    Ok(Some(RunErrorHandlerResult::new(output)))
}
