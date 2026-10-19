//! Tracing
//!
//! Web demo checklist: "Tracing waterfall".
//!
//! A custom `TracingProcessor` replaces the default exporters and receives every span of the
//! run (agent, turn, generation, function, ...). The web demo draws the same data as a
//! waterfall; here each finished span is printed.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p openai-agents-examples --example tracing
//! ```

mod common;

use std::sync::Arc;
use std::time::SystemTime;

use openai_agents::tracing::{set_trace_processors, Span, SpanData, Trace, TracingProcessor};
use openai_agents::{function_tool, Agent, RunConfig, RunOptions, Runner};

#[function_tool(description = "Get operations metrics for a city")]
async fn get_city_metrics(city: String) -> String {
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    format!("{city}: 1200 orders today, 98% on time")
}

/// Prints each span as it ends.
struct PrintSpans;

impl TracingProcessor for PrintSpans {
    fn on_trace_start(&self, _trace: &Trace) {}

    fn on_trace_end(&self, _trace: &Trace) {}

    fn on_span_start(&self, _span: &Span) {}

    fn on_span_end(&self, span: &Span) {
        let (kind, name) = match &span.data {
            SpanData::Agent { name } => ("agent", name.clone()),
            SpanData::Function { name } => ("function", name.clone()),
            SpanData::Generation { model } => ("generation", model.clone()),
            SpanData::Turn {
                turn, agent_name, ..
            } => ("turn", format!("{turn} · {agent_name}")),
            SpanData::Guardrail { name } => ("guardrail", name.clone()),
            SpanData::Handoff {
                from_agent,
                to_agent,
            } => (
                "handoff",
                format!("{from_agent} -> {}", to_agent.as_deref().unwrap_or("?")),
            ),
            SpanData::Task { .. } => ("task", "run".to_string()),
            SpanData::Response { .. } => ("response", "response".to_string()),
            SpanData::Custom { name } => ("custom", name.clone()),
        };
        let ms = span
            .ended_at
            .unwrap_or_else(SystemTime::now)
            .duration_since(span.started_at)
            .map_or(0, |d| d.as_millis());
        println!("[span] {kind:<10} {name} ({ms} ms)");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_trace_processors(vec![Arc::new(PrintSpans)]);

    let agent = Agent::new("Ops")
        .instructions("Use get_city_metrics for every city you are asked about.")
        .model(common::model()?)
        .tools(vec![get_city_metrics()]);

    let mut run_config = RunConfig::default();
    run_config.tracing.workflow_name = Some("tracing-example".into());
    let mut options = RunOptions::default();
    options.run_config = run_config;

    let input = "Give me the city metrics for Berlin.";
    let result = Runner::run(&agent, input, options).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
