//! Exporting traces, and counting what a run cost (Rust-only, X-018)
//!
//! A `TracingProcessor` is called from inside the run loop, so anything it does there is on the
//! critical path: a processor that writes to a file or a collector would make every span cost a
//! round trip, and a slow one would slow the agent down.
//!
//! `BatchedProcessor` does as little as it can in the callback — turn the span into JSON and hand
//! it to a queue — and writes on a thread of its own. A full queue drops the event instead of
//! waiting, and `dropped()` says how many were lost. `force_flush` asks the writer to catch up and
//! waits at most two seconds.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example trace_export
//! TRACE_DIR=/tmp/traces cargo run -p opai-agents-examples --example trace_export
//! ```
//!
//! No key is needed: without `OPENAI_API_KEY` the offline model answers.
//!
//! The JSONL file holds one object per event; `jq <file>` reads it.

mod common;

use std::sync::Arc;
use std::time::Duration;

use opai_agents::observability::{BatchedProcessor, JsonlSink, Metrics, MetricsSink, PriceTable};
use opai_agents::tracing::{set_trace_processors, TracingProcessor};
use opai_agents::{function_tool, Agent, RunOptions, Runner};

#[function_tool(description = "Get operations metrics for a city")]
async fn get_city_metrics(city: String) -> String {
    tokio::time::sleep(Duration::from_millis(300)).await;
    format!("{city}: 1200 orders today, 98% on time")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("TRACE_DIR").unwrap_or_else(|_| "/tmp/agents-traces".into());
    let path = std::path::PathBuf::from(&dir).join("traces.jsonl");
    let sink = JsonlSink::open(&path)?;
    println!("writing spans to {}", sink.path().display());

    // 1024 events may be waiting; a lone event sits at most 200 ms.
    let processor = BatchedProcessor::spawn(sink, 1024, Duration::from_millis(200));
    set_trace_processors(vec![processor.clone()]);

    let agent = Agent::new("Ops")
        .instructions("Use get_city_metrics for every city you are asked about.")
        .model(common::model_or_offline()?)
        .tools(vec![get_city_metrics()]);

    let result = Runner::run(
        &agent,
        "Give me the city metrics for Berlin.",
        RunOptions::new().workflow_name("trace-export"),
    )
    .await?;
    println!(
        "agent> {}",
        result.final_output_as_str().unwrap_or_default()
    );

    // The writer is behind: ask it to catch up before the process ends.
    processor.force_flush();
    println!(
        "exported to {} ({} events dropped)",
        path.display(),
        processor.dropped()
    );

    // The same events, counted instead of written: an operator's numbers, not a log.
    let metrics = Arc::new(Metrics::new());
    let mut prices = PriceTable::new();
    prices.set(
        std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".into()),
        opai_agents::observability::Price::new(2.50, 10.00),
    );
    let counting = BatchedProcessor::spawn(
        MetricsSink::new(Arc::clone(&metrics), Arc::new(prices)),
        1024,
        Duration::from_millis(200),
    );
    set_trace_processors(vec![counting.clone()]);
    Runner::run(
        &agent,
        "And for Paris?",
        RunOptions::new().workflow_name("trace-export"),
    )
    .await?;
    counting.force_flush();
    println!(
        "turns={} tool_calls={} tool_failures={} failure_rate={:.0}% tokens={}+{} cost=${:.4}",
        metrics.turns(),
        metrics.tool_calls(),
        metrics.tool_failures(),
        metrics.tool_failure_rate() * 100.0,
        metrics.input_tokens(),
        metrics.output_tokens(),
        metrics.cost_usd(),
    );
    Ok(())
}
