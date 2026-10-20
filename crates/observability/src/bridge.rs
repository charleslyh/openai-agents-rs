//! Spans as `tracing` events, so an agent run shows up in the application's own logs (Rust-only,
//! X-018).
//!
//! The SDK's trace ([`TracingProcessor`](opai_agents_core::tracing::TracingProcessor)) and the
//! Rust `tracing` crate share a name and nothing else: an application that has set up
//! `tracing-subscriber` — a formatter, `RUST_LOG`, a rolling file, `tokio-console` — would not see
//! a turn or a tool call unless something carries them across. [`TracingSink`] is that: each event
//! becomes a `tracing` event with the target `opai_agents`, so
//! `RUST_LOG=opai_agents=debug` turns it on and the application's subscriber decides where it
//! goes.
//!
//! | SDK event | Level |
//! |-----------|-------|
//! | span ended without error | `DEBUG` |
//! | span ended with an error | `WARN` |
//! | span started, trace started or ended | `TRACE` |
//!
//! Only span *ends* are `DEBUG`: a start carries no duration or outcome, and logging both would
//! double every line.
//!
//! These are events, not `tracing` spans. A `tracing` span is parented by whichever span is
//! entered on the current thread, and the export runs on the writer thread, where none is. The
//! SDK's own `span_id` and `parent_id` are fields instead, so the tree can still be rebuilt from
//! the log.
//!
//! The fields are the ones every exporter has — kind, name, ids, turn, token counts, error
//! message. No request body, tool argument or model output is among them, because the SDK's spans
//! do not hold any.
//!
//! ```rust
//! use opai_agents_observability::{BatchedProcessor, TracingSink};
//!
//! # fn install() -> std::sync::Arc<BatchedProcessor> {
//! BatchedProcessor::spawn(TracingSink::new(), 1024, std::time::Duration::from_millis(200))
//! # }
//! ```

use serde_json::Value;
use tracing::Level;

use super::batched::EventSink;
use super::export::ExportEvent;

/// The `tracing` target every event is emitted under.
pub const TARGET: &str = "opai_agents";

/// Emits each SDK event as a `tracing` event under [`TARGET`].
#[derive(Debug, Default)]
pub struct TracingSink;

impl TracingSink {
    /// A sink that writes into the application's `tracing` subscriber.
    pub fn new() -> Self {
        Self
    }
}

impl EventSink for TracingSink {
    fn write(&mut self, events: &[ExportEvent]) -> Result<(), String> {
        for event in events {
            emit(event);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "tracing"
    }
}

/// The level an event is logged at.
pub fn level_of(event: &ExportEvent) -> Level {
    match event {
        ExportEvent::SpanEnd(body) if body.get("error").is_some() => Level::WARN,
        ExportEvent::SpanEnd(_) => Level::DEBUG,
        _ => Level::TRACE,
    }
}

fn emit(event: &ExportEvent) {
    let (kind, body) = match event {
        ExportEvent::TraceStart(body) => ("trace_start", body),
        ExportEvent::TraceEnd(body) => ("trace_end", body),
        ExportEvent::SpanStart(body) => ("span_start", body),
        ExportEvent::SpanEnd(body) => ("span_end", body),
    };
    let text = |key: &str| body.get(key).and_then(Value::as_str);
    let usage = |key: &str| {
        body.get("usage")
            .and_then(|u| u.get(key))
            .and_then(Value::as_u64)
    };
    let turn = body.get("turn").and_then(Value::as_u64);
    let input_tokens = usage("input_tokens");
    let output_tokens = usage("output_tokens");
    let error = body
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str);
    let duration_ms = match (
        body.get("started_at").and_then(Value::as_f64),
        body.get("ended_at").and_then(Value::as_f64),
    ) {
        (Some(start), Some(end)) if end >= start => Some(((end - start) * 1000.0).round() as u64),
        _ => None,
    };

    // `event!` wants its level as a constant, so one arm per level; the fields are the same.
    macro_rules! log {
        ($level:expr) => {
            tracing::event!(
                target: "opai_agents",
                $level,
                kind,
                span_kind = text("span_kind"),
                name = text("name"),
                span_id = text("span_id"),
                parent_id = text("parent_id"),
                trace_id = text("trace_id"),
                workflow = text("workflow_name"),
                turn,
                input_tokens,
                output_tokens,
                duration_ms,
                error,
            )
        };
    }
    match level_of(event) {
        Level::WARN => log!(Level::WARN),
        Level::DEBUG => log!(Level::DEBUG),
        _ => log!(Level::TRACE),
    }
}
