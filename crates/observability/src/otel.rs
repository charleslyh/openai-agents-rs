//! Spans as OpenTelemetry spans, in whatever pipeline the process has installed (Rust-only, X-018).
//!
//! An agent run is a trace with spans — turns, generations, tool calls, handoffs — which is what an
//! OpenTelemetry backend already draws. [`OtelSink`] hands the same events to the tracer the
//! application installed, so a run shows up next to the HTTP requests and database queries around
//! it, in Jaeger, in Grafana or in a vendor's UI, without the SDK knowing which of those it is.
//!
//! Two things are lost on the way, and are said here rather than papered over:
//!
//! * **The SDK's ids are attributes.** The OpenTelemetry API hands out span and trace ids itself;
//!   it does not take someone else's. `agents.span_id` and `agents.parent_id` carry ours, so a
//!   span can still be matched to the trace it belongs to.
//! * **Parenting is what the caller has.** Our spans are exported one by one from a background
//!   thread, where there is no ambient span, so each becomes a root span carrying its parent id as
//!   an attribute. Matching them afterwards is what the ids are for.
//!
//! Only the API is depended on: the exporter — OTLP, Jaeger, stdout — is the application's choice,
//! which is also what keeps this crate from pulling a gRPC stack into every build.
//!
//! ```rust,no_run
//! // The application installs the pipeline it wants, once, at startup:
//! // opentelemetry_otlp::new_pipeline().tracing()...install_batch(opentelemetry_sdk::runtime::Tokio)
//! use opai_agents_observability::{BatchedProcessor, OtelSink};
//!
//! let processor = BatchedProcessor::spawn(OtelSink::new(), 1024, std::time::Duration::from_millis(200));
//! # let _ = processor;
//! ```

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use opentelemetry::global::tracer;
use opentelemetry::trace::{Span, SpanBuilder, Status};
use opentelemetry::{Context, KeyValue};
use serde_json::Value;

use super::batched::EventSink;
use super::export::ExportEvent;

/// The name this sink's tracer is registered under.
pub const TRACER_NAME: &str = "opai-agents";

/// Writes events as OpenTelemetry spans, through the tracer the process installed.
pub struct OtelSink {
    tracer: opentelemetry::global::BoxedTracer,
}

impl OtelSink {
    /// Use the global tracer provider.
    ///
    /// With no provider installed this writes nothing and costs nothing: the global tracer is a
    /// no-op until one is set.
    pub fn new() -> Self {
        Self {
            tracer: tracer(TRACER_NAME),
        }
    }
}

impl Default for OtelSink {
    fn default() -> Self {
        Self::new()
    }
}

impl EventSink for OtelSink {
    fn write(&mut self, events: &[ExportEvent]) -> Result<(), String> {
        for event in events {
            self.emit(event);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "opentelemetry"
    }
}

impl OtelSink {
    /// One OpenTelemetry span per ended span; a started span has no end time yet, and a trace has
    /// no span of its own in OpenTelemetry — its id is on every span already.
    fn emit(&self, event: &ExportEvent) {
        let body = match event {
            ExportEvent::SpanEnd(body) => body,
            _ => return,
        };
        let name = body
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("span")
            .to_string();
        let started = time_of(body.get("started_at"));
        let ended = time_of(body.get("ended_at")).or(started);
        let mut attributes = vec![
            KeyValue::new("agents.span_kind", kind_of(body)),
            KeyValue::new("agents.span_id", id_of(body, "span_id")),
        ];
        let parent = id_of(body, "parent_id");
        if !parent.is_empty() {
            attributes.push(KeyValue::new("agents.parent_id", parent));
        }
        if let Some(turn) = body.get("turn").and_then(Value::as_u64) {
            attributes.push(KeyValue::new("agents.turn", turn as i64));
        }
        let status = match body
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
        {
            Some(message) => {
                attributes.push(KeyValue::new("agents.error", message.to_string()));
                Status::error(message.to_string())
            }
            None => Status::Ok,
        };
        let mut builder = SpanBuilder::from_name(name).with_attributes(attributes);
        if let Some(started) = started {
            builder = builder.with_start_time(started);
        }
        // Exported from the writer thread, where there is no ambient span: a root span, with our
        // parent id as an attribute.
        let mut span = builder.start_with_context(&self.tracer, &Context::new());
        for (key, value) in usage_of(body.get("usage").unwrap_or(&Value::Null)) {
            span.set_attribute(KeyValue::new(key, value));
        }
        span.set_status(status);
        if let Some(ended) = ended {
            span.end_with_timestamp(ended);
        } else {
            span.end();
        }
    }
}

impl std::fmt::Debug for OtelSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OtelSink")
            .field("tracer", &TRACER_NAME)
            .finish()
    }
}

fn kind_of(body: &Value) -> String {
    body.get("span_kind")
        .and_then(Value::as_str)
        .unwrap_or("custom")
        .to_string()
}

fn id_of(body: &Value, key: &str) -> String {
    body.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// The SDK writes timestamps as seconds since the epoch; `None` when absent or not a number.
fn time_of(value: Option<&Value>) -> Option<SystemTime> {
    let seconds = value.and_then(Value::as_f64)?;
    let whole = seconds.floor() as u64;
    let nanos = ((seconds - seconds.floor()) * 1_000_000_000.0) as u32;
    Some(UNIX_EPOCH + Duration::new(whole, nanos))
}

/// Usage as `agents.usage.<field>` attributes, so a token count is a number a backend can chart.
fn usage_of(usage: &Value) -> Vec<(&'static str, f64)> {
    let mut out = Vec::new();
    for (field, key) in [
        ("input_tokens", "agents.usage.input_tokens"),
        ("output_tokens", "agents.usage.output_tokens"),
        ("total_tokens", "agents.usage.total_tokens"),
        ("requests", "agents.usage.requests"),
    ] {
        if let Some(value) = usage.get(field).and_then(Value::as_f64) {
            out.push((key, value));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_span_event_becomes_a_span() {
        // No provider is installed, so the tracer is a no-op: this checks the mapping runs and
        // reads the JSON, not that anything was exported.
        let mut sink = OtelSink::new();
        let event = ExportEvent::SpanEnd(json!({
            "span_kind": "turn",
            "name": "Ops",
            "span_id": "span_1",
            "parent_id": "span_0",
            "started_at": 1_700_000_000.5,
            "ended_at": 1_700_000_001.0,
            "turn": 1,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }));
        sink.write(&[event]).expect("write");
    }

    #[test]
    fn a_missing_timestamp_and_a_negative_one_are_handled() {
        assert_eq!(time_of(None), None);
        assert_eq!(time_of(Some(&json!("not a number"))), None);
        assert!(time_of(Some(&json!(1_700_000_000.25))).is_some());
    }

    #[test]
    fn usage_becomes_attributes_only_when_it_is_a_number() {
        let out = usage_of(&json!({"input_tokens": 10, "output_tokens": "many"}));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "agents.usage.input_tokens");
    }
}
