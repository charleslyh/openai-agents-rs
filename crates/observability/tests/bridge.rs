//! The bridge puts SDK events into the application's `tracing` subscriber.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use openai_agents_observability::bridge::{level_of, TARGET};
use openai_agents_observability::{EventSink, ExportEvent, TracingSink};
use serde_json::json;
use tracing::field::{Field, Visit};
use tracing::{Level, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::{registry, Layer};

#[derive(Debug, Clone)]
struct Captured {
    level: Level,
    target: String,
    fields: BTreeMap<String, String>,
}

#[derive(Default, Clone)]
struct Capture(Arc<Mutex<Vec<Captured>>>);

struct Fields<'a>(&'a mut BTreeMap<String, String>);

impl Visit for Fields<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().to_string(), format!("{value:?}"));
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
}

impl<S: Subscriber> Layer<S> for Capture {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut fields = BTreeMap::new();
        event.record(&mut Fields(&mut fields));
        self.0.lock().expect("capture").push(Captured {
            level: *event.metadata().level(),
            target: event.metadata().target().to_string(),
            fields,
        });
    }
}

fn run(events: &[ExportEvent]) -> Vec<Captured> {
    let capture = Capture::default();
    let subscriber = registry().with(capture.clone());
    tracing::subscriber::with_default(subscriber, || {
        TracingSink::new().write(events).expect("write");
    });
    let out = capture.0.lock().expect("capture").clone();
    out
}

fn span_end(extra: serde_json::Value) -> ExportEvent {
    let mut body = json!({
        "span_kind": "function",
        "name": "lookup",
        "span_id": "span_1",
        "parent_id": "span_0",
        "started_at": 10.0,
        "ended_at": 10.25,
    });
    for (k, v) in extra.as_object().expect("object") {
        body[k] = v.clone();
    }
    ExportEvent::SpanEnd(body)
}

#[test]
fn a_span_end_becomes_one_event_under_the_sdk_target() {
    let events = run(&[span_end(json!({}))]);
    assert_eq!(events.len(), 1);
    let event = &events[0];
    assert_eq!(event.target, TARGET);
    assert_eq!(event.level, Level::DEBUG);
    assert_eq!(event.fields["kind"], "span_end");
    assert_eq!(event.fields["span_kind"], "function");
    assert_eq!(event.fields["name"], "lookup");
    assert_eq!(event.fields["span_id"], "span_1");
    assert_eq!(event.fields["parent_id"], "span_0");
    assert_eq!(event.fields["duration_ms"], "250");
    assert!(!event.fields.contains_key("error"));
}

#[test]
fn a_failed_span_is_a_warning_carrying_the_message() {
    let events = run(&[span_end(json!({"error": {"message": "boom"}}))]);
    assert_eq!(events[0].level, Level::WARN);
    assert_eq!(events[0].fields["error"], "boom");
}

#[test]
fn usage_and_turn_travel_as_numbers() {
    let events = run(&[span_end(json!({
        "span_kind": "turn",
        "turn": 2,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }))]);
    assert_eq!(events[0].fields["turn"], "2");
    assert_eq!(events[0].fields["input_tokens"], "10");
    assert_eq!(events[0].fields["output_tokens"], "5");
}

#[test]
fn starts_and_trace_events_are_trace_level() {
    let events = run(&[
        ExportEvent::TraceStart(json!({"trace_id": "t", "workflow_name": "w"})),
        ExportEvent::SpanStart(json!({"span_kind": "turn", "span_id": "s"})),
        ExportEvent::TraceEnd(json!({"trace_id": "t"})),
    ]);
    assert_eq!(events.len(), 3);
    assert!(events.iter().all(|e| e.level == Level::TRACE));
    assert_eq!(events[0].fields["workflow"], "w");
    assert_eq!(events[0].fields["trace_id"], "t");
}

/// Spans carry no bodies, and the bridge adds none: an extra key in the JSON is not forwarded.
#[test]
fn only_the_documented_fields_are_forwarded() {
    let events = run(&[span_end(json!({"arguments": "SECRET", "output": "SECRET"}))]);
    for value in events[0].fields.values() {
        assert!(!value.contains("SECRET"), "forwarded: {value}");
    }
}

#[test]
fn the_level_follows_the_outcome() {
    assert_eq!(level_of(&span_end(json!({}))), Level::DEBUG);
    assert_eq!(
        level_of(&span_end(json!({"error": {"message": "x"}}))),
        Level::WARN
    );
    assert_eq!(level_of(&ExportEvent::SpanStart(json!({}))), Level::TRACE);
}
