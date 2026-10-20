//! What a span or trace looks like on its way out (Rust-only, X-018).
//!
//! [`Span`] and [`Trace`] are the SDK's own types and are not
//! `Serialize`, so an exporter needs a shape of its own. This one is deliberately flat: an event
//! kind, ids, a name, timestamps, and whatever the span carries — so a file, a console or an
//! OpenTelemetry collector can all take it without knowing about the SDK.
//!
//! Nothing here carries a request body, a tool argument or a model output: the SDK's spans never
//! hold them, and this module does not add them.

use serde_json::{json, Value};

use opai_agents_core::tracing::{Span, SpanData, Trace};

/// One thing that happened, ready to be written somewhere.
#[derive(Debug, Clone, PartialEq)]
pub enum ExportEvent {
    /// A trace opened.
    TraceStart(Value),
    /// A trace closed.
    TraceEnd(Value),
    /// A span opened.
    SpanStart(Value),
    /// A span closed; this is the one that carries the duration and any error.
    SpanEnd(Value),
}

impl ExportEvent {
    /// The JSON of this event, with a `kind` field saying which one it is.
    pub fn to_json(&self) -> Value {
        let (kind, body) = match self {
            Self::TraceStart(v) => ("trace_start", v),
            Self::TraceEnd(v) => ("trace_end", v),
            Self::SpanStart(v) => ("span_start", v),
            Self::SpanEnd(v) => ("span_end", v),
        };
        let mut out = body.clone();
        if let Value::Object(map) = &mut out {
            map.insert("kind".into(), json!(kind));
        }
        out
    }
}

/// Seconds since the epoch, as a float; `None` before it (a span that has not ended).
fn epoch_secs(time: Option<std::time::SystemTime>) -> Option<f64> {
    time.and_then(|t| {
        t.duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs_f64())
    })
}

/// A trace, as JSON.
pub fn trace_to_json(trace: &Trace) -> Value {
    json!({
        "trace_id": trace.trace_id,
        "workflow_name": trace.workflow_name,
        "group_id": trace.group_id,
        "metadata": trace.metadata,
    })
}

/// A span, as JSON: the fields every exporter wants, plus what its own kind carries.
pub fn span_to_json(span: &Span) -> Value {
    let mut out = json!({
        "span_id": span.span_id,
        "parent_id": span.parent_id,
        "started_at": epoch_secs(Some(span.started_at)),
        "ended_at": epoch_secs(span.ended_at),
    });
    let (kind, name, extra) = match &span.data {
        SpanData::Agent { name } => ("agent", Some(name.clone()), None),
        SpanData::Function { name } => ("function", Some(name.clone()), None),
        SpanData::Generation { model } => ("generation", Some(model.clone()), None),
        SpanData::Custom { name } => ("custom", Some(name.clone()), None),
        SpanData::Handoff {
            from_agent,
            to_agent,
        } => (
            "handoff",
            to_agent.clone(),
            Some(json!({"from_agent": from_agent})),
        ),
        SpanData::Response { response_id } => ("response", response_id.clone(), None),
        SpanData::Guardrail { name } => ("guardrail", Some(name.clone()), None),
        SpanData::Task { name, usage } => (
            "task",
            Some(name.clone()),
            usage.as_ref().map(|u| json!({"usage": u})),
        ),
        SpanData::Turn {
            turn,
            agent_name,
            usage,
        } => (
            "turn",
            Some(agent_name.clone()),
            Some(json!({"turn": turn, "usage": usage})),
        ),
    };
    if let Value::Object(map) = &mut out {
        map.insert("span_kind".into(), json!(kind));
        map.insert("name".into(), json!(name));
        if let Some(Value::Object(extra)) = extra {
            for (k, v) in extra {
                map.insert(k, v);
            }
        }
        if let Some(error) = &span.error {
            map.insert(
                "error".into(),
                json!({"message": error.message, "data": error.data}),
            );
        }
    }
    out
}
