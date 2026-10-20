//! Collects SDK tracing spans per run, for the waterfall in the page.
//!
//! Processors are process-global and `Span` carries no trace id. Every run therefore sets
//! `RunConfig.workflow_name` to its run id: the root `Task` span is named after it, and each
//! other span is attributed to the run of its parent.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use opai_agents::tracing::{set_trace_processors, Span, SpanData, Trace, TracingProcessor};

use crate::events::SpanDto;

/// More tracked spans than this means something leaked; start over rather than grow.
const MAX_TRACKED_SPANS: usize = 50_000;

#[derive(Default)]
struct Inner {
    /// span id -> run id
    owner: HashMap<String, String>,
    /// run id -> finished spans
    buckets: HashMap<String, Vec<SpanDto>>,
}

#[derive(Default)]
pub struct TraceCollector {
    inner: Mutex<Inner>,
}

impl TraceCollector {
    /// Replace the default processors (which keep every span forever) with this collector.
    pub fn install() -> Arc<Self> {
        let collector = Arc::new(Self::default());
        set_trace_processors(vec![Arc::clone(&collector) as Arc<dyn TracingProcessor>]);
        collector
    }

    /// Start collecting for `run_id`.
    pub fn register(&self, run_id: &str) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.buckets.entry(run_id.to_string()).or_default();
        }
    }

    /// Stop collecting for `run_id` and return what was gathered, oldest first.
    pub fn take(&self, run_id: &str) -> Vec<SpanDto> {
        let Ok(mut inner) = self.inner.lock() else {
            return Vec::new();
        };
        inner.owner.retain(|_, run| run != run_id);
        let mut spans = inner.buckets.remove(run_id).unwrap_or_default();
        spans.sort_by_key(|s| s.start_ms);
        spans
    }
}

fn millis(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn describe(data: &SpanData) -> (&'static str, String) {
    match data {
        SpanData::Agent { name } => ("agent", name.clone()),
        SpanData::Function { name } => ("function", name.clone()),
        SpanData::Generation { model } => ("generation", model.clone()),
        SpanData::Custom { name } => ("custom", name.clone()),
        SpanData::Handoff {
            from_agent,
            to_agent,
        } => (
            "handoff",
            format!("{from_agent} -> {}", to_agent.as_deref().unwrap_or("?")),
        ),
        SpanData::Response { .. } => ("response", "response".to_string()),
        SpanData::Guardrail { name } => ("guardrail", name.clone()),
        SpanData::Task { .. } => ("task", "run".to_string()),
        SpanData::Turn {
            turn, agent_name, ..
        } => ("turn", format!("turn {turn} · {agent_name}")),
    }
}

impl TracingProcessor for TraceCollector {
    fn on_trace_start(&self, _trace: &Trace) {}

    fn on_trace_end(&self, _trace: &Trace) {}

    fn on_span_start(&self, span: &Span) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        if inner.owner.len() > MAX_TRACKED_SPANS {
            inner.owner.clear();
        }
        let run = match (&span.parent_id, &span.data) {
            (Some(parent), _) => inner.owner.get(parent).cloned(),
            (None, SpanData::Task { name, .. }) if inner.buckets.contains_key(name) => {
                Some(name.clone())
            }
            _ => None,
        };
        if let Some(run) = run {
            inner.owner.insert(span.span_id.clone(), run);
        }
    }

    fn on_span_end(&self, span: &Span) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        let Some(run) = inner.owner.get(&span.span_id).cloned() else {
            return;
        };
        let (kind, name) = describe(&span.data);
        let dto = SpanDto {
            id: span.span_id.clone(),
            parent: span.parent_id.clone(),
            kind: kind.to_string(),
            name,
            start_ms: millis(span.started_at).into(),
            end_ms: millis(span.ended_at.unwrap_or_else(SystemTime::now)).into(),
            error: span
                .error
                .as_ref()
                .map(|e| crate::events::truncate(&e.message, 200)),
        };
        if let Some(bucket) = inner.buckets.get_mut(&run) {
            bucket.push(dto);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn span(id: &str, parent: Option<&str>, data: SpanData) -> Span {
        let start = SystemTime::UNIX_EPOCH + Duration::from_secs(10);
        Span {
            span_id: id.into(),
            parent_id: parent.map(str::to_string),
            data,
            started_at: start,
            ended_at: Some(start + Duration::from_millis(5)),
            error: None,
        }
    }

    #[test]
    fn spans_follow_their_parent_to_the_run() {
        let c = TraceCollector::default();
        c.register("run-1");
        let task = span(
            "t",
            None,
            SpanData::Task {
                name: "run-1".into(),
                usage: None,
            },
        );
        let child = span(
            "c",
            Some("t"),
            SpanData::Agent {
                name: "Triage".into(),
            },
        );
        let stranger = span(
            "s",
            None,
            SpanData::Task {
                name: "other".into(),
                usage: None,
            },
        );
        for s in [&task, &child, &stranger] {
            c.on_span_start(s);
        }
        for s in [&child, &task, &stranger] {
            c.on_span_end(s);
        }
        let spans = c.take("run-1");
        assert_eq!(spans.len(), 2);
        assert!(c.take("run-1").is_empty());
    }
}
