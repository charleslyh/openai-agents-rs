//! Plain trace and span data.

use std::time::SystemTime;

use serde_json::Value;

/// Per-run tracing settings (Python: `RunConfig.tracing` / `TracingConfig`).
#[derive(Debug, Clone, Default)]
pub struct TracingConfig {
    /// Whether the runner wraps the run in a task span and each turn in a turn span
    /// (Python: `include_task_and_turn_spans`; `None` means true).
    pub include_task_and_turn_spans: Option<bool>,
}

impl TracingConfig {
    /// Whether task and turn spans are created (Python: `include_task_and_turn_spans(config)`).
    pub fn includes_task_and_turn_spans(config: Option<&TracingConfig>) -> bool {
        config
            .and_then(|c| c.include_task_and_turn_spans)
            .unwrap_or(true)
    }
}

/// Optional trace identity supplied by the caller (Python: `RunConfig.trace_id` / `group_id` /
/// `trace_metadata`).
#[derive(Debug, Clone, Default)]
pub struct TraceConfig {
    /// Use this trace id instead of generating one.
    pub trace_id: Option<String>,
    /// Grouping identifier, e.g. a chat thread id.
    pub group_id: Option<String>,
    /// Arbitrary metadata attached to the trace.
    pub metadata: Option<Value>,
}

/// A workflow-level trace (Python: `Trace`).
#[derive(Debug, Clone)]
pub struct Trace {
    /// Trace id.
    pub trace_id: String,
    /// Workflow name.
    pub workflow_name: String,
    /// Grouping identifier (Python: `RunConfig.group_id`).
    pub group_id: Option<String>,
    /// Extra metadata (Python: `RunConfig.trace_metadata`).
    pub metadata: Option<Value>,
}

/// Span data kinds.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SpanData {
    /// Agent span.
    Agent {
        /// Agent name.
        name: String,
    },
    /// Function/tool span.
    Function {
        /// Tool name.
        name: String,
    },
    /// Model generation span.
    Generation {
        /// Model name if known.
        model: String,
    },
    /// Custom span.
    Custom {
        /// Span name.
        name: String,
    },
    /// Handoff span (Python: `HandoffSpanData`).
    Handoff {
        /// Agent the run was handed off from.
        from_agent: String,
        /// Agent the run was handed off to.
        to_agent: Option<String>,
    },
    /// Model response span (Python: `ResponseSpanData`).
    Response {
        /// Responses API response id when known.
        response_id: Option<String>,
    },
    /// Guardrail span (Python: `GuardrailSpanData`).
    Guardrail {
        /// Guardrail name.
        name: String,
    },
    /// One top-level run (Python: `TaskSpanData`).
    Task {
        /// Workflow name.
        name: String,
        /// Usage of the whole run: `input_tokens`, `output_tokens`, `cached_input_tokens`,
        /// `cache_write_input_tokens`, `requests`, `total_tokens`. Absent when nothing was used.
        usage: Option<Value>,
    },
    /// One agent-loop turn (Python: `TurnSpanData`).
    Turn {
        /// 1-based turn number.
        turn: usize,
        /// Agent that ran the turn.
        agent_name: String,
        /// Usage of the turn: `input_tokens`, `output_tokens`, `cached_input_tokens`,
        /// `cache_write_input_tokens`. Absent when nothing was used.
        usage: Option<Value>,
    },
}

/// An error recorded on a span (Python: `SpanError`).
///
/// A span that ends without one completed. Exporters (OTLP, Langfuse, …) map this to their
/// backend's error field, which is what makes a failed run visible on a timeline.
#[derive(Debug, Clone)]
pub struct SpanError {
    /// Human-readable message.
    pub message: String,
    /// Structured detail; `None` when there is none.
    pub data: Option<Value>,
}

/// A span within a trace (Python: `Span`).
#[derive(Debug, Clone)]
pub struct Span {
    /// Span id.
    pub span_id: String,
    /// Span this one is nested in (Python: `parent_id`); `None` for a direct child of the trace.
    pub parent_id: Option<String>,
    /// Span payload.
    pub data: SpanData,
    /// When the span was opened.
    pub started_at: SystemTime,
    /// When the span ended; `None` while it is open (including in `on_span_start`).
    pub ended_at: Option<SystemTime>,
    /// Error attached to the span, if the work it covers failed.
    pub error: Option<SpanError>,
}
