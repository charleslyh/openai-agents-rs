//! Agents tracing (Python: `agents.tracing` local subset).
//!
//! Cloud OpenAI export is intentionally unsupported (D-004).

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::Value;
use uuid::Uuid;

use crate::usage::Usage;

static TRACING_DISABLED: AtomicBool = AtomicBool::new(false);
static PROCESSOR: OnceLock<Mutex<Option<Arc<dyn TracingProcessor>>>> = OnceLock::new();

tokio::task_local! {
    /// Per-run override set by the runner (Python: `RunConfig.tracing_disabled`).
    static RUN_TRACING_DISABLED: bool;
}

/// The span new spans attach to, and what to fall back to when it ends.
#[derive(Debug, Clone, Default)]
struct CurrentSpan {
    id: Option<String>,
    parent: Option<String>,
    is_turn: bool,
}

tokio::task_local! {
    /// The current span of this task (Python: the `contextvars` current span). Only spans that
    /// own a stretch of the run (task, agent, turn) become current; leaf spans just read it.
    static CURRENT_SPAN: RefCell<CurrentSpan>;
}

fn current_span() -> CurrentSpan {
    CURRENT_SPAN.try_with(|c| c.borrow().clone()).unwrap_or_default()
}

fn set_current_span(span: CurrentSpan) {
    let _ = CURRENT_SPAN.try_with(|c| *c.borrow_mut() = span);
}

/// Id of the span new spans would attach to, when a run is in progress.
pub fn current_span_id() -> Option<String> {
    current_span().id
}

fn processor_slot() -> &'static Mutex<Option<Arc<dyn TracingProcessor>>> {
    PROCESSOR.get_or_init(|| Mutex::new(Some(Arc::new(InMemoryProcessor::default()))))
}

/// Disable or enable tracing globally (Python: `set_tracing_disabled`).
///
/// Per-run configuration takes precedence in the sense that tracing is off when *either* the
/// global flag or the current run's `RunConfig.tracing_disabled` is set.
pub fn set_tracing_disabled(disabled: bool) {
    TRACING_DISABLED.store(disabled, Ordering::SeqCst);
}

/// Whether tracing is disabled globally.
pub fn is_tracing_disabled() -> bool {
    TRACING_DISABLED.load(Ordering::SeqCst)
}

/// Whether tracing is disabled for the current execution context.
///
/// True when the global switch is off, or when the enclosing run disabled tracing.
pub fn tracing_disabled() -> bool {
    if is_tracing_disabled() {
        return true;
    }
    RUN_TRACING_DISABLED.try_with(|v| *v).unwrap_or(false)
}

/// Run `future` with a per-run tracing switch (Python: `RunConfig.tracing_disabled`).
///
/// The future also inherits the span that is current *where this function is called*, so a
/// task spawned from a run keeps its spans under the right parent.
pub fn with_run_tracing_disabled<F>(disabled: bool, future: F) -> impl std::future::Future<Output = F::Output>
where
    F: std::future::Future,
{
    let inherited = current_span();
    RUN_TRACING_DISABLED.scope(disabled, CURRENT_SPAN.scope(RefCell::new(inherited), future))
}

/// Run `future` with `span_id` as the current span, so spans created inside (for example a
/// nested run started by a tool) become its children. Unlike a span guard this is scoped to
/// the future, which makes it safe for futures that run concurrently.
pub fn with_current_span<F>(span_id: &str, future: F) -> impl std::future::Future<Output = F::Output>
where
    F: std::future::Future,
{
    let scoped = CurrentSpan {
        id: Some(span_id.to_string()),
        parent: current_span().id,
        is_turn: false,
    };
    CURRENT_SPAN.scope(RefCell::new(scoped), future)
}

/// Replace trace processors (Python: `set_trace_processors`).
pub fn set_trace_processors(processors: Vec<Arc<dyn TracingProcessor>>) {
    let merged = Arc::new(MultiProcessor { processors });
    *processor_slot().lock().expect("processor lock") = Some(merged);
}

/// Add a processor alongside the current one.
pub fn add_trace_processor(processor: Arc<dyn TracingProcessor>) {
    let mut slot = processor_slot().lock().expect("processor lock");
    let current = slot.take();
    let processors = match current {
        Some(p) => vec![p, processor],
        None => vec![processor],
    };
    *slot = Some(Arc::new(MultiProcessor { processors }));
}

/// Flush processors (Python: `flush_traces`).
///
/// Processors that buffer events are expected to implement [`TracingProcessor::force_flush`];
/// the default is a no-op, so this is safe to call at any time.
pub fn flush_traces() {
    if let Some(p) = processor_slot().lock().expect("processor lock").as_ref() {
        p.force_flush();
    }
}

/// Generate a trace id (Python: `gen_trace_id`).
pub fn gen_trace_id() -> String {
    format!("trace_{}", Uuid::new_v4())
}

/// Generate a span id (Python: `gen_span_id`).
pub fn gen_span_id() -> String {
    format!("span_{}", Uuid::new_v4())
}

/// Processor interface (Python: `TracingProcessor`).
pub trait TracingProcessor: Send + Sync {
    /// Called when a trace starts.
    fn on_trace_start(&self, trace: &Trace);
    /// Called when a trace ends.
    fn on_trace_end(&self, trace: &Trace);
    /// Called when a span starts.
    fn on_span_start(&self, span: &Span);
    /// Called when a span ends.
    fn on_span_end(&self, span: &Span);
    /// Flush any buffered events. Defaults to a no-op.
    fn force_flush(&self) {}
}

#[derive(Default)]
struct MultiProcessor {
    processors: Vec<Arc<dyn TracingProcessor>>,
}

impl TracingProcessor for MultiProcessor {
    fn on_trace_start(&self, trace: &Trace) {
        for p in &self.processors {
            p.on_trace_start(trace);
        }
    }
    fn on_trace_end(&self, trace: &Trace) {
        for p in &self.processors {
            p.on_trace_end(trace);
        }
    }
    fn on_span_start(&self, span: &Span) {
        for p in &self.processors {
            p.on_span_start(span);
        }
    }
    fn on_span_end(&self, span: &Span) {
        for p in &self.processors {
            p.on_span_end(span);
        }
    }
    fn force_flush(&self) {
        for p in &self.processors {
            p.force_flush();
        }
    }
}

/// In-memory processor for assertions in tests.
#[derive(Debug, Default)]
pub struct InMemoryProcessor {
    /// Traces when they started.
    pub started_traces: Mutex<Vec<Trace>>,
    /// Finished traces.
    pub traces: Mutex<Vec<Trace>>,
    /// Spans when they started.
    pub started_spans: Mutex<Vec<Span>>,
    /// Finished spans.
    pub spans: Mutex<Vec<Span>>,
}

impl TracingProcessor for InMemoryProcessor {
    fn on_trace_start(&self, trace: &Trace) {
        self.started_traces
            .lock()
            .expect("lock")
            .push(trace.clone());
    }
    fn on_trace_end(&self, trace: &Trace) {
        self.traces.lock().expect("lock").push(trace.clone());
    }
    fn on_span_start(&self, span: &Span) {
        self.started_spans.lock().expect("lock").push(span.clone());
    }
    fn on_span_end(&self, span: &Span) {
        self.spans.lock().expect("lock").push(span.clone());
    }
}

impl InMemoryProcessor {
    /// Install as the sole processor and return a handle.
    pub fn install() -> Arc<Self> {
        let proc = Arc::new(Self::default());
        set_trace_processors(vec![proc.clone()]);
        proc
    }
}

/// Per-run tracing settings (Python: `RunConfig.tracing` / `TracingConfig`).
#[derive(Debug, Clone, Default)]
pub struct TracingConfig {
    /// Key used to export this run's traces (Python: `api_key`). Cloud export is not supported
    /// (D-004), so the value is carried but never used.
    pub api_key: Option<String>,
    /// Whether the runner wraps the run in a task span and each turn in a turn span
    /// (Python: `include_task_and_turn_spans`; `None` means true).
    pub include_task_and_turn_spans: Option<bool>,
}

impl TracingConfig {
    /// Whether task and turn spans are created (Python: `include_task_and_turn_spans(config)`).
    pub fn includes_task_and_turn_spans(config: Option<&TracingConfig>) -> bool {
        config.and_then(|c| c.include_task_and_turn_spans).unwrap_or(true)
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

/// Guard that ends the trace on drop.
pub struct TraceGuard {
    trace: Trace,
    active: bool,
}

impl TraceGuard {
    /// Access the underlying trace.
    pub fn trace(&self) -> &Trace {
        &self.trace
    }
}

impl Drop for TraceGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Some(p) = processor_slot().lock().expect("lock").as_ref() {
            p.on_trace_end(&self.trace);
        }
    }
}

/// Span data kinds.
#[derive(Debug, Clone)]
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

/// A span within a trace (Python: `Span`).
#[derive(Debug, Clone)]
pub struct Span {
    /// Span id.
    pub span_id: String,
    /// Span this one is nested in (Python: `parent_id`); `None` for a direct child of the trace.
    pub parent_id: Option<String>,
    /// Span payload.
    pub data: SpanData,
}

/// Guard that ends the span on drop.
pub struct SpanGuard {
    span: Span,
    active: bool,
    /// The current span to put back when this guard owned the current-span slot.
    restore: Option<CurrentSpan>,
    usage: Usage,
}

impl SpanGuard {
    /// Access the underlying span.
    pub fn span(&self) -> &Span {
        &self.span
    }

    /// Add model usage to this span; task and turn spans export it when they end
    /// (Python: `attach_usage_to_span`). Other span kinds ignore it.
    pub fn add_usage(&mut self, usage: &Usage) {
        self.usage.add(usage);
    }
}

fn usage_is_empty(usage: &Usage) -> bool {
    usage.requests == 0
        && usage.input_tokens == 0
        && usage.output_tokens == 0
        && usage.total_tokens == 0
        && usage.input_tokens_details.cached_tokens == 0
        && usage.input_tokens_details.cache_write_tokens == 0
        && usage.output_tokens_details.reasoning_tokens == 0
}

/// Python: `turn_usage_to_span_data`.
fn turn_usage_json(usage: &Usage) -> Value {
    serde_json::json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_input_tokens": usage.input_tokens_details.cached_tokens,
        "cache_write_input_tokens": usage.input_tokens_details.cache_write_tokens,
    })
}

/// Python: `task_usage_to_span_data`.
fn task_usage_json(usage: &Usage) -> Value {
    let mut value = turn_usage_json(usage);
    value["requests"] = usage.requests.into();
    value["total_tokens"] = usage.total_tokens.into();
    value
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if self.active {
            if !usage_is_empty(&self.usage) {
                match &mut self.span.data {
                    SpanData::Task { usage, .. } => *usage = Some(task_usage_json(&self.usage)),
                    SpanData::Turn { usage, .. } => *usage = Some(turn_usage_json(&self.usage)),
                    _ => {}
                }
            }
            if let Some(p) = processor_slot().lock().expect("lock").as_ref() {
                p.on_span_end(&self.span);
            }
        }
        if let Some(previous) = self.restore.take() {
            set_current_span(previous);
        }
    }
}

/// Puts the current span back when dropped; see [`leave_turn_span`].
pub struct SpanScopeGuard {
    restore: Option<CurrentSpan>,
}

impl Drop for SpanScopeGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.restore.take() {
            set_current_span(previous);
        }
    }
}

/// While the guard lives, new spans attach to the parent of the current *turn* span instead of
/// the turn itself. Python closes the turn span before output guardrails run, so their spans
/// sit under the agent span. Does nothing when the current span is not a turn.
pub fn leave_turn_span() -> SpanScopeGuard {
    let current = current_span();
    if !current.is_turn {
        return SpanScopeGuard { restore: None };
    }
    set_current_span(CurrentSpan {
        id: current.parent.clone(),
        parent: None,
        is_turn: false,
    });
    SpanScopeGuard { restore: Some(current) }
}

/// Start a root trace (Python: `trace(...)`).
pub fn trace(workflow_name: &str) -> TraceGuard {
    trace_with_config(workflow_name, TraceConfig::default())
}

/// Start a root trace with an explicit id / group / metadata.
pub fn trace_with_config(workflow_name: &str, config: TraceConfig) -> TraceGuard {
    let t = Trace {
        trace_id: config.trace_id.unwrap_or_else(gen_trace_id),
        workflow_name: workflow_name.to_string(),
        group_id: config.group_id,
        metadata: config.metadata,
    };
    if tracing_disabled() {
        return TraceGuard { trace: t, active: false };
    }
    if let Some(p) = processor_slot().lock().expect("lock").as_ref() {
        p.on_trace_start(&t);
    }
    TraceGuard { trace: t, active: true }
}

/// Create a span under the current one. With `becomes_current` it also becomes the current span
/// until the guard drops (spans that own a stretch of the run); otherwise it is a leaf.
fn start_span(data: SpanData, becomes_current: bool) -> SpanGuard {
    let parent = current_span();
    let is_turn = matches!(data, SpanData::Turn { .. });
    let span = Span {
        span_id: gen_span_id(),
        parent_id: parent.id.clone(),
        data,
    };
    if tracing_disabled() {
        return SpanGuard { span, active: false, restore: None, usage: Usage::default() };
    }
    if let Some(p) = processor_slot().lock().expect("lock").as_ref() {
        p.on_span_start(&span);
    }
    let restore = becomes_current.then(|| {
        set_current_span(CurrentSpan {
            id: Some(span.span_id.clone()),
            parent: parent.id.clone(),
            is_turn,
        });
        parent
    });
    SpanGuard { span, active: true, restore, usage: Usage::default() }
}

/// Agent span (Python: `agent_span`).
pub fn agent_span(name: &str) -> SpanGuard {
    start_span(
        SpanData::Agent {
            name: name.to_string(),
        },
        true,
    )
}

/// Span for one whole run (Python: `task_span`). Spans created while it lives nest under it.
pub fn task_span(name: &str) -> SpanGuard {
    start_span(
        SpanData::Task {
            name: name.to_string(),
            usage: None,
        },
        true,
    )
}

/// Span for one agent-loop turn (Python: `turn_span`). Spans created while it lives nest under it.
pub fn turn_span(turn: usize, agent_name: &str) -> SpanGuard {
    start_span(
        SpanData::Turn {
            turn,
            agent_name: agent_name.to_string(),
            usage: None,
        },
        true,
    )
}

/// Function/tool span (Python: `function_span`).
pub fn function_span(name: &str) -> SpanGuard {
    start_span(
        SpanData::Function {
            name: name.to_string(),
        },
        false,
    )
}

/// Generation span (Python: `generation_span`).
pub fn generation_span(model: &str) -> SpanGuard {
    start_span(
        SpanData::Generation {
            model: model.to_string(),
        },
        false,
    )
}

/// Custom span (Python: `custom_span`).
pub fn custom_span(name: &str) -> SpanGuard {
    start_span(
        SpanData::Custom {
            name: name.to_string(),
        },
        false,
    )
}

/// Handoff span (Python: `handoff_span`).
pub fn handoff_span(from_agent: &str, to_agent: &str) -> SpanGuard {
    start_span(
        SpanData::Handoff {
            from_agent: from_agent.to_string(),
            to_agent: Some(to_agent.to_string()),
        },
        false,
    )
}

/// Model response span (Python: `response_span`).
pub fn response_span(response_id: Option<&str>) -> SpanGuard {
    start_span(
        SpanData::Response {
            response_id: response_id.map(str::to_string),
        },
        false,
    )
}

/// Guardrail span (Python: `guardrail_span`).
pub fn guardrail_span(name: &str) -> SpanGuard {
    start_span(
        SpanData::Guardrail {
            name: name.to_string(),
        },
        false,
    )
}
