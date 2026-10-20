//! Agents tracing (Python: `agents.tracing` local subset).
//!
//! Cloud OpenAI export is intentionally unsupported (D-004).

mod processor;
mod types;

pub use processor::{
    add_trace_processor, flush_traces, set_trace_processors, InMemoryProcessor, TracingProcessor,
};
pub use types::{Span, SpanData, SpanError, Trace, TraceConfig, TracingConfig};

pub(crate) use processor::TracingState;

use std::cell::RefCell;
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;

use serde_json::Value;
use uuid::Uuid;

use crate::usage::Usage;

tokio::task_local! {
    /// Per-run override set by the runner (Python: `RunConfig.tracing_disabled`).
    static RUN_TRACING_DISABLED: bool;
}

tokio::task_local! {
    /// The tracing state of the runtime this task's run belongs to.
    static CURRENT_STATE: Arc<TracingState>;
}

/// The state behind `Runtime::global()`, which adopts it: spans opened outside any run, and the
/// free `set_trace_processors`-style functions, use it.
pub(crate) fn global_state() -> Arc<TracingState> {
    static GLOBAL: OnceLock<Arc<TracingState>> = OnceLock::new();
    Arc::clone(GLOBAL.get_or_init(|| Arc::new(TracingState::new())))
}

/// The state of the run this code is part of, else the global one.
fn current_state() -> Arc<TracingState> {
    CURRENT_STATE
        .try_with(Arc::clone)
        .unwrap_or_else(|_| global_state())
}

/// Run `future` reporting to `state` (see `Runtime::scope`).
pub(crate) fn scope_state<F: std::future::Future>(
    state: Arc<TracingState>,
    future: F,
) -> impl std::future::Future<Output = F::Output> {
    CURRENT_STATE.scope(state, future)
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
    CURRENT_SPAN
        .try_with(|c| c.borrow().clone())
        .unwrap_or_default()
}

fn set_current_span(span: CurrentSpan) {
    let _ = CURRENT_SPAN.try_with(|c| *c.borrow_mut() = span);
}

/// Id of the span new spans would attach to, when a run is in progress.
pub fn current_span_id() -> Option<String> {
    current_span().id
}

/// Disable or enable tracing on `Runtime::global()` (Python: `set_tracing_disabled`).
///
/// Tracing is off when *either* the runtime's switch or the current run's
/// `RunConfig.tracing_disabled` is set. For a runtime of your own use
/// `Runtime::set_tracing_disabled`.
pub fn set_tracing_disabled(disabled: bool) {
    global_state().set_disabled(disabled);
}

/// Whether tracing is disabled on `Runtime::global()`.
pub fn is_tracing_disabled() -> bool {
    global_state().disabled()
}

/// Whether tracing is disabled for the current execution context.
///
/// True when the switch of the runtime in scope is off (a run's own runtime, else the global one),
/// or when the enclosing run disabled tracing.
pub fn tracing_disabled() -> bool {
    disabled_in(&current_state())
}

fn disabled_in(state: &TracingState) -> bool {
    state.disabled() || RUN_TRACING_DISABLED.try_with(|v| *v).unwrap_or(false)
}

/// Run `future` with a per-run tracing switch (Python: `RunConfig.tracing_disabled`).
///
/// The future also inherits the span and the tracing state (the runtime's processors and switch)
/// that are current *where this function is called*, so a task spawned from a run keeps its spans
/// under the right parent and reports them to the right processor. It does not carry the
/// runtime's model provider: a run started inside such a task passes
/// `RunOptions::runtime(Runtime::current())` itself.
pub fn with_run_tracing_disabled<F>(
    disabled: bool,
    future: F,
) -> impl std::future::Future<Output = F::Output>
where
    F: std::future::Future,
{
    let inherited = current_span();
    CURRENT_STATE.scope(
        current_state(),
        RUN_TRACING_DISABLED.scope(
            disabled,
            CURRENT_SPAN.scope(RefCell::new(inherited), future),
        ),
    )
}

/// Run `future` with `span_id` as the current span, so spans created inside (for example a
/// nested run started by a tool) become its children. Unlike a span guard this is scoped to
/// the future, which makes it safe for futures that run concurrently.
pub fn with_current_span<F>(
    span_id: &str,
    future: F,
) -> impl std::future::Future<Output = F::Output>
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

/// Generate a trace id (Python: `gen_trace_id`).
pub fn gen_trace_id() -> String {
    format!("trace_{}", Uuid::new_v4())
}

/// Generate a span id (Python: `gen_span_id`).
pub fn gen_span_id() -> String {
    format!("span_{}", Uuid::new_v4())
}

/// Guard that ends the trace on drop.
pub struct TraceGuard {
    trace: Trace,
    active: bool,
    /// The state the trace started on; it ends there too.
    state: Arc<TracingState>,
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
        if let Some(p) = self.state.processor() {
            p.on_trace_end(&self.trace);
        }
    }
}

/// Guard that ends the span on drop.
pub struct SpanGuard {
    span: Span,
    active: bool,
    /// The state the span started on; it ends there too.
    state: Arc<TracingState>,
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

    /// Record a failure on this span (Python: `attach_error_to_span`). The last error wins.
    pub fn set_error(&mut self, error: SpanError) {
        self.span.error = Some(error);
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
            self.span.ended_at = Some(SystemTime::now());
            if let Some(p) = self.state.processor() {
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
    SpanScopeGuard {
        restore: Some(current),
    }
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
    let state = current_state();
    if disabled_in(&state) {
        return TraceGuard {
            trace: t,
            active: false,
            state,
        };
    }
    if let Some(p) = state.processor() {
        p.on_trace_start(&t);
    }
    TraceGuard {
        trace: t,
        active: true,
        state,
    }
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
        started_at: SystemTime::now(),
        ended_at: None,
        error: None,
    };
    let state = current_state();
    if disabled_in(&state) {
        return SpanGuard {
            span,
            active: false,
            state,
            restore: None,
            usage: Usage::default(),
        };
    }
    if let Some(p) = state.processor() {
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
    SpanGuard {
        span,
        active: true,
        state,
        restore,
        usage: Usage::default(),
    }
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
