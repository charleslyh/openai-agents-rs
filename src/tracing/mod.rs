//! Agents tracing (Python: `agents.tracing` Phase-1 local subset).
//!
//! Cloud OpenAI export is intentionally unsupported (D-004).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use uuid::Uuid;

static TRACING_DISABLED: AtomicBool = AtomicBool::new(false);
static PROCESSOR: OnceLock<Mutex<Option<Arc<dyn TracingProcessor>>>> = OnceLock::new();

fn processor_slot() -> &'static Mutex<Option<Arc<dyn TracingProcessor>>> {
    PROCESSOR.get_or_init(|| Mutex::new(Some(Arc::new(InMemoryProcessor::default()))))
}

/// Disable or enable tracing globally (Python: `set_tracing_disabled`).
pub fn set_tracing_disabled(disabled: bool) {
    TRACING_DISABLED.store(disabled, Ordering::SeqCst);
}

/// Whether tracing is disabled.
pub fn is_tracing_disabled() -> bool {
    TRACING_DISABLED.load(Ordering::SeqCst)
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

/// Flush processors (no-op for in-memory).
pub fn flush_traces() {}

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
}

/// In-memory processor for assertions in tests.
#[derive(Debug, Default)]
pub struct InMemoryProcessor {
    /// Finished traces.
    pub traces: Mutex<Vec<Trace>>,
    /// Finished spans.
    pub spans: Mutex<Vec<Span>>,
}

impl TracingProcessor for InMemoryProcessor {
    fn on_trace_start(&self, _trace: &Trace) {}
    fn on_trace_end(&self, trace: &Trace) {
        self.traces.lock().expect("lock").push(trace.clone());
    }
    fn on_span_start(&self, _span: &Span) {}
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

/// A workflow-level trace (Python: `Trace`).
#[derive(Debug, Clone)]
pub struct Trace {
    /// Trace id.
    pub trace_id: String,
    /// Workflow name.
    pub workflow_name: String,
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

/// Span data kinds (Phase-1).
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
}

/// A span within a trace (Python: `Span`).
#[derive(Debug, Clone)]
pub struct Span {
    /// Span id.
    pub span_id: String,
    /// Span payload.
    pub data: SpanData,
}

/// Guard that ends the span on drop.
pub struct SpanGuard {
    span: Span,
    active: bool,
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Some(p) = processor_slot().lock().expect("lock").as_ref() {
            p.on_span_end(&self.span);
        }
    }
}

/// Start a root trace (Python: `trace(...)`).
pub fn trace(workflow_name: &str) -> TraceGuard {
    if is_tracing_disabled() {
        return TraceGuard {
            trace: Trace {
                trace_id: gen_trace_id(),
                workflow_name: workflow_name.to_string(),
            },
            active: false,
        };
    }
    let t = Trace {
        trace_id: gen_trace_id(),
        workflow_name: workflow_name.to_string(),
    };
    if let Some(p) = processor_slot().lock().expect("lock").as_ref() {
        p.on_trace_start(&t);
    }
    TraceGuard {
        trace: t,
        active: true,
    }
}

fn start_span(data: SpanData) -> SpanGuard {
    if is_tracing_disabled() {
        return SpanGuard {
            span: Span {
                span_id: gen_span_id(),
                data,
            },
            active: false,
        };
    }
    let span = Span {
        span_id: gen_span_id(),
        data,
    };
    if let Some(p) = processor_slot().lock().expect("lock").as_ref() {
        p.on_span_start(&span);
    }
    SpanGuard {
        span,
        active: true,
    }
}

/// Agent span (Python: `agent_span`).
pub fn agent_span(name: &str) -> SpanGuard {
    start_span(SpanData::Agent {
        name: name.to_string(),
    })
}

/// Function/tool span (Python: `function_span`).
pub fn function_span(name: &str) -> SpanGuard {
    start_span(SpanData::Function {
        name: name.to_string(),
    })
}

/// Generation span (Python: `generation_span`).
pub fn generation_span(model: &str) -> SpanGuard {
    start_span(SpanData::Generation {
        model: model.to_string(),
    })
}

/// Custom span (Python: `custom_span`).
pub fn custom_span(name: &str) -> SpanGuard {
    start_span(SpanData::Custom {
        name: name.to_string(),
    })
}
