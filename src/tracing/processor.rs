//! Trace processors: the interface, the fan-out, the in-memory sink and the global slot.

use std::sync::{Arc, Mutex, OnceLock};

use super::{Span, Trace};

static PROCESSOR: OnceLock<Mutex<Option<Arc<dyn TracingProcessor>>>> = OnceLock::new();

pub(super) fn processor_slot() -> &'static Mutex<Option<Arc<dyn TracingProcessor>>> {
    PROCESSOR.get_or_init(|| Mutex::new(Some(Arc::new(InMemoryProcessor::default()))))
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
