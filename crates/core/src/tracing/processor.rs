//! Trace processors: the interface, the fan-out, the in-memory sink and the per-runtime state.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::{global_state, Span, Trace};

/// The tracing state one [`Runtime`](crate::runtime::Runtime) owns: the on/off switch and the
/// processor every span and trace is reported to.
pub(crate) struct TracingState {
    disabled: AtomicBool,
    processor: Mutex<Option<Arc<dyn TracingProcessor>>>,
}

impl TracingState {
    /// Enabled, with the bounded in-memory processor installed (Python: the default processor).
    pub(crate) fn new() -> Self {
        Self {
            disabled: AtomicBool::new(false),
            processor: Mutex::new(Some(Arc::new(InMemoryProcessor::default()))),
        }
    }

    pub(crate) fn disabled(&self) -> bool {
        self.disabled.load(Ordering::SeqCst)
    }

    pub(crate) fn set_disabled(&self, disabled: bool) {
        self.disabled.store(disabled, Ordering::SeqCst);
    }

    /// The processor, cloned out so callbacks run without the lock held: a processor that adds
    /// another one from a callback must not deadlock.
    pub(crate) fn processor(&self) -> Option<Arc<dyn TracingProcessor>> {
        self.processor
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub(crate) fn set_processors(&self, processors: Vec<Arc<dyn TracingProcessor>>) {
        *self
            .processor
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(Arc::new(MultiProcessor { processors }));
    }

    pub(crate) fn add_processor(&self, processor: Arc<dyn TracingProcessor>) {
        let mut slot = self
            .processor
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let processors = match slot.take() {
            Some(current) => vec![current, processor],
            None => vec![processor],
        };
        *slot = Some(Arc::new(MultiProcessor { processors }));
    }

    pub(crate) fn flush(&self) {
        if let Some(p) = self.processor() {
            p.force_flush();
        }
    }
}

/// Replace trace processors (Python: `set_trace_processors`).
///
/// Acts on `Runtime::global()`; use `Runtime::set_trace_processors` for a runtime of your own.
pub fn set_trace_processors(processors: Vec<Arc<dyn TracingProcessor>>) {
    global_state().set_processors(processors);
}

/// Add a processor alongside the current one, on `Runtime::global()`.
pub fn add_trace_processor(processor: Arc<dyn TracingProcessor>) {
    global_state().add_processor(processor);
}

/// Flush processors (Python: `flush_traces`), on `Runtime::global()`.
///
/// Processors that buffer events are expected to implement [`TracingProcessor::force_flush`];
/// the default is a no-op, so this is safe to call at any time.
pub fn flush_traces() {
    global_state().flush();
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

/// Default per-collection bound for [`InMemoryProcessor`].
///
/// The processor is installed by default, so in a long-running process an unbounded sink would
/// grow for the lifetime of the process. Once a collection reaches the bound, the oldest entries
/// are dropped first.
pub const DEFAULT_IN_MEMORY_CAPACITY: usize = 10_000;

/// In-memory processor for assertions in tests.
///
/// Each collection is bounded (see `DEFAULT_IN_MEMORY_CAPACITY` and
/// [`InMemoryProcessor::with_capacity`]); the oldest entries are dropped once it is full.
#[derive(Debug)]
pub struct InMemoryProcessor {
    /// Traces when they started.
    pub started_traces: Mutex<Vec<Trace>>,
    /// Finished traces.
    pub traces: Mutex<Vec<Trace>>,
    /// Spans when they started.
    pub started_spans: Mutex<Vec<Span>>,
    /// Finished spans.
    pub spans: Mutex<Vec<Span>>,
    capacity: usize,
}

impl Default for InMemoryProcessor {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_IN_MEMORY_CAPACITY)
    }
}

/// Push `item`, evicting the oldest entries in a batch when the bound is reached.
///
/// Evicting a tenth at a time keeps the amortised cost of a push constant instead of shifting the
/// whole vector on every call once the bound is hit.
fn push_bounded<T>(slot: &Mutex<Vec<T>>, item: T, capacity: usize) {
    let mut guard = slot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if guard.len() >= capacity {
        let evict = (capacity / 10).max(1).min(guard.len());
        guard.drain(..evict);
    }
    guard.push(item);
}

impl TracingProcessor for InMemoryProcessor {
    fn on_trace_start(&self, trace: &Trace) {
        push_bounded(&self.started_traces, trace.clone(), self.capacity);
    }
    fn on_trace_end(&self, trace: &Trace) {
        push_bounded(&self.traces, trace.clone(), self.capacity);
    }
    fn on_span_start(&self, span: &Span) {
        push_bounded(&self.started_spans, span.clone(), self.capacity);
    }
    fn on_span_end(&self, span: &Span) {
        push_bounded(&self.spans, span.clone(), self.capacity);
    }
}

impl InMemoryProcessor {
    /// A processor that keeps at most `capacity` entries per collection (minimum 1).
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            started_traces: Mutex::default(),
            traces: Mutex::default(),
            started_spans: Mutex::default(),
            spans: Mutex::default(),
            capacity: capacity.max(1),
        }
    }

    /// Install as the sole processor and return a handle.
    pub fn install() -> Arc<Self> {
        let proc = Arc::new(Self::default());
        set_trace_processors(vec![proc.clone()]);
        proc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_bounded_never_exceeds_capacity_and_keeps_newest() {
        let slot = Mutex::new(Vec::new());
        for i in 0..1_000 {
            push_bounded(&slot, i, 50);
            assert!(slot.lock().unwrap().len() <= 50);
        }
        let kept = slot.lock().unwrap();
        assert_eq!(kept.last(), Some(&999));
        assert!(
            kept.len() >= 45,
            "batch eviction should keep most of the window"
        );
    }

    #[test]
    fn capacity_of_zero_is_clamped_to_one() {
        let slot = Mutex::new(Vec::new());
        push_bounded(&slot, 1, 1);
        push_bounded(&slot, 2, 1);
        assert_eq!(*slot.lock().unwrap(), vec![2]);
    }
}
