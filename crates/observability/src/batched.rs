//! Export without slowing the run down (Rust-only, X-018).
//!
//! [`TracingProcessor`] is a synchronous callback made from inside the run loop, so anything a
//! processor does there is on the critical path. A processor that writes to a file, a socket or a
//! collector would make every span cost a round trip, and a slow or blocked exporter would slow
//! the agent down or stop it.
//!
//! [`BatchedProcessor`] therefore does as little as it can in the callback — turn the span into
//! JSON and hand it to a bounded channel — and lets a background thread do the writing. When the
//! channel is full the event is dropped and counted: a run must never wait on its own telemetry.
//! [`BatchedProcessor::dropped`] and a warning say how many were lost, so a queue that is too small
//! is visible.
//!
//! The writer is a thread of its own rather than a spawned task, because writing is blocking IO and
//! because `force_flush` is synchronous: a task on a single-threaded runtime could not run while the
//! caller waited for it.

use std::sync::Arc;
use std::time::Duration;

use opai_agents_core::tracing::{Span, Trace, TracingProcessor};

use super::export::{span_to_json, trace_to_json, ExportEvent};
use super::queue::Queue;

pub use super::queue::FLUSH_TIMEOUT;

/// Writes batches of events somewhere. Implemented by [`super::ConsoleSink`] and
/// [`super::JsonlSink`].
pub trait EventSink: Send + 'static {
    /// Write `events`, oldest first. Errors are reported and counted by the caller; returning one
    /// does not stop the processor.
    fn write(&mut self, events: &[ExportEvent]) -> Result<(), String>;

    /// Name for log messages.
    fn name(&self) -> &'static str {
        "sink"
    }
}

/// A processor that hands events to a background thread.
pub struct BatchedProcessor {
    queue: Queue<ExportEvent>,
}

impl std::fmt::Debug for BatchedProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedProcessor")
            .field("capacity", &self.capacity())
            .field("dropped", &self.dropped())
            .finish()
    }
}

impl BatchedProcessor {
    /// Start exporting to `sink` in the background.
    ///
    /// `capacity` is how many events may be waiting to be written; `interval` is how long a lone
    /// event may sit in the queue. No runtime is needed: the writer runs on its own thread.
    pub fn spawn(mut sink: impl EventSink, capacity: usize, interval: Duration) -> Arc<Self> {
        let name = sink.name();
        // `try_send` from the synchronous callback never blocks, and the writer thread drains the
        // same channel.
        let queue = Queue::spawn(name, move |events| sink.write(events), capacity, interval);
        Arc::new(Self { queue })
    }

    /// How many events were lost: dropped from a full queue, or lost to a failed write.
    pub fn dropped(&self) -> u64 {
        self.queue.dropped()
    }

    /// How many events may be waiting to be written.
    pub fn capacity(&self) -> usize {
        self.queue.capacity()
    }
}

impl TracingProcessor for BatchedProcessor {
    fn on_trace_start(&self, trace: &Trace) {
        self.queue
            .send(ExportEvent::TraceStart(trace_to_json(trace)));
    }

    fn on_trace_end(&self, trace: &Trace) {
        self.queue.send(ExportEvent::TraceEnd(trace_to_json(trace)));
    }

    fn on_span_start(&self, span: &Span) {
        self.queue.send(ExportEvent::SpanStart(span_to_json(span)));
    }

    fn on_span_end(&self, span: &Span) {
        self.queue.send(ExportEvent::SpanEnd(span_to_json(span)));
    }

    /// Ask the writer thread to write what it has, and wait up to [`FLUSH_TIMEOUT`].
    ///
    /// The wait is safe from inside an async caller: the writer is a thread, not a task, so it makes
    /// progress even while this thread is parked.
    fn force_flush(&self) {
        self.queue.flush();
    }
}
