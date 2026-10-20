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

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::Arc;
use std::time::Duration;

use opai_agents_core::tracing::{Span, Trace, TracingProcessor};

use super::export::{span_to_json, trace_to_json, ExportEvent};

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

/// How long [`TracingProcessor::force_flush`] waits for the background task before giving up.
pub const FLUSH_TIMEOUT: Duration = Duration::from_secs(2);

/// Batch size that triggers a write even before the interval elapses.
const BATCH_LIMIT: usize = 128;

enum Message {
    Events(ExportEvent),
    Flush(SyncSender<()>),
}

/// A processor that hands events to a background thread.
pub struct BatchedProcessor {
    tx: SyncSender<Message>,
    /// Events lost because the channel was full, or because a write failed.
    dropped: Arc<AtomicU64>,
    capacity: usize,
}

impl std::fmt::Debug for BatchedProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedProcessor")
            .field("capacity", &self.capacity)
            .field("dropped", &self.dropped())
            .finish()
    }
}

impl BatchedProcessor {
    /// Start exporting to `sink` in the background.
    ///
    /// `capacity` is how many events may be waiting to be written; `interval` is how long a lone
    /// event may sit in the queue. No runtime is needed: the writer runs on its own thread.
    pub fn spawn(sink: impl EventSink, capacity: usize, interval: Duration) -> Arc<Self> {
        // `try_send` from the synchronous callback never blocks, and the writer thread drains the
        // same channel.
        let (tx, rx) = sync_channel(capacity.max(1));
        let dropped = Arc::new(AtomicU64::new(0));
        let writer_dropped = Arc::clone(&dropped);
        std::thread::spawn(move || run(sink, rx, interval, writer_dropped));
        Arc::new(Self {
            tx,
            dropped,
            capacity: capacity.max(1),
        })
    }

    /// How many events were lost: dropped from a full queue, or lost to a failed write.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// How many events may be waiting to be written.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Never block: a full queue means the exporter is behind, and the run comes first.
    fn send(&self, event: ExportEvent) {
        match self.tx.try_send(Message::Events(event)) {
            Ok(()) => {}
            Err(std::sync::mpsc::TrySendError::Full(_)) => {
                let dropped = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
                if dropped == 1 || dropped.is_multiple_of(100) {
                    tracing::warn!(
                        dropped,
                        capacity = self.capacity,
                        "trace events are being dropped: the exporter cannot keep up"
                    );
                }
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {}
        }
    }
}

impl TracingProcessor for BatchedProcessor {
    fn on_trace_start(&self, trace: &Trace) {
        self.send(ExportEvent::TraceStart(trace_to_json(trace)));
    }

    fn on_trace_end(&self, trace: &Trace) {
        self.send(ExportEvent::TraceEnd(trace_to_json(trace)));
    }

    fn on_span_start(&self, span: &Span) {
        self.send(ExportEvent::SpanStart(span_to_json(span)));
    }

    fn on_span_end(&self, span: &Span) {
        self.send(ExportEvent::SpanEnd(span_to_json(span)));
    }

    /// Ask the writer thread to write what it has, and wait up to [`FLUSH_TIMEOUT`].
    ///
    /// The wait is safe from inside an async caller: the writer is a thread, not a task, so it makes
    /// progress even while this thread is parked.
    fn force_flush(&self) {
        let (done_tx, done_rx) = std::sync::mpsc::sync_channel(1);
        if self.tx.try_send(Message::Flush(done_tx)).is_err() {
            return;
        }
        if done_rx.recv_timeout(FLUSH_TIMEOUT).is_err() {
            tracing::warn!("timed out waiting for trace events to be written");
        }
    }
}

/// Collect events on the writer thread, and write them in batches.
///
/// `recv_timeout` is both the queue wait and the timer: a quiet exporter still writes what it has
/// every `interval`.
fn run(
    mut sink: impl EventSink,
    rx: std::sync::mpsc::Receiver<Message>,
    interval: Duration,
    dropped: Arc<AtomicU64>,
) {
    let name = sink.name();
    let mut batch: Vec<ExportEvent> = Vec::new();
    loop {
        match rx.recv_timeout(interval) {
            Ok(Message::Events(event)) => {
                batch.push(event);
                if batch.len() >= BATCH_LIMIT {
                    flush(&mut sink, &mut batch, name, &dropped);
                }
            }
            Ok(Message::Flush(done)) => {
                flush(&mut sink, &mut batch, name, &dropped);
                let _ = done.send(());
            }
            // Nothing arrived in time: write what is queued, if anything.
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                flush(&mut sink, &mut batch, name, &dropped)
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                // The processor was dropped: write what is left and stop.
                flush(&mut sink, &mut batch, name, &dropped);
                break;
            }
        }
    }
}

/// Write `batch` and empty it; a failed write counts the events as lost.
fn flush(
    sink: &mut impl EventSink,
    batch: &mut Vec<ExportEvent>,
    name: &'static str,
    dropped: &AtomicU64,
) {
    if batch.is_empty() {
        return;
    }
    if let Err(e) = sink.write(batch) {
        tracing::warn!(sink = name, "could not write trace events: {e}");
        dropped.fetch_add(batch.len() as u64, Ordering::Relaxed);
    }
    batch.clear();
}
