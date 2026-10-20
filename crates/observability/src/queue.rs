//! A bounded queue with a writer thread of its own (Rust-only, X-018).
//!
//! Trace export and conversation export both run from inside the run loop, so both need the same
//! thing: hand the item over without waiting, write it elsewhere, and count what was lost. This is
//! that, once.
//!
//! The writer is a thread rather than a spawned task, because writing is blocking IO and because
//! [`Queue::flush`] is synchronous: a task on a single-threaded runtime could not run while the
//! caller waited for it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::Arc;
use std::time::Duration;

/// How long a flush waits for the writer thread before giving up.
pub const FLUSH_TIMEOUT: Duration = Duration::from_secs(2);

/// Batch size that triggers a write even before the interval elapses.
const BATCH_LIMIT: usize = 128;

enum Message<T> {
    Item(T),
    Flush(SyncSender<()>),
}

/// Items waiting for a writer thread.
pub(crate) struct Queue<T> {
    tx: SyncSender<Message<T>>,
    /// Items lost because the channel was full, or because a write failed.
    dropped: Arc<AtomicU64>,
    capacity: usize,
    name: &'static str,
}

impl<T: Send + 'static> Queue<T> {
    /// Start `write` on a thread of its own. `capacity` is how many items may be waiting;
    /// `interval` is how long a lone item may sit in the queue.
    pub(crate) fn spawn(
        name: &'static str,
        write: impl FnMut(&[T]) -> Result<(), String> + Send + 'static,
        capacity: usize,
        interval: Duration,
    ) -> Self {
        let capacity = capacity.max(1);
        let (tx, rx) = sync_channel(capacity);
        let dropped = Arc::new(AtomicU64::new(0));
        let writer_dropped = Arc::clone(&dropped);
        std::thread::spawn(move || run(write, rx, interval, name, writer_dropped));
        Self {
            tx,
            dropped,
            capacity,
            name,
        }
    }

    /// Never block: a full queue means the writer is behind, and the run comes first.
    pub(crate) fn send(&self, item: T) {
        match self.tx.try_send(Message::Item(item)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                let dropped = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
                if dropped == 1 || dropped.is_multiple_of(100) {
                    tracing::warn!(
                        sink = self.name,
                        dropped,
                        capacity = self.capacity,
                        "events are being dropped: the exporter cannot keep up"
                    );
                }
            }
            Err(TrySendError::Disconnected(_)) => {}
        }
    }

    /// Ask the writer to write what it has, and wait up to [`FLUSH_TIMEOUT`].
    pub(crate) fn flush(&self) {
        let (done_tx, done_rx) = sync_channel(1);
        if self.tx.try_send(Message::Flush(done_tx)).is_err() {
            return;
        }
        if done_rx.recv_timeout(FLUSH_TIMEOUT).is_err() {
            tracing::warn!(
                sink = self.name,
                "timed out waiting for events to be written"
            );
        }
    }

    pub(crate) fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Collect items on the writer thread, and write them in batches.
///
/// `recv_timeout` is both the queue wait and the timer: a quiet exporter still writes what it has
/// every `interval`.
fn run<T>(
    mut write: impl FnMut(&[T]) -> Result<(), String>,
    rx: Receiver<Message<T>>,
    interval: Duration,
    name: &'static str,
    dropped: Arc<AtomicU64>,
) {
    let mut batch: Vec<T> = Vec::new();
    loop {
        match rx.recv_timeout(interval) {
            Ok(Message::Item(item)) => {
                batch.push(item);
                if batch.len() >= BATCH_LIMIT {
                    flush(&mut write, &mut batch, name, &dropped);
                }
            }
            Ok(Message::Flush(done)) => {
                flush(&mut write, &mut batch, name, &dropped);
                let _ = done.send(());
            }
            Err(RecvTimeoutError::Timeout) => flush(&mut write, &mut batch, name, &dropped),
            Err(RecvTimeoutError::Disconnected) => {
                // The producer was dropped: write what is left and stop.
                flush(&mut write, &mut batch, name, &dropped);
                break;
            }
        }
    }
}

/// Write `batch` and empty it; a failed write counts the items as lost.
fn flush<T>(
    write: &mut impl FnMut(&[T]) -> Result<(), String>,
    batch: &mut Vec<T>,
    name: &'static str,
    dropped: &AtomicU64,
) {
    if batch.is_empty() {
        return;
    }
    if let Err(e) = write(batch) {
        tracing::warn!(sink = name, "could not write events: {e}");
        dropped.fetch_add(batch.len() as u64, Ordering::Relaxed);
    }
    batch.clear();
}
