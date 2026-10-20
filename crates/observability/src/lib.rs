//! Trace processors that export somewhere, without costing the run anything (Rust-only, X-018).
//!
//! Core defines [`TracingProcessor`](opai_agents_core::tracing::TracingProcessor) and ships an
//! in-memory sink; it has no file, console or network code. This crate is where those live:
//!
//! * [`BatchedProcessor`]: hands events to a background thread through a bounded queue, so a slow
//!   exporter costs the run nothing.
//! * [`ConsoleSink`] and [`JsonlSink`]: where the events go.
//! * `OtelSink` (`otel` feature): spans as OpenTelemetry spans, in whatever pipeline the process
//!   installed.
//! * [`TracingSink`]: events into the application's own `tracing` subscriber.
//! * [`metrics`]: turn counts, tool failure rate, token usage and cost, from the same stream.
//! * [`conversation`]: the other half a console needs — prompts, replies, tool arguments and
//!   results — recorded through `RunHooks` and handed to whatever the application attaches.
//!
//! Spans never carry a request body, a tool argument or a model output, so neither does anything
//! that exports them. Content has a channel of its own, [`conversation`], that is off until the
//! application turns it on.

pub mod batched;
pub mod bridge;
pub mod conversation;
pub mod export;
pub mod metrics;
#[cfg(feature = "otel")]
pub mod otel;
mod queue;
pub mod sinks;

pub use batched::{BatchedProcessor, EventSink, FLUSH_TIMEOUT};
pub use bridge::TracingSink;
pub use conversation::{
    ConversationEvent, ConversationHub, ConversationRecorder, ConversationSink, Limits,
    QueuedConversationSink,
};
pub use export::{span_to_json, trace_to_json, ExportEvent};
pub use metrics::{Metrics, MetricsSink, Price, PriceTable};
pub use sinks::{ConsoleSink, JsonlSink};

#[cfg(feature = "otel")]
pub use otel::OtelSink;
