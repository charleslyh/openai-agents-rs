//! Trace processors that export somewhere, without costing the run anything (Rust-only, X-018).
//!
//! Core defines [`TracingProcessor`](openai_agents_core::tracing::TracingProcessor) and ships an
//! in-memory sink; it has no file, console or network code. This crate is where those live:
//!
//! * [`BatchedProcessor`]: hands events to a background thread through a bounded queue, so a slow
//!   exporter costs the run nothing.
//! * [`ConsoleSink`] and [`JsonlSink`]: where the events go.
//! * [`metrics`]: turn counts, tool failure rate, token usage and cost, from the same stream.
//!
//! Spans never carry a request body, a tool argument or a model output, so neither does anything
//! here.

pub mod batched;
pub mod export;
pub mod metrics;
pub mod sinks;

pub use batched::{BatchedProcessor, EventSink, FLUSH_TIMEOUT};
pub use export::{span_to_json, trace_to_json, ExportEvent};
pub use metrics::{Metrics, MetricsSink, Price, PriceTable};
pub use sinks::{ConsoleSink, JsonlSink};
