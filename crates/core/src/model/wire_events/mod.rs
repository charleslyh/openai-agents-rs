//! Standard Responses API wire events emitted as [`crate::StreamEvent::RawResponse`].
//!
//! Public because every streaming [`crate::Model`] has to speak this vocabulary: the OpenAI
//! adapters, the scripted test double and any custom provider build on [`emit::WireEventEmitter`]
//! and [`response_object`]. A model that cannot stream natively can call
//! [`crate::model::replay_response`] instead.
//!
//! Python reference:
//! - `agents.models.openai_responses` — Responses events are forwarded verbatim (`yield chunk`).
//! - `agents.models.chatcmpl_stream_handler` — Chat Completions chunks are *synthesized into*
//!   the same Responses event vocabulary, using `FAKE_RESPONSES_ID` for ids.
//! - `agents.testing.model` — scripted steps expand into the same vocabulary.
//!
//! So every backend here emits the same shapes: each event carries `type` plus a monotonic
//! `sequence_number` starting at 0, and the terminal event is `response.completed` holding the
//! full `response` object.
//!
//! Not ported (documented in D-011): `response.output_text.annotation.added` and logprob payloads
//! are emitted empty.

use serde_json::{json, Value};

use crate::items::ResponseOutputItem;
use crate::usage::Usage;

pub mod emit;
pub mod typed;

pub use typed::{parse, WireEvent};

/// Placeholder id for objects synthesized from a non-Responses API
/// (Python: `agents.models.fake_id.FAKE_RESPONSES_ID`).
pub const FAKE_RESPONSES_ID: &str = "__fake_id__";

/// Monotonic event counter (Python: `chatcmpl_stream_handler.SequenceNumber`).
#[derive(Debug, Default)]
pub(crate) struct SequenceNumber(u64);

impl SequenceNumber {
    /// Start at 0.
    pub(crate) fn new() -> Self {
        Self(0)
    }

    /// Return the next number and advance.
    pub(crate) fn next(&mut self) -> u64 {
        let number = self.0;
        self.0 += 1;
        number
    }
}

/// Unix seconds, the unit the Responses API uses for `created_at`.
pub fn now_seconds() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// A `Response` object as carried by `response.created` / `response.completed`.
///
/// Mirrors the fields Python fills in (`openai_chatcompletions.py:760`, `testing/model.py:1183`).
pub fn response_object(
    id: &str,
    model: &str,
    status: &str,
    created_at: f64,
    output: &[ResponseOutputItem],
    usage: &Usage,
    tool_choice: &str,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "status": status,
        "output": output,
        "usage": usage.to_responses_usage(),
        "tools": [],
        "tool_choice": tool_choice,
        "parallel_tool_calls": false,
        "top_p": null,
        "temperature": null,
    })
}
