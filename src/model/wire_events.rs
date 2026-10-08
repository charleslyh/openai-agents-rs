//! Standard Responses API wire events emitted as [`crate::StreamEvent::RawResponse`].
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
use tokio::sync::mpsc::Sender;

use crate::items::ResponseOutputItem;
use crate::usage::Usage;

/// Placeholder id for objects synthesized from a non-Responses API
/// (Python: `agents.models.fake_id.FAKE_RESPONSES_ID`).
pub(crate) const FAKE_RESPONSES_ID: &str = "__fake_id__";

/// Response id `ScriptedModel` uses when a step declares none (Python: `"scripted-response"`).
pub(crate) const SCRIPTED_RESPONSE_ID: &str = "scripted-response";

/// Model name `ScriptedModel` reports (Python: `"scripted-model"`).
pub(crate) const SCRIPTED_MODEL: &str = "scripted-model";

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
pub(crate) fn now_seconds() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// A `Response` object as carried by `response.created` / `response.completed`.
///
/// Mirrors the fields Python fills in (`openai_chatcompletions.py:760`, `testing/model.py:1183`).
pub(crate) fn response_object(
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

/// Sends wire events, numbering them in emission order.
pub(crate) struct WireEventEmitter<'a> {
    tx: &'a Sender<Value>,
    seq: SequenceNumber,
}

impl<'a> WireEventEmitter<'a> {
    /// Bind the emitter to a raw-event channel.
    pub(crate) fn new(tx: &'a Sender<Value>) -> Self {
        Self {
            tx,
            seq: SequenceNumber::new(),
        }
    }

    async fn send(&mut self, ty: &str, fields: Value) {
        let mut event = json!({"type": ty, "sequence_number": self.seq.next()});
        if let (Some(base), Some(extra)) = (event.as_object_mut(), fields.as_object()) {
            for (key, value) in extra {
                base.insert(key.clone(), value.clone());
            }
        }
        let _ = self.tx.send(event).await;
    }

    /// `response.created` — emitted once, before any output item.
    pub(crate) async fn created(&mut self, response: Value) {
        self.send("response.created", json!({"response": response}))
            .await;
    }

    /// `response.output_item.added`.
    pub(crate) async fn output_item_added(&mut self, output_index: usize, item: Value) {
        self.send(
            "response.output_item.added",
            json!({"output_index": output_index, "item": item}),
        )
        .await;
    }

    /// `response.output_item.done`.
    pub(crate) async fn output_item_done(&mut self, output_index: usize, item: Value) {
        self.send(
            "response.output_item.done",
            json!({"output_index": output_index, "item": item}),
        )
        .await;
    }

    /// `response.content_part.added`.
    pub(crate) async fn content_part_added(
        &mut self,
        item_id: &str,
        output_index: usize,
        content_index: usize,
        part: Value,
    ) {
        self.send(
            "response.content_part.added",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": part,
            }),
        )
        .await;
    }

    /// `response.content_part.done`.
    pub(crate) async fn content_part_done(
        &mut self,
        item_id: &str,
        output_index: usize,
        content_index: usize,
        part: Value,
    ) {
        self.send(
            "response.content_part.done",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": part,
            }),
        )
        .await;
    }

    /// `response.output_text.delta`.
    pub(crate) async fn text_delta(
        &mut self,
        item_id: &str,
        output_index: usize,
        content_index: usize,
        delta: &str,
    ) {
        self.send(
            "response.output_text.delta",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "delta": delta,
                "logprobs": [],
            }),
        )
        .await;
    }

    /// `response.refusal.delta`.
    #[cfg_attr(not(feature = "openai"), allow(dead_code))]
    pub(crate) async fn refusal_delta(
        &mut self,
        item_id: &str,
        output_index: usize,
        content_index: usize,
        delta: &str,
    ) {
        self.send(
            "response.refusal.delta",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "delta": delta,
            }),
        )
        .await;
    }

    /// `response.refusal.done`.
    #[cfg_attr(not(feature = "openai"), allow(dead_code))]
    pub(crate) async fn refusal_done(
        &mut self,
        item_id: &str,
        output_index: usize,
        content_index: usize,
        refusal: &str,
    ) {
        self.send(
            "response.refusal.done",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "refusal": refusal,
            }),
        )
        .await;
    }

    /// `response.output_text.done`.
    pub(crate) async fn text_done(
        &mut self,
        item_id: &str,
        output_index: usize,
        content_index: usize,
        text: &str,
    ) {
        self.send(
            "response.output_text.done",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "text": text,
            }),
        )
        .await;
    }

    /// `response.reasoning_summary_part.added`.
    pub(crate) async fn reasoning_summary_part_added(
        &mut self,
        item_id: &str,
        output_index: usize,
        summary_index: usize,
    ) {
        self.send(
            "response.reasoning_summary_part.added",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": summary_index,
                "part": {"type": "summary_text", "text": ""},
            }),
        )
        .await;
    }

    /// `response.reasoning_summary_text.delta`.
    pub(crate) async fn reasoning_summary_text_delta(
        &mut self,
        item_id: &str,
        output_index: usize,
        summary_index: usize,
        delta: &str,
    ) {
        self.send(
            "response.reasoning_summary_text.delta",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": summary_index,
                "delta": delta,
            }),
        )
        .await;
    }

    /// `response.reasoning_summary_text.done`.
    pub(crate) async fn reasoning_summary_text_done(
        &mut self,
        item_id: &str,
        output_index: usize,
        summary_index: usize,
        text: &str,
    ) {
        self.send(
            "response.reasoning_summary_text.done",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": summary_index,
                "text": text,
            }),
        )
        .await;
    }

    /// `response.reasoning_summary_part.done`.
    pub(crate) async fn reasoning_summary_part_done(
        &mut self,
        item_id: &str,
        output_index: usize,
        summary_index: usize,
        text: &str,
    ) {
        self.send(
            "response.reasoning_summary_part.done",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": summary_index,
                "part": {"type": "summary_text", "text": text},
            }),
        )
        .await;
    }

    /// `response.reasoning_text.delta`.
    pub(crate) async fn reasoning_text_delta(
        &mut self,
        item_id: &str,
        output_index: usize,
        content_index: usize,
        delta: &str,
    ) {
        self.send(
            "response.reasoning_text.delta",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "delta": delta,
            }),
        )
        .await;
    }

    /// `response.reasoning_text.done`.
    pub(crate) async fn reasoning_text_done(
        &mut self,
        item_id: &str,
        output_index: usize,
        content_index: usize,
        text: &str,
    ) {
        self.send(
            "response.reasoning_text.done",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "text": text,
            }),
        )
        .await;
    }

    /// `response.function_call_arguments.delta`.
    pub(crate) async fn function_call_arguments_delta(
        &mut self,
        item_id: &str,
        output_index: usize,
        delta: &str,
    ) {
        self.send(
            "response.function_call_arguments.delta",
            json!({"item_id": item_id, "output_index": output_index, "delta": delta}),
        )
        .await;
    }

    /// `response.function_call_arguments.done`.
    pub(crate) async fn function_call_arguments_done(
        &mut self,
        item_id: &str,
        output_index: usize,
        name: &str,
        arguments: &str,
    ) {
        self.send(
            "response.function_call_arguments.done",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "name": name,
                "arguments": arguments,
            }),
        )
        .await;
    }

    /// `response.completed` — the terminal event, carrying the full response object.
    pub(crate) async fn completed(&mut self, response: Value) {
        self.send("response.completed", json!({"response": response}))
            .await;
    }
}

/// Emit a single terminal `response.completed`, continuing an existing sequence numbering.
///
/// Used when a gateway streams output events but never sends `response.completed`.
pub(crate) async fn emit_completed(tx: &Sender<Value>, response: Value, sequence_number: u64) {
    let _ = tx
        .send(json!({
            "type": "response.completed",
            "sequence_number": sequence_number,
            "response": response,
        }))
        .await;
}

/// The id an event refers to: `id`, else `call_id`, else the synthesized placeholder.
pub(crate) fn item_id(item: &Value) -> String {
    item.get("id")
        .or_else(|| item.get("call_id"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(FAKE_RESPONSES_ID)
        .to_string()
}

/// Copy of an item as announced by `output_item.added` (Python: `_in_progress_output_item`).
pub(crate) fn in_progress_item(item: &Value) -> Value {
    let mut copy = item.clone();
    match item.get("type").and_then(|t| t.as_str()) {
        Some("message") => {
            copy["status"] = json!("in_progress");
            copy["content"] = json!([]);
        }
        Some("reasoning") => {
            copy["status"] = json!("in_progress");
            copy["summary"] = json!([]);
            if copy.get("content").map(|c| c.is_array()).unwrap_or(false) {
                copy["content"] = json!([]);
            }
        }
        Some("function_call") => {
            copy["status"] = json!("in_progress");
            copy["arguments"] = json!("");
        }
        _ => {}
    }
    copy
}

/// Replay an already-complete response as the standard event sequence
/// (Python: `agents.testing.model` step expansion).
///
/// Used by `ScriptedModel`, by the default `Model::stream_response`, and by the OpenAI adapters
/// when a gateway answers `stream: true` with a plain JSON body.
pub(crate) async fn emit_response_stream(tx: &Sender<Value>, response: Value) {
    let output: Vec<Value> = response
        .get("output")
        .and_then(|o| o.as_array())
        .cloned()
        .unwrap_or_default();

    let mut opening = response.clone();
    opening["status"] = json!("in_progress");
    opening["output"] = json!([]);

    let mut emitter = WireEventEmitter::new(tx);
    emitter.created(opening).await;

    for (output_index, item) in output.iter().enumerate() {
        let id = item_id(item);
        emitter
            .output_item_added(output_index, in_progress_item(item))
            .await;
        match item.get("type").and_then(|t| t.as_str()) {
            Some("message") => {
                for (content_index, part) in item
                    .get("content")
                    .and_then(|c| c.as_array())
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    if part.get("type").and_then(|t| t.as_str()) != Some("output_text") {
                        continue;
                    }
                    let text = part.get("text").and_then(|t| t.as_str()).unwrap_or("");
                    emitter
                        .content_part_added(
                            &id,
                            output_index,
                            content_index,
                            json!({"type": "output_text", "text": "", "annotations": [], "logprobs": []}),
                        )
                        .await;
                    emitter
                        .text_delta(&id, output_index, content_index, text)
                        .await;
                    emitter
                        .text_done(&id, output_index, content_index, text)
                        .await;
                    emitter
                        .content_part_done(&id, output_index, content_index, part.clone())
                        .await;
                }
            }
            Some("reasoning") => {
                for (summary_index, summary) in item
                    .get("summary")
                    .and_then(|s| s.as_array())
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    let text = summary.get("text").and_then(|t| t.as_str()).unwrap_or("");
                    emitter
                        .reasoning_summary_part_added(&id, output_index, summary_index)
                        .await;
                    emitter
                        .reasoning_summary_text_delta(&id, output_index, summary_index, text)
                        .await;
                    emitter
                        .reasoning_summary_text_done(&id, output_index, summary_index, text)
                        .await;
                    emitter
                        .reasoning_summary_part_done(&id, output_index, summary_index, text)
                        .await;
                }
                for (content_index, content) in item
                    .get("content")
                    .and_then(|c| c.as_array())
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    let text = content.get("text").and_then(|t| t.as_str()).unwrap_or("");
                    emitter
                        .reasoning_text_delta(&id, output_index, content_index, text)
                        .await;
                    emitter
                        .reasoning_text_done(&id, output_index, content_index, text)
                        .await;
                }
            }
            Some("function_call") => {
                let name = item.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let arguments = item.get("arguments").and_then(|a| a.as_str()).unwrap_or("");
                emitter
                    .function_call_arguments_delta(&id, output_index, arguments)
                    .await;
                emitter
                    .function_call_arguments_done(&id, output_index, name, arguments)
                    .await;
            }
            _ => {}
        }
        emitter.output_item_done(output_index, item.clone()).await;
    }

    emitter.completed(response).await;
}
