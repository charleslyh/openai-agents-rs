//! Output-index bookkeeping for the Chat Completions stream, which is synthesized into the
//! Responses event vocabulary.

use serde_json::json;

use openai_agents_core::model::wire_events::emit::WireEventEmitter;
use openai_agents_core::model::wire_events::{now_seconds, response_object, FAKE_RESPONSES_ID};
use openai_agents_core::usage::Usage;

/// One synthesized tool call.
#[derive(Debug, Default, Clone)]
pub(super) struct ChatCall {
    pub(super) id: String,
    pub(super) name: String,
    pub(super) arguments: String,
    pub(super) output_index: usize,
    /// `output_item.added` already emitted.
    pub(super) announced: bool,
    /// Gemini's thought signature for this call, which it wants back with the call.
    pub(super) thought_signature: Option<String>,
}

/// Output-index bookkeeping and lazy event emission for the synthesized stream
/// (Python: `chatcmpl_stream_handler._StreamOutputLayout` + `StreamingState`).
#[derive(Debug, Default)]
pub(super) struct ChatStreamLayout {
    pub(super) created: bool,
    pub(super) reasoning_open: bool,
    pub(super) reasoning_summary_open: bool,
    pub(super) reasoning_content_open: bool,
    pub(super) reasoning_summary: String,
    pub(super) reasoning_content: String,
    pub(super) message_open: bool,
    pub(super) message_index: Option<usize>,
    pub(super) text_open: bool,
    pub(super) text_content_index: usize,
    pub(super) text: String,
    pub(super) refusal_open: bool,
    pub(super) refusal_content_index: usize,
    pub(super) refusal: String,
    pub(super) calls: std::collections::BTreeMap<usize, ChatCall>,
}

impl ChatStreamLayout {
    /// `response.created` precedes every output item (Python: emitted once, up front).
    pub(super) async fn ensure_created(
        &mut self,
        emitter: &mut WireEventEmitter<'_>,
        model: &str,
        response_id: Option<&str>,
    ) {
        if self.created {
            return;
        }
        self.created = true;
        let response = response_object(
            response_id.unwrap_or(FAKE_RESPONSES_ID),
            model,
            "in_progress",
            now_seconds(),
            &[],
            &Usage::default(),
            "auto",
        );
        emitter.created(response).await;
    }

    /// The reasoning item always occupies output index 0 (Python: `_reasoning_output_count`).
    pub(super) async fn open_reasoning(&mut self, emitter: &mut WireEventEmitter<'_>) {
        if self.reasoning_open {
            return;
        }
        self.reasoning_open = true;
        emitter
            .output_item_added(
                0,
                json!({"id": FAKE_RESPONSES_ID, "type": "reasoning", "summary": [], "content": []}),
            )
            .await;
    }

    pub(super) async fn open_reasoning_summary(&mut self, emitter: &mut WireEventEmitter<'_>) {
        self.open_reasoning(emitter).await;
        if self.reasoning_summary_open {
            return;
        }
        self.reasoning_summary_open = true;
        emitter
            .reasoning_summary_part_added(FAKE_RESPONSES_ID, 0, 0)
            .await;
    }

    pub(super) async fn open_reasoning_content(&mut self, emitter: &mut WireEventEmitter<'_>) {
        self.open_reasoning(emitter).await;
        self.reasoning_content_open = true;
    }

    /// Open the assistant message item (once). Text and refusal parts are added to it.
    pub(super) async fn open_message_item(&mut self, emitter: &mut WireEventEmitter<'_>) -> usize {
        if !self.message_open {
            // Python: reasoning takes index 0, then every tool call known at this point.
            let index = usize::from(self.reasoning_open) + self.calls.len();
            self.message_open = true;
            self.message_index = Some(index);
            emitter
                .output_item_added(
                    index,
                    json!({
                        "id": FAKE_RESPONSES_ID,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    }),
                )
                .await;
        }
        self.message_index()
    }

    /// Open the `output_text` part of the message (once).
    pub(super) async fn open_message(&mut self, emitter: &mut WireEventEmitter<'_>) {
        let index = self.open_message_item(emitter).await;
        if self.text_open {
            return;
        }
        self.text_open = true;
        self.text_content_index = usize::from(self.refusal_open);
        emitter
            .content_part_added(
                FAKE_RESPONSES_ID,
                index,
                self.text_content_index,
                json!({"type": "output_text", "text": "", "annotations": [], "logprobs": []}),
            )
            .await;
    }

    /// Open the `refusal` part of the message (once).
    pub(super) async fn open_refusal(&mut self, emitter: &mut WireEventEmitter<'_>) {
        let index = self.open_message_item(emitter).await;
        if self.refusal_open {
            return;
        }
        self.refusal_open = true;
        self.refusal_content_index = usize::from(self.text_open);
        emitter
            .content_part_added(
                FAKE_RESPONSES_ID,
                index,
                self.refusal_content_index,
                json!({"type": "refusal", "refusal": ""}),
            )
            .await;
    }

    pub(super) fn message_index(&self) -> usize {
        self.message_index.unwrap_or(0)
    }

    pub(super) fn call(&mut self, index: usize) -> &mut ChatCall {
        self.calls.entry(index).or_default()
    }

    pub(super) fn call_id(&mut self, index: usize, id: &str) {
        self.call(index).id.push_str(id);
    }

    pub(super) fn call_name(&mut self, index: usize, name: &str) {
        self.call(index).name.push_str(name);
    }

    pub(super) fn call_arguments(&mut self, index: usize, arguments: &str) {
        self.call(index).arguments.push_str(arguments);
    }

    /// Announce the call once (`output_item.added`) and return its output index.
    ///
    /// Python: `function_call_output_index` = reasoning slot + position in insertion order.
    pub(super) async fn register_call(
        &mut self,
        emitter: &mut WireEventEmitter<'_>,
        index: usize,
    ) -> usize {
        if let Some(call) = self.calls.get(&index) {
            if call.announced {
                return call.output_index;
            }
        }
        let announced = self.calls.values().filter(|c| c.announced).count();
        let output_index = usize::from(self.reasoning_open) + announced;
        let (id, name) = {
            let call = self.calls.entry(index).or_default();
            call.output_index = output_index;
            call.announced = true;
            (call.id.clone(), call.name.clone())
        };
        emitter
            .output_item_added(
                output_index,
                json!({
                    "id": FAKE_RESPONSES_ID,
                    "type": "function_call",
                    "name": name,
                    "arguments": "",
                    "call_id": id,
                    "status": "in_progress",
                }),
            )
            .await;
        output_index
    }
}
