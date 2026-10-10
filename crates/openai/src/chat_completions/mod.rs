//! Chat Completions model via async-openai config + HTTP.

use async_trait::async_trait;
use serde_json::{json, Value};

use openai_agents_core::error::ModelError;
use openai_agents_core::items::{ModelResponse, ResponseOutputItem};
use openai_agents_core::model::wire_events::emit::{emit_response_stream, WireEventEmitter};
use openai_agents_core::model::wire_events::{now_seconds, response_object, FAKE_RESPONSES_ID};
use openai_agents_core::model::{Model, ModelRequest};
use openai_agents_core::usage::Usage;

mod body;
mod stream_layout;

use body::{build_chat_body, chat_payload_to_model_response};
use stream_layout::ChatStreamLayout;

use super::chat_convert::{ChatConvertOptions, ReplayReasoningFn};
use super::{
    decorate_request, is_json_response, map_reqwest, status_error, OpenAiEndpoint, SseReader,
};

/// OpenAI Chat Completions model (Python: `OpenAIChatCompletionsModel`).
pub struct OpenAIChatCompletionsModel {
    endpoint: OpenAiEndpoint,
    model: String,
    replay_reasoning: Option<ReplayReasoningFn>,
    preserve_thinking_blocks: bool,
}

impl OpenAIChatCompletionsModel {
    /// Create with API key and optional custom base URL (for wiremock / proxies).
    pub fn new(
        model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: Option<&str>,
    ) -> Self {
        Self {
            endpoint: OpenAiEndpoint::new(api_key, base_url),
            model: model.into(),
            replay_reasoning: None,
            preserve_thinking_blocks: false,
        }
    }

    /// Create from a shared endpoint.
    pub fn with_endpoint(model: impl Into<String>, endpoint: OpenAiEndpoint) -> Self {
        Self {
            endpoint,
            model: model.into(),
            replay_reasoning: None,
            preserve_thinking_blocks: false,
        }
    }

    /// Decide per reasoning item whether it is sent back with later requests
    /// (Python: `should_replay_reasoning_content`). By default only DeepSeek models get their
    /// `reasoning_content` replayed.
    pub fn should_replay_reasoning_content<F>(mut self, replay: F) -> Self
    where
        F: Fn(&str, &Value) -> bool + Send + Sync + 'static,
    {
        self.replay_reasoning = Some(std::sync::Arc::new(replay));
        self
    }

    /// Send Claude thinking blocks back with later requests (Python: `preserve_thinking_blocks`).
    /// Claude with extended thinking behind a Chat Completions gateway that returns
    /// `thinking_blocks` needs them in the assistant message that made the tool calls.
    pub fn preserve_thinking_blocks(mut self, preserve: bool) -> Self {
        self.preserve_thinking_blocks = preserve;
        self
    }

    fn convert_options(&self) -> ChatConvertOptions {
        ChatConvertOptions {
            model: self.model.clone(),
            replay_reasoning: self.replay_reasoning.clone(),
            preserve_thinking_blocks: self.preserve_thinking_blocks,
        }
    }

    /// Whether the endpoint is OpenAI itself. Third-party servers often reject parameters that
    /// OpenAI accepts, so some defaults only apply here (Python: `ChatCmplHelpers.is_openai`).
    fn is_openai_endpoint(&self) -> bool {
        use async_openai::config::Config;
        let base = self.endpoint.config.api_base().to_lowercase();
        base.contains("://api.openai.com")
    }
}

#[async_trait]
impl Model for OpenAIChatCompletionsModel {
    fn get_retry_advice(
        &self,
        request: &openai_agents_core::retry::ModelRetryAdviceRequest,
    ) -> Option<openai_agents_core::retry::ModelRetryAdvice> {
        crate::retry_advice::openai_retry_advice(request)
    }

    async fn get_response(&self, request: ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        if request.conversation_id.is_some() {
            // Python: stored conversations are a Responses API feature; the Chat Completions
            // adapter has no way to honour them and must not silently drop the request.
            return Err(ModelError::Unsupported(
                "conversation_id is not supported by the Chat Completions API; use the Responses \
                 API or clear `RunOptions.conversation_id`"
                    .into(),
            ));
        }
        let body = build_chat_body(&self.model, &self.convert_options(), &request)?;
        // non-stream
        let resp = decorate_request(
            self.endpoint
                .http
                .post(self.endpoint.url("/chat/completions")),
            request.model_settings,
        )
        .headers(self.endpoint.auth_headers())
        .json(&body)
        .send()
        .await
        .map_err(map_reqwest)?;

        if !resp.status().is_success() {
            return Err(status_error("chat completions", resp).await);
        }
        let payload: Value = resp.json().await.map_err(map_reqwest)?;

        chat_payload_to_model_response(
            payload,
            request.model_settings.preserve_raw_usage == Some(true),
            &self.model,
        )
    }

    async fn stream_response(
        &self,
        request: ModelRequest<'_>,
        raw_tx: tokio::sync::mpsc::Sender<Value>,
    ) -> Result<ModelResponse, ModelError> {
        if request.conversation_id.is_some() {
            return Err(ModelError::Unsupported(
                "conversation_id is not supported by the Chat Completions API; use the Responses \
                 API or clear `RunOptions.conversation_id`"
                    .into(),
            ));
        }
        let mut body = build_chat_body(&self.model, &self.convert_options(), &request)?;
        body["stream"] = json!(true);
        // Python (`get_stream_options_param`): ask for a usage chunk when `include_usage` is set,
        // and by default only on OpenAI. Other servers often reject the unknown parameter, so
        // for them the default is to omit it (set `ModelSettings.include_usage` to force it).
        let include_usage = request
            .model_settings
            .include_usage
            .or(self.is_openai_endpoint().then_some(true));
        if let Some(include_usage) = include_usage {
            body["stream_options"] = json!({"include_usage": include_usage});
        }

        let resp = decorate_request(
            self.endpoint
                .http
                .post(self.endpoint.url("/chat/completions")),
            request.model_settings,
        )
        .headers(self.endpoint.auth_headers())
        .json(&body)
        .send()
        .await
        .map_err(map_reqwest)?;

        if !resp.status().is_success() {
            return Err(status_error("chat completions stream", resp).await);
        }

        // Wiremock / non-SSE providers may still return a full JSON body.
        if is_json_response(&resp) {
            let payload: Value = resp.json().await.map_err(map_reqwest)?;
            let response = chat_payload_to_model_response(
                payload,
                request.model_settings.preserve_raw_usage == Some(true),
                &self.model,
            )?;
            let wire = response_object(
                response.response_id.as_deref().unwrap_or(FAKE_RESPONSES_ID),
                &self.model,
                "completed",
                now_seconds(),
                &response.output,
                &response.usage,
                "auto",
            );
            emit_response_stream(&raw_tx, wire).await;
            return Ok(response);
        }

        use futures::StreamExt;
        let mut byte_stream = resp.bytes_stream();
        let mut reader = SseReader::new();
        let mut emitter = WireEventEmitter::new(&raw_tx);
        let mut layout = ChatStreamLayout::default();
        let mut response_id: Option<String> = None;
        let mut usage: Option<Usage> = None;
        let mut raw_usage: Option<Value> = None;
        let mut finish_reason: Option<String> = None;

        while let Some(chunk) = byte_stream.next().await {
            let chunk = chunk.map_err(map_reqwest)?;
            reader.feed(&chunk);
            while let Some(event) = reader.next_event() {
                let payload = event?;

                // Some servers report a failure as a `data: {"error": ...}` chunk instead of an
                // HTTP status; without this it would pass for an empty reply.
                if let Some(error) = payload.get("error").filter(|e| !e.is_null()) {
                    return Err(ModelError::Behavior(format!(
                        "the provider reported an error in the stream: {error}"
                    )));
                }
                if let Some(reason) = payload
                    .pointer("/choices/0/finish_reason")
                    .and_then(Value::as_str)
                {
                    finish_reason = Some(reason.to_string());
                }

                if response_id.is_none() {
                    response_id = payload
                        .get("id")
                        .and_then(|i| i.as_str())
                        .map(str::to_string);
                }
                layout
                    .ensure_created(&mut emitter, &self.model, response_id.as_deref())
                    .await;

                if let Some(u) = payload.get("usage").filter(|u| !u.is_null()) {
                    usage = Some(Usage::from_chat_usage(u));
                    if request.model_settings.preserve_raw_usage == Some(true) && u.is_object() {
                        raw_usage = Some(u.clone());
                    }
                }

                let delta = payload
                    .pointer("/choices/0/delta")
                    .cloned()
                    .unwrap_or(Value::Null);

                // DeepSeek / some gateways stream chain-of-thought as `reasoning_content`
                // before the visible `content` (Python: reasoning summary path).
                if let Some(reasoning) = delta.get("reasoning_content").and_then(|c| c.as_str()) {
                    if !reasoning.is_empty() {
                        layout.open_reasoning_summary(&mut emitter).await;
                        emitter
                            .reasoning_summary_text_delta(FAKE_RESPONSES_ID, 0, 0, reasoning)
                            .await;
                        layout.reasoning_summary.push_str(reasoning);
                    }
                }

                // Third-party gateways expose raw CoT as `reasoning` (Python: content path).
                if let Some(reasoning) = delta.get("reasoning").and_then(|c| c.as_str()) {
                    if !reasoning.is_empty() {
                        layout.open_reasoning_content(&mut emitter).await;
                        emitter
                            .reasoning_text_delta(FAKE_RESPONSES_ID, 0, 0, reasoning)
                            .await;
                        layout.reasoning_content.push_str(reasoning);
                    }
                }

                if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                    if !content.is_empty() {
                        layout.open_message(&mut emitter).await;
                        let index = layout.message_index();
                        emitter
                            .text_delta(
                                FAKE_RESPONSES_ID,
                                index,
                                layout.text_content_index,
                                content,
                            )
                            .await;
                        layout.text.push_str(content);
                    }
                }

                // The model declines to answer (Python: `response.refusal.delta`).
                if let Some(refusal) = delta.get("refusal").and_then(|r| r.as_str()) {
                    if !refusal.is_empty() {
                        layout.open_refusal(&mut emitter).await;
                        emitter
                            .refusal_delta(
                                FAKE_RESPONSES_ID,
                                layout.message_index(),
                                layout.refusal_content_index,
                                refusal,
                            )
                            .await;
                        layout.refusal.push_str(refusal);
                    }
                }

                if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                    for tc in tcs {
                        let idx = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                        if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                            if !id.is_empty() {
                                layout.call_id(idx, id);
                            }
                        }
                        if let Some(name) = tc.pointer("/function/name").and_then(|n| n.as_str()) {
                            if !name.is_empty() {
                                layout.call_name(idx, name);
                            }
                        }
                        if let Some(args) =
                            tc.pointer("/function/arguments").and_then(|a| a.as_str())
                        {
                            if !args.is_empty() {
                                layout.call_arguments(idx, args);
                                let index = layout.register_call(&mut emitter, idx).await;
                                emitter
                                    .function_call_arguments_delta(FAKE_RESPONSES_ID, index, args)
                                    .await;
                            }
                        }
                        // Gemini's OpenAI-compatible endpoint sends the signature as
                        // `extra_content.google.thought_signature`; LiteLLM as
                        // `provider_specific_fields.thought_signature` (Gemini models only).
                        let signature = tc
                            .pointer("/extra_content/google/thought_signature")
                            .or_else(|| {
                                self.model
                                    .to_lowercase()
                                    .contains("gemini")
                                    .then(|| {
                                        tc.pointer("/provider_specific_fields/thought_signature")
                                    })
                                    .flatten()
                            })
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty());
                        if let Some(signature) = signature {
                            layout.call(idx).thought_signature = Some(signature.to_string());
                        }
                        // A tool call can arrive without an arguments delta.
                        let _ = layout.register_call(&mut emitter, idx).await;
                    }
                }
            }
        }

        // Python: a stream that ends with nothing to show is explained by its finish reason.
        // Filtered output becomes a refusal; a completion cut off before any visible token is a
        // budget problem and an error.
        if !layout.message_open && layout.calls.is_empty() {
            match finish_reason.as_deref() {
                Some("content_filter") => {
                    let refusal = "Response withheld by the provider's content filter.";
                    layout.open_refusal(&mut emitter).await;
                    emitter
                        .refusal_delta(
                            FAKE_RESPONSES_ID,
                            layout.message_index(),
                            layout.refusal_content_index,
                            refusal,
                        )
                        .await;
                    layout.refusal.push_str(refusal);
                }
                Some("length") => {
                    return Err(ModelError::Behavior(
                        "Chat Completions stream terminated with finish_reason='length' but \
                         produced no assistant text, tool call, or refusal."
                            .into(),
                    ));
                }
                _ => {}
            }
        }

        // Finalize: `done` events, then the terminal `response.completed`.
        let mut tagged: Vec<(usize, Value)> = Vec::new();
        // Python: every streamed item records the model and response that produced it.
        let mut origin = serde_json::Map::new();
        origin.insert("model".into(), self.model.clone().into());
        if let Some(id) = response_id.as_deref().filter(|id| !id.is_empty()) {
            origin.insert("response_id".into(), id.into());
        }

        if layout.reasoning_open {
            let mut summary: Vec<Value> = Vec::new();
            if layout.reasoning_summary_open {
                emitter
                    .reasoning_summary_text_done(FAKE_RESPONSES_ID, 0, 0, &layout.reasoning_summary)
                    .await;
                emitter
                    .reasoning_summary_part_done(FAKE_RESPONSES_ID, 0, 0, &layout.reasoning_summary)
                    .await;
                summary.push(json!({"type": "summary_text", "text": layout.reasoning_summary}));
            }
            let mut content: Vec<Value> = Vec::new();
            if layout.reasoning_content_open {
                emitter
                    .reasoning_text_done(FAKE_RESPONSES_ID, 0, 0, &layout.reasoning_content)
                    .await;
                content.push(json!({"type": "reasoning_text", "text": layout.reasoning_content}));
            }
            let mut item = json!({
                "id": FAKE_RESPONSES_ID,
                "type": "reasoning",
                "summary": summary,
                "content": content,
            });
            item["provider_data"] = Value::Object(origin.clone());
            emitter.output_item_done(0, item.clone()).await;
            tagged.push((0, item));
        }

        if layout.message_open {
            let index = layout.message_index();
            let mut parts: Vec<(usize, Value)> = Vec::new();
            if layout.text_open {
                let part = json!({
                    "type": "output_text",
                    "text": layout.text,
                    "annotations": [],
                    "logprobs": [],
                });
                emitter
                    .content_part_done(
                        FAKE_RESPONSES_ID,
                        index,
                        layout.text_content_index,
                        part.clone(),
                    )
                    .await;
                parts.push((layout.text_content_index, part));
            }
            if layout.refusal_open {
                emitter
                    .refusal_done(
                        FAKE_RESPONSES_ID,
                        index,
                        layout.refusal_content_index,
                        &layout.refusal,
                    )
                    .await;
                let part = json!({"type": "refusal", "refusal": layout.refusal});
                emitter
                    .content_part_done(
                        FAKE_RESPONSES_ID,
                        index,
                        layout.refusal_content_index,
                        part.clone(),
                    )
                    .await;
                parts.push((layout.refusal_content_index, part));
            }
            parts.sort_by_key(|(content_index, _)| *content_index);
            let mut item = json!({
                "id": FAKE_RESPONSES_ID,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": parts.into_iter().map(|(_, part)| part).collect::<Vec<_>>(),
            });
            item["provider_data"] = Value::Object(origin.clone());
            emitter.output_item_done(index, item.clone()).await;
            tagged.push((index, item));
        }

        for (_, call) in layout.calls.iter() {
            // A server that streams a tool call without an id still needs one: the runner pairs
            // the tool result with the call by id.
            let call_id = if call.id.is_empty() {
                format!("call_{}", uuid::Uuid::new_v4().simple())
            } else {
                call.id.clone()
            };
            let mut item = json!({
                "id": FAKE_RESPONSES_ID,
                "type": "function_call",
                "name": call.name,
                "arguments": call.arguments,
                "call_id": call_id,
            });
            let mut data = origin.clone();
            if let Some(signature) = &call.thought_signature {
                data.insert("thought_signature".into(), signature.clone().into());
            }
            item["provider_data"] = Value::Object(data);
            emitter
                .output_item_done(call.output_index, item.clone())
                .await;
            tagged.push((call.output_index, item));
        }

        tagged.sort_by_key(|(index, _)| *index);
        let output: Vec<ResponseOutputItem> = tagged.into_iter().map(|(_, item)| item).collect();
        let id = response_id
            .clone()
            .unwrap_or_else(|| FAKE_RESPONSES_ID.to_string());
        let response = ModelResponse {
            output: output.clone(),
            usage: usage.unwrap_or(Usage {
                requests: 1,
                ..Usage::default()
            }),
            response_id,
            request_id: None,
            raw_usage,
        };
        let wire = response_object(
            &id,
            &self.model,
            "completed",
            now_seconds(),
            &output,
            &response.usage,
            "auto",
        );
        emitter.completed(wire).await;
        Ok(response)
    }
}
