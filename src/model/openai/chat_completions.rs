//! Chat Completions model via async-openai config + HTTP.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::ModelError;
use crate::items::{ItemHelpers, ModelResponse, ResponseOutputItem};
use crate::model::wire_events::{
    emit_response_stream, now_seconds, response_object, WireEventEmitter, FAKE_RESPONSES_ID,
};
use crate::model::{Model, ModelRequest};
use crate::usage::Usage;

use super::{
    apply_model_settings_chat, decorate_request, is_json_response, map_transport, merge_usage,
    tools_as_chat, OpenAiEndpoint, SseReader,
};

/// OpenAI Chat Completions model (Python: `OpenAIChatCompletionsModel`).
pub struct OpenAIChatCompletionsModel {
    endpoint: OpenAiEndpoint,
    model: String,
}

impl OpenAIChatCompletionsModel {
    /// Create with API key and optional custom base URL (for wiremock / proxies).
    pub fn new(model: impl Into<String>, api_key: impl Into<String>, base_url: Option<&str>) -> Self {
        Self {
            endpoint: OpenAiEndpoint::new(api_key, base_url),
            model: model.into(),
        }
    }

    /// Create from a shared endpoint.
    pub fn with_endpoint(model: impl Into<String>, endpoint: OpenAiEndpoint) -> Self {
        Self {
            endpoint,
            model: model.into(),
        }
    }
}

#[async_trait]
impl Model for OpenAIChatCompletionsModel {
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
        let body = build_chat_body(&self.model, &request);
        // non-stream
        let resp = decorate_request(
            self.endpoint.http.post(self.endpoint.url("/chat/completions")),
            request.model_settings,
        )
        .bearer_auth(self.endpoint.api_key())
        .json(&body)
        .send()
        .await
        .map_err(map_transport)?;

        let status = resp.status();
        let payload: Value = resp.json().await.map_err(map_transport)?;
        if !status.is_success() {
            return Err(ModelError::Transport(format!(
                "chat completions status={status} body={payload}"
            )));
        }

        chat_payload_to_model_response(payload)
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
        let mut body = build_chat_body(&self.model, &request);
        body["stream"] = json!(true);
        // Some providers want stream_options.include_usage
        // Python: `ModelSettings.include_usage` asks the provider for a usage chunk.
        body["stream_options"] = json!({
            "include_usage": request.model_settings.include_usage.unwrap_or(true)
        });

        let resp = decorate_request(
            self.endpoint.http.post(self.endpoint.url("/chat/completions")),
            request.model_settings,
        )
        .bearer_auth(self.endpoint.api_key())
        .json(&body)
        .send()
        .await
        .map_err(map_transport)?;

        let status = resp.status();
        if !status.is_success() {
            let payload: Value = resp.json().await.unwrap_or(Value::Null);
            return Err(ModelError::Transport(format!(
                "chat completions stream status={status} body={payload}"
            )));
        }

        // Wiremock / non-SSE providers may still return a full JSON body.
        if is_json_response(&resp) {
            let payload: Value = resp.json().await.map_err(map_transport)?;
            let response = chat_payload_to_model_response(payload)?;
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
        let mut usage: Option<(u64, u64)> = None;

        while let Some(chunk) = byte_stream.next().await {
            let chunk = chunk.map_err(map_transport)?;
            reader.feed(&chunk);
            while let Some(event) = reader.next_event() {
                let payload = event?;

                if response_id.is_none() {
                    response_id = payload
                        .get("id")
                        .and_then(|i| i.as_str())
                        .map(str::to_string);
                }
                layout
                    .ensure_created(&mut emitter, &self.model, response_id.as_deref())
                    .await;

                if let Some(u) = payload.get("usage") {
                    let input = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                    let output = u
                        .get("completion_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    usage = Some((input, output));
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
                            .text_delta(FAKE_RESPONSES_ID, index, 0, content)
                            .await;
                        layout.text.push_str(content);
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
                        // A tool call can arrive without an arguments delta.
                        let _ = layout.register_call(&mut emitter, idx).await;
                    }
                }
            }
        }

        // Finalize: `done` events, then the terminal `response.completed`.
        let mut tagged: Vec<(usize, Value)> = Vec::new();

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
            let item = json!({
                "id": FAKE_RESPONSES_ID,
                "type": "reasoning",
                "summary": summary,
                "content": content,
            });
            emitter.output_item_done(0, item.clone()).await;
            tagged.push((0, item));
        }

        if layout.message_open {
            let index = layout.message_index();
            let part = json!({
                "type": "output_text",
                "text": layout.text,
                "annotations": [],
                "logprobs": [],
            });
            emitter
                .content_part_done(FAKE_RESPONSES_ID, index, 0, part.clone())
                .await;
            let item = json!({
                "id": FAKE_RESPONSES_ID,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [part],
            });
            emitter.output_item_done(index, item.clone()).await;
            tagged.push((index, item));
        }

        for (_, call) in layout.calls.iter() {
            let call_id = if call.id.is_empty() {
                "call".to_string()
            } else {
                call.id.clone()
            };
            let item = json!({
                "id": FAKE_RESPONSES_ID,
                "type": "function_call",
                "name": call.name,
                "arguments": call.arguments,
                "call_id": call_id,
            });
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
            usage: merge_usage(usage),
            response_id,
            request_id: None,
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

/// One synthesized tool call.
#[derive(Debug, Default, Clone)]
struct ChatCall {
    id: String,
    name: String,
    arguments: String,
    output_index: usize,
    /// `output_item.added` already emitted.
    announced: bool,
}

/// Output-index bookkeeping and lazy event emission for the synthesized stream
/// (Python: `chatcmpl_stream_handler._StreamOutputLayout` + `StreamingState`).
#[derive(Debug, Default)]
struct ChatStreamLayout {
    created: bool,
    reasoning_open: bool,
    reasoning_summary_open: bool,
    reasoning_content_open: bool,
    reasoning_summary: String,
    reasoning_content: String,
    message_open: bool,
    message_index: Option<usize>,
    text: String,
    calls: std::collections::BTreeMap<usize, ChatCall>,
}

impl ChatStreamLayout {
    /// `response.created` precedes every output item (Python: emitted once, up front).
    async fn ensure_created(
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
    async fn open_reasoning(&mut self, emitter: &mut WireEventEmitter<'_>) {
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

    async fn open_reasoning_summary(&mut self, emitter: &mut WireEventEmitter<'_>) {
        self.open_reasoning(emitter).await;
        if self.reasoning_summary_open {
            return;
        }
        self.reasoning_summary_open = true;
        emitter
            .reasoning_summary_part_added(FAKE_RESPONSES_ID, 0, 0)
            .await;
    }

    async fn open_reasoning_content(&mut self, emitter: &mut WireEventEmitter<'_>) {
        self.open_reasoning(emitter).await;
        self.reasoning_content_open = true;
    }

    async fn open_message(&mut self, emitter: &mut WireEventEmitter<'_>) {
        if self.message_open {
            return;
        }
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
        emitter
            .content_part_added(
                FAKE_RESPONSES_ID,
                index,
                0,
                json!({"type": "output_text", "text": "", "annotations": [], "logprobs": []}),
            )
            .await;
    }

    fn message_index(&self) -> usize {
        self.message_index.unwrap_or(0)
    }

    fn call(&mut self, index: usize) -> &mut ChatCall {
        self.calls.entry(index).or_default()
    }

    fn call_id(&mut self, index: usize, id: &str) {
        self.call(index).id.push_str(id);
    }

    fn call_name(&mut self, index: usize, name: &str) {
        self.call(index).name.push_str(name);
    }

    fn call_arguments(&mut self, index: usize, arguments: &str) {
        self.call(index).arguments.push_str(arguments);
    }

    /// Announce the call once (`output_item.added`) and return its output index.
    ///
    /// Python: `function_call_output_index` = reasoning slot + position in insertion order.
    async fn register_call(&mut self, emitter: &mut WireEventEmitter<'_>, index: usize) -> usize {
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

fn build_chat_body(model: &str, request: &ModelRequest<'_>) -> Value {
    let messages = input_to_chat_messages(request.system_instructions, &request.input);
    let mut body = json!({
        "model": model,
        "messages": messages,
    });
    let tools = tools_as_chat(request.tools);
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
    }
    apply_model_settings_chat(&mut body, request.model_settings);
    // Python: structured output becomes `response_format` for Chat Completions.
    if let Some(schema) = request.output_schema.filter(|s| !s.is_plain_text()) {
        match schema.json_schema() {
            Ok(schema_value) => {
                body["response_format"] = json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.name(),
                        "schema": schema_value,
                        "strict": schema.is_strict_json_schema(),
                    }
                });
            }
            Err(_) => {}
        }
    }
    body
}

fn input_to_chat_messages(
    system: Option<&str>,
    input: &crate::model::ModelInput<'_>,
) -> Vec<Value> {
    let mut messages = Vec::new();
    if let Some(sys) = system {
        messages.push(json!({"role": "system", "content": sys}));
    }
    match input {
        crate::model::ModelInput::Text(t) => {
            messages.push(json!({"role": "user", "content": t}));
        }
        crate::model::ModelInput::Items(items) => {
            for item in *items {
                messages.extend(responses_item_to_chat_messages(item));
            }
        }
    }
    messages
}

fn responses_item_to_chat_messages(item: &Value) -> Vec<Value> {
    let ty = item.get("type").and_then(|t| t.as_str());
    match ty {
        Some("function_call") => {
            let call_id = item
                .get("call_id")
                .and_then(|c| c.as_str())
                .unwrap_or("call");
            vec![json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                        "arguments": item.get("arguments").and_then(|a| a.as_str()).unwrap_or("")
                    }
                }]
            })]
        }
        Some("function_call_output") => {
            vec![json!({
                "role": "tool",
                "tool_call_id": item.get("call_id").and_then(|c| c.as_str()).unwrap_or(""),
                "content": item.get("output").and_then(|o| o.as_str()).unwrap_or("")
            })]
        }
        Some("message") => {
            let role = item
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("assistant");
            let text = crate::items::extract_message_text(item).unwrap_or_default();
            vec![json!({"role": role, "content": text})]
        }
        _ => {
            if let Some(role) = item.get("role").and_then(|r| r.as_str()) {
                let content = item
                    .get("content")
                    .cloned()
                    .unwrap_or(Value::String(String::new()));
                vec![json!({"role": role, "content": content})]
            } else {
                vec![]
            }
        }
    }
}

fn chat_payload_to_model_response(payload: Value) -> Result<ModelResponse, ModelError> {
    let choice = payload
        .pointer("/choices/0/message")
        .ok_or_else(|| ModelError::Behavior("missing choices[0].message".into()))?;

    let mut output: Vec<ResponseOutputItem> = Vec::new();

    if let Some(tool_calls) = choice.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tool_calls {
            let id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("call");
            let name = tc
                .pointer("/function/name")
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let arguments = tc
                .pointer("/function/arguments")
                .and_then(|a| a.as_str())
                .unwrap_or("");
            output.push(ItemHelpers::function_tool_call(name, arguments, id));
        }
    }

    if let Some(content) = choice.get("content").and_then(|c| c.as_str()) {
        if !content.is_empty() {
            output.push(ItemHelpers::text_message(content));
        }
    }

    let usage = payload.get("usage").map(|u| {
        let input = u
            .get("prompt_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output_toks = u
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        (input, output_toks)
    });

    Ok(ModelResponse {
        output,
        usage: merge_usage(usage),
        response_id: payload
            .get("id")
            .and_then(|i| i.as_str())
            .map(str::to_string),
        request_id: None,
    })
}

#[allow(dead_code)]
fn _usage_ping() -> Usage {
    Usage::default()
}
