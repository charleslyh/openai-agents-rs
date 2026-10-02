//! HTTP mocks for OpenAI Responses and Chat Completions (feature `testing`).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use crate::model::openai::{OpenAIChatCompletionsModel, OpenAIResponsesModel};

/// One function/tool call in mock provider JSON.
#[derive(Debug, Clone)]
pub struct MockToolCall {
    /// Tool name.
    pub name: String,
    /// JSON arguments string.
    pub arguments: String,
    /// Provider call id.
    pub call_id: String,
}

impl MockToolCall {
    /// Build a mock tool call.
    pub fn new(
        name: impl Into<String>,
        arguments: impl Into<String>,
        call_id: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            arguments: arguments.into(),
            call_id: call_id.into(),
        }
    }
}

struct SequentialJson {
    bodies: Mutex<VecDeque<Value>>,
}

#[derive(Clone)]
struct SequentialResponder(Arc<SequentialJson>);

impl SequentialJson {
    fn new() -> SequentialResponder {
        SequentialResponder(Arc::new(Self {
            bodies: Mutex::new(VecDeque::new()),
        }))
    }

    fn push(&self, body: Value) {
        self.bodies.lock().expect("queue").push_back(body);
    }
}

impl SequentialResponder {
    fn push(&self, body: Value) {
        self.0.push(body);
    }
}

impl Respond for SequentialResponder {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        match self.0.bodies.lock().expect("queue").pop_front() {
            Some(body) => ResponseTemplate::new(200).set_body_json(body),
            None => ResponseTemplate::new(500).set_body_json(json!({
                "error": {"message": "MockResponses/MockCompletions: no queued body"}
            })),
        }
    }
}

/// Queued mock of `POST /v1/responses`.
pub struct MockResponses {
    server: MockServer,
    queue: SequentialResponder,
}

impl MockResponses {
    /// Bind a listener and mount the sequential responder.
    pub async fn start() -> Self {
        let server = MockServer::start().await;
        let queue = SequentialJson::new();
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(queue.clone())
            .mount(&server)
            .await;
        Self { server, queue }
    }

    /// Enqueue a raw Responses JSON body.
    pub fn enqueue(&self, body: Value) {
        self.queue.push(body);
    }

    /// Enqueue a completed assistant text message.
    pub fn enqueue_text(&self, text: impl Into<String>) {
        let text = text.into();
        self.enqueue(json!({
            "id": "resp_mock",
            "object": "response",
            "output": [{
                "id": "1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                    "logprobs": []
                }]
            }],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }));
    }

    /// Enqueue one or more `function_call` output items.
    pub fn enqueue_tool_calls(&self, calls: impl IntoIterator<Item = MockToolCall>) {
        let output: Vec<Value> = calls
            .into_iter()
            .map(|c| {
                json!({
                    "id": "1",
                    "type": "function_call",
                    "name": c.name,
                    "arguments": c.arguments,
                    "call_id": c.call_id
                })
            })
            .collect();
        self.enqueue(json!({
            "id": "resp_mock",
            "object": "response",
            "output": output,
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }));
    }

    /// Base URL including `/v1` for [`OpenAIResponsesModel`].
    pub fn base_url(&self) -> String {
        format!("{}/v1", self.server.uri())
    }

    /// Bind an [`OpenAIResponsesModel`] to this mock.
    pub fn model(&self, model: impl Into<String>) -> OpenAIResponsesModel {
        OpenAIResponsesModel::new(model, "sk-mock", Some(&self.base_url()))
    }

    /// Captured request JSON bodies in order.
    pub async fn request_bodies(&self) -> Vec<Value> {
        self.server
            .received_requests()
            .await
            .unwrap_or_default()
            .into_iter()
            .filter_map(|r| r.body_json().ok())
            .collect()
    }
}

/// Queued mock of `POST /v1/chat/completions`.
pub struct MockCompletions {
    server: MockServer,
    queue: SequentialResponder,
}

impl MockCompletions {
    /// Bind a listener and mount the sequential responder.
    pub async fn start() -> Self {
        let server = MockServer::start().await;
        let queue = SequentialJson::new();
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(queue.clone())
            .mount(&server)
            .await;
        Self { server, queue }
    }

    /// Enqueue a raw Chat Completions JSON body.
    pub fn enqueue(&self, body: Value) {
        self.queue.push(body);
    }

    /// Enqueue a stop message.
    pub fn enqueue_text(&self, text: impl Into<String>) {
        let text = text.into();
        self.enqueue(json!({
            "id": "chatcmpl_mock",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }));
    }

    /// Enqueue assistant `tool_calls`.
    pub fn enqueue_tool_calls(&self, calls: impl IntoIterator<Item = MockToolCall>) {
        let tool_calls: Vec<Value> = calls
            .into_iter()
            .map(|c| {
                json!({
                    "id": c.call_id,
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": c.arguments
                    }
                })
            })
            .collect();
        self.enqueue(json!({
            "id": "chatcmpl_mock",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": tool_calls
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }));
    }

    /// Base URL including `/v1` for [`OpenAIChatCompletionsModel`].
    pub fn base_url(&self) -> String {
        format!("{}/v1", self.server.uri())
    }

    /// Bind an [`OpenAIChatCompletionsModel`] to this mock.
    pub fn model(&self, model: impl Into<String>) -> OpenAIChatCompletionsModel {
        OpenAIChatCompletionsModel::new(model, "sk-mock", Some(&self.base_url()))
    }

    /// Captured request JSON bodies in order.
    pub async fn request_bodies(&self) -> Vec<Value> {
        self.server
            .received_requests()
            .await
            .unwrap_or_default()
            .into_iter()
            .filter_map(|r| r.body_json().ok())
            .collect()
    }
}
