//! OpenAI wiremock contract tests (verification layer 2).
#![cfg(feature = "openai")]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use openai_agents::{
    Agent, FunctionTool, ModelSettings, OpenAIChatCompletionsModel, OpenAIProvider,
    OpenAIResponsesModel, RunOptions, Runner, StreamEvent, ToolChoice, Truncation, Verbosity,
};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Test-local responder that replays queued bodies in order.
///
/// Only the tests in this file use it; the crate does not ship an HTTP mock layer.
struct QueuedBodies(Mutex<VecDeque<serde_json::Value>>);

impl QueuedBodies {
    fn new(bodies: Vec<serde_json::Value>) -> Self {
        Self(Mutex::new(bodies.into()))
    }
}

impl wiremock::Respond for QueuedBodies {
    fn respond(&self, _request: &wiremock::Request) -> ResponseTemplate {
        match self.0.lock().expect("queue").pop_front() {
            Some(body) => ResponseTemplate::new(200).set_body_json(body),
            None => ResponseTemplate::new(500)
                .set_body_json(json!({"error": {"message": "no queued body"}})),
        }
    }
}

fn weather_tool() -> FunctionTool {
    FunctionTool::new(
        "get_weather",
        "weather",
        json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }),
        |_ctx, args| async move {
            let v: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
            let city = v.get("city").and_then(|c| c.as_str()).unwrap_or("?");
            Ok(serde_json::Value::String(format!("sunny in {city}")))
        },
    )
}

/// A multi-turn Responses loop must round-trip `function_call_output` back to the API.
///
/// This is the assertion `ScriptedModel` cannot make: only the real adapter proves the second
/// request carries the tool result in Responses wire format.
#[tokio::test]
async fn responses_multi_turn_tool_loop() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(QueuedBodies::new(vec![
            json!({
                "id": "resp_1",
                "object": "response",
                "output": [{
                    "id": "1",
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Paris\"}",
                    "call_id": "call-w"
                }],
                "usage": {"input_tokens": 1, "output_tokens": 1}
            }),
            json!({
                "id": "resp_2",
                "object": "response",
                "output": [{
                    "id": "2",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "It is sunny in Paris.", "annotations": [], "logprobs": []}]
                }],
                "usage": {"input_tokens": 1, "output_tokens": 1}
            }),
        ]))
        .mount(&server)
        .await;

    let agent = Agent::new("assistant")
        .instructions("use tools")
        .model(Arc::new(OpenAIResponsesModel::new(
            "mock-model",
            "sk-test",
            Some(&format!("{}/v1", server.uri())),
        )))
        .tools(vec![weather_tool()]);

    let result = Runner::run(&agent, "weather in Paris?", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("It is sunny in Paris."));
    assert_eq!(result.raw_responses.len(), 2);

    let requests = server.received_requests().await.expect("reqs");
    assert_eq!(requests.len(), 2);
    let first: serde_json::Value = requests[0].body_json().expect("json");
    assert_eq!(first["model"], "mock-model");
    assert!(first["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t["name"] == "get_weather"));

    // The second request must replay the tool result as a `function_call_output` item.
    let second: serde_json::Value = requests[1].body_json().expect("json");
    let input = second["input"].as_array().expect("turn-2 input");
    assert!(
        input.iter().any(|i| i["type"] == "function_call_output"
            && i["call_id"] == "call-w"
            && i["output"].as_str().unwrap().contains("Paris")),
        "turn-2 input missing function_call_output: {input:?}"
    );
}

/// The same loop over Chat Completions must use `messages` with `role: tool`.
#[tokio::test]
async fn chat_multi_turn_tool_loop() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(QueuedBodies::new(vec![
            json!({
                "id": "chatcmpl_1",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call-w",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\":\"Tokyo\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }),
            json!({
                "id": "chatcmpl_2",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Tokyo is sunny."},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }),
        ]))
        .mount(&server)
        .await;

    let agent = Agent::new("assistant")
        .model(Arc::new(OpenAIChatCompletionsModel::new(
            "mock-chat",
            "sk-test",
            Some(&format!("{}/v1", server.uri())),
        )))
        .tools(vec![weather_tool()]);

    let result = Runner::run(&agent, "weather?", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("Tokyo is sunny."));
    assert_eq!(result.raw_responses.len(), 2);

    let requests = server.received_requests().await.expect("reqs");
    assert_eq!(requests.len(), 2);
    let first: serde_json::Value = requests[0].body_json().expect("json");
    assert!(first["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t["function"]["name"] == "get_weather"));

    // Chat Completions replays the result as a `tool` message keyed by `tool_call_id`.
    let second: serde_json::Value = requests[1].body_json().expect("json");
    let messages = second["messages"].as_array().expect("messages");
    assert!(
        messages.iter().any(|m| m["role"] == "tool"
            && m["tool_call_id"] == "call-w"
            && m["content"].as_str().unwrap().contains("Tokyo")),
        "turn-2 messages missing tool result: {messages:?}"
    );
}

/// Settings are stored as `f32`, so compare with a tolerance.
fn approx(actual: &serde_json::Value, expected: f64) {
    let got = actual.as_f64().unwrap_or(f64::NAN);
    assert!(
        (got - expected).abs() < 1e-4,
        "expected {expected}, got {got}"
    );
}

fn new_settings() -> ModelSettings {
    let mut metadata = serde_json::Map::new();
    metadata.insert("tenant".into(), json!("acme"));
    ModelSettings {
        temperature: Some(0.25),
        top_p: Some(0.9),
        frequency_penalty: Some(0.1),
        presence_penalty: Some(0.2),
        tool_choice: Some(ToolChoice::Function("noop".into())),
        parallel_tool_calls: Some(false),
        truncation: Some(Truncation::Auto),
        max_tokens: Some(512),
        reasoning: Some(json!({"effort": "medium"})),
        verbosity: Some(Verbosity::Low),
        metadata: Some(metadata),
        store: Some(false),
        top_logprobs: Some(3),
        response_include: Some(vec!["file_search_call.results".into()]),
        ..Default::default()
    }
}

#[tokio::test]
async fn responses_request_shape() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_test",
            "object": "response",
            "output": [{
                "id": "1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "pong", "annotations": [], "logprobs": []}]
            }],
            "usage": {"input_tokens": 3, "output_tokens": 1}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let model = Arc::new(OpenAIResponsesModel::new(
        "gpt-test",
        "sk-test",
        Some(&format!("{}/v1", server.uri())),
    ));
    let tool = FunctionTool::constant("noop", "noop", "x");
    let agent = Agent::new("a")
        .instructions("be brief")
        .model(model)
        .tools(vec![tool]);

    let result = Runner::run(&agent, "ping", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("pong"));

    let requests = server.received_requests().await.expect("reqs");
    assert_eq!(requests.len(), 1);
    let body: serde_json::Value = requests[0].body_json().expect("json");
    assert_eq!(body["model"], "gpt-test");
    assert_eq!(body["instructions"], "be brief");
    assert!(body["input"].is_array());
    assert!(body["tools"].as_array().unwrap().iter().any(|t| t["name"] == "noop"));
}

#[tokio::test]
async fn chat_completions_request_shape() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "pong"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let model = Arc::new(OpenAIChatCompletionsModel::new(
        "gpt-test",
        "sk-test",
        Some(&format!("{}/v1", server.uri())),
    ));
    let tool = FunctionTool::constant("noop", "noop", "x");
    let agent = Agent::new("a")
        .instructions("be brief")
        .model(model)
        .tools(vec![tool]);

    let result = Runner::run(&agent, "ping", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("pong"));

    let requests = server.received_requests().await.expect("reqs");
    assert_eq!(requests.len(), 1);
    let body: serde_json::Value = requests[0].body_json().expect("json");
    assert_eq!(body["model"], "gpt-test");
    assert!(body["messages"].as_array().unwrap().iter().any(|m| m["role"] == "system"));
    assert!(body["tools"].as_array().unwrap().iter().any(|t| {
        t["type"] == "function" && t["function"]["name"] == "noop"
    }));
}

/// Render a `text/event-stream` body from Responses / Chat Completions SSE events.
fn sse_body(events: &[serde_json::Value]) -> String {
    events.iter().map(|e| format!("data: {e}\n\n")).collect()
}

fn responses_message_item(id: &str, text: &str) -> serde_json::Value {
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}]
    })
}

/// Collect the `delta` strings of raw events with the given type.
fn deltas_of_kind(events: &[StreamEvent], kind: &str) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::RawResponse { data } => {
                let is_kind = data.get("type").and_then(|t| t.as_str()) == Some(kind);
                match (is_kind, data.get("delta").and_then(|d| d.as_str())) {
                    (true, Some(delta)) => Some(delta.to_string()),
                    _ => None,
                }
            }
            _ => None,
        })
        .collect()
}

/// The `type` of every raw wire event, in emission order.
fn raw_types(events: &[StreamEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::RawResponse { data } => {
                data.get("type").and_then(|t| t.as_str()).map(str::to_string)
            }
            _ => None,
        })
        .collect()
}

/// `sequence_number` of every raw wire event, in emission order.
fn sequence_numbers(events: &[StreamEvent]) -> Vec<u64> {
    events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::RawResponse { data } => data
                .get("sequence_number")
                .and_then(|s| s.as_u64()),
            _ => None,
        })
        .collect()
}

/// The `response` object carried by the terminal `response.completed` event.
fn completed_response(events: &[StreamEvent]) -> serde_json::Value {
    events
        .iter()
        .rev()
        .find_map(|e| match e {
            StreamEvent::RawResponse { data }
                if data.get("type").and_then(|t| t.as_str()) == Some("response.completed") =>
            {
                data.get("response").cloned()
            }
            _ => None,
        })
        .expect("response.completed event")
}

/// Responses API streams token deltas, not just a synthetic `response.completed` (D-011).
#[tokio::test]
async fn responses_stream_emits_token_deltas() {
    let server = MockServer::start().await;
    let message = responses_message_item("msg_1", "Hello world");
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body(&[
                    json!({"type": "response.created", "sequence_number": 0, "response": {"id": "resp_stream", "status": "in_progress"}}),
                    json!({"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": message}),
                    json!({"type": "response.reasoning_summary_text.delta", "sequence_number": 2, "item_id": "rs_1", "output_index": 0, "delta": "thinking"}),
                    json!({"type": "response.output_text.delta", "sequence_number": 3, "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "Hello"}),
                    json!({"type": "response.output_text.delta", "sequence_number": 4, "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": " world"}),
                    json!({"type": "response.output_item.done", "sequence_number": 5, "output_index": 0, "item": message}),
                    json!({"type": "response.completed", "sequence_number": 6, "response": {
                        "id": "resp_stream",
                        "output": [message],
                        "usage": {"input_tokens": 2, "output_tokens": 3}
                    }}),
                ])),
        )
        .expect(1)
        .mount(&server)
        .await;

    let agent = Agent::new("a").model(Arc::new(OpenAIResponsesModel::new(
        "gpt-test",
        "sk-test",
        Some(&format!("{}/v1", server.uri())),
    )));

    let mut streamed = Runner::run_streamed(agent, "ping", RunOptions::default());
    let events = streamed.collect_events().await.expect("events");

    assert_eq!(
        deltas_of_kind(&events, "response.output_text.delta"),
        ["Hello", " world"]
    );
    assert_eq!(
        deltas_of_kind(&events, "response.reasoning_summary_text.delta"),
        ["thinking"]
    );
    // Events are forwarded verbatim, so the wire `type`s and `sequence_number`s survive.
    assert_eq!(
        raw_types(&events),
        [
            "response.created",
            "response.output_item.added",
            "response.reasoning_summary_text.delta",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert_eq!(sequence_numbers(&events), [0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(
        completed_response(&events)["output"][0]["content"][0]["text"],
        "Hello world"
    );
    assert_eq!(
        streamed.final_output().and_then(|v| v.as_str().map(str::to_string)),
        Some("Hello world".into())
    );

    let body: serde_json::Value = server.received_requests().await.expect("reqs")[0]
        .body_json()
        .expect("json");
    assert_eq!(body["stream"], true);
}

/// Gateways that never send `response.completed` still yield the assembled output.
#[tokio::test]
async fn responses_stream_assembles_without_completed_event() {
    let server = MockServer::start().await;
    let message = responses_message_item("msg_1", "partial");
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body(&[
                    json!({"type": "response.created", "response": {"id": "resp_partial"}}),
                    json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "delta": "partial"}),
                    json!({"type": "response.output_item.done", "output_index": 0, "item": message}),
                ])),
        )
        .expect(1)
        .mount(&server)
        .await;

    let agent = Agent::new("a").model(Arc::new(OpenAIResponsesModel::new(
        "gpt-test",
        "sk-test",
        Some(&format!("{}/v1", server.uri())),
    )));

    let mut streamed = Runner::run_streamed(agent, "ping", RunOptions::default());
    let events = streamed.collect_events().await.expect("events");
    assert_eq!(
        deltas_of_kind(&events, "response.output_text.delta"),
        ["partial"]
    );
    // The synthesized terminal event continues the numbering of the forwarded events.
    assert_eq!(raw_types(&events).last().map(String::as_str), Some("response.completed"));
    assert_eq!(
        streamed.final_output().and_then(|v| v.as_str().map(str::to_string)),
        Some("partial".into())
    );
}

/// Chat Completions keeps streaming token deltas over the shared SSE reader.
#[tokio::test]
async fn chat_stream_emits_token_deltas() {
    let server = MockServer::start().await;
    let chunk = |content: &str| {
        json!({
            "id": "chatcmpl_stream",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": content}}]
        })
    };
    let mut events = vec![chunk("Hi"), chunk(" there")];
    events.push(json!({
        "id": "chatcmpl_stream",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2}
    }));
    let mut body = sse_body(&events);
    body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .expect(1)
        .mount(&server)
        .await;

    let agent = Agent::new("a").model(Arc::new(OpenAIChatCompletionsModel::new(
        "gpt-test",
        "sk-test",
        Some(&format!("{}/v1", server.uri())),
    )));

    let mut streamed = Runner::run_streamed(agent, "ping", RunOptions::default());
    let events = streamed.collect_events().await.expect("events");
    assert_eq!(
        deltas_of_kind(&events, "response.output_text.delta"),
        ["Hi", " there"]
    );
    // Chat Completions synthesizes the standard Responses event sequence.
    assert_eq!(
        raw_types(&events),
        [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    let expected: Vec<u64> = (0..raw_types(&events).len() as u64).collect();
    assert_eq!(sequence_numbers(&events), expected);
    let completed = completed_response(&events);
    assert_eq!(completed["object"], "response");
    assert_eq!(completed["status"], "completed");
    assert_eq!(
        completed["output"][0]["content"][0]["text"],
        "Hi there"
    );
    assert_eq!(
        streamed.final_output().and_then(|v| v.as_str().map(str::to_string)),
        Some("Hi there".into())
    );

    let body: serde_json::Value = server.received_requests().await.expect("reqs")[0]
        .body_json()
        .expect("json");
    assert_eq!(body["stream"], true);
}

/// Expanded `ModelSettings` must reach the Responses request with Responses field names.
#[tokio::test]
async fn responses_maps_expanded_model_settings() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_test",
            "object": "response",
            "output": [{
                "id": "1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "pong", "annotations": [], "logprobs": []}]
            }],
            "usage": {"input_tokens": 3, "output_tokens": 1}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let model = Arc::new(OpenAIResponsesModel::new(
        "gpt-test",
        "sk-test",
        Some(&format!("{}/v1", server.uri())),
    ));
    let agent = Agent::new("a")
        .model(model)
        .tools(vec![FunctionTool::constant("noop", "noop", "x")])
        .model_settings(new_settings());

    let _ = Runner::run(&agent, "ping", RunOptions::default())
        .await
        .expect("run");

    let body: serde_json::Value = server.received_requests().await.expect("reqs")[0]
        .body_json()
        .expect("json");
    approx(&body["temperature"], 0.25);
    approx(&body["top_p"], 0.9);
    approx(&body["frequency_penalty"], 0.1);
    approx(&body["presence_penalty"], 0.2);
    // Responses API names it `max_output_tokens`, not `max_tokens`.
    assert_eq!(body["max_output_tokens"], 512);
    assert!(body.get("max_tokens").is_none());
    assert_eq!(body["truncation"], "auto");
    assert_eq!(body["verbosity"], "low");
    assert_eq!(body["reasoning"]["effort"], "medium");
    assert_eq!(body["metadata"]["tenant"], "acme");
    assert_eq!(body["store"], false);
    assert_eq!(body["top_logprobs"], 3);
    assert_eq!(body["include"][0], "file_search_call.results");
    assert_eq!(body["parallel_tool_calls"], false);
    assert_eq!(
        body["tool_choice"],
        json!({"type": "function", "function": {"name": "noop"}})
    );
}

/// The same settings map onto Chat Completions field names.
#[tokio::test]
async fn chat_maps_expanded_model_settings() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "pong"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let model = Arc::new(OpenAIChatCompletionsModel::new(
        "gpt-test",
        "sk-test",
        Some(&format!("{}/v1", server.uri())),
    ));
    let agent = Agent::new("a")
        .model(model)
        .tools(vec![FunctionTool::constant("noop", "noop", "x")])
        .model_settings(new_settings());

    let _ = Runner::run(&agent, "ping", RunOptions::default())
        .await
        .expect("run");

    let body: serde_json::Value = server.received_requests().await.expect("reqs")[0]
        .body_json()
        .expect("json");
    assert_eq!(body["max_tokens"], 512);
    approx(&body["frequency_penalty"], 0.1);
    approx(&body["presence_penalty"], 0.2);
    assert_eq!(body["top_logprobs"], 3);
    assert_eq!(body["store"], false);
    assert_eq!(body["metadata"]["tenant"], "acme");
    // Responses-only settings must not leak into a Chat Completions request.
    assert!(body.get("truncation").is_none());
    assert!(body.get("max_output_tokens").is_none());
    assert!(body.get("include").is_none());
}

#[derive(serde::Deserialize, openai_agents::schemars::JsonSchema)]
#[allow(dead_code)]
struct WiremockAnswer {
    /// The answer.
    value: String,
}

/// Responses API advertises structured output through `text.format`.
#[tokio::test]
async fn responses_advertises_json_schema() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_test",
            "object": "response",
            "output": [{
                "id": "1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "{\"value\":\"ok\"}", "annotations": [], "logprobs": []}]
            }],
            "usage": {"input_tokens": 3, "output_tokens": 1}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let agent = Agent::new("a")
        .model(Arc::new(OpenAIResponsesModel::new(
            "gpt-test",
            "sk-test",
            Some(&format!("{}/v1", server.uri())),
        )))
        .output_type(Arc::new(
            openai_agents::AgentOutputSchema::of::<WiremockAnswer>().expect("schema"),
        ));

    let result = Runner::run(&agent, "ping", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output, json!({"value": "ok"}));

    let body: serde_json::Value = server.received_requests().await.expect("reqs")[0]
        .body_json()
        .expect("json");
    assert_eq!(body["text"]["format"]["type"], "json_schema");
    assert_eq!(body["text"]["format"]["strict"], true);
    assert_eq!(
        body["text"]["format"]["schema"]["properties"]["value"]["type"],
        "string"
    );
}

/// `OpenAIProvider` turns `Agent.model_name` into the API selected by
/// `set_default_openai_api` (Responses by default, Python-aligned).
#[tokio::test]
async fn openai_provider_resolves_model_name() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_test",
            "object": "response",
            "output": [{
                "id": "1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "via responses", "annotations": [], "logprobs": []}]
            }],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .expect(1)
        .mount(&server)
        .await;

    openai_agents::set_default_openai_api(openai_agents::DefaultOpenAiApi::Responses);
    let provider = OpenAIProvider::new("sk-test", Some(&format!("{}/v1", server.uri())), None);

    let agent = Agent::new("named").model_name("gpt-test");
    let mut opts = RunOptions::default();
    opts.run_config.model_provider = Some(Arc::new(provider));
    let result = Runner::run(&agent, "go", opts).await.expect("run");
    assert_eq!(result.final_output_as_str(), Some("via responses"));

    // Switching the global default routes the same name to Chat Completions. This runs in the
    // same test because `set_default_openai_api` is process-global.
    let chat_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "via chat"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .expect(1)
        .mount(&chat_server)
        .await;

    openai_agents::set_default_openai_api(openai_agents::DefaultOpenAiApi::ChatCompletions);
    let provider =
        OpenAIProvider::new("sk-test", Some(&format!("{}/v1", chat_server.uri())), None);
    let mut opts = RunOptions::default();
    opts.run_config.model_provider = Some(Arc::new(provider));
    let result = Runner::run(&agent, "go", opts).await.expect("run");
    assert_eq!(result.final_output_as_str(), Some("via chat"));

    // Restore the crate default for the rest of this binary.
    openai_agents::set_default_openai_api(openai_agents::DefaultOpenAiApi::Responses);
}

/// Chat Completions advertises structured output through `response_format`.
#[tokio::test]
async fn chat_advertises_json_schema() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "{\"value\":\"ok\"}"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let agent = Agent::new("a")
        .model(Arc::new(OpenAIChatCompletionsModel::new(
            "gpt-test",
            "sk-test",
            Some(&format!("{}/v1", server.uri())),
        )))
        .output_type(Arc::new(
            openai_agents::AgentOutputSchema::of::<WiremockAnswer>().expect("schema"),
        ));

    let result = Runner::run(&agent, "ping", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output, json!({"value": "ok"}));

    let body: serde_json::Value = server.received_requests().await.expect("reqs")[0]
        .body_json()
        .expect("json");
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(body["response_format"]["json_schema"]["strict"], true);
    assert_eq!(
        body["response_format"]["json_schema"]["schema"]["properties"]["value"]["type"],
        "string"
    );
}
