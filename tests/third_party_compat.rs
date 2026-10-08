//! Behavior against third-party OpenAI-compatible servers (wiremock).
//!
//! This SDK targets any server that speaks the Responses or Chat Completions protocol, and many of
//! them deviate from OpenAI in small ways. These tests pin what the adapters send and how they
//! cope with replies that are not byte-for-byte what OpenAI would answer.
#![cfg(feature = "openai")]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use openai_agents::{
    Agent, AgentsError, FunctionTool, ModelError, ModelSettings, OpenAIChatCompletionsModel,
    OpenAIResponsesModel, RunOptions, Runner,
};
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// Replays queued JSON bodies in order.
struct Queued(Mutex<VecDeque<Value>>);

impl Respond for Queued {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        match self.0.lock().unwrap().pop_front() {
            Some(body) => ResponseTemplate::new(200).set_body_json(body),
            None => ResponseTemplate::new(500).set_body_json(json!({"error": {"message": "empty"}})),
        }
    }
}

async fn chat_server(bodies: Vec<Value>) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Queued(Mutex::new(bodies.into())))
        .mount(&server)
        .await;
    server
}

async fn sse_server(body: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body.to_string()),
        )
        .mount(&server)
        .await;
    server
}

fn chat_model(server: &MockServer, model: &str) -> Arc<OpenAIChatCompletionsModel> {
    Arc::new(OpenAIChatCompletionsModel::new(
        model,
        "sk-test",
        Some(&format!("{}/v1", server.uri())),
    ))
}

fn completion(message: Value, finish_reason: &str) -> Value {
    json!({
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })
}

fn tool_call(id: Value, name: &str) -> Value {
    json!({"id": id, "type": "function", "function": {"name": name, "arguments": "{}"}})
}

fn echo_tool(name: &str) -> FunctionTool {
    FunctionTool::constant(name, "echo", format!("{name}-result"))
}

async fn request_bodies(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|r| r.body_json().unwrap())
        .collect()
}

/// Two tool calls in one turn go back as one assistant message followed by both results. Most
/// servers (and every strict chat template) reject one assistant message per call.
#[tokio::test]
async fn parallel_tool_calls_are_sent_as_one_assistant_message() {
    let server = chat_server(vec![
        completion(
            json!({"role": "assistant", "content": null,
                   "tool_calls": [tool_call(json!("c1"), "alpha"), tool_call(json!("c2"), "beta")]}),
            "tool_calls",
        ),
        completion(json!({"role": "assistant", "content": "done"}), "stop"),
    ])
    .await;
    let agent = Agent::new("a")
        .model(chat_model(&server, "any-model"))
        .tools(vec![echo_tool("alpha"), echo_tool("beta")]);
    let result = Runner::run(&agent, "go", RunOptions::default()).await.expect("run");
    assert_eq!(result.final_output_as_str(), Some("done"));

    let second = &request_bodies(&server).await[1]["messages"];
    let roles: Vec<&str> = second.as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "tool"], "{second}");
    assert_eq!(second[1]["tool_calls"].as_array().unwrap().len(), 2);
    assert_eq!(second[2]["tool_call_id"], "c1");
    assert_eq!(second[3]["content"], "beta-result");
}

/// Typed user messages keep their text (they used to be sent as an empty string).
#[tokio::test]
async fn typed_user_messages_keep_their_content() {
    let server = chat_server(vec![completion(json!({"role": "assistant", "content": "ok"}), "stop")]).await;
    let agent = Agent::new("a").model(chat_model(&server, "m"));
    let input = vec![json!({"type": "message", "role": "user",
                            "content": [{"type": "input_text", "text": "hello there"}]})];
    Runner::run(&agent, input, RunOptions::default()).await.expect("run");
    let messages = &request_bodies(&server).await[0]["messages"];
    assert_eq!(messages[0]["content"][0], json!({"type": "text", "text": "hello there"}));
}

/// The structured-output schema is always named `final_output`, a name every server accepts
/// (a Rust type path such as `crate::Answer` is not a valid schema name).
#[tokio::test]
async fn structured_output_schema_name_is_valid() {
    #[derive(serde::Deserialize, openai_agents::schemars::JsonSchema)]
    #[allow(dead_code)]
    struct Answer {
        value: String,
    }
    let schema = Arc::new(openai_agents::AgentOutputSchema::of::<Answer>().unwrap());

    let server = chat_server(vec![completion(
        json!({"role": "assistant", "content": "{\"value\":\"x\"}"}),
        "stop",
    )])
    .await;
    let agent = Agent::new("a").model(chat_model(&server, "m")).output_type(schema.clone());
    Runner::run(&agent, "go", RunOptions::default()).await.expect("chat run");
    let body = &request_bodies(&server).await[0];
    assert_eq!(body["response_format"]["json_schema"]["name"], "final_output");

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1", "object": "response", "status": "completed", "model": "m",
            "output": [{"id": "m1", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "{\"value\":\"x\"}", "annotations": []}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        })))
        .mount(&server)
        .await;
    let model = Arc::new(OpenAIResponsesModel::new("m", "sk", Some(&format!("{}/v1", server.uri()))));
    let agent = Agent::new("a").model(model).output_type(schema);
    Runner::run(&agent, "go", RunOptions::default()).await.expect("responses run");
    assert_eq!(request_bodies(&server).await[0]["text"]["format"]["name"], "final_output");
}

/// `stream_options` is a parameter some servers reject, so it is only sent by default to OpenAI
/// itself; `include_usage` opts in everywhere else.
#[tokio::test]
async fn stream_options_are_opt_in_for_third_party_servers() {
    let sse = "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";
    for (include_usage, expected) in [(None, None), (Some(true), Some(true)), (Some(false), Some(false))] {
        let server = sse_server(sse).await;
        let mut options = RunOptions::default();
        options.run_config.model_settings = Some(ModelSettings {
            include_usage,
            ..ModelSettings::default()
        });
        let agent = Agent::new("a").model(chat_model(&server, "m"));
        let mut streamed = Runner::run_streamed(agent, "go", options);
        streamed.collect_events().await.expect("events");
        let body = &request_bodies(&server).await[0];
        assert_eq!(body["stream"], true);
        assert_eq!(
            body.get("stream_options").map(|o| o["include_usage"].as_bool().unwrap()),
            expected,
            "include_usage={include_usage:?}"
        );
    }
}

/// Servers differ from OpenAI in small ways; none of these should break the tool loop.
#[tokio::test]
async fn lenient_reply_shapes_still_drive_the_tool_loop() {
    let server = chat_server(vec![
        completion(
            json!({
                "role": "assistant",
                // content as a list of parts, a tool call without an id, arguments as an object
                "content": [{"type": "text", "text": "thinking..."}],
                "tool_calls": [{"type": "function", "function": {"name": "alpha", "arguments": {"q": 1}}}]
            }),
            "tool_calls",
        ),
        completion(json!({"role": "assistant", "content": "done"}), "stop"),
    ])
    .await;
    let agent = Agent::new("a").model(chat_model(&server, "m")).tools(vec![echo_tool("alpha")]);
    let result = Runner::run(&agent, "go", RunOptions::default()).await.expect("run");
    assert_eq!(result.final_output_as_str(), Some("done"));

    let second = &request_bodies(&server).await[1]["messages"];
    assert_eq!(second[1]["content"], "thinking...", "text and call share one message");
    let call = &second[1]["tool_calls"][0];
    let id = call["id"].as_str().unwrap();
    assert!(id.starts_with("call_") && id.len() > 8, "generated id: {id}");
    assert_eq!(call["function"]["arguments"], "{\"q\":1}");
    assert_eq!(second[2]["tool_call_id"], id, "the result is paired with the generated id");
}

/// An empty completion is explained by `finish_reason` (Python parity): filtered output is a
/// refusal the caller can handle, truncation before any token is a behavior error.
#[tokio::test]
async fn finish_reason_explains_empty_completions() {
    let empty = json!({"role": "assistant", "content": null});
    let server = chat_server(vec![completion(empty.clone(), "content_filter")]).await;
    let agent = Agent::new("a").model(chat_model(&server, "m"));
    match Runner::run(&agent, "go", RunOptions::default()).await.unwrap_err() {
        AgentsError::ModelRefusal(e) => assert!(e.refusal.contains("content filter"), "{}", e.refusal),
        other => panic!("expected a refusal, got {other}"),
    }

    let server = chat_server(vec![completion(empty, "length")]).await;
    let agent = Agent::new("a").model(chat_model(&server, "m"));
    let error = Runner::run(&agent, "go", RunOptions::default()).await.unwrap_err();
    assert!(matches!(error, AgentsError::Model(ModelError::Behavior(_))), "{error}");

    // A reply that has content is left alone whatever the finish reason says.
    let server = chat_server(vec![completion(json!({"role": "assistant", "content": "cut"}), "length")]).await;
    let agent = Agent::new("a").model(chat_model(&server, "m"));
    let result = Runner::run(&agent, "go", RunOptions::default()).await.unwrap();
    assert_eq!(result.final_output_as_str(), Some("cut"));
}

/// A reply `refusal` field becomes a refusal (it used to be dropped, leaving an empty answer).
#[tokio::test]
async fn reply_refusal_field_is_a_refusal() {
    let server = chat_server(vec![completion(
        json!({"role": "assistant", "content": null, "refusal": "I can't help with that"}),
        "stop",
    )])
    .await;
    let agent = Agent::new("a").model(chat_model(&server, "m"));
    match Runner::run(&agent, "go", RunOptions::default()).await.unwrap_err() {
        AgentsError::ModelRefusal(e) => assert_eq!(e.refusal, "I can't help with that"),
        other => panic!("expected a refusal, got {other}"),
    }
}

/// DeepSeek wants the chain of thought of a tool-calling turn sent back; other models do not.
#[tokio::test]
async fn reasoning_content_is_replayed_for_deepseek_only() {
    for (model, replayed) in [("deepseek-reasoner", true), ("llama-3", false)] {
        let server = chat_server(vec![
            completion(
                json!({"role": "assistant", "content": null, "reasoning_content": "I should call alpha",
                       "tool_calls": [tool_call(json!("c1"), "alpha")]}),
                "tool_calls",
            ),
            completion(json!({"role": "assistant", "content": "done"}), "stop"),
        ])
        .await;
        let agent = Agent::new("a").model(chat_model(&server, model)).tools(vec![echo_tool("alpha")]);
        Runner::run(&agent, "go", RunOptions::default()).await.expect("run");
        let assistant = &request_bodies(&server).await[1]["messages"][1];
        assert_eq!(
            assistant.get("reasoning_content").and_then(Value::as_str),
            replayed.then_some("I should call alpha"),
            "{model}: {assistant}"
        );
    }

    // The rule can be overridden per model instance.
    let server = chat_server(vec![
        completion(
            json!({"role": "assistant", "content": null, "reasoning_content": "plan",
                   "tool_calls": [tool_call(json!("c1"), "alpha")]}),
            "tool_calls",
        ),
        completion(json!({"role": "assistant", "content": "done"}), "stop"),
    ])
    .await;
    let model = OpenAIChatCompletionsModel::new("qwen", "sk", Some(&format!("{}/v1", server.uri())))
        .should_replay_reasoning_content(|model, _| model.starts_with("qwen"));
    let agent = Agent::new("a").model(Arc::new(model)).tools(vec![echo_tool("alpha")]);
    Runner::run(&agent, "go", RunOptions::default()).await.expect("run");
    assert_eq!(request_bodies(&server).await[1]["messages"][1]["reasoning_content"], "plan");
}

fn chunk(delta: Value, finish: Option<&str>) -> String {
    let payload = json!({"id": "c", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
    format!("data: {payload}\n\n")
}

/// Streaming: a refusal arrives as `delta.refusal`, and surfaces as a refusal.
#[tokio::test]
async fn streamed_refusal_deltas_become_a_refusal() {
    let body = [
        chunk(json!({"refusal": "I can't "}), None),
        chunk(json!({"refusal": "do that"}), Some("stop")),
        "data: [DONE]\n\n".to_string(),
    ]
    .concat();
    let server = sse_server(&body).await;
    let agent = Agent::new("a").model(chat_model(&server, "m"));
    let mut streamed = Runner::run_streamed(agent, "go", RunOptions::default());
    let events = streamed.collect_events().await;
    match events {
        Err(AgentsError::ModelRefusal(e)) => assert_eq!(e.refusal, "I can't do that"),
        other => panic!("expected a refusal, got {:?}", other.map(|e| e.len())),
    }
}

/// Streaming mirrors the non-streaming finish-reason rules, and a mid-stream `error` chunk is an
/// error rather than an empty answer.
#[tokio::test]
async fn streamed_failures_are_reported() {
    let cases: [(Vec<String>, &str); 3] = [
        (vec![chunk(json!({}), Some("content_filter"))], "refusal"),
        (vec![chunk(json!({}), Some("length"))], "behavior"),
        (vec![format!("data: {}\n\n", json!({"error": {"message": "overloaded"}}))], "behavior"),
    ];
    for (chunks, kind) in cases {
        let server = sse_server(&chunks.concat()).await;
        let agent = Agent::new("a").model(chat_model(&server, "m"));
        let mut streamed = Runner::run_streamed(agent, "go", RunOptions::default());
        match (streamed.collect_events().await, kind) {
            (Err(AgentsError::ModelRefusal(_)), "refusal") => {}
            (Err(AgentsError::Model(ModelError::Behavior(_))), "behavior") => {}
            (other, _) => panic!("{kind}: unexpected {:?}", other.map(|e| e.len())),
        }
    }
}

/// Streaming tool calls without an id still pair with their result, and SSE framing quirks
/// (comments, CRLF, `data:` without a space) are tolerated.
#[tokio::test]
async fn streamed_tool_call_without_id_and_odd_sse_framing() {
    let first = [
        ": keep-alive\r\n\r\n".to_string(),
        format!(
            "data:{}\r\n\r\n",
            json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"name": "alpha", "arguments": "{}"}}]}}]})
        ),
        "data: [DONE]\r\n\r\n".to_string(),
    ]
    .concat();
    let second = [chunk(json!({"content": "done"}), Some("stop")), "data: [DONE]\n\n".to_string()].concat();
    let server = MockServer::start().await;
    let bodies = Mutex::new(VecDeque::from([first, second]));
    struct Sse(Mutex<VecDeque<String>>);
    impl Respond for Sse {
        fn respond(&self, _: &Request) -> ResponseTemplate {
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(self.0.lock().unwrap().pop_front().unwrap_or_default())
        }
    }
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Sse(bodies))
        .mount(&server)
        .await;

    let agent = Agent::new("a").model(chat_model(&server, "m")).tools(vec![echo_tool("alpha")]);
    let mut streamed = Runner::run_streamed(agent, "go", RunOptions::default());
    streamed.collect_events().await.expect("events");
    assert_eq!(streamed.final_output().and_then(|v| v.as_str().map(str::to_string)), Some("done".into()));

    let second = &request_bodies(&server).await[1]["messages"];
    let id = second[1]["tool_calls"][0]["id"].as_str().unwrap();
    assert!(id.starts_with("call_"), "{id}");
    assert_eq!(second[2]["tool_call_id"], id);
}

fn settings(f: impl FnOnce(&mut ModelSettings)) -> RunOptions {
    let mut model_settings = ModelSettings::default();
    f(&mut model_settings);
    let mut options = RunOptions::default();
    options.run_config.model_settings = Some(model_settings);
    options
}

/// `extra_query` reaches the URL of both APIs: scalars as text, arrays as repeated keys, `null`
/// skipped (Python: `ModelSettings.extra_query`).
#[tokio::test]
async fn extra_query_is_sent_with_the_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "c", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"}}]
        })))
        .mount(&server)
        .await;
    let options = settings(|s| {
        s.extra_query = Some(
            json!({"api-version": "2024-06-01", "n": 3, "tag": ["a", "b"], "skip": null})
                .as_object()
                .unwrap()
                .clone(),
        );
    });
    let agent = Agent::new("a").model(chat_model(&server, "m"));
    Runner::run(&agent, "go", options).await.expect("run");
    let url = server.received_requests().await.unwrap()[0].url.clone();
    let pairs: Vec<(String, String)> =
        url.query_pairs().map(|(k, v)| (k.into_owned(), v.into_owned())).collect();
    assert_eq!(
        pairs,
        [("api-version", "2024-06-01"), ("n", "3"), ("tag", "a"), ("tag", "b")]
            .map(|(k, v)| (k.to_string(), v.to_string()))
    );
}

/// `preserve_raw_usage` keeps the provider's usage object, including fields `Usage` does not
/// model, for plain and streamed Chat Completions; without it nothing is kept.
#[tokio::test]
async fn preserve_raw_usage_keeps_the_provider_usage_object() {
    let usage = json!({"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5, "cost": 0.0012});
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "c", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"}}],
            "usage": usage
        })))
        .mount(&server)
        .await;
    let agent = Agent::new("a").model(chat_model(&server, "m"));

    let result = Runner::run(&agent, "go", settings(|s| s.preserve_raw_usage = Some(true)))
        .await
        .expect("run");
    assert_eq!(result.raw_responses[0].raw_usage, Some(usage.clone()));
    assert_eq!(result.usage.total_tokens, 5);

    let result = Runner::run(&agent, "go", RunOptions::default()).await.expect("run");
    assert_eq!(result.raw_responses[0].raw_usage, None, "off by default");

    let sse = format!(
        "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
        json!({"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"}}]}),
        json!({"id": "c", "choices": [], "usage": usage}),
    );
    let server = sse_server(&sse).await;
    let agent = Agent::new("a").model(chat_model(&server, "m"));
    let mut streamed = Runner::run_streamed(
        agent,
        "go",
        settings(|s| {
            s.preserve_raw_usage = Some(true);
            s.include_usage = Some(true);
        }),
    );
    streamed.collect_events().await.expect("events");
    let snapshot = streamed.snapshot.lock().unwrap();
    assert_eq!(snapshot.raw_responses[0].raw_usage, Some(usage));
}
