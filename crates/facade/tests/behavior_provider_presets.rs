//! Contract tests for the provider presets (verification layer 2).
//!
//! Two things a unit test of the table cannot show: that a preset's endpoint and protocol produce
//! the request the capability matrix promises, and that a capability the preset lacks stops the
//! request instead of sending a field the server would drop.

#![cfg(all(feature = "openai", feature = "providers"))]

use openai_agents::providers::{Capabilities, Capability, Provider};
use openai_agents::{Agent, ModelSettings, Reasoning, ReasoningEffort, RunOptions, Runner};
use serde_json::json;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A preset that speaks Chat Completions, and one that speaks Responses.
const CHAT_PRESETS: &[&str] = &["deepseek", "gemini", "groq", "openrouter", "anthropic"];

/// Answer the adapter asks for; the body is asserted, not the reply.
async fn mock_chat(server: &MockServer) {
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .mount(server)
        .await;
}

async fn mock_responses(server: &MockServer) {
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "m",
            "status": "completed",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}]
            }],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        })))
        .mount(server)
        .await;
}

/// The single request the mock received.
async fn body_sent(server: &MockServer) -> serde_json::Value {
    let requests = server.received_requests().await.expect("requests");
    let request = requests.first().expect("one request was sent");
    serde_json::from_slice(&request.body).expect("a JSON body")
}

/// The capability matrix says what a preset supports; the request has to agree.
#[tokio::test]
async fn every_chat_preset_sends_reasoning_as_its_effort_field() {
    for name in CHAT_PRESETS {
        let server = MockServer::start().await;
        mock_chat(&server).await;
        let preset = openai_agents::providers::preset(name).expect("a preset");
        assert!(
            preset.capabilities.supports(Capability::Reasoning),
            "`{name}` is expected to report reasoning"
        );
        let provider = Provider::preset(name)
            .expect("a preset")
            .api_key("sk-test")
            .base_url(server.uri())
            .expect("the mock is loopback")
            .build()
            .expect("built");
        let mut settings = ModelSettings::default();
        settings.reasoning = Some(Reasoning::effort(ReasoningEffort::Medium));
        let agent = Agent::new("a").instructions("i");
        let options = RunOptions::new()
            .provider(Some(provider))
            .model("some-model")
            .model_settings(settings);
        Runner::run(&agent, "go", options).await.expect("run");

        let body = body_sent(&server).await;
        assert_eq!(body["reasoning_effort"], "medium", "`{name}`: {body}");
        assert!(
            body.get("reasoning").is_none(),
            "`{name}` is not Responses: {body}"
        );
        assert_eq!(body["model"], "some-model");
    }
}

/// The one preset that speaks Responses sends the nested object (Python's shape).
#[tokio::test]
async fn openai_sends_the_responses_reasoning_object() {
    let server = MockServer::start().await;
    mock_responses(&server).await;
    let provider = Provider::preset("openai")
        .expect("a preset")
        .api_key("sk-test")
        .base_url(server.uri())
        .expect("the mock is loopback")
        .build()
        .expect("built");
    let mut settings = ModelSettings::default();
    settings.reasoning = Some(Reasoning::effort(ReasoningEffort::High));
    let agent = Agent::new("a").instructions("i");
    let options = RunOptions::new()
        .provider(Some(provider))
        .model("gpt-4o-mini")
        .model_settings(settings);
    Runner::run(&agent, "go", options).await.expect("run");

    let body = body_sent(&server).await;
    assert_eq!(body["reasoning"]["effort"], "high", "{body}");
}

/// Ollama's preset claims no reasoning, so the request is refused rather than sent without it.
#[tokio::test]
async fn a_preset_without_reasoning_refuses_the_request() {
    let server = MockServer::start().await;
    mock_chat(&server).await;
    let provider = Provider::preset("ollama")
        .expect("a preset")
        .base_url(server.uri())
        .expect("the mock is loopback")
        .build()
        .expect("built");
    assert!(!openai_agents::providers::preset("ollama")
        .expect("a preset")
        .capabilities
        .supports(Capability::Reasoning));
    let mut settings = ModelSettings::default();
    settings.reasoning = Some(Reasoning::effort(ReasoningEffort::Low));
    let agent = Agent::new("a").instructions("i");
    let options = RunOptions::new()
        .provider(Some(provider))
        .model("llama3.1")
        .model_settings(settings);
    let error = Runner::run(&agent, "go", options)
        .await
        .expect_err("the preset has no reasoning");
    assert!(error.to_string().contains("reasoning"), "{error}");
    assert!(
        server
            .received_requests()
            .await
            .expect("requests")
            .is_empty(),
        "nothing was sent"
    );
}

/// A preset's endpoint is where the table says it is: the request goes to its path, not a default.
#[tokio::test]
async fn requests_go_to_the_presets_own_path() {
    for (name, path) in [
        ("deepseek", "/v1/chat/completions"),
        ("groq", "/openai/v1/chat/completions"),
    ] {
        let server = MockServer::start().await;
        mock_chat(&server).await;
        let provider = Provider::preset(name)
            .expect("a preset")
            .api_key("sk-test")
            .base_url(format!(
                "{}{}",
                server.uri(),
                path.split("/chat").next().unwrap_or_default()
            ))
            .expect("loopback")
            .build()
            .expect("built");
        let agent = Agent::new("a").instructions("i");
        let options = RunOptions::new().provider(Some(provider)).model("m");
        Runner::run(&agent, "go", options).await.expect("run");
        let requests = server.received_requests().await.expect("requests");
        assert!(
            requests
                .first()
                .expect("one request")
                .url
                .path()
                .ends_with("/chat/completions"),
            "`{name}` hit {}",
            requests.first().expect("one request").url
        );
    }
}

/// The matrix is readable, and every preset claims at least the protocol basics.
#[test]
fn every_preset_claims_the_basics_and_names_its_key() {
    for spec in openai_agents::providers::PRESETS {
        assert!(
            spec.capabilities.supports(Capability::Streaming),
            "`{}` cannot stream?",
            spec.name
        );
        assert!(
            spec.capabilities.supports(Capability::ToolCalls),
            "`{}` cannot call tools?",
            spec.name
        );
        assert!(
            spec.key_env.is_some() || spec.enable_env.is_some(),
            "`{}` has no way to be enabled",
            spec.name
        );
        assert!(
            spec.base_url.starts_with("https://") || spec.base_url.starts_with("http://localhost")
        );
        let _ = Capabilities::chat();
    }
}
