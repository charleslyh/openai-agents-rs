//! `CompatibleProvider` and the host-based default API of `OpenAIProvider` (D-013, D-I).
//!
//! This binary never calls `set_default_openai_api`, so the process-global stays unset and the
//! host-based default is what is under test.
#![cfg(feature = "openai")]

use std::sync::Arc;

use openai_agents::{
    Agent, CompatibleProvider, DefaultOpenAiApi, MultiProvider, OpenAIProvider, RunOptions, Runner,
};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn chat_reply(text: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "chatcmpl_1", "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    }))
}

fn responses_reply(text: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "resp_1", "object": "response",
        "output": [{"id": "1", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": text,
                                 "annotations": [], "logprobs": []}]}],
        "usage": {"input_tokens": 1, "output_tokens": 1}
    }))
}

async fn run_with(provider: Arc<dyn openai_agents::ModelProvider>, model: &str) -> String {
    let mut options = RunOptions::default();
    options.run_config.model_provider = Some(provider);
    Runner::run(&Agent::new("a").model_name(model), "go", options)
        .await
        .expect("run")
        .final_output_as_str()
        .expect("text")
        .to_string()
}

async fn authorization_headers(server: &MockServer) -> Vec<Option<String>> {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|r| r.headers.get("authorization").map(|v| v.to_str().unwrap().to_string()))
        .collect()
}

#[tokio::test]
async fn compatible_provider_speaks_chat_completions_without_a_key_by_default() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(chat_reply("via chat"))
        .expect(1)
        .mount(&server)
        .await;
    let provider = CompatibleProvider::new(format!("{}/v1", server.uri()));
    assert_eq!(run_with(Arc::new(provider), "m").await, "via chat");
    assert_eq!(authorization_headers(&server).await, vec![None], "no key, no Authorization");
}

#[tokio::test]
async fn compatible_provider_sends_the_key_and_can_choose_responses() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(responses_reply("via responses"))
        .expect(1)
        .mount(&server)
        .await;
    let provider = CompatibleProvider::new(format!("{}/v1", server.uri()))
        .api_key("secret")
        .api(DefaultOpenAiApi::Responses);
    assert_eq!(run_with(Arc::new(provider), "m").await, "via responses");
    assert_eq!(authorization_headers(&server).await, vec![Some("Bearer secret".to_string())]);
}

#[tokio::test]
async fn compatible_provider_is_routed_by_prefix_and_has_a_default_model() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(chat_reply("routed"))
        .expect(2)
        .mount(&server)
        .await;
    let base = format!("{}/v1", server.uri());
    let router = MultiProvider::new()
        .register("local", Arc::new(CompatibleProvider::new(&base).default_model("qwen")));
    let router: Arc<dyn openai_agents::ModelProvider> = Arc::new(router);
    assert_eq!(run_with(router.clone(), "local/qwen2.5").await, "routed");
    let sent: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(sent["model"], "qwen2.5", "the prefix is stripped");

    let provider = CompatibleProvider::new(&base).default_model("qwen");
    let model = openai_agents::ModelProvider::get_model(&provider, None).expect("default model");
    drop(model);
    assert_eq!(run_with(Arc::new(provider), "qwen").await, "routed");
}

/// D-I: an `OpenAIProvider` pointed at a server other than OpenAI defaults to Chat Completions.
#[tokio::test]
async fn openai_provider_defaults_to_chat_completions_for_other_hosts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(chat_reply("chat by host"))
        .expect(1)
        .mount(&server)
        .await;
    let provider = OpenAIProvider::new("sk", Some(&format!("{}/v1", server.uri())), None);
    assert_eq!(run_with(Arc::new(provider), "m").await, "chat by host");

    // An explicit provider-level choice still wins.
    let responses_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(responses_reply("explicit"))
        .expect(1)
        .mount(&responses_server)
        .await;
    let provider = OpenAIProvider::new("sk", Some(&format!("{}/v1", responses_server.uri())), None)
        .api(DefaultOpenAiApi::Responses);
    assert_eq!(run_with(Arc::new(provider), "m").await, "explicit");
}
