//! OpenAI wiremock contract tests (verification layer 2).
#![cfg(feature = "openai")]

use std::sync::Arc;

use openai_agents::{
    Agent, FunctionTool, OpenAIChatCompletionsModel, OpenAIResponsesModel, RunOptions, Runner,
};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

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
