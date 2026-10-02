//! Chat Completions model via async-openai config + HTTP.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::ModelError;
use crate::items::{ItemHelpers, ModelResponse, ResponseOutputItem};
use crate::model::{Model, ModelRequest};
use crate::usage::Usage;

use super::{
    apply_model_settings_chat, map_transport, merge_usage, tools_as_chat, OpenAiEndpoint,
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
        let messages = input_to_chat_messages(request.system_instructions, &request.input);
        let mut body = json!({
            "model": self.model,
            "messages": messages,
        });
        let tools = tools_as_chat(request.tools);
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }
        apply_model_settings_chat(&mut body, request.model_settings);

        let resp = self
            .endpoint
            .http
            .post(self.endpoint.url("/chat/completions"))
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
