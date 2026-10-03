//! Chat Completions model via async-openai config + HTTP.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::ModelError;
use crate::items::{ItemHelpers, ModelResponse, ResponseOutputItem};
use crate::model::{Model, ModelRequest};
use crate::usage::Usage;

use super::{
    apply_model_settings_chat, decorate_request, map_transport, merge_usage, tools_as_chat,
    OpenAiEndpoint,
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
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        if content_type.contains("application/json") {
            let payload: Value = resp.json().await.map_err(map_transport)?;
            let response = chat_payload_to_model_response(payload)?;
            for item in &response.output {
                if let Some(text) = crate::items::extract_message_text(item) {
                    let _ = raw_tx
                        .send(json!({
                            "type": "output_text.delta",
                            "delta": text,
                        }))
                        .await;
                }
            }
            let _ = raw_tx
                .send(json!({
                    "type": "response.completed",
                    "response_id": response.response_id,
                    "output": response.output,
                }))
                .await;
            return Ok(response);
        }

        use futures::StreamExt;
        let mut byte_stream = resp.bytes_stream();
        let mut buffer = String::new();
        let mut response_id: Option<String> = None;
        let mut text = String::new();
        // index -> (id, name, arguments)
        let mut tool_calls: std::collections::BTreeMap<usize, (String, String, String)> =
            std::collections::BTreeMap::new();
        let mut usage: Option<(u64, u64)> = None;

        while let Some(chunk) = byte_stream.next().await {
            let chunk = chunk.map_err(map_transport)?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = buffer.find('\n') {
                let mut line = buffer[..pos].to_string();
                buffer.drain(..=pos);
                if line.ends_with('\r') {
                    line.pop();
                }
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data == "[DONE]" {
                    continue;
                }
                let payload: Value = serde_json::from_str(data).map_err(|e| {
                    ModelError::Behavior(format!("invalid SSE JSON: {e}; data={data}"))
                })?;

                if response_id.is_none() {
                    response_id = payload
                        .get("id")
                        .and_then(|i| i.as_str())
                        .map(str::to_string);
                }

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

                // DeepSeek / some gateways stream chain-of-thought as
                // `reasoning_content` before visible `content`.
                if let Some(reasoning) = delta.get("reasoning_content").and_then(|c| c.as_str()) {
                    if !reasoning.is_empty() {
                        let _ = raw_tx
                            .send(json!({
                                "type": "reasoning_text.delta",
                                "delta": reasoning,
                            }))
                            .await;
                    }
                }

                if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                    if !content.is_empty() {
                        text.push_str(content);
                        let _ = raw_tx
                            .send(json!({
                                "type": "output_text.delta",
                                "delta": content,
                            }))
                            .await;
                    }
                }

                if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                    for tc in tcs {
                        let idx = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                        let entry = tool_calls.entry(idx).or_insert_with(|| {
                            (
                                String::new(),
                                String::new(),
                                String::new(),
                            )
                        });
                        if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                            if !id.is_empty() {
                                entry.0 = id.to_string();
                            }
                        }
                        if let Some(name) = tc.pointer("/function/name").and_then(|n| n.as_str()) {
                            if !name.is_empty() {
                                entry.1.push_str(name);
                            }
                        }
                        if let Some(args) =
                            tc.pointer("/function/arguments").and_then(|a| a.as_str())
                        {
                            entry.2.push_str(args);
                        }
                    }
                }
            }
        }

        let mut output: Vec<ResponseOutputItem> = Vec::new();
        for (id, name, arguments) in tool_calls.into_values() {
            let id = if id.is_empty() { "call".to_string() } else { id };
            output.push(ItemHelpers::function_tool_call(name, arguments, id));
        }
        if !text.is_empty() {
            output.push(ItemHelpers::text_message(text));
        }

        let response = ModelResponse {
            output,
            usage: merge_usage(usage),
            response_id,
            request_id: None,
        };
        let _ = raw_tx
            .send(json!({
                "type": "response.completed",
                "response_id": response.response_id,
                "output": response.output,
            }))
            .await;
        Ok(response)
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
