//! Responses API model via async-openai config + HTTP.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::ModelError;
use crate::items::ModelResponse;
use crate::model::{Model, ModelRequest};

use super::{
    apply_model_settings_responses, decorate_request, map_transport, merge_usage,
    tools_as_responses, OpenAiEndpoint,
};

/// OpenAI Responses API model (Python: `OpenAIResponsesModel`).
pub struct OpenAIResponsesModel {
    endpoint: OpenAiEndpoint,
    model: String,
}

impl OpenAIResponsesModel {
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
impl Model for OpenAIResponsesModel {
    async fn get_response(&self, request: ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        let input_items = request.input.to_owned_items();
        let mut body = json!({
            "model": self.model,
            "input": input_items,
        });
        if let Some(instr) = request.system_instructions {
            body["instructions"] = json!(instr);
        }
        let tools = tools_as_responses(request.tools);
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }
        if let Some(prev) = request.previous_response_id {
            body["previous_response_id"] = json!(prev);
        }
        if let Some(conv) = request.conversation_id {
            body["conversation"] = json!(conv);
        }
        apply_model_settings_responses(&mut body, request.model_settings);
        // Python: structured output becomes `text.format = {type: json_schema, ...}`.
        if let Some(schema) = request.output_schema.filter(|s| !s.is_plain_text()) {
            body["text"] = json!({
                "format": {
                    "type": "json_schema",
                    "name": schema.name(),
                    "schema": schema.json_schema().map_err(map_transport)?,
                    "strict": schema.is_strict_json_schema(),
                }
            });
        }

        let resp = decorate_request(
            self.endpoint.http.post(self.endpoint.url("/responses")),
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
                "responses status={status} body={payload}"
            )));
        }

        responses_payload_to_model_response(payload)
    }
}

fn responses_payload_to_model_response(payload: Value) -> Result<ModelResponse, ModelError> {
    let output = payload
        .get("output")
        .and_then(|o| o.as_array())
        .cloned()
        .unwrap_or_default();

    let usage = payload.get("usage").map(|u| {
        let input = u
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output_toks = u
            .get("output_tokens")
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
