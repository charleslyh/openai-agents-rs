//! Responses API model via async-openai config + HTTP.

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::usage::Usage;
use crate::error::ModelError;
use crate::items::{ModelResponse, ResponseOutputItem};
use crate::model::wire_events::{
    emit_completed, emit_response_stream, now_seconds, response_object, FAKE_RESPONSES_ID,
};
use crate::model::{Model, ModelRequest};

use super::{
    apply_model_settings_responses, decorate_request, is_json_response, map_reqwest, map_transport, status_error,
    tools_as_responses, OpenAiEndpoint, SseReader,
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
    fn get_retry_advice(
        &self,
        request: &crate::retry::ModelRetryAdviceRequest,
    ) -> Option<crate::retry::ModelRetryAdvice> {
        crate::retry::openai_retry_advice(request)
    }

    async fn get_response(&self, request: ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        let body = build_responses_body(&self.model, &request)?;

        let resp = decorate_request(
            self.endpoint.http.post(self.endpoint.url("/responses")),
            request.model_settings,
        )
        .bearer_auth(self.endpoint.api_key())
        .json(&body)
        .send()
        .await
        .map_err(map_reqwest)?;

        if !resp.status().is_success() {
            return Err(status_error("responses", resp).await);
        }
        let payload: Value = resp.json().await.map_err(map_reqwest)?;

        responses_payload_to_model_response(payload)
    }

    /// Streams the Responses API as SSE (D-011).
    ///
    /// Wire events are forwarded verbatim (Python: `openai_responses.py` does `yield chunk`), so
    /// consumers see `response.output_text.delta`, `response.output_item.*` and so on. The
    /// assembled `output` comes from `response.completed` when the gateway sends it, otherwise
    /// from the `response.output_item.done` events seen.
    async fn stream_response(
        &self,
        request: ModelRequest<'_>,
        raw_tx: tokio::sync::mpsc::Sender<Value>,
    ) -> Result<ModelResponse, ModelError> {
        let mut body = build_responses_body(&self.model, &request)?;
        body["stream"] = json!(true);

        let resp = decorate_request(
            self.endpoint.http.post(self.endpoint.url("/responses")),
            request.model_settings,
        )
        .bearer_auth(self.endpoint.api_key())
        .json(&body)
        .send()
        .await
        .map_err(map_reqwest)?;

        if !resp.status().is_success() {
            return Err(status_error("responses stream", resp).await);
        }

        // Wiremock / non-SSE providers may still return a full JSON body.
        if is_json_response(&resp) {
            let payload: Value = resp.json().await.map_err(map_reqwest)?;
            let response = responses_payload_to_model_response(payload)?;
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
        let mut response_id: Option<String> = None;
        // output_index -> item; only used when the stream ends without `response.completed`.
        let mut items: BTreeMap<usize, Value> = BTreeMap::new();
        let mut completed: Option<Value> = None;
        // Sequence number of the last forwarded event, so a synthesized terminal event can
        // continue the numbering instead of restarting it.
        let mut last_sequence: Option<u64> = None;

        while let Some(chunk) = byte_stream.next().await {
            let chunk = chunk.map_err(map_reqwest)?;
            reader.feed(&chunk);
            while let Some(event) = reader.next_event() {
                let event = event?;
                // Python forwards every wire event, including failure events.
                if let Some(sequence) = event.get("sequence_number").and_then(|s| s.as_u64()) {
                    last_sequence = Some(sequence);
                }
                let _ = raw_tx.send(event.clone()).await;
                match event.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                    // `response.created` carries the id before any content arrives.
                    "response.created" | "response.queued" | "response.in_progress"
                        if response_id.is_none() =>
                    {
                        response_id = event
                            .pointer("/response/id")
                            .and_then(|i| i.as_str())
                            .map(str::to_string);
                    }
                    "response.output_item.done" => {
                        if let Some(item) = event.get("item") {
                            let index = event
                                .get("output_index")
                                .and_then(|i| i.as_u64())
                                .unwrap_or(items.len() as u64)
                                as usize;
                            items.insert(index, item.clone());
                        }
                    }
                    "response.completed" | "response.incomplete" => {
                        completed = event.get("response").cloned();
                    }
                    "response.failed" | "error" => {
                        return Err(ModelError::Behavior(format!(
                            "responses stream {}: {}",
                            failure_kind(&event),
                            failure_detail(&event)
                        )));
                    }
                    _ => {}
                }
            }
        }

        let response = match completed {
            Some(payload) => {
                let mut response = responses_payload_to_model_response(payload)?;
                if response.response_id.is_none() {
                    response.response_id = response_id;
                }
                response
            }
            None => {
                // The gateway never sent `response.completed`: synthesize the terminal event so
                // consumers still see one, continuing the sequence numbering.
                let output: Vec<ResponseOutputItem> = items.into_values().collect();
                let id = response_id
                    .clone()
                    .unwrap_or_else(|| FAKE_RESPONSES_ID.to_string());
                let wire = response_object(
                    &id,
                    &self.model,
                    "completed",
                    now_seconds(),
                    &output,
                    &Usage::default(),
                    "auto",
                );
                emit_completed(&raw_tx, wire, last_sequence.map_or(0, |s| s + 1)).await;
                ModelResponse {
                    output,
                    usage: Usage::default(),
                    response_id,
                    request_id: None,
                }
            }
        };
        Ok(response)
    }
}

fn build_responses_body(model: &str, request: &ModelRequest<'_>) -> Result<Value, ModelError> {
    let input_items = request.input.to_owned_items();
    let mut body = json!({
        "model": model,
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
    // Python: structured output becomes `text.format = {type: json_schema, ...}`.
    if let Some(schema) = request.output_schema.filter(|s| !s.is_plain_text()) {
        body["text"] = json!({
            "format": {
                "type": "json_schema",
                "name": "final_output",
                "schema": schema.json_schema().map_err(map_transport)?,
                "strict": schema.is_strict_json_schema(),
            }
        });
    }
    // Applied last, like Python: `extra_args` collides with anything already in the request
    // and `extra_body` overrides everything.
    apply_model_settings_responses(&mut body, request.model_settings)?;
    Ok(body)
}

fn failure_kind(event: &Value) -> &str {
    match event.get("type").and_then(|t| t.as_str()) {
        Some("error") => "error",
        _ => "failed",
    }
}

fn failure_detail(event: &Value) -> String {
    event
        .pointer("/response/error/message")
        .or_else(|| event.get("message"))
        .and_then(|m| m.as_str())
        .unwrap_or("unknown error")
        .to_string()
}

fn responses_payload_to_model_response(payload: Value) -> Result<ModelResponse, ModelError> {
    let output = payload
        .get("output")
        .and_then(|o| o.as_array())
        .cloned()
        .unwrap_or_default();

    // Python: a response without `usage` represents zero requests.
    let usage = payload
        .get("usage")
        .map(Usage::from_responses_usage)
        .unwrap_or_default();

    Ok(ModelResponse {
        output,
        usage,
        response_id: payload
            .get("id")
            .and_then(|i| i.as_str())
            .map(str::to_string),
        request_id: None,
    })
}
