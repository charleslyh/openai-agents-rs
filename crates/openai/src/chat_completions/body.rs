//! Building the request body and reading the response of a Chat Completions call.

use serde_json::{json, Value};

use opai_agents_core::error::ModelError;
use opai_agents_core::items::{ModelResponse, ResponseOutputItem};
use opai_agents_core::model::ModelRequest;
use opai_agents_core::usage::Usage;

use super::super::chat_convert::{
    chat_message_to_output_items_with, items_to_chat_messages, ChatConvertOptions,
};
use super::super::{apply_model_settings_chat, tools_as_chat};

pub(super) fn build_chat_body(
    model: &str,
    options: &ChatConvertOptions,
    request: &ModelRequest<'_>,
) -> Result<Value, ModelError> {
    let messages = input_to_chat_messages(request.system_instructions, &request.input, options)?;
    let mut body = json!({
        "model": model,
        "messages": messages,
    });
    let tools = tools_as_chat(request.tools);
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
    }
    // Python: structured output becomes `response_format` for Chat Completions.
    if let Some(schema) = request.output_schema.filter(|s| !s.is_plain_text()) {
        let schema_value = schema
            .json_schema()
            .map_err(|e| ModelError::Behavior(format!("invalid output JSON schema: {e}")))?;
        body["response_format"] = json!({
            "type": "json_schema",
            "json_schema": {
                "name": "final_output",
                "schema": schema_value,
                "strict": schema.is_strict_json_schema(),
            }
        });
    }
    // Applied last, like Python: `extra_args` collides with anything already in the request
    // and `extra_body` overrides everything.
    apply_model_settings_chat(&mut body, request.model_settings)?;
    Ok(body)
}

fn input_to_chat_messages(
    system: Option<&str>,
    input: &opai_agents_core::model::ModelInput<'_>,
    options: &ChatConvertOptions,
) -> Result<Vec<Value>, ModelError> {
    let mut messages = Vec::new();
    if let Some(sys) = system {
        messages.push(json!({"role": "system", "content": sys}));
    }
    match input {
        opai_agents_core::model::ModelInput::Text(t) => {
            messages.push(json!({"role": "user", "content": t}));
        }
        opai_agents_core::model::ModelInput::Items(items) => {
            messages.extend(items_to_chat_messages(items, options)?);
        }
    }
    Ok(messages)
}

pub(super) fn chat_payload_to_model_response(
    payload: Value,
    preserve_raw_usage: bool,
    model: &str,
) -> Result<ModelResponse, ModelError> {
    let choice = payload
        .pointer("/choices/0")
        .ok_or_else(|| ModelError::Behavior("missing choices[0]".into()))?;
    let mut message = choice
        .get("message")
        .filter(|m| m.is_object())
        .cloned()
        .ok_or_else(|| ModelError::Behavior("missing choices[0].message".into()))?;

    let has_output = |m: &Value| {
        let non_empty = |key: &str| match m.get(key) {
            Some(Value::String(s)) => !s.is_empty(),
            Some(Value::Array(a)) => !a.is_empty(),
            _ => false,
        };
        non_empty("content") || non_empty("refusal") || non_empty("tool_calls")
    };
    // Python: a completion with nothing in it is explained by its finish reason. Filtered output
    // becomes a refusal the caller can handle; a completion cut off before any visible token is a
    // budget problem, not a refusal.
    match choice.get("finish_reason").and_then(Value::as_str) {
        Some("content_filter") if !has_output(&message) => {
            message["refusal"] = json!("Response withheld by the provider's content filter.");
        }
        Some("length") if !has_output(&message) => {
            return Err(ModelError::Behavior(
                "Chat Completions response terminated with finish_reason='length' but produced \
                 no assistant text, tool call, or refusal."
                    .into(),
            ));
        }
        _ => {}
    }
    // Python: every output item records the model (and response) that produced it.
    let mut provider_data = json!({"model": model});
    if let Some(id) = payload
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
    {
        provider_data["response_id"] = id.into();
    }
    let output: Vec<ResponseOutputItem> =
        chat_message_to_output_items_with(&message, Some(&provider_data));

    let usage = chat_usage_or_completed_request(payload.get("usage"));
    let raw_usage = payload
        .get("usage")
        .filter(|u| preserve_raw_usage && u.is_object())
        .cloned();

    Ok(ModelResponse {
        output,
        usage,
        response_id: payload
            .get("id")
            .and_then(|i| i.as_str())
            .map(str::to_string),
        request_id: None,
        raw_usage,
    })
}

/// Python (`openai_chatcompletions.py`): the request counts even when the provider omits usage.
fn chat_usage_or_completed_request(usage: Option<&Value>) -> Usage {
    match usage.filter(|u| !u.is_null()) {
        Some(u) => Usage::from_chat_usage(u),
        None => Usage {
            requests: 1,
            ..Usage::default()
        },
    }
}
