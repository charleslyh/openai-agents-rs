//! Turning `ModelSettings` and tools into request bodies and headers.

use crate::error::ModelError;
use crate::model_settings::ModelSettings;
use crate::tool::FunctionTool;

/// Settings shared by both OpenAI APIs.
pub(crate) fn apply_model_settings_common(body: &mut serde_json::Value, settings: &ModelSettings) {
    if let Some(t) = settings.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    if let Some(p) = settings.top_p {
        body["top_p"] = serde_json::json!(p);
    }
    if let Some(f) = settings.frequency_penalty {
        body["frequency_penalty"] = serde_json::json!(f);
    }
    if let Some(f) = settings.presence_penalty {
        body["presence_penalty"] = serde_json::json!(f);
    }
    if let Some(tc) = &settings.tool_choice {
        body["tool_choice"] = tc.to_request_value();
    }
    if let Some(p) = settings.parallel_tool_calls {
        body["parallel_tool_calls"] = serde_json::json!(p);
    }
    // `extra_args` / `extra_body` are applied by the callers, after every mapped field, so that
    // `extra_body` keeps the highest precedence (openai SDK `_merge_mappings`).
}

/// Apply `ModelSettings.extra_args`, filling only keys nothing else has set.
///
/// Python merges `extra_args` into the API call kwargs and raises `TypeError` when a key is
/// already provided (`openai_chatcompletions.py:725`, `openai_responses.py:1059`). Rust has no
/// kwargs layer, so the same collision is reported as [`ModelError::Behavior`] with Python's
/// message. `extra_body` is applied afterwards and still wins.
pub(crate) fn apply_extra_args(
    body: &mut serde_json::Value,
    settings: &ModelSettings,
    api: &str,
) -> Result<(), ModelError> {
    let Some(extra) = settings.extra_args.as_ref() else {
        return Ok(());
    };
    let mut duplicates: Vec<&str> = extra
        .keys()
        .filter(|key| body.get(key.as_str()).is_some())
        .map(String::as_str)
        .collect();
    if !duplicates.is_empty() {
        duplicates.sort_unstable();
        let keys = duplicates
            .iter()
            .map(|k| format!("'{k}'"))
            .collect::<Vec<_>>()
            .join(", ");
        let noun = if duplicates.len() == 1 {
            "keyword argument"
        } else {
            "keyword arguments"
        };
        return Err(ModelError::Behavior(format!(
            "extra_args: {api}.create() got multiple values for {noun} {keys}"
        )));
    }
    if let Some(base) = body.as_object_mut() {
        for (key, value) in extra {
            base.insert(key.clone(), value.clone());
        }
    }
    Ok(())
}

/// Apply `ModelSettings.extra_body`, overriding whatever the request already holds.
///
/// Python passes this as the OpenAI SDK's nested `extra_body` argument (`docs/models/index.md`:
/// "remains a nested `extra_body` argument"), which the SDK merges over the request body, so it
/// has the highest precedence — above both mapped settings and [`apply_extra_args`].
pub(crate) fn apply_extra_body(body: &mut serde_json::Value, settings: &ModelSettings) {
    if let (Some(base), Some(serde_json::Value::Object(extra))) =
        (body.as_object_mut(), settings.extra_body.as_ref())
    {
        for (key, value) in extra {
            base.insert(key.clone(), value.clone());
        }
    }
}

/// Chat Completions mapping (Python: `OpenAIChatCompletionsModel`).
pub(crate) fn apply_model_settings_chat(
    body: &mut serde_json::Value,
    settings: &ModelSettings,
) -> Result<(), ModelError> {
    apply_model_settings_common(body, settings);
    if let Some(m) = settings.max_tokens {
        body["max_tokens"] = serde_json::json!(m);
    }
    if let Some(t) = settings.top_logprobs {
        body["top_logprobs"] = serde_json::json!(t);
    }
    if let Some(s) = settings.store {
        body["store"] = serde_json::json!(s);
    }
    if let Some(meta) = &settings.metadata {
        body["metadata"] = serde_json::Value::Object(meta.clone());
    }
    apply_extra_args(body, settings, "chat.completions")?;
    apply_extra_body(body, settings);
    Ok(())
}

/// Responses API mapping (Python: `OpenAIResponsesModel`).
pub(crate) fn apply_model_settings_responses(
    body: &mut serde_json::Value,
    settings: &ModelSettings,
) -> Result<(), ModelError> {
    apply_model_settings_common(body, settings);
    // Responses API names this `max_output_tokens`.
    if let Some(m) = settings.max_tokens {
        body["max_output_tokens"] = serde_json::json!(m);
    }
    if let Some(t) = settings.truncation {
        body["truncation"] = serde_json::json!(match t {
            crate::model_settings::Truncation::Auto => "auto",
            crate::model_settings::Truncation::Disabled => "disabled",
        });
    }
    if let Some(r) = &settings.reasoning {
        body["reasoning"] = r.clone();
    }
    if let Some(v) = settings.verbosity {
        body["verbosity"] = serde_json::json!(match v {
            crate::model_settings::Verbosity::Low => "low",
            crate::model_settings::Verbosity::Medium => "medium",
            crate::model_settings::Verbosity::High => "high",
        });
    }
    if let Some(meta) = &settings.metadata {
        body["metadata"] = serde_json::Value::Object(meta.clone());
    }
    if let Some(s) = settings.store {
        body["store"] = serde_json::json!(s);
    }
    if let Some(t) = settings.top_logprobs {
        body["top_logprobs"] = serde_json::json!(t);
    }
    if let Some(inc) = &settings.response_include {
        body["include"] = serde_json::json!(inc);
    }
    apply_extra_args(body, settings, "responses")?;
    apply_extra_body(body, settings);
    Ok(())
}

/// Apply `timeout` / `extra_headers` to an outgoing request.
pub(crate) fn decorate_request(
    request: reqwest::RequestBuilder,
    settings: &ModelSettings,
) -> reqwest::RequestBuilder {
    let mut request = request;
    if let Some(t) = settings.timeout {
        if t > 0.0 {
            request = request.timeout(std::time::Duration::from_secs_f32(t));
        }
    }
    if let Some(headers) = &settings.extra_headers {
        for (k, v) in headers {
            if let Some(v) = v.as_str() {
                request = request.header(k.as_str(), v);
            }
        }
    }
    if let Some(query) = &settings.extra_query {
        let pairs = query_pairs(query);
        if !pairs.is_empty() {
            request = request.query(&pairs);
        }
    }
    request
}

/// Flatten `ModelSettings.extra_query` into `key=value` pairs: scalars as text, arrays as
/// repeated keys, `null` skipped, objects as compact JSON.
fn query_pairs(query: &serde_json::Map<String, serde_json::Value>) -> Vec<(String, String)> {
    use serde_json::Value;
    fn text(value: &Value) -> Option<String> {
        match value {
            Value::Null => None,
            Value::String(s) => Some(s.clone()),
            Value::Bool(b) => Some(b.to_string()),
            Value::Number(n) => Some(n.to_string()),
            other => Some(other.to_string()),
        }
    }
    let mut pairs = Vec::new();
    for (key, value) in query {
        match value {
            Value::Array(items) => {
                pairs.extend(items.iter().filter_map(text).map(|v| (key.clone(), v)));
            }
            other => pairs.extend(text(other).map(|v| (key.clone(), v))),
        }
    }
    pairs
}

pub(crate) fn tools_as_chat(tools: &[FunctionTool]) -> Vec<serde_json::Value> {
    tools.iter().map(|t| t.to_chat_tool()).collect()
}

pub(crate) fn tools_as_responses(tools: &[FunctionTool]) -> Vec<serde_json::Value> {
    tools.iter().map(|t| t.to_function_tool_param()).collect()
}
