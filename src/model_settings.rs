//! Model settings (Python: `agents.model_settings.ModelSettings`).
//!
//! The frequently used subset of Python's 25 fields is supported. `tool_choice`,
//! `truncation` and `verbosity` are typed enums because Rust callers get no benefit from
//! untyped strings; every other setting maps one-to-one onto the provider request.

use serde::de::Deserializer;
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// How the model is forced (or not) to call tools (Python: `ModelSettings.tool_choice`).
#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    /// Let the model decide (`"auto"`).
    Auto,
    /// The model must call a tool (`"required"`).
    Required,
    /// The model must not call a tool (`"none"`).
    None,
    /// Force a specific function tool by name.
    Function(String),
    /// Provider-specific object, e.g. an MCP tool choice.
    Other(Value),
}

impl ToolChoice {
    /// Wire representation for the Chat Completions / Responses request body.
    pub fn to_request_value(&self) -> Value {
        match self {
            Self::Auto => Value::String("auto".into()),
            Self::Required => Value::String("required".into()),
            Self::None => Value::String("none".into()),
            Self::Function(name) => {
                serde_json::json!({"type": "function", "function": {"name": name}})
            }
            Self::Other(v) => v.clone(),
        }
    }
}

impl From<&str> for ToolChoice {
    fn from(value: &str) -> Self {
        match value {
            "auto" => Self::Auto,
            "required" => Self::Required,
            "none" => Self::None,
            other => Self::Function(other.to_string()),
        }
    }
}

impl From<String> for ToolChoice {
    fn from(value: String) -> Self {
        value.as_str().into()
    }
}

impl Serialize for ToolChoice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_request_value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        Ok(match value {
            Value::String(s) => s.as_str().into(),
            Value::Object(map) => match map.get("function").and_then(|f| f.get("name")) {
                Some(Value::String(name)) => Self::Function(name.clone()),
                _ => Self::Other(Value::Object(map)),
            },
            other => Self::Other(other),
        })
    }
}

/// Context truncation strategy (Python: `ModelSettings.truncation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Truncation {
    /// Let the provider truncate when needed.
    Auto,
    /// Never truncate; error instead.
    Disabled,
}

/// Verbosity constraint (Python: `ModelSettings.verbosity`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verbosity {
    /// Terse answers.
    Low,
    /// Default verbosity.
    Medium,
    /// Detailed answers.
    High,
}

/// Tunable parameters forwarded to the model provider.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelSettings {
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling.
    pub top_p: Option<f32>,
    /// Penalizes tokens by frequency.
    pub frequency_penalty: Option<f32>,
    /// Penalizes tokens by presence.
    pub presence_penalty: Option<f32>,
    /// Tool choice hint.
    pub tool_choice: Option<ToolChoice>,
    /// Whether the model may emit several tool calls in one turn.
    pub parallel_tool_calls: Option<bool>,
    /// Context truncation strategy.
    pub truncation: Option<Truncation>,
    /// Maximum output tokens.
    pub max_tokens: Option<u32>,
    /// Reasoning configuration (provider-specific object).
    pub reasoning: Option<Value>,
    /// Verbosity constraint.
    pub verbosity: Option<Verbosity>,
    /// Metadata attached to the request.
    pub metadata: Option<Map<String, Value>>,
    /// Whether the provider should store the response.
    pub store: Option<bool>,
    /// Number of top logprobs to return.
    pub top_logprobs: Option<u32>,
    /// Ask streaming Chat Completions to include a usage chunk.
    pub include_usage: Option<bool>,
    /// Extra output fields to include (Responses API `include`).
    pub response_include: Option<Vec<String>>,
    /// Extra JSON fields merged into the provider request.
    pub extra_body: Option<Value>,
    /// Extra HTTP headers for the provider request.
    pub extra_headers: Option<Map<String, Value>>,
    /// Arbitrary keyword arguments forwarded to the provider.
    pub extra_args: Option<Map<String, Value>>,
    /// Per-attempt timeout in seconds.
    pub timeout: Option<f32>,
}

impl ModelSettings {
    /// Produce a new instance by overlaying non-`None` values from `override`.
    ///
    /// Equivalent to Python's `ModelSettings.resolve(override)`: `extra_body`,
    /// `extra_headers` and `extra_args` dictionaries are merged rather than replaced, and
    /// `tool_choice` may be cleared back to `None` by the reset logic in the runner.
    pub fn resolve(&self, override_settings: Option<&ModelSettings>) -> ModelSettings {
        let Some(o) = override_settings else {
            return self.clone();
        };
        let mut out = self.clone();
        macro_rules! overlay {
            ($($field:ident),* $(,)?) => {
                $(if o.$field.is_some() {
                    out.$field = o.$field.clone();
                })*
            };
        }
        overlay!(
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            tool_choice,
            parallel_tool_calls,
            truncation,
            max_tokens,
            reasoning,
            verbosity,
            store,
            top_logprobs,
            include_usage,
            response_include,
            timeout,
        );
        out.metadata = merge_maps(self.metadata.as_ref(), o.metadata.as_ref());
        out.extra_body = merge_values(self.extra_body.as_ref(), o.extra_body.as_ref());
        out.extra_headers = merge_maps(self.extra_headers.as_ref(), o.extra_headers.as_ref());
        out.extra_args = merge_maps(self.extra_args.as_ref(), o.extra_args.as_ref());
        out
    }
}

fn merge_maps(base: Option<&Map<String, Value>>, overlay: Option<&Map<String, Value>>) -> Option<Map<String, Value>> {
    match (base, overlay) {
        (None, None) => None,
        (Some(b), None) => Some(b.clone()),
        (None, Some(o)) => Some(o.clone()),
        (Some(b), Some(o)) => {
            let mut merged = b.clone();
            for (k, v) in o {
                merged.insert(k.clone(), v.clone());
            }
            Some(merged)
        }
    }
}

fn merge_values(base: Option<&Value>, overlay: Option<&Value>) -> Option<Value> {
    match (base, overlay) {
        (None, None) => None,
        (Some(b), None) => Some(b.clone()),
        (None, Some(o)) => Some(o.clone()),
        (Some(Value::Object(b)), Some(Value::Object(o))) => {
            let mut merged = b.clone();
            for (k, v) in o {
                merged.insert(k.clone(), v.clone());
            }
            Some(Value::Object(merged))
        }
        (_, Some(o)) => Some(o.clone()),
    }
}

/// Default model settings used when an agent does not override them.
pub fn get_default_model_settings() -> ModelSettings {
    ModelSettings::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, Value)]) -> Map<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn resolve_overlays_non_none_values() {
        let base = ModelSettings {
            temperature: Some(0.1),
            max_tokens: Some(10),
            ..Default::default()
        };
        let overlay = ModelSettings {
            temperature: Some(0.9),
            ..Default::default()
        };
        let merged = base.resolve(Some(&overlay));
        assert_eq!(merged.temperature, Some(0.9));
        assert_eq!(merged.max_tokens, Some(10));
    }

    #[test]
    fn resolve_merges_dictionaries() {
        let base = ModelSettings {
            extra_args: Some(map(&[("a", Value::from(1))])),
            extra_headers: Some(map(&[("x", Value::from("1"))])),
            ..Default::default()
        };
        let overlay = ModelSettings {
            extra_args: Some(map(&[("b", Value::from(2))])),
            ..Default::default()
        };
        let merged = base.resolve(Some(&overlay));
        assert_eq!(merged.extra_args.as_ref().unwrap().len(), 2);
        assert_eq!(merged.extra_headers.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn tool_choice_round_trips_through_json() {
        for choice in [
            ToolChoice::Auto,
            ToolChoice::Required,
            ToolChoice::None,
            ToolChoice::Function("lookup".into()),
            ToolChoice::Other(serde_json::json!({"type": "mcp"})),
        ] {
            let value = serde_json::to_value(&choice).expect("serialize");
            let back: ToolChoice = serde_json::from_value(value).expect("deserialize");
            assert_eq!(back, choice);
        }
        assert_eq!(
            ToolChoice::Function("x".into()).to_request_value(),
            serde_json::json!({"type": "function", "function": {"name": "x"}})
        );
        assert_eq!(ToolChoice::from("none"), ToolChoice::None);
        assert_eq!(ToolChoice::from("other"), ToolChoice::Function("other".into()));
    }
}
