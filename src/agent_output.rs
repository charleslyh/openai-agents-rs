//! Structured output schemas (Python: `agents.agent_output`).
//!
//! An agent may declare an `output_type`; the SDK then advertises a JSON Schema to the model
//! and validates the response before it becomes the run's `final_output`.

use std::any::{Any, TypeId};
use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::{AgentsError, ModelError, UserError};
use crate::strict_schema::ensure_strict_json_schema;

/// Key used when the target type cannot be represented as a JSON Schema object
/// (Python: `_WRAPPER_DICT_KEY`).
pub const WRAPPER_DICT_KEY: &str = "response";

/// Validates a parsed JSON value against the declared output type.
pub type OutputValidator = Arc<dyn Fn(&Value) -> Result<(), String> + Send + Sync>;

/// Captures the JSON schema of an output type and validates JSON produced by the model.
///
/// Python: `agents.agent_output.AgentOutputSchemaBase`.
pub trait AgentOutputSchemaBase: Send + Sync + std::fmt::Debug {
    /// Whether the output type is plain text rather than a JSON object.
    fn is_plain_text(&self) -> bool;
    /// Name of the output type, sent to the provider as the schema name.
    fn name(&self) -> &str;
    /// JSON Schema for the output. Errors when the type is plain text.
    fn json_schema(&self) -> Result<&Value, AgentsError>;
    /// Whether the schema was normalized into OpenAI's strict form.
    fn is_strict_json_schema(&self) -> bool;
    /// Validate a JSON string against the output type.
    ///
    /// Returns the validated value, or a [`ModelError::Behavior`] when the model produced
    /// something that does not match (Python raises `ModelBehaviorError`).
    fn validate_json(&self, json_str: &str) -> Result<Value, ModelError>;
}

/// Output schema derived from a Rust type (Python: `AgentOutputSchema(output_type, ...)`).
#[derive(Clone)]
pub struct AgentOutputSchema {
    name: String,
    schema: Value,
    strict: bool,
    wrapped: bool,
    plain_text: bool,
    validator: OutputValidator,
}

impl std::fmt::Debug for AgentOutputSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentOutputSchema")
            .field("name", &self.name)
            .field("strict", &self.strict)
            .field("wrapped", &self.wrapped)
            .field("plain_text", &self.plain_text)
            .finish()
    }
}

impl AgentOutputSchema {
    /// Build a schema for `T`, normalized to OpenAI strict mode when `strict` is set.
    ///
    /// `T = String` mirrors Python's `output_type=str` and is treated as plain text.
    /// Types whose schema root is not an object (for example `Vec<T>`) are wrapped in
    /// `{"response": ...}`, matching Python's `_WRAPPER_DICT_KEY` behaviour.
    pub fn of<T>() -> Result<Self, AgentsError>
    where
        T: Any + DeserializeOwned + crate::schemars::JsonSchema + Send + Sync + 'static,
    {
        Self::of_with::<T>(true)
    }

    /// Like [`Self::of`] but allows opting out of strict mode
    /// (Python: `AgentOutputSchema(T, strict_json_schema=False)`).
    pub fn of_with<T>(strict: bool) -> Result<Self, AgentsError>
    where
        T: Any + DeserializeOwned + crate::schemars::JsonSchema + Send + Sync + 'static,
    {
        let name = std::any::type_name::<T>().to_string();
        let plain_text = TypeId::of::<T>() == TypeId::of::<String>();
        if plain_text {
            return Ok(Self {
                name,
                schema: Value::Null,
                strict,
                wrapped: false,
                plain_text: true,
                validator: Arc::new(|_| Ok(())),
            });
        }

        let raw = serde_json::to_value(crate::schemars::schema_for!(T))
            .map_err(|e| UserError::new(format!("could not build a JSON schema for `{name}`: {e}")))?;

        let is_object = raw.get("type") == Some(&Value::String("object".into()));
        let (schema, wrapped) = if is_object {
            (raw, false)
        } else {
            (
                serde_json::json!({
                    "type": "object",
                    "properties": { WRAPPER_DICT_KEY: raw },
                    "required": [WRAPPER_DICT_KEY],
                    "additionalProperties": false,
                }),
                true,
            )
        };

        let schema = if strict {
            ensure_strict_json_schema(&schema).map_err(|e| {
                UserError::new(format!(
                    "Strict JSON schema is enabled, but `{name}` is not valid. Either make the \
                     output type strict, or build it with `of_with::<T>(false)` ({e})"
                ))
            })?
        } else {
            schema
        };

        Ok(Self {
            name,
            schema,
            strict,
            wrapped,
            plain_text: false,
            validator: Arc::new(|value: &Value| {
                serde_json::from_value::<T>(value.clone())
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            }),
        })
    }
}

impl AgentOutputSchemaBase for AgentOutputSchema {
    fn is_plain_text(&self) -> bool {
        self.plain_text
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn json_schema(&self) -> Result<&Value, AgentsError> {
        if self.plain_text {
            return Err(UserError::new("Output type is plain text, so no JSON schema is available").into());
        }
        Ok(&self.schema)
    }

    fn is_strict_json_schema(&self) -> bool {
        self.strict
    }

    fn validate_json(&self, json_str: &str) -> Result<Value, ModelError> {
        if self.plain_text {
            return Ok(Value::String(json_str.to_string()));
        }
        let parsed: Value = serde_json::from_str(json_str).map_err(|e| {
            ModelError::Behavior(format!("Model returned invalid JSON for the output type: {e}"))
        })?;

        if self.wrapped {
            let object = parsed.as_object().ok_or_else(|| {
                ModelError::Behavior(format!(
                    "Expected an object with a `{WRAPPER_DICT_KEY}` key for JSON: {json_str}"
                ))
            })?;
            let inner = object.get(WRAPPER_DICT_KEY).ok_or_else(|| {
                ModelError::Behavior(format!(
                    "Could not find key `{WRAPPER_DICT_KEY}` in JSON: {json_str}"
                ))
            })?;
            return self.validate_value(inner, json_str);
        }

        self.validate_value(&parsed, json_str)
    }
}

impl AgentOutputSchema {
    fn validate_value(&self, value: &Value, json_str: &str) -> Result<Value, ModelError> {
        (self.validator)(value).map_err(|e| {
            ModelError::Behavior(format!(
                "Model output did not match the declared output type: {e} (JSON: {json_str})"
            ))
        })?;
        Ok(value.clone())
    }
}

/// A user-supplied schema that bypasses type-driven schema generation.
///
/// Useful when the JSON Schema is authored elsewhere
/// (Python: subclassing `AgentOutputSchemaBase`).
pub struct CustomOutputSchema {
    name: String,
    schema: Value,
    strict: bool,
}

impl std::fmt::Debug for CustomOutputSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomOutputSchema")
            .field("name", &self.name)
            .field("strict", &self.strict)
            .finish()
    }
}

impl CustomOutputSchema {
    /// Create from a ready-made JSON Schema.
    pub fn new(name: impl Into<String>, schema: Value, strict: bool) -> Self {
        Self {
            name: name.into(),
            schema,
            strict,
        }
    }
}

impl AgentOutputSchemaBase for CustomOutputSchema {
    fn is_plain_text(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn json_schema(&self) -> Result<&Value, AgentsError> {
        Ok(&self.schema)
    }

    fn is_strict_json_schema(&self) -> bool {
        self.strict
    }

    fn validate_json(&self, json_str: &str) -> Result<Value, ModelError> {
        serde_json::from_str::<Value>(json_str).map_err(|e| {
            ModelError::Behavior(format!("Model returned invalid JSON for the output type: {e}"))
        })
    }
}

/// Helper: wrap a schema in an `Arc<dyn AgentOutputSchemaBase>`.
pub fn output_schema<T>() -> Result<Arc<dyn AgentOutputSchemaBase>, AgentsError>
where
    T: Any + DeserializeOwned + crate::schemars::JsonSchema + Send + Sync + 'static,
{
    Ok(Arc::new(AgentOutputSchema::of::<T>()?))
}
