//! Error types aligned with Python `agents.exceptions` (supported subset).

use thiserror::Error;

/// Top-level Agents SDK error.
#[derive(Debug, Error)]
pub enum AgentsError {
    /// Maximum number of turns was exceeded.
    #[error(transparent)]
    MaxTurns(#[from] MaxTurnsExceeded),
    /// Model / provider failure.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// The model refused to answer (Python: `ModelRefusalError`).
    #[error(transparent)]
    ModelRefusal(#[from] ModelRefusalError),
    /// Invalid user configuration or input.
    #[error(transparent)]
    User(#[from] UserError),
    /// An input guardrail halted the run (Python: `InputGuardrailTripwireTriggered`).
    #[error(transparent)]
    InputGuardrailTripwire(#[from] InputGuardrailTripwireTriggered),
    /// An output guardrail halted the run (Python: `OutputGuardrailTripwireTriggered`).
    #[error(transparent)]
    OutputGuardrailTripwire(#[from] OutputGuardrailTripwireTriggered),
    /// Tool execution failure that should abort the run.
    #[error("tool error: {message}")]
    Tool {
        /// Human-readable message.
        message: String,
        /// Underlying cause, when there is one.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    /// A tool input guardrail halted the run (Python: `ToolInputGuardrailTripwireTriggered`).
    #[error(transparent)]
    ToolInputGuardrailTripwire(#[from] ToolInputGuardrailTripwireTriggered),
    /// A tool output guardrail halted the run (Python: `ToolOutputGuardrailTripwireTriggered`).
    #[error(transparent)]
    ToolOutputGuardrailTripwire(#[from] ToolOutputGuardrailTripwireTriggered),
    /// A function tool exceeded its timeout with `ToolTimeoutBehavior::RaiseException`
    /// (Python: `ToolTimeoutError`).
    #[error(transparent)]
    ToolTimeout(#[from] ToolTimeoutError),
    /// Unexpected internal failure.
    #[error("internal error: {message}")]
    Internal {
        /// Human-readable message.
        message: String,
        /// Underlying cause, when there is one.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

impl AgentsError {
    /// A tool failure described by `message`.
    pub fn tool(message: impl Into<String>) -> Self {
        Self::Tool {
            message: message.into(),
            source: None,
        }
    }

    /// A tool failure that preserves `source` in the error chain.
    pub fn tool_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Tool {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// An internal failure described by `message`.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
            source: None,
        }
    }

    /// An internal failure that preserves `source` in the error chain.
    pub fn internal_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Internal {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

/// Raised when a tool input guardrail asks to stop the run.
#[derive(Debug, Error, Clone)]
#[error("Tool input guardrail `{guardrail_name}` triggered a tripwire")]
pub struct ToolInputGuardrailTripwireTriggered {
    /// Name of the guardrail that tripped.
    pub guardrail_name: String,
    /// What the guardrail returned.
    pub output: crate::tool_guardrails::ToolGuardrailFunctionOutput,
}

/// Raised when a tool output guardrail asks to stop the run.
#[derive(Debug, Error, Clone)]
#[error("Tool output guardrail `{guardrail_name}` triggered a tripwire")]
pub struct ToolOutputGuardrailTripwireTriggered {
    /// Name of the guardrail that tripped.
    pub guardrail_name: String,
    /// What the guardrail returned.
    pub output: crate::tool_guardrails::ToolGuardrailFunctionOutput,
}

/// Raised when a function tool invocation exceeds its timeout (Python: `ToolTimeoutError`).
#[derive(Debug, Error, Clone)]
#[error("{}", crate::tool::default_tool_timeout_error_message(tool_name, *timeout_seconds))]
pub struct ToolTimeoutError {
    /// Name of the tool that timed out.
    pub tool_name: String,
    /// The configured timeout.
    pub timeout_seconds: f64,
}

/// Raised when `max_turns` is exceeded (Python: `MaxTurnsExceeded`).
#[derive(Debug, Error, Clone)]
#[error("Max turns exceeded: {max_turns}")]
pub struct MaxTurnsExceeded {
    /// Configured maximum turns.
    pub max_turns: usize,
}

/// Raised when the model refuses to produce the requested output
/// (Python: `ModelRefusalError`).
#[derive(Debug, Error, Clone)]
#[error("Model refused to produce output: {refusal}")]
pub struct ModelRefusalError {
    /// The refusal text returned by the model.
    pub refusal: String,
}

/// Model-layer errors.
#[derive(Debug, Error, Clone)]
pub enum ModelError {
    /// Scripted model ran out of steps or received an unexpected call.
    #[error("model script error: {0}")]
    Script(String),
    /// HTTP / transport failure without more structure (for example an undecodable body).
    #[error("model transport error: {0}")]
    Transport(String),
    /// The provider answered with a non-success HTTP status.
    #[error("model transport error: {0}")]
    Status(ModelStatusError),
    /// The request never produced a response: connection, TLS, timeout or body-read failure.
    #[error("model transport error: {0}")]
    Connection(ModelConnectionError),
    /// One model attempt exceeded `ModelSettings.timeout` (Python: `ModelTimeoutError`).
    #[error(transparent)]
    Timeout(ModelTimeoutError),
    /// Provider returned an unusable payload.
    #[error("model behavior error: {0}")]
    Behavior(String),
    /// Feature or API not available in this build.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// A non-success HTTP answer from the provider (Python: `openai.APIStatusError`).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelStatusError {
    /// HTTP status code.
    pub status_code: u16,
    /// Human-readable message (status and body).
    pub message: String,
    /// Parsed JSON body, or the raw text as a string.
    pub body: serde_json::Value,
    /// Response headers, names lower-cased.
    pub headers: std::collections::BTreeMap<String, String>,
}

impl std::fmt::Display for ModelStatusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl ModelStatusError {
    /// A response header by case-insensitive name.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }

    /// The provider's error code (`body.error.code`, falling back to `body.code`).
    pub fn error_code(&self) -> Option<&str> {
        self.body
            .pointer("/error/code")
            .or_else(|| self.body.get("code"))
            .and_then(serde_json::Value::as_str)
    }

    /// The provider's request id (`x-request-id`).
    pub fn request_id(&self) -> Option<&str> {
        self.header("x-request-id")
    }

    /// Seconds the provider asked the client to wait (`retry-after-ms`, then `retry-after`).
    pub fn retry_after(&self) -> Option<f64> {
        crate::retry::retry_after_from_headers(self.header("retry-after-ms"), self.header("retry-after"))
    }
}

/// A request that failed before an HTTP answer arrived (Python: `openai.APIConnectionError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConnectionError {
    /// Human-readable message.
    pub message: String,
    /// Whether the transport timed out (Python: `APITimeoutError`).
    pub is_timeout: bool,
}

impl std::fmt::Display for ModelConnectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// One model attempt exceeded its timeout (Python: `ModelTimeoutError`).
#[derive(Debug, Error, Clone, PartialEq)]
#[error("Model call timed out after {timeout_seconds} seconds.")]
pub struct ModelTimeoutError {
    /// The configured timeout.
    pub timeout_seconds: f64,
}

/// Raised when an input guardrail trips (Python: `InputGuardrailTripwireTriggered`).
#[derive(Debug, Error, Clone)]
#[error("Input guardrail `{}` triggered a tripwire", result.guardrail_name)]
pub struct InputGuardrailTripwireTriggered {
    /// The guardrail result that tripped.
    pub result: crate::guardrail::InputGuardrailResult,
}

/// Raised when an output guardrail trips (Python: `OutputGuardrailTripwireTriggered`).
#[derive(Debug, Error, Clone)]
#[error("Output guardrail `{}` triggered a tripwire", result.guardrail_name)]
pub struct OutputGuardrailTripwireTriggered {
    /// The guardrail result that tripped.
    pub result: crate::guardrail::OutputGuardrailResult,
}

/// Invalid configuration or caller misuse (Python: `UserError`).
#[derive(Debug, Error, Clone)]
#[error("{0}")]
pub struct UserError(pub String);

impl UserError {
    /// Create a user error from a displayable message.
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}
