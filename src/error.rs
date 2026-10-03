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

/// Raised when `max_turns` is exceeded (Python: `MaxTurnsExceeded`).
#[derive(Debug, Error, Clone)]
#[error("Max turns exceeded: {max_turns}")]
pub struct MaxTurnsExceeded {
    /// Configured maximum turns.
    pub max_turns: usize,
}

/// Model-layer errors.
#[derive(Debug, Error)]
pub enum ModelError {
    /// Scripted model ran out of steps or received an unexpected call.
    #[error("model script error: {0}")]
    Script(String),
    /// HTTP / transport failure.
    #[error("model transport error: {0}")]
    Transport(String),
    /// Provider returned an unusable payload.
    #[error("model behavior error: {0}")]
    Behavior(String),
    /// Feature or API not available in this build.
    #[error("unsupported: {0}")]
    Unsupported(String),
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
