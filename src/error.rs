//! Error types aligned with Python `agents.exceptions` (Phase-1 subset).

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
    /// Tool execution failure that should abort the run.
    #[error("tool error: {0}")]
    Tool(String),
    /// Unexpected internal failure.
    #[error("internal error: {0}")]
    Internal(String),
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
