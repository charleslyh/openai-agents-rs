//! The one error this crate has: a guardrail that could not be built.

use std::fmt;

/// A guardrail was configured in a way that could never work (Rust-only, X-020).
///
/// A keyword filter with no keywords, a pattern that is not a regular expression and a length
/// limit of zero all build a guardrail that does nothing or rejects everything. Returning this at
/// construction means that mistake is found when the program starts, not when a filter that looked
/// fine turns out never to have matched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildError {
    message: String,
}

impl BuildError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// What was wrong.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "guardrail configuration: {}", self.message)
    }
}

impl std::error::Error for BuildError {}
