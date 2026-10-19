//! Run context (Python: `agents.run_context.RunContextWrapper`).
//!
//! The SDK is not generic over the user context. Business data travels in an
//! [`Extensions`] map keyed by Rust type; the wrapper itself holds the strongly typed,
//! SDK-owned state (usage). Missing data is reported by [`RunContextWrapper::require`] with
//! an actionable error instead of silently returning `None`.

use std::any::Any;
use std::sync::{Arc, Mutex};

use crate::error::{AgentsError, UserError};
use crate::extensions::Extensions;
use crate::usage::Usage;

pub use crate::extensions::ContextValue;

/// Wrapper for the per-run context shared with tools, guardrails and hooks.
///
/// Cloning is cheap: every field is behind an `Arc`.
#[derive(Clone, Default)]
pub struct RunContextWrapper {
    extensions: Extensions,
    usage: Arc<Mutex<Usage>>,
}

impl std::fmt::Debug for RunContextWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunContextWrapper")
            .field("extensions", &self.extensions)
            .field("usage", &self.usage())
            .finish()
    }
}

impl RunContextWrapper {
    /// Create a wrapper around an optional single user context value.
    pub fn new(context: Option<ContextValue>) -> Self {
        let mut extensions = Extensions::new();
        if let Some(value) = context {
            extensions.insert_shared(value);
        }
        Self::from_extensions(extensions)
    }

    /// Create a wrapper around a set of typed extensions.
    pub fn from_extensions(extensions: Extensions) -> Self {
        Self {
            extensions,
            usage: Arc::new(Mutex::new(Usage::default())),
        }
    }

    /// Wrap a concrete context value.
    pub fn with<T: Any + Send + Sync>(context: T) -> Self {
        Self::from_extensions(Extensions::new().with(context))
    }

    /// Whether any user data was supplied.
    pub fn has_context(&self) -> bool {
        !self.extensions.is_empty()
    }

    /// The typed extensions supplied for this run.
    pub fn extensions(&self) -> &Extensions {
        &self.extensions
    }

    /// Borrow the value of type `T`, if there is one.
    pub fn get<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.extensions.get::<T>()
    }

    /// Borrow the value of type `T` or fail with an actionable error.
    pub fn require<T: Any + Send + Sync>(&self) -> Result<&T, AgentsError> {
        self.get::<T>().ok_or_else(|| self.missing::<T>(None))
    }

    /// Same as [`RunContextWrapper::get`] (Python-style name).
    pub fn context<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.get::<T>()
    }

    /// Same as [`RunContextWrapper::require`] (Python-style name).
    pub fn try_context<T: Any + Send + Sync>(&self) -> Result<&T, AgentsError> {
        self.require::<T>()
    }

    /// The error for a missing value, optionally naming the tool that asked for it.
    pub(crate) fn missing<T: Any>(&self, asked_by: Option<&str>) -> AgentsError {
        let wanted = std::any::type_name::<T>();
        let who = asked_by
            .map(|name| format!(" (requested by `{name}`)"))
            .unwrap_or_default();
        if self.extensions.is_empty() {
            return UserError::new(format!(
                "no run context was provided{who}; add a value of type `{wanted}` with \
                 `RunOptions::with` or `RunOptions::context`"
            ))
            .into();
        }
        let held = self.extensions.known_type_names();
        let held = if held.is_empty() {
            String::new()
        } else {
            format!("; the run holds: {}", held.join(", "))
        };
        UserError::new(format!("run context is not of type `{wanted}`{who}{held}")).into()
    }

    /// Snapshot of the aggregated usage recorded so far.
    pub fn usage(&self) -> Usage {
        self.usage.lock().map(|u| u.clone()).unwrap_or_default()
    }

    /// Add usage observed by a model call.
    pub fn add_usage(&self, usage: &Usage) {
        if let Ok(mut current) = self.usage.lock() {
            current.add(usage);
        }
    }

    /// Replace the accumulated usage (used when resuming a run).
    pub fn set_usage(&self, usage: Usage) {
        if let Ok(mut current) = self.usage.lock() {
            *current = usage;
        }
    }
}

/// Context passed to tool invocations (Python: `ToolContext` subset).
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Tool name being invoked.
    pub tool_name: String,
    /// Call id from the model.
    pub tool_call_id: String,
    /// Raw JSON arguments string from the model.
    pub tool_arguments: String,
    /// Run context shared with guardrails and hooks.
    pub run_context: RunContextWrapper,
}

impl ToolContext {
    /// Build a tool context from the run context.
    pub fn new(
        tool_name: impl Into<String>,
        tool_call_id: impl Into<String>,
        tool_arguments: impl Into<String>,
        run_context: RunContextWrapper,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            tool_call_id: tool_call_id.into(),
            tool_arguments: tool_arguments.into(),
            run_context,
        }
    }

    /// Convenience accessor for the user context.
    pub fn context<T: std::any::Any + Send + Sync>(&self) -> Option<&T> {
        self.run_context.get::<T>()
    }

    /// Borrow the value of type `T`, or fail with an error that names this tool.
    pub fn require<T: std::any::Any + Send + Sync>(&self) -> Result<&T, AgentsError> {
        self.run_context
            .get::<T>()
            .ok_or_else(|| self.run_context.missing::<T>(Some(&self.tool_name)))
    }
}
