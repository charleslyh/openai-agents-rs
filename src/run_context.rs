//! Run context (Python: `agents.run_context.RunContextWrapper`).
//!
//! The SDK is not generic over the user context: it is carried as
//! `Arc<dyn Any + Send + Sync>` and downcast on demand. This keeps `Agent` / `Runner`
//! free of type parameters at the cost of discovering context type errors at runtime, so
//! [`RunContextWrapper::try_context`] reports a precise error instead of silently returning
//! `None`.

use std::any::Any;
use std::sync::{Arc, Mutex};

use crate::error::{AgentsError, UserError};
use crate::usage::Usage;

/// A user-supplied context value.
pub type ContextValue = Arc<dyn Any + Send + Sync>;

/// Wrapper for the per-run context shared with tools, guardrails and hooks.
///
/// Cloning is cheap: every field is behind an `Arc`.
#[derive(Clone, Default)]
pub struct RunContextWrapper {
    context: Option<ContextValue>,
    usage: Arc<Mutex<Usage>>,
}

impl std::fmt::Debug for RunContextWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunContextWrapper")
            .field("has_context", &self.context.is_some())
            .field("usage", &self.usage())
            .finish()
    }
}

impl RunContextWrapper {
    /// Create a wrapper around an optional user context.
    pub fn new(context: Option<ContextValue>) -> Self {
        Self {
            context,
            usage: Arc::new(Mutex::new(Usage::default())),
        }
    }

    /// Wrap a concrete context value.
    pub fn with<T: Any + Send + Sync>(context: T) -> Self {
        Self::new(Some(Arc::new(context)))
    }

    /// Whether a user context was supplied.
    pub fn has_context(&self) -> bool {
        self.context.is_some()
    }

    /// Borrow the user context, if it is of type `T`.
    pub fn context<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.context.as_ref()?.downcast_ref::<T>()
    }

    /// Borrow the user context or fail with an actionable error.
    pub fn try_context<T: Any + Send + Sync>(&self) -> Result<&T, AgentsError> {
        let Some(value) = &self.context else {
            return Err(UserError::new(format!(
                "no run context was provided; pass `RunOptions::context` with a value of type `{}`",
                std::any::type_name::<T>()
            ))
            .into());
        };
        value.downcast_ref::<T>().ok_or_else(|| {
            UserError::new(format!(
                "run context is not of type `{}`",
                std::any::type_name::<T>()
            ))
            .into()
        })
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
