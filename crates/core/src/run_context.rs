//! Run context (Python: `agents.run_context.RunContextWrapper`).
//!
//! The SDK is not generic over the user context. Business data travels in an
//! [`Extensions`] map keyed by Rust type; the wrapper itself holds the strongly typed,
//! SDK-owned state (usage). Missing data is reported by [`RunContextWrapper::require`] with
//! an actionable error instead of silently returning `None`.

use std::any::Any;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, MutexGuard};

use tokio_util::sync::CancellationToken;

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
    /// Deferred capabilities the model has loaded, as `(agent name, capability name)`.
    loaded: Arc<Mutex<BTreeSet<(String, String)>>>,
    /// Cancels the run; tool calls get a child of it ([`ToolContext::cancellation`]).
    cancellation: CancellationToken,
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
            loaded: Arc::default(),
            cancellation: CancellationToken::new(),
        }
    }

    /// Use `token` as the run's cancellation token.
    #[must_use]
    pub(crate) fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = token;
        self
    }

    /// The token that cancels this run. Cancelling it asks the run to stop: the model call is
    /// dropped, tool calls already running are left to finish, and the run ends with
    /// [`RunResult::is_cancelled`](crate::result::RunResult::is_cancelled) set.
    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    /// Whether the run was asked to stop.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
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

    /// The loaded set. A poisoned lock still holds a valid set (every update is one insert), so
    /// it is used as is rather than panicking.
    fn loaded_set(&self) -> MutexGuard<'_, BTreeSet<(String, String)>> {
        self.loaded.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether the model has loaded deferred capability `capability` of `agent`.
    pub(crate) fn is_capability_loaded(&self, agent: &str, capability: &str) -> bool {
        self.loaded_set()
            .contains(&(agent.to_owned(), capability.to_owned()))
    }

    /// Mark a deferred capability loaded; `false` when it already was.
    pub(crate) fn load_capability(&self, agent: &str, capability: &str) -> bool {
        self.loaded_set()
            .insert((agent.to_owned(), capability.to_owned()))
    }

    /// Every loaded `(agent, capability)` pair, sorted. Saved in a paused run's state.
    pub(crate) fn loaded_capabilities(&self) -> Vec<(String, String)> {
        self.loaded_set().iter().cloned().collect()
    }

    /// Replace the loaded set (used when resuming a run).
    pub(crate) fn set_loaded_capabilities(&self, loaded: &[(String, String)]) {
        *self.loaded_set() = loaded.iter().cloned().collect();
    }
}

/// Context passed to tool invocations (Python: `ToolContext` subset).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolContext {
    /// Tool name being invoked.
    pub tool_name: String,
    /// Call id from the model.
    pub tool_call_id: String,
    /// Raw JSON arguments string from the model.
    pub tool_arguments: String,
    /// Run context shared with guardrails and hooks.
    pub run_context: RunContextWrapper,
    cancellation: CancellationToken,
}

impl ToolContext {
    /// Build a tool context from the run context. Its cancellation token is a child of the
    /// run's.
    pub fn new(
        tool_name: impl Into<String>,
        tool_call_id: impl Into<String>,
        tool_arguments: impl Into<String>,
        run_context: RunContextWrapper,
    ) -> Self {
        let cancellation = run_context.cancellation().child_token();
        Self {
            tool_name: tool_name.into(),
            tool_call_id: tool_call_id.into(),
            tool_arguments: tool_arguments.into(),
            run_context,
            cancellation,
        }
    }

    /// The cancellation token of this call, a child of the run's token.
    ///
    /// Cancellation is cooperative: the SDK never drops a tool that is running. A long tool
    /// should watch this token (`ctx.cancellation().cancelled().await` in a `select!`, or
    /// [`is_cancelled`](Self::is_cancelled) between steps) and return early; whatever it returns
    /// is still recorded as its output. Calls that had not started when the run was cancelled
    /// never run.
    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    /// Whether this call, or the run it belongs to, was cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
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
