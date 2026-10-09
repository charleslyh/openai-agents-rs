//! Runner-managed model retries (Python: `agents.retry` and `run_internal.model_retry`).
//!
//! Retries are opt-in: set [`ModelRetrySettings`] on [`crate::ModelSettings::retry`]. With a
//! `policy`, a failed model request is replayed up to `max_retries` times, waiting `backoff`
//! between attempts (or the provider's `Retry-After`). Without one nothing is retried, except the
//! legacy `conversation_locked` replay that Python keeps for compatibility.
//!
//! The Rust SDK has no hidden client-side retries (Python's OpenAI client retries on its own and
//! the runner has to switch that off), so there is nothing to disable here.

mod delay;
mod executor;
mod policies;
mod policy;
mod settings;

/// Built-in policies (Python: `retry_policies`).
pub mod retry_policies {
    pub use super::policies::retry_policies::*;
}

pub(crate) use delay::{apply_retry_attempt_usage, default_retry_delay};
pub(crate) use executor::{call_with_retry, RetryCall};
pub use policy::{
    ModelRetryAdvice, ModelRetryAdviceRequest, ModelRetryNormalizedError, ReplaySafety,
    RetryDecision, RetryPolicy, RetryPolicyContext,
};
pub use settings::{
    ModelRetryBackoffSettings, ModelRetrySettings, COMPATIBILITY_CONVERSATION_LOCKED_RETRIES,
    DEFAULT_BACKOFF_JITTER, DEFAULT_BACKOFF_MULTIPLIER, DEFAULT_INITIAL_DELAY_SECONDS,
    DEFAULT_MAX_DELAY_SECONDS,
};
