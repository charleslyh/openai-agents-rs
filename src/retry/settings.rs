//! Retry settings and their defaults.
//! Runner-managed model retries (Python: `agents.retry` and `run_internal.model_retry`).
//!
//! Retries are opt-in: set [`ModelRetrySettings`] on [`crate::ModelSettings::retry`]. With a
//! `policy`, a failed model request is replayed up to `max_retries` times, waiting `backoff`
//! between attempts (or the provider's `Retry-After`). Without one nothing is retried, except the
//! legacy `conversation_locked` replay that Python keeps for compatibility.
//!
//! The Rust SDK has no hidden client-side retries (Python's OpenAI client retries on its own and
//! the runner has to switch that off), so there is nothing to disable here.

use serde::{Deserialize, Serialize};

use super::RetryPolicy;

/// Delay before the first retry when no backoff is configured.
pub const DEFAULT_INITIAL_DELAY_SECONDS: f64 = 0.25;
/// Longest wait between attempts when no backoff is configured.
pub const DEFAULT_MAX_DELAY_SECONDS: f64 = 2.0;
/// Growth factor of the wait between attempts.
pub const DEFAULT_BACKOFF_MULTIPLIER: f64 = 2.0;
/// Whether the wait is randomized by default.
pub const DEFAULT_BACKOFF_JITTER: bool = true;
/// How many times a `conversation_locked` error is replayed without any policy.
pub const COMPATIBILITY_CONVERSATION_LOCKED_RETRIES: u32 = 3;

/// Backoff configuration (Python: `ModelRetryBackoffSettings`). `None` fields use the defaults.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelRetryBackoffSettings {
    /// Delay in seconds before the first retry.
    pub initial_delay: Option<f64>,
    /// Longest delay in seconds between retries.
    pub max_delay: Option<f64>,
    /// Factor the delay grows by after every retry.
    pub multiplier: Option<f64>,
    /// Randomize the delay by +-12.5%.
    pub jitter: Option<bool>,
}

impl ModelRetryBackoffSettings {
    /// Overlay the non-`None` fields of `over` (Python: `_merge_backoff_settings`).
    fn merged_with(self, over: ModelRetryBackoffSettings) -> Self {
        Self {
            initial_delay: over.initial_delay.or(self.initial_delay),
            max_delay: over.max_delay.or(self.max_delay),
            multiplier: over.multiplier.or(self.multiplier),
            jitter: over.jitter.or(self.jitter),
        }
    }
}

/// Opt-in retry settings for model calls (Python: `ModelRetrySettings`).
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct ModelRetrySettings {
    /// Retries allowed after the first request.
    pub max_retries: Option<u32>,
    /// Backoff applied when the policy gives no explicit delay.
    pub backoff: Option<ModelRetryBackoffSettings>,
    /// Decides whether a failure is retried. Runtime-only: it is not serialized.
    #[serde(skip)]
    pub policy: Option<RetryPolicy>,
}

impl std::fmt::Debug for ModelRetrySettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelRetrySettings")
            .field("max_retries", &self.max_retries)
            .field("backoff", &self.backoff)
            .field("has_policy", &self.policy.is_some())
            .finish()
    }
}

impl PartialEq for ModelRetrySettings {
    fn eq(&self, other: &Self) -> bool {
        self.max_retries == other.max_retries
            && self.backoff == other.backoff
            && self.policy == other.policy
    }
}

impl ModelRetrySettings {
    /// Retry up to `max_retries` times with `policy`.
    pub fn new(max_retries: u32, policy: RetryPolicy) -> Self {
        Self {
            max_retries: Some(max_retries),
            backoff: None,
            policy: Some(policy),
        }
    }

    /// Use `backoff` between attempts.
    pub fn with_backoff(mut self, backoff: ModelRetryBackoffSettings) -> Self {
        self.backoff = Some(backoff);
        self
    }

    /// Overlay `over` on top of `self` (Python: `_merge_retry_settings`): non-`None` fields win and
    /// the backoff is merged field by field.
    pub(crate) fn merged_with(&self, over: &ModelRetrySettings) -> ModelRetrySettings {
        ModelRetrySettings {
            max_retries: over.max_retries.or(self.max_retries),
            backoff: match (self.backoff, over.backoff) {
                (None, b) => b,
                (a, None) => a,
                (Some(a), Some(b)) => Some(a.merged_with(b)),
            },
            policy: over.policy.clone().or_else(|| self.policy.clone()),
        }
    }
}
