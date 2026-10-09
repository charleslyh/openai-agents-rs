//! Decisions, normalized errors and the retry policy contract.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::ModelError;

/// Whether provider-side work may already have happened (Python: `replay_safety`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplaySafety {
    /// Replaying the request cannot duplicate work.
    Safe,
    /// The provider may already have acted on the request.
    Unsafe,
}

/// Normalized facts about a failure, as seen by retry policies
/// (Python: `ModelRetryNormalizedError`).
///
/// Values returned from [`ModelRetryAdvice::normalized`] only override the fields they set with
/// the `with_*` methods, like Python's explicit-field tracking.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelRetryNormalizedError {
    /// HTTP status of the failure.
    pub status_code: Option<u16>,
    /// Provider error code.
    pub error_code: Option<String>,
    /// Error message.
    pub message: Option<String>,
    /// Provider request id.
    pub request_id: Option<String>,
    /// Seconds the provider asked to wait.
    pub retry_after: Option<f64>,
    /// The call was cancelled by the caller.
    pub is_abort: bool,
    /// The request failed to reach the provider.
    pub is_network_error: bool,
    /// The request timed out.
    pub is_timeout: bool,
    explicit: Vec<&'static str>,
}

macro_rules! explicit_setter {
    ($name:ident, $field:literal, $ty:ty) => {
        /// Set this field and mark it as overriding the SDK's own reading of the error.
        pub fn $name(mut self, value: $ty) -> Self {
            self.$name = value;
            self.explicit.push($field);
            self
        }
    };
}

impl ModelRetryNormalizedError {
    /// An override that sets nothing yet.
    pub fn new() -> Self {
        Self::default()
    }

    explicit_setter!(status_code, "status_code", Option<u16>);
    explicit_setter!(error_code, "error_code", Option<String>);
    explicit_setter!(message, "message", Option<String>);
    explicit_setter!(request_id, "request_id", Option<String>);
    explicit_setter!(retry_after, "retry_after", Option<f64>);
    explicit_setter!(is_abort, "is_abort", bool);
    explicit_setter!(is_network_error, "is_network_error", bool);
    explicit_setter!(is_timeout, "is_timeout", bool);

    pub(crate) fn from_error(error: &ModelError) -> Self {
        let mut n = Self {
            message: Some(error.to_string()),
            ..Self::default()
        };
        match error {
            ModelError::Status(e) => {
                n.status_code = Some(e.status_code);
                n.error_code = e.error_code().map(str::to_string);
                n.request_id = e.request_id().map(str::to_string);
                n.retry_after = e.retry_after();
            }
            ModelError::Connection(e) => {
                n.is_network_error = true;
                n.is_timeout = e.is_timeout;
            }
            ModelError::Timeout(_) => n.is_timeout = true,
            _ => {}
        }
        // Python also recognizes network failures by their message.
        let text = error.to_string().to_lowercase();
        if text.contains("connection error")
            || text.contains("network error")
            || text.contains("socket hang up")
            || text.contains("connection closed")
        {
            n.is_network_error = true;
        }
        n
    }

    pub(crate) fn apply_override(&mut self, over: &ModelRetryNormalizedError) {
        for field in &over.explicit {
            match *field {
                "status_code" => self.status_code = over.status_code,
                "error_code" => self.error_code = over.error_code.clone(),
                "message" => self.message = over.message.clone(),
                "request_id" => self.request_id = over.request_id.clone(),
                "retry_after" => self.retry_after = over.retry_after,
                "is_network_error" => self.is_network_error = over.is_network_error,
                "is_timeout" => self.is_timeout = over.is_timeout,
                // A provider can add abort evidence but never clear one the SDK inferred.
                "is_abort" => self.is_abort = self.is_abort || over.is_abort,
                _ => {}
            }
        }
    }
}

/// Provider-specific retry guidance (Python: `ModelRetryAdvice`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelRetryAdvice {
    /// The provider's opinion: retry (`true`), do not (`false`), or no opinion.
    pub suggested: Option<bool>,
    /// Seconds the provider asked to wait.
    pub retry_after: Option<f64>,
    /// Whether replaying the request is safe; `None` means unknown.
    pub replay_safety: Option<ReplaySafety>,
    /// Why the provider gave this advice.
    pub reason: Option<String>,
    /// Corrections to the SDK's reading of the error.
    pub normalized: Option<ModelRetryNormalizedError>,
}

/// What an adapter is asked when it derives retry advice (Python: `ModelRetryAdviceRequest`).
#[derive(Debug, Clone)]
pub struct ModelRetryAdviceRequest {
    /// The failure.
    pub error: ModelError,
    /// 1-based attempt that failed.
    pub attempt: u32,
    /// Whether the failed call was streamed.
    pub stream: bool,
    /// `previous_response_id` of the request.
    pub previous_response_id: Option<String>,
    /// `conversation_id` of the request.
    pub conversation_id: Option<String>,
}

/// A retry policy's verdict (Python: `RetryDecision`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RetryDecision {
    /// Retry the request.
    pub retry: bool,
    /// Seconds to wait first; `None` uses the provider's `Retry-After` or the backoff.
    pub delay: Option<f64>,
    /// Why.
    pub reason: Option<String>,
    /// Explicit approval to replay a request the provider marked replay-unsafe. An ordinary
    /// `retry: true` never bypasses replay protection.
    pub approve_unsafe_replay: bool,
    pub(crate) hard_veto: bool,
    pub(crate) delegable_replay_veto: bool,
    pub(crate) approves_replay: bool,
}

impl From<bool> for RetryDecision {
    fn from(retry: bool) -> Self {
        Self {
            retry,
            ..Self::default()
        }
    }
}

impl RetryDecision {
    /// Retry.
    pub fn yes() -> Self {
        true.into()
    }

    /// Do not retry.
    pub fn no() -> Self {
        false.into()
    }

    /// Wait `seconds` before retrying.
    pub fn with_delay(mut self, seconds: f64) -> Self {
        self.delay = Some(seconds);
        self
    }

    /// Attach a reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Approve replaying a request the provider marked unsafe to replay.
    pub fn with_approve_unsafe_replay(mut self) -> Self {
        self.approve_unsafe_replay = true;
        self
    }

    pub(crate) fn hard_veto(mut self) -> Self {
        self.hard_veto = true;
        self
    }

    pub(crate) fn delegable_replay_veto(mut self) -> Self {
        self.hard_veto = true;
        self.delegable_replay_veto = true;
        self
    }

    pub(crate) fn replay_safe_approval(mut self) -> Self {
        self.approves_replay = true;
        self
    }

    pub(crate) fn merge_positive(
        existing: RetryDecision,
        incoming: RetryDecision,
    ) -> RetryDecision {
        let mut merged = RetryDecision {
            retry: true,
            delay: existing.delay,
            reason: existing.reason,
            approve_unsafe_replay: existing.approve_unsafe_replay || incoming.approve_unsafe_replay,
            ..Default::default()
        };
        if existing.approves_replay {
            merged = merged.replay_safe_approval();
        }
        if incoming.delay.is_some() {
            merged.delay = incoming.delay;
        }
        if incoming.reason.is_some() {
            merged.reason = incoming.reason;
        }
        if incoming.approves_replay {
            merged = merged.replay_safe_approval();
        }
        merged
    }

    pub(crate) fn resolve_delegable_veto(
        veto: RetryDecision,
        approving: RetryDecision,
    ) -> RetryDecision {
        if !approving.retry || !approving.approve_unsafe_replay {
            return veto;
        }
        let mut resolved = RetryDecision {
            retry: true,
            delay: approving.delay,
            reason: approving.reason.or(veto.reason),
            approve_unsafe_replay: true,
            ..Default::default()
        };
        if approving.approves_replay {
            resolved = resolved.replay_safe_approval();
        }
        resolved
    }
}

/// What a retry policy sees about a failure (Python: `RetryPolicyContext`).
#[derive(Debug, Clone)]
pub struct RetryPolicyContext {
    /// The failure.
    pub error: ModelError,
    /// 1-based attempt that failed.
    pub attempt: u32,
    /// Retries allowed after the first request.
    pub max_retries: u32,
    /// Whether the failed call was streamed.
    pub stream: bool,
    /// Normalized facts about the failure.
    pub normalized: ModelRetryNormalizedError,
    /// The adapter's advice, if it gave any.
    pub provider_advice: Option<ModelRetryAdvice>,
    /// `previous_response_id` of the request.
    pub previous_response_id: Option<String>,
    /// `conversation_id` of the request.
    pub conversation_id: Option<String>,
}

impl RetryPolicyContext {
    /// Provider replay classification; `None` is unknown.
    pub fn replay_safety(&self) -> Option<ReplaySafety> {
        self.provider_advice.as_ref().and_then(|a| a.replay_safety)
    }

    /// Whether the request depends on server-side conversation state.
    pub fn stateful_request(&self) -> bool {
        self.previous_response_id.is_some() || self.conversation_id.is_some()
    }
}

type PolicyFuture = Pin<Box<dyn Future<Output = RetryDecision> + Send>>;

/// Decides whether a failed model request is retried (Python: `RetryPolicy`).
#[derive(Clone)]
pub struct RetryPolicy(Arc<dyn Fn(RetryPolicyContext) -> PolicyFuture + Send + Sync>);

impl std::fmt::Debug for RetryPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RetryPolicy(..)")
    }
}

impl PartialEq for RetryPolicy {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl RetryPolicy {
    /// Build a policy from an async closure returning a `bool` or a [`RetryDecision`].
    pub fn new<F, Fut, D>(f: F) -> Self
    where
        F: Fn(RetryPolicyContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = D> + Send + 'static,
        D: Into<RetryDecision>,
    {
        Self(Arc::new(move |ctx| {
            let fut = f(ctx);
            Box::pin(async move { fut.await.into() })
        }))
    }

    /// Evaluate the policy.
    pub async fn evaluate(&self, context: RetryPolicyContext) -> RetryDecision {
        (self.0)(context).await
    }
}
