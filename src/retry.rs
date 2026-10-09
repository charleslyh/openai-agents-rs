//! Runner-managed model retries (Python: `agents.retry` and `run_internal.model_retry`).
//!
//! Retries are opt-in: set [`ModelRetrySettings`] on [`crate::ModelSettings::retry`]. With a
//! `policy`, a failed model request is replayed up to `max_retries` times, waiting `backoff`
//! between attempts (or the provider's `Retry-After`). Without one nothing is retried, except the
//! legacy `conversation_locked` replay that Python keeps for compatibility.
//!
//! The Rust SDK has no hidden client-side retries (Python's OpenAI client retries on its own and
//! the runner has to switch that off), so there is nothing to disable here.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::items::ModelResponse;
use crate::usage::{RequestUsage, Usage};

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

    fn from_error(error: &ModelError) -> Self {
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

    fn apply_override(&mut self, over: &ModelRetryNormalizedError) {
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
    hard_veto: bool,
    delegable_replay_veto: bool,
    approves_replay: bool,
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

    fn hard_veto(mut self) -> Self {
        self.hard_veto = true;
        self
    }

    fn delegable_replay_veto(mut self) -> Self {
        self.hard_veto = true;
        self.delegable_replay_veto = true;
        self
    }

    fn replay_safe_approval(mut self) -> Self {
        self.approves_replay = true;
        self
    }

    fn merge_positive(existing: RetryDecision, incoming: RetryDecision) -> RetryDecision {
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

    fn resolve_delegable_veto(veto: RetryDecision, approving: RetryDecision) -> RetryDecision {
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

/// Built-in policies (Python: `retry_policies`).
pub mod retry_policies {
    use super::*;

    /// Never retry.
    pub fn never() -> RetryPolicy {
        RetryPolicy::new(|_| async { false })
    }

    /// Follow the provider's advice (Python: `provider_suggested`).
    pub fn provider_suggested() -> RetryPolicy {
        RetryPolicy::new(|ctx: RetryPolicyContext| async move {
            let advice = ctx.provider_advice.as_ref();
            let reason = advice.and_then(|a| a.reason.clone());
            let retry_after = advice.and_then(|a| a.retry_after);
            match advice.and_then(|a| a.suggested) {
                None => RetryDecision::no(),
                Some(false) => {
                    let mut decision = RetryDecision::no();
                    decision.reason = reason;
                    if ctx.replay_safety() == Some(ReplaySafety::Unsafe) {
                        decision.delegable_replay_veto()
                    } else {
                        decision.hard_veto()
                    }
                }
                Some(true) => {
                    let mut decision = RetryDecision::yes();
                    decision.delay = retry_after;
                    decision.reason = reason;
                    if ctx.replay_safety() == Some(ReplaySafety::Safe) {
                        decision.replay_safe_approval()
                    } else {
                        decision
                    }
                }
            }
        })
    }

    /// Retry network failures and timeouts.
    pub fn network_error() -> RetryPolicy {
        RetryPolicy::new(|ctx: RetryPolicyContext| async move {
            ctx.normalized.is_network_error || ctx.normalized.is_timeout
        })
    }

    /// Retry when the provider sent a `Retry-After`, waiting that long.
    pub fn retry_after() -> RetryPolicy {
        RetryPolicy::new(|ctx: RetryPolicyContext| async move {
            let delay = ctx
                .normalized
                .retry_after
                .or_else(|| ctx.provider_advice.as_ref().and_then(|a| a.retry_after));
            match delay {
                Some(delay) => RetryDecision::yes().with_delay(delay),
                None => RetryDecision::no(),
            }
        })
    }

    /// Retry the given HTTP statuses.
    pub fn http_status(statuses: impl IntoIterator<Item = u16>) -> RetryPolicy {
        let allowed: Vec<u16> = statuses.into_iter().collect();
        RetryPolicy::new(move |ctx: RetryPolicyContext| {
            let retry = ctx
                .normalized
                .status_code
                .is_some_and(|code| allowed.contains(&code));
            async move { retry }
        })
    }

    /// Retry only when every policy agrees (Python: `all`).
    pub fn all(policies: Vec<RetryPolicy>) -> RetryPolicy {
        if policies.is_empty() {
            return never();
        }
        RetryPolicy::new(move |ctx: RetryPolicyContext| {
            let policies = policies.clone();
            async move {
                let mut merged = RetryDecision::yes();
                let mut delegable: Option<RetryDecision> = None;
                for policy in &policies {
                    let decision = policy.evaluate(ctx.clone()).await;
                    if decision.hard_veto {
                        if decision.delegable_replay_veto {
                            delegable.get_or_insert(decision);
                            continue;
                        }
                        return decision;
                    }
                    if !decision.retry {
                        return decision;
                    }
                    if decision.delay.is_some() {
                        merged.delay = decision.delay;
                    }
                    if decision.reason.is_some() {
                        merged.reason = decision.reason.clone();
                    }
                    if decision.approve_unsafe_replay {
                        merged.approve_unsafe_replay = true;
                    }
                    if decision.approves_replay {
                        merged = merged.replay_safe_approval();
                    }
                }
                match delegable {
                    Some(veto) => RetryDecision::resolve_delegable_veto(veto, merged),
                    None => merged,
                }
            }
        })
    }

    /// Retry when any policy agrees (Python: `any`).
    pub fn any(policies: Vec<RetryPolicy>) -> RetryPolicy {
        if policies.is_empty() {
            return never();
        }
        RetryPolicy::new(move |ctx: RetryPolicyContext| {
            let policies = policies.clone();
            async move {
                let mut first_positive: Option<RetryDecision> = None;
                let mut last_negative: Option<RetryDecision> = None;
                let mut delegable: Option<RetryDecision> = None;
                for policy in &policies {
                    let decision = policy.evaluate(ctx.clone()).await;
                    if decision.hard_veto {
                        if decision.delegable_replay_veto {
                            delegable.get_or_insert(decision);
                            continue;
                        }
                        return decision;
                    }
                    if decision.retry {
                        first_positive = Some(match first_positive {
                            None => decision,
                            Some(existing) => RetryDecision::merge_positive(existing, decision),
                        });
                        continue;
                    }
                    last_negative = Some(decision);
                }
                if let Some(veto) = delegable {
                    return match first_positive {
                        None => veto,
                        Some(positive) => RetryDecision::resolve_delegable_veto(veto, positive),
                    };
                }
                first_positive
                    .or(last_negative)
                    .unwrap_or_else(RetryDecision::no)
            }
        })
    }
}

/// `retry-after-ms` (milliseconds) then `retry-after` (seconds or an HTTP date), as seconds
/// (Python: `get_retry_after`).
pub(crate) fn retry_after_from_headers(ms: Option<&str>, value: Option<&str>) -> Option<f64> {
    if let Some(ms) = ms.and_then(|v| v.trim().parse::<f64>().ok()) {
        if ms >= 0.0 {
            return Some(ms / 1000.0);
        }
    }
    let value = value?.trim();
    if let Ok(seconds) = value.parse::<f64>() {
        return (seconds >= 0.0).then_some(seconds);
    }
    // HTTP-date form: `Wed, 21 Oct 2026 07:28:00 GMT`.
    let until = httpdate_to_unix(value)?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs_f64();
    Some((until - now).max(0.0))
}

/// Seconds since the epoch for an IMF-fixdate (`Sun, 06 Nov 1994 08:49:37 GMT`).
fn httpdate_to_unix(value: &str) -> Option<f64> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    if parts.len() != 6 || parts[5] != "GMT" {
        return None;
    }
    let day: i64 = parts[1].parse().ok()?;
    let month = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    .iter()
    .position(|m| *m == parts[2])? as i64
        + 1;
    let year: i64 = parts[3].parse().ok()?;
    let mut time = parts[4].split(':');
    let (h, m, s): (i64, i64, i64) = (
        time.next()?.parse().ok()?,
        time.next()?.parse().ok()?,
        time.next()?.parse().ok()?,
    );
    // Days from civil (Howard Hinnant's algorithm).
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some((days * 86_400 + h * 3600 + m * 60 + s) as f64)
}

fn random_unit() -> f64 {
    (uuid::Uuid::new_v4().as_u128() >> 75) as f64 / (1u128 << 53) as f64
}

/// Wait before retry number `attempt` (Python: `_default_retry_delay`).
pub(crate) fn default_retry_delay(attempt: u32, backoff: Option<ModelRetryBackoffSettings>) -> f64 {
    let b = backoff.unwrap_or_default();
    let initial = b.initial_delay.unwrap_or(DEFAULT_INITIAL_DELAY_SECONDS);
    let max = b.max_delay.unwrap_or(DEFAULT_MAX_DELAY_SECONDS);
    let multiplier = b.multiplier.unwrap_or(DEFAULT_BACKOFF_MULTIPLIER);
    let jitter = b.jitter.unwrap_or(DEFAULT_BACKOFF_JITTER);
    let base = (initial * multiplier.powi(attempt.saturating_sub(1) as i32)).min(max);
    if !jitter {
        return base;
    }
    (base * (0.875 + random_unit() * 0.25)).clamp(0.0, max)
}

/// Add the usage of failed attempts to a successful response (Python: `apply_retry_attempt_usage`):
/// each failed attempt counts as a request with a zero-token entry.
pub(crate) fn apply_retry_attempt_usage(usage: &mut Usage, failed_attempts: u32) {
    if failed_attempts == 0 {
        return;
    }
    let mut entries = std::mem::take(&mut usage.request_usage_entries);
    if entries.is_empty() {
        entries.push(RequestUsage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
            input_tokens_details: usage.input_tokens_details,
            output_tokens_details: usage.output_tokens_details,
        });
    }
    usage.requests = usage.requests.max(1) + u64::from(failed_attempts);
    let mut all = vec![RequestUsage::default(); failed_attempts as usize];
    all.extend(entries);
    usage.request_usage_entries = all;
}

fn is_conversation_locked(error: &ModelError) -> bool {
    matches!(error, ModelError::Status(e) if e.status_code == 400 && e.error_code() == Some("conversation_locked"))
}

/// Python keeps replaying `conversation_locked` unless the caller sets `max_retries = 0`.
fn preserves_conversation_locked_compat(settings: Option<&ModelRetrySettings>) -> bool {
    settings.is_none_or(|s| s.max_retries.is_none_or(|n| n > 0))
}

/// Everything the retry loop needs to know about one model call.
pub(crate) struct RetryCall<'a> {
    pub settings: Option<&'a ModelRetrySettings>,
    pub previous_response_id: Option<&'a str>,
    pub conversation_id: Option<&'a str>,
    /// Per-attempt timeout in seconds (`ModelSettings.timeout`).
    pub timeout: Option<f32>,
    pub stream: bool,
    /// Set by a streaming attempt once it forwarded an event that makes a replay unsafe.
    pub emitted_unsafe_event: Option<&'a AtomicBool>,
}

/// Decide whether `error` is retried (Python: `_evaluate_retry`).
async fn evaluate_retry(
    call: &RetryCall<'_>,
    error: &ModelError,
    attempt: u32,
    provider_advice: Option<ModelRetryAdvice>,
) -> RetryDecision {
    let max_retries = call.settings.and_then(|s| s.max_retries).unwrap_or(0);
    if attempt > max_retries {
        return RetryDecision::no();
    }

    let mut normalized = ModelRetryNormalizedError::from_error(error);
    if let Some(advice) = &provider_advice {
        if let Some(after) = advice.retry_after {
            normalized.retry_after = Some(after);
        }
        if let Some(over) = &advice.normalized {
            normalized.apply_override(over);
        }
    }
    let context = RetryPolicyContext {
        error: error.clone(),
        attempt,
        max_retries,
        stream: call.stream,
        normalized: normalized.clone(),
        provider_advice: provider_advice.clone(),
        previous_response_id: call.previous_response_id.map(str::to_string),
        conversation_id: call.conversation_id.map(str::to_string),
    };
    let advice_reason = provider_advice.as_ref().and_then(|a| a.reason.clone());
    let marks_unsafe = context.replay_safety() == Some(ReplaySafety::Unsafe);
    let marks_safe = context.replay_safety() == Some(ReplaySafety::Safe);
    let emitted = call
        .emitted_unsafe_event
        .is_some_and(|flag| flag.load(Ordering::SeqCst));

    // Aborts and failures after user-visible streamed output are absolute vetoes.
    if normalized.is_abort || emitted {
        return RetryDecision::no().with_reason_opt(advice_reason);
    }
    // A provider-unsafe streamed failure stays blocked before the policy runs.
    if marks_unsafe && call.stream {
        return RetryDecision::no().with_reason_opt(advice_reason);
    }

    let Some(policy) = call.settings.and_then(|s| s.policy.as_ref()) else {
        return RetryDecision::no();
    };
    let decision = policy.evaluate(context.clone()).await;
    if !decision.retry {
        return decision;
    }

    let reason = decision.reason.clone().or(advice_reason.clone());
    // A stateful request fails closed: the follow-up depends on server-side state, so only a
    // provider-owned approval, or the application's approval of a provider-unsafe failure,
    // lets it be replayed.
    if context.stateful_request()
        && !(decision.approves_replay
            || marks_safe
            || (decision.approve_unsafe_replay && marks_unsafe))
    {
        return RetryDecision::no().with_reason_opt(reason);
    }
    // Provider-marked replay unsafety needs an explicit approval.
    if marks_unsafe && !(decision.approves_replay || decision.approve_unsafe_replay) {
        return RetryDecision::no().with_reason_opt(reason);
    }

    let delay = decision.delay.unwrap_or_else(|| {
        normalized
            .retry_after
            .unwrap_or_else(|| default_retry_delay(attempt, call.settings.and_then(|s| s.backoff)))
    });
    RetryDecision::yes()
        .with_delay(delay)
        .with_reason_opt(reason)
}

impl RetryDecision {
    fn with_reason_opt(mut self, reason: Option<String>) -> Self {
        if reason.is_some() {
            self.reason = reason;
        }
        self
    }
}

async fn sleep_for_retry(seconds: f64) {
    if seconds > 0.0 {
        tokio::time::sleep(Duration::from_secs_f64(seconds)).await;
    }
}

/// Run one attempt under the per-attempt timeout (Python: `_await_model_attempt`).
async fn await_model_attempt<Fut>(
    attempt: Fut,
    timeout: Option<f32>,
) -> Result<ModelResponse, ModelError>
where
    Fut: Future<Output = Result<ModelResponse, ModelError>>,
{
    match timeout {
        Some(seconds) if seconds > 0.0 => {
            match tokio::time::timeout(Duration::from_secs_f32(seconds), attempt).await {
                Ok(result) => result,
                Err(_) => Err(ModelError::Timeout(crate::error::ModelTimeoutError {
                    timeout_seconds: f64::from(seconds),
                })),
            }
        }
        _ => attempt.await,
    }
}

/// Call a model, replaying failed attempts according to the retry settings
/// (Python: `get_response_with_retry` / `stream_response_with_retry`).
///
/// `attempt` starts one fresh request each time it is called; `advice` asks the adapter for
/// provider guidance about a failure.
pub(crate) async fn call_with_retry<F, Fut, A>(
    call: RetryCall<'_>,
    advice: A,
    mut attempt: F,
) -> Result<ModelResponse, ModelError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<ModelResponse, ModelError>>,
    A: Fn(&ModelRetryAdviceRequest) -> Option<ModelRetryAdvice>,
{
    let mut policy_attempt: u32 = 1;
    let mut failed_policy_attempts: u32 = 0;
    let mut compatibility_retries: u32 = 0;

    loop {
        if let Some(flag) = call.emitted_unsafe_event {
            flag.store(false, Ordering::SeqCst);
        }
        match await_model_attempt(attempt(), call.timeout).await {
            Ok(mut response) => {
                apply_retry_attempt_usage(
                    &mut response.usage,
                    failed_policy_attempts + compatibility_retries,
                );
                return Ok(response);
            }
            Err(error) => {
                if is_conversation_locked(&error)
                    && preserves_conversation_locked_compat(call.settings)
                    && compatibility_retries < COMPATIBILITY_CONVERSATION_LOCKED_RETRIES
                {
                    compatibility_retries += 1;
                    sleep_for_retry(f64::from(1u32 << (compatibility_retries - 1))).await;
                    continue;
                }
                let provider_advice = advice(&ModelRetryAdviceRequest {
                    error: error.clone(),
                    attempt: policy_attempt,
                    stream: call.stream,
                    previous_response_id: call.previous_response_id.map(str::to_string),
                    conversation_id: call.conversation_id.map(str::to_string),
                });
                let decision = evaluate_retry(&call, &error, policy_attempt, provider_advice).await;
                if !decision.retry {
                    return Err(error);
                }
                ::tracing::debug!(
                    "Retrying failed model request in {:?}s (attempt {policy_attempt})",
                    decision.delay
                );
                sleep_for_retry(decision.delay.unwrap_or(0.0)).await;
                policy_attempt += 1;
                failed_policy_attempts += 1;
            }
        }
    }
}

/// OpenAI's retry advice for a failure (Python: `get_openai_retry_advice`).
pub(crate) fn openai_retry_advice(request: &ModelRetryAdviceRequest) -> Option<ModelRetryAdvice> {
    let error = &request.error;
    let (retry_after, normalized) = match error {
        ModelError::Status(e) => {
            let after = e.retry_after();
            (
                after,
                ModelRetryNormalizedError::new()
                    .status_code(Some(e.status_code))
                    .error_code(e.error_code().map(str::to_string))
                    .message(Some(error.to_string()))
                    .request_id(e.request_id().map(str::to_string))
                    .retry_after(after)
                    .is_abort(false)
                    .is_network_error(false)
                    .is_timeout(false),
            )
        }
        ModelError::Connection(e) => (
            None,
            ModelRetryNormalizedError::new()
                .status_code(None)
                .message(Some(error.to_string()))
                .retry_after(None)
                .is_abort(false)
                .is_network_error(!e.is_timeout)
                .is_timeout(e.is_timeout),
        ),
        _ => return None,
    };
    let reason = Some(error.to_string());

    if let ModelError::Status(e) = error {
        match e
            .header("x-should-retry")
            .map(|h| h.trim().to_lowercase())
            .as_deref()
        {
            Some("true") => {
                return Some(ModelRetryAdvice {
                    suggested: Some(true),
                    retry_after,
                    replay_safety: Some(ReplaySafety::Safe),
                    reason,
                    normalized: Some(normalized),
                })
            }
            Some("false") => {
                return Some(ModelRetryAdvice {
                    suggested: Some(false),
                    retry_after,
                    replay_safety: None,
                    reason,
                    normalized: Some(normalized),
                })
            }
            _ => {}
        }
    }
    if normalized.is_network_error || normalized.is_timeout {
        return Some(ModelRetryAdvice {
            suggested: Some(true),
            retry_after,
            replay_safety: None,
            reason,
            normalized: Some(normalized),
        });
    }
    let status = normalized.status_code;
    if matches!(status, Some(408 | 409 | 429)) || status.is_some_and(|s| s >= 500) {
        let stateful = request.previous_response_id.is_some() || request.conversation_id.is_some();
        return Some(ModelRetryAdvice {
            suggested: Some(true),
            retry_after,
            replay_safety: stateful.then_some(ReplaySafety::Safe),
            reason,
            normalized: Some(normalized),
        });
    }
    retry_after.map(|after| ModelRetryAdvice {
        suggested: None,
        retry_after: Some(after),
        replay_safety: None,
        reason,
        normalized: Some(normalized),
    })
}
