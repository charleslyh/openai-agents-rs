//! The retry executor: how a model call is replayed after a failure.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::error::ModelError;
use crate::items::ModelResponse;

use super::{
    apply_retry_attempt_usage, default_retry_delay, ModelRetryAdvice, ModelRetryAdviceRequest,
    ModelRetryNormalizedError, ModelRetrySettings, ReplaySafety, RetryDecision, RetryPolicyContext,
    COMPATIBILITY_CONVERSATION_LOCKED_RETRIES,
};

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
pub(crate) async fn evaluate_retry(
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
pub(crate) async fn await_model_attempt<Fut>(
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
