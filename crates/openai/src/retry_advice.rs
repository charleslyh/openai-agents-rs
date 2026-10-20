//! OpenAI-specific retry advice (Python: `get_openai_retry_advice`).

use opai_agents_core::error::ModelError;
use opai_agents_core::retry::{
    ModelRetryAdvice, ModelRetryAdviceRequest, ModelRetryNormalizedError, ReplaySafety,
};

/// OpenAI's retry advice for a failure (Python: `get_openai_retry_advice`).
pub fn openai_retry_advice(request: &ModelRetryAdviceRequest) -> Option<ModelRetryAdvice> {
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
