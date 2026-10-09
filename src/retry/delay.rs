//! Backoff and usage bookkeeping between attempts.

use crate::usage::Usage;

use super::{
    ModelRetryBackoffSettings, DEFAULT_BACKOFF_JITTER, DEFAULT_BACKOFF_MULTIPLIER,
    DEFAULT_INITIAL_DELAY_SECONDS, DEFAULT_MAX_DELAY_SECONDS,
};
use crate::usage::RequestUsage;

/// `retry-after-ms` (milliseconds) then `retry-after` (seconds or an HTTP date), as seconds
/// (Python: `get_retry_after`).
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
