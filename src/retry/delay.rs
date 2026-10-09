//! Backoff and usage bookkeeping between attempts.

use crate::usage::Usage;

use super::{
    ModelRetryBackoffSettings, DEFAULT_BACKOFF_JITTER, DEFAULT_BACKOFF_MULTIPLIER,
    DEFAULT_INITIAL_DELAY_SECONDS, DEFAULT_MAX_DELAY_SECONDS,
};
use crate::usage::RequestUsage;

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
