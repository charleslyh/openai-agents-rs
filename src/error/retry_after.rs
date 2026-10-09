//! Turning a provider's `Retry-After` headers into seconds.
//!
//! Lives in `error` because [`super::ModelStatusError::retry_after`] is the only caller, and
//! keeping it here lets the error module stay a leaf (Python: `get_retry_after`).

/// `retry-after-ms` (milliseconds) then `retry-after` (seconds or an HTTP date), as seconds.
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
