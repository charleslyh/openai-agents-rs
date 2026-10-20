//! What time it is, in the shape the model asked for (Rust-only, X-019).
//!
//! A model has no clock of its own, and "today", "next Friday" and "how long until" are questions
//! it is asked constantly. This answers them from the system clock, in UTC or in a named offset,
//! and always as an RFC 3339 string — one shape, so the model does not have to guess which of
//! three formats it got.
//!
//! The offset is a fixed `+HH:MM` / `-HH:MM` / `Z` and not a time zone name: named zones need a
//! database of their own, and a tool that answers "what time is it" has no business pulling one in.

use std::time::{SystemTime, UNIX_EPOCH};

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use opai_agents_core::error::UserError;
use opai_agents_core::{AgentsError, FunctionTool};

/// Arguments of the `clock` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ClockArgs {
    /// Offset from UTC as `+HH:MM`, `-HH:MM`, `Z` or `+HH`. Leave it out for UTC.
    #[serde(default)]
    pub offset: Option<String>,
    /// A whole number of seconds to add, for "in two hours" or "an hour ago".
    #[serde(default)]
    pub add_seconds: Option<i64>,
}

/// The `clock` tool: no arguments needed, so an empty object works.
pub fn clock() -> FunctionTool {
    FunctionTool::typed(
        "clock",
        "What time it is now, as an RFC 3339 timestamp. Optional: `offset` (`+02:00`, `-05:30`, \
         `Z`) and `add_seconds` (a whole number of seconds to add).",
        |_ctx, args: ClockArgs| async move {
            let now = now()?;
            let shifted = match args.add_seconds {
                Some(seconds) => shift(now, seconds)?,
                None => now,
            };
            Ok(json!({
                "now": format_rfc3339(shifted, parse_offset(args.offset.as_deref())?),
                "unix_seconds": shifted,
            }))
        },
    )
}

/// Seconds since the epoch.
fn now() -> Result<i64, AgentsError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .map_err(|_| AgentsError::from(UserError::new("clock: the system clock is before 1970")))
}

fn shift(unix_seconds: i64, by: i64) -> Result<i64, AgentsError> {
    unix_seconds.checked_add(by).ok_or_else(|| {
        AgentsError::from(UserError::new("clock: that shift overflows the timestamp"))
    })
}

/// `+HH:MM` / `-HH:MM` / `+HH` / `Z` / none, as seconds east of UTC.
pub fn parse_offset(offset: Option<&str>) -> Result<i32, AgentsError> {
    let Some(offset) = offset else { return Ok(0) };
    let text = offset.trim();
    if text.is_empty() || text.eq_ignore_ascii_case("z") {
        return Ok(0);
    }
    let (sign, rest) = match text.as_bytes().first() {
        Some(b'+') => (1, &text[1..]),
        Some(b'-') => (-1, &text[1..]),
        _ => {
            return Err(AgentsError::from(UserError::new(format!(
                "clock: {offset:?} is not an offset: use +HH:MM, -HH:MM or Z"
            ))))
        }
    };
    let (hours, minutes) = match rest.split_once(':') {
        Some((h, m)) => (h, m),
        None => (rest, "00"),
    };
    let hours: i32 = hours.parse().map_err(|_| bad_offset(offset))?;
    let minutes: i32 = minutes.parse().map_err(|_| bad_offset(offset))?;
    if !(0..=23).contains(&hours) || !(0..=59).contains(&minutes) {
        return Err(bad_offset(offset));
    }
    Ok(sign * (hours * 3600 + minutes * 60))
}

fn bad_offset(offset: &str) -> AgentsError {
    AgentsError::from(UserError::new(format!(
        "clock: {offset:?} is not an offset: use +HH:MM, -HH:MM or Z"
    )))
}

/// `unix_seconds` as `YYYY-MM-DDThh:mm:ss+HH:MM`, the one shape this tool answers in.
pub fn format_rfc3339(unix_seconds: i64, offset_seconds: i32) -> String {
    let shifted = unix_seconds + i64::from(offset_seconds);
    let days = shifted.div_euclid(86_400);
    let seconds_of_day = shifted.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3600;
    let minute = seconds_of_day % 3600 / 60;
    let second = seconds_of_day % 60;
    let (offset_hours, offset_minutes) = (offset_seconds / 3600, offset_seconds % 3600 / 60);
    if offset_seconds == 0 {
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
    } else {
        format!(
            "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}{:+03}:{:02}",
            offset_hours,
            offset_minutes.abs()
        )
    }
}

/// Days since the epoch as a calendar date (Howard Hinnant's algorithm, valid for the whole range
/// an `i64` timestamp can reach).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_are_read() {
        assert_eq!(parse_offset(None).unwrap(), 0);
        assert_eq!(parse_offset(Some("Z")).unwrap(), 0);
        assert_eq!(parse_offset(Some("+02:00")).unwrap(), 7200);
        assert_eq!(parse_offset(Some("-05:30")).unwrap(), -19_800);
        assert_eq!(parse_offset(Some("+02")).unwrap(), 7200);
        for bad in ["2:00", "+99:00", "+02:99", "Europe/Berlin", "+aa:bb"] {
            assert!(parse_offset(Some(bad)).is_err(), "`{bad}` must be refused");
        }
    }

    #[test]
    fn the_format_is_one_shape() {
        // 2024-02-29, the leap day, at 11:34:56 UTC.
        assert_eq!(format_rfc3339(1_709_206_496, 0), "2024-02-29T11:34:56Z");
        assert_eq!(
            format_rfc3339(1_709_206_496, 7200),
            "2024-02-29T13:34:56+02:00"
        );
        assert_eq!(
            format_rfc3339(1_709_206_496, -19_800),
            "2024-02-29T06:04:56-05:30"
        );
    }

    #[test]
    fn a_shift_that_overflows_is_an_error() {
        assert!(shift(i64::MAX, 1).is_err());
        assert_eq!(shift(100, -1).unwrap(), 99);
    }
}
