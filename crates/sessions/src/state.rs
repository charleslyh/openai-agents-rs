//! What a paused run looks like in storage, shared by every backend (Rust-only, X-016).
//!
//! A run that stops for an approval may outlive the process, so what comes back from a file, from
//! Redis or from Postgres is data from outside: it may have been edited, it may have been written
//! by an older build, and it may have outlived its time.
//!
//! One decides that here, once, rather than per backend, so no backend can forget it:
//!
//! * **Integrity.** With a signing key configured, every state carries an HMAC-SHA256 of its own
//!   JSON. A state whose signature does not match fails to load instead of resuming — an approval
//!   edited into a different call must never be honoured. The comparison takes the same time
//!   whatever the first byte is.
//! * **Schema.** [`decode`] reads the version alongside the state, and [`to_state`] runs the
//!   migration hook before checking it.
//! * **Expiry.** [`decode`] reports an expired state as absent, and the backend deletes it.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use serde_json::{json, Value};
use sha2::Sha256;

use opai_agents_core::error::UserError;
use opai_agents_core::{
    finish_load, AgentsError, RunState, RunStateMigration, RUN_STATE_SCHEMA_VERSION,
};

type HmacSha256 = Hmac<Sha256>;

/// Read the signing key from `env`, warning once when it is not set.
///
/// Without a key a store still works, but a state edited outside the process would be resumed, so
/// that is said out loud.
pub(crate) fn key_from_env(env: &str) -> Option<Vec<u8>> {
    let key = std::env::var(env)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .map(String::into_bytes);
    if key.is_none() {
        tracing::warn!(
            "{env} is not set: paused run states are stored unsigned, so one edited outside this \
             process would be resumed"
        );
    }
    key
}

/// The stored form of `state`: version, state, optional expiry, optional signature.
pub(crate) fn wrap(
    state: &RunState,
    ttl: Option<Duration>,
    signing_key: Option<&[u8]>,
) -> Result<Value, AgentsError> {
    let state_json = state.to_json();
    // Signed over the exact bytes that are read back, so the signature can be checked before
    // anything is parsed.
    let canonical = serde_json::to_string(&state_json)
        .map_err(|e| UserError::new(format!("cannot serialize a paused run: {e}")))?;
    let mut envelope = json!({
        "schema_version": RUN_STATE_SCHEMA_VERSION,
        "state": state_json,
    });
    if let Some(ttl) = ttl {
        envelope["expires_at"] = json!(now_seconds().saturating_add(ttl.as_secs()));
    }
    if let Some(key) = signing_key {
        envelope["mac"] = json!(sign(key, &canonical));
    }
    Ok(envelope)
}

/// The state inside `envelope`, or `None` when it has expired (the backend deletes it then).
///
/// `label` says where the state came from, so the error names the file, the key or the row.
pub(crate) fn decode(
    envelope: &Value,
    signing_key: Option<&[u8]>,
    label: &str,
) -> Result<Option<Value>, AgentsError> {
    if let Some(expires_at) = envelope.get("expires_at").and_then(Value::as_u64) {
        if now_seconds() >= expires_at {
            return Ok(None);
        }
    }
    let state_json = envelope.get("state").cloned().unwrap_or(Value::Null);
    let canonical = serde_json::to_string(&state_json)
        .map_err(|e| UserError::new(format!("cannot read the stored run: {e}")))?;
    match (signing_key, envelope.get("mac").and_then(Value::as_str)) {
        (Some(key), Some(mac)) if !verify(key, &canonical, mac) => Err(UserError::new(format!(
            "{label} was changed after it was stored: the signature does not match. Delete it and \
             start the run again"
        ))
        .into()),
        // Signed store, unsigned state: someone removed the signature.
        (Some(_), None) => Err(UserError::new(format!(
            "{label} carries no signature, but a signing key is set. Delete it and start the run \
             again"
        ))
        .into()),
        _ => Ok(Some(state_json)),
    }
}

/// Turn stored JSON into a [`RunState`], running `migrate` first and checking the version after.
pub(crate) fn to_state(
    state_json: Value,
    migrate: Option<&RunStateMigration>,
) -> Result<RunState, AgentsError> {
    let starting_agent = state_json
        .get("starting_agent_name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let state = RunState::from_json(&starting_agent, state_json)?;
    finish_load(state, migrate)
}

/// Refuse an id or key that would be more than a name: no separators (they are path or key
/// separators somewhere), no control characters, no leading dot, and not unbounded.
///
/// Only the server-backed stores use it: the file stores have their own, stricter check that also
/// has to produce a file name.
#[cfg(any(feature = "redis", feature = "postgres", test))]
pub(crate) fn validate_id(id: &str, what: &str) -> Result<(), AgentsError> {
    if id.is_empty() {
        return Err(UserError::new(format!("the {what} must not be empty")).into());
    }
    if id.len() > 200 {
        return Err(UserError::new(format!("the {what} is longer than 200 characters")).into());
    }
    if id.starts_with('.') {
        return Err(UserError::new(format!("the {what} must not start with a dot")).into());
    }
    if id.chars().any(|c| c.is_control() || c == '/' || c == '\\') {
        return Err(UserError::new(format!(
            "the {what} must not contain a separator or a control character"
        ))
        .into());
    }
    Ok(())
}

pub(crate) fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// HMAC-SHA256 of `payload`, as lower-case hex.
fn sign(key: &[u8], payload: &str) -> String {
    let mut mac = match HmacSha256::new_from_slice(key) {
        Ok(mac) => mac,
        Err(_) => return String::new(),
    };
    mac.update(payload.as_bytes());
    hex(mac.finalize().into_bytes().as_slice())
}

/// Compare without letting the first differing byte change how long it takes.
fn verify(key: &[u8], payload: &str, mac: &str) -> bool {
    let expected = sign(key, payload);
    constant_time_eq(expected.as_bytes(), mac.as_bytes())
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .fold(0u8, |acc, (a, b)| acc | (a ^ b))
        == 0
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0xf) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signing_is_stable_and_comparison_is_exact() {
        assert_eq!(sign(b"k", "payload"), sign(b"k", "payload"));
        assert_ne!(sign(b"k", "payload"), sign(b"k", "payload2"));
        assert!(verify(b"k", "payload", &sign(b"k", "payload")));
        assert!(!verify(b"k", "payload", &sign(b"k", "other")));
        assert!(!verify(b"k", "payload", "short"));
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
    }

    #[test]
    fn an_id_that_is_more_than_a_name_is_refused() {
        assert!(validate_id("ops", "session id").is_ok());
        for bad in ["", "a/b", "..", ".hidden", "a\u{1}b"] {
            assert!(validate_id(bad, "session id").is_err(), "`{bad}` must fail");
        }
    }
}
