//! The Redis backend, against a real server: `REDIS_URL=redis://127.0.0.1:6379 cargo test`.
//!
//! Without `REDIS_URL` these are skipped rather than failed: a backend that talks to a server
//! cannot be checked without one, and a repository test run should not need to start one. The
//! suite is the same [`session_contract`] every backend runs, plus what is specific to Redis — a
//! TTL, and a paused run that is still there after a reconnect.
#![cfg(feature = "redis")]

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use opai_agents_core::{
    session_contract, RunState, RunStateStore, Session, RUN_STATE_SCHEMA_VERSION,
};
use opai_agents_sessions::{RedisRunStateStore, RedisSession};
use serde_json::json;

fn url() -> Option<String> {
    std::env::var("REDIS_URL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// A session id of its own, so a test does not read what another left behind.
fn unique(name: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("{name}-{nanos}")
}

#[tokio::test]
async fn redis_passes_the_session_contract() {
    let Some(url) = url() else {
        eprintln!("REDIS_URL is not set: skipping the Redis session contract");
        return;
    };
    let session = RedisSession::open(&url, unique("contract"))
        .await
        .expect("open");
    session_contract::assert_session_contract(&session).await;
}

#[tokio::test]
async fn redis_keeps_every_concurrent_write() {
    let Some(url) = url() else {
        eprintln!("REDIS_URL is not set: skipping");
        return;
    };
    let session: Arc<dyn Session> = Arc::new(
        RedisSession::open(&url, unique("concurrent"))
            .await
            .expect("open"),
    );
    session_contract::assert_concurrent_writes(session, 16).await;
}

#[tokio::test]
async fn a_history_with_a_ttl_is_still_readable_until_it_expires() {
    let Some(url) = url() else {
        eprintln!("REDIS_URL is not set: skipping");
        return;
    };
    let session = RedisSession::open(&url, unique("ttl"))
        .await
        .expect("open")
        .with_ttl(Duration::from_secs(60));
    session
        .add_items(vec![
            json!({"type": "message", "role": "user", "content": "hi"}),
        ])
        .await
        .expect("add");
    assert_eq!(session.get_items(None).await.expect("read").len(), 1);
    session.clear_session().await.expect("clear");
}

/// A paused run written by one connection is read by another: what resuming in a new process is.
#[tokio::test]
async fn a_paused_run_survives_a_reconnect() {
    let Some(url) = url() else {
        eprintln!("REDIS_URL is not set: skipping");
        return;
    };
    let key = unique("run");
    let store = RedisRunStateStore::unsigned(&url).await.expect("open");
    assert!(store.load(&key).await.expect("load").is_none());

    let state = state();
    store.save(&key, &state, None).await.expect("save");

    let again = RedisRunStateStore::unsigned(&url)
        .await
        .expect("open again");
    let loaded = again.load(&key).await.expect("load").expect("a state");
    assert_eq!(loaded.starting_agent_name, state.starting_agent_name);

    again.delete(&key).await.expect("delete");
    assert!(store.load(&key).await.expect("load").is_none());
    again.delete(&key).await.expect("deleting twice is fine");
}

/// A state edited in Redis — the server is writable by other things too — is refused, not resumed.
#[tokio::test]
async fn a_signed_store_refuses_a_state_that_was_changed() {
    let Some(url) = url() else {
        eprintln!("REDIS_URL is not set: skipping");
        return;
    };
    // One test owns the variable: `cargo test` runs tests of a binary in parallel threads.
    std::env::set_var("AGENTS_STATE_KEY", "s3cret");
    let store = RedisRunStateStore::from_env(&url).await.expect("open");
    std::env::remove_var("AGENTS_STATE_KEY");
    assert!(
        store.is_signed(),
        "the key is picked up from the environment"
    );

    let key = unique("signed");
    store.save(&key, &state(), None).await.expect("save");

    // Change the state the way someone with access to the server could.
    let mut conn = RedisSession::open(&url, "unused")
        .await
        .expect("open")
        .connection();
    let raw: Option<String> = redis_cmd_get(&mut conn, &format!("agents:state:{key}")).await;
    let mut envelope: serde_json::Value =
        serde_json::from_str(&raw.expect("a stored state")).expect("json");
    envelope["state"]["current_agent_name"] = json!("someone-else");
    redis_cmd_set(
        &mut conn,
        &format!("agents:state:{key}"),
        &serde_json::to_string(&envelope).expect("json"),
    )
    .await;

    let error = store
        .load(&key)
        .await
        .expect_err("the signature no longer matches");
    assert!(
        error.to_string().contains("changed after it was stored"),
        "{error}"
    );
    store.delete(&key).await.expect("delete");
}

fn state() -> RunState {
    RunState::from_json(
        "ops",
        json!({
            "$schemaVersion": RUN_STATE_SCHEMA_VERSION,
            "starting_agent_name": "ops",
            "input": {"type": "text", "text": "ship it"},
            "pending_response": {"output": [], "usage": {}},
        }),
    )
    .expect("a minimal state")
}

async fn redis_cmd_get(conn: &mut redis::aio::MultiplexedConnection, key: &str) -> Option<String> {
    redis::cmd("GET")
        .arg(key)
        .query_async(conn)
        .await
        .expect("get")
}

async fn redis_cmd_set(conn: &mut redis::aio::MultiplexedConnection, key: &str, value: &str) {
    redis::cmd("SET")
        .arg(key)
        .arg(value)
        .query_async::<()>(conn)
        .await
        .expect("set");
}
