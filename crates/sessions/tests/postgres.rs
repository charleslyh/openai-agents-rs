//! The Postgres backend, against a real server:
//! `DATABASE_URL=postgres://postgres@localhost/agents cargo test`.
//!
//! Without `DATABASE_URL` these are skipped rather than failed: a backend that talks to a server
//! cannot be checked without one, and a repository test run should not need to start one. The
//! suite is the same [`session_contract`] every backend runs, plus a paused run in a row.
#![cfg(feature = "postgres")]

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use opai_agents_core::{
    session_contract, RunState, RunStateStore, Session, RUN_STATE_SCHEMA_VERSION,
};
use opai_agents_sessions::{PostgresRunStateStore, PostgresSession};
use serde_json::json;

/// The table the session tests use; each run drops it first.
const TABLE: &str = "agents_sessions_test";
/// The table the paused-run tests use.
const STATE_TABLE: &str = "agents_run_states_test";

fn url() -> Option<String> {
    std::env::var("DATABASE_URL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn unique(name: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("{name}-{nanos}")
}

#[tokio::test]
async fn postgres_passes_the_session_contract() {
    let Some(url) = url() else {
        eprintln!("DATABASE_URL is not set: skipping the Postgres session contract");
        return;
    };
    let session = PostgresSession::connect(&url, unique("contract"), TABLE)
        .await
        .expect("open");
    session.create_table().await.expect("create the table");
    session_contract::assert_session_contract(&session).await;
}

#[tokio::test]
async fn postgres_keeps_every_concurrent_write() {
    let Some(url) = url() else {
        eprintln!("DATABASE_URL is not set: skipping");
        return;
    };
    let session = PostgresSession::connect(&url, unique("concurrent"), TABLE)
        .await
        .expect("open");
    session.create_table().await.expect("create the table");
    let session: Arc<dyn Session> = Arc::new(session);
    session_contract::assert_concurrent_writes(session, 16).await;
}

/// Two sessions share a table without seeing each other's history.
#[tokio::test]
async fn sessions_in_one_table_are_separate() {
    let Some(url) = url() else {
        eprintln!("DATABASE_URL is not set: skipping");
        return;
    };
    let one = PostgresSession::connect(&url, unique("one"), TABLE)
        .await
        .expect("open");
    one.create_table().await.expect("create the table");
    let two = PostgresSession::connect(&url, unique("two"), TABLE)
        .await
        .expect("open");
    one.add_items(vec![
        json!({"type": "message", "role": "user", "content": "for one"}),
    ])
    .await
    .expect("add");
    assert_eq!(one.get_items(None).await.expect("read").len(), 1);
    assert!(
        two.get_items(None).await.expect("read").is_empty(),
        "another session's history must not appear"
    );
    one.clear_session().await.expect("clear");
    two.clear_session().await.expect("clear");
}

/// A paused run written by one connection is read by another: what resuming in a new process is.
#[tokio::test]
async fn a_paused_run_survives_a_reconnect() {
    let Some(url) = url() else {
        eprintln!("DATABASE_URL is not set: skipping");
        return;
    };
    let store = PostgresRunStateStore::unsigned(&url, STATE_TABLE)
        .await
        .expect("open");
    store.create_table().await.expect("create the table");
    let key = unique("run");
    assert!(store.load(&key).await.expect("load").is_none());

    let state = state();
    store.save(&key, &state, None).await.expect("save");
    // Saving again replaces the row rather than failing on the primary key.
    store.save(&key, &state, None).await.expect("save again");

    let again = PostgresRunStateStore::unsigned(&url, STATE_TABLE)
        .await
        .expect("open again");
    let loaded = again.load(&key).await.expect("load").expect("a state");
    assert_eq!(loaded.starting_agent_name, state.starting_agent_name);

    again.delete(&key).await.expect("delete");
    assert!(store.load(&key).await.expect("load").is_none());
    again.delete(&key).await.expect("deleting twice is fine");
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
