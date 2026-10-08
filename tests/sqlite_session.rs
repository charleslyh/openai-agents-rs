//! `SqliteSession` (feature `sqlite`): storage semantics, persistence, runner integration and
//! file compatibility with the Python SDK's `SQLiteSession`.
#![cfg(feature = "sqlite")]

use std::path::PathBuf;
use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{Agent, AgentsError, RunOptions, Runner, Session, SqliteSession};
use serde_json::{json, Value};

/// A database path in the temp dir, removed (with its WAL files) on drop.
struct TempDb(PathBuf);

impl TempDb {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!("agents-rs-{}.db", uuid::Uuid::new_v4())))
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", self.0.display()));
        }
    }
}

fn msg(role: &str, text: &str) -> Value {
    json!({"role": role, "content": text})
}

#[tokio::test]
async fn stores_reads_limits_pops_and_clears() {
    let session = SqliteSession::in_memory("s1").expect("open");
    assert_eq!(session.session_id(), "s1");
    assert!(session.get_items(None).await.unwrap().is_empty());
    assert!(session.pop_item().await.unwrap().is_none());

    session.add_items(vec![msg("user", "a"), msg("assistant", "b")]).await.unwrap();
    session.add_items(vec![msg("user", "c")]).await.unwrap();
    session.add_items(vec![]).await.unwrap();

    let all = session.get_items(None).await.unwrap();
    assert_eq!(all.iter().map(|i| i["content"].as_str().unwrap()).collect::<Vec<_>>(), ["a", "b", "c"]);
    let latest = session.get_items(Some(2)).await.unwrap();
    assert_eq!(latest.iter().map(|i| i["content"].as_str().unwrap()).collect::<Vec<_>>(), ["b", "c"]);
    assert!(session.get_items(Some(0)).await.unwrap().is_empty());
    assert_eq!(session.get_items(Some(10)).await.unwrap().len(), 3);

    assert_eq!(session.pop_item().await.unwrap(), Some(msg("user", "c")));
    assert_eq!(session.get_items(None).await.unwrap().len(), 2);
    session.clear_session().await.unwrap();
    assert!(session.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn items_persist_and_sessions_in_one_file_are_isolated() {
    let db = TempDb::new();
    {
        let a = SqliteSession::open("a", &db.0).unwrap();
        let b = SqliteSession::open("b", &db.0).unwrap();
        a.add_items(vec![msg("user", "from a")]).await.unwrap();
        b.add_items(vec![msg("user", "from b")]).await.unwrap();
    }
    let a = SqliteSession::open("a", &db.0).unwrap();
    assert_eq!(a.get_items(None).await.unwrap(), vec![msg("user", "from a")]);
    a.clear_session().await.unwrap();
    let b = SqliteSession::open("b", &db.0).unwrap();
    assert_eq!(b.get_items(None).await.unwrap(), vec![msg("user", "from b")], "clear is per session");
}

#[tokio::test]
async fn unreadable_rows_are_skipped_and_do_not_count_toward_the_limit() {
    let db = TempDb::new();
    let session = SqliteSession::open("s", &db.0).unwrap();
    session.add_items(vec![msg("user", "a"), msg("user", "b")]).await.unwrap();
    let raw = rusqlite::Connection::open(&db.0).unwrap();
    raw.execute("INSERT INTO agent_messages (session_id, message_data) VALUES ('s', 'not json')", [])
        .unwrap();

    assert_eq!(session.get_items(None).await.unwrap().len(), 2);
    let latest = session.get_items(Some(2)).await.unwrap();
    assert_eq!(latest, vec![msg("user", "a"), msg("user", "b")], "the corrupt tail row is not an item");
    assert_eq!(session.pop_item().await.unwrap(), Some(msg("user", "b")), "pop drops the corrupt row first");
}

#[tokio::test]
async fn table_names_are_validated_and_can_be_customized() {
    let db = TempDb::new();
    let err = SqliteSession::open_with_tables("s", &db.0, "bad name; DROP", "m").unwrap_err();
    assert!(matches!(err, AgentsError::User(_)), "{err}");

    let custom = SqliteSession::open_with_tables("s", &db.0, "my_sessions", "my_messages").unwrap();
    custom.add_items(vec![msg("user", "x")]).await.unwrap();
    let raw = rusqlite::Connection::open(&db.0).unwrap();
    let count: i64 = raw.query_row("SELECT COUNT(*) FROM my_messages", [], |r| r.get(0)).unwrap();
    assert_eq!(count, 1);
}

#[tokio::test]
async fn runner_resumes_a_conversation_from_a_reopened_database() {
    let db = TempDb::new();
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::text_message("first answer")),
        ModelStep::from(ItemHelpers::text_message("second answer")),
    ]));
    let agent = Agent::new("a").model(model.clone());
    let run = |session: SqliteSession| {
        let mut options = RunOptions::default();
        options.session = Some(session.shared());
        options
    };

    Runner::run(&agent, "one", run(SqliteSession::open("conv", &db.0).unwrap())).await.unwrap();
    // A fresh session object over the same file, like a new process.
    let result = Runner::run(&agent, "two", run(SqliteSession::open("conv", &db.0).unwrap()))
        .await
        .unwrap();
    assert_eq!(result.final_output_as_str(), Some("second answer"));
    let second_input = model.calls()[1].input.clone();
    assert_eq!(second_input.as_array().map(Vec::len), Some(3), "{second_input}");
    let stored = SqliteSession::open("conv", &db.0).unwrap().get_items(None).await.unwrap();
    assert_eq!(stored.len(), 4);
}

/// The schema matches Python's, so each SDK reads what the other wrote. Skipped when the Python
/// oracle venv (`scripts/setup_venv.sh`) is not installed.
#[tokio::test]
async fn database_files_are_interchangeable_with_python() {
    let python = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".venv/bin/python");
    if !python.exists() {
        eprintln!("skipping: no .venv (run scripts/setup_venv.sh)");
        return;
    }
    let db = TempDb::new();
    let script = r#"
import asyncio, sys
from agents.memory import SQLiteSession
async def main(path):
    s = SQLiteSession("shared", path)
    print(len(await s.get_items()))
    await s.add_items([{"role": "user", "content": "from python"}])
asyncio.run(main(sys.argv[1]))
"#;
    let run_python = || {
        let out = std::process::Command::new(&python)
            .args(["-c", script])
            .arg(&db.0)
            .output()
            .expect("python");
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };

    assert_eq!(run_python(), "0");
    let session = SqliteSession::open("shared", &db.0).unwrap();
    assert_eq!(session.get_items(None).await.unwrap(), vec![msg("user", "from python")]);
    session.add_items(vec![msg("assistant", "from rust")]).await.unwrap();
    drop(session);
    assert_eq!(run_python(), "2", "Python sees the item Rust added");
}
