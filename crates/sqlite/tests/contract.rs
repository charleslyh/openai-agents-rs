//! The bundled SQLite backend runs the same contract as every other [`Session`].

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use openai_agents_core::session_contract;
use openai_agents_sqlite::SqliteSession;

/// A path no other test will pick, so parallel test binaries do not share a database.
fn temp_db(name: &str) -> std::path::PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let path = std::env::temp_dir().join(format!("agents-contract-{name}-{unique}.db"));
    let _ = std::fs::remove_file(&path);
    path
}

#[tokio::test]
async fn sqlite_passes_the_session_contract() {
    let path = temp_db("file");
    // The session id comes first, then the file. (The other way round creates a database named
    // `contract` in the working directory, and tests the wrong file.)
    let session = SqliteSession::open("contract", &path).expect("open");
    session_contract::assert_session_contract(&session).await;
    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn sqlite_passes_the_contract_in_memory() {
    let session = SqliteSession::in_memory("contract").expect("open");
    session_contract::assert_session_contract(&session).await;
}

/// Concurrent writes from several tasks all land, with no lost update.
#[tokio::test]
async fn sqlite_keeps_every_concurrent_write() {
    let session: Arc<dyn openai_agents_core::Session> =
        Arc::new(SqliteSession::in_memory("shared").expect("open"));
    session_contract::assert_concurrent_writes(Arc::clone(&session), 16).await;
}
