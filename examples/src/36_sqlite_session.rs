//! A conversation in a SQLite database (Rust-only)
//!
//! `FileSession` keeps one directory per conversation. `SqliteSession` keeps many conversations
//! in one database file, each under its own session id, with the Python SDK's table layout — so a
//! single `.db` file can serve every user of a small service, and `sqlite3` can inspect it.
//!
//! The example opens the same file twice, as two separate processes would:
//!
//! 1. Run 1 stores a fact under session id `alice` and a different one under `bob`.
//! 2. Run 2 reopens the file with brand-new `SqliteSession` handles and asks each session for the
//!    fact it was told. The sessions do not see each other's history.
//!
//! `DB_PATH` chooses the file (default `/tmp/agents-sqlite-session.db`); `KEEP=1` skips the reset
//! at the start so that re-running the binary continues the same conversations.
//!
//! Postgres and Redis follow the same shape: `PostgresSession` and `RedisSession` implement the
//! same `Session` trait behind the facade's `postgres` / `redis` features, so only the line that
//! creates the session changes.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example sqlite_session
//! DB_PATH=./agents.db KEEP=1 cargo run -p opai-agents-examples --example sqlite_session
//! ```
//!
//! Without `OPENAI_API_KEY` the offline model answers with a fixed message, so the history is
//! stored and reloaded but the reply cannot show memory; use a real key to see it.

mod common;

use std::sync::Arc;

use opai_agents::{Agent, RunOptions, Runner, Session, SqliteSession};

fn new_agent() -> Result<Agent, Box<dyn std::error::Error>> {
    Ok(Agent::new("Assistant")
        .instructions("Answer in one short sentence, using the conversation so far.")
        .model(common::model_or_offline()?))
}

async fn ask(
    session: &Arc<dyn Session>,
    label: &str,
    question: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n[{label}] user> {question}");
    let result = Runner::run(
        &new_agent()?,
        question,
        RunOptions::new().session(Arc::clone(session)),
    )
    .await?;
    println!(
        "[{label}] agent> {}",
        result.final_output_as_str().unwrap_or_default()
    );
    println!(
        "[{label}] stored {} items",
        session.get_items(None).await?.len()
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var("DB_PATH").unwrap_or_else(|_| "/tmp/agents-sqlite-session.db".into());
    println!("database: {path}");

    // ---- Process 1: two users, two session ids, one file. ----
    {
        let alice: Arc<dyn Session> = SqliteSession::open("alice", &path)?.shared();
        let bob: Arc<dyn Session> = SqliteSession::open("bob", &path)?.shared();
        if std::env::var_os("KEEP").is_none() {
            alice.clear_session().await?;
            bob.clear_session().await?;
        }
        ask(
            &alice,
            "run 1 alice",
            "My favourite colour is teal. Remember that.",
        )
        .await?;
        ask(
            &bob,
            "run 1 bob",
            "My favourite colour is orange. Remember that.",
        )
        .await?;
        // Both handles and their connections are dropped here; only the file is left.
    }

    // ---- Process 2: fresh handles on the same file. ----
    println!("\n[restart] reopened the database");
    let alice: Arc<dyn Session> = SqliteSession::open("alice", &path)?.shared();
    let bob: Arc<dyn Session> = SqliteSession::open("bob", &path)?.shared();
    ask(&alice, "run 2 alice", "What is my favourite colour?").await?;
    ask(&bob, "run 2 bob", "What is my favourite colour?").await?;
    Ok(())
}
