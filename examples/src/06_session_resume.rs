//! A conversation that survives being closed (Rust-only)
//!
//! A `Session` is where a conversation lives. The agent keeps nothing: every run loads the stored
//! history into the model input and appends the new turn when it ends. So a conversation can be
//! stopped and picked up again by building a *new* session handle on the same storage, and a *new*
//! agent, which is also what a restart of your service looks like.
//!
//! The example walks through the whole life of one conversation:
//!
//! 1. **Create** a session on a backend and an agent that uses it.
//! 2. **Talk**: two turns, the second relying on the first.
//! 3. **Terminate**: drop the session handle and the agent. Only the storage is left.
//! 4. **Reopen**: a new handle on the same storage. The stored history is printed, then a question
//!    that only the earlier turns can answer; a real model replies with the name and the colour.
//!
//! The backend is the one function `open` below, picked with `SESSION_BACKEND`:
//!
//! * `file` (default): a JSONL file per session in `SESSION_DIR` (default
//!   `/tmp/agents-sessions`). `FileSession`.
//! * `sqlite`: rows in a SQLite database at `DB_PATH` (default `/tmp/agents-sessions.db`).
//!   `SqliteSession`.
//! * `memory`: `InMemorySession`, which lives and dies with the process. Reopening gives an empty
//!   conversation, and the example shows exactly that, as the contrast with the other two.
//!
//! `RedisSession` and `PostgresSession` (facade features `redis` and `postgres`) implement the same
//! trait, so they replace `open` and nothing else. They need a server, so this example does not
//! run them.
//!
//! `KEEP=1` skips the reset at the start, so running the binary again continues the stored
//! conversation across real process restarts.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example session_resume
//! SESSION_BACKEND=sqlite cargo run -p opai-agents-examples --example session_resume
//! SESSION_BACKEND=memory cargo run -p opai-agents-examples --example session_resume
//! ```
//!
//! Without `OPENAI_API_KEY` the offline model answers with a fixed message: the history is still
//! stored and printed on reopen, but the reply cannot show memory. Use a real key to see it.

mod common;

use std::sync::Arc;

use opai_agents::{FileSession, InMemorySession, Session, SqliteSession};

const SESSION_ID: &str = "demo";

/// Create the session handle. The only backend-specific code in this example.
fn open(backend: &str) -> Result<Arc<dyn Session>, Box<dyn std::error::Error>> {
    Ok(match backend {
        "file" => {
            let dir =
                std::env::var("SESSION_DIR").unwrap_or_else(|_| "/tmp/agents-sessions".into());
            println!("backend: file, history in {dir}/{SESSION_ID}.jsonl");
            FileSession::shared(dir, SESSION_ID)?
        }
        "sqlite" => {
            let path =
                std::env::var("DB_PATH").unwrap_or_else(|_| "/tmp/agents-sessions.db".into());
            println!("backend: sqlite, database {path}, session id `{SESSION_ID}`");
            SqliteSession::open(SESSION_ID, path)?.shared()
        }
        "memory" => {
            println!("backend: memory, gone when this handle is dropped");
            InMemorySession::shared(SESSION_ID)
        }
        other => {
            return Err(
                format!("unknown SESSION_BACKEND `{other}`; use file, sqlite or memory").into(),
            )
        }
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = std::env::var("SESSION_BACKEND").unwrap_or_else(|_| "file".into());

    // ---- 1 and 2. Create the session and talk. ----
    {
        let session = open(&backend)?;
        if std::env::var_os("KEEP").is_none() {
            session.clear_session().await?;
        }
        println!("\n== conversation");
        common::chat::ask(&session, "turn 1", "My name is Mina. Remember that.").await?;
        common::chat::ask(&session, "turn 2", "My favourite colour is teal.").await?;
        common::chat::show_history(&session, "stored").await?;

        // ---- 3. Terminate. ----
        println!("\n== terminate: session handle and agent dropped");
    }

    // ---- 4. Reopen: new handle, new agent, same storage. ----
    println!("\n== reopen");
    let session = open(&backend)?;
    common::chat::show_history(&session, "restored").await?;
    common::chat::ask(
        &session,
        "turn 3",
        "What is my name and my favourite colour?",
    )
    .await?;
    common::chat::show_history(&session, "after").await?;
    Ok(())
}
