//! One database, many conversations (Rust-only)
//!
//! A service rarely has one conversation: it has one per user, per ticket, per chat window.
//! A `Session` is addressed by an id, and a backend can host any number of ids side by side.
//! `SqliteSession` does it in one database file: every session shares the file and its tables, and
//! the id keeps their histories apart. (`RedisSession` and `PostgresSession` work the same way:
//! one server, many ids.)
//!
//! The example hosts three users in one file and shows what that means in practice:
//!
//! 1. **One handle per id.** `session_for(user)` is all a request handler needs: open the id,
//!    run, let go.
//! 2. **No crosstalk.** The turns of the three users are interleaved, yet each session ends up
//!    holding only its own messages (printed from the stored history, so it shows even offline).
//! 3. **Independent lifecycle.** Clearing one conversation leaves the others untouched.
//! 4. **Reopen from the file.** The surviving conversations continue from their own history; a
//!    real model answers each user from what that user said.
//!
//! `DB_PATH` chooses the file (default `/tmp/agents-multi-session.db`).
//!
//! ```bash
//! cargo run -p opai-agents-examples --example multi_session
//! ```
//!
//! Without `OPENAI_API_KEY` the offline model gives fixed replies; the histories printed after
//! each step still show the isolation. Use a real key to see the answers differ.

mod common;

use std::sync::Arc;

use opai_agents::{Session, SqliteSession};

/// What a request handler does per user: open that user's conversation in the shared database.
fn session_for(path: &str, user: &str) -> Result<Arc<dyn Session>, Box<dyn std::error::Error>> {
    Ok(SqliteSession::open(user, path)?.shared())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var("DB_PATH").unwrap_or_else(|_| "/tmp/agents-multi-session.db".into());
    println!("one database: {path}");

    let users = [("alice", "teal"), ("bob", "orange"), ("carol", "green")];
    for (user, _) in users {
        session_for(&path, user)?.clear_session().await?;
    }

    // ---- 1 and 2. Interleaved turns: each user, then each user again. ----
    println!("\n== interleaved conversations");
    for (user, colour) in users {
        let session = session_for(&path, user)?;
        let message = format!("I am {user}. My favourite colour is {colour}.");
        common::chat::ask(&session, user, &message).await?;
    }
    for (user, _) in users {
        let session = session_for(&path, user)?;
        common::chat::ask(&session, user, "What is my favourite colour?").await?;
    }

    println!("\n== what each session holds (only its own messages)");
    for (user, _) in users {
        common::chat::show_history(&session_for(&path, user)?, user).await?;
    }

    // ---- 3. Independent lifecycle. ----
    println!("\n== clear bob's conversation; the others are untouched");
    session_for(&path, "bob")?.clear_session().await?;
    for (user, _) in users {
        let items = session_for(&path, user)?.get_items(None).await?.len();
        println!("[{user}] {items} stored items");
    }

    // ---- 4. Reopen from the file: the surviving conversations continue. ----
    println!("\n== reopen alice and carol from the file");
    for user in ["alice", "carol"] {
        let session = session_for(&path, user)?;
        common::chat::ask(
            &session,
            user,
            "Remind me: what did I tell you about myself?",
        )
        .await?;
    }
    Ok(())
}
