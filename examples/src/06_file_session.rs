//! A conversation that outlives the process (Rust-only, X-017)
//!
//! `InMemorySession` forgets everything when the process ends. `FileSession` keeps one JSON item
//! per line in a directory, so the history is still there after a restart — and two processes can
//! share it, because writes take a lock.
//!
//! The example runs one turn, prints what was stored, then drops the session and opens the same
//! directory again: the second `open` reads what the first wrote, which is what a restart looks
//! like.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example file_session
//! SESSION_DIR=./.sessions cargo run -p opai-agents-examples --example file_session
//! ```
//!
//! No key is needed: without `OPENAI_API_KEY` the offline model answers.

mod common;

use std::sync::Arc;

use opai_agents::{Agent, FileSession, RunOptions, Runner, Session};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("SESSION_DIR").unwrap_or_else(|_| "/tmp/agents-file-session".into());

    let file = FileSession::open(&dir, "demo")?;
    println!("history: {}", file.path().display());
    let session: Arc<dyn Session> = Arc::new(file);

    let agent = Agent::new("Assistant")
        .instructions("Answer in one short sentence.")
        .model(common::model_or_offline()?);

    let result = Runner::run(
        &agent,
        "My favourite colour is teal. Remember that.",
        RunOptions::new().session(Arc::clone(&session)),
    )
    .await?;
    println!(
        "agent> {}",
        result.final_output_as_str().unwrap_or_default()
    );

    // Everything this process holds goes away; only the file is left.
    let stored = session.get_items(None).await?.len();
    drop(session);

    let reopened: Arc<dyn Session> = FileSession::shared(&dir, "demo")?;
    let reread = reopened.get_items(None).await?.len();
    println!("stored {stored} items, read back {reread} items in a new process");
    Ok(())
}
