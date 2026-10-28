//! A conversation that outlives the process (Rust-only, X-017)
//!
//! `InMemorySession` forgets everything when the process ends. `FileSession` keeps one JSON item
//! per line in a directory, so the history is still there after a restart — and the next run can
//! answer from it.
//!
//! The example plays two "sessions of use" in one binary:
//!
//! 1. Run 1 tells the agent a fact (a favourite colour), then every in-process handle is dropped.
//! 2. Run 2 reopens the same directory with a brand-new `FileSession` and a brand-new `Agent`,
//!    and asks a question that can only be answered from run 1. The runner loads the stored
//!    history into the model input, so a real model replies "teal".
//!
//! Set `KEEP=1` to skip the reset at start: re-running the binary then continues the same
//! conversation, which is what a real restart looks like.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example file_session
//! SESSION_DIR=./.sessions cargo run -p opai-agents-examples --example file_session
//! ```
//!
//! Without `OPENAI_API_KEY` the offline model answers with a fixed message, so the history is
//! stored and reloaded but the reply cannot show memory; use a real key to see it.

mod common;

use std::sync::Arc;

use opai_agents::{Agent, FileSession, RunOptions, Runner, Session};

fn new_agent() -> Result<Agent, Box<dyn std::error::Error>> {
    Ok(Agent::new("Assistant")
        .instructions("Answer in one short sentence, using the conversation so far.")
        .model(common::model_or_offline()?))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("SESSION_DIR").unwrap_or_else(|_| "/tmp/agents-file-session".into());

    // ---- Process 1: tell the agent something. ----
    {
        let file = FileSession::open(&dir, "demo")?;
        println!("history file: {}", file.path().display());
        let session: Arc<dyn Session> = Arc::new(file);
        if std::env::var_os("KEEP").is_none() {
            session.clear_session().await?;
        }

        let q1 = "My favourite colour is teal. Remember that.";
        println!("\n[run 1] user> {q1}");
        let result = Runner::run(
            &new_agent()?,
            q1,
            RunOptions::new().session(Arc::clone(&session)),
        )
        .await?;
        println!(
            "[run 1] agent> {}",
            result.final_output_as_str().unwrap_or_default()
        );
        println!(
            "[run 1] stored {} items",
            session.get_items(None).await?.len()
        );
        // Session, agent and runner state all go out of scope here; only the file is left.
    }

    // ---- Process 2: a fresh session on the same directory, a fresh agent. ----
    let reopened: Arc<dyn Session> = FileSession::shared(&dir, "demo")?;
    println!("\n[restart] reopened the session from disk");

    let q2 = "What is my favourite colour?";
    println!("\n[run 2] user> {q2}");
    let result = Runner::run(
        &new_agent()?,
        q2,
        RunOptions::new().session(Arc::clone(&reopened)),
    )
    .await?;
    println!(
        "[run 2] agent> {}",
        result.final_output_as_str().unwrap_or_default()
    );
    println!(
        "[run 2] stored {} items (run 1 + run 2)",
        reopened.get_items(None).await?.len()
    );
    Ok(())
}
