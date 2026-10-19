//! Session and paused-run storage that outlives the process (Rust-only, X-016).
//!
//! Core defines the contracts — [`Session`](openai_agents_core::Session) for a conversation,
//! [`RunStateStore`](openai_agents_core::RunStateStore) for a run that stopped for an approval —
//! and ships nothing that writes to disk or talks to a server. This crate is where those live:
//!
//! * [`FileSession`]: a conversation in a JSONL file, locked across processes.
//! * [`FileRunStateStore`]: a paused run per file, signed so a state edited outside the process is
//!   refused instead of resumed.
//! * Redis (`redis` feature) and Postgres (`postgres` feature) for a session shared by more than
//!   one machine.
//!
//! Every backend runs [`openai_agents_core::session_contract`], so they differ in where they put
//! the history and not in what the history does.

pub mod file;
pub mod state_file;

pub use file::FileSession;
pub use state_file::{FileRunStateStore, STATE_KEY_ENV};
