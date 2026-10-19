//! Session and paused-run storage that outlives the process (Rust-only, X-016).
//!
//! Core defines the contracts — [`Session`](openai_agents_core::Session) for a conversation,
//! [`RunStateStore`](openai_agents_core::RunStateStore) for a run that stopped for an approval —
//! and ships nothing that writes to disk or talks to a server. This crate is where those live:
//!
//! * [`FileSession`]: a conversation in a JSONL file, locked across processes.
//! * Redis (`redis` feature) and Postgres (`postgres` feature) for a session shared by more than
//!   one machine.
//!
//! Every backend runs [`openai_agents_core::session_contract`], so they differ in where they put
//! the history and not in what the history does.

pub mod file;

pub use file::FileSession;
