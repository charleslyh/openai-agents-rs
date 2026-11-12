//! Session and paused-run storage that outlives the process (Rust-only, X-016).
//!
//! Core defines the contracts — [`Session`](opai_agents_core::Session) for a conversation,
//! [`RunStateStore`](opai_agents_core::RunStateStore) for a run that stopped for an approval —
//! and ships nothing that writes to disk or talks to a server. This crate is where those live:
//!
//! * [`FileSession`]: a conversation in a JSONL file, locked across processes.
//! * [`FileBlobStore`]: offloaded tool outputs, one file each, for
//!   [`OffloadToolOutputs`](opai_agents_core::OffloadToolOutputs).
//! * [`FileRunStateStore`]: a paused run per file, signed so a state edited outside the process is
//!   refused instead of resumed.
//! * [`RedisSession`] and [`RedisRunStateStore`] (`redis` feature): the same, in a list and a key,
//!   for a history more than one process shares.
//! * [`SqliteSession`] (`sqlite` feature): a database file or in-memory, with the Python SDK's
//!   schema so either SDK can read the file.
//! * [`SqliteFactStore`] (`sqlite` feature): long-term facts for
//!   [`FactMemory`](opai_agents_core::FactMemory), in the same kind of file.
//! * [`PostgresSession`] and [`PostgresRunStateStore`] (`postgres` feature): the same, in rows, for
//!   a deployment that already has a database.
//!
//! Every backend runs [`opai_agents_core::session_contract`], so they differ in where they put
//! the history and not in what the history does.
//!
//! The crates that talk to a server sit behind features, so a build that keeps history in memory
//! or in a file compiles no Redis client and no database driver.

pub mod blob_file;
pub mod file;
mod state;
pub mod state_file;

#[cfg(feature = "sqlite")]
pub mod facts_sqlite;
#[cfg(feature = "postgres")]
pub mod postgres;
#[cfg(feature = "redis")]
pub mod redis;
#[cfg(feature = "sqlite")]
pub mod sqlite;

pub use blob_file::FileBlobStore;
pub use file::FileSession;
pub use state_file::{FileRunStateStore, STATE_KEY_ENV};

#[cfg(feature = "sqlite")]
pub use facts_sqlite::SqliteFactStore;
#[cfg(feature = "postgres")]
pub use postgres::{PostgresRunStateStore, PostgresSession};
#[cfg(feature = "redis")]
pub use redis::{RedisRunStateStore, RedisSession};
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteSession;
