//! Agent teams for the Rust OpenAI Agents SDK (Rust-only extension, experimental).
//!
//! A team is a lead plus long-lived teammates. Each teammate runs its own agent loop in its own
//! task: it sleeps until it has mail, works, reports back, and sleeps again. Teammates talk through
//! mailboxes and divide work through a shared task list. See `docs/AGENT-TEAMS.md` for the design.

#![deny(missing_docs)]

pub mod budget;
pub mod error;
pub mod mailbox;
pub mod taskboard;
pub mod types;

pub use budget::Budget;
pub use error::TeamError;
pub use mailbox::{InMemoryMailbox, Mailbox, DEFAULT_INBOX_CAPACITY};
pub use taskboard::{InMemoryTaskBoard, TaskBoard};
pub use types::{
    ClaimOutcome, MemberId, MemberInfo, MemberState, Message, MessageKind, NewTask, Task, TaskId,
    TaskStatus, TaskUpdate,
};
