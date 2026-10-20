//! Plain data shared by every part of the team layer.
//!
//! All types are `serde`-serialisable on purpose: the in-memory mailbox and task board are the
//! first implementations of their traits, not the last, and a persistent backend should not have to
//! invent its own wire format.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Name of a team member. The coordinating agent is always [`MemberId::lead`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MemberId(String);

impl MemberId {
    /// The id of the lead.
    pub const LEAD: &'static str = "lead";

    /// Wrap a name. Validation of spawned names happens in `TeamHandle::spawn_member`.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// The lead.
    pub fn lead() -> Self {
        Self(Self::LEAD.to_string())
    }

    /// Whether this is the lead.
    pub fn is_lead(&self) -> bool {
        self.0 == Self::LEAD
    }

    /// The name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MemberId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for MemberId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for MemberId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

/// Opaque id of a task.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TaskId(String);

impl TaskId {
    /// Wrap an id.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The id as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for TaskId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

/// Where a member is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemberState {
    /// Waiting for mail; costs no model call.
    Idle,
    /// Running the agent.
    Working,
    /// Paused until the lead resolves a tool call.
    Blocked,
    /// A shutdown was requested; the member is finishing its current work.
    ShuttingDown,
    /// Left the team in an orderly way.
    Stopped,
    /// Ended because of an error or a panic.
    Failed,
}

impl MemberState {
    /// Whether the member will never run again.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Stopped | Self::Failed)
    }
}

/// What a message is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageKind {
    /// Free text from one member to another.
    Text,
    /// Sent by a member to the lead when it finishes a piece of work.
    IdleNotice,
    /// The lead asks a member to finish and leave.
    ShutdownRequest,
    /// A member confirms it is leaving.
    ShutdownResponse,
    /// A member asks the lead to resolve a paused tool call.
    ApprovalRequest,
    /// The lead's answer to an approval request.
    ApprovalResponse,
    /// A problem report (member failed, budget exceeded).
    Alert,
}

/// One message in a mailbox.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Unique id.
    pub id: String,
    /// Sender. Set by the team from the caller's identity, never taken from tool arguments.
    pub from: MemberId,
    /// Recipient.
    pub to: MemberId,
    /// Purpose.
    pub kind: MessageKind,
    /// Text. Untrusted when it comes from another agent.
    pub body: String,
    /// Milliseconds since the Unix epoch.
    pub sent_at_ms: u64,
}

impl Message {
    /// A new message stamped with a fresh id and the current time.
    pub fn new(from: MemberId, to: MemberId, kind: MessageKind, body: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            from,
            to,
            kind,
            body: body.into(),
            sent_at_ms: now_ms(),
        }
    }
}

/// Milliseconds since the Unix epoch (0 if the clock is before it).
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Progress of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Waiting for an owner.
    Pending,
    /// Claimed and being worked on.
    InProgress,
    /// Done. Final.
    Completed,
}

/// A unit of work on the shared board.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    /// Id.
    pub id: TaskId,
    /// Short title.
    pub title: String,
    /// What to do.
    pub description: String,
    /// Progress.
    pub status: TaskStatus,
    /// Who works on it, once claimed.
    pub owner: Option<MemberId>,
    /// Tasks that must be completed first.
    pub blocked_by: Vec<TaskId>,
    /// Outcome recorded on completion.
    pub result: Option<String>,
    /// Who created it.
    pub created_by: MemberId,
}

/// Input of `TaskBoard::create`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewTask {
    /// Short title.
    pub title: String,
    /// What to do.
    pub description: String,
    /// Tasks that must be completed first. They must already exist.
    pub blocked_by: Vec<TaskId>,
    /// Who creates it.
    pub created_by: MemberId,
}

/// Input of `TaskBoard::update`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskUpdate {
    /// New status: `InProgress` to `Completed`, or back to `Pending` to give the task up.
    pub status: Option<TaskStatus>,
    /// Outcome, recorded with `Completed`.
    pub result: Option<String>,
}

/// Result of an attempt to claim a task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum ClaimOutcome {
    /// The caller owns the task now.
    Claimed {
        /// The task after the claim.
        task: Task,
    },
    /// Someone else got there first (or the caller already owns it).
    AlreadyTaken {
        /// The current owner.
        owner: Option<MemberId>,
    },
    /// Dependencies are not completed yet.
    Blocked {
        /// The unfinished dependencies.
        waiting_on: Vec<TaskId>,
    },
    /// The task is already completed.
    AlreadyCompleted,
    /// No task with this id.
    NotFound,
}

/// Public view of a member.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemberInfo {
    /// Name.
    pub id: MemberId,
    /// Role the member was created from.
    pub role: String,
    /// State.
    pub state: MemberState,
    /// Messages delivered but not yet fully handled.
    pub unprocessed: usize,
}
