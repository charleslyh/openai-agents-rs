//! Observing a team and steering it from the outside.
//!
//! [`TeamEvent`]s are for watching (a UI, a log, a test). [`TeamHooks`] are for deciding: they can
//! veto a teammate going idle or a task being completed, which is how a quality gate is attached.

use async_trait::async_trait;
use tokio::sync::broadcast;

use crate::types::{MemberId, MemberState, MessageKind, Task, TaskId};

/// How many events a subscriber may fall behind before it starts losing the oldest ones.
pub const EVENT_BUFFER: usize = 256;

/// Something that happened in the team.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TeamEvent {
    /// A teammate was created.
    MemberSpawned {
        /// Its name.
        member: MemberId,
        /// The role it was created from.
        role: String,
    },
    /// A member moved to another state.
    MemberStateChanged {
        /// The member.
        member: MemberId,
        /// The new state.
        state: MemberState,
    },
    /// A message was delivered. The text is not part of the event.
    MessageSent {
        /// Sender.
        from: MemberId,
        /// Recipient.
        to: MemberId,
        /// Purpose.
        kind: MessageKind,
    },
    /// A task was created.
    TaskCreated {
        /// The task.
        task: TaskId,
        /// Who created it.
        by: MemberId,
    },
    /// A task was claimed.
    TaskClaimed {
        /// The task.
        task: TaskId,
        /// The new owner.
        by: MemberId,
    },
    /// A task was completed.
    TaskCompleted {
        /// The task.
        task: TaskId,
        /// Who completed it.
        by: MemberId,
    },
    /// A tool call of a teammate is waiting for the lead's decision.
    ApprovalRequested {
        /// The member.
        member: MemberId,
        /// The tool call.
        call_id: String,
        /// The tool.
        tool: String,
    },
    /// A waiting tool call was decided.
    ApprovalResolved {
        /// The member.
        member: MemberId,
        /// The tool call.
        call_id: String,
    },
    /// A team limit stopped work. Reported once.
    BudgetExceeded {
        /// Which limit.
        reason: String,
    },
    /// A member ended because of an error or a panic.
    MemberFailed {
        /// The member.
        member: MemberId,
        /// What happened (no message bodies).
        reason: String,
    },
    /// The team was shut down.
    TeamShutdown,
}

/// Fan-out of events to any number of subscribers.
///
/// Delivery never blocks the team: a subscriber that falls more than [`EVENT_BUFFER`] events behind
/// gets a `Lagged` error on its next receive and skips ahead.
#[derive(Clone)]
pub(crate) struct EventBus {
    sender: broadcast::Sender<TeamEvent>,
}

impl EventBus {
    pub(crate) fn new() -> Self {
        Self {
            sender: broadcast::channel(EVENT_BUFFER).0,
        }
    }

    pub(crate) fn emit(&self, event: TeamEvent) {
        // No subscribers is the normal case, not an error.
        let _ = self.sender.send(event);
    }

    pub(crate) fn subscribe(&self) -> broadcast::Receiver<TeamEvent> {
        self.sender.subscribe()
    }
}

/// What a hook decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookDecision {
    /// Go ahead.
    Allow,
    /// Do not. The reason is handed to the member so it can act on it.
    Reject(String),
}

/// Quality gates around the team's work. Every method defaults to [`HookDecision::Allow`].
#[async_trait]
pub trait TeamHooks: Send + Sync {
    /// A teammate finished its work and is about to report and go idle. `summary` is its answer.
    ///
    /// Rejecting makes the member keep working with the reason as its next input, at most three
    /// times in a row (and each extra run counts against the budget).
    async fn on_teammate_idle(&self, _member: &MemberId, _summary: &str) -> HookDecision {
        HookDecision::Allow
    }

    /// A member marks `task` completed. Rejecting leaves the task in progress and returns the
    /// reason to the member.
    async fn on_task_completed(&self, _task: &Task, _by: &MemberId) -> HookDecision {
        HookDecision::Allow
    }
}

/// Hooks that allow everything.
pub(crate) struct AllowAll;

impl TeamHooks for AllowAll {}
