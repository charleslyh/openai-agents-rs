//! Errors of the team layer.
//!
//! Every variant is meant to be read by a model: the text says what was wrong and what can be
//! done instead, because tool errors are handed back to the agent that made the call.

use crate::types::{MemberId, TaskId};

/// What went wrong in a team operation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TeamError {
    /// The addressed member does not exist (or has already stopped).
    #[error("unknown team member `{0}`")]
    UnknownMember(MemberId),
    /// The role name was not registered with `TeamBuilder::role`.
    #[error("unknown role `{role}`; available roles: {available}")]
    UnknownRole {
        /// The requested role.
        role: String,
        /// Comma separated registered roles.
        available: String,
    },
    /// The member name is not allowed or already taken.
    #[error("invalid member name `{name}`: {reason}")]
    InvalidName {
        /// The requested name.
        name: String,
        /// Why it was refused.
        reason: String,
    },
    /// The recipient's inbox is full; try again later or message someone else.
    #[error("the inbox of `{0}` is full; retry later")]
    MailboxFull(MemberId),
    /// A message or task text exceeds the size limit.
    #[error("{what} is {len} bytes, over the limit of {max} bytes")]
    TooLarge {
        /// Which field.
        what: &'static str,
        /// Actual size.
        len: usize,
        /// Allowed size.
        max: usize,
    },
    /// A team limit was reached.
    #[error("team budget exceeded: {0}")]
    BudgetExceeded(String),
    /// The task does not exist.
    #[error("unknown task `{0}`")]
    UnknownTask(TaskId),
    /// The caller may not change this task.
    #[error("task `{task}` is not owned by `{caller}`")]
    NotOwner {
        /// The task.
        task: TaskId,
        /// The member who tried.
        caller: MemberId,
    },
    /// The requested status change is not allowed.
    #[error("invalid transition: {0}")]
    InvalidTransition(String),
    /// A hook refused the change; the reason is for the member to act on.
    #[error("rejected: {0}")]
    Rejected(String),
    /// The team has been shut down.
    #[error("the team is shut down")]
    Closed,
    /// No pending approval matches.
    #[error("no pending approval `{call_id}` for `{member}`")]
    UnknownApproval {
        /// The member.
        member: MemberId,
        /// The tool call id.
        call_id: String,
    },
    /// Bad input that does not fit another variant.
    #[error("{0}")]
    Invalid(String),
}

impl From<TeamError> for opai_agents_core::AgentsError {
    fn from(error: TeamError) -> Self {
        opai_agents_core::AgentsError::tool(error.to_string())
    }
}
