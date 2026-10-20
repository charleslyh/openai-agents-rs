//! Roles: the templates a lead can create teammates from.
//!
//! A lead never describes a teammate freely. It picks a registered [`Role`] by name and adds
//! instructions for the job. The model, the tools and the turn limit come from the role, which the
//! application wrote. A lead that has been talked into something by untrusted text therefore cannot
//! pick a stronger model, add a tool or lift a restriction.

use std::sync::Arc;

use opai_agents_core::{Agent, InMemorySession, Session};

use crate::types::MemberId;

/// Makes the session a teammate keeps its conversation in.
pub type SessionFactory = Arc<dyn Fn(&MemberId) -> Arc<dyn Session> + Send + Sync>;

/// A template for teammates.
#[derive(Clone)]
pub struct Role {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) agent: Agent,
    pub(crate) max_turns: Option<usize>,
    pub(crate) session_factory: Option<SessionFactory>,
}

impl Role {
    /// A role called `name` (lowercase letters, digits, `-` and `_`, at most 32 characters) whose
    /// teammates run `agent`.
    pub fn new(name: impl Into<String>, agent: Agent) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            agent,
            max_turns: None,
            session_factory: None,
        }
    }

    /// What this role is good for; shown to the lead.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Model turns of one run of a teammate with this role. Capped by `Budget::max_turns_per_run`.
    #[must_use]
    pub fn max_turns(mut self, turns: usize) -> Self {
        self.max_turns = Some(turns);
        self
    }

    /// Keep teammates' conversations in sessions made by `factory` instead of in memory.
    #[must_use]
    pub fn session_factory(
        mut self,
        factory: impl Fn(&MemberId) -> Arc<dyn Session> + Send + Sync + 'static,
    ) -> Self {
        self.session_factory = Some(Arc::new(factory));
        self
    }

    /// The role's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn make_session(&self, member: &MemberId, team_id: &str) -> Arc<dyn Session> {
        match &self.session_factory {
            Some(factory) => factory(member),
            None => Arc::new(InMemorySession::new(format!("{team_id}/{member}"))),
        }
    }
}

impl std::fmt::Debug for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Role")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("max_turns", &self.max_turns)
            .finish()
    }
}

/// Whether `name` is usable as a role or member name.
pub(crate) fn valid_name(name: &str) -> Result<(), String> {
    let mut chars = name.chars();
    let first_ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit());
    if !first_ok
        || name.len() > 32
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
    {
        return Err(
            "use 1 to 32 characters: lowercase letters, digits, `-` and `_`, starting with a \
             letter or digit"
                .to_string(),
        );
    }
    Ok(())
}
