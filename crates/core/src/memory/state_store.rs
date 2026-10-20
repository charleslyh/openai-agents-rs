//! Where a paused run is kept while it waits (Rust-only, X-016).
//!
//! A run that stops for an approval can outlive the process that started it: Python keeps it in a
//! `RunState` you serialize yourself. [`RunStateStore`] is the storage side of that: save a state
//! under a key, load it back, delete it when it is done. The file, Redis and Postgres backends
//! live in `opai-agents-sessions`; this is the contract they implement.
//!
//! Two things a store must get right, and this module gives it the vocabulary for both:
//!
//! * **Schema.** A state records `RUN_STATE_SCHEMA_VERSION` when it is saved and is refused on
//!   load when the version is not one [`SUPPORTED_RUN_STATE_SCHEMAS`]. A deployment that upgrades
//!   across a schema change hands a [`RunStateMigration`] to the store, which runs it on load.
//! * **Integrity.** A stored state is data from outside the process. A store that can sign it
//!   (HMAC, key from the environment) should, and must refuse one whose signature does not match;
//!   [`RunStateStore::load`] failing is the signal, not a silent empty result.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::error::AgentsError;
use crate::run_state::{RunState, SUPPORTED_RUN_STATE_SCHEMAS};

/// Storage for paused runs (Rust-only).
#[async_trait]
pub trait RunStateStore: Send + Sync {
    /// Store `state` under `key`, replacing any state already there. `ttl` is how long the store
    /// should keep it; `None` means until it is deleted.
    async fn save(
        &self,
        key: &str,
        state: &RunState,
        ttl: Option<Duration>,
    ) -> Result<(), AgentsError>;

    /// The state stored under `key`, or `None` when there is none (or it expired).
    ///
    /// Fails when the state is there but unusable: a schema it cannot read, or a signature that
    /// does not match. Never returns `None` for those — a tampered state is an error.
    async fn load(&self, key: &str) -> Result<Option<RunState>, AgentsError>;

    /// Forget the state stored under `key`. Deleting a missing key is not an error.
    async fn delete(&self, key: &str) -> Result<(), AgentsError>;
}

/// Turns a state written by an older version into one this version can resume.
///
/// Called with the loaded state before it is handed back; return the state to resume, or an error
/// saying why it cannot be adapted.
pub type RunStateMigration = Arc<dyn Fn(RunState) -> Result<RunState, AgentsError> + Send + Sync>;

/// Whether `version` is a schema this build can read without a migration.
pub fn is_supported_schema(version: &str) -> bool {
    SUPPORTED_RUN_STATE_SCHEMAS.contains(&version)
}

/// Run `migrate` on `state` when one was given, and check the result is a schema this build reads.
///
/// Stores call this on load, so every backend applies migrations the same way.
pub fn finish_load(
    state: RunState,
    migrate: Option<&RunStateMigration>,
) -> Result<RunState, AgentsError> {
    let state = match migrate {
        Some(migrate) => migrate(state)?,
        None => state,
    };
    let version = state
        .to_json()
        .get("$schemaVersion")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    if !is_supported_schema(&version) {
        return Err(crate::error::UserError::new(format!(
            "a run state was migrated to schema `{version}`, which this build cannot read \
             (it reads {})",
            SUPPORTED_RUN_STATE_SCHEMAS.join(", ")
        ))
        .into());
    }
    Ok(state)
}
