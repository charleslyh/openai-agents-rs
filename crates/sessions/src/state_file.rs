//! A paused run in a file, signed so an edited state is refused rather than resumed (X-016).
//!
//! A run that stops for an approval may outlive the process: the state is written here, and read
//! back — perhaps hours later, perhaps by another process. That makes it data from outside, so it
//! is treated as such:
//!
//! * **Integrity.** With `AGENTS_STATE_KEY` set, each state carries an HMAC-SHA256 of its own
//!   JSON. A state whose signature does not match fails to load instead of resuming: an approval
//!   someone edited into a different call must never be honoured. Comparing takes the same time
//!   whatever the first byte is.
//! * **Schema.** The version is checked on load, and a [`RunStateMigration`] given to the store can
//!   adapt a state written by an older build before it is resumed.
//! * **Expiry.** `save` can set a TTL; an expired state reads as absent and is deleted.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use fs2::FileExt;
use serde_json::Value;

use opai_agents_core::error::UserError;
use opai_agents_core::{AgentsError, RunState, RunStateMigration, RunStateStore};

use super::file::file_name;
use super::state::{decode, key_from_env, to_state, wrap};

/// Where the signing key is read from.
pub const STATE_KEY_ENV: &str = "AGENTS_STATE_KEY";

/// A paused run per file, in `dir`.
///
/// Every read and write takes a cross-process lock, and a write goes to a temporary file that is
/// renamed into place, so a crash cannot leave a half state behind.
pub struct FileRunStateStore {
    dir: PathBuf,
    /// Signing key, when one is configured.
    key: Option<Vec<u8>>,
    migrate: Option<RunStateMigration>,
}

impl FileRunStateStore {
    /// Use `dir` for states, signing them when `AGENTS_STATE_KEY` is set.
    pub fn from_env(dir: impl Into<PathBuf>) -> Result<Self, AgentsError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(|e| {
            UserError::new(format!(
                "cannot create state directory {}: {e}",
                dir.display()
            ))
        })?;
        let key = key_from_env(STATE_KEY_ENV);
        Ok(Self {
            dir,
            key,
            migrate: None,
        })
    }

    /// Use `dir` without signing. For tests, and for deployments that store states somewhere only
    /// this process can write.
    pub fn unsigned(dir: impl Into<PathBuf>) -> Result<Self, AgentsError> {
        let mut store = Self::from_env(dir)?;
        store.key = None;
        Ok(store)
    }

    /// Adapt states written by an older build before they are resumed.
    #[must_use]
    pub fn with_migration(mut self, migrate: RunStateMigration) -> Self {
        self.migrate = Some(migrate);
        self
    }

    /// Whether states are signed.
    pub fn is_signed(&self) -> bool {
        self.key.is_some()
    }

    /// Hold the lock on the whole directory for the rest of the operation.
    fn lock(&self) -> Result<File, AgentsError> {
        let path = self.dir.join(".state.lock");
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| UserError::new(format!("cannot open the lock {}: {e}", path.display())))?;
        file.lock_exclusive()
            .map_err(|e| UserError::new(format!("cannot lock {}: {e}", path.display())))?;
        Ok(file)
    }

    fn path_for(&self, key: &str) -> Result<PathBuf, AgentsError> {
        let name = file_name(key)?;
        Ok(self.dir.join(format!("{name}.state.json")))
    }
}

#[async_trait]
impl RunStateStore for FileRunStateStore {
    async fn save(
        &self,
        key: &str,
        state: &RunState,
        ttl: Option<Duration>,
    ) -> Result<(), AgentsError> {
        let _lock = self.lock()?;
        let path = self.path_for(key)?;
        let envelope = wrap(state, ttl, self.key.as_deref())?;
        atomic_write(&path, &envelope)
    }

    async fn load(&self, key: &str) -> Result<Option<RunState>, AgentsError> {
        let _lock = self.lock()?;
        let path = self.path_for(key)?;
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(UserError::new(format!("cannot read {}: {e}", path.display())).into())
            }
        };
        let envelope: Value = serde_json::from_slice(&bytes)
            .map_err(|e| UserError::new(format!("{} is not a stored run: {e}", path.display())))?;
        let label = path.display().to_string();
        let Some(state_json) = decode(&envelope, self.key.as_deref(), &label)? else {
            std::fs::remove_file(&path).ok();
            return Ok(None);
        };
        Ok(Some(to_state(state_json, self.migrate.as_ref())?))
    }

    async fn delete(&self, key: &str) -> Result<(), AgentsError> {
        let _lock = self.lock()?;
        let path = self.path_for(key)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(UserError::new(format!("cannot delete {}: {e}", path.display())).into()),
        }
    }
}

impl std::fmt::Debug for FileRunStateStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileRunStateStore")
            .field("dir", &self.dir)
            .field("signed", &self.key.is_some())
            .field("has_migration", &self.migrate.is_some())
            .finish()
    }
}

/// Write `value` to `path` through a temporary file, so a crash cannot leave a half state.
fn atomic_write(path: &Path, value: &Value) -> Result<(), AgentsError> {
    let tmp = path.with_extension("state.json.tmp");
    let bytes = serde_json::to_vec(value)
        .map_err(|e| UserError::new(format!("cannot serialize a paused run: {e}")))?;
    std::fs::write(&tmp, bytes)
        .map_err(|e| UserError::new(format!("cannot write {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        UserError::new(format!(
            "cannot replace {} with {}: {e}",
            path.display(),
            tmp.display()
        ))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::now_seconds;
    use opai_agents_core::RUN_STATE_SCHEMA_VERSION;
    use serde_json::json;
    use std::sync::Arc;

    fn dir(name: &str) -> PathBuf {
        let unique = now_seconds();
        let path = std::env::temp_dir().join(format!("agents-state-{name}-{unique}"));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    /// A state to store: any paused run will do, and `RunState::from_json` needs a starting agent.
    fn state() -> RunState {
        RunState::from_json(
            "ops",
            json!({
                "$schemaVersion": RUN_STATE_SCHEMA_VERSION,
                "starting_agent_name": "ops",
                "input": {"type": "text", "text": "ship it"},
                "pending_response": {"output": [], "usage": {}},
            }),
        )
        .expect("a minimal state")
    }

    #[tokio::test]
    async fn a_state_round_trips_and_can_be_deleted() {
        let dir = dir("roundtrip");
        let store = FileRunStateStore::unsigned(&dir).expect("open");
        assert!(store.load("k").await.expect("load").is_none());
        store.save("k", &state(), None).await.expect("save");
        assert!(store.load("k").await.expect("load").is_some());
        store.delete("k").await.expect("delete");
        assert!(store.load("k").await.expect("load").is_none());
        store.delete("k").await.expect("deleting twice is fine");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An expired state reads as absent, and is removed.
    #[tokio::test]
    async fn an_expired_state_is_absent() {
        let dir = dir("ttl");
        let store = FileRunStateStore::unsigned(&dir).expect("open");
        store
            .save("k", &state(), Some(Duration::from_secs(1)))
            .await
            .expect("save");
        // Rewrite the expiry as if it had passed.
        let path = dir.join("k.state.json");
        let mut envelope: Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("json");
        envelope["expires_at"] = json!(now_seconds().saturating_sub(1));
        std::fs::write(&path, serde_json::to_vec(&envelope).expect("json")).expect("write");
        assert!(store.load("k").await.expect("load").is_none());
        assert!(!path.exists(), "an expired state is removed");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A state edited outside the process is an error, not a resume; so is one whose signature was
    /// removed. Both halves are in one test on purpose: `cargo test` runs tests of a binary in
    /// parallel threads, and `AGENTS_STATE_KEY` is process-wide.
    #[tokio::test]
    async fn a_signed_store_refuses_a_state_that_was_changed() {
        let dir = dir("signed");
        std::env::set_var(STATE_KEY_ENV, "s3cret");
        let store = FileRunStateStore::from_env(&dir).expect("open");
        std::env::remove_var(STATE_KEY_ENV);
        assert!(
            store.is_signed(),
            "the key is picked up from the environment"
        );
        store.save("k", &state(), None).await.expect("save");

        let path = dir.join("k.state.json");
        let mut envelope: Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("json");
        // As if someone had pointed the paused run at another agent.
        envelope["state"]["current_agent_name"] = json!("someone-else");
        std::fs::write(&path, serde_json::to_vec(&envelope).expect("json")).expect("write");
        let error = store
            .load("k")
            .await
            .expect_err("the signature no longer matches");
        assert!(
            error.to_string().contains("changed after it was stored"),
            "{error}"
        );

        store.save("k", &state(), None).await.expect("save again");
        let mut envelope: Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("json");
        envelope.as_object_mut().expect("object").remove("mac");
        std::fs::write(&path, serde_json::to_vec(&envelope).expect("json")).expect("write");
        let error = store.load("k").await.expect_err("no signature");
        assert!(error.to_string().contains("no signature"), "{error}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The migration hook runs on load, and its result is checked against the schema.
    #[tokio::test]
    async fn a_migration_adapts_a_state_on_load() {
        let dir = dir("migrate");
        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        let store = FileRunStateStore::unsigned(&dir)
            .expect("open")
            .with_migration(Arc::new(move |mut state: RunState| {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
                state.allow_agent_graph_change();
                Ok(state)
            }) as RunStateMigration);
        store.save("k", &state(), None).await.expect("save");
        store.load("k").await.expect("load").expect("a state");
        assert!(
            ran.load(std::sync::atomic::Ordering::SeqCst),
            "the hook ran"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Keys are file names, so they are validated.
    #[tokio::test]
    async fn an_unsafe_key_is_refused() {
        let dir = dir("keys");
        let store = FileRunStateStore::unsigned(&dir).expect("open");
        for key in ["../outside", "a/b", "", "..", ".hidden"] {
            assert!(
                store.load(key).await.is_err() || store.save(key, &state(), None).await.is_err(),
                "`{key}` must be refused"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
