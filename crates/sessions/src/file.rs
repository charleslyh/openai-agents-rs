//! A session in a JSONL file: one item per line, newest last.
//!
//! The backend for a single machine: no server to run, and the history is readable with `cat`.
//! Writes take a cross-process lock, so a second process touching the same session waits instead
//! of interleaving lines, and a rewrite goes to a temporary file that is renamed into place, so a
//! crash mid-write cannot leave a half history behind.
//!
//! The session id becomes the file name, so it is validated: no separators, no leading dot, no
//! `.` or `..`, and an existing path that is a symlink is refused rather than followed.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use async_trait::async_trait;
use fs2::FileExt;
use serde_json::Value;

use openai_agents_core::error::UserError;
use openai_agents_core::{AgentsError, Session, SessionSettings};

/// A session stored as one JSON item per line.
///
/// The file IO is blocking (as SQLite's is): call it from a task where that is acceptable, or from
/// `spawn_blocking`.
#[derive(Debug)]
pub struct FileSession {
    id: String,
    path: PathBuf,
    lock_path: PathBuf,
    /// Serialises this process; the file lock covers the others.
    in_process: Mutex<()>,
}

impl FileSession {
    /// Open (or create) the session `session_id` in directory `dir`.
    pub fn open(
        dir: impl Into<PathBuf>,
        session_id: impl Into<String>,
    ) -> Result<Self, AgentsError> {
        let id = session_id.into();
        let name = file_name(&id)?;
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(|e| {
            UserError::new(format!(
                "cannot create session directory {}: {e}",
                dir.display()
            ))
        })?;
        Ok(Self {
            id,
            path: dir.join(format!("{name}.jsonl")),
            lock_path: dir.join(format!("{name}.lock")),
            in_process: Mutex::new(()),
        })
    }

    /// The file this session keeps.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Create an empty session behind an `Arc`, ready for `RunOptions::session`.
    pub fn shared(
        dir: impl Into<PathBuf>,
        session_id: impl Into<String>,
    ) -> Result<std::sync::Arc<Self>, AgentsError> {
        Ok(std::sync::Arc::new(Self::open(dir, session_id)?))
    }

    /// Hold the in-process mutex and the cross-process lock for the rest of the operation.
    fn lock(&self) -> Result<(std::sync::MutexGuard<'_, ()>, File), AgentsError> {
        let guard = self
            .in_process
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        // The lock file is only a place to flock: it is never written, so it is never truncated.
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&self.lock_path)
            .map_err(|e| {
                UserError::new(format!(
                    "cannot open the lock {}: {e}",
                    self.lock_path.display()
                ))
            })?;
        lock.lock_exclusive().map_err(|e| {
            UserError::new(format!("cannot lock {}: {e}", self.lock_path.display()))
        })?;
        Ok((guard, lock))
    }

    fn read_all(&self) -> Result<Vec<Value>, AgentsError> {
        let (_guard, _lock) = self.lock()?;
        self.check_not_symlink()?;
        let file = match File::open(&self.path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => {
                return Err(
                    UserError::new(format!("cannot read {}: {e}", self.path.display())).into(),
                )
            }
        };
        let mut items = Vec::new();
        for (index, line) in BufReader::new(file).lines().enumerate() {
            let line = line
                .map_err(|e| UserError::new(format!("cannot read {}: {e}", self.path.display())))?;
            if line.trim().is_empty() {
                continue;
            }
            // A file is data from outside the process: a broken line is an error, not a skip.
            let value: Value = serde_json::from_str(&line).map_err(|e| {
                UserError::new(format!(
                    "{}: line {} is not JSON: {e}",
                    self.path.display(),
                    index + 1
                ))
            })?;
            items.push(value);
        }
        Ok(items)
    }

    /// Replace the whole file: write a temporary one, then rename it over.
    fn write_all(&self, items: &[Value]) -> Result<(), AgentsError> {
        let (_guard, _lock) = self.lock()?;
        self.check_not_symlink()?;
        let tmp = self.path.with_extension("jsonl.tmp");
        {
            let mut file = File::create(&tmp)
                .map_err(|e| UserError::new(format!("cannot write {}: {e}", tmp.display())))?;
            for item in items {
                serde_json::to_writer(&mut file, item).map_err(|e| {
                    UserError::new(format!(
                        "cannot serialize an item for {}: {e}",
                        tmp.display()
                    ))
                })?;
                file.write_all(b"\n")
                    .map_err(|e| UserError::new(format!("cannot write {}: {e}", tmp.display())))?;
            }
            file.sync_all()
                .map_err(|e| UserError::new(format!("cannot flush {}: {e}", tmp.display())))?;
        }
        std::fs::rename(&tmp, &self.path).map_err(|e| {
            UserError::new(format!(
                "cannot replace {} with {}: {e}",
                self.path.display(),
                tmp.display()
            ))
        })?;
        Ok(())
    }

    fn check_not_symlink(&self) -> Result<(), AgentsError> {
        match std::fs::symlink_metadata(&self.path) {
            Ok(meta) if meta.file_type().is_symlink() => Err(UserError::new(format!(
                "{} is a symlink; refusing to follow it out of the session directory",
                self.path.display()
            ))
            .into()),
            _ => Ok(()),
        }
    }
}

#[async_trait]
impl Session for FileSession {
    fn session_id(&self) -> &str {
        &self.id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        None
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<Value>, AgentsError> {
        let items = self.read_all()?;
        Ok(match limit {
            Some(n) if n < items.len() => items[items.len() - n..].to_vec(),
            _ => items,
        })
    }

    async fn add_items(&self, items: Vec<Value>) -> Result<(), AgentsError> {
        if items.is_empty() {
            return Ok(());
        }
        let (_guard, _lock) = self.lock()?;
        self.check_not_symlink()?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| {
                UserError::new(format!("cannot append to {}: {e}", self.path.display()))
            })?;
        for item in &items {
            serde_json::to_writer(&mut file, item).map_err(|e| {
                UserError::new(format!(
                    "cannot serialize an item for {}: {e}",
                    self.path.display()
                ))
            })?;
            file.write_all(b"\n").map_err(|e| {
                UserError::new(format!("cannot append to {}: {e}", self.path.display()))
            })?;
        }
        file.sync_all()
            .map_err(|e| UserError::new(format!("cannot flush {}: {e}", self.path.display())))?;
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<Value>, AgentsError> {
        let mut items = self.read_all()?;
        let popped = items.pop();
        if popped.is_some() {
            self.write_all(&items)?;
        }
        Ok(popped)
    }

    async fn clear_session(&self) -> Result<(), AgentsError> {
        self.write_all(&[])
    }

    /// One atomic write, so a reader never sees the session emptied and not yet refilled.
    async fn replace_items(&self, items: Vec<Value>) -> Result<(), AgentsError> {
        self.write_all(&items)
    }
}

/// Turn an id into a file name, refusing anything that could leave the directory.
pub(crate) fn file_name(id: &str) -> Result<String, AgentsError> {
    let reject = |why: &str| {
        UserError::new(format!(
            "session id `{id}` cannot be a file name: {why}. Use letters, digits, `_`, `-` and `.`"
        ))
    };
    if id.is_empty() {
        return Err(reject("it is empty").into());
    }
    if id.contains(['/', '\\']) {
        return Err(reject("it contains a path separator").into());
    }
    if id == "." || id == ".." || id.starts_with('.') {
        return Err(reject("it starts with a dot, or is `.` or `..`").into());
    }
    if id.chars().any(|c| c.is_control() || c == '\0') {
        return Err(reject("it contains a control character").into());
    }
    if id.len() > 128 {
        return Err(reject("it is longer than 128 characters").into());
    }
    let cleaned: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    Ok(cleaned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let path = std::env::temp_dir().join(format!("agents-file-session-{name}-{unique}"));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    #[test]
    fn unsafe_session_ids_are_refused() {
        for id in [
            "",
            ".",
            "..",
            ".hidden",
            "a/b",
            "..\\x",
            "/etc/passwd",
            "a\0b",
        ] {
            assert!(
                FileSession::open(std::env::temp_dir(), id).is_err(),
                "`{id}` must be refused"
            );
        }
        // Spaces and other oddities are made safe rather than refused.
        let session = FileSession::open(std::env::temp_dir(), "my session!").expect("accepted");
        assert!(session.path().ends_with("my_session_.jsonl"));
        assert!(
            !FileSession::open(std::env::temp_dir(), "x".repeat(200)).is_ok(),
            "a very long id is refused"
        );
    }

    #[test]
    fn a_symlinked_history_is_not_followed() {
        let dir = dir("symlink");
        let session = FileSession::open(&dir, "s").expect("open");
        std::fs::remove_file(session.path()).ok();
        let outside = std::env::temp_dir().join("agents-file-session-outside.jsonl");
        std::fs::write(&outside, "not an item\n").expect("write");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, session.path()).expect("symlink");
        #[cfg(not(unix))]
        return;
        #[cfg(unix)]
        {
            let error = tokio::runtime::Runtime::new()
                .expect("runtime")
                .block_on(session.get_items(None))
                .expect_err("a symlink is refused");
            assert!(error.to_string().contains("symlink"), "{error}");
        }
    }

    /// The contract every backend shares.
    #[tokio::test]
    async fn file_session_passes_the_contract() {
        let dir = dir("contract");
        let session = FileSession::open(&dir, "contract").expect("open");
        openai_agents_core::session_contract::assert_session_contract(&session).await;
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn file_session_keeps_every_concurrent_write() {
        let dir = dir("concurrent");
        let session = FileSession::shared(&dir, "shared").expect("open");
        openai_agents_core::session_contract::assert_concurrent_writes(session, 16).await;
        std::fs::remove_dir_all(&dir).ok();
    }
}
