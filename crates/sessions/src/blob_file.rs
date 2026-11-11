//! Offloaded tool outputs on disk: one file per output, one directory per namespace.
//!
//! [`FileBlobStore`] is the file-backed [`BlobStore`] for
//! [`OffloadToolOutputs`](opai_agents_core::OffloadToolOutputs). Layout:
//! `<root>/<namespace digest>/<reference>.txt`.
//!
//! * The namespace never reaches a path as text: the directory is named after a digest of it, so
//!   `../x` or an absolute path in a namespace cannot leave `root`, and two namespaces cannot
//!   collide through character replacement.
//! * References must have the shape made by `reference_for` (32 lowercase hex characters); anything
//!   else is refused before it touches the file system.
//! * A write goes to a temporary file that is renamed into place, so a reader never sees half a
//!   file. Writing a reference that already exists changes nothing.
//! * A symlinked namespace directory is not followed out of `root`.
//! * On Unix, files are `0600` and directories `0700`: offloaded outputs can hold anything a tool
//!   returned.
//!
//! The text is written as it is, not encrypted. Put `root` where only the process can read it.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use opai_agents_core::{is_valid_reference, reference_for, AgentsError, BlobStore, UserError};

/// Most bytes one blob may hold. A larger one is refused, and the history keeps its output.
pub const MAX_BLOB_BYTES: usize = 16 * 1024 * 1024;

/// A [`BlobStore`] that keeps each blob in a file under `root`.
#[derive(Debug, Clone)]
pub struct FileBlobStore {
    root: PathBuf,
}

impl FileBlobStore {
    /// Store under `root`, created when missing.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, AgentsError> {
        let root = root.into();
        create_dir(&root)?;
        Ok(Self { root })
    }

    /// Like [`open`](Self::open), shared.
    pub fn shared(root: impl Into<PathBuf>) -> Result<Arc<Self>, AgentsError> {
        Ok(Arc::new(Self::open(root)?))
    }

    /// The directory the blobs live under.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn namespace_dir(&self, namespace: &str) -> PathBuf {
        self.root.join(reference_for(namespace))
    }
}

fn io_error(action: &str, path: &Path, error: std::io::Error) -> AgentsError {
    UserError::new(format!("cannot {action} {}: {error}", path.display())).into()
}

fn create_dir(path: &Path) -> Result<(), AgentsError> {
    fs::create_dir_all(path).map_err(|e| io_error("create", path, e))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .map_err(|e| io_error("restrict", path, e))?;
    }
    Ok(())
}

fn refuse_symlink(path: &Path) -> Result<(), AgentsError> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => Err(UserError::new(format!(
            "{} is a symlink; refusing to follow it out of the blob directory",
            path.display()
        ))
        .into()),
        _ => Ok(()),
    }
}

fn check_key(key: &str) -> Result<(), AgentsError> {
    if is_valid_reference(key) {
        Ok(())
    } else {
        Err(UserError::new("a blob key must be 32 lowercase hex characters").into())
    }
}

fn write_blob(dir: &Path, key: &str, text: &str) -> Result<(), AgentsError> {
    if text.len() > MAX_BLOB_BYTES {
        return Err(UserError::new(format!(
            "a blob may hold at most {MAX_BLOB_BYTES} bytes, got {}",
            text.len()
        ))
        .into());
    }
    refuse_symlink(dir)?;
    create_dir(dir)?;
    let path = dir.join(format!("{key}.txt"));
    if path.exists() {
        return Ok(());
    }
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let tmp = dir.join(format!("{key}.{}.{unique}.tmp", std::process::id()));
    {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&tmp).map_err(|e| io_error("write", &tmp, e))?;
        file.write_all(text.as_bytes())
            .and_then(|()| file.flush())
            .map_err(|e| io_error("write", &tmp, e))?;
    }
    fs::rename(&tmp, &path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        io_error("move into place", &path, e)
    })
}

fn read_blob(dir: &Path, key: &str) -> Result<Option<String>, AgentsError> {
    refuse_symlink(dir)?;
    let path = dir.join(format!("{key}.txt"));
    refuse_symlink(&path)?;
    match fs::read_to_string(&path) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(io_error("read", &path, e)),
    }
}

fn remove_namespace(dir: &Path) -> Result<(), AgentsError> {
    refuse_symlink(dir)?;
    match fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(io_error("remove", dir, e)),
    }
}

async fn blocking<T, F>(work: F) -> Result<T, AgentsError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, AgentsError> + Send + 'static,
{
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| AgentsError::internal(format!("blob task failed: {e}")))?
}

#[async_trait]
impl BlobStore for FileBlobStore {
    async fn put(&self, namespace: &str, key: &str, text: &str) -> Result<(), AgentsError> {
        check_key(key)?;
        let dir = self.namespace_dir(namespace);
        let (key, text) = (key.to_string(), text.to_string());
        blocking(move || write_blob(&dir, &key, &text)).await
    }

    async fn get(&self, namespace: &str, key: &str) -> Result<Option<String>, AgentsError> {
        check_key(key)?;
        let dir = self.namespace_dir(namespace);
        let key = key.to_string();
        blocking(move || read_blob(&dir, &key)).await
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), AgentsError> {
        let dir = self.namespace_dir(namespace);
        blocking(move || remove_namespace(&dir)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(name: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        std::env::temp_dir().join(format!("agents-blobs-{name}-{unique}"))
    }

    #[tokio::test]
    async fn a_blob_round_trips_and_writing_it_again_is_harmless() {
        let dir = root("roundtrip");
        let store = FileBlobStore::open(&dir).unwrap();
        let key = reference_for("hello");
        store.put("s1", &key, "hello").await.unwrap();
        store.put("s1", &key, "hello").await.unwrap();
        assert_eq!(
            store.get("s1", &key).await.unwrap().as_deref(),
            Some("hello")
        );
        assert_eq!(store.get("s1", &reference_for("nope")).await.unwrap(), None);
        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn namespaces_are_kept_apart_and_can_be_deleted() {
        let dir = root("namespaces");
        let store = FileBlobStore::open(&dir).unwrap();
        let key = reference_for("x");
        store.put("a", &key, "x").await.unwrap();
        assert_eq!(store.get("b", &key).await.unwrap(), None);
        store.put("b", &key, "x").await.unwrap();
        store.delete_namespace("a").await.unwrap();
        assert_eq!(store.get("a", &key).await.unwrap(), None);
        assert!(store.get("b", &key).await.unwrap().is_some());
        store.delete_namespace("never-used").await.unwrap();
        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_hostile_namespace_or_key_cannot_leave_the_root() {
        let dir = root("hostile");
        let store = FileBlobStore::open(&dir).unwrap();
        let key = reference_for("x");
        for namespace in ["../escape", "/etc", "a/../../b", ""] {
            store.put(namespace, &key, "x").await.unwrap();
            assert!(store.namespace_dir(namespace).starts_with(&dir));
        }
        assert!(!dir.parent().unwrap().join("escape").exists());
        for bad in ["../../etc/passwd", "", &"A".repeat(32), "abc"] {
            assert!(store.put("n", bad, "x").await.is_err(), "{bad}");
            assert!(store.get("n", bad).await.is_err(), "{bad}");
        }
        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn an_oversized_blob_is_refused() {
        let dir = root("big");
        let store = FileBlobStore::open(&dir).unwrap();
        let text = "x".repeat(MAX_BLOB_BYTES + 1);
        assert!(store.put("n", &reference_for(&text), &text).await.is_err());
        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlinked_namespace_directory_is_not_followed() {
        let dir = root("symlink");
        let outside = root("outside");
        fs::create_dir_all(&outside).unwrap();
        let store = FileBlobStore::open(&dir).unwrap();
        std::os::unix::fs::symlink(&outside, store.namespace_dir("n")).unwrap();
        let key = reference_for("x");
        assert!(store.put("n", &key, "x").await.is_err());
        assert!(store.get("n", &key).await.is_err());
        assert!(fs::read_dir(&outside).unwrap().next().is_none());
        let _ = fs::remove_dir_all(dir);
        let _ = fs::remove_dir_all(outside);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn files_are_private_to_the_owner() {
        use std::os::unix::fs::PermissionsExt;
        let dir = root("perms");
        let store = FileBlobStore::open(&dir).unwrap();
        let key = reference_for("x");
        store.put("n", &key, "x").await.unwrap();
        let ns = store.namespace_dir("n");
        let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&ns), 0o700);
        assert_eq!(mode(&ns.join(format!("{key}.txt"))), 0o600);
        let _ = fs::remove_dir_all(dir);
    }
}
