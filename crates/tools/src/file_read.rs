//! Reading files inside one directory, and nowhere else (Rust-only, X-019).
//!
//! "Read the config" is a reasonable thing for an agent to do; "read `/etc/shadow`" is not, and the
//! difference is one `../../../`. These tools take a directory at construction time and answer only
//! for paths inside it:
//!
//! * the root is canonicalised once, so a symlinked root is resolved before anything is compared;
//! * a requested path is joined to the root and canonicalised, which resolves `..` and any symlink,
//!   and then has to start with the root — a symlink that leaves the directory therefore leaves the
//!   answer as an error, even though it "looks" like it is inside;
//! * an absolute path is refused: everything the tool reads is named relative to its root;
//! * a file over [`MAX_FILE_BYTES`] is refused instead of being read into memory.
//!
//! Writing is not part of these tools. An agent that must write needs a tool built for that, with
//! its own rules, not a read tool with the check turned off.

use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use openai_agents_core::error::UserError;
use openai_agents_core::{AgentsError, FunctionTool};

/// The largest file `file_read` returns.
pub const MAX_FILE_BYTES: u64 = 1024 * 1024;

/// The most entries `file_list` returns in one answer.
pub const MAX_LIST_ENTRIES: usize = 500;

/// Arguments of the `file_read` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FileArgs {
    /// Path relative to the tool's directory, e.g. `notes/today.md`. Absolute paths are refused.
    pub path: String,
}

/// A tool that reads one file, as text, from `dir`.
pub fn file_read(dir: impl Into<PathBuf>) -> Result<FunctionTool, AgentsError> {
    let root = root_of(dir)?;
    Ok(FunctionTool::typed(
        "file_read",
        "Read a file as text. The path is relative to the tool's directory; absolute paths and \
         anything outside it are refused.",
        move |_ctx, args: FileArgs| {
            let root = root.clone();
            async move {
                let path = resolve(&root, &args.path)?;
                let (text, bytes) = read_text(&path, MAX_FILE_BYTES)?;
                Ok(json!({"path": args.path, "text": text, "bytes": bytes}))
            }
        },
    ))
}

/// A tool that lists one directory inside `dir`.
pub fn file_list(dir: impl Into<PathBuf>) -> Result<FunctionTool, AgentsError> {
    let root = root_of(dir)?;
    Ok(FunctionTool::typed(
        "file_list",
        "List the files and directories in one directory of the tool's directory. The path is \
         relative to it; absolute paths and anything outside it are refused.",
        move |_ctx, args: FileArgs| {
            let root = root.clone();
            async move {
                let path = resolve(&root, &args.path)?;
                if !path.is_dir() {
                    return Err(tool_err(format!("{} is not a directory", args.path)));
                }
                let mut entries = Vec::new();
                for entry in std::fs::read_dir(&path)
                    .map_err(|e| tool_err(format!("cannot list {}: {e}", args.path)))?
                {
                    let entry =
                        entry.map_err(|e| tool_err(format!("cannot list {}: {e}", args.path)))?;
                    let name = entry.file_name().to_string_lossy().to_string();
                    let kind = entry
                        .file_type()
                        .map(|t| if t.is_dir() { "directory" } else { "file" })
                        .unwrap_or("other");
                    entries.push(json!({"name": name, "kind": kind}));
                    if entries.len() >= MAX_LIST_ENTRIES {
                        break;
                    }
                }
                entries.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
                Ok(json!({"path": args.path, "entries": entries}))
            }
        },
    ))
}

/// Canonicalise the directory once: a symlinked root resolved now is a root compared the same way
/// for every later request.
fn root_of(dir: impl Into<PathBuf>) -> Result<PathBuf, AgentsError> {
    let dir = dir.into();
    let root = dir.canonicalize().map_err(|e| {
        tool_err(format!(
            "{} is not a directory it can use: {e}",
            dir.display()
        ))
    })?;
    if !root.is_dir() {
        return Err(tool_err(format!("{} is not a directory", root.display())));
    }
    Ok(root)
}

/// `requested` inside `root`, or an error: this is the check the whole tool is for.
pub fn resolve(root: &Path, requested: &str) -> Result<PathBuf, AgentsError> {
    let requested = requested.trim();
    if requested.is_empty() {
        return Ok(root.to_path_buf());
    }
    if requested.starts_with('/') || Path::new(requested).is_absolute() {
        return Err(tool_err(format!(
            "{requested} is absolute: name it relative to the tool's directory"
        )));
    }
    let candidate = root.join(requested);
    let resolved = candidate.canonicalize().map_err(|_| {
        tool_err(format!(
            "{requested} does not exist inside the tool's directory"
        ))
    })?;
    if !resolved.starts_with(root) {
        return Err(tool_err(format!(
            "{requested} is outside the tool's directory"
        )));
    }
    Ok(resolved)
}

/// Read `path` as text, refusing a file bigger than `max_bytes`.
fn read_text(path: &Path, max_bytes: u64) -> Result<(String, u64), AgentsError> {
    let bytes = std::fs::metadata(path)
        .map_err(|e| tool_err(format!("cannot read {}: {e}", path.display())))?
        .len();
    if bytes > max_bytes {
        return Err(tool_err(format!(
            "{} is {bytes} bytes, and the largest file this tool reads is {max_bytes}",
            path.display()
        )));
    }
    let text = std::fs::read_to_string(path)
        .map_err(|_| tool_err(format!("{} is not UTF-8 text", path.display())))?;
    Ok((text, bytes))
}

fn tool_err(message: impl Into<String>) -> AgentsError {
    AgentsError::from(UserError::new(format!("files: {}", message.into())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory of its own per test: tests run in parallel inside one process, so a name made
    /// from the process id alone would be shared, and one test would delete another's files.
    fn sandbox() -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "agents-files-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("inside")).expect("create");
        std::fs::write(dir.join("inside/notes.txt"), "hello").expect("write");
        std::fs::write(dir.join("root.txt"), "root").expect("write");
        dir
    }

    #[test]
    fn a_path_inside_is_resolved() {
        let dir = sandbox();
        let root = root_of(&dir).expect("root");
        assert_eq!(
            std::fs::read_to_string(resolve(&root, "inside/notes.txt").expect("resolve")).unwrap(),
            "hello"
        );
        assert_eq!(resolve(&root, "").unwrap(), root);
        // `..` that stays inside is fine; this one leaves.
        assert!(resolve(&root, "../outside").is_err());
        assert!(resolve(&root, "/etc/passwd").is_err());
        assert!(resolve(&root, "nope.txt").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The classic escape: a link that sits inside and points outside.
    #[test]
    fn a_symlink_that_leaves_is_refused() {
        let dir = sandbox();
        let outside = std::env::temp_dir().join(format!(
            "agents-outside-{}-{}",
            std::process::id(),
            dir.file_name().and_then(|n| n.to_str()).unwrap_or("x")
        ));
        std::fs::create_dir_all(&outside).expect("create");
        std::fs::write(outside.join("secret.txt"), "secret").expect("write");
        let root = root_of(&dir).expect("root");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, dir.join("escape")).expect("symlink");

        #[cfg(unix)]
        assert!(
            resolve(&root, "escape").is_err() || resolve(&root, "escape/secret.txt").is_err(),
            "a link out of the sandbox must not be followed"
        );

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&outside).ok();
    }

    /// A file over the limit is refused instead of being read into memory.
    #[test]
    fn a_file_over_the_limit_is_refused() {
        let dir = sandbox();
        std::fs::write(dir.join("big.txt"), "x".repeat(64)).expect("write");
        let path = resolve(&root_of(&dir).expect("root"), "big.txt").expect("resolve");
        assert!(
            read_text(&path, 8).is_err(),
            "64 bytes over an 8 byte limit"
        );
        assert_eq!(read_text(&path, 64).expect("read").1, 64);
        std::fs::remove_dir_all(&dir).ok();
    }
}
