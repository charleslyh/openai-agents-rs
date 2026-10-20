//! The file tools through their public constructors (X-019): what the model could ask for, and
//! what it gets. The unit tests check path resolution; these check the tools built on it.

use std::path::{Path, PathBuf};

use opai_agents_core::{FunctionTool, RunContextWrapper, ToolContext};
use opai_agents_tools::{file_list, file_read};
use serde_json::{json, Value};

struct Sandbox {
    root: PathBuf,
    outside: PathBuf,
}

impl Sandbox {
    fn new(name: &str) -> Self {
        let base = std::env::temp_dir().join(format!("agents-tools-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("root");
        let outside = base.join("outside");
        std::fs::create_dir_all(root.join("docs")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(root.join("docs/readme.txt"), "inside").unwrap();
        std::fs::write(root.join("top.txt"), "top").unwrap();
        std::fs::write(outside.join("secret.txt"), "SECRET").unwrap();
        Self { root, outside }
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        if let Some(base) = self.root.parent() {
            let _ = std::fs::remove_dir_all(base);
        }
    }
}

async fn call(tool: &FunctionTool, path: &str) -> Result<Value, String> {
    let args = json!({"path": path}).to_string();
    let ctx = ToolContext::new(
        tool.name.clone(),
        "call-1",
        args.clone(),
        RunContextWrapper::default(),
    );
    (tool.on_invoke_tool)(ctx, args)
        .await
        .map(|out| out.value().cloned().unwrap_or(Value::Null))
        .map_err(|e| e.to_string())
}

#[tokio::test]
async fn reading_inside_the_directory_works() {
    let sandbox = Sandbox::new("read");
    let read = file_read(&sandbox.root).unwrap();
    let out = call(&read, "docs/readme.txt").await.unwrap();
    assert_eq!(out["text"], "inside");
    assert_eq!(call(&read, "top.txt").await.unwrap()["text"], "top");
}

#[tokio::test]
async fn nothing_outside_the_directory_can_be_read() {
    let sandbox = Sandbox::new("escape");
    let read = file_read(&sandbox.root).unwrap();
    let secret = sandbox.outside.join("secret.txt");
    let attempts = [
        "../outside/secret.txt".to_string(),
        "docs/../../outside/secret.txt".to_string(),
        "./../outside/secret.txt".to_string(),
        secret.display().to_string(),
        "/etc/passwd".to_string(),
        "docs/../../../../../../etc/passwd".to_string(),
        "..".to_string(),
    ];
    for attempt in attempts {
        let result = call(&read, &attempt).await;
        match result {
            Err(_) => {}
            Ok(out) => assert!(
                !out.to_string().contains("SECRET") && !out.to_string().contains("root:"),
                "`{attempt}` returned {out}"
            ),
        }
    }
}

#[cfg(unix)]
#[tokio::test]
async fn a_symlink_out_of_the_directory_is_not_followed() {
    use std::os::unix::fs::symlink;
    let sandbox = Sandbox::new("symlink");
    // A link to a file, a link to a directory, a chain of links, and a link that points back in.
    symlink(
        sandbox.outside.join("secret.txt"),
        sandbox.root.join("link-file"),
    )
    .unwrap();
    symlink(&sandbox.outside, sandbox.root.join("link-dir")).unwrap();
    symlink(
        sandbox.root.join("link-file"),
        sandbox.root.join("link-chain"),
    )
    .unwrap();
    symlink(sandbox.root.join("top.txt"), sandbox.root.join("link-in")).unwrap();

    let read = file_read(&sandbox.root).unwrap();
    let list = file_list(&sandbox.root).unwrap();

    for path in ["link-file", "link-dir/secret.txt", "link-chain"] {
        let result = call(&read, path).await;
        assert!(result.is_err(), "`{path}` left the directory: {result:?}");
    }
    assert!(call(&list, "link-dir").await.is_err());
    // A link that stays inside is fine.
    assert_eq!(call(&read, "link-in").await.unwrap()["text"], "top");
}

#[cfg(unix)]
#[tokio::test]
async fn a_root_that_is_itself_a_symlink_is_resolved_once() {
    use std::os::unix::fs::symlink;
    let sandbox = Sandbox::new("root-link");
    let alias = sandbox.root.parent().unwrap().join("alias");
    symlink(&sandbox.root, &alias).unwrap();
    let read = file_read(&alias).unwrap();
    assert_eq!(call(&read, "top.txt").await.unwrap()["text"], "top");
    assert!(call(&read, "../outside/secret.txt").await.is_err());
}

#[tokio::test]
async fn listing_stays_inside_and_hides_nothing_it_may_show() {
    let sandbox = Sandbox::new("list");
    let list = file_list(&sandbox.root).unwrap();
    let out = call(&list, "").await.unwrap();
    let names: Vec<&str> = out["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["docs", "top.txt"]);
    assert!(call(&list, "..").await.is_err());
    assert!(call(&list, "top.txt").await.is_err(), "not a directory");
}

#[tokio::test]
async fn a_file_over_the_limit_is_not_read() {
    let sandbox = Sandbox::new("big");
    let big = vec![b'x'; (opai_agents_tools::MAX_FILE_BYTES + 1) as usize];
    std::fs::write(sandbox.root.join("big.txt"), big).unwrap();
    let read = file_read(&sandbox.root).unwrap();
    let err = call(&read, "big.txt").await.expect_err("too big");
    assert!(err.contains("largest file"), "{err}");
}

#[tokio::test]
async fn a_binary_file_is_refused_not_mangled() {
    let sandbox = Sandbox::new("binary");
    std::fs::write(sandbox.root.join("blob.bin"), [0xff, 0xfe, 0x00, 0x01]).unwrap();
    let read = file_read(&sandbox.root).unwrap();
    let err = call(&read, "blob.bin").await.expect_err("not text");
    assert!(err.contains("not UTF-8"), "{err}");
}

#[test]
fn a_directory_that_does_not_exist_is_an_error_at_construction() {
    assert!(file_read(Path::new("/definitely/not/here")).is_err());
    assert!(file_list(Path::new("/definitely/not/here")).is_err());
}

#[test]
fn a_file_is_not_a_root() {
    let sandbox = Sandbox::new("file-root");
    assert!(file_read(sandbox.root.join("top.txt")).is_err());
}
