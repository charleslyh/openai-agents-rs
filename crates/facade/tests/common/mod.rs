//! Helpers shared by the integration tests that need files from the repository.
//!
//! The tests live in `crates/facade`, but the oracle venv, the vendored Python SDK and the
//! `conformance/` assets sit at the repository root, so `CARGO_MANIFEST_DIR` is not the place to
//! look for them. Every test that locates one of those goes through here, so a move of either
//! side is a one-line change instead of a silent skip.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// The repository root (the directory that holds `conformance/`, `vendor/` and `.venv/`).
pub fn repo_root() -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/facade sits two levels below the repository root")
        .to_path_buf();
    assert!(
        root.join("conformance").is_dir(),
        "repository root not found at {}: expected a `conformance/` directory there",
        root.display()
    );
    root
}

/// The oracle venv's interpreter, or `None` when the venv has not been created.
///
/// Tests that need it skip themselves in that case, so make the skip visible: a missing venv
/// means these tests checked nothing, and CI creates the venv before running them.
pub fn oracle_python() -> Option<PathBuf> {
    let python = repo_root().join(".venv/bin/python");
    if python.exists() {
        Some(python)
    } else {
        eprintln!("skipping: no .venv (run conformance/setup_venv.sh)");
        None
    }
}
