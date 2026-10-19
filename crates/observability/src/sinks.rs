//! Where exported events go: stdout, or a JSONL file.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

use super::batched::EventSink;
use super::export::ExportEvent;

/// Print one line per event, for a terminal or a container log.
///
/// ```rust
/// use openai_agents_observability::{BatchedProcessor, ConsoleSink};
///
/// # fn install() -> std::sync::Arc<BatchedProcessor> {
/// BatchedProcessor::spawn(ConsoleSink, 1024, std::time::Duration::from_millis(200))
/// # }
/// ```
#[derive(Debug, Default)]
pub struct ConsoleSink;

impl EventSink for ConsoleSink {
    fn write(&mut self, events: &[ExportEvent]) -> Result<(), String> {
        for event in events {
            println!("{}", event.to_json());
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "console"
    }
}

/// Append one JSON object per line to a file: the shape a log shipper and `jq` both read.
///
/// The file is opened in append mode for each batch, so rotating it underneath is fine.
#[derive(Debug)]
pub struct JsonlSink {
    path: PathBuf,
}

impl JsonlSink {
    /// Append to `path`, creating it if needed. Returns an error when the file cannot be opened, so
    /// a bad path is reported instead of silently exporting nothing.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, String> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
            }
        }
        // Touch the file, so a path that cannot be written is an error now.
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        Ok(Self { path })
    }

    /// The file events are written to.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl EventSink for JsonlSink {
    fn write(&mut self, events: &[ExportEvent]) -> Result<(), String> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| format!("cannot open {}: {e}", self.path.display()))?;
        for event in events {
            serde_json::to_writer(&mut file, &event.to_json())
                .map_err(|e| format!("cannot serialize an event: {e}"))?;
            file.write_all(b"\n")
                .map_err(|e| format!("cannot write {}: {e}", self.path.display()))?;
        }
        file.flush()
            .map_err(|e| format!("cannot flush {}: {e}", self.path.display()))
    }

    fn name(&self) -> &'static str {
        "jsonl"
    }
}
