//! Run results (Python: `agents.result` subset).

use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::error::AgentsError;
use crate::items::{ItemHelpers, InputLike, ModelResponse, ResponseInputItem, RunItem};
use crate::stream_events::StreamEvent;
use crate::usage::Usage;

/// Result of a completed `Runner::run` (Python: `RunResult`).
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Original input.
    pub input: InputLike,
    /// New items generated during the run.
    pub new_items: Vec<RunItem>,
    /// Raw model responses.
    pub raw_responses: Vec<ModelResponse>,
    /// Final output (string or JSON value).
    pub final_output: serde_json::Value,
    /// Last agent that ran.
    pub last_agent_name: String,
    /// Max turns configured for the run.
    pub max_turns: Option<usize>,
    /// Aggregated usage across raw responses.
    pub usage: Usage,
}

impl RunResult {
    /// Name of the last agent (Python: `last_agent.name`).
    pub fn last_agent<'a>(&self, starting: &'a Agent) -> &'a Agent {
        let _ = &self.last_agent_name;
        starting
    }

    /// Convenience: last response id.
    pub fn last_response_id(&self) -> Option<&str> {
        self.raw_responses
            .last()
            .and_then(|r| r.response_id.as_deref())
    }

    /// Build a continuation input list (Python: `to_input_list`).
    pub fn to_input_list(&self) -> Vec<ResponseInputItem> {
        let mut items = ItemHelpers::input_to_new_input_list(&self.input);
        for item in &self.new_items {
            items.push(item.raw_item().clone());
        }
        items
    }

    /// Final output as a string when possible.
    pub fn final_output_as_str(&self) -> Option<&str> {
        self.final_output.as_str()
    }
}

/// Shared mutable snapshot for a streaming run.
#[derive(Debug, Default)]
pub struct StreamingSnapshot {
    /// Current agent name.
    pub current_agent_name: String,
    /// Current turn number.
    pub current_turn: usize,
    /// Whether the run finished.
    pub is_complete: bool,
    /// Final output once complete.
    pub final_output: Option<serde_json::Value>,
    /// Items accumulated so far.
    pub new_items: Vec<RunItem>,
    /// Raw responses accumulated so far.
    pub raw_responses: Vec<ModelResponse>,
    /// Usage so far.
    pub usage: Usage,
    /// Error if the background task failed after events stopped.
    pub error: Option<String>,
}

/// Streaming run handle (Python: `RunResultStreaming`).
pub struct RunResultStreaming {
    /// Live snapshot updated by the background loop.
    pub snapshot: Arc<Mutex<StreamingSnapshot>>,
    /// Max turns configured.
    pub max_turns: Option<usize>,
    rx: mpsc::Receiver<Result<StreamEvent, AgentsError>>,
}

impl RunResultStreaming {
    /// Create from channel + snapshot (used by `Runner::run_streamed`).
    pub(crate) fn new(
        snapshot: Arc<Mutex<StreamingSnapshot>>,
        max_turns: Option<usize>,
        rx: mpsc::Receiver<Result<StreamEvent, AgentsError>>,
    ) -> Self {
        Self {
            snapshot,
            max_turns,
            rx,
        }
    }

    /// Whether the agent has finished running.
    pub fn is_complete(&self) -> bool {
        self.snapshot.lock().expect("snapshot").is_complete
    }

    /// Current agent name.
    pub fn current_agent_name(&self) -> String {
        self.snapshot.lock().expect("snapshot").current_agent_name.clone()
    }

    /// Final output once available.
    pub fn final_output(&self) -> Option<serde_json::Value> {
        self.snapshot.lock().expect("snapshot").final_output.clone()
    }

    /// Receive the next stream event.
    ///
    /// Returns `None` when the stream is exhausted. Errors from the run loop are returned as
    /// `Err` on the event that failed (and may also be stored on the snapshot).
    pub async fn next_event(&mut self) -> Option<Result<StreamEvent, AgentsError>> {
        self.rx.recv().await
    }

    /// Collect all remaining events into a vector (convenience for tests).
    pub async fn collect_events(&mut self) -> Result<Vec<StreamEvent>, AgentsError> {
        let mut out = Vec::new();
        while let Some(ev) = self.next_event().await {
            out.push(ev?);
        }
        let snap = self.snapshot.lock().expect("snapshot");
        if let Some(err) = &snap.error {
            if !snap.is_complete {
                return Err(AgentsError::Internal(err.clone()));
            }
        }
        Ok(out)
    }
}
