//! Run results (Python: `agents.result` subset).

use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::error::AgentsError;
use crate::guardrail::{InputGuardrailResult, OutputGuardrailResult};
use crate::items::{ItemHelpers, InputLike, ModelResponse, ResponseInputItem, RunItem, ToolApprovalItem};
use crate::model_settings::ModelSettings;
use crate::run_state::{ApprovalStore, RunState};
use crate::stream_events::StreamEvent;
use crate::usage::Usage;

/// Result of a completed or interrupted `Runner::run` (Python: `RunResult`).
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Original input.
    pub input: InputLike,
    /// New items generated during the run.
    pub new_items: Vec<RunItem>,
    /// Raw model responses.
    pub raw_responses: Vec<ModelResponse>,
    /// Final output (string or JSON value). `Null` when interrupted pending approval.
    pub final_output: serde_json::Value,
    /// Last agent that ran (Python: `RunResult.last_agent`).
    pub last_agent: Arc<Agent>,
    /// Name of the last agent that ran.
    pub last_agent_name: String,
    /// Max turns configured for the run.
    pub max_turns: Option<usize>,
    /// Aggregated usage across raw responses.
    pub usage: Usage,
    /// Pending tool approvals (Python: `interruptions`). Empty when the run finished.
    pub interruptions: Vec<ToolApprovalItem>,
    /// Results of the input guardrails that ran (Python: `input_guardrail_results`).
    pub input_guardrail_results: Vec<InputGuardrailResult>,
    /// Results of the output guardrails that ran (Python: `output_guardrail_results`).
    pub output_guardrail_results: Vec<OutputGuardrailResult>,
    /// Internal snapshot used by [`Self::to_state`] when interrupted.
    pub(crate) interrupt_state: Option<InterruptSnapshot>,
}

/// Fields needed to rebuild [`RunState`] after an interruption.
#[derive(Debug, Clone)]
pub(crate) struct InterruptSnapshot {
    pub(crate) starting_agent_name: String,
    pub(crate) current_input_items: Vec<ResponseInputItem>,
    pub(crate) turn: usize,
    pub(crate) previous_response_id: Option<String>,
    pub(crate) model_settings: ModelSettings,
    pub(crate) pending_response: ModelResponse,
    pub(crate) approvals: ApprovalStore,
    pub(crate) nested_agent_runs: std::collections::HashMap<String, RunState>,
}

impl RunResult {
    /// The last agent that ran (Python: `RunResult.last_agent`).
    ///
    /// After a handoff this is the agent the run was handed off to, not the starting agent.
    pub fn last_agent(&self) -> &Agent {
        self.last_agent.as_ref()
    }

    /// Parse the final output into a concrete type (Python: `final_output` is already typed
    /// via `output_type`).
    pub fn final_output_as<T: serde::de::DeserializeOwned>(&self) -> Result<T, AgentsError> {
        serde_json::from_value::<T>(self.final_output.clone()).map_err(|e| {
            AgentsError::tool(format!(
                "could not decode final output as {}: {e}",
                std::any::type_name::<T>()
            ))
        })
    }

    /// Whether the run paused for human approval.
    pub fn is_interrupted(&self) -> bool {
        !self.interruptions.is_empty()
    }

    /// Convenience: last response id.
    pub fn last_response_id(&self) -> Option<&str> {
        self.raw_responses
            .last()
            .and_then(|r| r.response_id.as_deref())
    }

    /// Build a continuation input list (Python: `to_input_list`).
    ///
    /// Skips [`RunItem::ToolApproval`] entries (they are not model inputs).
    pub fn to_input_list(&self) -> Vec<ResponseInputItem> {
        let mut items = ItemHelpers::input_to_new_input_list(&self.input);
        for item in &self.new_items {
            if item.is_model_input() {
                items.push(item.raw_item().clone());
            }
        }
        items
    }

    /// Final output as a string when possible.
    pub fn final_output_as_str(&self) -> Option<&str> {
        self.final_output.as_str()
    }

    /// Convert an interrupted result into a resumable [`RunState`] (Python: `to_state`).
    pub fn to_state(&self) -> Result<RunState, AgentsError> {
        let snap = self.interrupt_state.as_ref().ok_or_else(|| {
            AgentsError::User(crate::error::UserError::new(
                "RunResult.to_state() requires a paused run with interruptions",
            ))
        })?;
        if self.interruptions.is_empty() {
            return Err(AgentsError::User(crate::error::UserError::new(
                "RunResult.to_state() requires non-empty interruptions",
            )));
        }
        Ok(RunState {
            input: self.input.clone(),
            starting_agent_name: snap.starting_agent_name.clone(),
            current_agent_name: self.last_agent_name.clone(),
            current_input_items: snap.current_input_items.clone(),
            generated_items: self.new_items.clone(),
            raw_responses: self.raw_responses.clone(),
            usage: self.usage.clone(),
            max_turns: self.max_turns.unwrap_or(crate::run::DEFAULT_MAX_TURNS),
            turn: snap.turn,
            previous_response_id: snap.previous_response_id.clone(),
            model_settings: snap.model_settings.clone(),
            interruptions: self.interruptions.clone(),
            pending_response: snap.pending_response.clone(),
            approvals: snap.approvals.clone(),
            nested_agent_runs: snap.nested_agent_runs.clone(),
        })
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
    /// Pending interruptions when the stream paused for HITL.
    pub interruptions: Vec<ToolApprovalItem>,
    /// Error if the background task failed after events stopped.
    pub error: Option<String>,
    /// A graceful cancel was requested: stop before the next turn begins.
    pub cancel_after_turn: bool,
    /// The run was cancelled, so `final_output` is intentionally absent.
    pub is_cancelled: bool,
}

/// How [`RunResultStreaming::cancel`] stops a run (Python: `cancel(mode=...)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CancelMode {
    /// Stop now: the background task is aborted and the event queue is closed.
    #[default]
    Immediate,
    /// Let the current turn (model call and tool calls) finish, then stop before the next one.
    AfterTurn,
}

/// Streaming run handle (Python: `RunResultStreaming`).
pub struct RunResultStreaming {
    /// Live snapshot updated by the background loop.
    pub snapshot: Arc<Mutex<StreamingSnapshot>>,
    /// Max turns configured.
    pub max_turns: Option<usize>,
    rx: mpsc::Receiver<Result<StreamEvent, AgentsError>>,
    task: Option<tokio::task::JoinHandle<()>>,
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
            task: None,
        }
    }

    /// Remember the background task so [`cancel`](Self::cancel) can abort it.
    pub(crate) fn with_task(mut self, task: tokio::task::JoinHandle<()>) -> Self {
        self.task = Some(task);
        self
    }

    /// Cancel the run (Python: `RunResultStreaming.cancel`).
    ///
    /// `Immediate` aborts the background task and drops queued events. `AfterTurn` lets the
    /// current turn finish and stops before the next one; keep consuming events until the stream
    /// ends. Either way the run reports no `final_output` and [`is_cancelled`](Self::is_cancelled)
    /// becomes true. Cancelling a finished run has no effect.
    pub fn cancel(&mut self, mode: CancelMode) {
        let mut snap = self.snapshot.lock().expect("snapshot");
        if snap.is_complete {
            return;
        }
        match mode {
            CancelMode::AfterTurn => snap.cancel_after_turn = true,
            CancelMode::Immediate => {
                snap.is_cancelled = true;
                snap.is_complete = true;
                snap.final_output = None;
                drop(snap);
                if let Some(task) = &self.task {
                    task.abort();
                }
                self.rx.close();
                while self.rx.try_recv().is_ok() {}
            }
        }
    }

    /// Whether the run was cancelled before producing a final output.
    pub fn is_cancelled(&self) -> bool {
        self.snapshot.lock().expect("snapshot").is_cancelled
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
                return Err(AgentsError::internal(err.clone()));
            }
        }
        Ok(out)
    }
}
