//! The agent loop (Python: `run.py::_run_impl` and `run_internal/run_loop.py`):
//! one iteration per model call, with tools, handoffs, guardrails and session saving.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::error::AgentsError;
use crate::items::{InputLike, ItemHelpers};
use crate::memory::prepare_input_with_session;
use crate::result::{RunResult, StreamingSnapshot};
use crate::run_state::RunState;
use crate::stream_events::StreamEvent;
use crate::tool_guardrails::{ToolInputGuardrailResult, ToolOutputGuardrailResult};

use super::config::RunOptions;
use super::session_writer::SessionWriter;
use super::turn::validate_run_options;
use super::turn_loop::{LoopInit, TurnLoop};

tokio::task_local! {
    pub(crate) static NESTED_RESUME_STATES: RefCell<HashMap<String, RunState>>;
}

/// Take a nested resume state for an outer tool call id (used by `Agent.as_tool`).
pub(crate) fn take_nested_resume_state(call_id: &str) -> Option<RunState> {
    NESTED_RESUME_STATES
        .try_with(|cell| cell.borrow_mut().remove(call_id))
        .ok()
        .flatten()
}

/// Tool guardrail results collected while a run executes tools.
#[derive(Debug, Default)]
pub(crate) struct ToolGuardrailLog {
    pub(crate) input: Vec<ToolInputGuardrailResult>,
    pub(crate) output: Vec<ToolOutputGuardrailResult>,
}

pub(crate) type SharedToolGuardrailLog = Arc<Mutex<ToolGuardrailLog>>;

pub(crate) type EventTx = mpsc::Sender<Result<StreamEvent, AgentsError>>;

/// A spawned child task that is aborted when its owner goes away.
///
/// `tokio::spawn` detaches: dropping a `JoinHandle` leaves the task running. A run that is
/// cancelled (`CancelMode::Immediate` aborts the run task, which drops its futures) or fails
/// must not leave its input guardrails or event forwarder running, so children are held in this
/// wrapper. Awaiting it behaves like awaiting the handle.
pub(super) struct AbortOnDrop<T>(pub(super) tokio::task::JoinHandle<T>);

impl<T> std::future::Future for AbortOnDrop<T> {
    type Output = Result<T, tokio::task::JoinError>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        std::pin::Pin::new(&mut self.0).poll(cx)
    }
}

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        // A no-op once the task has finished.
        self.0.abort();
    }
}

pub(crate) enum LoopStart {
    /// `prepared` is the model input built from the session, when there is one.
    Fresh {
        input: InputLike,
        prepared: Option<Vec<Value>>,
    },
    /// Boxed: a `RunState` carries the whole paused transcript, and `Fresh` is tiny, so the
    /// unboxed variant made every `LoopStart` as large as a run snapshot.
    Resume { state: Box<RunState> },
}

pub(crate) async fn emit(tx: &Option<EventTx>, event: StreamEvent) {
    if let Some(tx) = tx {
        let _ = tx.send(Ok(event)).await;
    }
}

/// Run the loop with the run's tracing switch in scope.
///
/// Spans created anywhere inside the run must respect `RunConfig.tracing_disabled`, not just
/// the global switch (Python: `RunConfig.tracing_disabled` short-circuits the whole trace).
pub(crate) async fn run_loop(
    starting_agent: Agent,
    start: LoopStart,
    options: RunOptions,
    events: Option<EventTx>,
    snapshot: Option<Arc<Mutex<StreamingSnapshot>>>,
) -> Result<RunResult, AgentsError> {
    let run_tracing_disabled = options.run_config.tracing.disabled;
    let session = options.session.clone();
    let reasoning_policy = options.run_config.model.reasoning_item_id_policy;
    // Python (`prepare_input_with_session`): stored history comes before the new input.
    let mut writer = session
        .clone()
        .map(|s| SessionWriter::new(s, reasoning_policy));
    let start = match (start, &session) {
        (LoopStart::Fresh { input, .. }, Some(session)) => {
            let (prepared, to_save) = prepare_input_with_session(
                session.as_ref(),
                options.run_config.session.settings,
                options.run_config.session.input_callback.as_ref(),
                ItemHelpers::input_to_new_input_list(&input),
            )
            .await?;
            if let Some(writer) = writer.as_mut() {
                writer.pending_input = Some(to_save);
            }
            LoopStart::Fresh {
                input,
                prepared: Some(prepared),
            }
        }
        (LoopStart::Resume { state }, Some(_)) => {
            if let Some(writer) = writer.as_mut() {
                // A state from before turn-by-turn saving has saved nothing: save it all now.
                writer.saved_items = state.session_saved_items;
                writer.pending_input = (!state.session_input_saved)
                    .then(|| ItemHelpers::input_to_new_input_list(&state.input));
            }
            LoopStart::Resume { state }
        }
        (start, _) => start,
    };
    let tool_guardrail_log = SharedToolGuardrailLog::default();
    if let LoopStart::Resume { state } = &start {
        // Results from before the pause belong to the resumed run's result too.
        let mut log = tool_guardrail_log
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        log.input = state.tool_input_guardrail_results.clone();
        log.output = state.tool_output_guardrail_results.clone();
    }
    let mut result = crate::tracing::with_run_tracing_disabled(
        run_tracing_disabled,
        run_loop_inner(
            starting_agent,
            start,
            options,
            events,
            snapshot,
            Arc::clone(&tool_guardrail_log),
            writer.as_mut(),
        ),
    )
    .await?;
    {
        let mut log = tool_guardrail_log
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        result.tool_input_guardrail_results = std::mem::take(&mut log.input);
        result.tool_output_guardrail_results = std::mem::take(&mut log.output);
    }
    result.reasoning_item_id_policy = reasoning_policy;
    if let Some(writer) = writer.as_mut() {
        if result.interruptions.is_empty() {
            // The run's trace has ended by now, so work a session does here (a compaction
            // summary, say) must not emit spans of its own.
            crate::tracing::with_run_tracing_disabled(true, writer.flush(&result.new_items))
                .await?;
        } else if let Some(snapshot) = result.interrupt_state.as_mut() {
            // A paused run keeps the turn in flight out of the session until it is
            // resumed; remember how far it got so the resume saves only the rest.
            snapshot.session_saved_items = writer.saved_items;
            snapshot.session_input_saved = writer.pending_input.is_none();
        }
    }
    Ok(result)
}

/// Validate the options, open the run's trace and drive the turn loop.
async fn run_loop_inner(
    starting_agent: Agent,
    start: LoopStart,
    options: RunOptions,
    events: Option<EventTx>,
    snapshot: Option<Arc<Mutex<StreamingSnapshot>>>,
    tool_guardrail_log: SharedToolGuardrailLog,
    session_writer: Option<&mut SessionWriter>,
) -> Result<RunResult, AgentsError> {
    let max_tool_concurrency = validate_run_options(&options)?;
    let workflow = options
        .run_config
        .tracing
        .workflow_name
        .clone()
        .unwrap_or_else(|| "Agent workflow".into());

    // Python: `trace(workflow_name, trace_id=..., group_id=..., metadata=...)`. A disabled run
    // still builds the guard, but it is inactive so no events reach the processors.
    let _root = crate::tracing::trace_with_config(
        &workflow,
        crate::tracing::TraceConfig {
            trace_id: options.run_config.tracing.trace_id.clone(),
            group_id: options.run_config.tracing.group_id.clone(),
            metadata: options.run_config.tracing.metadata.clone(),
        },
    );

    TurnLoop::new(LoopInit {
        starting_agent,
        start,
        options,
        events,
        snapshot,
        tool_guardrail_log,
        session_writer,
        workflow,
        max_tool_concurrency,
    })?
    .run()
    .await
}
