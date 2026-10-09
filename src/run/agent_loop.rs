//! The agent loop (Python: `run.py::_run_impl` and `run_internal/run_loop.py`):
//! one iteration per model call, with tools, handoffs, guardrails and session saving.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::agent::{Agent, FunctionToolResult, ToolUseBehavior};
use crate::error::{
    AgentsError, InputGuardrailTripwireTriggered, MaxTurnsExceeded, ModelError, ModelRefusalError,
    OutputGuardrailTripwireTriggered, UserError,
};
use crate::guardrail::{InputGuardrail, InputGuardrailResult, OutputGuardrail};
use crate::handoffs::{nest_handoff_history, Handoff, HandoffInputData};
use crate::items::apply_reasoning_item_id_policy;
use crate::items::{
    extract_message_text, is_function_call, is_reasoning, required_function_call_parts,
    HandoffCallItem, HandoffOutputItem, InputLike, ItemHelpers, MessageOutputItem, ModelResponse,
    ReasoningItem, RunItem, ToolApprovalItem, ToolCallItem,
};
use crate::lifecycle::RunHooks;
use crate::memory::{prepare_input_with_session, Session};
use crate::model::{
    default_model_provider, Model, ModelInput, ModelProvider, ModelRef, ModelRequest, ModelTracing,
};
use crate::result::{InterruptSnapshot, RunResult, StreamingSnapshot};
use crate::run_context::RunContextWrapper;
use crate::run_state::{ApprovalStore, RunState};
use crate::stream_events::{RunItemStreamName, StreamEvent};
use crate::tool_guardrails::{ToolInputGuardrailResult, ToolOutputGuardrailResult};
use crate::tracing::{
    agent_span, generation_span, handoff_span, task_span, turn_span, SpanGuard, TracingConfig,
};
use crate::usage::Usage;

use super::config::{
    resolve_blocked_message, CallModelData, ModelInputData, OutputGuardrailBlockedMessage,
    RunConfig, RunOptions,
};
use super::errors::{
    accept_handler_output, build_run_error_data, invoke_run_error_handler, RunHandledError,
};
use super::tools::{
    execute_planned_tools, finalize_tool_output, has_tool_output, plan_tool_calls,
    push_tool_output, resolve_tool_name_collisions, tool_output_item, tools_for_agent,
    validate_tool_timeout, value_to_tool_string, ToolPlan, MULTIPLE_HANDOFFS_MESSAGE,
};

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
struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

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
    let run_tracing_disabled = options.run_config.tracing_disabled;
    let session = options.session.clone();
    let reasoning_policy = options.run_config.reasoning_item_id_policy;
    // Python (`prepare_input_with_session`): stored history comes before the new input.
    let mut writer = session
        .clone()
        .map(|s| SessionWriter::new(s, reasoning_policy));
    let start = match (start, &session) {
        (LoopStart::Fresh { input, .. }, Some(session)) => {
            let (prepared, to_save) = prepare_input_with_session(
                session.as_ref(),
                options.run_config.session_settings,
                options.run_config.session_input_callback.as_ref(),
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
        let mut log = tool_guardrail_log.lock().expect("tool guardrail log");
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
        let mut log = tool_guardrail_log.lock().expect("tool guardrail log");
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
            // A run paused for approval keeps the turn in flight out of the session until it is
            // resumed; remember how far it got so the resume saves only the rest.
            snapshot.session_saved_items = writer.saved_items;
            snapshot.session_input_saved = writer.pending_input.is_none();
        }
    }
    Ok(result)
}

/// Writes a run's input and generated items to its session as the run progresses.
///
/// Python saves the input before the first model call and each turn's items when the turn ends,
/// so a run that fails or is cut off keeps the turns it completed. `saved_items` counts the
/// leading `generated_items` already stored.
struct SessionWriter {
    session: Arc<dyn Session>,
    policy: Option<crate::items::ReasoningItemIdPolicy>,
    /// The run's input, until it has been saved.
    pending_input: Option<Vec<Value>>,
    saved_items: usize,
}

impl SessionWriter {
    fn new(session: Arc<dyn Session>, policy: Option<crate::items::ReasoningItemIdPolicy>) -> Self {
        Self {
            session,
            policy,
            pending_input: None,
            saved_items: 0,
        }
    }

    /// Save the turn whose final output an output guardrail withheld.
    ///
    /// Python (`blocked_output.py`) keeps the turn's tool calls but replaces every tool output
    /// with the placeholder, and leaves out the model's own messages. A turn that holds a
    /// reasoning item is not saved at all, because reasoning cannot be replayed without the
    /// items it led up to. Earlier turns were saved when they ended.
    async fn save_blocked_turn(
        &mut self,
        items: &[RunItem],
        placeholder: &str,
    ) -> Result<(), AgentsError> {
        let mut out = self.pending_input.take().unwrap_or_default();
        let turn = &items[self.saved_items.min(items.len())..];
        if !turn.iter().any(|i| matches!(i, RunItem::Reasoning(_))) {
            for item in turn {
                match item {
                    RunItem::ToolCall(_) => out.push(item.raw_item().clone()),
                    RunItem::ToolCallOutput(_) => {
                        let mut raw = item.raw_item().clone();
                        raw["output"] = Value::String(placeholder.to_string());
                        out.push(raw);
                    }
                    _ => {}
                }
            }
        }
        self.saved_items = items.len();
        if !out.is_empty() {
            self.session.add_items(out).await?;
        }
        Ok(())
    }

    /// Save the input (once) and every model-visible item not saved yet.
    async fn flush(&mut self, items: &[RunItem]) -> Result<(), AgentsError> {
        let mut out = self.pending_input.take().unwrap_or_default();
        let start = self.saved_items.min(items.len());
        out.extend(
            items[start..]
                .iter()
                .filter(|i| i.is_model_input())
                .map(|i| apply_reasoning_item_id_policy(i.raw_item(), self.policy)),
        );
        self.saved_items = items.len();
        if !out.is_empty() {
            self.session.add_items(out).await?;
        }
        Ok(())
    }
}

async fn run_loop_inner(
    starting_agent: Agent,
    start: LoopStart,
    options: RunOptions,
    events: Option<EventTx>,
    snapshot: Option<Arc<Mutex<StreamingSnapshot>>>,
    tool_guardrail_log: SharedToolGuardrailLog,
    mut session_writer: Option<&mut SessionWriter>,
) -> Result<RunResult, AgentsError> {
    // Python (`run_config.py:591-592`): `max_turns` is `int | None` and `None` disables the
    // limit; `run.py:1507` only compares when it is not `None`. `RunOptions::default()` keeps the
    // Python default of `Some(DEFAULT_MAX_TURNS)` (D-008), but an explicit `None` must mean
    // "run until the agent stops" (D-042), not "fall back to 10".
    let max_turns: Option<usize> = options.max_turns;
    // Python raises `ValueError` when the config is built; Rust has no constructor to hook, so
    // the run reports it as a `UserError` before doing any work.
    let max_tool_concurrency = options
        .run_config
        .tool_execution
        .and_then(|c| c.max_function_tool_concurrency);
    if matches!(
        &options.run_config.output_guardrail_blocked_message,
        Some(OutputGuardrailBlockedMessage::Text(text)) if text.is_empty()
    ) {
        return Err(UserError::new("output_guardrail_blocked_message must be non-empty").into());
    }
    if max_tool_concurrency == Some(0) {
        return Err(UserError::new(
            "tool_execution.max_function_tool_concurrency must be at least 1",
        )
        .into());
    }
    let workflow = options
        .run_config
        .workflow_name
        .clone()
        .unwrap_or_else(|| "Agent workflow".into());

    // Python: `trace(workflow_name, trace_id=..., group_id=..., metadata=...)`. A disabled run
    // still builds the guard, but it is inactive so no events reach the processors.
    let _root = crate::tracing::trace_with_config(
        &workflow,
        crate::tracing::TraceConfig {
            trace_id: options.run_config.trace_id.clone(),
            group_id: options.run_config.group_id.clone(),
            metadata: options.run_config.trace_metadata.clone(),
        },
    );

    // Python: the run is wrapped in a task span; each agent gets an agent span (replaced on
    // handoff) and each turn a turn span, unless `tracing.include_task_and_turn_spans` is off.
    let use_task_and_turn_spans =
        TracingConfig::includes_task_and_turn_spans(options.run_config.tracing.as_ref());
    let mut task_guard: Option<SpanGuard> = use_task_and_turn_spans.then(|| task_span(&workflow));
    // Held only for its drop: replacing it ends the previous agent's span.
    #[allow(unused_assignments)]
    let mut agent_guard: Option<SpanGuard> = None;
    let mut agent_span_due = true;

    let starting_agent_name = starting_agent.name.clone();
    // Every agent reachable so far by name, for handoffs bound late (`handoff_to_name`).
    let mut known_agents: HashMap<String, Agent> = HashMap::new();
    collect_agents(&starting_agent, &mut known_agents);
    let mut current_agent = starting_agent;
    // Agents that have emitted tool calls, used to reset `tool_choice` (Python:
    // `AgentToolUseTracker` + `maybe_reset_tool_choice`).
    let mut agents_used_tools: HashSet<String> = HashSet::new();

    // Python runs input guardrails once, on the first turn of a fresh run; a resumed run already
    // has their results.
    let resuming = matches!(start, LoopStart::Resume { .. });
    let mut generated_items: Vec<RunItem> = Vec::new();
    let mut raw_responses: Vec<ModelResponse> = Vec::new();
    let mut usage = Usage::default();
    let input: InputLike;
    let original_input_len: usize;
    let mut current_input_items: Vec<_>;
    let mut previous_response_id = options.previous_response_id.clone();
    let mut turn = 0usize;
    let mut resume_pending: Option<(ModelResponse, ApprovalStore)> = None;
    let mut nested_agent_runs: HashMap<String, RunState> = HashMap::new();
    let mut live_approvals = ApprovalStore::default();
    let context = RunContextWrapper::new(options.context.clone());
    let mut input_guardrail_results: Vec<InputGuardrailResult> = Vec::new();

    match start {
        LoopStart::Fresh {
            input: fresh,
            prepared,
        } => {
            input = fresh;
            // With a session, `run_loop` already merged history and the new input.
            current_input_items =
                prepared.unwrap_or_else(|| ItemHelpers::input_to_new_input_list(&input));
            original_input_len = current_input_items.len();
        }
        LoopStart::Resume { state } => {
            let state = *state;
            input = state.input.clone();
            original_input_len = ItemHelpers::input_to_new_input_list(&input).len();
            current_agent = resolve_agent_by_name(&current_agent, &state.current_agent_name)?;
            generated_items = state.generated_items;
            // Drop prior ToolApproval placeholders; resume will re-create if still pending.
            generated_items.retain(|i| !matches!(i, RunItem::ToolApproval(_)));
            raw_responses = state.raw_responses;
            input_guardrail_results = state.input_guardrail_results;
            usage = state.usage;
            current_input_items = state.current_input_items;
            previous_response_id = state.previous_response_id.or(previous_response_id);
            turn = state.turn.saturating_sub(1); // loop will increment
            nested_agent_runs = state.nested_agent_runs.clone();
            live_approvals = state.approvals.clone();
            context.set_usage(usage.clone());
            resume_pending = Some((state.pending_response, state.approvals));
        }
    }

    emit(
        &events,
        StreamEvent::AgentUpdated {
            agent_name: current_agent.name.clone(),
        },
    )
    .await;

    // Python splits input guardrails by `run_in_parallel`: blocking ones finish before the first
    // model call, the rest run concurrently with it.
    let (parallel_guardrails, blocking_guardrails): (Vec<InputGuardrail>, Vec<InputGuardrail>) =
        options
            .run_config
            .input_guardrails
            .iter()
            .chain(current_agent.input_guardrails.iter())
            .filter(|_| !resuming)
            .cloned()
            .partition(|g| g.run_in_parallel);
    let guardrail_agent = Arc::new(current_agent.clone());
    let guardrail_input = input.clone();
    let guardrail_context = context.clone();
    let run_input_guardrails = move |guardrails: Vec<InputGuardrail>| {
        let agent = Arc::clone(&guardrail_agent);
        let run_input = guardrail_input.clone();
        let ctx = guardrail_context.clone();
        async move {
            let futs = guardrails.into_iter().map(|g| {
                let agent = Arc::clone(&agent);
                let run_input = run_input.clone();
                let ctx = ctx.clone();
                async move { g.run(agent, run_input, ctx).await }
            });
            futures::future::join_all(futs).await
        }
    };
    // Both groups start inside the first turn (Python runs them under the turn span): blocking
    // ones finish first, then the parallel ones are spawned alongside the model call.
    let mut parallel_guardrails = Some(parallel_guardrails);
    let mut blocking_guardrails = Some(blocking_guardrails);
    let mut pending_input_guardrails: Option<AbortOnDrop<Vec<InputGuardrailResult>>> = None;

    // `on_agent_start` is called once per agent, including after each handoff.
    if let Some(h) = &options.hooks {
        h.on_agent_start(context.clone(), &current_agent).await;
    }
    if let Some(ah) = &current_agent.hooks {
        ah.on_start(context.clone(), &current_agent).await;
    }

    loop {
        turn += 1;
        // Save what the previous turn produced (and, before the first model call, the input).
        // The first turn of a resumed run is the one that was paused: it is saved when it ends.
        if resume_pending.is_none() {
            if let Some(writer) = session_writer.as_deref_mut() {
                writer.flush(&generated_items).await?;
            }
        }
        if let Some(snap) = &snapshot {
            let mut s = snap.lock().expect("snapshot");
            s.current_turn = turn;
            s.current_agent_name = current_agent.name.clone();
        }
        // Graceful cancel (`CancelMode::AfterTurn`): stop before a new turn begins.
        if turn > 1 {
            let cancelled = snapshot
                .as_ref()
                .map(|snap| snap.lock().expect("snapshot").cancel_after_turn)
                .unwrap_or(false);
            if cancelled {
                return Ok(finished_result(
                    input,
                    generated_items,
                    raw_responses,
                    Value::Null,
                    Arc::new(current_agent.clone()),
                    max_turns,
                    usage,
                ));
            }
        }
        // Python (`run.py:1506-1507`): `current_turn += 1` first, then `if max_turns is not
        // None and current_turn > max_turns`. With no limit the comparison is skipped entirely.
        let exceeded = match max_turns {
            Some(limit) if turn > limit => Some(limit),
            _ => None,
        };
        if let Some(limit) = exceeded {
            let error = MaxTurnsExceeded { max_turns: limit };
            // Python (`run.py:1508`): `SpanError(message="Max turns exceeded", data={"max_turns": n})`
            // on the current span, so a run that ran out of turns shows up on the timeline.
            if let Some(guard) = task_guard.as_mut() {
                guard.set_error(crate::tracing::SpanError {
                    message: "Max turns exceeded".to_string(),
                    data: Some(json!({ "max_turns": limit })),
                });
            }
            // Python (`finalize_max_turns_handler_output`): validate the handler's output,
            // record it as an assistant message, then run the end hooks and output guardrails.
            let run_data = build_run_error_data(
                &input,
                &generated_items,
                &raw_responses,
                &current_agent,
                options.run_config.reasoning_item_id_policy,
            );
            let Some(handled) = invoke_run_error_handler(
                options.error_handlers.max_turns.clone(),
                RunHandledError::MaxTurns(error.clone()),
                &context,
                run_data,
            )
            .await?
            else {
                return Err(error.into());
            };
            let final_output =
                accept_handler_output(&current_agent, handled, &mut generated_items)?;
            return finalize_run(
                input,
                generated_items,
                raw_responses,
                final_output,
                &current_agent,
                max_turns,
                usage,
                &context,
                options.hooks.as_ref(),
                &options.run_config.output_guardrails,
                input_guardrail_results,
                None,
                session_writer.as_deref_mut(),
            )
            .await;
        }

        // A handoff ended the previous agent's span; the new agent's starts with its first turn.
        if agent_span_due {
            drop(agent_guard.take());
            agent_guard = Some(agent_span(&current_agent.name));
            agent_span_due = false;
        }
        let mut turn_guard: Option<SpanGuard> =
            use_task_and_turn_spans.then(|| turn_span(turn, &current_agent.name));

        if let Some(blocking) = blocking_guardrails.take().filter(|g| !g.is_empty()) {
            let results = run_input_guardrails(blocking).await;
            if let Some(r) = results.iter().find(|r| r.output.tripwire_triggered) {
                return Err(InputGuardrailTripwireTriggered { result: r.clone() }.into());
            }
            input_guardrail_results.extend(results);
        }
        if let Some(parallel) = parallel_guardrails.take().filter(|g| !g.is_empty()) {
            // `tokio::spawn` does not inherit task-locals, so the run's tracing switch and the
            // current span are re-established inside the task; otherwise spans opened by a
            // guardrail escape `RunConfig.tracing_disabled` and lose their parent.
            let run_tracing_disabled = options.run_config.tracing_disabled;
            let fut = run_input_guardrails(parallel);
            pending_input_guardrails = Some(AbortOnDrop(tokio::spawn(
                crate::tracing::with_run_tracing_disabled(run_tracing_disabled, fut),
            )));
        }

        // Python recomputes model settings at the start of every turn
        // (`get_model_settings` then `maybe_reset_tool_choice`), so `tool_choice` is cleared
        // once the agent has used tools — including when the previous turn ended early via
        // `stop_on_first_tool` / `StopAtTools`.
        let mut model_settings = current_agent
            .model_settings
            .resolve(options.run_config.model_settings.as_ref());
        if current_agent.reset_tool_choice && agents_used_tools.contains(&current_agent.name) {
            model_settings.tool_choice = None;
        }

        let model = resolve_model(&current_agent, &options.run_config)?;
        // Python re-evaluates `is_enabled` every turn; disabled tools and handoffs are hidden
        // from the model and calls to them are treated as unknown.
        let (enabled_tools, enabled_handoffs) = resolve_tool_name_collisions(
            current_agent.all_function_tools(&context).await?,
            current_agent.enabled_handoffs(&context).await,
            options.run_config.tool_name_collision_policy,
        )?;
        for tool in &enabled_tools {
            validate_tool_timeout(tool)?;
        }
        let tools = tools_for_agent(enabled_tools, &enabled_handoffs);

        let (response, approvals_map, from_resume) = if let Some((pending, approvals)) =
            resume_pending.take()
        {
            (pending, approvals, true)
        } else {
            let _gen_span =
                generation_span(current_agent.model_name.as_deref().unwrap_or("scripted"));
            // Python: `instructions` may be a callable of `(context, agent)`.
            let resolved_instructions = current_agent.resolve_instructions(&context).await;
            // Python (`maybe_filter_model_input`): the filter runs before the LLM hooks and
            // only affects this call.
            let model_data = match &options.run_config.call_model_input_filter {
                Some(filter) => {
                    filter(CallModelData {
                        model_data: ModelInputData {
                            input: current_input_items.clone(),
                            instructions: resolved_instructions,
                        },
                        agent: Arc::new(current_agent.clone()),
                        context: context.clone(),
                    })
                    .await?
                }
                None => ModelInputData {
                    input: current_input_items.clone(),
                    instructions: resolved_instructions,
                },
            };
            let system_instructions = model_data.instructions;
            let model_input_items = model_data.input;
            if let Some(h) = &options.hooks {
                h.on_llm_start(
                    context.clone(),
                    &current_agent,
                    system_instructions.as_deref(),
                    &model_input_items,
                )
                .await;
            }
            if let Some(ah) = &current_agent.hooks {
                ah.on_llm_start(
                    context.clone(),
                    &current_agent,
                    system_instructions.as_deref(),
                    &model_input_items,
                )
                .await;
            }
            let (response, early_guardrails) = {
                // Everything an attempt reads is a plain reference, so each attempt rebuilds its
                // request and a retry can start it again from scratch.
                let model_ref: &dyn Model = &*model;
                let instructions_ref = system_instructions.as_deref();
                let input_ref = &model_input_items;
                let settings_ref = &model_settings;
                let tools_ref = &tools;
                let events_ref = &events;
                let emitted_unsafe = Arc::new(AtomicBool::new(false));
                // Python maps `trace_include_sensitive_data=False` to
                // `ModelTracing.ENABLED_WITHOUT_DATA`, so adapters drop payloads.
                let tracing_mode = if crate::tracing::tracing_disabled() {
                    ModelTracing::Disabled
                } else if options.run_config.trace_include_sensitive_data {
                    ModelTracing::Enabled
                } else {
                    ModelTracing::EnabledWithoutData
                };
                let previous_ref = previous_response_id.as_deref();
                let conversation_ref = options.conversation_id.as_deref();
                let output_schema_ref = current_agent.output_type.as_deref();
                let attempt_flag = Arc::clone(&emitted_unsafe);
                let model_fut = crate::retry::call_with_retry(
                    crate::retry::RetryCall {
                        settings: model_settings.retry.as_ref(),
                        previous_response_id: previous_ref,
                        conversation_id: conversation_ref,
                        timeout: model_settings.timeout,
                        stream: events.is_some(),
                        emitted_unsafe_event: events.is_some().then_some(&*emitted_unsafe),
                    },
                    |request| model_ref.get_retry_advice(request),
                    move || {
                        let attempt_flag = Arc::clone(&attempt_flag);
                        async move {
                            let req = ModelRequest {
                                system_instructions: instructions_ref,
                                input: ModelInput::Items(input_ref),
                                model_settings: settings_ref,
                                tools: tools_ref,
                                tracing: tracing_mode,
                                previous_response_id: previous_ref,
                                conversation_id: conversation_ref,
                                output_schema: output_schema_ref,
                            };
                            if events_ref.is_some() {
                                let (raw_tx, mut raw_rx) = mpsc::channel::<Value>(64);
                                let forward = {
                                    let events = events_ref.clone();
                                    AbortOnDrop(tokio::spawn(async move {
                                        while let Some(data) = raw_rx.recv().await {
                                            // Python: once an event other than `response.created` /
                                            // `response.in_progress` reached the consumer, a
                                            // replay would show it output twice.
                                            let kind = data.get("type").and_then(Value::as_str);
                                            if !matches!(
                                                kind,
                                                Some("response.created" | "response.in_progress")
                                            ) {
                                                attempt_flag.store(true, Ordering::SeqCst);
                                            }
                                            emit(&events, StreamEvent::RawResponse { data }).await;
                                        }
                                    }))
                                };
                                let outcome = model_ref.stream_response(req, raw_tx).await;
                                let _ = forward.await;
                                outcome
                            } else {
                                model_ref.get_response(req).await
                            }
                        }
                    },
                );
                let model_fut = async move { model_fut.await.map_err(AgentsError::from) };
                tokio::pin!(model_fut);
                // Python cancels the in-flight model call when a concurrent input guardrail
                // trips, so a tripwire never waits for (or pays for) the full response.
                let mut early_guardrails = None;
                let response = match pending_input_guardrails.as_mut() {
                    Some(task) => tokio::select! {
                        joined = &mut *task => {
                            let results = joined.map_err(|e| {
                                AgentsError::internal(format!("input guardrail task failed: {e}"))
                            })?;
                            if let Some(r) = results.iter().find(|r| r.output.tripwire_triggered) {
                                return Err(
                                    InputGuardrailTripwireTriggered { result: r.clone() }.into()
                                );
                            }
                            early_guardrails = Some(results);
                            (&mut model_fut).await?
                        }
                        response = &mut model_fut => response?,
                    },
                    None => (&mut model_fut).await?,
                };
                (response, early_guardrails)
            };
            if let Some(results) = early_guardrails {
                pending_input_guardrails = None;
                input_guardrail_results.extend(results);
            }
            if let Some(id) = &response.response_id {
                previous_response_id = Some(id.clone());
            }
            usage.add(&response.usage);
            for guard in [task_guard.as_mut(), turn_guard.as_mut()]
                .into_iter()
                .flatten()
            {
                guard.add_usage(&response.usage);
            }
            context.add_usage(&response.usage);
            raw_responses.push(response.clone());
            if let Some(snap) = &snapshot {
                let mut s = snap.lock().expect("snapshot");
                s.new_items = generated_items.clone();
                s.raw_responses = raw_responses.clone();
                s.usage = usage.clone();
            }
            if let Some(h) = &options.hooks {
                h.on_llm_end(context.clone(), &current_agent, &response)
                    .await;
            }
            if let Some(ah) = &current_agent.hooks {
                ah.on_llm_end(context.clone(), &current_agent, &response)
                    .await;
            }
            (response, live_approvals.clone(), false)
        };

        // Input guardrails ran alongside the first turn; a tripwire aborts the run.
        if let Some(task) = pending_input_guardrails.take() {
            let results = task
                .await
                .map_err(|e| AgentsError::internal(format!("input guardrail task failed: {e}")))?;
            for r in &results {
                if r.output.tripwire_triggered {
                    return Err(InputGuardrailTripwireTriggered { result: r.clone() }.into());
                }
            }
            input_guardrail_results.extend(results);
        }

        let mut function_calls: Vec<Value> = Vec::new();
        let mut reasoning_items: Vec<Value> = Vec::new();
        let mut messages: Vec<Value> = Vec::new();
        for item in response.output.iter().cloned() {
            if is_function_call(&item) {
                function_calls.push(item);
            } else if is_reasoning(&item) {
                reasoning_items.push(item);
            } else {
                messages.push(item);
            }
        }

        if !from_resume {
            for reason in &reasoning_items {
                let item = RunItem::Reasoning(ReasoningItem {
                    agent_name: current_agent.name.clone(),
                    raw_item: reason.clone(),
                });
                emit(
                    &events,
                    StreamEvent::RunItem {
                        name: RunItemStreamName::ReasoningItemCreated,
                        item: item.clone(),
                    },
                )
                .await;
                generated_items.push(item);
            }
            for msg in &messages {
                let item = RunItem::Message(MessageOutputItem {
                    agent_name: current_agent.name.clone(),
                    raw_item: msg.clone(),
                });
                emit(
                    &events,
                    StreamEvent::RunItem {
                        name: RunItemStreamName::MessageOutputCreated,
                        item: item.clone(),
                    },
                )
                .await;
                generated_items.push(item);
            }
            for call in &function_calls {
                let (tool_name, _, _) = required_function_call_parts(call)?;
                let is_handoff = enabled_handoffs.iter().any(|h| h.tool_name == tool_name);
                // Python separates handoff tool calls into `HandoffCallItem` during
                // `process_model_response`, so `new_items` distinguishes them from plain tools.
                let item = if is_handoff {
                    RunItem::HandoffCall(HandoffCallItem {
                        agent_name: current_agent.name.clone(),
                        raw_item: call.clone(),
                    })
                } else {
                    RunItem::ToolCall(ToolCallItem {
                        agent_name: current_agent.name.clone(),
                        raw_item: call.clone(),
                    })
                };
                emit(
                    &events,
                    StreamEvent::RunItem {
                        name: if is_handoff {
                            RunItemStreamName::HandoffRequested
                        } else {
                            RunItemStreamName::ToolCalled
                        },
                        item: item.clone(),
                    },
                )
                .await;
                generated_items.push(item);
            }
        }

        if function_calls.is_empty() {
            let last_message = messages
                .iter()
                .rev()
                .find(|m| m.get("type").and_then(Value::as_str) == Some("message"));
            let final_text = messages
                .iter()
                .filter_map(extract_message_text)
                .collect::<Vec<_>>()
                .join("");
            // Only this response is reported to `model_refusal` / `invalid_final_output`
            // handlers, like Python (`raw_responses=[new_response]`).
            let error_data = |generated: &[RunItem]| {
                build_run_error_data(
                    &input,
                    generated,
                    std::slice::from_ref(&response),
                    &current_agent,
                    options.run_config.reasoning_item_id_policy,
                )
            };
            // Python (`execute_tools_and_side_effects`): a refusal ends the turn with
            // `ModelRefusalError` unless a `model_refusal` handler supplies the output.
            let refusal = last_message.and_then(crate::items::extract_message_refusal);
            let final_output = if let Some(refusal) = refusal {
                let error = ModelRefusalError { refusal };
                let handled = invoke_run_error_handler(
                    options.error_handlers.model_refusal.clone(),
                    RunHandledError::ModelRefusal(error.clone()),
                    &context,
                    error_data(&generated_items),
                )
                .await?;
                let Some(handled) = handled else {
                    return Err(error.into());
                };
                accept_handler_output(&current_agent, handled, &mut generated_items)?
            } else {
                match current_agent.output_type.as_deref() {
                    Some(schema) if !schema.is_plain_text() => {
                        let invalid = if final_text.is_empty() {
                            ModelError::Behavior(
                                "Model returned no final output for the structured output type."
                                    .into(),
                            )
                        } else {
                            match schema.validate_json(&final_text) {
                                Ok(value) => {
                                    return finalize_run(
                                        input,
                                        generated_items,
                                        raw_responses,
                                        value,
                                        &current_agent,
                                        max_turns,
                                        usage,
                                        &context,
                                        options.hooks.as_ref(),
                                        &options.run_config.output_guardrails,
                                        input_guardrail_results,
                                        None,
                                        session_writer.as_deref_mut(),
                                    )
                                    .await;
                                }
                                Err(error) => error,
                            }
                        };
                        let handled = invoke_run_error_handler(
                            options.error_handlers.invalid_final_output.clone(),
                            RunHandledError::InvalidFinalOutput(invalid.clone()),
                            &context,
                            error_data(&generated_items),
                        )
                        .await?;
                        match handled {
                            Some(handled) => accept_handler_output(
                                &current_agent,
                                handled,
                                &mut generated_items,
                            )?,
                            // Python: an empty structured answer that no handler fixes asks the
                            // model again; an unparsable one raises.
                            None if final_text.is_empty() => {
                                for item in &response.output {
                                    current_input_items.push(apply_reasoning_item_id_policy(
                                        item,
                                        options.run_config.reasoning_item_id_policy,
                                    ));
                                }
                                continue;
                            }
                            None => return Err(invalid.into()),
                        }
                    }
                    _ => Value::String(final_text),
                }
            };
            return finalize_run(
                input,
                generated_items,
                raw_responses,
                final_output,
                &current_agent,
                max_turns,
                usage,
                &context,
                options.hooks.as_ref(),
                &options.run_config.output_guardrails,
                input_guardrail_results,
                None,
                session_writer.as_deref_mut(),
            )
            .await;
        }

        // Python records tool usage for the turn so `tool_choice` can be reset on the next
        // turn (`AgentToolUseTracker.record_processed_response`).
        agents_used_tools.insert(current_agent.name.clone());

        // Python (`execute_tools_and_side_effects`) runs the function tools of the turn first and
        // then performs the first handoff; extra handoffs are answered but ignored.
        let mut handoff_calls: Vec<(Handoff, Value, String)> = Vec::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        for call in &function_calls {
            let (name, _args, call_id) = required_function_call_parts(call)?;
            match enabled_handoffs.iter().find(|h| h.tool_name == name) {
                Some(h) => handoff_calls.push((h.clone(), call.clone(), call_id)),
                None => tool_calls.push(call.clone()),
            }
        }

        let tool_plan = plan_tool_calls(
            &current_agent,
            &tools,
            &tool_calls,
            &approvals_map,
            &generated_items,
            &options.run_config,
            &context,
        )
        .await?;

        if !tool_plan.interruptions.is_empty() {
            // Execute tools that do not need approval in this turn before pausing.
            if !tool_plan.to_invoke.is_empty() {
                let auto_plan = ToolPlan {
                    interruptions: Vec::new(),
                    ready_outputs: Vec::new(),
                    to_invoke: tool_plan.to_invoke.clone(),
                    already_done: Vec::new(),
                };
                let (auto_results, nested_interruptions, nested_states) = execute_planned_tools(
                    &auto_plan,
                    &nested_agent_runs,
                    &current_agent,
                    &context,
                    options.hooks.as_ref(),
                    max_tool_concurrency,
                    &tool_guardrail_log,
                )
                .await?;
                for (_tool, output, call_id) in &auto_results {
                    let out_item = tool_output_item(&current_agent.name, call_id, output);
                    emit(
                        &events,
                        StreamEvent::RunItem {
                            name: RunItemStreamName::ToolOutput,
                            item: out_item.clone(),
                        },
                    )
                    .await;
                    if !has_tool_output(&generated_items, call_id) {
                        generated_items.push(out_item);
                    }
                }
                if !nested_interruptions.is_empty() {
                    for (call_id, state) in nested_states {
                        nested_agent_runs.insert(call_id, state);
                    }
                    for approval in &nested_interruptions {
                        generated_items.push(RunItem::ToolApproval(approval.clone()));
                    }
                    return Ok(interrupted_result(
                        input,
                        generated_items,
                        raw_responses,
                        Arc::new(current_agent.clone()),
                        max_turns,
                        usage,
                        nested_interruptions,
                        InterruptSnapshot {
                            starting_agent_name,
                            current_input_items,
                            turn,
                            previous_response_id,
                            model_settings,
                            pending_response: response,
                            approvals: live_approvals,
                            nested_agent_runs,
                            session_saved_items: 0,
                            session_input_saved: false,
                            input_guardrail_results: input_guardrail_results.clone(),
                        },
                    ));
                }
            }
            for approval in &tool_plan.interruptions {
                generated_items.push(RunItem::ToolApproval(approval.clone()));
            }
            return Ok(interrupted_result(
                input,
                generated_items,
                raw_responses,
                Arc::new(current_agent.clone()),
                max_turns,
                usage,
                tool_plan.interruptions,
                InterruptSnapshot {
                    starting_agent_name,
                    current_input_items,
                    turn,
                    previous_response_id,
                    model_settings,
                    pending_response: response,
                    approvals: live_approvals,
                    nested_agent_runs,
                    session_saved_items: 0,
                    session_input_saved: false,
                    input_guardrail_results: input_guardrail_results.clone(),
                },
            ));
        }

        // All calls decided — execute remaining tools (rejections already resolved to outputs).
        let (tool_results, nested_interruptions, nested_states) = execute_planned_tools(
            &tool_plan,
            &nested_agent_runs,
            &current_agent,
            &context,
            options.hooks.as_ref(),
            max_tool_concurrency,
            &tool_guardrail_log,
        )
        .await?;
        if !nested_interruptions.is_empty() {
            for (call_id, state) in nested_states {
                nested_agent_runs.insert(call_id, state);
            }
            for approval in &nested_interruptions {
                generated_items.push(RunItem::ToolApproval(approval.clone()));
            }
            return Ok(interrupted_result(
                input,
                generated_items,
                raw_responses,
                Arc::new(current_agent.clone()),
                max_turns,
                usage,
                nested_interruptions,
                InterruptSnapshot {
                    starting_agent_name,
                    current_input_items,
                    turn,
                    previous_response_id,
                    model_settings,
                    pending_response: response,
                    approvals: live_approvals,
                    nested_agent_runs,
                    session_saved_items: 0,
                    session_input_saved: false,
                    input_guardrail_results: input_guardrail_results.clone(),
                },
            ));
        }
        // Clear nested states for completed outer tool calls.
        for (_tool, _output, call_id) in &tool_results {
            nested_agent_runs.remove(call_id);
        }

        // Python builds a `ToolCallOutputItem` for every completed tool result *before*
        // deciding whether one of them is the final output
        // (`_build_tool_result_items`, then `check_for_final_output_from_tools`), so a
        // `stop_on_first_tool` run still records the sibling outputs it produced.
        for (_tool, output, call_id) in &tool_results {
            push_tool_output(
                &events,
                &mut generated_items,
                &current_agent.name,
                call_id,
                output,
            )
            .await;
        }

        if !handoff_calls.is_empty() {
            let turn_start_len = current_input_items.len();
            for item in &response.output {
                current_input_items.push(apply_reasoning_item_id_policy(
                    item,
                    options.run_config.reasoning_item_id_policy,
                ));
            }
            for (_tool, output, call_id) in &tool_results {
                current_input_items.push(ItemHelpers::function_call_output(
                    call_id,
                    value_to_tool_string(output),
                ));
            }
            // Python: every handoff after the first gets a plain tool output.
            for (_h, _call, call_id) in handoff_calls.iter().skip(1) {
                let message = Value::String(MULTIPLE_HANDOFFS_MESSAGE.to_string());
                push_tool_output(
                    &events,
                    &mut generated_items,
                    &current_agent.name,
                    call_id,
                    &message,
                )
                .await;
                current_input_items.push(ItemHelpers::function_call_output(
                    call_id,
                    MULTIPLE_HANDOFFS_MESSAGE.to_string(),
                ));
            }

            let (h, call, call_id) = handoff_calls[0].clone();
            let _handoff_span = handoff_span(&current_agent.name, &h.agent.name);

            // Python (`on_invoke_handoff`): validate the arguments, then run `on_handoff`
            // before the handoff output is recorded.
            if let Some(on_handoff) = &h.on_handoff {
                let input = if h.input_json_schema.is_some() {
                    let (_, args, _) = required_function_call_parts(&call)?;
                    Some(serde_json::from_str::<Value>(&args).map_err(|e| {
                        ModelError::Behavior(format!(
                            "Invalid JSON input for handoff `{}`: {e}",
                            h.tool_name
                        ))
                    })?)
                } else {
                    None
                };
                on_handoff(context.clone(), input).await?;
            }

            let transfer = crate::pyjson::dumps(&json!({"assistant": h.agent.name}));
            let source_agent_name = current_agent.name.clone();
            let target_agent_name = h.agent.name.clone();
            let out_item = RunItem::HandoffOutput(HandoffOutputItem {
                agent_name: source_agent_name.clone(),
                raw_item: ItemHelpers::function_call_output(&call_id, transfer.clone()),
                source_agent_name,
                target_agent_name,
            });
            emit(
                &events,
                StreamEvent::RunItem {
                    name: RunItemStreamName::HandoffOccured,
                    item: out_item.clone(),
                },
            )
            .await;
            generated_items.push(out_item);
            current_input_items.push(ItemHelpers::function_call_output(&call_id, transfer));

            let source_agent = current_agent.clone();
            current_agent = if h.late_bound {
                known_agents.get(&h.agent.name).cloned().ok_or_else(|| {
                    UserError::new(format!(
                        "Handoff `{}` targets agent `{}`, which is not reachable from the \
                         starting agent. Add it with `handoff(agent)` somewhere in the graph.",
                        h.tool_name, h.agent.name
                    ))
                })?
            } else {
                (*h.agent).clone()
            };
            collect_agents(&current_agent, &mut known_agents);

            // Python: `hooks.on_handoff(context, from_agent, to_agent)` and the agent-level
            // `on_handoff(context, agent=new_agent, source=old_agent)`.
            if let Some(hooks) = &options.hooks {
                hooks
                    .on_handoff(context.clone(), &source_agent, &current_agent)
                    .await;
            }
            if let Some(ah) = &source_agent.hooks {
                ah.on_handoff(context.clone(), &current_agent, &source_agent)
                    .await;
            }

            // Python: the handoff's own filter wins over `RunConfig.handoff_input_filter`.
            let input_filter = h
                .input_filter
                .clone()
                .or_else(|| options.run_config.handoff_input_filter.clone());
            let server_managed =
                options.previous_response_id.is_some() || options.conversation_id.is_some();
            let mut should_nest = h
                .nest_handoff_history
                .unwrap_or(options.run_config.nest_handoff_history);
            if input_filter.is_some() && server_managed {
                return Err(UserError::new(
                    "Server-managed conversations do not support handoff input filters. \
                     Remove Handoff.input_filter or RunConfig.handoff_input_filter, \
                     or disable conversation_id and previous_response_id.",
                )
                .into());
            }
            if should_nest && server_managed {
                ::tracing::warn!(
                    "Server-managed conversations do not support nest_handoff_history for handoff \
                     {} -> {}. Disabling nested handoff history.",
                    source_agent.name,
                    current_agent.name
                );
                should_nest = false;
            }
            if input_filter.is_some() || should_nest {
                let original_len = original_input_len.min(turn_start_len);
                let data = HandoffInputData {
                    input_history: current_input_items[..original_len].to_vec(),
                    pre_handoff_items: current_input_items[original_len..turn_start_len].to_vec(),
                    new_items: current_input_items[turn_start_len..].to_vec(),
                    run_context: context.clone(),
                };
                // Python: an explicit filter replaces automatic nesting.
                let next = match input_filter {
                    Some(filter) => filter(data).await?,
                    None => nest_handoff_history(
                        data,
                        options.run_config.handoff_history_mapper.as_ref(),
                    ),
                };
                current_input_items = next
                    .input_history
                    .into_iter()
                    .chain(next.pre_handoff_items)
                    .chain(next.new_items)
                    .collect();
            }

            emit(
                &events,
                StreamEvent::AgentUpdated {
                    agent_name: current_agent.name.clone(),
                },
            )
            .await;
            if let Some(hooks) = &options.hooks {
                hooks.on_agent_start(context.clone(), &current_agent).await;
            }
            if let Some(ah) = &current_agent.hooks {
                ah.on_start(context.clone(), &current_agent).await;
            }
            agent_span_due = true;
            continue;
        }

        match &current_agent.tool_use_behavior {
            ToolUseBehavior::StopOnFirstTool => {
                if let Some((_tool, output, _call_id)) = tool_results.first() {
                    return finalize_run(
                        input,
                        generated_items,
                        raw_responses,
                        finalize_tool_output(&current_agent, output),
                        &current_agent,
                        max_turns,
                        usage,
                        &context,
                        options.hooks.as_ref(),
                        &options.run_config.output_guardrails,
                        input_guardrail_results,
                        Some(&options.run_config),
                        session_writer.as_deref_mut(),
                    )
                    .await;
                }
            }
            ToolUseBehavior::StopAtTools { stop_at_tool_names } => {
                for (tool, output, _call_id) in &tool_results {
                    if stop_at_tool_names.iter().any(|n| n == &tool.name) {
                        return finalize_run(
                            input,
                            generated_items,
                            raw_responses,
                            finalize_tool_output(&current_agent, output),
                            &current_agent,
                            max_turns,
                            usage,
                            &context,
                            options.hooks.as_ref(),
                            &options.run_config.output_guardrails,
                            input_guardrail_results,
                            Some(&options.run_config),
                            session_writer.as_deref_mut(),
                        )
                        .await;
                    }
                }
            }
            ToolUseBehavior::Custom(decide) => {
                let results: Vec<FunctionToolResult> = tool_results
                    .iter()
                    .map(|(tool, output, call_id)| FunctionToolResult {
                        tool_name: tool.name.clone(),
                        call_id: call_id.clone(),
                        output: output.clone(),
                    })
                    .collect();
                let decision = decide(&context, &results);
                if decision.is_final_output {
                    let output = decision.final_output.unwrap_or(Value::Null);
                    return finalize_run(
                        input,
                        generated_items,
                        raw_responses,
                        finalize_tool_output(&current_agent, &output),
                        &current_agent,
                        max_turns,
                        usage,
                        &context,
                        options.hooks.as_ref(),
                        &options.run_config.output_guardrails,
                        input_guardrail_results,
                        Some(&options.run_config),
                        session_writer.as_deref_mut(),
                    )
                    .await;
                }
            }
            ToolUseBehavior::RunLlmAgain => {}
        }

        for item in &response.output {
            current_input_items.push(apply_reasoning_item_id_policy(
                item,
                options.run_config.reasoning_item_id_policy,
            ));
        }
        for (_tool, output, call_id) in &tool_results {
            if !has_tool_output(&generated_items, call_id) {
                let out_item = tool_output_item(&current_agent.name, call_id, output);
                emit(
                    &events,
                    StreamEvent::RunItem {
                        name: RunItemStreamName::ToolOutput,
                        item: out_item.clone(),
                    },
                )
                .await;
                generated_items.push(out_item);
            }
            current_input_items.push(ItemHelpers::function_call_output(
                call_id,
                value_to_tool_string(output),
            ));
        }
    }
}

/// Run `on_agent_end` hooks plus output guardrails, then build the final [`RunResult`].
///
/// Python runs output guardrails concurrently after the final output is known and raises
/// `OutputGuardrailTripwireTriggered` if any of them tripped.
#[allow(clippy::too_many_arguments)]
async fn finalize_run(
    input: InputLike,
    new_items: Vec<RunItem>,
    raw_responses: Vec<ModelResponse>,
    final_output: Value,
    agent: &Agent,
    max_turns: Option<usize>,
    usage: Usage,
    context: &RunContextWrapper,
    hooks: Option<&Arc<dyn RunHooks>>,
    extra_output_guardrails: &[OutputGuardrail],
    input_guardrail_results: Vec<InputGuardrailResult>,
    tool_origin: Option<&RunConfig>,
    session_writer: Option<&mut SessionWriter>,
) -> Result<RunResult, AgentsError> {
    // Python closes the turn span before the run's end hooks and output guardrails.
    let _outside_turn = crate::tracing::leave_turn_span();
    if let Some(h) = hooks {
        h.on_agent_end(context.clone(), agent, &final_output).await;
    }
    if let Some(ah) = &agent.hooks {
        ah.on_end(context.clone(), agent, &final_output).await;
    }

    let guardrails: Vec<OutputGuardrail> = extra_output_guardrails
        .iter()
        .chain(agent.output_guardrails.iter())
        .cloned()
        .collect();
    let mut output_guardrail_results = Vec::new();
    if !guardrails.is_empty() {
        let agent = Arc::new(agent.clone());
        let futs = guardrails.iter().map(|g| {
            let g = g.clone();
            let agent = Arc::clone(&agent);
            let output = final_output.clone();
            let ctx = context.clone();
            async move { g.run(agent, output, ctx).await }
        });
        let results = futures::future::join_all(futs).await;
        let tripped = results
            .iter()
            .find(|r| r.output.tripwire_triggered)
            .cloned();
        output_guardrail_results = results;
        if let Some(mut result) = tripped {
            // Python (`blocked_output.py`): a final output that came from a tool is withheld, so
            // the error carries a data-free placeholder instead of the tool's output, and so
            // does the session.
            if let Some(run_config) = tool_origin {
                let placeholder =
                    resolve_blocked_message(run_config, &result.guardrail_name, &agent, context);
                if let Some(writer) = session_writer {
                    writer.save_blocked_turn(&new_items, &placeholder).await?;
                }
                result.agent_output = Value::String(placeholder);
                result.output.output_info = Value::Null;
            }
            return Err(OutputGuardrailTripwireTriggered { result }.into());
        }
    }

    let mut result = finished_result(
        input,
        new_items,
        raw_responses,
        final_output,
        Arc::new(agent.clone()),
        max_turns,
        usage,
    );
    result.input_guardrail_results = input_guardrail_results;
    result.output_guardrail_results = output_guardrail_results;
    Ok(result)
}

fn finished_result(
    input: InputLike,
    new_items: Vec<RunItem>,
    raw_responses: Vec<ModelResponse>,
    final_output: Value,
    last_agent: Arc<Agent>,
    max_turns: Option<usize>,
    usage: Usage,
) -> RunResult {
    RunResult {
        input,
        new_items,
        raw_responses,
        final_output,
        last_agent_name: last_agent.name.clone(),
        last_agent,
        max_turns,
        usage,
        interruptions: Vec::new(),
        input_guardrail_results: Vec::new(),
        output_guardrail_results: Vec::new(),
        tool_input_guardrail_results: Vec::new(),
        tool_output_guardrail_results: Vec::new(),
        reasoning_item_id_policy: None,
        interrupt_state: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn interrupted_result(
    input: InputLike,
    new_items: Vec<RunItem>,
    raw_responses: Vec<ModelResponse>,
    last_agent: Arc<Agent>,
    max_turns: Option<usize>,
    usage: Usage,
    interruptions: Vec<ToolApprovalItem>,
    interrupt_state: InterruptSnapshot,
) -> RunResult {
    RunResult {
        input,
        new_items,
        raw_responses,
        final_output: Value::Null,
        last_agent_name: last_agent.name.clone(),
        last_agent,
        max_turns,
        usage,
        interruptions,
        input_guardrail_results: interrupt_state.input_guardrail_results.clone(),
        output_guardrail_results: Vec::new(),
        tool_input_guardrail_results: Vec::new(),
        tool_output_guardrail_results: Vec::new(),
        reasoning_item_id_policy: None,
        interrupt_state: Some(interrupt_state),
    }
}

/// Record `root` and every agent it owns through `handoff(..)`, by name. Late-bound handoffs
/// are stand-ins and are skipped; the first agent seen under a name is kept.
fn collect_agents(root: &Agent, into: &mut HashMap<String, Agent>) {
    if into.contains_key(&root.name) {
        return;
    }
    into.insert(root.name.clone(), root.clone());
    for h in root.handoffs.iter().filter(|h| !h.late_bound) {
        collect_agents(&h.agent, into);
    }
}

fn resolve_agent_by_name(root: &Agent, name: &str) -> Result<Agent, AgentsError> {
    if root.name == name {
        return Ok(root.clone());
    }
    let mut stack = vec![root.clone()];
    while let Some(agent) = stack.pop() {
        for h in agent.handoffs.iter().filter(|h| !h.late_bound) {
            if h.agent.name == name {
                return Ok((*h.agent).clone());
            }
            stack.push((*h.agent).clone());
        }
    }
    Err(UserError::new(format!("Agent `{name}` not found in starting agent graph")).into())
}

static DEFAULT_MODEL_PROVIDER: OnceLock<Arc<dyn ModelProvider>> = OnceLock::new();

/// Lazily-created default provider (Python: `RunConfig.model_provider` fallback).
pub fn default_provider() -> &'static Arc<dyn ModelProvider> {
    DEFAULT_MODEL_PROVIDER.get_or_init(default_model_provider)
}

/// Resolve the model for a turn.
///
/// Order matches Python: `RunConfig.model` (instance or name) → `Agent.model` →
/// `Agent.model_name`, with names resolved through the configured provider.
fn resolve_model(agent: &Agent, run_config: &RunConfig) -> Result<Arc<dyn Model>, AgentsError> {
    let provider: &Arc<dyn ModelProvider> = match &run_config.model_provider {
        Some(p) => p,
        None => default_provider(),
    };

    if let Some(ModelRef::Instance(m)) = &run_config.model {
        return Ok(Arc::clone(m));
    }
    if let Some(ModelRef::Name(name)) = &run_config.model {
        return Ok(provider.get_model(Some(name))?);
    }
    if let Some(m) = &agent.model {
        return Ok(Arc::clone(m));
    }
    if let Some(name) = &agent.model_name {
        return Ok(provider.get_model(Some(name))?);
    }
    Err(UserError::new(
        "No model configured. Bind a Model (e.g. ScriptedModel) via `Agent::model`, set \
         `Agent::model_name` / `RunConfig.model` to a name resolvable by a ModelProvider, or \
         enable the `openai` feature with OPENAI_API_KEY set.",
    )
    .into())
}
