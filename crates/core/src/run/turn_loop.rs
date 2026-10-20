//! The turn loop (Python: `run.py::_run_impl` and `run_internal/run_loop.py`): one iteration per
//! model call, with tools, handoffs, guardrails and session saving.
//!
//! [`TurnLoop`] owns everything a run mutates between turns. Each phase of a turn is a method with
//! one job, so the order of a turn reads top to bottom in [`TurnLoop::step`].

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::error::{AgentsError, InputGuardrailTripwireTriggered, ModelError, UserError};
use crate::guardrail::{run_group, single_transform, InputGuardrail, InputGuardrailResult};
use crate::handoffs::Handoff;
use crate::items::{
    apply_reasoning_item_id_policy, required_function_call_parts, InputLike, ItemHelpers,
    ModelResponse, ResponseInputItem, RunItem,
};
use crate::model::{CallModelData, Model, ModelInput, ModelInputData, ModelRequest, ModelTracing};
use crate::result::{InterruptSnapshot, RunResult, StreamingSnapshot};
use crate::run_context::RunContextWrapper;
use crate::run_state::{ResolutionStore, RunState};
use crate::stream_events::StreamEvent;
use crate::tracing::{agent_span, generation_span, task_span, turn_span, SpanGuard, TracingConfig};
use crate::usage::Usage;

use super::agent_loop::{emit, AbortOnDrop, EventTx, LoopStart, SharedToolGuardrailLog};
use super::config::RunOptions;
use super::finalize::{finalize_run, finished_result, interrupted_result, FinalizeEnv, RunRecord};
use super::resolve::{
    agent_graph_fingerprint, collect_agents, resolve_agent_by_name, resolve_model,
};
use super::session_writer::SessionWriter;
use super::tools::{plan_tool_calls, push_tool_output};
use super::turn::{
    execute_turn_tools, final_output_without_tools, handle_max_turns_exceeded, perform_handoff,
    prepare_turn, record_response_items, record_turn_output, split_response_output,
    tool_use_behavior_output, turn_was_cancelled, HandoffTurn, ResponseParts, ToolTurn, TurnEnd,
    TurnOutput, TurnSetup,
};

/// A function-tool result: the tool, its output and the model's call id.
type ToolResult = (crate::tool::FunctionTool, Value, String);

/// Handoff calls (with their resolved handoff) and plain tool calls found in one response.
type SplitCalls = (Vec<(Handoff, Value, String)>, Vec<Value>);

/// Everything [`TurnLoop::new`] needs.
pub(super) struct LoopInit<'a> {
    pub(super) starting_agent: Arc<Agent>,
    pub(super) start: LoopStart,
    pub(super) options: RunOptions,
    pub(super) events: Option<EventTx>,
    pub(super) snapshot: Option<Arc<Mutex<StreamingSnapshot>>>,
    pub(super) tool_guardrail_log: SharedToolGuardrailLog,
    pub(super) session_writer: Option<&'a mut SessionWriter>,
    pub(super) workflow: String,
    pub(super) max_tool_concurrency: Option<usize>,
}

/// The input guardrails of a run.
///
/// Python splits them by `run_in_parallel`: blocking ones finish before the first model call, the
/// rest run concurrently with it.
struct InputGuardrails {
    blocking: Option<Vec<InputGuardrail>>,
    parallel: Option<Vec<InputGuardrail>>,
    pending: Option<AbortOnDrop<Vec<InputGuardrailResult>>>,
    agent: Arc<Agent>,
    input: InputLike,
}

impl InputGuardrails {
    /// A resumed run already has the results of its guardrails, so it gets none.
    fn new(options: &RunOptions, agent: &Arc<Agent>, input: &InputLike, resuming: bool) -> Self {
        let (parallel, blocking): (Vec<InputGuardrail>, Vec<InputGuardrail>) = options
            .run_config
            .input_guardrails
            .iter()
            .chain(agent.all_input_guardrails())
            .filter(|_| !resuming)
            .cloned()
            .partition(|g| g.run_in_parallel);
        Self {
            blocking: Some(blocking),
            parallel: Some(parallel),
            pending: None,
            agent: Arc::clone(agent),
            input: input.clone(),
        }
    }

    /// A future that runs `guardrails` side by side. It owns its inputs so it can be spawned.
    fn runner(
        &self,
        guardrails: Vec<InputGuardrail>,
        context: &RunContextWrapper,
    ) -> impl Future<Output = Vec<InputGuardrailResult>> + Send + 'static {
        let agent = Arc::clone(&self.agent);
        let run_input = self.input.clone();
        let ctx = context.clone();
        async move {
            let futs = guardrails.into_iter().map(|g| {
                let agent = Arc::clone(&agent);
                let run_input = run_input.clone();
                let ctx = ctx.clone();
                async move { g.run(agent, run_input, ctx).await }
            });
            run_group(futs, |r: &InputGuardrailResult| &r.output).await
        }
    }
}

/// The first tripped guardrail of `results`, as the error that aborts the run.
fn tripwire(results: &[InputGuardrailResult]) -> Result<(), AgentsError> {
    match results.iter().find(|r| r.output.tripwire_triggered()) {
        Some(r) => Err(InputGuardrailTripwireTriggered { result: r.clone() }.into()),
        None => Ok(()),
    }
}

/// A guardrail that runs alongside the model cannot change what the model already received.
fn reject_parallel_transform(results: &[InputGuardrailResult]) -> Result<(), AgentsError> {
    match results.iter().find(|r| r.output.transformed().is_some()) {
        Some(r) => Err(UserError::new(format!(
            "input guardrail `{}` returned Verdict::Transform but runs in parallel with the model, \
             which has already received the original input; build it with \
             `.run_in_parallel(false)`",
            r.guardrail_name
        ))
        .into()),
        None => Ok(()),
    }
}

/// Where a run stands when its loop starts: fresh, or restored from a paused [`RunState`].
struct Progress {
    current_agent: Arc<Agent>,
    input: InputLike,
    original_input_len: usize,
    current_input_items: Vec<Value>,
    generated_items: Vec<RunItem>,
    raw_responses: Vec<ModelResponse>,
    usage: Usage,
    previous_response_id: Option<String>,
    turn: usize,
    resume_pending: Option<(ModelResponse, ResolutionStore)>,
    nested_agent_runs: HashMap<String, RunState>,
    live_resolutions: ResolutionStore,
    input_guardrail_results: Vec<InputGuardrailResult>,
}

impl Progress {
    fn restore(
        start: LoopStart,
        starting_agent: Arc<Agent>,
        previous_response_id: Option<String>,
        context: &RunContextWrapper,
    ) -> Result<Self, AgentsError> {
        match start {
            LoopStart::Fresh { input, prepared } => {
                // With a session, `run_loop` already merged history and the new input.
                let current_input_items =
                    prepared.unwrap_or_else(|| ItemHelpers::input_to_new_input_list(&input));
                Ok(Self {
                    current_agent: starting_agent,
                    original_input_len: current_input_items.len(),
                    current_input_items,
                    input,
                    generated_items: Vec::new(),
                    raw_responses: Vec::new(),
                    usage: Usage::default(),
                    previous_response_id,
                    turn: 0,
                    resume_pending: None,
                    nested_agent_runs: HashMap::new(),
                    live_resolutions: ResolutionStore::default(),
                    input_guardrail_results: Vec::new(),
                })
            }
            LoopStart::Resume { state } => {
                let state = *state;
                let current_agent =
                    resolve_agent_by_name(&starting_agent, &state.current_agent_name)?;
                let mut generated_items = state.generated_items;
                // Drop prior ToolInterruption placeholders; resume will re-create if still pending.
                generated_items.retain(|i| !matches!(i, RunItem::ToolInterruption(_)));
                context.set_usage(state.usage.clone());
                context.set_loaded_capabilities(&state.loaded_capabilities);
                Ok(Self {
                    current_agent,
                    original_input_len: ItemHelpers::input_to_new_input_list(&state.input).len(),
                    input: state.input,
                    current_input_items: state.current_input_items,
                    generated_items,
                    raw_responses: state.raw_responses,
                    usage: state.usage,
                    previous_response_id: state.previous_response_id.or(previous_response_id),
                    // The loop increments the counter before the first turn.
                    turn: state.turn.saturating_sub(1),
                    resume_pending: Some((state.pending_response, state.resolutions.clone())),
                    nested_agent_runs: state.nested_agent_runs,
                    live_resolutions: state.resolutions,
                    input_guardrail_results: state.input_guardrail_results,
                })
            }
        }
    }
}

/// The state of a running agent loop.
pub(super) struct TurnLoop<'a> {
    // Span guards come first: fields drop in declaration order, and the agent's span has to end
    // before the task span that contains it.
    agent_guard: Option<SpanGuard>,
    task_guard: Option<SpanGuard>,
    use_task_and_turn_spans: bool,
    agent_span_due: bool,

    options: RunOptions,
    events: Option<EventTx>,
    snapshot: Option<Arc<Mutex<StreamingSnapshot>>>,
    tool_guardrail_log: SharedToolGuardrailLog,
    session_writer: Option<&'a mut SessionWriter>,
    context: RunContextWrapper,
    max_turns: Option<usize>,
    max_tool_concurrency: Option<usize>,
    guardrails: InputGuardrails,

    starting_agent: Arc<Agent>,
    current_agent: Arc<Agent>,
    /// Every agent reachable so far by name, for handoffs bound late (`handoff_to_name`).
    known_agents: HashMap<String, Arc<Agent>>,
    /// Agents that have emitted tool calls, used to reset `tool_choice` (Python:
    /// `AgentToolUseTracker` + `maybe_reset_tool_choice`).
    agents_used_tools: HashSet<String>,

    input: InputLike,
    original_input_len: usize,
    current_input_items: Vec<Value>,
    generated_items: Vec<RunItem>,
    raw_responses: Vec<ModelResponse>,
    usage: Usage,
    previous_response_id: Option<String>,
    turn: usize,
    resume_pending: Option<(ModelResponse, ResolutionStore)>,
    nested_agent_runs: HashMap<String, RunState>,
    live_resolutions: ResolutionStore,
    input_guardrail_results: Vec<InputGuardrailResult>,
}

impl<'a> TurnLoop<'a> {
    pub(super) fn new(init: LoopInit<'a>) -> Result<Self, AgentsError> {
        let LoopInit {
            starting_agent,
            start,
            options,
            events,
            snapshot,
            tool_guardrail_log,
            session_writer,
            workflow,
            max_tool_concurrency,
        } = init;
        // Python: the run is wrapped in a task span; each agent gets an agent span (replaced on
        // handoff) and each turn a turn span, unless `tracing.include_task_and_turn_spans` is off.
        let use_task_and_turn_spans =
            TracingConfig::includes_task_and_turn_spans(options.run_config.tracing.config.as_ref());
        let task_guard = use_task_and_turn_spans.then(|| task_span(&workflow));

        let starting = Arc::clone(&starting_agent);
        let mut known_agents = HashMap::new();
        collect_agents(&starting_agent, &mut known_agents);
        let resuming = matches!(start, LoopStart::Resume { .. });
        let context = options.run_context();
        let Progress {
            current_agent,
            input,
            original_input_len,
            current_input_items,
            generated_items,
            raw_responses,
            usage,
            previous_response_id,
            turn,
            resume_pending,
            nested_agent_runs,
            live_resolutions,
            input_guardrail_results,
        } = Progress::restore(
            start,
            starting_agent,
            options.previous_response_id.clone(),
            &context,
        )?;
        let guardrails = InputGuardrails::new(&options, &current_agent, &input, resuming);
        Ok(Self {
            agent_guard: None,
            task_guard,
            use_task_and_turn_spans,
            agent_span_due: true,
            max_turns: options.max_turns,
            options,
            events,
            snapshot,
            tool_guardrail_log,
            session_writer,
            context,
            max_tool_concurrency,
            guardrails,
            starting_agent: starting,
            current_agent,
            known_agents,
            agents_used_tools: HashSet::new(),
            input,
            original_input_len,
            current_input_items,
            generated_items,
            raw_responses,
            usage,
            previous_response_id,
            turn,
            resume_pending,
            nested_agent_runs,
            live_resolutions,
            input_guardrail_results,
        })
    }

    /// Run turns until the run finishes, pauses or fails.
    pub(super) async fn run(mut self) -> Result<RunResult, AgentsError> {
        self.announce_agent().await;
        loop {
            if let Some(result) = self.step().await? {
                return Ok(result);
            }
        }
    }

    /// `on_agent_start` is called once per agent, including after each handoff.
    async fn announce_agent(&self) {
        emit(
            &self.events,
            StreamEvent::AgentUpdated {
                agent_name: self.current_agent.name.clone(),
            },
        )
        .await;
        if let Some(h) = &self.options.hooks {
            h.on_agent_start(self.context.clone(), &self.current_agent)
                .await;
        }
        if let Some(ah) = &self.current_agent.hooks {
            ah.on_start(self.context.clone(), &self.current_agent).await;
        }
    }

    /// One turn. `Some` ends the run with that result; `None` goes on to the next turn.
    async fn step(&mut self) -> Result<Option<RunResult>, AgentsError> {
        if let Some(done) = self.begin_turn().await? {
            return Ok(Some(done));
        }
        let mut turn_guard = self.open_spans();
        self.start_input_guardrails().await?;

        let mut setup = prepare_turn(
            &self.current_agent,
            &self.agents_used_tools,
            &self.options,
            &self.context,
        )
        .await?;
        let resolved = resolve_model(&self.current_agent, &self.options.run_config).await?;
        resolved.apply_to(&mut setup.model_settings);
        let model = resolved.model;
        let (response, resolutions, from_resume) = match self.resume_pending.take() {
            Some((pending, resolutions)) => (pending, resolutions, true),
            None => {
                let response = self.call_model(&setup, &*model, &mut turn_guard).await?;
                (response, self.live_resolutions.clone(), false)
            }
        };
        self.settle_input_guardrails().await?;

        let parts = split_response_output(&response);
        if !from_resume {
            // Python separates handoff tool calls from plain ones while recording, so the names
            // the turn enabled have to be known before the items are built.
            let handoff_names: Vec<String> = setup
                .enabled_handoffs
                .iter()
                .map(|h| h.tool_name.clone())
                .collect();
            record_response_items(
                &parts,
                &handoff_names,
                &self.current_agent.name,
                &self.events,
                &mut self.generated_items,
            )
            .await?;
        }
        if parts.function_calls.is_empty() {
            return self.complete_without_tools(&parts, &response).await;
        }
        self.complete_with_tools(&parts, response, setup, &resolutions)
            .await
    }

    /// Before a turn: save what the previous one produced, publish progress, honour a graceful
    /// cancel and the turn limit. `Some` ends the run.
    async fn begin_turn(&mut self) -> Result<Option<RunResult>, AgentsError> {
        self.turn += 1;
        // Save what the previous turn produced (and, before the first model call, the input). The
        // first turn of a resumed run is the one that was paused: it is saved when it ends.
        if self.resume_pending.is_none() && !self.input_waits_for_guardrails() {
            if let Some(writer) = self.session_writer.as_deref_mut() {
                writer.flush(&self.generated_items).await?;
            }
        }
        if let Some(snap) = &self.snapshot {
            let mut s = snap
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            s.current_turn = self.turn;
            s.current_agent_name = self.current_agent.name.clone();
        }
        // Graceful cancel (`CancelMode::AfterTurn`): stop before a new turn begins.
        if turn_was_cancelled(self.turn, self.snapshot.as_ref()) {
            let record = self.take_record();
            return Ok(Some(finished_result(
                record,
                Value::Null,
                Arc::clone(&self.current_agent),
            )));
        }
        // Python (`run.py:1506-1507`): the turn counter is incremented first and only compared
        // when `max_turns` is not `None`.
        let done_items = self.generated_items.clone();
        let exceeded = handle_max_turns_exceeded(
            self.turn,
            &TurnEnd {
                input: &self.input,
                raw_responses: &self.raw_responses,
                agent: &self.current_agent,
                max_turns: self.max_turns,
                options: &self.options,
            },
            &done_items,
            &mut self.generated_items,
            &self.context,
            self.task_guard.as_mut(),
        )
        .await?;
        match exceeded {
            Some(output) => Ok(Some(self.finish(output, false).await?)),
            None => Ok(None),
        }
    }

    /// The turn's spans. A handoff ended the previous agent's span; the new agent's starts with
    /// its first turn.
    fn open_spans(&mut self) -> Option<SpanGuard> {
        if self.agent_span_due {
            drop(self.agent_guard.take());
            self.agent_guard = Some(agent_span(&self.current_agent.name));
            self.agent_span_due = false;
        }
        self.use_task_and_turn_spans
            .then(|| turn_span(self.turn, &self.current_agent.name))
    }

    /// Both groups start inside the first turn (Python runs them under the turn span): blocking
    /// ones finish first, then the parallel ones are spawned alongside the model call.
    async fn start_input_guardrails(&mut self) -> Result<(), AgentsError> {
        if let Some(blocking) = self.guardrails.blocking.take().filter(|g| !g.is_empty()) {
            let results = self.guardrails.runner(blocking, &self.context).await;
            tripwire(&results)?;
            let transform = single_transform(
                &results,
                |r: &InputGuardrailResult| &r.output,
                |r: &InputGuardrailResult| r.guardrail_name.as_str(),
            )?
            .map(|(name, content)| (name.to_owned(), content.clone()));
            if let Some((name, content)) = transform {
                self.apply_input_transform(&name, content)?;
            }
            self.input_guardrail_results.extend(results);
        }
        if let Some(parallel) = self.guardrails.parallel.take().filter(|g| !g.is_empty()) {
            // `tokio::spawn` does not inherit task-locals, so the run's tracing switch is
            // re-established inside the task; otherwise spans opened by a guardrail escape
            // `RunConfig.tracing_disabled`.
            let disabled = self.options.run_config.tracing.disabled;
            let fut = self.guardrails.runner(parallel, &self.context);
            self.guardrails.pending = Some(AbortOnDrop(tokio::spawn(
                crate::tracing::with_run_tracing_disabled(disabled, fut),
            )));
        }
        Ok(())
    }

    /// Input guardrails ran alongside the first turn; a tripwire aborts the run.
    async fn settle_input_guardrails(&mut self) -> Result<(), AgentsError> {
        if let Some(task) = self.guardrails.pending.take() {
            let results = task
                .await
                .map_err(|e| AgentsError::internal(format!("input guardrail task failed: {e}")))?;
            tripwire(&results)?;
            reject_parallel_transform(&results)?;
            self.input_guardrail_results.extend(results);
        }
        Ok(())
    }

    /// Replace the run's new input with what a blocking input guardrail returned, everywhere the
    /// original would otherwise go: the model input, the parallel guardrails, the session and the
    /// result. History loaded from a session is left alone.
    fn apply_input_transform(
        &mut self,
        guardrail: &str,
        content: Value,
    ) -> Result<(), AgentsError> {
        let new_input = match content {
            Value::String(text) => InputLike::Text(text),
            Value::Array(items) => InputLike::Items(items),
            other => {
                return Err(UserError::new(format!(
                    "input guardrail `{guardrail}` returned a transform that is neither a string \
                     nor an array of input items: {other}"
                ))
                .into())
            }
        };
        let old_items = ItemHelpers::input_to_new_input_list(&self.input);
        let new_items = ItemHelpers::input_to_new_input_list(&new_input);
        let start = self.current_input_items.len().checked_sub(old_items.len());
        let in_place = start.filter(|&i| self.current_input_items[i..] == old_items[..]);
        let Some(start) = in_place else {
            return Err(UserError::new(format!(
                "input guardrail `{guardrail}` returned Verdict::Transform, but the run's input was \
                 already changed by the session input callback, so it cannot be replaced safely"
            ))
            .into());
        };
        tracing::debug!(guardrail, "input replaced by an input guardrail transform");
        self.current_input_items.truncate(start);
        self.current_input_items.extend(new_items.iter().cloned());
        self.original_input_len = self.original_input_len + new_items.len() - old_items.len();
        // A `pending_input` of `None` means the input was already saved; leave it alone.
        if let Some(writer) = self.session_writer.as_deref_mut() {
            if writer.pending_input.is_some() {
                writer.pending_input = Some(new_items);
            }
        }
        self.guardrails.input = new_input.clone();
        self.input = new_input;
        Ok(())
    }

    /// Whether this turn leaves the run's input unsaved because blocking input guardrails have
    /// not run yet: they may rewrite it with a `Verdict::Transform`, and what the session already
    /// holds cannot be rewritten. Later turns have no blocking guardrails left.
    fn input_waits_for_guardrails(&self) -> bool {
        self.guardrails
            .blocking
            .as_ref()
            .is_some_and(|g| !g.is_empty())
            && self
                .session_writer
                .as_ref()
                .is_some_and(|w| w.pending_input.is_some())
    }

    /// Ask the model for this turn's response and record it.
    async fn call_model(
        &mut self,
        setup: &TurnSetup,
        model: &dyn Model,
        turn_guard: &mut Option<SpanGuard>,
    ) -> Result<ModelResponse, AgentsError> {
        let _gen_span = generation_span(
            self.current_agent
                .model_name
                .as_deref()
                .unwrap_or("scripted"),
        );
        let (instructions, input_items) = self.model_input().await?;
        self.notify_llm_start(instructions.as_deref(), &input_items)
            .await;
        let (response, early) = self
            .request_response(model, setup, instructions.as_deref(), &input_items)
            .await?;
        if let Some(results) = early {
            self.guardrails.pending = None;
            self.input_guardrail_results.extend(results);
        }
        self.record_response(&response, turn_guard).await;
        Ok(response)
    }

    /// The instructions and input items for this call, after `call_model_input_filter` and any
    /// `ModelMiddleware`.
    async fn model_input(&self) -> Result<(Option<String>, Vec<ResponseInputItem>), AgentsError> {
        // Python: `instructions` may be a callable of `(context, agent)`.
        let instructions = self.current_agent.resolve_instructions(&self.context).await;
        let data = ModelInputData {
            input: self.current_input_items.clone(),
            instructions,
        };
        // Python (`maybe_filter_model_input`): the filter runs before the LLM hooks and only
        // affects this call.
        let mut data = match &self.options.run_config.model.input_filter {
            Some(filter) => {
                filter(CallModelData {
                    model_data: data,
                    agent: Arc::clone(&self.current_agent),
                    context: self.context.clone(),
                })
                .await?
            }
            None => data,
        };
        // Rust-only extension on top of the Python-parity filter above (see
        // `crate::model_middleware`): each step runs in the order it was added.
        for middleware in &self.options.run_config.model.middleware {
            data = middleware
                .prepare(data, &self.current_agent, &self.context)
                .await?;
        }
        Ok((data.instructions, data.input))
    }

    async fn notify_llm_start(
        &self,
        instructions: Option<&str>,
        input_items: &[ResponseInputItem],
    ) {
        if let Some(h) = &self.options.hooks {
            h.on_llm_start(
                self.context.clone(),
                &self.current_agent,
                instructions,
                input_items,
            )
            .await;
        }
        if let Some(ah) = &self.current_agent.hooks {
            ah.on_llm_start(
                self.context.clone(),
                &self.current_agent,
                instructions,
                input_items,
            )
            .await;
        }
    }

    /// Call the model with retries. Python cancels the in-flight call when a concurrent input
    /// guardrail trips, so a tripwire never waits for (or pays for) the full response; results of
    /// a guardrail that finished first are returned alongside the response.
    async fn request_response(
        &mut self,
        model: &dyn Model,
        setup: &TurnSetup,
        instructions: Option<&str>,
        input_items: &[ResponseInputItem],
    ) -> Result<(ModelResponse, Option<Vec<InputGuardrailResult>>), AgentsError> {
        // Everything an attempt reads is a plain reference, so each attempt rebuilds its request
        // and a retry can start it again from scratch.
        let emitted_unsafe = Arc::new(AtomicBool::new(false));
        // Python maps `trace_include_sensitive_data=False` to
        // `ModelTracing.ENABLED_WITHOUT_DATA`, so adapters drop payloads.
        let tracing_mode = if crate::tracing::tracing_disabled() {
            ModelTracing::Disabled
        } else if self.options.run_config.tracing.include_sensitive_data {
            ModelTracing::Enabled
        } else {
            ModelTracing::EnabledWithoutData
        };
        let previous_ref = self.previous_response_id.as_deref();
        let conversation_ref = self.options.conversation_id.as_deref();
        let output_schema_ref = self.current_agent.output_type.as_deref();
        let events_ref = &self.events;
        let settings_ref = &setup.model_settings;
        let tools_ref = &setup.tools;
        let attempt_flag = Arc::clone(&emitted_unsafe);
        let model_fut = crate::retry::call_with_retry(
            crate::retry::RetryCall {
                settings: settings_ref.retry.as_ref(),
                previous_response_id: previous_ref,
                conversation_id: conversation_ref,
                timeout: settings_ref.timeout,
                stream: events_ref.is_some(),
                emitted_unsafe_event: events_ref.is_some().then_some(&*emitted_unsafe),
            },
            |request| model.get_retry_advice(request),
            move || {
                let attempt_flag = Arc::clone(&attempt_flag);
                async move {
                    let req = ModelRequest {
                        system_instructions: instructions,
                        input: ModelInput::Items(input_items),
                        model_settings: settings_ref,
                        tools: tools_ref,
                        tracing: tracing_mode,
                        previous_response_id: previous_ref,
                        conversation_id: conversation_ref,
                        output_schema: output_schema_ref,
                    };
                    attempt_model_call(model, req, events_ref, attempt_flag).await
                }
            },
        );
        let model_fut = async move { model_fut.await.map_err(AgentsError::from) };
        tokio::pin!(model_fut);
        let mut early_guardrails = None;
        let response = match self.guardrails.pending.as_mut() {
            Some(task) => tokio::select! {
                joined = &mut *task => {
                    let results = joined.map_err(|e| {
                        AgentsError::internal(format!("input guardrail task failed: {e}"))
                    })?;
                    tripwire(&results)?;
                    reject_parallel_transform(&results)?;
                    early_guardrails = Some(results);
                    (&mut model_fut).await?
                }
                response = &mut model_fut => response?,
            },
            None => (&mut model_fut).await?,
        };
        Ok((response, early_guardrails))
    }

    /// Account for a response: usage, the snapshot streaming consumers read, and `on_llm_end`.
    async fn record_response(
        &mut self,
        response: &ModelResponse,
        turn_guard: &mut Option<SpanGuard>,
    ) {
        if let Some(id) = &response.response_id {
            self.previous_response_id = Some(id.clone());
        }
        self.usage.add(&response.usage);
        for guard in [self.task_guard.as_mut(), turn_guard.as_mut()]
            .into_iter()
            .flatten()
        {
            guard.add_usage(&response.usage);
        }
        self.context.add_usage(&response.usage);
        self.raw_responses.push(response.clone());
        if let Some(snap) = &self.snapshot {
            let mut s = snap
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            s.new_items = self.generated_items.clone();
            s.raw_responses = self.raw_responses.clone();
            s.usage = self.usage.clone();
        }
        if let Some(h) = &self.options.hooks {
            h.on_llm_end(self.context.clone(), &self.current_agent, response)
                .await;
        }
        if let Some(ah) = &self.current_agent.hooks {
            ah.on_llm_end(self.context.clone(), &self.current_agent, response)
                .await;
        }
    }

    /// A response with no tool calls either ends the run or sends the model round again.
    async fn complete_without_tools(
        &mut self,
        parts: &ResponseParts,
        response: &ModelResponse,
    ) -> Result<Option<RunResult>, AgentsError> {
        let outcome = final_output_without_tools(
            parts,
            response,
            &self.input,
            &mut self.generated_items,
            &self.current_agent,
            &self.options,
            &self.context,
        )
        .await?;
        match outcome {
            TurnOutput::Final(output) => Ok(Some(self.finish(output, false).await?)),
            TurnOutput::Again => {
                for item in &response.output {
                    self.current_input_items
                        .push(apply_reasoning_item_id_policy(
                            item,
                            self.options.run_config.model.reasoning_item_id_policy,
                        ));
                }
                Ok(None)
            }
        }
    }

    /// Run the turn's tools; a pause for a resolution ends the run with an interrupted result.
    async fn complete_with_tools(
        &mut self,
        parts: &ResponseParts,
        response: ModelResponse,
        setup: TurnSetup,
        resolutions: &ResolutionStore,
    ) -> Result<Option<RunResult>, AgentsError> {
        // Python records tool usage for the turn so `tool_choice` can be reset on the next turn
        // (`AgentToolUseTracker.record_processed_response`).
        self.agents_used_tools
            .insert(self.current_agent.name.clone());
        let (handoff_calls, tool_calls) = split_calls(parts, &setup.enabled_handoffs)?;

        let plan = plan_tool_calls(
            &self.current_agent,
            &setup.tools,
            &tool_calls,
            resolutions,
            &self.generated_items,
            &self.options.run_config,
            &self.context,
        )
        .await?;
        let (tool_results, interrupted) = execute_turn_tools(
            plan,
            ToolTurn {
                agent: &self.current_agent,
                context: &self.context,
                options: &self.options,
                nested_agent_runs: &self.nested_agent_runs,
                max_tool_concurrency: self.max_tool_concurrency,
                guardrail_log: &self.tool_guardrail_log,
                events: &self.events,
            },
            &mut self.generated_items,
        )
        .await?;
        if let Some(signal) = interrupted {
            return Ok(Some(self.pause(signal, response, setup.model_settings)));
        }
        self.after_tools(&handoff_calls, &response, &tool_results)
            .await
    }

    /// After the tools ran: record their outputs, then hand off, finish on a tool result, or go
    /// round again.
    async fn after_tools(
        &mut self,
        handoff_calls: &[(Handoff, Value, String)],
        response: &ModelResponse,
        tool_results: &[ToolResult],
    ) -> Result<Option<RunResult>, AgentsError> {
        // Clear nested states for completed outer tool calls.
        for (_tool, _output, call_id) in tool_results {
            self.nested_agent_runs.remove(call_id);
        }
        // Python builds a `ToolCallOutputItem` for every completed tool result *before* deciding
        // whether one of them is the final output (`_build_tool_result_items`, then
        // `check_for_final_output_from_tools`), so a `stop_on_first_tool` run still records the
        // sibling outputs it produced.
        for (_tool, output, call_id) in tool_results {
            push_tool_output(
                &self.events,
                &mut self.generated_items,
                &self.current_agent.name,
                call_id,
                output,
            )
            .await;
        }

        let handed_off = perform_handoff(
            HandoffTurn {
                handoff_calls,
                response,
                tool_results,
                original_input_len: self.original_input_len,
                events: &self.events,
                options: &self.options,
                context: &self.context,
            },
            &mut self.current_agent,
            &mut self.known_agents,
            &mut self.current_input_items,
            &mut self.generated_items,
        )
        .await?;
        if handed_off {
            // A handoff ended this agent's span; the new agent's starts with its first turn.
            self.agent_span_due = true;
            return Ok(None);
        }

        if let Some(output) = tool_use_behavior_output(
            &self.current_agent.tool_use_behavior,
            &self.current_agent,
            tool_results,
            &self.context,
        ) {
            return Ok(Some(self.finish(output, true).await?));
        }

        record_turn_output(
            response,
            tool_results,
            &self.current_agent.name,
            &self.events,
            &mut self.generated_items,
            &mut self.current_input_items,
            self.options.run_config.model.reasoning_item_id_policy,
        )
        .await;
        Ok(None)
    }

    /// End the run with `output`. `from_tool` marks an output that came from a tool, which a
    /// blocking output guardrail withholds.
    async fn finish(&mut self, output: Value, from_tool: bool) -> Result<RunResult, AgentsError> {
        let record = self.take_record();
        let input_guardrail_results = std::mem::take(&mut self.input_guardrail_results);
        finalize_run(
            record,
            output,
            &self.current_agent,
            FinalizeEnv {
                context: &self.context,
                hooks: self.options.hooks.as_ref(),
                extra_output_guardrails: &self.options.run_config.output_guardrails,
                input_guardrail_results,
                tool_origin: from_tool.then_some(&self.options.run_config),
                session_writer: self.session_writer.as_deref_mut(),
            },
        )
        .await
    }

    /// End the run paused, with everything needed to resume it.
    fn pause(
        &mut self,
        signal: super::turn::InterruptSignal,
        response: ModelResponse,
        model_settings: crate::model_settings::ModelSettings,
    ) -> RunResult {
        for (call_id, state) in signal.nested_states {
            self.nested_agent_runs.insert(call_id, state);
        }
        for interruption in &signal.interruptions {
            self.generated_items
                .push(RunItem::ToolInterruption(interruption.clone()));
        }
        let snapshot = InterruptSnapshot {
            starting_agent_name: self.starting_agent.name.clone(),
            agent_graph: Some(agent_graph_fingerprint(&self.starting_agent)),
            current_input_items: std::mem::take(&mut self.current_input_items),
            turn: self.turn,
            previous_response_id: self.previous_response_id.take(),
            model_settings,
            pending_response: response,
            resolutions: std::mem::take(&mut self.live_resolutions),
            nested_agent_runs: std::mem::take(&mut self.nested_agent_runs),
            session_saved_items: 0,
            session_input_saved: false,
            input_guardrail_results: self.input_guardrail_results.clone(),
            loaded_capabilities: self.context.loaded_capabilities(),
        };
        let record = self.take_record();
        interrupted_result(
            record,
            Arc::clone(&self.current_agent),
            signal.interruptions,
            snapshot,
        )
    }

    /// Move the run's accumulated record out; the loop is not used again afterwards.
    fn take_record(&mut self) -> RunRecord {
        RunRecord {
            input: std::mem::replace(&mut self.input, InputLike::Text(String::new())),
            new_items: std::mem::take(&mut self.generated_items),
            raw_responses: std::mem::take(&mut self.raw_responses),
            max_turns: self.max_turns,
            usage: std::mem::take(&mut self.usage),
        }
    }
}

/// Separate the response's handoff calls from its plain tool calls.
fn split_calls(
    parts: &ResponseParts,
    enabled_handoffs: &[Handoff],
) -> Result<SplitCalls, AgentsError> {
    let mut handoff_calls = Vec::new();
    let mut tool_calls = Vec::new();
    for call in &parts.function_calls {
        let (name, _args, call_id) = required_function_call_parts(call)?;
        match enabled_handoffs.iter().find(|h| h.tool_name == name) {
            Some(h) => handoff_calls.push((h.clone(), call.clone(), call_id)),
            None => tool_calls.push(call.clone()),
        }
    }
    Ok((handoff_calls, tool_calls))
}

/// One attempt at the model call, streaming raw events to the consumer when there is one.
async fn attempt_model_call(
    model: &dyn Model,
    req: ModelRequest<'_>,
    events: &Option<EventTx>,
    attempt_flag: Arc<AtomicBool>,
) -> Result<ModelResponse, ModelError> {
    if events.is_none() {
        return model.get_response(req).await;
    }
    let (raw_tx, mut raw_rx) = mpsc::channel::<Value>(64);
    let forward = {
        let events = events.clone();
        AbortOnDrop(tokio::spawn(async move {
            while let Some(data) = raw_rx.recv().await {
                // Python: once an event other than `response.created` / `response.in_progress`
                // reached the consumer, a replay would show it output twice.
                let kind = data.get("type").and_then(Value::as_str);
                if !matches!(kind, Some("response.created" | "response.in_progress")) {
                    attempt_flag.store(true, Ordering::SeqCst);
                }
                emit(&events, StreamEvent::RawResponse { data }).await;
            }
        }))
    };
    let outcome = model.stream_response(req, raw_tx).await;
    let _ = forward.await;
    outcome
}
