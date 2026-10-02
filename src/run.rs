//! Runner and run configuration (Python: `agents.run` / `run_config`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::agent::{Agent, AsToolConfig, ToolUseBehavior};
use crate::error::{AgentsError, MaxTurnsExceeded, ModelError, UserError};
use crate::handoffs::Handoff;
use crate::items::{
    extract_message_text, function_call_parts, is_function_call, InputLike, ItemHelpers,
    MessageOutputItem, ModelResponse, ResponseOutputItem, RunItem, ToolApprovalItem, ToolCallItem,
    ToolCallOutputItem,
};
use crate::model::{Model, ModelInput, ModelRequest, ModelTracing};
use crate::model_settings::ModelSettings;
use crate::result::{InterruptSnapshot, RunResult, RunResultStreaming, StreamingSnapshot};
use crate::run_state::{ApprovalDecision, ApprovalStore, RunState};
use crate::stream_events::{RunItemStreamName, StreamEvent};
use crate::tool::{FunctionTool, ToolContext, ToolResult};
use crate::tracing::{agent_span, function_span, generation_span, trace};
use crate::usage::Usage;

tokio::task_local! {
    static NESTED_RESUME_STATES: RefCell<HashMap<String, RunState>>;
}

/// Take a nested resume state for an outer tool call id (used by `Agent.as_tool`).
pub(crate) fn take_nested_resume_state(call_id: &str) -> Option<RunState> {
    NESTED_RESUME_STATES
        .try_with(|cell| cell.borrow_mut().remove(call_id))
        .ok()
        .flatten()
}

/// Default max turns (Python: `DEFAULT_MAX_TURNS = 10`).
pub const DEFAULT_MAX_TURNS: usize = 10;

/// Which OpenAI HTTP API to use by default (Python: `set_default_openai_api`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DefaultOpenAiApi {
    /// Responses API (Python default).
    #[default]
    Responses,
    /// Chat Completions API.
    ChatCompletions,
}

static DEFAULT_OPENAI_API: OnceLock<Mutex<DefaultOpenAiApi>> = OnceLock::new();

fn default_openai_api_slot() -> &'static Mutex<DefaultOpenAiApi> {
    DEFAULT_OPENAI_API.get_or_init(|| Mutex::new(DefaultOpenAiApi::Responses))
}

/// Set the default OpenAI API (Python: `set_default_openai_api`).
pub fn set_default_openai_api(api: DefaultOpenAiApi) {
    *default_openai_api_slot().lock().expect("api lock") = api;
}

/// Get the default OpenAI API.
pub fn get_default_openai_api() -> DefaultOpenAiApi {
    *default_openai_api_slot().lock().expect("api lock")
}

/// Run configuration subset (Python: `RunConfig`).
#[derive(Clone, Default)]
pub struct RunConfig {
    /// Override model instance for the whole run.
    pub model: Option<Arc<dyn Model>>,
    /// Override model settings (merged over agent settings).
    pub model_settings: Option<ModelSettings>,
    /// Disable tracing for this run.
    pub tracing_disabled: bool,
    /// Workflow name for the root trace.
    pub workflow_name: Option<String>,
}

impl std::fmt::Debug for RunConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunConfig")
            .field("model_bound", &self.model.is_some())
            .field("model_settings", &self.model_settings)
            .field("tracing_disabled", &self.tracing_disabled)
            .field("workflow_name", &self.workflow_name)
            .finish()
    }
}

/// Per-call options for [`Runner::run`] / [`Runner::run_streamed`].
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Max turns before [`MaxTurnsExceeded`].
    pub max_turns: Option<usize>,
    /// Run configuration.
    pub run_config: RunConfig,
    /// Previous Responses API response id.
    pub previous_response_id: Option<String>,
    /// Conversation id.
    pub conversation_id: Option<String>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            max_turns: Some(DEFAULT_MAX_TURNS),
            run_config: RunConfig {
                workflow_name: Some("Agent workflow".into()),
                ..Default::default()
            },
            previous_response_id: None,
            conversation_id: None,
        }
    }
}

type EventTx = mpsc::Sender<Result<StreamEvent, AgentsError>>;

/// Facade entry point (Python: `Runner`).
pub struct Runner;

impl Runner {
    /// Run an agent asynchronously (Python: `Runner.run` with string/list input).
    pub async fn run(
        starting_agent: &Agent,
        input: impl Into<InputLike>,
        options: RunOptions,
    ) -> Result<RunResult, AgentsError> {
        run_loop(
            starting_agent.clone(),
            LoopStart::Fresh {
                input: input.into(),
            },
            options,
            None,
            None,
        )
        .await
    }

    /// Resume a paused run after approve/reject (Python: `Runner.run(agent, state)`).
    pub async fn run_state(
        starting_agent: &Agent,
        state: RunState,
        options: RunOptions,
    ) -> Result<RunResult, AgentsError> {
        if state.starting_agent_name != starting_agent.name {
            return Err(UserError::new(format!(
                "RunState starting agent `{}` does not match `{}`",
                state.starting_agent_name, starting_agent.name
            ))
            .into());
        }
        run_loop(
            starting_agent.clone(),
            LoopStart::Resume { state },
            options,
            None,
            None,
        )
        .await
    }

    /// Run in streaming mode (Python: `Runner.run_streamed`).
    pub fn run_streamed(
        starting_agent: Agent,
        input: impl Into<InputLike>,
        options: RunOptions,
    ) -> RunResultStreaming {
        let input = input.into();
        let max_turns = options.max_turns;
        let (tx, rx) = mpsc::channel(64);
        let snapshot = Arc::new(Mutex::new(StreamingSnapshot {
            current_agent_name: starting_agent.name.clone(),
            ..Default::default()
        }));
        let snap = Arc::clone(&snapshot);
        tokio::spawn(async move {
            let result = run_loop(
                starting_agent,
                LoopStart::Fresh { input },
                options,
                Some(tx.clone()),
                Some(Arc::clone(&snap)),
            )
            .await;
            match result {
                Ok(r) => {
                    let mut s = snap.lock().expect("snapshot");
                    s.is_complete = true;
                    s.final_output = if r.interruptions.is_empty() {
                        Some(r.final_output.clone())
                    } else {
                        None
                    };
                    s.interruptions = r.interruptions.clone();
                    s.new_items = r.new_items;
                    s.raw_responses = r.raw_responses;
                    s.usage = r.usage;
                    s.current_agent_name = r.last_agent_name;
                }
                Err(e) => {
                    {
                        let mut s = snap.lock().expect("snapshot");
                        s.error = Some(e.to_string());
                        s.is_complete = true;
                    }
                    let _ = tx.send(Err(e)).await;
                }
            }
        });
        RunResultStreaming::new(snapshot, max_turns, rx)
    }

    /// Blocking wrapper (Python: `run_sync` → Rust `run_blocking`, see D-001).
    pub fn run_blocking(
        starting_agent: &Agent,
        input: impl Into<InputLike>,
        options: RunOptions,
    ) -> Result<RunResult, AgentsError> {
        let input = input.into();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| AgentsError::Internal(e.to_string()))?;
        rt.block_on(Self::run(starting_agent, input, options))
    }
}

enum LoopStart {
    Fresh { input: InputLike },
    Resume { state: RunState },
}

async fn emit(tx: &Option<EventTx>, event: StreamEvent) {
    if let Some(tx) = tx {
        let _ = tx.send(Ok(event)).await;
    }
}

fn tools_for_agent(agent: &Agent) -> Vec<FunctionTool> {
    let mut tools = agent.enabled_tools();
    for h in &agent.handoffs {
        tools.push(handoff_as_tool(h));
    }
    tools
}

fn handoff_as_tool(h: &Handoff) -> FunctionTool {
    FunctionTool::new(
        h.tool_name.clone(),
        h.tool_description.clone(),
        json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }),
        |_ctx, _args| async { Ok(Value::Null) },
    )
}

async fn run_loop(
    starting_agent: Agent,
    start: LoopStart,
    options: RunOptions,
    events: Option<EventTx>,
    snapshot: Option<Arc<Mutex<StreamingSnapshot>>>,
) -> Result<RunResult, AgentsError> {
    let max_turns = options.max_turns.unwrap_or(DEFAULT_MAX_TURNS);
    let workflow = options
        .run_config
        .workflow_name
        .clone()
        .unwrap_or_else(|| "Agent workflow".into());

    let _root = if options.run_config.tracing_disabled {
        None
    } else {
        Some(trace(&workflow))
    };

    let starting_agent_name = starting_agent.name.clone();
    let mut current_agent = starting_agent;
    let mut model_settings = current_agent.model_settings.clone();
    if let Some(override_settings) = &options.run_config.model_settings {
        merge_settings(&mut model_settings, override_settings);
    }

    let mut generated_items: Vec<RunItem> = Vec::new();
    let mut raw_responses: Vec<ModelResponse> = Vec::new();
    let mut usage = Usage::default();
    let input: InputLike;
    let mut current_input_items: Vec<_>;
    let mut previous_response_id = options.previous_response_id.clone();
    let mut turn = 0usize;
    let mut resume_pending: Option<(ModelResponse, ApprovalStore)> = None;
    let mut nested_agent_runs: HashMap<String, RunState> = HashMap::new();
    let mut live_approvals = ApprovalStore::default();

    match start {
        LoopStart::Fresh { input: fresh } => {
            input = fresh;
            current_input_items = ItemHelpers::input_to_new_input_list(&input);
        }
        LoopStart::Resume { state } => {
            input = state.input.clone();
            current_agent = resolve_agent_by_name(&current_agent, &state.current_agent_name)?;
            model_settings = state.model_settings.clone();
            if let Some(override_settings) = &options.run_config.model_settings {
                merge_settings(&mut model_settings, override_settings);
            }
            generated_items = state.generated_items;
            // Drop prior ToolApproval placeholders; resume will re-create if still pending.
            generated_items.retain(|i| !matches!(i, RunItem::ToolApproval(_)));
            raw_responses = state.raw_responses;
            usage = state.usage;
            current_input_items = state.current_input_items;
            previous_response_id = state.previous_response_id.or(previous_response_id);
            turn = state.turn.saturating_sub(1); // loop will increment
            nested_agent_runs = state.nested_agent_runs.clone();
            live_approvals = state.approvals.clone();
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

    loop {
        turn += 1;
        if let Some(snap) = &snapshot {
            let mut s = snap.lock().expect("snapshot");
            s.current_turn = turn;
            s.current_agent_name = current_agent.name.clone();
        }
        if turn > max_turns {
            return Err(MaxTurnsExceeded { max_turns }.into());
        }

        let model = resolve_model(&current_agent, &options.run_config)?;
        let tools = tools_for_agent(&current_agent);

        let _agent_span = agent_span(&current_agent.name);

        let (response, approvals_map, from_resume) =
            if let Some((pending, approvals)) = resume_pending.take() {
                (pending, approvals, true)
            } else {
                let _gen_span =
                    generation_span(current_agent.model_name.as_deref().unwrap_or("scripted"));
                let response = {
                    let req = ModelRequest {
                        system_instructions: current_agent.instructions.as_deref(),
                        input: ModelInput::Items(&current_input_items),
                        model_settings: &model_settings,
                        tools: &tools,
                        tracing: if options.run_config.tracing_disabled {
                            ModelTracing::Disabled
                        } else {
                            ModelTracing::Enabled
                        },
                        previous_response_id: previous_response_id.as_deref(),
                        conversation_id: options.conversation_id.as_deref(),
                    };
                    if events.is_some() {
                        let (raw_tx, mut raw_rx) = mpsc::channel::<Value>(64);
                        let forward = {
                            let events = events.clone();
                            tokio::spawn(async move {
                                while let Some(data) = raw_rx.recv().await {
                                    emit(&events, StreamEvent::RawResponse { data }).await;
                                }
                            })
                        };
                        let response = model.stream_response(req, raw_tx).await?;
                        let _ = forward.await;
                        response
                    } else {
                        model.get_response(req).await?
                    }
                };
                if events.is_none() {
                    emit(
                        &events,
                        StreamEvent::RawResponse {
                            data: json!({
                                "type": "response.completed",
                                "response_id": response.response_id,
                                "output": response.output,
                            }),
                        },
                    )
                    .await;
                }
                if let Some(id) = &response.response_id {
                    previous_response_id = Some(id.clone());
                }
                usage.add(&response.usage);
                raw_responses.push(response.clone());
                if let Some(snap) = &snapshot {
                    let mut s = snap.lock().expect("snapshot");
                    s.new_items = generated_items.clone();
                    s.raw_responses = raw_responses.clone();
                    s.usage = usage.clone();
                }
                (response, live_approvals.clone(), false)
            };

        let (function_calls, messages): (Vec<_>, Vec<_>) = response
            .output
            .iter()
            .cloned()
            .partition(|item| is_function_call(item));

        if !from_resume {
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
                let item = RunItem::ToolCall(ToolCallItem {
                    agent_name: current_agent.name.clone(),
                    raw_item: call.clone(),
                });
                let is_handoff = function_call_parts(call)
                    .map(|(n, _, _)| current_agent.handoffs.iter().any(|h| h.tool_name == n))
                    .unwrap_or(false);
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
            let final_text = messages
                .iter()
                .filter_map(|m| extract_message_text(m))
                .collect::<Vec<_>>()
                .join("");
            return Ok(finished_result(
                input,
                generated_items,
                raw_responses,
                Value::String(final_text),
                current_agent.name.clone(),
                max_turns,
                usage,
            ));
        }

        // Prefer the first handoff call if present (Python ignores extras).
        let mut handoff_done = false;
        for call in &function_calls {
            let Some((name, _args, call_id)) = function_call_parts(call) else {
                continue;
            };
            let Some(h) = current_agent
                .handoffs
                .iter()
                .find(|h| h.tool_name == name)
                .cloned()
            else {
                continue;
            };

            let transfer = json!({"assistant": h.agent.name}).to_string();
            let out_item =
                tool_output_item(&current_agent.name, &call_id, &Value::String(transfer.clone()));
            emit(
                &events,
                StreamEvent::RunItem {
                    name: RunItemStreamName::HandoffOccured,
                    item: out_item.clone(),
                },
            )
            .await;
            generated_items.push(out_item);

            for item in &response.output {
                current_input_items.push(item.clone());
            }
            current_input_items.push(ItemHelpers::function_call_output(&call_id, transfer));

            current_agent = (*h.agent).clone();
            model_settings = current_agent.model_settings.clone();
            if let Some(override_settings) = &options.run_config.model_settings {
                merge_settings(&mut model_settings, override_settings);
            }

            emit(
                &events,
                StreamEvent::AgentUpdated {
                    agent_name: current_agent.name.clone(),
                },
            )
            .await;
            handoff_done = true;
            break;
        }
        if handoff_done {
            continue;
        }

        let tool_plan = plan_tool_calls(
            &current_agent,
            &tools,
            &function_calls,
            &approvals_map,
            &generated_items,
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
                let (auto_results, nested_interruptions, nested_states) =
                    execute_planned_tools(&auto_plan, &nested_agent_runs).await?;
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
                        current_agent.name.clone(),
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
                current_agent.name.clone(),
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
                },
            ));
        }

        // All calls decided — execute remaining tools (rejections already resolved to outputs).
        let (tool_results, nested_interruptions, nested_states) =
            execute_planned_tools(&tool_plan, &nested_agent_runs).await?;
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
                current_agent.name.clone(),
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
                },
            ));
        }
        // Clear nested states for completed outer tool calls.
        for (_tool, _output, call_id) in &tool_results {
            nested_agent_runs.remove(call_id);
        }

        match &current_agent.tool_use_behavior {
            ToolUseBehavior::StopOnFirstTool => {
                let (_tool, output, call_id) = &tool_results[0];
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
                return Ok(finished_result(
                    input,
                    generated_items,
                    raw_responses,
                    output.clone(),
                    current_agent.name.clone(),
                    max_turns,
                    usage,
                ));
            }
            ToolUseBehavior::StopAtTools { stop_at_tool_names } => {
                for (tool, output, call_id) in &tool_results {
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
                    if stop_at_tool_names.iter().any(|n| n == &tool.name) {
                        return Ok(finished_result(
                            input,
                            generated_items,
                            raw_responses,
                            output.clone(),
                            current_agent.name.clone(),
                            max_turns,
                            usage,
                        ));
                    }
                }
            }
            ToolUseBehavior::RunLlmAgain => {
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
                }
            }
        }

        for item in &response.output {
            current_input_items.push(item.clone());
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

        if current_agent.reset_tool_choice {
            model_settings.tool_choice = None;
        }
    }
}

struct ToolPlan {
    /// Still waiting on human decisions.
    interruptions: Vec<ToolApprovalItem>,
    /// Rejected calls: (tool, rejection output, call_id).
    ready_outputs: Vec<(FunctionTool, Value, String)>,
    /// Tools that should be invoked now: (tool, arguments, call_id).
    to_invoke: Vec<(FunctionTool, String, String)>,
    /// Already-done call ids (skip invoke, output already in generated_items).
    already_done: Vec<(FunctionTool, Value, String)>,
}

async fn plan_tool_calls(
    agent: &Agent,
    tools: &[FunctionTool],
    function_calls: &[ResponseOutputItem],
    approvals: &ApprovalStore,
    generated_items: &[RunItem],
) -> Result<ToolPlan, AgentsError> {
    let mut plan = ToolPlan {
        interruptions: Vec::new(),
        ready_outputs: Vec::new(),
        to_invoke: Vec::new(),
        already_done: Vec::new(),
    };

    for call in function_calls {
        let (name, arguments, call_id) = function_call_parts(call)
            .ok_or_else(|| ModelError::Behavior("malformed function_call item".into()))?;
        let tool = tools
            .iter()
            .find(|t| t.name == name)
            .cloned()
            .ok_or_else(|| UserError::new(format!("Tool not found: {name}")))?;

        if let Some(existing) = existing_tool_output(generated_items, &call_id) {
            plan.already_done.push((tool, existing, call_id));
            continue;
        }

        match approvals.status(&agent.name, &name, &call_id) {
            Some(ApprovalDecision::Approved) => {
                plan.to_invoke.push((tool, arguments, call_id));
            }
            Some(ApprovalDecision::Rejected { message }) => {
                plan.ready_outputs
                    .push((tool, Value::String(message), call_id));
            }
            None => {
                if tool.needs_approval.requires(&arguments, &call_id).await {
                    plan.interruptions.push(ToolApprovalItem {
                        agent_name: agent.name.clone(),
                        tool_name: name,
                        call_id,
                        arguments,
                        raw_item: call.clone(),
                    });
                } else {
                    plan.to_invoke.push((tool, arguments, call_id));
                }
            }
        }
    }
    Ok(plan)
}

async fn execute_planned_tools(
    plan: &ToolPlan,
    nested_resume: &HashMap<String, RunState>,
) -> Result<
    (
        Vec<(FunctionTool, Value, String)>,
        Vec<ToolApprovalItem>,
        HashMap<String, RunState>,
    ),
    AgentsError,
> {
    let mut results: Vec<(FunctionTool, Value, String)> = Vec::new();
    let mut order: Vec<String> = Vec::new();
    let mut nested_interruptions = Vec::new();
    let mut nested_states = HashMap::new();

    for (tool, output, call_id) in &plan.already_done {
        order.push(call_id.clone());
        results.push((tool.clone(), output.clone(), call_id.clone()));
    }
    for (tool, output, call_id) in &plan.ready_outputs {
        order.push(call_id.clone());
        results.push((tool.clone(), output.clone(), call_id.clone()));
    }

    for (_, _, call_id) in &plan.to_invoke {
        order.push(call_id.clone());
    }

    let resume_map = nested_resume.clone();
    let invoke_futs = plan.to_invoke.iter().map(|(tool, arguments, call_id)| {
        let tool = tool.clone();
        let arguments = arguments.clone();
        let call_id = call_id.clone();
        let resume_map = resume_map.clone();
        async move {
            let _fs = function_span(&tool.name);
            let ctx = ToolContext {
                tool_name: tool.name.clone(),
                tool_call_id: call_id.clone(),
                tool_arguments: arguments.clone(),
            };
            let result = NESTED_RESUME_STATES
                .scope(RefCell::new(resume_map), async {
                    (tool.on_invoke_tool)(ctx, arguments).await
                })
                .await?;
            Ok::<_, AgentsError>((tool, call_id, result))
        }
    });
    let invoked = futures::future::try_join_all(invoke_futs).await?;
    for (tool, call_id, result) in invoked {
        if !result.interruptions.is_empty() {
            if let Some(state) = result.nested_state {
                nested_states.insert(call_id.clone(), *state);
            }
            nested_interruptions.extend(result.interruptions);
            continue;
        }
        let output = result.output.unwrap_or(Value::Null);
        results.push((tool, output, call_id));
    }

    let index: HashMap<&str, usize> = order
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    results.sort_by_key(|(_, _, id)| index.get(id.as_str()).copied().unwrap_or(usize::MAX));
    Ok((results, nested_interruptions, nested_states))
}

fn has_tool_output(items: &[RunItem], call_id: &str) -> bool {
    items.iter().any(|gi| {
        matches!(
            gi,
            RunItem::ToolCallOutput(o)
                if o.raw_item.get("call_id").and_then(|c| c.as_str()) == Some(call_id)
        )
    })
}

fn existing_tool_output(items: &[RunItem], call_id: &str) -> Option<Value> {
    items.iter().find_map(|gi| match gi {
        RunItem::ToolCallOutput(o)
            if o.raw_item.get("call_id").and_then(|c| c.as_str()) == Some(call_id) =>
        {
            Some(o.output.clone())
        }
        _ => None,
    })
}

fn finished_result(
    input: InputLike,
    new_items: Vec<RunItem>,
    raw_responses: Vec<ModelResponse>,
    final_output: Value,
    last_agent_name: String,
    max_turns: usize,
    usage: Usage,
) -> RunResult {
    RunResult {
        input,
        new_items,
        raw_responses,
        final_output,
        last_agent_name,
        max_turns: Some(max_turns),
        usage,
        interruptions: Vec::new(),
        interrupt_state: None,
    }
}

fn interrupted_result(
    input: InputLike,
    new_items: Vec<RunItem>,
    raw_responses: Vec<ModelResponse>,
    last_agent_name: String,
    max_turns: usize,
    usage: Usage,
    interruptions: Vec<ToolApprovalItem>,
    interrupt_state: InterruptSnapshot,
) -> RunResult {
    RunResult {
        input,
        new_items,
        raw_responses,
        final_output: Value::Null,
        last_agent_name,
        max_turns: Some(max_turns),
        usage,
        interruptions,
        interrupt_state: Some(interrupt_state),
    }
}

fn resolve_agent_by_name(root: &Agent, name: &str) -> Result<Agent, AgentsError> {
    if root.name == name {
        return Ok(root.clone());
    }
    let mut stack = vec![root.clone()];
    while let Some(agent) = stack.pop() {
        for h in &agent.handoffs {
            if h.agent.name == name {
                return Ok((*h.agent).clone());
            }
            stack.push((*h.agent).clone());
        }
    }
    Err(UserError::new(format!("Agent `{name}` not found in starting agent graph")).into())
}

/// Build an `Agent.as_tool` FunctionTool (Python: `Agent.as_tool`).
pub(crate) fn build_agent_as_tool(agent: &Agent, config: AsToolConfig) -> FunctionTool {
    let nested_agent = agent.clone();
    let tool_name = config
        .name
        .unwrap_or_else(|| crate::handoffs::transform_string_function_style(&nested_agent.name));
    let tool_description = config.description.unwrap_or_else(|| {
        nested_agent
            .handoff_description
            .clone()
            .unwrap_or_else(|| format!("Agent tool: {}", nested_agent.name))
    });
    let max_turns = config.max_turns;
    let needs = config.needs_approval;

    FunctionTool::new_with_result(
        tool_name,
        tool_description,
        json!({
            "type": "object",
            "properties": {
                "input": { "type": "string", "description": "Input for the agent" }
            },
            "required": ["input"],
            "additionalProperties": false
        }),
        move |ctx, args| {
            let nested_agent = nested_agent.clone();
            async move {
                let input_text = extract_as_tool_input(&args)?;
                let mut opts = RunOptions::default();
                if let Some(mt) = max_turns {
                    opts.max_turns = Some(mt);
                }

                let result = if let Some(state) = take_nested_resume_state(&ctx.tool_call_id) {
                    Runner::run_state(&nested_agent, state, opts).await?
                } else {
                    Runner::run(&nested_agent, input_text, opts).await?
                };

                if result.is_interrupted() {
                    let nested_state = result.to_state()?;
                    return Ok(ToolResult::interrupted(
                        result.interruptions.clone(),
                        nested_state,
                    ));
                }

                Ok(ToolResult::output(result.final_output))
            }
        },
    )
    .with_needs_approval(needs)
}

fn extract_as_tool_input(args: &str) -> Result<String, AgentsError> {
    let v: Value = serde_json::from_str(args)
        .map_err(|e| AgentsError::Tool(format!("invalid agent tool args: {e}")))?;
    if let Some(s) = v.get("input").and_then(|x| x.as_str()) {
        return Ok(s.to_string());
    }
    if let Some(s) = v.as_str() {
        return Ok(s.to_string());
    }
    Err(AgentsError::Tool(
        "agent tool requires string field `input`".into(),
    ))
}

fn tool_output_item(agent_name: &str, call_id: &str, output: &Value) -> RunItem {
    let raw = ItemHelpers::function_call_output(call_id, value_to_tool_string(output));
    RunItem::ToolCallOutput(ToolCallOutputItem {
        agent_name: agent_name.to_string(),
        raw_item: raw,
        output: output.clone(),
    })
}

fn value_to_tool_string(output: &Value) -> String {
    match output {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn merge_settings(base: &mut ModelSettings, overlay: &ModelSettings) {
    if overlay.temperature.is_some() {
        base.temperature = overlay.temperature;
    }
    if overlay.top_p.is_some() {
        base.top_p = overlay.top_p;
    }
    if overlay.max_tokens.is_some() {
        base.max_tokens = overlay.max_tokens;
    }
    if overlay.tool_choice.is_some() {
        base.tool_choice = overlay.tool_choice.clone();
    }
    if overlay.parallel_tool_calls.is_some() {
        base.parallel_tool_calls = overlay.parallel_tool_calls;
    }
    if overlay.extra_body.is_some() {
        base.extra_body = overlay.extra_body.clone();
    }
}

fn resolve_model(
    agent: &Agent,
    run_config: &RunConfig,
) -> Result<Arc<dyn Model>, AgentsError> {
    if let Some(m) = &run_config.model {
        return Ok(Arc::clone(m));
    }
    if let Some(m) = &agent.model {
        return Ok(Arc::clone(m));
    }
    Err(UserError::new(
        "No model configured on Agent or RunConfig. Bind a Model (e.g. ScriptedModel) or enable the `openai` feature and use an OpenAI model.",
    )
    .into())
}
