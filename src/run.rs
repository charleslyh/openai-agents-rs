//! Runner and run configuration (Python: `agents.run` / `run_config`).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use futures::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::agent::{Agent, AsToolConfig, FunctionToolResult, ToolUseBehavior};
use crate::error::{
    AgentsError, InputGuardrailTripwireTriggered, MaxTurnsExceeded, ModelError,
    OutputGuardrailTripwireTriggered, ToolTimeoutError, UserError,
};
use crate::guardrail::{InputGuardrail, InputGuardrailResult, OutputGuardrail};
use crate::handoffs::{
    nest_handoff_history, Handoff, HandoffHistoryMapper, HandoffInputData, HandoffInputFilter,
};
use crate::items::{
    extract_message_text, is_function_call, is_reasoning, required_function_call_parts,
    HandoffCallItem, HandoffOutputItem, InputLike, ItemHelpers, MessageOutputItem, ModelResponse,
    ReasoningItem, ResponseOutputItem, RunItem, ToolApprovalItem, ToolCallItem, ToolCallOutputItem,
};
use crate::lifecycle::RunHooks;
use crate::memory::{
    prepare_input_with_session, Session, SessionInputCallback, SessionSettings,
};
use crate::model::{
    default_model_provider, Model, ModelInput, ModelProvider, ModelRef, ModelRequest, ModelTracing,
};
use crate::model_settings::ModelSettings;
use crate::result::{InterruptSnapshot, RunResult, RunResultStreaming, StreamingSnapshot};
use crate::run_context::{ContextValue, RunContextWrapper};
use crate::run_state::{ApprovalDecision, ApprovalStore, RunState};
use crate::stream_events::{RunItemStreamName, StreamEvent};
use crate::tool::{
    default_tool_timeout_error_message, FunctionTool, ToolContext, ToolResult, ToolTimeoutBehavior,
    DEFAULT_APPROVAL_REJECTION_MESSAGE,
};
use crate::tool_guardrails::{
    run_tool_input_guardrails, run_tool_output_guardrails, ToolInputGuardrailResult,
    ToolOutputGuardrailResult,
};
use crate::tracing::{agent_span, function_span, generation_span, handoff_span};
use crate::model::wire_events::FAKE_RESPONSES_ID;
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

/// Default for `trace_include_sensitive_data`, mirroring Python's
/// `OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA` (defaults to true).
pub fn default_trace_include_sensitive_data() -> bool {
    match std::env::var("OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA") {
        Ok(raw) => !matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        Err(_) => true,
    }
}

/// The input and instructions about to be sent to the model
/// (Python: `ModelInputData`).
#[derive(Debug, Clone)]
pub struct ModelInputData {
    /// Responses input items.
    pub input: Vec<Value>,
    /// System instructions.
    pub instructions: Option<String>,
}

/// Payload given to [`RunConfig::call_model_input_filter`] (Python: `CallModelData`).
#[derive(Clone)]
pub struct CallModelData {
    /// The model input as prepared by the runner.
    pub model_data: ModelInputData,
    /// The agent about to be called.
    pub agent: Arc<Agent>,
    /// The run context.
    pub context: RunContextWrapper,
}

/// Rewrites the model input right before each model call (Python: `CallModelInputFilter`).
///
/// The filtered data is used for this call only; the run history is unchanged.
pub type CallModelInputFilter = Arc<
    dyn Fn(
            CallModelData,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<ModelInputData, AgentsError>> + Send>,
        > + Send
        + Sync,
>;

/// Name of the stand-in tool recorded for a call to a tool the agent does not have.
const TOOL_NOT_FOUND_PLACEHOLDER: &str = "__tool_not_found__";

/// Output sent for every handoff after the first in one turn (Python: same literal).
const MULTIPLE_HANDOFFS_MESSAGE: &str = "Multiple handoffs detected, ignoring this one.";

/// Snapshot of the run handed to a run error handler (Python: `RunErrorData`).
#[derive(Debug, Clone)]
pub struct RunErrorData {
    /// The run input.
    pub input: InputLike,
    /// Run items generated so far.
    pub new_items: Vec<RunItem>,
    /// The input followed by every model-visible generated item.
    pub history: Vec<Value>,
    /// The model-visible generated items.
    pub output: Vec<Value>,
    /// Raw model responses so far.
    pub raw_responses: Vec<ModelResponse>,
    /// The agent that was running.
    pub last_agent: Arc<Agent>,
}

/// Input of a run error handler (Python: `RunErrorHandlerInput`).
#[derive(Clone)]
pub struct RunErrorHandlerInput {
    /// The error that stopped the run.
    pub error: MaxTurnsExceeded,
    /// The run context.
    pub context: RunContextWrapper,
    /// The run so far.
    pub run_data: RunErrorData,
}

/// What a run error handler produces (Python: `RunErrorHandlerResult`).
#[derive(Debug, Clone)]
pub struct RunErrorHandlerResult {
    /// The final output to finish the run with.
    pub final_output: Value,
    /// Whether the synthesized assistant message joins the run's items (default true).
    pub include_in_history: bool,
}

impl RunErrorHandlerResult {
    /// Finish the run with `final_output`, recorded in the history.
    pub fn new(final_output: impl Into<Value>) -> Self {
        Self {
            final_output: final_output.into(),
            include_in_history: true,
        }
    }
}

/// Turns an error into a final output; `None` re-raises the error (Python: `RunErrorHandler`).
pub type RunErrorHandler = Arc<
    dyn Fn(
            RunErrorHandlerInput,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Option<RunErrorHandlerResult>, AgentsError>> + Send>,
        > + Send
        + Sync,
>;

/// Error handlers keyed by error kind (Python: `RunErrorHandlers`; only `max_turns` is ported).
#[derive(Clone, Default)]
pub struct RunErrorHandlers {
    /// Called when `max_turns` is exceeded.
    pub max_turns: Option<RunErrorHandler>,
}

impl RunErrorHandlers {
    /// Handle `MaxTurnsExceeded` with an async closure.
    pub fn on_max_turns<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunErrorHandlerInput) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Option<RunErrorHandlerResult>, AgentsError>>
            + Send
            + 'static,
    {
        self.max_turns = Some(Arc::new(move |input| Box::pin(f(input))));
        self
    }
}

/// SDK-side execution settings for local tool calls (Python: `ToolExecutionConfig`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolExecutionConfig {
    /// At most this many function tools of one turn run at once; `None` starts them all
    /// (Python: `max_function_tool_concurrency`). Must be at least 1.
    pub max_function_tool_concurrency: Option<usize>,
}

/// What to do when a function tool and a handoff (or two tools) share a name
/// (Python: `RunConfig.tool_name_collision_policy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolNameCollisionPolicy {
    /// Log a warning and expose only the winner of each collision (default).
    #[default]
    Warn,
    /// Fail with a [`UserError`] before the model is called.
    Error,
}

/// Which tool error a [`ToolErrorFormatter`] is formatting (Python: `ToolErrorFormatterArgs.kind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolErrorKind {
    /// A human rejected the tool call without giving a message.
    ApprovalRejected,
    /// The model called a tool the agent does not have
    /// (only with [`ToolNotFoundBehavior::ReturnErrorToModel`]).
    ToolNotFound,
}

/// Data passed to [`RunConfig::tool_error_formatter`] (Python: `ToolErrorFormatterArgs`).
///
/// Python also passes `tool_type`; every Rust tool is a function tool, so it is omitted.
#[derive(Debug, Clone)]
pub struct ToolErrorFormatterArgs {
    /// The category of tool error being formatted.
    pub kind: ToolErrorKind,
    /// Name of the tool the model called.
    pub tool_name: String,
    /// The tool call id.
    pub call_id: String,
    /// The SDK default message for this error kind.
    pub default_message: String,
    /// The active run context.
    pub run_context: RunContextWrapper,
}

/// Rewrites the model-visible text of a tool error; `None` keeps the default
/// (Python: `ToolErrorFormatter`).
pub type ToolErrorFormatter = Arc<
    dyn Fn(
            ToolErrorFormatterArgs,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>>
        + Send
        + Sync,
>;

/// What to do when the model calls a tool the agent does not have
/// (Python: `RunConfig.tool_not_found_behavior`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolNotFoundBehavior {
    /// Fail the run with a model behavior error (Python: `"raise_error"`, default).
    #[default]
    RaiseError,
    /// Send an error output back to the model so it can recover
    /// (Python: `"return_error_to_model"`).
    ReturnErrorToModel,
}

/// Run configuration subset (Python: `RunConfig`).
#[derive(Clone)]
pub struct RunConfig {
    /// Override model for the whole run: an instance, or a name resolved by
    /// [`model_provider`](Self::model_provider) (Python: `RunConfig.model`).
    pub model: Option<ModelRef>,
    /// Provider used to resolve string model names (Python: `RunConfig.model_provider`).
    ///
    /// When unset, names go through [`default_model_provider`].
    pub model_provider: Option<Arc<dyn ModelProvider>>,
    /// Override model settings (merged over agent settings).
    pub model_settings: Option<ModelSettings>,
    /// Disable tracing for this run.
    pub tracing_disabled: bool,
    /// Workflow name for the root trace.
    pub workflow_name: Option<String>,
    /// Reuse this trace id instead of generating one (Python: `RunConfig.trace_id`).
    pub trace_id: Option<String>,
    /// Grouping identifier for the trace, e.g. a chat thread id (Python: `RunConfig.group_id`).
    pub group_id: Option<String>,
    /// Extra metadata attached to the trace (Python: `RunConfig.trace_metadata`).
    pub trace_metadata: Option<Value>,
    /// Whether tool inputs/outputs and model payloads may be recorded
    /// (Python: `RunConfig.trace_include_sensitive_data`).
    pub trace_include_sensitive_data: bool,
    /// Input guardrails applied to the run (Python: `RunConfig.input_guardrails`).
    pub input_guardrails: Vec<InputGuardrail>,
    /// Output guardrails applied to the run (Python: `RunConfig.output_guardrails`).
    pub output_guardrails: Vec<OutputGuardrail>,
    /// Behavior when the model calls an unknown tool (Python: `RunConfig.tool_not_found_behavior`).
    pub tool_not_found_behavior: ToolNotFoundBehavior,
    /// Execution settings for local function tools (Python: `RunConfig.tool_execution`).
    pub tool_execution: Option<ToolExecutionConfig>,
    /// Merge session history with the new input (Python: `RunConfig.session_input_callback`).
    pub session_input_callback: Option<SessionInputCallback>,
    /// Session read settings for this run, overlaid on the session's own
    /// (Python: `RunConfig.session_settings`).
    pub session_settings: Option<SessionSettings>,
    /// Collision handling for tool and handoff names (Python: `tool_name_collision_policy`).
    pub tool_name_collision_policy: ToolNameCollisionPolicy,
    /// Customize approval-rejection and tool-not-found messages
    /// (Python: `RunConfig.tool_error_formatter`).
    pub tool_error_formatter: Option<ToolErrorFormatter>,
    /// Compact earlier history into summary messages when handing off
    /// (Python: `RunConfig.nest_handoff_history`, opt-in, default false).
    pub nest_handoff_history: bool,
    /// Custom mapper for the nested history (Python: `RunConfig.handoff_history_mapper`);
    /// only used when nesting is on.
    pub handoff_history_mapper: Option<HandoffHistoryMapper>,
    /// Default filter for every handoff without its own (Python: `RunConfig.handoff_input_filter`).
    pub handoff_input_filter: Option<HandoffInputFilter>,
    /// Edit the model input just before each call (Python: `RunConfig.call_model_input_filter`).
    pub call_model_input_filter: Option<CallModelInputFilter>,
}

impl RunConfig {
    /// Install a [`ToolErrorFormatter`] from an async closure.
    pub fn with_tool_error_formatter<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(ToolErrorFormatterArgs) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Option<String>> + Send + 'static,
    {
        self.tool_error_formatter = Some(Arc::new(move |args| Box::pin(f(args))));
        self
    }

    /// Install a [`CallModelInputFilter`] from an async closure.
    pub fn with_call_model_input_filter<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(CallModelData) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<ModelInputData, AgentsError>> + Send + 'static,
    {
        self.call_model_input_filter = Some(Arc::new(move |data| Box::pin(f(data))));
        self
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            model: None,
            model_provider: None,
            model_settings: None,
            tracing_disabled: false,
            workflow_name: None,
            trace_id: None,
            group_id: None,
            trace_metadata: None,
            trace_include_sensitive_data: default_trace_include_sensitive_data(),
            input_guardrails: Vec::new(),
            output_guardrails: Vec::new(),
            tool_not_found_behavior: ToolNotFoundBehavior::default(),
            call_model_input_filter: None,
            handoff_input_filter: None,
            nest_handoff_history: false,
            handoff_history_mapper: None,
            tool_error_formatter: None,
            tool_name_collision_policy: ToolNameCollisionPolicy::default(),
            tool_execution: None,
            session_input_callback: None,
            session_settings: None,
        }
    }
}

impl std::fmt::Debug for RunConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunConfig")
            .field("model_bound", &self.model.is_some())
            .field("model_settings", &self.model_settings)
            .field("tracing_disabled", &self.tracing_disabled)
            .field("workflow_name", &self.workflow_name)
            .field("trace_id", &self.trace_id)
            .field("group_id", &self.group_id)
            .field(
                "trace_include_sensitive_data",
                &self.trace_include_sensitive_data,
            )
            .finish()
    }
}

/// Per-call options for [`Runner::run`] / [`Runner::run_streamed`].
#[derive(Clone)]
pub struct RunOptions {
    /// Max turns before [`MaxTurnsExceeded`].
    pub max_turns: Option<usize>,
    /// Run configuration.
    pub run_config: RunConfig,
    /// Previous Responses API response id.
    pub previous_response_id: Option<String>,
    /// Conversation id.
    pub conversation_id: Option<String>,
    /// User context shared with tools, guardrails and hooks (Python: `RunOptions.context`).
    pub context: Option<ContextValue>,
    /// Run-level lifecycle hooks (Python: `RunOptions.hooks`).
    pub hooks: Option<Arc<dyn RunHooks>>,
    /// Conversation memory (Python: `Runner.run(session=...)`).
    ///
    /// History is prepended to a fresh run's input. When the run finishes (not when it pauses
    /// for approval, and not on error) its input and new items are appended. After an approval
    /// pause, pass the same session to [`Runner::run_state`] so the resumed run is saved.
    pub session: Option<Arc<dyn Session>>,
    /// Handlers that turn run errors into a final output (Python: `error_handlers=`).
    pub error_handlers: RunErrorHandlers,
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
            context: None,
            hooks: None,
            session: None,
            error_handlers: RunErrorHandlers::default(),
        }
    }
}

impl std::fmt::Debug for RunOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunOptions")
            .field("max_turns", &self.max_turns)
            .field("run_config", &self.run_config)
            .field("previous_response_id", &self.previous_response_id)
            .field("conversation_id", &self.conversation_id)
            .field("has_context", &self.context.is_some())
            .field("has_hooks", &self.hooks.is_some())
            .field("has_session", &self.session.is_some())
            .finish()
    }
}

/// Tool guardrail results collected while a run executes tools.
#[derive(Debug, Default)]
struct ToolGuardrailLog {
    input: Vec<ToolInputGuardrailResult>,
    output: Vec<ToolOutputGuardrailResult>,
}

type SharedToolGuardrailLog = Arc<Mutex<ToolGuardrailLog>>;

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
                prepared: None,
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
        let task = tokio::spawn(async move {
            let result = run_loop(
                starting_agent,
                LoopStart::Fresh { input, prepared: None },
                options,
                Some(tx.clone()),
                Some(Arc::clone(&snap)),
            )
            .await;
            match result {
                Ok(r) => {
                    let mut s = snap.lock().expect("snapshot");
                    s.is_complete = true;
                    // A graceful cancel returns a result without a real final output.
                    let stopped_early = s.cancel_after_turn;
                    if stopped_early {
                        s.is_cancelled = true;
                    }
                    s.final_output = if r.interruptions.is_empty() && !stopped_early {
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
        RunResultStreaming::new(snapshot, max_turns, rx).with_task(task)
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
            .map_err(|e| AgentsError::internal_with_source("could not start a Tokio runtime", e))?;
        rt.block_on(Self::run(starting_agent, input, options))
    }
}

enum LoopStart {
    /// `prepared` is the model input built from the session, when there is one.
    Fresh {
        input: InputLike,
        prepared: Option<Vec<Value>>,
    },
    Resume { state: RunState },
}

async fn emit(tx: &Option<EventTx>, event: StreamEvent) {
    if let Some(tx) = tx {
        let _ = tx.send(Ok(event)).await;
    }
}

/// Python (`resolve_tool_name_collisions`): a name used twice is an error under
/// [`ToolNameCollisionPolicy::Error`]; otherwise a handoff beats a tool, and the last entry of the
/// winning kind is kept.
fn resolve_tool_name_collisions(
    tools: Vec<FunctionTool>,
    handoffs: Vec<Handoff>,
    policy: ToolNameCollisionPolicy,
) -> Result<(Vec<FunctionTool>, Vec<Handoff>), AgentsError> {
    let mut owners: Vec<(String, Vec<(bool, usize)>)> = Vec::new();
    let mut add = |name: &str, is_handoff: bool, index: usize| {
        match owners.iter_mut().find(|(n, _)| n == name) {
            Some((_, entries)) => entries.push((is_handoff, index)),
            None => owners.push((name.to_string(), vec![(is_handoff, index)])),
        }
    };
    for (i, t) in tools.iter().enumerate() {
        add(&t.name, false, i);
    }
    for (i, h) in handoffs.iter().enumerate() {
        if !h.tool_name.is_empty() {
            add(&h.tool_name, true, i);
        }
    }

    let mut drop_tools = HashSet::new();
    let mut drop_handoffs = HashSet::new();
    for (name, entries) in owners.iter().filter(|(_, e)| e.len() > 1) {
        let handoff_count = entries.iter().filter(|(is_handoff, _)| *is_handoff).count();
        let message = if handoff_count == 0 {
            format!(
                "Ambiguous function tool configuration: the tool name `{name}` is used by \
                 multiple tools. Assign a unique name to every colliding function tool."
            )
        } else if handoff_count == entries.len() {
            format!(
                "Ambiguous handoff configuration: the handoff tool name `{name}` is used by \
                 multiple handoffs. Pass a unique tool name to each handoff."
            )
        } else {
            format!(
                "Ambiguous tool routing configuration: the tool name `{name}` is used by both \
                 a function tool and a handoff. Assign a unique name to every colliding \
                 function tool and handoff."
            )
        };
        if policy == ToolNameCollisionPolicy::Error {
            return Err(UserError::new(message).into());
        }
        ::tracing::warn!("{message}");
        let winner = entries
            .iter()
            .rev()
            .find(|(is_handoff, _)| *is_handoff)
            .or_else(|| entries.last())
            .copied()
            .expect("collision has entries");
        for entry in entries.iter().filter(|e| **e != winner) {
            if entry.0 {
                drop_handoffs.insert(entry.1);
            } else {
                drop_tools.insert(entry.1);
            }
        }
    }

    let tools = tools
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !drop_tools.contains(i))
        .map(|(_, t)| t)
        .collect();
    let handoffs = handoffs
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !drop_handoffs.contains(i))
        .map(|(_, h)| h)
        .collect();
    Ok((tools, handoffs))
}

/// Python (`_validate_function_tool_timeout_config`): finite and greater than zero.
fn validate_tool_timeout(tool: &FunctionTool) -> Result<(), AgentsError> {
    match tool.timeout_seconds {
        Some(seconds) if !seconds.is_finite() => Err(UserError::new(
            "FunctionTool timeout_seconds must be a finite number.",
        )
        .into()),
        Some(seconds) if seconds <= 0.0 => Err(UserError::new(
            "FunctionTool timeout_seconds must be greater than 0.",
        )
        .into()),
        _ => Ok(()),
    }
}

fn tools_for_agent(mut tools: Vec<FunctionTool>, handoffs: &[Handoff]) -> Vec<FunctionTool> {
    for h in handoffs {
        tools.push(handoff_as_tool(h));
    }
    tools
}

fn handoff_as_tool(h: &Handoff) -> FunctionTool {
    FunctionTool::new(
        h.tool_name.clone(),
        h.tool_description.clone(),
        h.input_json_schema.clone().unwrap_or_else(|| {
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            })
        }),
        |_ctx, _args| async { Ok(Value::Null) },
    )
}

/// Run the loop with the run's tracing switch in scope.
///
/// Spans created anywhere inside the run must respect `RunConfig.tracing_disabled`, not just
/// the global switch (Python: `RunConfig.tracing_disabled` short-circuits the whole trace).
async fn run_loop(
    starting_agent: Agent,
    start: LoopStart,
    options: RunOptions,
    events: Option<EventTx>,
    snapshot: Option<Arc<Mutex<StreamingSnapshot>>>,
) -> Result<RunResult, AgentsError> {
    let run_tracing_disabled = options.run_config.tracing_disabled;
    let session = options.session.clone();
    // Python (`prepare_input_with_session`): stored history comes before the new input.
    let mut session_input_to_save: Option<Vec<Value>> = None;
    let start = match (start, &session) {
        (LoopStart::Fresh { input, .. }, Some(session)) => {
            let (prepared, to_save) = prepare_input_with_session(
                session.as_ref(),
                options.run_config.session_settings,
                options.run_config.session_input_callback.as_ref(),
                ItemHelpers::input_to_new_input_list(&input),
            )
            .await?;
            session_input_to_save = Some(to_save);
            LoopStart::Fresh {
                input,
                prepared: Some(prepared),
            }
        }
        (start, _) => start,
    };
    let tool_guardrail_log = SharedToolGuardrailLog::default();
    let mut result = crate::tracing::with_run_tracing_disabled(
        run_tracing_disabled,
        run_loop_inner(
            starting_agent,
            start,
            options,
            events,
            snapshot,
            Arc::clone(&tool_guardrail_log),
        ),
    )
    .await?;
    {
        let mut log = tool_guardrail_log.lock().expect("tool guardrail log");
        result.tool_input_guardrail_results = std::mem::take(&mut log.input);
        result.tool_output_guardrail_results = std::mem::take(&mut log.output);
    }
    // Python saves the turn's input and generated items to the session; a run paused for
    // approval is saved when it is resumed with the same session.
    if let Some(session) = session {
        if result.interruptions.is_empty() {
            let mut items = session_input_to_save
                .unwrap_or_else(|| ItemHelpers::input_to_new_input_list(&result.input));
            items.extend(
                result
                    .new_items
                    .iter()
                    .filter(|i| i.is_model_input())
                    .map(|i| i.raw_item().clone()),
            );
            session.add_items(items).await?;
        }
    }
    Ok(result)
}

async fn run_loop_inner(
    starting_agent: Agent,
    start: LoopStart,
    options: RunOptions,
    events: Option<EventTx>,
    snapshot: Option<Arc<Mutex<StreamingSnapshot>>>,
    tool_guardrail_log: SharedToolGuardrailLog,
) -> Result<RunResult, AgentsError> {
    let max_turns = options.max_turns.unwrap_or(DEFAULT_MAX_TURNS);
    // Python raises `ValueError` when the config is built; Rust has no constructor to hook, so
    // the run reports it as a `UserError` before doing any work.
    let max_tool_concurrency = options
        .run_config
        .tool_execution
        .and_then(|c| c.max_function_tool_concurrency);
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

    let starting_agent_name = starting_agent.name.clone();
    let mut current_agent = starting_agent;
    // Agents that have emitted tool calls, used to reset `tool_choice` (Python:
    // `AgentToolUseTracker` + `maybe_reset_tool_choice`).
    let mut agents_used_tools: HashSet<String> = HashSet::new();

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
            input = state.input.clone();
            original_input_len = ItemHelpers::input_to_new_input_list(&input).len();
            current_agent = resolve_agent_by_name(&current_agent, &state.current_agent_name)?;
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
            .cloned()
            .partition(|g| g.run_in_parallel);
    let run_input_guardrails = |guardrails: Vec<InputGuardrail>| {
        let agent = Arc::new(current_agent.clone());
        let run_input = input.clone();
        let ctx = context.clone();
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
    let mut pending_input_guardrails = if parallel_guardrails.is_empty() {
        None
    } else {
        // `tokio::spawn` does not inherit task-locals, so the run's tracing switch has to be
        // re-established inside the task; otherwise spans opened by a guardrail (Python runs
        // them concurrently with the first turn) escape `RunConfig.tracing_disabled`.
        let run_tracing_disabled = options.run_config.tracing_disabled;
        let fut = run_input_guardrails(parallel_guardrails);
        Some(tokio::spawn(crate::tracing::with_run_tracing_disabled(
            run_tracing_disabled,
            fut,
        )))
    };

    // `on_agent_start` is called once per agent, including after each handoff.
    if let Some(h) = &options.hooks {
        h.on_agent_start(context.clone(), &current_agent).await;
    }
    if let Some(ah) = &current_agent.hooks {
        ah.on_start(context.clone(), &current_agent).await;
    }

    if !blocking_guardrails.is_empty() {
        let results = run_input_guardrails(blocking_guardrails).await;
        if let Some(r) = results.iter().find(|r| r.output.tripwire_triggered) {
            return Err(InputGuardrailTripwireTriggered { result: r.clone() }.into());
        }
        input_guardrail_results = results;
    }

    loop {
        turn += 1;
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
        if turn > max_turns {
            let error = MaxTurnsExceeded { max_turns };
            let Some(handler) = options.error_handlers.max_turns.clone() else {
                return Err(error.into());
            };
            // Python (`finalize_max_turns_handler_output`): validate the handler's output,
            // record it as an assistant message, then run the end hooks and output guardrails.
            let run_data = build_run_error_data(&input, &generated_items, &raw_responses, &current_agent);
            let handled = handler(RunErrorHandlerInput {
                error: error.clone(),
                context: context.clone(),
                run_data,
            })
            .await?;
            let Some(handled) = handled else {
                return Err(error.into());
            };
            let (final_output, text) =
                validate_handler_final_output(&current_agent, handled.final_output)?;
            if handled.include_in_history {
                let mut message = ItemHelpers::text_message(text);
                message["id"] = Value::String(FAKE_RESPONSES_ID.to_string());
                generated_items.push(RunItem::Message(MessageOutputItem {
                    agent_name: current_agent.name.clone(),
                    raw_item: message,
                }));
            }
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
            )
            .await;
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
            current_agent.enabled_tools(&context).await,
            current_agent.enabled_handoffs(&context).await,
            options.run_config.tool_name_collision_policy,
        )?;
        for tool in &enabled_tools {
            validate_tool_timeout(tool)?;
        }
        let tools = tools_for_agent(enabled_tools, &enabled_handoffs);

        let _agent_span = agent_span(&current_agent.name);

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
                                    tokio::spawn(async move {
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
                                    })
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
            let final_text = messages
                .iter()
                .filter_map(|m| extract_message_text(m))
                .collect::<Vec<_>>()
                .join("");
            // Python (`execute_tools_and_side_effects`): with a non-plain-text output schema the
            // message must validate against it; otherwise the run is a model behavior error.
            let final_output = match current_agent.output_type.as_deref() {
                Some(schema) if !schema.is_plain_text() => {
                    if final_text.is_empty() {
                        return Err(ModelError::Behavior(
                            "Model returned no final output for the structured output type.".into(),
                        )
                        .into());
                    }
                    schema.validate_json(&final_text)?
                }
                _ => Value::String(final_text),
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
                current_input_items.push(item.clone());
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
            current_agent = (*h.agent).clone();

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
                    )
                    .await;
                }
            }
            ToolUseBehavior::RunLlmAgain => {}
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
    max_turns: usize,
    usage: Usage,
    context: &RunContextWrapper,
    hooks: Option<&Arc<dyn RunHooks>>,
    extra_output_guardrails: &[OutputGuardrail],
    input_guardrail_results: Vec<InputGuardrailResult>,
) -> Result<RunResult, AgentsError> {
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
        if let Some(result) = tripped {
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

/// Shape a tool result into the run's final output.
///
/// Python (`_maybe_finalize_from_tool_results`) coerces the tool output to `str` unless the
/// agent declared a non-plain-text `output_type`.
fn finalize_tool_output(agent: &Agent, output: &Value) -> Value {
    match agent.output_type.as_deref() {
        Some(schema) if !schema.is_plain_text() => output.clone(),
        _ => Value::String(value_to_tool_string(output)),
    }
}

/// Append (and stream) a tool output item unless one already exists for the call id.
async fn push_tool_output(
    events: &Option<EventTx>,
    generated_items: &mut Vec<RunItem>,
    agent_name: &str,
    call_id: &str,
    output: &Value,
) {
    if has_tool_output(generated_items, call_id) {
        return;
    }
    let out_item = tool_output_item(agent_name, call_id, output);
    emit(
        events,
        StreamEvent::RunItem {
            name: RunItemStreamName::ToolOutput,
            item: out_item.clone(),
        },
    )
    .await;
    generated_items.push(out_item);
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

fn build_run_error_data(
    input: &InputLike,
    items: &[RunItem],
    raw_responses: &[ModelResponse],
    agent: &Agent,
) -> RunErrorData {
    let output: Vec<Value> = items
        .iter()
        .filter(|i| i.is_model_input())
        .map(|i| i.raw_item().clone())
        .collect();
    let mut history = ItemHelpers::input_to_new_input_list(input);
    history.extend(output.iter().cloned());
    RunErrorData {
        input: input.clone(),
        new_items: items.to_vec(),
        history,
        output,
        raw_responses: raw_responses.to_vec(),
        last_agent: Arc::new(agent.clone()),
    }
}

/// Python (`validate_handler_final_output` + `format_final_output_text`): a structured agent's
/// handler output must validate against its schema. Returns the output and its message text.
fn validate_handler_final_output(
    agent: &Agent,
    output: Value,
) -> Result<(Value, String), AgentsError> {
    let invalid = || UserError::new("Invalid run error handler final_output for structured output.");
    let Some(schema) = agent.output_type.as_deref().filter(|s| !s.is_plain_text()) else {
        let text = value_to_tool_string(&output);
        return Ok((output, text));
    };
    // Python wraps non-object outputs under `response`; try the value as given, then wrapped.
    let wrapped = json!({ crate::agent_output::WRAPPER_DICT_KEY: output.clone() });
    for payload in [&output, &wrapped] {
        let text = serde_json::to_string(payload).map_err(|_| invalid())?;
        if let Ok(validated) = schema.validate_json(&text) {
            return Ok((validated, text));
        }
    }
    Err(invalid().into())
}

/// Run `RunConfig.tool_error_formatter`; `None` keeps `default_message`.
async fn format_tool_error(
    run_config: &RunConfig,
    context: &RunContextWrapper,
    kind: ToolErrorKind,
    tool_name: &str,
    call_id: &str,
    default_message: String,
) -> String {
    let Some(formatter) = &run_config.tool_error_formatter else {
        return default_message;
    };
    formatter(ToolErrorFormatterArgs {
        kind,
        tool_name: tool_name.to_string(),
        call_id: call_id.to_string(),
        default_message: default_message.clone(),
        run_context: context.clone(),
    })
    .await
    .unwrap_or(default_message)
}

async fn plan_tool_calls(
    agent: &Agent,
    tools: &[FunctionTool],
    function_calls: &[ResponseOutputItem],
    approvals: &ApprovalStore,
    generated_items: &[RunItem],
    run_config: &RunConfig,
    context: &RunContextWrapper,
) -> Result<ToolPlan, AgentsError> {
    let mut plan = ToolPlan {
        interruptions: Vec::new(),
        ready_outputs: Vec::new(),
        to_invoke: Vec::new(),
        already_done: Vec::new(),
    };

    for call in function_calls {
        let (name, arguments, call_id) = required_function_call_parts(call)?;
        let Some(tool) = tools.iter().find(|t| t.name == name).cloned() else {
            // Python: `ModelBehaviorError("Tool X not found in agent Y")`, unless the run asks
            // for the error to be returned to the model.
            if run_config.tool_not_found_behavior != ToolNotFoundBehavior::ReturnErrorToModel {
                return Err(ModelError::Behavior(format!(
                    "Tool {name} not found in agent {}",
                    agent.name
                ))
                .into());
            }
            // The placeholder keeps the tool name out of `StopAtTools` matching.
            let placeholder = FunctionTool::constant(TOOL_NOT_FOUND_PLACEHOLDER, "", "");
            let message = format_tool_error(
                run_config,
                context,
                ToolErrorKind::ToolNotFound,
                &name,
                &call_id,
                format!("Tool '{name}' not found."),
            )
            .await;
            plan.ready_outputs
                .push((placeholder, Value::String(message), call_id));
            continue;
        };

        if let Some(existing) = existing_tool_output(generated_items, &call_id) {
            plan.already_done.push((tool, existing, call_id));
            continue;
        }

        match approvals.status(&agent.name, &name, &call_id) {
            Some(ApprovalDecision::Approved) => {
                plan.to_invoke.push((tool, arguments, call_id));
            }
            Some(ApprovalDecision::Rejected { message }) => {
                // Python: an explicit rejection message wins; the formatter only replaces the
                // default one. The default is stored at reject time, so an explicit message
                // equal to it is treated as the default.
                let message = if message == DEFAULT_APPROVAL_REJECTION_MESSAGE {
                    format_tool_error(
                        run_config,
                        context,
                        ToolErrorKind::ApprovalRejected,
                        &name,
                        &call_id,
                        message,
                    )
                    .await
                } else {
                    message
                };
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
    agent: &Agent,
    context: &RunContextWrapper,
    hooks: Option<&Arc<dyn RunHooks>>,
    max_concurrency: Option<usize>,
    guardrail_log: &SharedToolGuardrailLog,
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
    let hooks = hooks.cloned();
    let agent_hooks = agent.hooks.clone();
    let invoke_futs: Vec<_> = plan.to_invoke.iter().map(|(tool, arguments, call_id)| {
        let tool = tool.clone();
        let arguments = arguments.clone();
        let call_id = call_id.clone();
        let resume_map = resume_map.clone();
        let agent = agent.clone();
        let guardrail_log = Arc::clone(guardrail_log);
        let context = context.clone();
        let hooks = hooks.clone();
        let agent_hooks = agent_hooks.clone();
        async move {
            let _fs = function_span(&tool.name);
            let ctx = ToolContext::new(
                tool.name.clone(),
                call_id.clone(),
                arguments.clone(),
                context.clone(),
            );
            let hook_ctx = ctx.clone();
            // Python: input guardrails run before the start hooks; a rejection replaces the
            // call, so neither the hooks, the body nor the output guardrails run.
            let guardrail_agent = Arc::new(agent.clone());
            let mut input_results = Vec::new();
            let rejection = run_tool_input_guardrails(
                &tool.tool_input_guardrails,
                &ctx,
                &guardrail_agent,
                &mut input_results,
            )
            .await;
            guardrail_log
                .lock()
                .expect("tool guardrail log")
                .input
                .extend(input_results);
            if let Some(message) = rejection? {
                return Ok::<_, AgentsError>((
                    tool,
                    call_id,
                    ToolResult::output(Value::String(message)),
                ));
            }
            if let Some(h) = &hooks {
                h.on_tool_start(hook_ctx.clone(), &agent, &tool).await;
            }
            if let Some(ah) = &agent_hooks {
                ah.on_tool_start(hook_ctx.clone(), &agent, &tool).await;
            }
            // Python (`failure_error_function`): a failing tool is reported to the model so it
            // can retry, instead of aborting the run.
            let call = NESTED_RESUME_STATES.scope(RefCell::new(resume_map), async {
                (tool.on_invoke_tool)(ctx, arguments).await
            });
            // Python applies the timeout outside the failure handler, so `RaiseException`
            // fails the run instead of being reported to the model.
            let outcome = match tool.timeout_seconds {
                None => call.await,
                Some(seconds) => {
                    match tokio::time::timeout(Duration::from_secs_f64(seconds), call).await {
                        Ok(outcome) => outcome,
                        Err(_) => {
                            let timeout = ToolTimeoutError {
                                tool_name: tool.name.clone(),
                                timeout_seconds: seconds,
                            };
                            if tool.timeout_behavior == ToolTimeoutBehavior::RaiseException {
                                return Err(timeout.into());
                            }
                            let message = match &tool.timeout_error_function {
                                Some(format) => format(&context, &AgentsError::from(timeout)),
                                None => default_tool_timeout_error_message(&tool.name, seconds),
                            };
                            Ok(ToolResult::output(Value::String(message)))
                        }
                    }
                }
            };
            // Python (`failure_error_function`): a failing tool is reported to the model so it
            // can retry, instead of aborting the run.
            let result = match outcome {
                Ok(result) => result,
                Err(error) => {
                    let message = tool.failure_error_function.handle(&context, error)?;
                    ToolResult::output(Value::String(message))
                }
            };
            let mut result = result;
            // Python: output guardrails see the result (including an error message) unless the
            // call paused for a nested approval.
            if result.interruptions.is_empty() {
                let mut output_results = Vec::new();
                let guarded = run_tool_output_guardrails(
                    &tool.tool_output_guardrails,
                    &hook_ctx,
                    &guardrail_agent,
                    result.output.clone().unwrap_or(Value::Null),
                    &mut output_results,
                )
                .await;
                guardrail_log
                    .lock()
                    .expect("tool guardrail log")
                    .output
                    .extend(output_results);
                result.output = Some(guarded?);
            }
            let output = result.output.clone().unwrap_or(Value::Null);
            if let Some(h) = &hooks {
                h.on_tool_end(hook_ctx.clone(), &agent, &tool, &output)
                    .await;
            }
            if let Some(ah) = &agent_hooks {
                ah.on_tool_end(hook_ctx, &agent, &tool, &output).await;
            }
            Ok::<_, AgentsError>((tool, call_id, result))
        }
    })
    .collect();
    // Python starts every call of the turn unless `max_function_tool_concurrency` is set; a slot
    // frees as soon as a call finishes, and the first failure cancels the rest.
    // The window is driven by hand over concrete futures: boxing them would need a `Send` proof
    // that the recursive agent-as-tool call chain makes impossible to infer.
    let limit = max_concurrency.unwrap_or(usize::MAX).max(1);
    let mut waiting = invoke_futs.into_iter().enumerate();
    let mut running = futures::stream::FuturesUnordered::new();
    let mut invoked = Vec::new();
    loop {
        while running.len() < limit {
            match waiting.next() {
                Some((index, fut)) => running.push(async move { (index, fut.await) }),
                None => break,
            }
        }
        match running.next().await {
            Some((index, Ok(done))) => invoked.push((index, done)),
            Some((_, Err(error))) => return Err(error),
            None => break,
        }
    }
    invoked.sort_by_key(|(index, _)| *index);
    for (_, (tool, call_id, result)) in invoked {
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
    last_agent: Arc<Agent>,
    max_turns: usize,
    usage: Usage,
) -> RunResult {
    RunResult {
        input,
        new_items,
        raw_responses,
        final_output,
        last_agent_name: last_agent.name.clone(),
        last_agent,
        max_turns: Some(max_turns),
        usage,
        interruptions: Vec::new(),
        input_guardrail_results: Vec::new(),
        output_guardrail_results: Vec::new(),
        tool_input_guardrail_results: Vec::new(),
        tool_output_guardrail_results: Vec::new(),
        interrupt_state: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn interrupted_result(
    input: InputLike,
    new_items: Vec<RunItem>,
    raw_responses: Vec<ModelResponse>,
    last_agent: Arc<Agent>,
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
        last_agent_name: last_agent.name.clone(),
        last_agent,
        max_turns: Some(max_turns),
        usage,
        interruptions,
        input_guardrail_results: Vec::new(),
        output_guardrail_results: Vec::new(),
        tool_input_guardrail_results: Vec::new(),
        tool_output_guardrail_results: Vec::new(),
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
        .map_err(|e| AgentsError::tool_with_source("invalid agent tool args", e))?;
    if let Some(s) = v.get("input").and_then(|x| x.as_str()) {
        return Ok(s.to_string());
    }
    if let Some(s) = v.as_str() {
        return Ok(s.to_string());
    }
    Err(AgentsError::tool(
        "agent tool requires string field `input`",
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
