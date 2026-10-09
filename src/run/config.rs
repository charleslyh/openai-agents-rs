//! Run configuration: `RunConfig`, `RunOptions` and the settings hanging off them
//! (Python: `run_config.py`).

use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::agent::Agent;
use crate::error::AgentsError;
use crate::guardrail::{InputGuardrail, OutputGuardrail};
use crate::handoffs::{HandoffHistoryMapper, HandoffInputFilter};
use crate::items::ReasoningItemIdPolicy;
use crate::lifecycle::RunHooks;
use crate::memory::{Session, SessionInputCallback, SessionSettings};
use crate::model::{ModelProvider, ModelRef};
use crate::model_settings::ModelSettings;
use crate::run_context::{ContextValue, RunContextWrapper};
use crate::tracing::TracingConfig;

use super::errors::RunErrorHandlers;

/// Default max turns (Python: `DEFAULT_MAX_TURNS = 10`).
pub const DEFAULT_MAX_TURNS: usize = 10;

/// Which HTTP API the OpenAI-protocol models use by default (Python: `set_default_openai_api`).
///
/// Unlike Python, whose default is Responses, the default here is Chat Completions: this SDK
/// targets any server that speaks the OpenAI protocol, and Chat Completions is the one nearly
/// all of them implement (D-I).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DefaultOpenAiApi {
    /// Responses API.
    Responses,
    /// Chat Completions API (default).
    #[default]
    ChatCompletions,
}

static DEFAULT_OPENAI_API: Mutex<Option<DefaultOpenAiApi>> = Mutex::new(None);

/// Set the default OpenAI API (Python: `set_default_openai_api`).
///
/// An explicit choice applies to every provider that has not been given its own (see D-I).
pub fn set_default_openai_api(api: DefaultOpenAiApi) {
    *DEFAULT_OPENAI_API.lock().expect("api lock") = Some(api);
}

/// Get the default API: the explicit choice, else Chat Completions.
pub fn get_default_openai_api() -> DefaultOpenAiApi {
    explicit_default_openai_api().unwrap_or_default()
}

/// The API chosen with [`set_default_openai_api`], if any.
pub(crate) fn explicit_default_openai_api() -> Option<DefaultOpenAiApi> {
    *DEFAULT_OPENAI_API.lock().expect("api lock")
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

/// Default text that replaces a tool output withheld by an output guardrail
/// (Python: `OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT`).
pub const OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT: &str = "Output withheld by an output guardrail.";

/// Data passed to an output guardrail blocked-message formatter
/// (Python: `OutputGuardrailBlockedMessageArgs`).
#[derive(Debug, Clone)]
pub struct OutputGuardrailBlockedMessageArgs {
    /// The SDK default placeholder.
    pub default_message: String,
    /// Name of the output guardrail that tripped.
    pub guardrail_name: String,
    /// The agent whose final output was rejected.
    pub agent: Arc<Agent>,
    /// The run context.
    pub run_context: RunContextWrapper,
}

/// Placeholder for a tool output rejected by an output guardrail
/// (Python: `RunConfig.output_guardrail_blocked_message`).
#[derive(Clone)]
pub enum OutputGuardrailBlockedMessage {
    /// A fixed, non-empty message.
    Text(String),
    /// A synchronous formatter; `None`, an empty string or a panic falls back to the default.
    Formatter(Arc<dyn Fn(OutputGuardrailBlockedMessageArgs) -> Option<String> + Send + Sync>),
}

impl std::fmt::Debug for OutputGuardrailBlockedMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text(text) => f.debug_tuple("Text").field(text).finish(),
            Self::Formatter(_) => f.write_str("Formatter(..)"),
        }
    }
}

impl OutputGuardrailBlockedMessage {
    /// Build a formatter variant from a closure.
    pub fn formatter<F>(f: F) -> Self
    where
        F: Fn(OutputGuardrailBlockedMessageArgs) -> Option<String> + Send + Sync + 'static,
    {
        Self::Formatter(Arc::new(f))
    }
}

/// Resolve the placeholder for a tripped tool-output guardrail
/// (Python: `_resolve_output_guardrail_blocked_message`). Never fails: any problem with the
/// configured message yields the default.
pub(crate) fn resolve_blocked_message(
    run_config: &RunConfig,
    guardrail_name: &str,
    agent: &Agent,
    context: &RunContextWrapper,
) -> String {
    let default = OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT.to_string();
    let resolved = match &run_config.output_guardrail_blocked_message {
        None => return default,
        Some(OutputGuardrailBlockedMessage::Text(text)) => Some(text.clone()),
        Some(OutputGuardrailBlockedMessage::Formatter(format)) => {
            let args = OutputGuardrailBlockedMessageArgs {
                default_message: default.clone(),
                guardrail_name: guardrail_name.to_string(),
                agent: Arc::new(agent.clone()),
                run_context: context.clone(),
            };
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| format(args)))
                .ok()
                .flatten()
        }
    };
    resolved.filter(|text| !text.is_empty()).unwrap_or(default)
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
    /// Fail with a `UserError` before the model is called.
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
///
/// Build one with `RunConfig { field, ..Default::default() }`; the `..Default::default()` tail is
/// what keeps a caller compiling when a field is added, so always include it.
#[derive(Clone)]
pub struct RunConfig {
    /// Override model for the whole run: an instance, or a name resolved by
    /// [`model_provider`](Self::model_provider) (Python: `RunConfig.model`).
    pub model: Option<ModelRef>,
    /// Provider used to resolve string model names (Python: `RunConfig.model_provider`).
    ///
    /// When unset, names go through [`crate::model::default_model_provider`].
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
    /// Tracing settings for this run (Python: `RunConfig.tracing`).
    pub tracing: Option<TracingConfig>,
    /// Placeholder shown instead of a tool output that an output guardrail rejected
    /// (Python: `RunConfig.output_guardrail_blocked_message`).
    pub output_guardrail_blocked_message: Option<OutputGuardrailBlockedMessage>,
    /// Whether reasoning item ids are kept in the input the runner builds
    /// (Python: `RunConfig.reasoning_item_id_policy`; `None` preserves them).
    pub reasoning_item_id_policy: Option<ReasoningItemIdPolicy>,
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
            reasoning_item_id_policy: None,
            tracing: None,
            output_guardrail_blocked_message: None,
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

/// Per-call options for [`crate::run::Runner::run`] / [`crate::run::Runner::run_streamed`].
///
/// [`RunOptions::default()`] supplies the default turn limit and workflow name, so prefer
/// `RunOptions { field, ..Default::default() }` over a full literal.
#[derive(Clone)]
pub struct RunOptions {
    /// Max turns before [`crate::error::MaxTurnsExceeded`].
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
    /// pause, pass the same session to [`crate::run::Runner::run_state`] so the resumed run is saved.
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
