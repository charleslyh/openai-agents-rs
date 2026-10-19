//! Run configuration: `RunConfig`, `RunOptions` and the settings hanging off them
//! (Python: `run_config.py`).

use std::sync::Arc;

use crate::agent::Agent;
use crate::extensions::Extensions;
use crate::guardrail::{InputGuardrail, OutputGuardrail};
use crate::lifecycle::RunHooks;
use crate::memory::Session;
use crate::run_context::{ContextValue, RunContextWrapper};

use super::errors::RunErrorHandlers;
use super::options::{HandoffOptions, ModelOptions, SessionOptions, ToolOptions, TracingOptions};

/// Default max turns (Python: `DEFAULT_MAX_TURNS = 10`).
pub const DEFAULT_MAX_TURNS: usize = 10;

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
    let resolved = match &run_config.tools.guardrail_blocked_message {
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

/// Data passed to [`ToolOptions::error_formatter`](super::ToolOptions::error_formatter) (Python: `ToolErrorFormatterArgs`).
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

/// Run configuration (Python: `RunConfig`), grouped by topic.
///
/// Start from `RunConfig::default()` and set what you need, either through the chain methods
/// (`RunConfig::default().with_tracing(TracingOptions::new().workflow_name("support"))`) or by
/// assigning fields (`config.model.settings = Some(settings)`). The struct is
/// `#[non_exhaustive]`, so a literal with `..Default::default()` only works inside this crate.
#[derive(Clone, Default, Debug)]
#[non_exhaustive]
pub struct RunConfig {
    /// Model selection, settings and input filtering.
    pub model: ModelOptions,
    /// Trace naming, ids and sensitive-data policy.
    pub tracing: TracingOptions,
    /// How session history joins the run.
    pub session: SessionOptions,
    /// What a handoff passes on.
    pub handoff: HandoffOptions,
    /// How tool calls are executed and reported.
    pub tools: ToolOptions,
    /// Input guardrails applied to the run (Python: `RunConfig.input_guardrails`).
    pub input_guardrails: Vec<InputGuardrail>,
    /// Output guardrails applied to the run (Python: `RunConfig.output_guardrails`).
    pub output_guardrails: Vec<OutputGuardrail>,
}

impl RunConfig {
    /// Replace the model group.
    #[must_use]
    pub fn with_model(mut self, model: ModelOptions) -> Self {
        self.model = model;
        self
    }

    /// Replace the tracing group.
    #[must_use]
    pub fn with_tracing(mut self, tracing: TracingOptions) -> Self {
        self.tracing = tracing;
        self
    }

    /// Replace the session group.
    #[must_use]
    pub fn with_session(mut self, session: SessionOptions) -> Self {
        self.session = session;
        self
    }

    /// Replace the handoff group.
    #[must_use]
    pub fn with_handoff(mut self, handoff: HandoffOptions) -> Self {
        self.handoff = handoff;
        self
    }

    /// Replace the tool group.
    #[must_use]
    pub fn with_tools(mut self, tools: ToolOptions) -> Self {
        self.tools = tools;
        self
    }

    /// Add an input guardrail for the whole run.
    #[must_use]
    pub fn input_guardrail(mut self, guardrail: InputGuardrail) -> Self {
        self.input_guardrails.push(guardrail);
        self
    }

    /// Add an output guardrail for the whole run.
    #[must_use]
    pub fn output_guardrail(mut self, guardrail: OutputGuardrail) -> Self {
        self.output_guardrails.push(guardrail);
        self
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
    /// Typed business data, one value per type (Rust-only; see [`Extensions`]).
    ///
    /// [`RunOptions::context`] is merged in as one more slot when the run starts.
    pub extensions: Extensions,
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
            run_config: RunConfig::default()
                .with_tracing(TracingOptions::new().workflow_name("Agent workflow")),
            previous_response_id: None,
            conversation_id: None,
            context: None,
            extensions: Extensions::new(),
            hooks: None,
            session: None,
            error_handlers: RunErrorHandlers::default(),
        }
    }
}

impl RunOptions {
    /// Options with the defaults of [`RunOptions::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a typed value that tools, guardrails and hooks read with `ctx.get::<T>()`.
    ///
    /// A second value of the same type replaces the first.
    #[must_use]
    pub fn with<T: std::any::Any + Send + Sync>(mut self, value: T) -> Self {
        self.extensions.insert(value);
        self
    }

    /// Set the single user context value (Python: `context=`); it is stored under its own type.
    #[must_use]
    pub fn context<T: std::any::Any + Send + Sync>(mut self, value: T) -> Self {
        self.context = Some(Arc::new(value));
        self
    }

    /// The run context handed to tools and hooks: the extensions plus [`RunOptions::context`].
    pub(crate) fn run_context(&self) -> RunContextWrapper {
        let mut extensions = self.extensions.clone();
        if let Some(value) = &self.context {
            extensions.insert_shared(Arc::clone(value));
        }
        RunContextWrapper::from_extensions(extensions)
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
            .field("extensions", &self.extensions)
            .field("has_hooks", &self.hooks.is_some())
            .field("has_session", &self.session.is_some())
            .finish()
    }
}
