//! Run configuration: `RunConfig`, `RunOptions` and the settings hanging off them
//! (Python: `run_config.py`).

use std::sync::Arc;

use crate::agent::Agent;
use crate::extensions::Extensions;
use crate::guardrail::{InputGuardrail, OutputGuardrail};
use crate::items::ToolInterruptionItem;
use crate::lifecycle::RunHooks;
use crate::memory::Session;
use crate::model::ModelRef;
use crate::model_settings::ModelSettings;
use crate::run_context::{ContextValue, RunContextWrapper};
use crate::run_state::ToolResolution;

use super::errors::RunErrorHandlers;
use super::hitl::ToolResolver;
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
    agent: &Arc<Agent>,
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
                agent: Arc::clone(agent),
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

impl ToolExecutionConfig {
    /// Defaults: every tool of a turn runs at once.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run at most `max` function tools of one turn at once.
    #[must_use]
    pub fn concurrency(mut self, max: usize) -> Self {
        self.max_function_tool_concurrency = Some(max);
        self
    }
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
/// Start from [`RunOptions::new`] and chain what the call needs:
///
/// ```ignore
/// let options = RunOptions::new().max_turns(5).temperature(0.2).session(session);
/// ```
///
/// [`RunOptions::default()`] supplies the default turn limit and workflow name. The struct is
/// `#[non_exhaustive]`, so a literal does not compile outside this crate; assign fields or chain.
#[derive(Clone)]
#[non_exhaustive]
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
    /// Resolves paused tool calls in-process instead of returning them (Rust-only, X-009); set
    /// with [`RunOptions::resolve_with`].
    pub resolver: Option<ToolResolver>,
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
            resolver: None,
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

    /// Resolve paused tool calls with `resolve` instead of returning them (Rust-only, X-009).
    ///
    /// When a tool call needs resolution (human approval, or work done elsewhere), the run asks
    /// `resolve` about each paused call and resumes, until it finishes. Without it the run returns
    /// early with [`RunResult::interruptions`](crate::result::RunResult::interruptions). Streamed
    /// runs fail with a `UserError` when a resolver is set.
    ///
    /// ```
    /// use openai_agents_core::{RunOptions, ToolResolution};
    ///
    /// let options = RunOptions::new().resolve_with(|call| async move {
    ///     if call.tool_name == "delete_file" {
    ///         ToolResolution::reject("Deleting files is not allowed.")
    ///     } else {
    ///         ToolResolution::Run
    ///     }
    /// });
    /// # let _ = options;
    /// ```
    #[must_use]
    pub fn resolve_with<F, Fut>(mut self, resolve: F) -> Self
    where
        F: Fn(ToolInterruptionItem) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ToolResolution> + Send + 'static,
    {
        self.resolver = Some(ToolResolver::new(resolve));
        self
    }

    /// Stop with [`crate::error::MaxTurnsExceeded`] after `turns` turns.
    #[must_use]
    pub fn max_turns(mut self, turns: usize) -> Self {
        self.max_turns = Some(turns);
        self
    }

    /// Remove the turn limit (Python: `max_turns=None`).
    #[must_use]
    pub fn unlimited_turns(mut self) -> Self {
        self.max_turns = None;
        self
    }

    /// Replace the whole [`RunConfig`].
    #[must_use]
    pub fn config(mut self, run_config: RunConfig) -> Self {
        self.run_config = run_config;
        self
    }

    /// Run `model` (an instance or a name) instead of each agent's own.
    #[must_use]
    pub fn model(mut self, model: impl Into<ModelRef>) -> Self {
        self.run_config.model.model_ref = Some(model.into());
        self
    }

    /// Merge `settings` over the agent's model settings.
    #[must_use]
    pub fn model_settings(mut self, settings: ModelSettings) -> Self {
        self.run_config.model.settings = Some(settings);
        self
    }

    /// Edit the run-level model settings in place, starting from the current ones.
    ///
    /// `options.tune(|s| s.temperature = Some(0.2))` is the one-line way to change one knob.
    #[must_use]
    pub fn tune(mut self, edit: impl FnOnce(&mut ModelSettings)) -> Self {
        edit(
            self.run_config
                .model
                .settings
                .get_or_insert_with(Default::default),
        );
        self
    }

    /// Sampling temperature for every model call of the run.
    #[must_use]
    pub fn temperature(self, temperature: f32) -> Self {
        self.tune(|s| s.temperature = Some(temperature))
    }

    /// Output token limit for every model call of the run.
    #[must_use]
    pub fn max_tokens(self, max_tokens: u32) -> Self {
        self.tune(|s| s.max_tokens = Some(max_tokens))
    }

    /// Reasoning for every model call of the run: `.reasoning(ReasoningEffort::High)` or a full
    /// [`Reasoning`](crate::model_settings::Reasoning) with a summary.
    #[must_use]
    pub fn reasoning(self, reasoning: impl Into<crate::model_settings::Reasoning>) -> Self {
        let reasoning = reasoning.into();
        self.tune(|s| s.reasoning = Some(reasoning))
    }

    /// Name the root trace.
    #[must_use]
    pub fn workflow_name(mut self, name: impl Into<String>) -> Self {
        self.run_config.tracing.workflow_name = Some(name.into());
        self
    }

    /// Edit the tracing group: workflow name, ids, metadata, processors, sensitive data.
    #[must_use]
    pub fn tracing(mut self, edit: impl FnOnce(&mut TracingOptions)) -> Self {
        edit(&mut self.run_config.tracing);
        self
    }

    /// Edit the tool group: unknown tools, concurrency, collisions, error texts.
    #[must_use]
    pub fn tools(mut self, edit: impl FnOnce(&mut ToolOptions)) -> Self {
        edit(&mut self.run_config.tools);
        self
    }

    /// Edit the session group: stored-history limit and the input callback.
    #[must_use]
    pub fn session_opts(mut self, edit: impl FnOnce(&mut SessionOptions)) -> Self {
        edit(&mut self.run_config.session);
        self
    }

    /// Conversation memory for the run.
    #[must_use]
    pub fn session(mut self, session: Arc<dyn Session>) -> Self {
        self.session = Some(session);
        self
    }

    /// Run-level lifecycle hooks.
    #[must_use]
    pub fn hooks(mut self, hooks: Arc<dyn RunHooks>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Chain onto a previous Responses API response.
    #[must_use]
    pub fn previous_response_id(mut self, id: impl Into<String>) -> Self {
        self.previous_response_id = Some(id.into());
        self
    }

    /// Continue a server-side conversation.
    #[must_use]
    pub fn conversation_id(mut self, id: impl Into<String>) -> Self {
        self.conversation_id = Some(id.into());
        self
    }

    /// Turn run errors into outputs with `handlers`.
    #[must_use]
    pub fn error_handlers(mut self, handlers: RunErrorHandlers) -> Self {
        self.error_handlers = handlers;
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
            .field("has_resolver", &self.resolver.is_some())
            .finish()
    }
}
