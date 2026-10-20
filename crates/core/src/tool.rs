//! Function tools (Python: `agents.tool.FunctionTool` supported subset).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::AgentsError;

pub use crate::error::{default_tool_timeout_error_message, DEFAULT_APPROVAL_REJECTION_MESSAGE};
use crate::extensions::TypeRequirement;
use crate::items::ToolInterruptionItem;
use crate::run_context::RunContextWrapper;

pub use crate::run_context::ToolContext;
use crate::run_state::RunState;
use crate::tool_guardrails::{ToolInputGuardrail, ToolOutputGuardrail};

/// Text returned to the model when a tool fails and no custom handler is set
/// (Python: `default_tool_error_function`).
pub const DEFAULT_TOOL_ERROR_MESSAGE: &str =
    "An error occurred while running the tool. Please try again.";

/// Formats a tool failure as the text sent back to the model
/// (Python: `failure_error_function`).
pub type ToolErrorFn = Arc<dyn Fn(&RunContextWrapper, &AgentsError) -> String + Send + Sync>;

/// What happens when a function tool returns an error (Python: `failure_error_function`).
#[derive(Clone, Default)]
pub enum ToolFailureHandling {
    /// Send [`DEFAULT_TOOL_ERROR_MESSAGE`] to the model; the exception is not exposed.
    #[default]
    Default,
    /// Send the formatter's text to the model.
    Custom(ToolErrorFn),
    /// Abort the run with the error (Python: `failure_error_function=None`).
    Raise,
}

impl ToolFailureHandling {
    /// Resolve the failure into model-visible text, or hand the error back to abort the run.
    pub fn handle(
        &self,
        context: &RunContextWrapper,
        error: AgentsError,
    ) -> Result<String, AgentsError> {
        match self {
            Self::Default => Ok(DEFAULT_TOOL_ERROR_MESSAGE.to_string()),
            Self::Custom(f) => Ok(f(context, &error)),
            Self::Raise => Err(error),
        }
    }
}

/// What happens when a tool exceeds its timeout (Python: `ToolTimeoutBehavior`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolTimeoutBehavior {
    /// Return a model-visible timeout message and keep running (default).
    #[default]
    ErrorAsResult,
    /// Fail the run with [`crate::ToolTimeoutError`].
    RaiseException,
}

/// Result of invoking a tool (Python: `FunctionToolResult` subset).
///
/// `interruptions` and `nested_state` are internal scheduling concepts (how the run loop pauses
/// for a human decision) and are not part of the public shape: build a value with
/// [`ToolOutput::output`] or [`ToolOutput::interrupted`], not a struct literal.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolOutput {
    /// Tool output when the call completed (absent when interrupted).
    pub(crate) output: Option<Value>,
    /// Nested / bubbled interruptions waiting for a resolution.
    pub(crate) interruptions: Vec<ToolInterruptionItem>,
    /// Nested agent run state when `Agent.as_tool` paused for HITL.
    pub(crate) nested_state: Option<Box<RunState>>,
}

impl ToolOutput {
    /// Successful tool output.
    pub fn output(value: Value) -> Self {
        Self {
            output: Some(value),
            interruptions: Vec::new(),
            nested_state: None,
        }
    }

    /// Paused for nested interruptions.
    pub fn interrupted(interruptions: Vec<ToolInterruptionItem>, nested_state: RunState) -> Self {
        Self {
            output: None,
            interruptions,
            nested_state: Some(Box::new(nested_state)),
        }
    }

    /// Whether the call paused instead of completing.
    pub fn is_interrupted(&self) -> bool {
        !self.interruptions.is_empty()
    }

    /// The completed output, if the call did not pause.
    pub fn value(&self) -> Option<&Value> {
        self.output.as_ref()
    }
}

/// Async tool invoker signature.
pub type ToolInvoker = Arc<
    dyn Fn(
            ToolContext,
            String,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, AgentsError>> + Send>>
        + Send
        + Sync,
>;

/// Per-call pause policy (Python: `needs_approval` callable).
///
/// Arguments: parsed JSON params object, tool call id → whether the call pauses.
pub type NeedsResolutionFn =
    Arc<dyn Fn(Value, String) -> Pin<Box<dyn Future<Output = bool> + Send>> + Send + Sync>;

/// When a function tool call pauses before it runs, so the host can resolve it
/// (Python: `FunctionTool.needs_approval`).
///
/// The run stops with the call in `interruptions`; the host resolves it on the
/// [`RunState`] — approve or reject it (human approval), or hand back a result it produced
/// elsewhere ([`RunState::provide_output`]) — and resumes the run.
#[derive(Clone)]
pub enum NeedsResolution {
    /// Fixed always / never.
    Fixed(bool),
    /// Per-call policy.
    Dynamic(NeedsResolutionFn),
}

impl Default for NeedsResolution {
    fn default() -> Self {
        Self::Fixed(false)
    }
}

impl From<bool> for NeedsResolution {
    fn from(value: bool) -> Self {
        Self::Fixed(value)
    }
}

impl NeedsResolution {
    /// Decide per call from the arguments the model sent and the call id.
    ///
    /// The common case is a threshold: pause when the call is large enough to matter. `params` is
    /// the parsed argument object (empty when the arguments are not a JSON object).
    pub fn dynamic<F, Fut>(f: F) -> Self
    where
        F: Fn(Value, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        Self::Dynamic(Arc::new(move |params, call_id| {
            Box::pin(f(params, call_id))
        }))
    }

    /// Evaluate whether this call pauses for resolution.
    pub async fn requires(&self, arguments_json: &str, call_id: &str) -> bool {
        match self {
            Self::Fixed(v) => *v,
            Self::Dynamic(f) => {
                let params = serde_json::from_str::<Value>(arguments_json)
                    .unwrap_or(Value::Object(Default::default()));
                f(params, call_id.to_string()).await
            }
        }
    }
}

/// Dynamic enablement policy: `(context, agent) -> enabled`.
pub type IsEnabledFn = Arc<
    dyn Fn(
            RunContextWrapper,
            Arc<crate::agent::Agent>,
        ) -> Pin<Box<dyn Future<Output = bool> + Send>>
        + Send
        + Sync,
>;

/// Whether a tool or handoff is offered to the model (Python: `is_enabled`).
#[derive(Clone)]
pub enum ToolEnabled {
    /// Always (or never) enabled.
    Fixed(bool),
    /// Decided per turn from the run context and the current agent.
    Dynamic(IsEnabledFn),
}

impl Default for ToolEnabled {
    fn default() -> Self {
        Self::Fixed(true)
    }
}

impl From<bool> for ToolEnabled {
    fn from(value: bool) -> Self {
        Self::Fixed(value)
    }
}

impl ToolEnabled {
    /// Build a dynamic policy from an async closure.
    pub fn dynamic<F, Fut>(f: F) -> Self
    where
        F: Fn(RunContextWrapper, Arc<crate::agent::Agent>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        Self::Dynamic(Arc::new(move |ctx, agent| Box::pin(f(ctx, agent))))
    }

    /// Evaluate the policy. The agent is shared with the callback, not copied.
    pub async fn resolve(
        &self,
        context: &RunContextWrapper,
        agent: &Arc<crate::agent::Agent>,
    ) -> bool {
        match self {
            Self::Fixed(v) => *v,
            Self::Dynamic(f) => f(context.clone(), Arc::clone(agent)).await,
        }
    }
}

impl std::fmt::Debug for ToolEnabled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed(v) => write!(f, "Fixed({v})"),
            Self::Dynamic(_) => f.write_str("Dynamic(..)"),
        }
    }
}

/// Name, description and parameter schema a [`Tool`] shows the model.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolSpec {
    /// Tool name shown to the LLM.
    pub name: String,
    /// Tool description shown to the LLM.
    pub description: String,
    /// JSON Schema for parameters.
    pub params_json_schema: Value,
    /// Whether the schema is strict.
    pub strict_json_schema: bool,
}

impl ToolSpec {
    /// A spec with an explicit JSON Schema. The schema is strict, like the one `typed` builds.
    ///
    /// The struct is `#[non_exhaustive]`, so this and [`ToolSpec::typed`] are how a crate outside
    /// the SDK builds one.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        params_json_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            params_json_schema,
            strict_json_schema: true,
        }
    }

    /// A spec whose schema comes from `Args`, the same one [`FunctionTool::typed`] advertises.
    ///
    /// Pair it with `serde_json::from_str::<Args>(&ctx.tool_arguments)` in [`Tool::call`], so the
    /// schema the model sees and the parse the tool does cannot drift apart.
    pub fn typed<Args: schemars::JsonSchema>(
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        let name = name.into();
        let schema = typed_params_schema::<Args>(&name);
        Self::new(name, description, schema)
    }

    /// Accept non-strict schemas (or not). Strict is the default.
    #[must_use]
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict_json_schema = strict;
        self
    }
}

/// How a [`Tool`] is scheduled: enablement, pause-for-resolution and failure/timeout handling.
///
/// Grouped apart from [`ToolSpec`] because a tool's shape and its scheduling policy change for
/// different reasons; `Default` is "always enabled, never pauses, default error text, no timeout".
/// The struct is `#[non_exhaustive]`: start from [`ToolPolicy::new`] and chain.
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct ToolPolicy {
    /// Whether the tool is enabled.
    pub is_enabled: ToolEnabled,
    /// Whether / when a call pauses before invoke so the host can resolve it.
    pub needs_resolution: NeedsResolution,
    /// How a returned error is surfaced.
    pub failure_error_function: ToolFailureHandling,
    /// Timeout for each invocation, in seconds.
    pub timeout_seconds: Option<f64>,
    /// How a timeout is handled.
    pub timeout_behavior: ToolTimeoutBehavior,
    /// Formats the model-visible timeout text for `ErrorAsResult`.
    pub timeout_error_function: Option<ToolErrorFn>,
}

impl ToolPolicy {
    /// The default policy: always enabled, never pauses, default error text, no timeout.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pause every call (`true`) or none (`false`) for a resolution, or decide per call with a
    /// [`NeedsResolution::Dynamic`].
    #[must_use]
    pub fn needs_resolution(mut self, policy: impl Into<NeedsResolution>) -> Self {
        self.needs_resolution = policy.into();
        self
    }

    /// Offer the tool to the model always (`true`), never (`false`), or decide each turn.
    #[must_use]
    pub fn enabled(mut self, enabled: impl Into<ToolEnabled>) -> Self {
        self.is_enabled = enabled.into();
        self
    }

    /// Limit each invocation to `seconds`. Must be finite and greater than zero; checked when the
    /// run starts.
    #[must_use]
    pub fn timeout(mut self, seconds: f64) -> Self {
        self.timeout_seconds = Some(seconds);
        self
    }

    /// Choose what a timeout does.
    #[must_use]
    pub fn timeout_behavior(mut self, behavior: ToolTimeoutBehavior) -> Self {
        self.timeout_behavior = behavior;
        self
    }

    /// Choose how a returned error reaches the model (or aborts the run).
    #[must_use]
    pub fn on_failure(mut self, handling: ToolFailureHandling) -> Self {
        self.failure_error_function = handling;
        self
    }

    /// Format the model-visible timeout text.
    #[must_use]
    pub fn timeout_message<F>(mut self, f: F) -> Self
    where
        F: Fn(&RunContextWrapper, &AgentsError) -> String + Send + Sync + 'static,
    {
        self.timeout_error_function = Some(Arc::new(f));
        self
    }
}

/// A tool the model can call (Rust-only: Python has no equivalent trait, only the
/// `FunctionTool` dataclass).
///
/// [`FunctionTool`] is the SDK's own implementation and the concrete type the run loop executes.
/// Writing `impl Tool for MyTool` and converting it once with [`FunctionTool::from_tool`] runs it
/// through that exact same machinery, so a custom tool and a built-in one behave identically —
/// "use the trait" and "build a `FunctionTool` directly" are the same construction, not two
/// parallel code paths.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Name, description and parameter schema shown to the model.
    fn spec(&self) -> ToolSpec;

    /// Enablement, pause-for-resolution and failure/timeout policy.
    ///
    /// Defaults to [`ToolPolicy::default`]: always enabled, never pauses, default error message,
    /// no timeout.
    fn policy(&self) -> ToolPolicy {
        ToolPolicy::default()
    }

    /// Types this tool reads from [`Extensions`](crate::Extensions).
    ///
    /// The runner checks these before the first model turn and fails with the agent and tool
    /// names when one is missing, instead of surfacing a type error from inside the tool body.
    /// Defaults to none — a tool that only uses [`ToolContext::require`] without declaring it
    /// still works, it just fails later.
    fn requires(&self) -> &[TypeRequirement] {
        &[]
    }

    /// Invoke the tool. `ctx.tool_arguments` carries the raw JSON arguments from the model.
    async fn call(&self, ctx: &ToolContext) -> Result<ToolOutput, AgentsError>;
}

/// What a [`TypedTool`] is, declared on the argument struct (Rust-only).
///
/// Usually derived: `#[derive(ToolMeta)]` with a `#[tool(name = "...", ...)]` attribute fills in the
/// name, description and policy next to the fields that make up the schema. Implement it by hand
/// when the metadata has to be computed.
pub trait ToolMeta {
    /// Tool name shown to the model.
    const NAME: &'static str;
    /// Tool description shown to the model.
    const DESCRIPTION: &'static str;

    /// Enablement, pause-for-resolution and failure/timeout policy. Defaults to
    /// [`ToolPolicy::default`].
    fn policy() -> ToolPolicy {
        ToolPolicy::default()
    }

    /// Types the tool reads from [`Extensions`](crate::Extensions); see [`Tool::requires`].
    fn requires() -> Vec<TypeRequirement> {
        Vec::new()
    }
}

/// A tool whose argument struct *is* the tool (Rust-only).
///
/// The struct supplies the parameter schema (`JsonSchema`, so field doc comments become the
/// descriptions the model reads), the parse (`Deserialize`) and, through [`ToolMeta`], the name and
/// policy. `run` receives the already-parsed value. [`FunctionTool::from_typed`] turns it into the
/// SDK's tool type.
///
/// The tool has no state of its own; what it needs from the run comes from
/// [`ToolContext::require`] and is declared with `requires(..)`. A tool with state of its own
/// implements [`Tool`] instead.
#[async_trait]
pub trait TypedTool:
    ToolMeta + serde::de::DeserializeOwned + schemars::JsonSchema + Send + 'static
{
    /// What the tool answers with; serialized to JSON for the model.
    type Output: serde::Serialize + Send;

    /// Run the call. An `Err` is shown to the model according to the tool's failure policy.
    async fn run(self, ctx: &ToolContext) -> Result<Self::Output, AgentsError>;
}

/// A function tool exposed to the model (Python: `FunctionTool`).
#[derive(Clone)]
pub struct FunctionTool {
    /// Tool name shown to the LLM.
    pub name: String,
    /// Tool description shown to the LLM.
    pub description: String,
    /// JSON Schema for parameters.
    pub params_json_schema: Value,
    /// Invocation callback.
    pub on_invoke_tool: ToolInvoker,
    /// Whether the schema is strict.
    pub strict_json_schema: bool,
    /// Whether the tool is enabled.
    pub is_enabled: ToolEnabled,
    /// Whether / when a call pauses before invoke so the host can resolve it
    /// (Python: `needs_approval`).
    pub needs_resolution: NeedsResolution,
    /// How a returned error is surfaced (Python: `failure_error_function`).
    pub failure_error_function: ToolFailureHandling,
    /// Checks that run before the tool body (Python: `tool_input_guardrails`).
    pub tool_input_guardrails: Vec<ToolInputGuardrail>,
    /// Checks that run on the tool result (Python: `tool_output_guardrails`).
    pub tool_output_guardrails: Vec<ToolOutputGuardrail>,
    /// Timeout for each invocation, in seconds (Python: `timeout_seconds`).
    pub timeout_seconds: Option<f64>,
    /// How a timeout is handled (Python: `timeout_behavior`).
    pub timeout_behavior: ToolTimeoutBehavior,
    /// Formats the model-visible timeout text for `ErrorAsResult`
    /// (Python: `timeout_error_function`).
    pub timeout_error_function: Option<ToolErrorFn>,
    /// Types this tool needs in [`Extensions`](crate::Extensions), checked before the first
    /// model turn (Rust-only, see [`Tool::requires`]).
    pub requires: Vec<TypeRequirement>,
}

impl std::fmt::Debug for FunctionTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("params_json_schema", &self.params_json_schema)
            .field("strict_json_schema", &self.strict_json_schema)
            .field("is_enabled", &self.is_enabled)
            .field(
                "needs_resolution",
                &matches!(self.needs_resolution, NeedsResolution::Fixed(true)),
            )
            .finish()
    }
}

impl FunctionTool {
    /// Create a tool with an async invoker that returns a JSON [`Value`].
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        params_json_schema: Value,
        on_invoke: F,
    ) -> Self
    where
        F: Fn(ToolContext, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, AgentsError>> + Send + 'static,
    {
        let on_invoke = Arc::new(on_invoke);
        Self::new_with_result(name, description, params_json_schema, move |ctx, args| {
            let on_invoke = Arc::clone(&on_invoke);
            async move { Ok(ToolOutput::output(on_invoke(ctx, args).await?)) }
        })
    }

    /// Create a tool whose invoker can return nested interruptions (`Agent.as_tool`).
    pub fn new_with_result<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        params_json_schema: Value,
        on_invoke: F,
    ) -> Self
    where
        F: Fn(ToolContext, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolOutput, AgentsError>> + Send + 'static,
    {
        let on_invoke = Arc::new(on_invoke);
        Self {
            name: name.into(),
            description: description.into(),
            params_json_schema,
            on_invoke_tool: Arc::new(move |ctx, args| {
                let on_invoke = Arc::clone(&on_invoke);
                Box::pin(async move { on_invoke(ctx, args).await })
            }),
            strict_json_schema: true,
            is_enabled: ToolEnabled::Fixed(true),
            needs_resolution: NeedsResolution::Fixed(false),
            failure_error_function: ToolFailureHandling::Default,
            tool_input_guardrails: Vec::new(),
            tool_output_guardrails: Vec::new(),
            timeout_seconds: None,
            timeout_behavior: ToolTimeoutBehavior::default(),
            timeout_error_function: None,
            requires: Vec::new(),
        }
    }

    /// Declare that this tool needs a value of type `T` in [`Extensions`](crate::Extensions)
    /// (Rust-only, see [`Tool::requires`]).
    ///
    /// The runner checks the declaration before the first model turn and fails with the agent,
    /// tool and type names when the run does not carry one.
    #[must_use]
    pub fn require<T: std::any::Any>(mut self) -> Self {
        self.requires.push(TypeRequirement::of::<T>());
        self
    }

    /// Run `guardrails` before the tool body (Python: `tool_input_guardrails`).
    pub fn with_tool_input_guardrails(mut self, guardrails: Vec<ToolInputGuardrail>) -> Self {
        self.tool_input_guardrails = guardrails;
        self
    }

    /// Run `guardrails` on the tool result (Python: `tool_output_guardrails`).
    pub fn with_tool_output_guardrails(mut self, guardrails: Vec<ToolOutputGuardrail>) -> Self {
        self.tool_output_guardrails = guardrails;
        self
    }

    /// Limit each invocation to `seconds` (Python: `timeout_seconds`).
    ///
    /// Must be finite and greater than zero; this is checked when the run starts.
    pub fn with_timeout(mut self, seconds: f64) -> Self {
        self.timeout_seconds = Some(seconds);
        self
    }

    /// Choose what a timeout does (Python: `timeout_behavior`).
    pub fn with_timeout_behavior(mut self, behavior: ToolTimeoutBehavior) -> Self {
        self.timeout_behavior = behavior;
        self
    }

    /// Format the timeout message sent to the model (Python: `timeout_error_function`).
    pub fn with_timeout_error_function<F>(mut self, f: F) -> Self
    where
        F: Fn(&RunContextWrapper, &AgentsError) -> String + Send + Sync + 'static,
    {
        self.timeout_error_function = Some(Arc::new(f));
        self
    }

    /// Format tool errors for the model with `f` instead of the default message.
    pub fn with_failure_error_function<F>(mut self, f: F) -> Self
    where
        F: Fn(&RunContextWrapper, &AgentsError) -> String + Send + Sync + 'static,
    {
        self.failure_error_function = ToolFailureHandling::Custom(Arc::new(f));
        self
    }

    /// Abort the run when this tool errors (Python: `failure_error_function=None`).
    pub fn raise_on_error(mut self) -> Self {
        self.failure_error_function = ToolFailureHandling::Raise;
        self
    }

    /// Enable or disable the tool statically or per turn (Python: `is_enabled`).
    pub fn with_is_enabled(mut self, enabled: impl Into<ToolEnabled>) -> Self {
        self.is_enabled = enabled.into();
        self
    }

    /// Pause every call (`true`) or none (`false`) for resolution before invoke.
    pub fn with_needs_resolution(mut self, needs: impl Into<NeedsResolution>) -> Self {
        self.needs_resolution = needs.into();
        self
    }

    /// Decide per call, from its arguments and call id, whether it pauses for resolution.
    pub fn with_needs_resolution_fn<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(Value, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        let f = Arc::new(f);
        self.needs_resolution = NeedsResolution::Dynamic(Arc::new(move |params, call_id| {
            let f = Arc::clone(&f);
            Box::pin(async move { f(params, call_id).await })
        }));
        self
    }

    /// Create a tool that returns a constant JSON/string value (handy for tests).
    pub fn constant(
        name: impl Into<String>,
        description: impl Into<String>,
        return_value: impl Into<String>,
    ) -> Self {
        let return_value = return_value.into();
        Self::new(
            name,
            description,
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            move |_ctx, _args| {
                let return_value = return_value.clone();
                async move { Ok(Value::String(return_value)) }
            },
        )
    }

    /// Convert to a Responses/Chat tool definition JSON fragment.
    pub fn to_function_tool_param(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.params_json_schema,
            "strict": self.strict_json_schema
        })
    }

    /// Chat Completions tools array entry.
    pub fn to_chat_tool(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params_json_schema,
                "strict": self.strict_json_schema
            }
        })
    }

    /// Adapt any [`Tool`] implementation into a `FunctionTool`.
    ///
    /// Converts once, at construction; afterwards the run loop sees an ordinary `FunctionTool`
    /// and a hand-written `impl Tool` is indistinguishable from one built with
    /// [`FunctionTool::new`].
    pub fn from_tool<T>(tool: T) -> Self
    where
        T: Tool + 'static,
    {
        let tool = Arc::new(tool);
        let spec = tool.spec();
        let policy = tool.policy();
        let requires = tool.requires().to_vec();
        let invoke_tool = Arc::clone(&tool);
        Self {
            name: spec.name,
            description: spec.description,
            params_json_schema: spec.params_json_schema,
            on_invoke_tool: Arc::new(move |ctx, _args| {
                let invoke_tool = Arc::clone(&invoke_tool);
                Box::pin(async move { invoke_tool.call(&ctx).await })
            }),
            strict_json_schema: spec.strict_json_schema,
            is_enabled: policy.is_enabled,
            needs_resolution: policy.needs_resolution,
            failure_error_function: policy.failure_error_function,
            tool_input_guardrails: Vec::new(),
            tool_output_guardrails: Vec::new(),
            timeout_seconds: policy.timeout_seconds,
            timeout_behavior: policy.timeout_behavior,
            timeout_error_function: policy.timeout_error_function,
            requires,
        }
    }

    /// Build the tool a [`TypedTool`] describes: schema from the struct, name, description,
    /// policy and requirements from its [`ToolMeta`].
    pub fn from_typed<T: TypedTool>() -> Self {
        let schema = typed_params_schema::<T>(T::NAME);
        let policy = T::policy();
        let mut tool =
            Self::new_with_result(T::NAME, T::DESCRIPTION, schema, |ctx, args| async move {
                let parsed: T = serde_json::from_str(&args).map_err(|e| {
                    AgentsError::tool_with_source(
                        format!("invalid arguments for tool `{}`", T::NAME),
                        e,
                    )
                })?;
                let out = parsed.run(&ctx).await?;
                let value = serde_json::to_value(out).map_err(|e| {
                    AgentsError::tool_with_source(
                        format!("could not serialize the result of tool `{}`", T::NAME),
                        e,
                    )
                })?;
                Ok(ToolOutput::output(value))
            });
        tool.is_enabled = policy.is_enabled;
        tool.needs_resolution = policy.needs_resolution;
        tool.failure_error_function = policy.failure_error_function;
        tool.timeout_seconds = policy.timeout_seconds;
        tool.timeout_behavior = policy.timeout_behavior;
        tool.timeout_error_function = policy.timeout_error_function;
        tool.requires = T::requires();
        tool
    }

    /// Create a tool whose parameters are a typed `Deserialize` + `JsonSchema` struct, instead of
    /// a hand-written JSON Schema.
    ///
    /// This is what `#[function_tool]` generates for a plain function; call it directly when the
    /// macro does not fit (a closure that captures state, or parameters built at runtime).
    pub fn typed<Args, F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        on_invoke: F,
    ) -> Self
    where
        Args: serde::de::DeserializeOwned + schemars::JsonSchema,
        F: Fn(ToolContext, Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, AgentsError>> + Send + 'static,
    {
        Self::typed_with_result::<Args, _, _>(name, description, move |ctx, args| {
            let fut = on_invoke(ctx, args);
            async move { Ok(ToolOutput::output(fut.await?)) }
        })
    }

    /// [`FunctionTool::typed`] whose invoker can return nested interruptions (`Agent.as_tool`).
    pub fn typed_with_result<Args, F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        on_invoke: F,
    ) -> Self
    where
        Args: serde::de::DeserializeOwned + schemars::JsonSchema,
        F: Fn(ToolContext, Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolOutput, AgentsError>> + Send + 'static,
    {
        let name = name.into();
        let schema = typed_params_schema::<Args>(&name);
        let on_invoke = Arc::new(on_invoke);
        Self::new_with_result(name, description, schema, move |ctx, args| {
            let on_invoke = Arc::clone(&on_invoke);
            async move {
                let parsed: Args = serde_json::from_str(&args).map_err(|e| {
                    AgentsError::tool_with_source("invalid arguments for typed tool", e)
                })?;
                on_invoke(ctx, parsed).await
            }
        })
    }
}

#[async_trait]
impl Tool for FunctionTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.clone(),
            description: self.description.clone(),
            params_json_schema: self.params_json_schema.clone(),
            strict_json_schema: self.strict_json_schema,
        }
    }

    fn policy(&self) -> ToolPolicy {
        ToolPolicy {
            is_enabled: self.is_enabled.clone(),
            needs_resolution: self.needs_resolution.clone(),
            failure_error_function: self.failure_error_function.clone(),
            timeout_seconds: self.timeout_seconds,
            timeout_behavior: self.timeout_behavior,
            timeout_error_function: self.timeout_error_function.clone(),
        }
    }

    fn requires(&self) -> &[TypeRequirement] {
        &self.requires
    }

    async fn call(&self, ctx: &ToolContext) -> Result<ToolOutput, AgentsError> {
        (self.on_invoke_tool)(ctx.clone(), ctx.tool_arguments.clone()).await
    }
}

/// `schemars::schema_for!(Args)`, normalized to a strict schema (Python: the same conversion
/// `#[function_tool]` runs on its generated args struct).
fn typed_params_schema<Args: schemars::JsonSchema>(tool_name: &str) -> Value {
    let raw = serde_json::to_value(schemars::schema_for!(Args)).unwrap_or_else(|e| {
        panic!("tool `{tool_name}`: could not serialize the parameter JSON schema: {e}")
    });
    let mut schema = crate::strict_schema::ensure_strict_json_schema(&raw).unwrap_or_else(|e| {
        panic!("tool `{tool_name}`: parameters cannot be converted to a strict JSON schema: {e}")
    });
    // Python's pydantic schemas do not carry a `$schema` keyword; drop it so the advertised tool
    // schema matches the macro-generated one byte for byte.
    if let Some(root) = schema.as_object_mut() {
        root.remove("$schema");
    }
    schema
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_context::RunContextWrapper;

    fn ctx(args: &str) -> ToolContext {
        ToolContext::new("probe", "call-1", args, RunContextWrapper::default())
    }

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct AddArgs {
        a: i64,
        b: i64,
    }

    /// A hand-written `impl Tool`, as a library consumer would write one.
    struct Doubler;

    #[async_trait]
    impl Tool for Doubler {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "doubler".into(),
                description: "doubles a number".into(),
                params_json_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "n": { "type": "integer" } },
                    "required": ["n"],
                    "additionalProperties": false
                }),
                strict_json_schema: true,
            }
        }

        fn policy(&self) -> ToolPolicy {
            ToolPolicy {
                needs_resolution: NeedsResolution::Fixed(true),
                ..ToolPolicy::default()
            }
        }

        async fn call(&self, ctx: &ToolContext) -> Result<ToolOutput, AgentsError> {
            let parsed: serde_json::Value = serde_json::from_str(&ctx.tool_arguments).unwrap();
            let n = parsed["n"].as_i64().unwrap_or(0);
            Ok(ToolOutput::output(Value::from(n * 2)))
        }
    }

    #[tokio::test]
    async fn from_tool_adapts_a_custom_tool_into_a_function_tool() {
        let tool = FunctionTool::from_tool(Doubler);
        assert_eq!(tool.name, "doubler");
        assert_eq!(tool.description, "doubles a number");
        assert!(tool.strict_json_schema);
        // The policy carried over too: `Doubler` always pauses for resolution.
        assert!(matches!(
            tool.needs_resolution,
            NeedsResolution::Fixed(true)
        ));

        let out = (tool.on_invoke_tool)(ctx(r#"{"n":21}"#), r#"{"n":21}"#.into())
            .await
            .unwrap();
        assert_eq!(out.value(), Some(&Value::from(42)));
    }

    #[test]
    fn a_typed_spec_advertises_the_same_schema_as_typed() {
        let spec = ToolSpec::typed::<AddArgs>("add", "adds two numbers");
        let tool =
            FunctionTool::typed::<AddArgs, _, _>("add", "adds two numbers", |_c, a| async move {
                Ok(Value::from(a.a + a.b))
            });
        assert_eq!(spec.params_json_schema, tool.params_json_schema);
        assert!(spec.strict_json_schema);
        assert!(!spec.strict(false).strict_json_schema);
    }

    #[tokio::test]
    async fn the_policy_builders_set_what_they_name() {
        let policy = ToolPolicy::new()
            .needs_resolution(NeedsResolution::dynamic(|params, call_id| async move {
                params["n"].as_u64().unwrap_or(0) > 5 && call_id != "skip"
            }))
            .enabled(false)
            .timeout(2.5)
            .timeout_behavior(ToolTimeoutBehavior::RaiseException)
            .on_failure(ToolFailureHandling::Raise)
            .timeout_message(|_ctx, _err| "too slow".to_string());

        assert_eq!(policy.timeout_seconds, Some(2.5));
        assert_eq!(policy.timeout_behavior, ToolTimeoutBehavior::RaiseException);
        assert!(matches!(
            policy.failure_error_function,
            ToolFailureHandling::Raise
        ));
        assert!(matches!(policy.is_enabled, ToolEnabled::Fixed(false)));
        assert!(policy.timeout_error_function.is_some());

        let rule = &policy.needs_resolution;
        assert!(rule.requires(r#"{"n": 9}"#, "c1").await);
        assert!(!rule.requires(r#"{"n": 1}"#, "c1").await);
        assert!(
            !rule.requires(r#"{"n": 9}"#, "skip").await,
            "the call id is passed through"
        );
        // Arguments that are not a JSON object count as "no parameters", not as a panic.
        assert!(!rule.requires("not json", "c1").await);
    }

    #[test]
    fn the_default_policy_is_the_documented_one() {
        let policy = ToolPolicy::new();
        assert!(matches!(
            policy.needs_resolution,
            NeedsResolution::Fixed(false)
        ));
        assert!(matches!(policy.is_enabled, ToolEnabled::Fixed(true)));
        assert!(matches!(
            policy.failure_error_function,
            ToolFailureHandling::Default
        ));
        assert_eq!(policy.timeout_seconds, None);
    }

    #[tokio::test]
    async fn typed_builds_a_schema_from_the_args_struct_and_parses_them() {
        let tool = FunctionTool::typed::<AddArgs, _, _>(
            "add",
            "adds two numbers",
            |_ctx, args| async move { Ok(Value::from(args.a + args.b)) },
        );
        let schema = tool.params_json_schema.clone();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
        let props = schema["properties"].as_object().unwrap();
        assert!(props.contains_key("a") && props.contains_key("b"));

        let out = (tool.on_invoke_tool)(ctx(r#"{"a":2,"b":3}"#), r#"{"a":2,"b":3}"#.into())
            .await
            .unwrap();
        assert_eq!(out.value(), Some(&Value::from(5)));
    }

    #[tokio::test]
    async fn typed_rejects_arguments_that_do_not_match_the_schema() {
        let tool = FunctionTool::typed::<AddArgs, _, _>(
            "add",
            "adds two numbers",
            |_ctx, args| async move { Ok(Value::from(args.a + args.b)) },
        );
        let err = (tool.on_invoke_tool)(ctx(r#"{"a":"nope"}"#), r#"{"a":"nope"}"#.into())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("invalid arguments"));
    }

    #[tokio::test]
    async fn typed_with_result_reaches_the_raw_result_constructor() {
        let tool =
            FunctionTool::typed_with_result::<AddArgs, _, _>("add", "d", |_ctx, args| async move {
                Ok(ToolOutput::output(Value::from(args.a + args.b)))
            });
        let out = (tool.on_invoke_tool)(ctx(r#"{"a":1,"b":2}"#), r#"{"a":1,"b":2}"#.into())
            .await
            .unwrap();
        assert!(!out.is_interrupted());
        assert_eq!(out.value(), Some(&Value::from(3)));
    }
}
