//! [`FunctionTool`]: the concrete tool the run loop executes, and its constructors.

use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use super::*;

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

    /// The strict `parameters` schema every typed constructor and `#[function_tool]` advertises
    /// for `Args`. Public for the macro's expansion; not part of the stable API.
    #[doc(hidden)]
    pub fn parameters_schema<Args: schemars::JsonSchema>(
        tool_name: &str,
    ) -> Result<Value, AgentsError> {
        typed_params_schema::<Args>(tool_name)
    }

    /// Build the tool a [`TypedTool`] describes: schema from its `Args`, name, description,
    /// policy and requirements from its [`ToolMeta`]. The tool value is kept and shared by every
    /// call, so state it holds lives as long as the `FunctionTool`.
    ///
    /// # Panics
    ///
    /// When `T::Args` has no strict JSON schema; use [`FunctionTool::try_from_typed`] for an error.
    pub fn from_typed<T: TypedTool>(tool: T) -> Self {
        Self::try_from_typed(tool).unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`FunctionTool::from_typed`] that returns an error instead of panicking when `T::Args` has
    /// no strict JSON schema.
    pub fn try_from_typed<T: TypedTool>(tool: T) -> Result<Self, AgentsError> {
        let tool = Arc::new(tool);
        let name = tool.name().to_string();
        let schema = typed_params_schema::<T::Args>(&name)?;
        let policy = tool.policy();
        let requires = tool.requires();
        let call_name = name.clone();
        let call_tool = Arc::clone(&tool);
        let mut built = Self::new_with_result(
            name,
            tool.description().to_string(),
            schema,
            move |ctx, args| {
                let tool = Arc::clone(&call_tool);
                let name = call_name.clone();
                async move {
                    let parsed: T::Args = serde_json::from_str(&args).map_err(|e| {
                        AgentsError::tool_with_source(
                            format!("invalid arguments for tool `{name}`"),
                            e,
                        )
                    })?;
                    let out = tool.run(parsed, &ctx).await?;
                    let value = serde_json::to_value(out).map_err(|e| {
                        AgentsError::tool_with_source(
                            format!("could not serialize the result of tool `{name}`"),
                            e,
                        )
                    })?;
                    Ok(ToolOutput::output(value))
                }
            },
        );
        built.is_enabled = policy.is_enabled;
        built.needs_resolution = policy.needs_resolution;
        built.failure_error_function = policy.failure_error_function;
        built.timeout_seconds = policy.timeout_seconds;
        built.timeout_behavior = policy.timeout_behavior;
        built.timeout_error_function = policy.timeout_error_function;
        built.requires = requires;
        Ok(built)
    }

    /// Create a tool whose parameters are a typed `Deserialize` + `JsonSchema` struct, instead of
    /// a hand-written JSON Schema.
    ///
    /// This is what `#[function_tool]` generates for a plain function; call it directly when the
    /// macro does not fit (a closure that captures state, or parameters built at runtime).
    ///
    /// # Panics
    ///
    /// When `Args` has no strict JSON schema; use [`FunctionTool::try_typed`] to get an error
    /// instead. The schema depends only on the type, so this fails the first time the tool is
    /// built, never on a model's input.
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
        Self::try_typed::<Args, F, Fut>(name, description, on_invoke)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`FunctionTool::typed`] that returns an error instead of panicking when `Args` has no
    /// strict JSON schema.
    pub fn try_typed<Args, F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        on_invoke: F,
    ) -> Result<Self, AgentsError>
    where
        Args: serde::de::DeserializeOwned + schemars::JsonSchema,
        F: Fn(ToolContext, Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, AgentsError>> + Send + 'static,
    {
        Self::try_typed_with_result::<Args, _, _>(name, description, move |ctx, args| {
            let fut = on_invoke(ctx, args);
            async move { Ok(ToolOutput::output(fut.await?)) }
        })
    }

    /// [`FunctionTool::typed`] whose invoker can return nested interruptions (`Agent.as_tool`).
    ///
    /// # Panics
    ///
    /// When `Args` has no strict JSON schema; see [`FunctionTool::try_typed_with_result`].
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
        Self::try_typed_with_result::<Args, F, Fut>(name, description, on_invoke)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`FunctionTool::typed_with_result`] that returns an error instead of panicking when `Args`
    /// has no strict JSON schema.
    pub fn try_typed_with_result<Args, F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        on_invoke: F,
    ) -> Result<Self, AgentsError>
    where
        Args: serde::de::DeserializeOwned + schemars::JsonSchema,
        F: Fn(ToolContext, Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolOutput, AgentsError>> + Send + 'static,
    {
        let name = name.into();
        let schema = typed_params_schema::<Args>(&name)?;
        let on_invoke = Arc::new(on_invoke);
        Ok(Self::new_with_result(
            name,
            description,
            schema,
            move |ctx, args| {
                let on_invoke = Arc::clone(&on_invoke);
                async move {
                    let parsed: Args = serde_json::from_str(&args).map_err(|e| {
                        AgentsError::tool_with_source("invalid arguments for typed tool", e)
                    })?;
                    on_invoke(ctx, parsed).await
                }
            },
        ))
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
pub(super) fn typed_params_schema<Args: schemars::JsonSchema>(
    tool_name: &str,
) -> Result<Value, AgentsError> {
    let raw = serde_json::to_value(schemars::schema_for!(Args)).map_err(|e| {
        UserError::new(format!(
            "tool `{tool_name}`: could not serialize the parameter JSON schema: {e}"
        ))
    })?;
    let mut schema = crate::strict_schema::ensure_strict_json_schema(&raw).map_err(|e| {
        UserError::new(format!(
            "tool `{tool_name}`: parameters cannot be converted to a strict JSON schema: {e}"
        ))
    })?;
    if let Some(root) = schema.as_object_mut() {
        // Python's pydantic schemas do not carry a `$schema` keyword.
        root.remove("$schema");
        // `title` is the Rust type name and `description` repeats the tool's own description
        // (schemars copies the struct's doc comment into both). The tool carries its identity
        // beside `parameters`; inside it the model needs only the properties.
        root.remove("title");
        root.remove("description");
        // `$defs` is dead weight once every `$ref` has been inlined.
        let defs = root.remove("$defs");
        if let Some(defs) = defs {
            if contains_ref(root.values()) {
                root.insert("$defs".to_string(), defs);
            }
        }
    }
    Ok(schema)
}

fn contains_ref<'a>(mut values: impl Iterator<Item = &'a Value>) -> bool {
    fn walk(value: &Value) -> bool {
        match value {
            Value::Object(map) => map.contains_key("$ref") || map.values().any(walk),
            Value::Array(items) => items.iter().any(walk),
            _ => false,
        }
    }
    values.any(walk)
}
