//! `Agent.as_tool`: run an agent as a function tool (Python: `Agent.as_tool`).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::agent::Agent;
use crate::error::AgentsError;
use crate::extensions::ContextValue;
use crate::result::RunResultStreaming;
use crate::stream_events::StreamEvent;
use crate::tool::{FunctionTool, ToolOutput};

use super::agent_loop::{take_nested_resume_state, LoopStart};
use super::config::RunOptions;

/// One event of a nested agent run, handed to [`AsToolConfig::on_stream`]
/// (Python: `AgentToolStreamEvent`).
///
/// Every event of the nested run is forwarded in order, including its own tool calls and
/// handoffs. `agent_name` follows the nested run through handoffs: it changes right after an
/// [`StreamEvent::AgentUpdated`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AgentToolStreamEvent {
    /// The event from the nested run.
    pub event: StreamEvent,
    /// Name of the nested agent that produced it.
    pub agent_name: String,
    /// Name of the agent-tool, as the outer model called it.
    pub tool_name: String,
    /// Id of the outer model's tool call that started the nested run.
    pub tool_call_id: String,
}

/// Callback receiving the stream events of a nested run (Python: `on_stream`).
pub type AgentToolStreamHandler =
    Arc<dyn Fn(AgentToolStreamEvent) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// Options for [`Agent::as_tool`] (Python: `Agent.as_tool` kwargs subset).
///
/// The Python keyword list is much longer (D-043), so fields will be added here: build one with
/// `AsToolConfig { field, ..Default::default() }` and the tail keeps you compiling.
///
/// # Run context
///
/// The nested run is an extension of the outer tool call, so by default it sees the same user
/// context as the outer run: everything given with `RunOptions::context` / `RunOptions::with`.
/// Precedence: [`AsToolConfig::context`] (explicit value, nothing from the outer run) >
/// `inherit_context(false)` (no context at all) > inherit (default).
///
/// The context may hold secrets (tokens, clients): by default every tool of the nested agent can
/// read them. Use `inherit_context(false)` for a sub-agent that must not.
///
/// ```
/// use opai_agents_core::{Agent, AsToolConfig};
///
/// struct Hub;
/// let director = Agent::new("director");
/// // Inherits the outer run's context (default).
/// let _inherit = director.as_tool(AsToolConfig::default());
/// // Isolated: the nested run has no context.
/// let _isolated = director.as_tool(AsToolConfig::default().inherit_context(false));
/// // Explicit: the nested run sees only this value.
/// let _explicit = director.as_tool(AsToolConfig::default().context(Hub));
/// ```
#[derive(Clone)]
pub struct AsToolConfig {
    /// Override tool name (default: sanitized agent name).
    pub name: Option<String>,
    /// Tool description (default: handoff_description or agent name).
    pub description: Option<String>,
    /// Whether a call of the agent-tool pauses for resolution before the nested run starts
    /// (Python: `needs_approval`).
    pub needs_resolution: bool,
    /// Max turns for the nested run.
    pub max_turns: Option<usize>,
    /// Receives every stream event of the nested run (Python: `on_stream`). When set, the nested
    /// agent runs in streaming mode; the tool's result is the same either way.
    ///
    /// The handler is awaited for each event before the next is taken, so a slow handler slows
    /// the nested run down instead of queueing events without bound (Python buffers up to
    /// `on_stream_max_pending_events` and then fails the tool; see `docs/DEVIATIONS.md`).
    pub on_stream: Option<AgentToolStreamHandler>,
    /// Whether the nested run inherits the outer run's user context (default `true`). Ignored
    /// when [`AsToolConfig::context`] is set.
    pub inherit_context: bool,
    /// Explicit context for the nested run. When set, only this value is visible to the nested
    /// run; nothing is inherited from the outer run.
    pub context: Option<ContextValue>,
}

impl Default for AsToolConfig {
    fn default() -> Self {
        Self {
            name: None,
            description: None,
            needs_resolution: false,
            max_turns: None,
            on_stream: None,
            inherit_context: true,
            context: None,
        }
    }
}

/// How a nested run gets its user context; resolved once from [`AsToolConfig`].
#[derive(Clone)]
enum ContextPolicy {
    Inherit,
    Isolated,
    Explicit(ContextValue),
}

impl AsToolConfig {
    /// Give the nested run exactly this context, instead of the outer run's.
    #[must_use]
    pub fn context<T: std::any::Any + Send + Sync>(mut self, value: T) -> Self {
        self.context = Some(Arc::new(value));
        self
    }

    /// Choose whether the nested run inherits the outer run's context (default `true`).
    #[must_use]
    pub fn inherit_context(mut self, inherit: bool) -> Self {
        self.inherit_context = inherit;
        self
    }

    /// Stream the nested run's events to `handler` (Python: `on_stream`).
    #[must_use]
    pub fn on_stream<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(AgentToolStreamEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.on_stream = Some(Arc::new(move |event| Box::pin(handler(event))));
        self
    }
}

impl std::fmt::Debug for AsToolConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsToolConfig")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("needs_resolution", &self.needs_resolution)
            .field("max_turns", &self.max_turns)
            .field("on_stream", &self.on_stream.as_ref().map(|_| "set"))
            .field("inherit_context", &self.inherit_context)
            .field("context", &self.context.as_ref().map(|_| "set"))
            .finish()
    }
}

impl Agent {
    /// Expose this agent as a function tool (Python: `Agent.as_tool`).
    ///
    /// Nested tool interruptions surface on the outer run's `interruptions`; resolve them on the
    /// outer [`crate::RunState`], then resume with [`crate::Runner::run_state`].
    ///
    /// Lives next to the runner it calls, so `agent` stays independent of `run`.
    pub fn as_tool(&self, config: AsToolConfig) -> FunctionTool {
        build_agent_as_tool(self, config)
    }
}

/// Build an `Agent.as_tool` FunctionTool (Python: `Agent.as_tool`).
fn build_agent_as_tool(agent: &Agent, config: AsToolConfig) -> FunctionTool {
    // Shared by every call of the tool; each call starts a run on the same `Arc`.
    let nested_agent = Arc::new(agent.clone());
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
    let needs = config.needs_resolution;
    let on_stream = config.on_stream;
    let policy = match config.context {
        Some(value) => ContextPolicy::Explicit(value),
        None if config.inherit_context => ContextPolicy::Inherit,
        None => ContextPolicy::Isolated,
    };

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
            let nested_agent = Arc::clone(&nested_agent);
            let on_stream = on_stream.clone();
            let policy = policy.clone();
            async move {
                let input_text = extract_as_tool_input(&args)?;
                if let Some(handler) = on_stream {
                    return run_streamed_nested(
                        nested_agent,
                        input_text,
                        max_turns,
                        handler,
                        &policy,
                        &ctx,
                    )
                    .await;
                }
                // The nested run stops with the outer one.
                let opts = nested_run_options(&ctx, max_turns, ctx.cancellation().clone(), &policy);

                let result = if let Some(state) = take_nested_resume_state(&ctx.tool_call_id) {
                    super::run_state_shared(nested_agent, state, opts).await?
                } else {
                    super::run_shared(nested_agent, input_text.into(), opts).await?
                };

                if result.is_interrupted() {
                    let nested_state = result.to_state()?;
                    return Ok(ToolOutput::interrupted(
                        result.interruptions.clone(),
                        nested_state,
                    ));
                }

                Ok(ToolOutput::output(result.final_output))
            }
        },
    )
    .with_needs_resolution(needs)
}

/// Options of a nested run, shared by the plain, resumed and streamed paths so they agree on
/// the context.
fn nested_run_options(
    ctx: &crate::tool::ToolContext,
    max_turns: Option<usize>,
    cancellation: tokio_util::sync::CancellationToken,
    policy: &ContextPolicy,
) -> RunOptions {
    let mut opts = RunOptions::default();
    opts.cancellation = Some(cancellation);
    if let Some(mt) = max_turns {
        opts.max_turns = Some(mt);
    }
    match policy {
        // The outer extensions already include `RunOptions::context`; `opts.context` stays
        // `None` so the same value is not inserted twice.
        ContextPolicy::Inherit => opts.extensions = ctx.run_context.extensions().clone(),
        ContextPolicy::Isolated => {}
        ContextPolicy::Explicit(value) => opts.context = Some(Arc::clone(value)),
    }
    opts
}

/// The nested run in streaming mode: every event goes to `handler`, then the outcome is turned
/// into the same [`ToolOutput`] the non-streaming path returns.
async fn run_streamed_nested(
    nested_agent: Arc<Agent>,
    input_text: String,
    max_turns: Option<usize>,
    handler: AgentToolStreamHandler,
    policy: &ContextPolicy,
    ctx: &crate::tool::ToolContext,
) -> Result<ToolOutput, AgentsError> {
    // A child token: the nested run stops with the outer one, but dropping this tool call (a
    // timeout, an outer cancel) cancels only the nested run, never the outer.
    let opts = nested_run_options(ctx, max_turns, ctx.cancellation().child_token(), policy);

    let mut stream: RunResultStreaming =
        if let Some(state) = take_nested_resume_state(&ctx.tool_call_id) {
            super::check_resumable(&state, &nested_agent)?;
            super::spawn_streamed(
                Arc::clone(&nested_agent),
                LoopStart::Resume {
                    state: Box::new(state),
                },
                opts,
            )
        } else {
            super::spawn_streamed(
                Arc::clone(&nested_agent),
                LoopStart::Fresh {
                    input: input_text.into(),
                    prepared: None,
                },
                opts,
            )
        }
        .cancel_on_drop(true);

    let mut current_agent = nested_agent.name.clone();
    while let Some(event) = stream.next_event().await {
        let event = event?;
        if let StreamEvent::AgentUpdated { agent_name } = &event {
            current_agent = agent_name.clone();
        }
        handler(AgentToolStreamEvent {
            event,
            agent_name: current_agent.clone(),
            tool_name: ctx.tool_name.clone(),
            tool_call_id: ctx.tool_call_id.clone(),
        })
        .await;
    }

    if stream.is_interrupted() {
        let nested_state = stream.to_state()?;
        let interruptions = stream
            .snapshot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .interruptions
            .clone();
        return Ok(ToolOutput::interrupted(interruptions, nested_state));
    }
    Ok(ToolOutput::output(
        stream.final_output().unwrap_or(Value::Null),
    ))
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
