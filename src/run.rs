//! Runner and run configuration (Python: `agents.run` / `run_config`).

use std::sync::{Arc, Mutex, OnceLock};

use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::agent::{Agent, ToolUseBehavior};
use crate::error::{AgentsError, MaxTurnsExceeded, ModelError, UserError};
use crate::handoffs::Handoff;
use crate::items::{
    extract_message_text, function_call_parts, is_function_call, InputLike, ItemHelpers,
    MessageOutputItem, ModelResponse, RunItem, ToolCallItem, ToolCallOutputItem,
};
use crate::model::{Model, ModelInput, ModelRequest, ModelTracing};
use crate::model_settings::ModelSettings;
use crate::result::{RunResult, RunResultStreaming, StreamingSnapshot};
use crate::stream_events::{RunItemStreamName, StreamEvent};
use crate::tool::{FunctionTool, ToolContext};
use crate::tracing::{agent_span, function_span, generation_span, trace};
use crate::usage::Usage;

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
    /// Run an agent asynchronously (Python: `Runner.run`).
    pub async fn run(
        starting_agent: &Agent,
        input: impl Into<InputLike>,
        options: RunOptions,
    ) -> Result<RunResult, AgentsError> {
        run_loop(starting_agent.clone(), input.into(), options, None, None).await
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
                input,
                options,
                Some(tx.clone()),
                Some(Arc::clone(&snap)),
            )
            .await;
            match result {
                Ok(r) => {
                    let mut s = snap.lock().expect("snapshot");
                    s.is_complete = true;
                    s.final_output = Some(r.final_output);
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
    input: InputLike,
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

    let mut current_agent = starting_agent;
    let mut model_settings = current_agent.model_settings.clone();
    if let Some(override_settings) = &options.run_config.model_settings {
        merge_settings(&mut model_settings, override_settings);
    }

    let mut generated_items: Vec<RunItem> = Vec::new();
    let mut raw_responses: Vec<ModelResponse> = Vec::new();
    let mut usage = Usage::default();
    let mut current_input_items = ItemHelpers::input_to_new_input_list(&input);
    let mut previous_response_id = options.previous_response_id.clone();

    emit(
        &events,
        StreamEvent::AgentUpdated {
            agent_name: current_agent.name.clone(),
        },
    )
    .await;

    let mut turn = 0usize;
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
        let _gen_span = generation_span(current_agent.model_name.as_deref().unwrap_or("scripted"));

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
            model.get_response(req).await?
        };

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

        if let Some(id) = &response.response_id {
            previous_response_id = Some(id.clone());
        }
        usage.add(&response.usage);

        let (function_calls, messages): (Vec<_>, Vec<_>) = response
            .output
            .iter()
            .cloned()
            .partition(|item| is_function_call(item));

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

        raw_responses.push(response.clone());
        if let Some(snap) = &snapshot {
            let mut s = snap.lock().expect("snapshot");
            s.new_items = generated_items.clone();
            s.raw_responses = raw_responses.clone();
            s.usage = usage.clone();
        }

        if function_calls.is_empty() {
            let final_text = messages
                .iter()
                .filter_map(|m| extract_message_text(m))
                .collect::<Vec<_>>()
                .join("");
            return Ok(RunResult {
                input,
                new_items: generated_items,
                raw_responses,
                final_output: Value::String(final_text),
                last_agent_name: current_agent.name.clone(),
                max_turns: Some(max_turns),
                usage,
            });
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
            let out_item = tool_output_item(&current_agent.name, &call_id, &Value::String(transfer.clone()));
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

        // Parallel function tools, matching Python `asyncio.gather` default (order preserved).
        let tool_futs = function_calls.iter().map(|call| {
            let tools = tools.clone();
            let call = call.clone();
            async move {
                let (name, arguments, call_id) = function_call_parts(&call)
                    .ok_or_else(|| ModelError::Behavior("malformed function_call item".into()))?;
                let tool = tools
                    .iter()
                    .find(|t| t.name == name)
                    .cloned()
                    .ok_or_else(|| UserError::new(format!("Tool not found: {name}")))?;
                let _fs = function_span(&name);
                let ctx = ToolContext {
                    tool_name: name.clone(),
                    tool_call_id: call_id.clone(),
                    tool_arguments: arguments.clone(),
                };
                let output = (tool.on_invoke_tool)(ctx, arguments).await?;
                Ok::<_, AgentsError>((tool, output, call_id))
            }
        });
        let tool_results = futures::future::try_join_all(tool_futs).await?;

        match &current_agent.tool_use_behavior {
            ToolUseBehavior::StopOnFirstTool => {
                let (_tool, output, call_id) = &tool_results[0];
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
                return Ok(RunResult {
                    input,
                    new_items: generated_items,
                    raw_responses,
                    final_output: output.clone(),
                    last_agent_name: current_agent.name.clone(),
                    max_turns: Some(max_turns),
                    usage,
                });
            }
            ToolUseBehavior::StopAtTools { stop_at_tool_names } => {
                for (tool, output, call_id) in &tool_results {
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
                    if stop_at_tool_names.iter().any(|n| n == &tool.name) {
                        return Ok(RunResult {
                            input,
                            new_items: generated_items,
                            raw_responses,
                            final_output: output.clone(),
                            last_agent_name: current_agent.name.clone(),
                            max_turns: Some(max_turns),
                            usage,
                        });
                    }
                }
            }
            ToolUseBehavior::RunLlmAgain => {
                for (_tool, output, call_id) in &tool_results {
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

        for item in &response.output {
            current_input_items.push(item.clone());
        }
        for (_tool, output, call_id) in &tool_results {
            let already = generated_items.iter().any(|gi| {
                matches!(gi, RunItem::ToolCallOutput(o) if o.raw_item.get("call_id").and_then(|c| c.as_str()) == Some(call_id.as_str()))
            });
            if !already {
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
