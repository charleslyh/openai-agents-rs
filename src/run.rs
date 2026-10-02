//! Runner and run configuration (Python: `agents.run` / `run_config` Phase-1).

use std::sync::{Arc, Mutex, OnceLock};

use serde_json::Value;

use crate::agent::{Agent, ToolUseBehavior};
use crate::error::{AgentsError, MaxTurnsExceeded, ModelError, UserError};
use crate::items::{
    extract_message_text, function_call_parts, is_function_call, InputLike, ItemHelpers,
    MessageOutputItem, ModelResponse, RunItem, ToolCallItem, ToolCallOutputItem,
};
use crate::model::{Model, ModelInput, ModelRequest, ModelTracing};
use crate::model_settings::ModelSettings;
use crate::result::RunResult;
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

/// Per-call options for [`Runner::run`].
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

/// Facade entry point (Python: `Runner`).
pub struct Runner;

impl Runner {
    /// Run an agent asynchronously (Python: `Runner.run`).
    pub async fn run(
        starting_agent: &Agent,
        input: impl Into<InputLike>,
        options: RunOptions,
    ) -> Result<RunResult, AgentsError> {
        let input = input.into();
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

        let model = resolve_model(starting_agent, &options.run_config)?;
        let mut model_settings = starting_agent.model_settings.clone();
        if let Some(override_settings) = &options.run_config.model_settings {
            merge_settings(&mut model_settings, override_settings);
        }

        let mut generated_items: Vec<RunItem> = Vec::new();
        let mut raw_responses: Vec<ModelResponse> = Vec::new();
        let mut usage = Usage::default();
        let mut current_input_items = ItemHelpers::input_to_new_input_list(&input);
        let mut previous_response_id = options.previous_response_id.clone();
        let tools = starting_agent.enabled_tools();

        let mut turn = 0usize;
        loop {
            turn += 1;
            if turn > max_turns {
                return Err(MaxTurnsExceeded { max_turns }.into());
            }

            let _agent_span = agent_span(&starting_agent.name);
            let _gen_span = generation_span(
                starting_agent
                    .model_name
                    .as_deref()
                    .unwrap_or("scripted"),
            );

            let response = {
                let req = ModelRequest {
                    system_instructions: starting_agent.instructions.as_deref(),
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
                generated_items.push(RunItem::Message(MessageOutputItem {
                    agent_name: starting_agent.name.clone(),
                    raw_item: msg.clone(),
                }));
            }
            for call in &function_calls {
                generated_items.push(RunItem::ToolCall(ToolCallItem {
                    agent_name: starting_agent.name.clone(),
                    raw_item: call.clone(),
                }));
            }

            raw_responses.push(response.clone());

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
                    last_agent_name: starting_agent.name.clone(),
                    max_turns: Some(max_turns),
                    usage,
                });
            }

            let mut tool_results: Vec<(FunctionTool, Value, String)> = Vec::new();
            for call in &function_calls {
                let (name, arguments, call_id) = function_call_parts(call).ok_or_else(|| {
                    ModelError::Behavior("malformed function_call item".into())
                })?;
                let tool = tools
                    .iter()
                    .find(|t| t.name == name)
                    .cloned()
                    .ok_or_else(|| {
                        UserError::new(format!("Tool not found: {name}"))
                    })?;

                let _fs = function_span(&name);
                let ctx = ToolContext {
                    tool_name: name.clone(),
                    tool_call_id: call_id.clone(),
                    tool_arguments: arguments.clone(),
                };
                let output = (tool.on_invoke_tool)(ctx, arguments).await.map_err(|e| {
                    // Match Python default: surface as string unless it's a hard AgentsError::Tool
                    match e {
                        AgentsError::Tool(msg) => AgentsError::Tool(msg),
                        other => other,
                    }
                })?;
                tool_results.push((tool, output, call_id));
            }

            // tool_use_behavior
            match &starting_agent.tool_use_behavior {
                ToolUseBehavior::StopOnFirstTool => {
                    let (_tool, output, call_id) = &tool_results[0];
                    let out_item = tool_output_item(&starting_agent.name, call_id, output);
                    generated_items.push(out_item);
                    return Ok(RunResult {
                        input,
                        new_items: generated_items,
                        raw_responses,
                        final_output: output.clone(),
                        last_agent_name: starting_agent.name.clone(),
                        max_turns: Some(max_turns),
                        usage,
                    });
                }
                ToolUseBehavior::StopAtTools { stop_at_tool_names } => {
                    for (tool, output, call_id) in &tool_results {
                        let out_item = tool_output_item(&starting_agent.name, call_id, output);
                        generated_items.push(out_item.clone());
                        if stop_at_tool_names.iter().any(|n| n == &tool.name) {
                            return Ok(RunResult {
                                input,
                                new_items: generated_items,
                                raw_responses,
                                final_output: output.clone(),
                                last_agent_name: starting_agent.name.clone(),
                                max_turns: Some(max_turns),
                                usage,
                            });
                        }
                    }
                }
                ToolUseBehavior::RunLlmAgain => {
                    for (_tool, output, call_id) in &tool_results {
                        let out_item = tool_output_item(&starting_agent.name, call_id, output);
                        generated_items.push(out_item);
                    }
                }
            }

            // Append model outputs + tool outputs for next turn
            for item in &response.output {
                current_input_items.push(item.clone());
            }
            for (_tool, output, call_id) in &tool_results {
                // Avoid double-insert when StopAtTools already recorded outputs in generated_items
                // but still need them on the model input list.
                let already = generated_items.iter().any(|gi| {
                    matches!(gi, RunItem::ToolCallOutput(o) if o.raw_item.get("call_id").and_then(|c| c.as_str()) == Some(call_id.as_str()))
                });
                if !already {
                    let out_item = tool_output_item(&starting_agent.name, call_id, output);
                    generated_items.push(out_item);
                }
                current_input_items.push(ItemHelpers::function_call_output(
                    call_id,
                    value_to_tool_string(output),
                ));
            }

            if starting_agent.reset_tool_choice {
                model_settings.tool_choice = None;
            }
        }
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