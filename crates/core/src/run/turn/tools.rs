//! The tools of a turn: preparing the tool set, running it, and reading its answer as the run's
//! final output when the agent says so.
//!
//! Split out of the other turn phases because all three are about tools, and together they are
//! long enough to hide the phases that are not.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::agent::{Agent, FunctionToolResult, ToolUseBehavior};
use crate::error::{AgentsError, ModelError, ModelRefusalError};
use crate::handoffs::Handoff;
use crate::items::{InputLike, ModelResponse, RunItem, ToolInterruptionItem};
use crate::run_context::RunContextWrapper;
use crate::run_state::RunState;
use crate::stream_events::{RunItemStreamName, StreamEvent};
use crate::tool::FunctionTool;

use super::super::agent_loop::{emit, EventTx, SharedToolGuardrailLog};
use super::super::config::RunOptions;
use super::super::errors::{
    accept_handler_output, build_run_error_data, invoke_run_error_handler, RunHandledError,
};
use super::{final_text, ResponseParts, TurnOutput};
use crate::run::tools::ToolPlan;

/// The model settings and tool set a turn runs with.
///
/// Python recomputes both at the start of every turn (`get_model_settings`, then
/// `maybe_reset_tool_choice`, then the `is_enabled` re-evaluation), so a tool disabled mid-run
/// disappears from the next call.
pub(crate) struct TurnSetup {
    pub(crate) model_settings: crate::model_settings::ModelSettings,
    pub(crate) tools: Vec<FunctionTool>,
    pub(crate) enabled_handoffs: Vec<Handoff>,
}

/// What a turn's tool phase needs to read, bundled so the phase takes three arguments.
#[derive(Clone, Copy)]
pub(crate) struct ToolTurn<'a> {
    pub(crate) agent: &'a Agent,
    pub(crate) context: &'a RunContextWrapper,
    pub(crate) options: &'a RunOptions,
    pub(crate) nested_agent_runs: &'a HashMap<String, RunState>,
    pub(crate) max_tool_concurrency: Option<usize>,
    pub(crate) guardrail_log: &'a SharedToolGuardrailLog,
    pub(crate) events: &'a Option<EventTx>,
}

/// Results of the tools that ran, plus the pause signal when a resolution is needed.
pub(crate) type TurnToolOutcome = (Vec<(FunctionTool, Value, String)>, Option<InterruptSignal>);

/// Why the run has to pause and wait for a human decision.
pub(crate) struct InterruptSignal {
    /// The interruptions that paused the run.
    pub(crate) interruptions: Vec<ToolInterruptionItem>,
    /// Nested run states to carry over, keyed by outer tool call id.
    pub(crate) nested_states: HashMap<String, RunState>,
}

/// The final output of a turn whose response contained no tool call, or the reason to ask again.
///
/// Covers the two ways such a turn can fail and still be rescued: a refusal
/// (`ModelRefusalError`) and a final message that does not match `output_type`
/// (`InvalidFinalOutput`). Both consult a handler before the run gives up, and both see only this
/// response as `raw_responses`, like Python (`raw_responses=[new_response]`).
pub(crate) async fn final_output_without_tools(
    parts: &ResponseParts,
    response: &ModelResponse,
    input: &InputLike,
    generated_items: &mut Vec<RunItem>,
    agent: &Agent,
    options: &RunOptions,
    context: &RunContextWrapper,
) -> Result<TurnOutput, AgentsError> {
    let last_message = &parts
        .messages
        .iter()
        .rev()
        .find(|m| m.get("type").and_then(Value::as_str) == Some("message"));
    let text = final_text(&parts.messages);
    let error_data = |generated: &[RunItem]| {
        build_run_error_data(
            input,
            generated,
            std::slice::from_ref(response),
            agent,
            options.run_config.model.reasoning_item_id_policy,
        )
    };
    // Python (`execute_tools_and_side_effects`): a refusal ends the turn with `ModelRefusalError`
    // unless a `model_refusal` handler supplies the output.
    let refusal = last_message.and_then(crate::items::extract_message_refusal);
    let output = if let Some(refusal) = refusal {
        let error = ModelRefusalError { refusal };
        let handled = invoke_run_error_handler(
            options.error_handlers.model_refusal.clone(),
            RunHandledError::ModelRefusal(error.clone()),
            context,
            error_data(generated_items),
        )
        .await?;
        let Some(handled) = handled else {
            return Err(error.into());
        };
        accept_handler_output(agent, handled, generated_items)?
    } else {
        match agent.output_type.as_deref() {
            Some(schema) if !schema.is_plain_text() => {
                let invalid = if text.is_empty() {
                    ModelError::Behavior(
                        "Model returned no final output for the structured output type.".into(),
                    )
                } else {
                    match schema.validate_json(&text) {
                        Ok(value) => return Ok(TurnOutput::Final(value)),
                        Err(error) => error,
                    }
                };
                let handled = invoke_run_error_handler(
                    options.error_handlers.invalid_final_output.clone(),
                    RunHandledError::InvalidFinalOutput(invalid.clone()),
                    context,
                    error_data(generated_items),
                )
                .await?;
                match handled {
                    Some(handled) => accept_handler_output(agent, handled, generated_items)?,
                    // Python: an empty structured answer that no handler fixes asks the model
                    // again; an unparsable one raises.
                    None if text.is_empty() => return Ok(TurnOutput::Again),
                    None => return Err(invalid.into()),
                }
            }
            _ => Value::String(text),
        }
    };
    Ok(TurnOutput::Final(output))
}

/// Resolves the model settings and the tool set for one turn.
pub(crate) async fn prepare_turn(
    agent: &Agent,
    agents_used_tools: &HashSet<String>,
    options: &RunOptions,
    context: &RunContextWrapper,
) -> Result<TurnSetup, AgentsError> {
    // Python recomputes model settings at the start of every turn (`get_model_settings` then
    // `maybe_reset_tool_choice`), so `tool_choice` is cleared once the agent has used tools —
    // including when the previous turn ended early via `stop_on_first_tool` / `StopAtTools`.
    let mut model_settings = agent
        .model_settings
        .resolve(options.run_config.model.settings.as_ref());
    if agent.reset_tool_choice && agents_used_tools.contains(&agent.name) {
        model_settings.tool_choice = None;
    }
    // Python re-evaluates `is_enabled` every turn; disabled tools and handoffs are hidden from the
    // model and calls to them are treated as unknown.
    let (enabled_tools, enabled_handoffs) = crate::run::tools::resolve_tool_name_collisions(
        agent.all_function_tools(context).await?,
        agent.enabled_handoffs(context).await,
        options.run_config.tools.name_collision_policy,
    )?;
    for tool in &enabled_tools {
        crate::run::tools::validate_tool_timeout(tool)?;
    }
    Ok(TurnSetup {
        model_settings,
        tools: crate::run::tools::tools_for_agent(enabled_tools, &enabled_handoffs),
        enabled_handoffs,
    })
}

/// Plans and executes this turn's function tools.
///
/// Python (`execute_tools_and_side_effects`) runs the function tools first and then performs the
/// first handoff. Three paths can pause the run for a resolution — an interrupted plan, its
/// resolution-free subset, and a nested agent run — and all three end the run the same way, so this
/// reports which one fired and the loop builds the single `interrupted_result`.
///
/// Returns the tool results and, when the run has to pause, the signal describing why.
pub(crate) async fn execute_turn_tools(
    plan: ToolPlan,
    env: ToolTurn<'_>,
    generated_items: &mut Vec<RunItem>,
) -> Result<TurnToolOutcome, AgentsError> {
    let ToolTurn {
        agent,
        context,
        options,
        nested_agent_runs,
        max_tool_concurrency,
        guardrail_log,
        events,
    } = env;
    if !plan.interruptions.is_empty() {
        // Execute tools that do not need a resolution in this turn before pausing.
        if !plan.to_invoke.is_empty() {
            let auto_plan = ToolPlan {
                interruptions: Vec::new(),
                ready_outputs: Vec::new(),
                to_invoke: plan.to_invoke.clone(),
                already_done: Vec::new(),
            };
            let (auto_results, nested_interruptions, nested_states) =
                crate::run::tools::execute_planned_tools(
                    &auto_plan,
                    nested_agent_runs,
                    agent,
                    context,
                    options.hooks.as_ref(),
                    max_tool_concurrency,
                    guardrail_log,
                )
                .await?;
            for (_tool, output, call_id) in &auto_results {
                let out_item = crate::run::tools::tool_output_item(&agent.name, call_id, output);
                emit(
                    events,
                    StreamEvent::RunItem {
                        name: RunItemStreamName::ToolOutput,
                        item: out_item.clone(),
                    },
                )
                .await;
                if !crate::run::tools::has_tool_output(generated_items, call_id) {
                    generated_items.push(out_item);
                }
            }
            if !nested_interruptions.is_empty() {
                return Ok((
                    auto_results,
                    Some(InterruptSignal {
                        interruptions: nested_interruptions,
                        nested_states,
                    }),
                ));
            }
        }
        return Ok((
            Vec::new(),
            Some(InterruptSignal {
                interruptions: plan.interruptions,
                nested_states: HashMap::new(),
            }),
        ));
    }

    // All calls decided — execute remaining tools (rejections already resolved to outputs).
    let (tool_results, nested_interruptions, nested_states) =
        crate::run::tools::execute_planned_tools(
            &plan,
            nested_agent_runs,
            agent,
            context,
            options.hooks.as_ref(),
            max_tool_concurrency,
            guardrail_log,
        )
        .await?;
    let signal = (!nested_interruptions.is_empty()).then(|| InterruptSignal {
        interruptions: nested_interruptions,
        nested_states,
    });
    Ok((tool_results, signal))
}

/// The run's final output when the agent's `tool_use_behavior` says one of this turn's tool
/// results ends the run, and `None` when the model should be asked again.
pub(crate) fn tool_use_behavior_output(
    behavior: &ToolUseBehavior,
    agent: &Agent,
    tool_results: &[(FunctionTool, Value, String)],
    context: &RunContextWrapper,
) -> Option<Value> {
    use crate::run::tools::finalize_tool_output;

    match behavior {
        ToolUseBehavior::StopOnFirstTool => tool_results
            .first()
            .map(|(_tool, output, _call_id)| finalize_tool_output(agent, output)),
        ToolUseBehavior::StopAtTools { stop_at_tool_names } => {
            tool_results.iter().find_map(|(tool, output, _call_id)| {
                stop_at_tool_names
                    .iter()
                    .any(|n| n == &tool.name)
                    .then(|| finalize_tool_output(agent, output))
            })
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
            let decision = decide(context, &results);
            decision
                .is_final_output
                .then(|| finalize_tool_output(agent, &decision.final_output.unwrap_or(Value::Null)))
        }
        ToolUseBehavior::RunLlmAgain => None,
    }
}
