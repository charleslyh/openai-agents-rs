//! One turn of the agent loop: the phases [`run_loop_inner`](super::agent_loop::run_loop_inner)
//! runs between two model calls.
//!
//! This module exists so the loop reads as a sequence of named phases instead of one 1100-line
//! body. Each function here is a *phase*, not a helper arbitrarily pulled out: it takes what the
//! loop has at that point and answers one question, and the loop stays the only place that decides
//! what happens next.
//!
//! Two rules keep this honest:
//!
//! - Nothing here loops back into the agent loop or calls the model. A phase reports; it does not
//!   act on the report.
//! - A phase that can end the run returns [`TurnOutcome`] rather than calling
//!   [`finalize_run`](super::finalize::finalize_run) itself, because finalising needs nearly every
//!   piece of loop state. Passing them in again at the call site is cheaper than hiding that fact.

use serde_json::Value;

use crate::agent::{Agent, FunctionToolResult, ToolUseBehavior};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use crate::error::{AgentsError, MaxTurnsExceeded, ModelError, ModelRefusalError, UserError};
use crate::handoffs::{nest_handoff_history, Handoff, HandoffInputData};
use crate::items::{
    extract_message_text, is_function_call, is_reasoning, required_function_call_parts,
    HandoffCallItem, InputLike, ItemHelpers, MessageOutputItem, ModelResponse, ReasoningItem,
    RunItem, ToolApprovalItem, ToolCallItem,
};
use crate::result::StreamingSnapshot;
use crate::run_context::RunContextWrapper;
use crate::run_state::RunState;
use crate::stream_events::{RunItemStreamName, StreamEvent};
use crate::tool::FunctionTool;
use crate::tracing::SpanGuard;

use super::agent_loop::{emit, EventTx, SharedToolGuardrailLog};
use super::config::{OutputGuardrailBlockedMessage, RunOptions};
use super::errors::{
    accept_handler_output, build_run_error_data, invoke_run_error_handler, RunHandledError,
};
use super::tools::ToolPlan;

/// Rejects the option combinations Python rejects when it builds its `RunConfig`, and returns the
/// tool concurrency the loop needs.
///
/// Rust has no constructor to hook, so the run reports them as a [`UserError`] before doing any
/// work. Separate from the loop because it reads only `options` and returns early.
pub(crate) fn validate_run_options(options: &RunOptions) -> Result<Option<usize>, AgentsError> {
    let max_tool_concurrency = options
        .run_config
        .tool_execution
        .and_then(|c| c.max_function_tool_concurrency);
    if matches!(
        &options.run_config.output_guardrail_blocked_message,
        Some(OutputGuardrailBlockedMessage::Text(text)) if text.is_empty()
    ) {
        return Err(UserError::new("output_guardrail_blocked_message must be non-empty").into());
    }
    if max_tool_concurrency == Some(0) {
        return Err(UserError::new(
            "tool_execution.max_function_tool_concurrency must be at least 1",
        )
        .into());
    }
    Ok(max_tool_concurrency)
}

/// Splits a model response's output into the three kinds the loop handles, in Python's
/// `process_model_response` order: function calls, reasoning items, messages.
pub(crate) fn split_response_output(response: &ModelResponse) -> ResponseParts {
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
    ResponseParts {
        function_calls,
        reasoning_items,
        messages,
    }
}

/// A model response's output, split by kind. See [`split_response_output`].
pub(crate) struct ResponseParts {
    pub(crate) function_calls: Vec<Value>,
    pub(crate) reasoning_items: Vec<Value>,
    pub(crate) messages: Vec<Value>,
}

/// The text of a final answer, when the model answered without calling a tool.
///
/// Python joins every assistant message in the response, so a reply split across several message
/// items reads as one string.
pub(crate) fn final_text(messages: &[Value]) -> String {
    messages
        .iter()
        .filter_map(extract_message_text)
        .collect::<Vec<_>>()
        .join("")
}

/// Records the items a model response produced: reasoning, messages, then tool or handoff calls.
///
/// Skipped for a resumed turn, whose items were already recorded before the run was paused.
pub(crate) async fn record_response_items(
    parts: &ResponseParts,
    handoff_names: &[String],
    agent_name: &str,
    events: &Option<EventTx>,
    generated_items: &mut Vec<RunItem>,
) -> Result<(), AgentsError> {
    use crate::items::required_function_call_parts;

    for reason in &parts.reasoning_items {
        let item = RunItem::Reasoning(ReasoningItem {
            agent_name: agent_name.to_string(),
            raw_item: reason.clone(),
        });
        emit(
            events,
            StreamEvent::RunItem {
                name: RunItemStreamName::ReasoningItemCreated,
                item: item.clone(),
            },
        )
        .await;
        generated_items.push(item);
    }
    for msg in &parts.messages {
        let item = RunItem::Message(MessageOutputItem {
            agent_name: agent_name.to_string(),
            raw_item: msg.clone(),
        });
        emit(
            events,
            StreamEvent::RunItem {
                name: RunItemStreamName::MessageOutputCreated,
                item: item.clone(),
            },
        )
        .await;
        generated_items.push(item);
    }
    for call in &parts.function_calls {
        let (tool_name, _, _) = required_function_call_parts(call)?;
        // Python separates handoff tool calls into `HandoffCallItem` during
        // `process_model_response`, so `new_items` distinguishes them from plain tools.
        let is_handoff = handoff_names.iter().any(|n| n == &tool_name);
        let item = if is_handoff {
            RunItem::HandoffCall(HandoffCallItem {
                agent_name: agent_name.to_string(),
                raw_item: call.clone(),
            })
        } else {
            RunItem::ToolCall(ToolCallItem {
                agent_name: agent_name.to_string(),
                raw_item: call.clone(),
            })
        };
        emit(
            events,
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
    Ok(())
}

/// The run's final output when the agent's `tool_use_behavior` says one of this turn's tool
/// results ends the run, and `None` when the model should be asked again.
pub(crate) fn tool_use_behavior_output(
    behavior: &ToolUseBehavior,
    agent: &Agent,
    tool_results: &[(FunctionTool, Value, String)],
    context: &RunContextWrapper,
) -> Option<Value> {
    use super::tools::finalize_tool_output;

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

/// What the loop needs to end the run, borrowed for the duration of a budget check.
///
/// Bundled so the check stays within a sane argument count. The items themselves are passed
/// separately, because the `max_turns` handler appends to them while the error data still has to
/// describe the state as it was.
pub(crate) struct TurnEnd<'a> {
    pub(crate) input: &'a InputLike,
    pub(crate) raw_responses: &'a [ModelResponse],
    pub(crate) agent: &'a Agent,
    pub(crate) max_turns: Option<usize>,
    pub(crate) options: &'a RunOptions,
}

/// Whether `CancelMode::AfterTurn` asked the run to stop before this turn begins.
///
/// Python checks this between turns, so the run finishes the turn it is on; `turn > 1` keeps the
/// first turn from cancelling itself.
pub(crate) fn turn_was_cancelled(
    turn: usize,
    snapshot: Option<&Arc<Mutex<StreamingSnapshot>>>,
) -> bool {
    turn > 1
        && snapshot
            .map(|snap| snap.lock().expect("snapshot").cancel_after_turn)
            .unwrap_or(false)
}

/// Handles `max_turns` being exceeded: records the error on the current span, then lets a handler
/// supply the run's output.
///
/// Python (`run.py:1506-1507`) increments the turn counter first and only compares when `max_turns`
/// is not `None`, so an explicit `None` means "run until the agent stops". Returns `Ok(None)` when
/// the budget is fine; with no handler the error propagates, which is what Python does.
pub(crate) async fn handle_max_turns_exceeded(
    turn: usize,
    end: &TurnEnd<'_>,
    done_items: &[RunItem],
    generated_items: &mut Vec<RunItem>,
    context: &RunContextWrapper,
    task_guard: Option<&mut SpanGuard>,
) -> Result<Option<Value>, AgentsError> {
    let Some(limit) = end.max_turns.filter(|limit| turn > *limit) else {
        return Ok(None);
    };
    let error = MaxTurnsExceeded { max_turns: limit };
    // Python (`run.py:1508`): `SpanError(message="Max turns exceeded", data={"max_turns": n})`
    // on the current span, so a run that ran out of turns shows up on the timeline.
    if let Some(guard) = task_guard {
        guard.set_error(crate::tracing::SpanError {
            message: "Max turns exceeded".to_string(),
            data: Some(serde_json::json!({ "max_turns": limit })),
        });
    }
    // Python (`finalize_max_turns_handler_output`): validate the handler's output, record it as an
    // assistant message, then run the end hooks and output guardrails. Cloned because the handler
    // appends to `generated_items` and the error data must describe the state as it was.
    let run_data = build_run_error_data(
        end.input,
        done_items,
        end.raw_responses,
        end.agent,
        end.options.run_config.reasoning_item_id_policy,
    );
    let Some(handled) = invoke_run_error_handler(
        end.options.error_handlers.max_turns.clone(),
        RunHandledError::MaxTurns(error.clone()),
        context,
        run_data,
    )
    .await?
    else {
        return Err(error.into());
    };
    let output = accept_handler_output(end.agent, handled, generated_items)?;
    Ok(Some(output))
}

/// What a turn that called no tool produced.
pub(crate) enum TurnOutput {
    /// The run ends with this output.
    Final(Value),
    /// Ask the model again: an empty structured answer that no handler fixed.
    Again,
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
            options.run_config.reasoning_item_id_policy,
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

/// Performs the first handoff of the turn and switches the loop to its target agent.
///
/// Python (`execute_tools_and_side_effects`) performs only the first handoff; any extra one is
/// answered with a plain tool output and ignored. Returns `false` when `handoff_calls` is empty.
///
/// `agent` is replaced in place, and `current_input_items` is rewritten when the handoff's filter
/// or history nesting applies, so both are `&mut`.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn perform_handoff(
    handoff_calls: &[(Handoff, Value, String)],
    response: &ModelResponse,
    tool_results: &[(FunctionTool, Value, String)],
    agent: &mut Agent,
    known_agents: &mut HashMap<String, Agent>,
    current_input_items: &mut Vec<Value>,
    generated_items: &mut Vec<RunItem>,
    original_input_len: usize,
    events: &Option<EventTx>,
    options: &RunOptions,
    context: &RunContextWrapper,
) -> Result<bool, AgentsError> {
    if handoff_calls.is_empty() {
        return Ok(false);
    }
    let turn_start_len = current_input_items.len();
    for item in &response.output {
        current_input_items.push(crate::items::apply_reasoning_item_id_policy(
            item,
            options.run_config.reasoning_item_id_policy,
        ));
    }
    for (_tool, output, call_id) in tool_results {
        current_input_items.push(ItemHelpers::function_call_output(
            call_id,
            super::tools::value_to_tool_string(output),
        ));
    }
    // Python: every handoff after the first gets a plain tool output.
    for (_h, _call, call_id) in handoff_calls.iter().skip(1) {
        let message = Value::String(super::tools::MULTIPLE_HANDOFFS_MESSAGE.to_string());
        super::tools::push_tool_output(events, generated_items, &agent.name, call_id, &message)
            .await;
        current_input_items.push(ItemHelpers::function_call_output(
            call_id,
            super::tools::MULTIPLE_HANDOFFS_MESSAGE.to_string(),
        ));
    }

    let (h, call, call_id) = handoff_calls[0].clone();
    let _handoff_span = crate::tracing::handoff_span(&agent.name, &h.agent.name);

    // Python (`on_invoke_handoff`): validate the arguments, then run `on_handoff` before the
    // handoff output is recorded.
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

    let transfer = crate::support::pyjson::dumps(&serde_json::json!({"assistant": h.agent.name}));
    let source_agent_name = agent.name.clone();
    let target_agent_name = h.agent.name.clone();
    let out_item = RunItem::HandoffOutput(crate::items::HandoffOutputItem {
        agent_name: source_agent_name.clone(),
        raw_item: ItemHelpers::function_call_output(&call_id, transfer.clone()),
        source_agent_name,
        target_agent_name,
    });
    emit(
        events,
        StreamEvent::RunItem {
            name: RunItemStreamName::HandoffOccured,
            item: out_item.clone(),
        },
    )
    .await;
    generated_items.push(out_item);
    current_input_items.push(ItemHelpers::function_call_output(&call_id, transfer));

    let source_agent = agent.clone();
    *agent = if h.late_bound {
        known_agents.get(&h.agent.name).cloned().ok_or_else(|| {
            UserError::new(format!(
                "Handoff `{}` targets agent `{}`, which is not reachable from the starting \
                 agent. Add it with `handoff(agent)` somewhere in the graph.",
                h.tool_name, h.agent.name
            ))
        })?
    } else {
        (*h.agent).clone()
    };
    super::resolve::collect_agents(agent, known_agents);

    // Python: `hooks.on_handoff(context, from_agent, to_agent)` and the agent-level
    // `on_handoff(context, agent=new_agent, source=old_agent)`.
    if let Some(hooks) = &options.hooks {
        hooks
            .on_handoff(context.clone(), &source_agent, agent)
            .await;
    }
    if let Some(ah) = &source_agent.hooks {
        ah.on_handoff(context.clone(), agent, &source_agent).await;
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
            agent.name
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
            None => nest_handoff_history(data, options.run_config.handoff_history_mapper.as_ref()),
        };
        *current_input_items = next
            .input_history
            .into_iter()
            .chain(next.pre_handoff_items)
            .chain(next.new_items)
            .collect();
    }

    emit(
        events,
        StreamEvent::AgentUpdated {
            agent_name: agent.name.clone(),
        },
    )
    .await;
    if let Some(hooks) = &options.hooks {
        hooks.on_agent_start(context.clone(), agent).await;
    }
    if let Some(ah) = &agent.hooks {
        ah.on_start(context.clone(), agent).await;
    }
    Ok(true)
}

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
        .resolve(options.run_config.model_settings.as_ref());
    if agent.reset_tool_choice && agents_used_tools.contains(&agent.name) {
        model_settings.tool_choice = None;
    }
    // Python re-evaluates `is_enabled` every turn; disabled tools and handoffs are hidden from the
    // model and calls to them are treated as unknown.
    let (enabled_tools, enabled_handoffs) = super::tools::resolve_tool_name_collisions(
        agent.all_function_tools(context).await?,
        agent.enabled_handoffs(context).await,
        options.run_config.tool_name_collision_policy,
    )?;
    for tool in &enabled_tools {
        super::tools::validate_tool_timeout(tool)?;
    }
    Ok(TurnSetup {
        model_settings,
        tools: super::tools::tools_for_agent(enabled_tools, &enabled_handoffs),
        enabled_handoffs,
    })
}

/// Feeds a turn's model output and tool results back into the next call's input.
///
/// Python builds a `ToolCallOutputItem` for every completed tool result *before* deciding whether
/// one of them is the final output (`_build_tool_result_items`, then
/// `check_for_final_output_from_tools`), which is why this runs ahead of
/// [`tool_use_behavior_output`].
pub(crate) async fn record_turn_output(
    response: &ModelResponse,
    tool_results: &[(FunctionTool, Value, String)],
    agent_name: &str,
    events: &Option<EventTx>,
    generated_items: &mut Vec<RunItem>,
    current_input_items: &mut Vec<Value>,
    reasoning_item_id_policy: Option<crate::items::ReasoningItemIdPolicy>,
) {
    for item in &response.output {
        current_input_items.push(crate::items::apply_reasoning_item_id_policy(
            item,
            reasoning_item_id_policy,
        ));
    }
    for (_tool, output, call_id) in tool_results {
        if !super::tools::has_tool_output(generated_items, call_id) {
            let out_item = super::tools::tool_output_item(agent_name, call_id, output);
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
        current_input_items.push(ItemHelpers::function_call_output(
            call_id,
            super::tools::value_to_tool_string(output),
        ));
    }
}

/// Why the run has to pause and wait for a human decision.
pub(crate) struct InterruptSignal {
    /// The interruptions that paused the run.
    pub(crate) interruptions: Vec<ToolApprovalItem>,
    /// Nested run states to carry over, keyed by outer tool call id.
    pub(crate) nested_states: HashMap<String, RunState>,
}

/// Plans and executes this turn's function tools.
///
/// Python (`execute_tools_and_side_effects`) runs the function tools first and then performs the
/// first handoff. Three paths can pause the run for approval — an interrupted plan, its
/// approval-free subset, and a nested agent run — and all three end the run the same way, so this
/// reports which one fired and the loop builds the single `interrupted_result`.
///
/// Returns `Ok(None)` when the turn ran to completion.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn execute_turn_tools(
    plan: ToolPlan,
    agent: &Agent,
    context: &RunContextWrapper,
    options: &RunOptions,
    generated_items: &mut Vec<RunItem>,
    nested_agent_runs: &HashMap<String, RunState>,
    max_tool_concurrency: Option<usize>,
    guardrail_log: &SharedToolGuardrailLog,
    events: &Option<EventTx>,
) -> Result<Option<(Vec<(FunctionTool, Value, String)>, Option<InterruptSignal>)>, AgentsError> {
    if !plan.interruptions.is_empty() {
        // Execute tools that do not need approval in this turn before pausing.
        if !plan.to_invoke.is_empty() {
            let auto_plan = ToolPlan {
                interruptions: Vec::new(),
                ready_outputs: Vec::new(),
                to_invoke: plan.to_invoke.clone(),
                already_done: Vec::new(),
            };
            let (auto_results, nested_interruptions, nested_states) =
                super::tools::execute_planned_tools(
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
                let out_item = super::tools::tool_output_item(&agent.name, call_id, output);
                emit(
                    events,
                    StreamEvent::RunItem {
                        name: RunItemStreamName::ToolOutput,
                        item: out_item.clone(),
                    },
                )
                .await;
                if !super::tools::has_tool_output(generated_items, call_id) {
                    generated_items.push(out_item);
                }
            }
            if !nested_interruptions.is_empty() {
                return Ok(Some((
                    auto_results,
                    Some(InterruptSignal {
                        interruptions: nested_interruptions,
                        nested_states,
                    }),
                )));
            }
        }
        return Ok(Some((
            Vec::new(),
            Some(InterruptSignal {
                interruptions: plan.interruptions,
                nested_states: HashMap::new(),
            }),
        )));
    }

    // All calls decided — execute remaining tools (rejections already resolved to outputs).
    let (tool_results, nested_interruptions, nested_states) = super::tools::execute_planned_tools(
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
    Ok(Some((tool_results, signal)))
}
