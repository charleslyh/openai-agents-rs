//! One turn of the agent loop: the phases [`run_loop_inner`](super::agent_loop::run_loop_inner)
//! runs between two model calls.
//!
//! This module exists so the loop reads as a sequence of named phases instead of one 1100-line
//! body. Each function here is a *phase*, not a helper arbitrarily pulled out: it takes what the
//! loop has at that point and answers one question, and the loop stays the only place that decides
//! what happens next.
//!
//! [`perform_handoff`](handoff::perform_handoff) lives in its own file: it is the one phase that
//! replaces the running agent, and it is long enough to hide the rest.
//!
//! Two rules keep this honest:
//!
//! - Nothing here loops back into the agent loop or calls the model. A phase reports; it does not
//!   act on the report.
//! - A phase that can end the run returns [`TurnOutcome`] rather than calling
//!   [`finalize_run`](super::finalize::finalize_run) itself, because finalising needs nearly every
//!   piece of loop state. Passing them in again at the call site is cheaper than hiding that fact.

use serde_json::Value;

use crate::agent::Agent;
use std::sync::{Arc, Mutex};

use crate::error::{AgentsError, MaxTurnsExceeded, UserError};
use crate::items::{
    extract_message_text, is_function_call, is_reasoning, HandoffCallItem, InputLike, ItemHelpers,
    MessageOutputItem, ModelResponse, ReasoningItem, RunItem, ToolCallItem,
};
use crate::result::StreamingSnapshot;
use crate::run_context::RunContextWrapper;
use crate::stream_events::{RunItemStreamName, StreamEvent};
use crate::tool::FunctionTool;
use crate::tracing::SpanGuard;

use super::agent_loop::{emit, EventTx};
use super::config::{OutputGuardrailBlockedMessage, RunOptions};
use super::errors::{
    accept_handler_output, build_run_error_data, invoke_run_error_handler, RunHandledError,
};

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
            .map(|snap| {
                snap.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .cancel_after_turn
            })
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

mod handoff;
mod tools;

pub(crate) use handoff::{perform_handoff, HandoffTurn};
pub(crate) use tools::{
    execute_turn_tools, final_output_without_tools, prepare_turn, tool_use_behavior_output,
    InterruptSignal, ToolTurn, TurnSetup,
};
