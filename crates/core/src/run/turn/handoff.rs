//! Performing a handoff: switching the run to another agent.
//!
//! Kept apart from the other turn phases because it is the only one that replaces the running
//! agent and rewrites the conversation, and it is long enough to hide the rest.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use crate::agent::Agent;
use crate::error::{AgentsError, ModelError, UserError};
use crate::handoffs::{nest_handoff_history, Handoff, HandoffInputData};
use crate::items::{required_function_call_parts, ItemHelpers, ModelResponse, RunItem};
use crate::run_context::RunContextWrapper;
use crate::stream_events::{RunItemStreamName, StreamEvent};
use crate::tool::FunctionTool;

use crate::run::agent_loop::{emit, EventTx};
use crate::run::config::RunOptions;
use crate::run::resolve::collect_agents;
use crate::run::tools::{push_tool_output, value_to_tool_string, MULTIPLE_HANDOFFS_MESSAGE};
use crate::tracing::handoff_span;

/// What a handoff reads from the turn that requested it.
#[derive(Clone, Copy)]
pub(crate) struct HandoffTurn<'a> {
    pub(crate) handoff_calls: &'a [(Handoff, Value, String)],
    pub(crate) response: &'a ModelResponse,
    pub(crate) tool_results: &'a [(FunctionTool, Value, String)],
    pub(crate) original_input_len: usize,
    pub(crate) events: &'a Option<EventTx>,
    pub(crate) options: &'a RunOptions,
    pub(crate) context: &'a RunContextWrapper,
}

/// Performs the first handoff of the turn and switches the loop to its target agent.
///
/// Python (`execute_tools_and_side_effects`) performs only the first handoff; any extra one is
/// answered with a plain tool output and ignored. Returns `false` when `handoff_calls` is empty.
///
/// `agent` is replaced in place, and `current_input_items` is rewritten when the handoff's filter
/// or history nesting applies, so both are `&mut`.
pub(crate) async fn perform_handoff(
    turn: HandoffTurn<'_>,
    agent: &mut Arc<Agent>,
    known_agents: &mut HashMap<String, Arc<Agent>>,
    current_input_items: &mut Vec<Value>,
    generated_items: &mut Vec<RunItem>,
) -> Result<bool, AgentsError> {
    let HandoffTurn {
        handoff_calls,
        response,
        tool_results,
        original_input_len,
        events,
        options,
        context,
    } = turn;
    if handoff_calls.is_empty() {
        return Ok(false);
    }
    let turn_start_len = current_input_items.len();
    for item in &response.output {
        current_input_items.push(crate::items::apply_reasoning_item_id_policy(
            item,
            options.run_config.model.reasoning_item_id_policy,
        ));
    }
    for (_tool, output, call_id) in tool_results {
        current_input_items.push(ItemHelpers::function_call_output(
            call_id,
            value_to_tool_string(output),
        ));
    }
    // Python: every handoff after the first gets a plain tool output.
    for (_h, _call, call_id) in handoff_calls.iter().skip(1) {
        let message = Value::String(MULTIPLE_HANDOFFS_MESSAGE.to_string());
        push_tool_output(events, generated_items, &agent.name, call_id, &message).await;
        current_input_items.push(ItemHelpers::function_call_output(
            call_id,
            MULTIPLE_HANDOFFS_MESSAGE.to_string(),
        ));
    }

    let (h, call, call_id) = handoff_calls[0].clone();
    let _handoff_span = handoff_span(&agent.name, &h.agent.name);

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

    // Only the `Arc`s move: the previous agent stays alive for the hooks below.
    let target = if h.late_bound {
        known_agents.get(&h.agent.name).cloned().ok_or_else(|| {
            UserError::new(format!(
                "Handoff `{}` targets agent `{}`, which is not reachable from the starting \
                 agent. Add it with `handoff(agent)` somewhere in the graph.",
                h.tool_name, h.agent.name
            ))
        })?
    } else {
        Arc::clone(&h.agent)
    };
    let source_agent = std::mem::replace(agent, target);
    collect_agents(agent, known_agents);

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
        .or_else(|| options.run_config.handoff.input_filter.clone());
    let server_managed =
        options.previous_response_id.is_some() || options.conversation_id.is_some();
    let mut should_nest = h
        .nest_handoff_history
        .unwrap_or(options.run_config.handoff.nest_history);
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
            None => nest_handoff_history(data, options.run_config.handoff.history_mapper.as_ref()),
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
