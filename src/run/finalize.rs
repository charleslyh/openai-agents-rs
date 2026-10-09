//! Turn end: hooks, output guardrails and building the [`RunResult`].

use std::sync::Arc;

use serde_json::Value;

use crate::agent::Agent;
use crate::error::AgentsError;
use crate::error::OutputGuardrailTripwireTriggered;
use crate::guardrail::{InputGuardrailResult, OutputGuardrail};
use crate::items::{InputLike, ModelResponse, RunItem, ToolApprovalItem};
use crate::lifecycle::RunHooks;
use crate::result::{InterruptSnapshot, RunResult};
use crate::run_context::RunContextWrapper;
use crate::usage::Usage;

use super::config::{resolve_blocked_message, RunConfig};
use super::session_writer::SessionWriter;

/// Run `on_agent_end` hooks plus output guardrails, then build the final [`RunResult`].
///
/// Python runs output guardrails concurrently after the final output is known and raises
/// `OutputGuardrailTripwireTriggered` if any of them tripped.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn finalize_run(
    input: InputLike,
    new_items: Vec<RunItem>,
    raw_responses: Vec<ModelResponse>,
    final_output: Value,
    agent: &Agent,
    max_turns: Option<usize>,
    usage: Usage,
    context: &RunContextWrapper,
    hooks: Option<&Arc<dyn RunHooks>>,
    extra_output_guardrails: &[OutputGuardrail],
    input_guardrail_results: Vec<InputGuardrailResult>,
    tool_origin: Option<&RunConfig>,
    session_writer: Option<&mut SessionWriter>,
) -> Result<RunResult, AgentsError> {
    // Python closes the turn span before the run's end hooks and output guardrails.
    let _outside_turn = crate::tracing::leave_turn_span();
    if let Some(h) = hooks {
        h.on_agent_end(context.clone(), agent, &final_output).await;
    }
    if let Some(ah) = &agent.hooks {
        ah.on_end(context.clone(), agent, &final_output).await;
    }

    let guardrails: Vec<OutputGuardrail> = extra_output_guardrails
        .iter()
        .chain(agent.output_guardrails.iter())
        .cloned()
        .collect();
    let mut output_guardrail_results = Vec::new();
    if !guardrails.is_empty() {
        let agent = Arc::new(agent.clone());
        let futs = guardrails.iter().map(|g| {
            let g = g.clone();
            let agent = Arc::clone(&agent);
            let output = final_output.clone();
            let ctx = context.clone();
            async move { g.run(agent, output, ctx).await }
        });
        let results = futures::future::join_all(futs).await;
        let tripped = results
            .iter()
            .find(|r| r.output.tripwire_triggered)
            .cloned();
        output_guardrail_results = results;
        if let Some(mut result) = tripped {
            // Python (`blocked_output.py`): a final output that came from a tool is withheld, so
            // the error carries a data-free placeholder instead of the tool's output, and so
            // does the session.
            if let Some(run_config) = tool_origin {
                let placeholder =
                    resolve_blocked_message(run_config, &result.guardrail_name, &agent, context);
                if let Some(writer) = session_writer {
                    writer.save_blocked_turn(&new_items, &placeholder).await?;
                }
                result.agent_output = Value::String(placeholder);
                result.output.output_info = Value::Null;
            }
            return Err(OutputGuardrailTripwireTriggered { result }.into());
        }
    }

    let mut result = finished_result(
        input,
        new_items,
        raw_responses,
        final_output,
        Arc::new(agent.clone()),
        max_turns,
        usage,
    );
    result.input_guardrail_results = input_guardrail_results;
    result.output_guardrail_results = output_guardrail_results;
    Ok(result)
}

pub(crate) fn finished_result(
    input: InputLike,
    new_items: Vec<RunItem>,
    raw_responses: Vec<ModelResponse>,
    final_output: Value,
    last_agent: Arc<Agent>,
    max_turns: Option<usize>,
    usage: Usage,
) -> RunResult {
    RunResult {
        input,
        new_items,
        raw_responses,
        final_output,
        last_agent_name: last_agent.name.clone(),
        last_agent,
        max_turns,
        usage,
        interruptions: Vec::new(),
        input_guardrail_results: Vec::new(),
        output_guardrail_results: Vec::new(),
        tool_input_guardrail_results: Vec::new(),
        tool_output_guardrail_results: Vec::new(),
        reasoning_item_id_policy: None,
        interrupt_state: None,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn interrupted_result(
    input: InputLike,
    new_items: Vec<RunItem>,
    raw_responses: Vec<ModelResponse>,
    last_agent: Arc<Agent>,
    max_turns: Option<usize>,
    usage: Usage,
    interruptions: Vec<ToolApprovalItem>,
    interrupt_state: InterruptSnapshot,
) -> RunResult {
    RunResult {
        input,
        new_items,
        raw_responses,
        final_output: Value::Null,
        last_agent_name: last_agent.name.clone(),
        last_agent,
        max_turns,
        usage,
        interruptions,
        input_guardrail_results: interrupt_state.input_guardrail_results.clone(),
        output_guardrail_results: Vec::new(),
        tool_input_guardrail_results: Vec::new(),
        tool_output_guardrail_results: Vec::new(),
        reasoning_item_id_policy: None,
        interrupt_state: Some(interrupt_state),
    }
}
