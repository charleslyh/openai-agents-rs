//! Turn end: hooks, output guardrails and building the [`RunResult`].

use std::sync::Arc;

use serde_json::Value;

use crate::agent::Agent;
use crate::error::AgentsError;
use crate::error::OutputGuardrailTripwireTriggered;
use crate::guardrail::{
    run_group, single_transform, InputGuardrailResult, OutputGuardrail, OutputGuardrailResult,
};
use crate::items::{InputLike, ModelResponse, RunItem, ToolInterruptionItem};
use crate::lifecycle::RunHooks;
use crate::result::{InterruptSnapshot, RunResult};
use crate::run_context::RunContextWrapper;
use crate::usage::Usage;

use super::config::{resolve_blocked_message, RunConfig};
use super::session_writer::SessionWriter;

/// What a run has accumulated by the time it ends, whether it finished or paused.
pub(crate) struct RunRecord {
    pub(crate) input: InputLike,
    pub(crate) new_items: Vec<RunItem>,
    pub(crate) raw_responses: Vec<ModelResponse>,
    pub(crate) max_turns: Option<usize>,
    pub(crate) usage: Usage,
}

/// The surroundings [`finalize_run`] consults while it ends the run.
pub(crate) struct FinalizeEnv<'a> {
    pub(crate) context: &'a RunContextWrapper,
    pub(crate) hooks: Option<&'a Arc<dyn RunHooks>>,
    pub(crate) extra_output_guardrails: &'a [OutputGuardrail],
    pub(crate) input_guardrail_results: Vec<InputGuardrailResult>,
    /// Set when the final output came from a tool, so a blocked output is withheld.
    pub(crate) tool_origin: Option<&'a RunConfig>,
    pub(crate) session_writer: Option<&'a mut SessionWriter>,
}

/// Run `on_agent_end` hooks plus output guardrails, then build the final [`RunResult`].
///
/// Python runs output guardrails concurrently after the final output is known and raises
/// `OutputGuardrailTripwireTriggered` if any of them tripped.
pub(crate) async fn finalize_run(
    record: RunRecord,
    mut final_output: Value,
    agent: &Arc<Agent>,
    env: FinalizeEnv<'_>,
) -> Result<RunResult, AgentsError> {
    let RunRecord {
        input,
        new_items,
        raw_responses,
        max_turns,
        usage,
    } = record;
    let FinalizeEnv {
        context,
        hooks,
        extra_output_guardrails,
        input_guardrail_results,
        tool_origin,
        session_writer,
    } = env;
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
        .chain(agent.all_output_guardrails())
        .cloned()
        .collect();
    let mut output_guardrail_results = Vec::new();
    if !guardrails.is_empty() {
        let futs = guardrails.iter().map(|g| {
            let g = g.clone();
            let agent = Arc::clone(agent);
            let output = final_output.clone();
            let ctx = context.clone();
            async move { g.run(agent, output, ctx).await }
        });
        let results = run_group(futs, |r: &OutputGuardrailResult| &r.output).await;
        let tripped = results
            .iter()
            .find(|r| r.output.tripwire_triggered())
            .cloned();
        // Rust-only (X-012): a transform replaces the final output the caller receives; the
        // model's message in `new_items` (and the session) is left as the model wrote it.
        let transform = if tripped.is_none() {
            single_transform(
                &results,
                |r: &OutputGuardrailResult| &r.output,
                |r: &OutputGuardrailResult| r.guardrail_name.as_str(),
            )?
            .map(|(_, content)| content.clone())
        } else {
            None
        };
        output_guardrail_results = results;
        if let Some(content) = transform {
            final_output = content;
        }
        if let Some(mut result) = tripped {
            // Python (`blocked_output.py`): a final output that came from a tool is withheld, so
            // the error carries a data-free placeholder instead of the tool's output, and so
            // does the session.
            if let Some(run_config) = tool_origin {
                let placeholder =
                    resolve_blocked_message(run_config, &result.guardrail_name, agent, context);
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
        RunRecord {
            input,
            new_items,
            raw_responses,
            max_turns,
            usage,
        },
        final_output,
        Arc::clone(agent),
    );
    result.input_guardrail_results = input_guardrail_results;
    result.output_guardrail_results = output_guardrail_results;
    Ok(result)
}

pub(crate) fn finished_result(
    record: RunRecord,
    final_output: Value,
    last_agent: Arc<Agent>,
) -> RunResult {
    let RunRecord {
        input,
        new_items,
        raw_responses,
        max_turns,
        usage,
    } = record;
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
        is_cancelled: false,
        interrupt_state: None,
    }
}

pub(crate) fn interrupted_result(
    record: RunRecord,
    last_agent: Arc<Agent>,
    interruptions: Vec<ToolInterruptionItem>,
    interrupt_state: InterruptSnapshot,
) -> RunResult {
    let RunRecord {
        input,
        new_items,
        raw_responses,
        max_turns,
        usage,
    } = record;
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
        is_cancelled: false,
        interrupt_state: Some(interrupt_state),
    }
}
