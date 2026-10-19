//! Planning the function tool calls of one turn (Python: `run_internal/tool_planning.py`).

use std::collections::HashSet;

use serde_json::{json, Value};

use crate::agent::Agent;
use crate::error::{AgentsError, ModelError, UserError};
use crate::handoffs::Handoff;
use crate::items::{
    required_function_call_parts, ResponseOutputItem, RunItem, ToolInterruptionItem,
};
use crate::run_context::RunContextWrapper;
use crate::run_state::{ResolutionStore, ToolResolution};
use crate::tool::{FunctionTool, DEFAULT_APPROVAL_REJECTION_MESSAGE};

use crate::run::config::{
    RunConfig, ToolErrorFormatterArgs, ToolErrorKind, ToolNameCollisionPolicy, ToolNotFoundBehavior,
};

use super::output::existing_tool_output;

/// Name of the stand-in tool recorded for a call to a tool the agent does not have.
const TOOL_NOT_FOUND_PLACEHOLDER: &str = "__tool_not_found__";

/// Output sent for every handoff after the first in one turn (Python: same literal).
pub(crate) const MULTIPLE_HANDOFFS_MESSAGE: &str = "Multiple handoffs detected, ignoring this one.";

/// Python (`resolve_tool_name_collisions`): a name used twice is an error under
/// [`ToolNameCollisionPolicy::Error`]; otherwise a handoff beats a tool, and the last entry of the
/// winning kind is kept.
pub(crate) fn resolve_tool_name_collisions(
    tools: Vec<FunctionTool>,
    handoffs: Vec<Handoff>,
    policy: ToolNameCollisionPolicy,
) -> Result<(Vec<FunctionTool>, Vec<Handoff>), AgentsError> {
    let mut owners: Vec<(String, Vec<(bool, usize)>)> = Vec::new();
    let mut add = |name: &str, is_handoff: bool, index: usize| match owners
        .iter_mut()
        .find(|(n, _)| n == name)
    {
        Some((_, entries)) => entries.push((is_handoff, index)),
        None => owners.push((name.to_string(), vec![(is_handoff, index)])),
    };
    for (i, t) in tools.iter().enumerate() {
        add(&t.name, false, i);
    }
    for (i, h) in handoffs.iter().enumerate() {
        if !h.tool_name.is_empty() {
            add(&h.tool_name, true, i);
        }
    }

    let mut drop_tools = HashSet::new();
    let mut drop_handoffs = HashSet::new();
    for (name, entries) in owners.iter().filter(|(_, e)| e.len() > 1) {
        let handoff_count = entries.iter().filter(|(is_handoff, _)| *is_handoff).count();
        let message = if handoff_count == 0 {
            format!(
                "Ambiguous function tool configuration: the tool name `{name}` is used by \
                 multiple tools. Assign a unique name to every colliding function tool."
            )
        } else if handoff_count == entries.len() {
            format!(
                "Ambiguous handoff configuration: the handoff tool name `{name}` is used by \
                 multiple handoffs. Pass a unique tool name to each handoff."
            )
        } else {
            format!(
                "Ambiguous tool routing configuration: the tool name `{name}` is used by both \
                 a function tool and a handoff. Assign a unique name to every colliding \
                 function tool and handoff."
            )
        };
        if policy == ToolNameCollisionPolicy::Error {
            return Err(UserError::new(message).into());
        }
        ::tracing::warn!("{message}");
        let winner = entries
            .iter()
            .rev()
            .find(|(is_handoff, _)| *is_handoff)
            .or_else(|| entries.last())
            .copied()
            .expect("collision has entries");
        for entry in entries.iter().filter(|e| **e != winner) {
            if entry.0 {
                drop_handoffs.insert(entry.1);
            } else {
                drop_tools.insert(entry.1);
            }
        }
    }

    let tools = tools
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !drop_tools.contains(i))
        .map(|(_, t)| t)
        .collect();
    let handoffs = handoffs
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !drop_handoffs.contains(i))
        .map(|(_, h)| h)
        .collect();
    Ok((tools, handoffs))
}

/// Python (`_validate_function_tool_timeout_config`): finite and greater than zero.
pub(crate) fn validate_tool_timeout(tool: &FunctionTool) -> Result<(), AgentsError> {
    match tool.timeout_seconds {
        Some(seconds) if !seconds.is_finite() => {
            Err(UserError::new("FunctionTool timeout_seconds must be a finite number.").into())
        }
        Some(seconds) if seconds <= 0.0 => {
            Err(UserError::new("FunctionTool timeout_seconds must be greater than 0.").into())
        }
        _ => Ok(()),
    }
}

pub(crate) fn tools_for_agent(
    mut tools: Vec<FunctionTool>,
    handoffs: &[Handoff],
) -> Vec<FunctionTool> {
    for h in handoffs {
        tools.push(handoff_as_tool(h));
    }
    tools
}

pub(crate) fn handoff_as_tool(h: &Handoff) -> FunctionTool {
    FunctionTool::new(
        h.tool_name.clone(),
        h.tool_description.clone(),
        h.input_json_schema.clone().unwrap_or_else(|| {
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            })
        }),
        |_ctx, _args| async { Ok(Value::Null) },
    )
}

pub(crate) struct ToolPlan {
    /// Still waiting for a resolution.
    pub(crate) interruptions: Vec<ToolInterruptionItem>,
    /// Calls resolved without running: (tool, rejection message or provided output, call_id).
    pub(crate) ready_outputs: Vec<(FunctionTool, Value, String)>,
    /// Tools that should be invoked now: (tool, arguments, call_id).
    pub(crate) to_invoke: Vec<(FunctionTool, String, String)>,
    /// Already-done call ids (skip invoke, output already in generated_items).
    pub(crate) already_done: Vec<(FunctionTool, Value, String)>,
}

/// Run `RunConfig.tool_error_formatter`; `None` keeps `default_message`.
pub(crate) async fn format_tool_error(
    run_config: &RunConfig,
    context: &RunContextWrapper,
    kind: ToolErrorKind,
    tool_name: &str,
    call_id: &str,
    default_message: String,
) -> String {
    let Some(formatter) = &run_config.tools.error_formatter else {
        return default_message;
    };
    formatter(ToolErrorFormatterArgs {
        kind,
        tool_name: tool_name.to_string(),
        call_id: call_id.to_string(),
        default_message: default_message.clone(),
        run_context: context.clone(),
    })
    .await
    .unwrap_or(default_message)
}

pub(crate) async fn plan_tool_calls(
    agent: &Agent,
    tools: &[FunctionTool],
    function_calls: &[ResponseOutputItem],
    resolutions: &ResolutionStore,
    generated_items: &[RunItem],
    run_config: &RunConfig,
    context: &RunContextWrapper,
) -> Result<ToolPlan, AgentsError> {
    let mut plan = ToolPlan {
        interruptions: Vec::new(),
        ready_outputs: Vec::new(),
        to_invoke: Vec::new(),
        already_done: Vec::new(),
    };

    for call in function_calls {
        let (name, arguments, call_id) = required_function_call_parts(call)?;
        let Some(tool) = tools.iter().find(|t| t.name == name).cloned() else {
            // Python: `ModelBehaviorError("Tool X not found in agent Y")`, unless the run asks
            // for the error to be returned to the model.
            if run_config.tools.not_found_behavior != ToolNotFoundBehavior::ReturnErrorToModel {
                return Err(ModelError::Behavior(format!(
                    "Tool {name} not found in agent {}",
                    agent.name
                ))
                .into());
            }
            // The placeholder keeps the tool name out of `StopAtTools` matching.
            let placeholder = FunctionTool::constant(TOOL_NOT_FOUND_PLACEHOLDER, "", "");
            let message = format_tool_error(
                run_config,
                context,
                ToolErrorKind::ToolNotFound,
                &name,
                &call_id,
                format!("Tool '{name}' not found."),
            )
            .await;
            plan.ready_outputs
                .push((placeholder, Value::String(message), call_id));
            continue;
        };

        if let Some(existing) = existing_tool_output(generated_items, &call_id) {
            plan.already_done.push((tool, existing, call_id));
            continue;
        }

        match resolutions.status(&agent.name, &name, &call_id) {
            Some(ToolResolution::Run) => {
                plan.to_invoke.push((tool, arguments, call_id));
            }
            Some(ToolResolution::Reject { message }) => {
                // Python: an explicit rejection message wins; the formatter only replaces the
                // default one. The default is stored at reject time, so an explicit message
                // equal to it is treated as the default.
                let message = if message == DEFAULT_APPROVAL_REJECTION_MESSAGE {
                    format_tool_error(
                        run_config,
                        context,
                        ToolErrorKind::ApprovalRejected,
                        &name,
                        &call_id,
                        message,
                    )
                    .await
                } else {
                    message
                };
                plan.ready_outputs
                    .push((tool, Value::String(message), call_id));
            }
            Some(ToolResolution::Output { output }) => {
                plan.ready_outputs.push((tool, output, call_id));
            }
            None => {
                if tool.needs_resolution.requires(&arguments, &call_id).await {
                    plan.interruptions.push(ToolInterruptionItem {
                        agent_name: agent.name.clone(),
                        tool_name: name,
                        call_id,
                        arguments,
                        raw_item: call.clone(),
                    });
                } else {
                    plan.to_invoke.push((tool, arguments, call_id));
                }
            }
        }
    }
    Ok(plan)
}
