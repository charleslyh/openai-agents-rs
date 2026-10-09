//! The function tools of one turn: resolving name collisions, planning which calls to make,
//! and executing them (Python: `run_internal/tool_planning.py` and `tool_execution.py`).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use serde_json::{json, Value};

use crate::agent::{Agent, AsToolConfig};
use crate::error::{AgentsError, ModelError, ToolTimeoutError, UserError};
use crate::handoffs::Handoff;
use crate::items::{
    required_function_call_parts, ItemHelpers, ResponseOutputItem, RunItem, ToolApprovalItem,
    ToolCallOutputItem,
};
use crate::lifecycle::RunHooks;
use crate::run_context::RunContextWrapper;
use crate::run_state::{ApprovalDecision, ApprovalStore, RunState};
use crate::stream_events::{RunItemStreamName, StreamEvent};
use crate::tool::{
    default_tool_timeout_error_message, FunctionTool, ToolContext, ToolResult, ToolTimeoutBehavior,
    DEFAULT_APPROVAL_REJECTION_MESSAGE,
};
use crate::tool_guardrails::{run_tool_input_guardrails, run_tool_output_guardrails};
use crate::tracing::function_span;

use super::agent_loop::{
    emit, take_nested_resume_state, EventTx, SharedToolGuardrailLog, NESTED_RESUME_STATES,
};
use super::config::{
    RunConfig, RunOptions, ToolErrorFormatterArgs, ToolErrorKind, ToolNameCollisionPolicy,
    ToolNotFoundBehavior,
};
use super::Runner;

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

/// Shape a tool result into the run's final output.
///
/// Python (`_maybe_finalize_from_tool_results`) coerces the tool output to `str` unless the
/// agent declared a non-plain-text `output_type`.
pub(crate) fn finalize_tool_output(agent: &Agent, output: &Value) -> Value {
    match agent.output_type.as_deref() {
        Some(schema) if !schema.is_plain_text() => output.clone(),
        _ => Value::String(value_to_tool_string(output)),
    }
}

/// Append (and stream) a tool output item unless one already exists for the call id.
pub(crate) async fn push_tool_output(
    events: &Option<EventTx>,
    generated_items: &mut Vec<RunItem>,
    agent_name: &str,
    call_id: &str,
    output: &Value,
) {
    if has_tool_output(generated_items, call_id) {
        return;
    }
    let out_item = tool_output_item(agent_name, call_id, output);
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

pub(crate) struct ToolPlan {
    /// Still waiting on human decisions.
    pub(crate) interruptions: Vec<ToolApprovalItem>,
    /// Rejected calls: (tool, rejection output, call_id).
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
    let Some(formatter) = &run_config.tool_error_formatter else {
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
    approvals: &ApprovalStore,
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
            if run_config.tool_not_found_behavior != ToolNotFoundBehavior::ReturnErrorToModel {
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

        match approvals.status(&agent.name, &name, &call_id) {
            Some(ApprovalDecision::Approved) => {
                plan.to_invoke.push((tool, arguments, call_id));
            }
            Some(ApprovalDecision::Rejected { message }) => {
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
            None => {
                if tool.needs_approval.requires(&arguments, &call_id).await {
                    plan.interruptions.push(ToolApprovalItem {
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

pub(crate) async fn execute_planned_tools(
    plan: &ToolPlan,
    nested_resume: &HashMap<String, RunState>,
    agent: &Agent,
    context: &RunContextWrapper,
    hooks: Option<&Arc<dyn RunHooks>>,
    max_concurrency: Option<usize>,
    guardrail_log: &SharedToolGuardrailLog,
) -> Result<
    (
        Vec<(FunctionTool, Value, String)>,
        Vec<ToolApprovalItem>,
        HashMap<String, RunState>,
    ),
    AgentsError,
> {
    let mut results: Vec<(FunctionTool, Value, String)> = Vec::new();
    let mut order: Vec<String> = Vec::new();
    let mut nested_interruptions = Vec::new();
    let mut nested_states = HashMap::new();

    for (tool, output, call_id) in &plan.already_done {
        order.push(call_id.clone());
        results.push((tool.clone(), output.clone(), call_id.clone()));
    }
    for (tool, output, call_id) in &plan.ready_outputs {
        order.push(call_id.clone());
        results.push((tool.clone(), output.clone(), call_id.clone()));
    }

    for (_, _, call_id) in &plan.to_invoke {
        order.push(call_id.clone());
    }

    let resume_map = nested_resume.clone();
    let hooks = hooks.cloned();
    let agent_hooks = agent.hooks.clone();
    let invoke_futs: Vec<_> = plan
        .to_invoke
        .iter()
        .map(|(tool, arguments, call_id)| {
            let tool = tool.clone();
            let arguments = arguments.clone();
            let call_id = call_id.clone();
            let resume_map = resume_map.clone();
            let agent = agent.clone();
            let guardrail_log = Arc::clone(guardrail_log);
            let context = context.clone();
            let hooks = hooks.clone();
            let agent_hooks = agent_hooks.clone();
            async move {
                let mut _fs = function_span(&tool.name);
                let ctx = ToolContext::new(
                    tool.name.clone(),
                    call_id.clone(),
                    arguments.clone(),
                    context.clone(),
                );
                let hook_ctx = ctx.clone();
                // Python: input guardrails run before the start hooks; a rejection replaces the
                // call, so neither the hooks, the body nor the output guardrails run.
                let guardrail_agent = Arc::new(agent.clone());
                let mut input_results = Vec::new();
                let rejection = run_tool_input_guardrails(
                    &tool.tool_input_guardrails,
                    &ctx,
                    &guardrail_agent,
                    &mut input_results,
                )
                .await;
                guardrail_log
                    .lock()
                    .expect("tool guardrail log")
                    .input
                    .extend(input_results);
                if let Some(message) = rejection? {
                    return Ok::<_, AgentsError>((
                        tool,
                        call_id,
                        ToolResult::output(Value::String(message)),
                    ));
                }
                if let Some(h) = &hooks {
                    h.on_tool_start(hook_ctx.clone(), &agent, &tool).await;
                }
                if let Some(ah) = &agent_hooks {
                    ah.on_tool_start(hook_ctx.clone(), &agent, &tool).await;
                }
                // Python (`failure_error_function`): a failing tool is reported to the model so it
                // can retry, instead of aborting the run.
                // A run the tool starts (an agent used as a tool) nests under this function span.
                let span_id = _fs.span().span_id.clone();
                let call = crate::tracing::with_current_span(
                    &span_id,
                    NESTED_RESUME_STATES.scope(RefCell::new(resume_map), async {
                        (tool.on_invoke_tool)(ctx, arguments).await
                    }),
                );
                // Python applies the timeout outside the failure handler, so `RaiseException`
                // fails the run instead of being reported to the model.
                let outcome = match tool.timeout_seconds {
                    None => call.await,
                    Some(seconds) => {
                        match tokio::time::timeout(Duration::from_secs_f64(seconds), call).await {
                            Ok(outcome) => outcome,
                            Err(_) => {
                                let timeout = ToolTimeoutError {
                                    tool_name: tool.name.clone(),
                                    timeout_seconds: seconds,
                                };
                                if tool.timeout_behavior == ToolTimeoutBehavior::RaiseException {
                                    return Err(timeout.into());
                                }
                                let message = match &tool.timeout_error_function {
                                    Some(format) => format(&context, &AgentsError::from(timeout)),
                                    None => default_tool_timeout_error_message(&tool.name, seconds),
                                };
                                Ok(ToolResult::output(Value::String(message)))
                            }
                        }
                    }
                };
                // Python (`failure_error_function`): a failing tool is reported to the model so it
                // can retry, instead of aborting the run.
                let result = match outcome {
                    Ok(result) => result,
                    Err(error) => {
                        // Python (`_build_handled_function_tool_error_handler`): `SpanError(message=
                        // "Error running tool")`. Only the tool name goes into `data`: the error text
                        // can hold user data and the sensitivity flag does not reach this scope.
                        _fs.set_error(crate::tracing::SpanError {
                            message: "Error running tool".to_string(),
                            data: Some(json!({ "tool_name": tool.name })),
                        });
                        let message = tool.failure_error_function.handle(&context, error)?;
                        ToolResult::output(Value::String(message))
                    }
                };
                let mut result = result;
                // Python: output guardrails see the result (including an error message) unless the
                // call paused for a nested approval.
                if result.interruptions.is_empty() {
                    let mut output_results = Vec::new();
                    let guarded = run_tool_output_guardrails(
                        &tool.tool_output_guardrails,
                        &hook_ctx,
                        &guardrail_agent,
                        result.output.clone().unwrap_or(Value::Null),
                        &mut output_results,
                    )
                    .await;
                    guardrail_log
                        .lock()
                        .expect("tool guardrail log")
                        .output
                        .extend(output_results);
                    result.output = Some(guarded?);
                }
                let output = result.output.clone().unwrap_or(Value::Null);
                if let Some(h) = &hooks {
                    h.on_tool_end(hook_ctx.clone(), &agent, &tool, &output)
                        .await;
                }
                if let Some(ah) = &agent_hooks {
                    ah.on_tool_end(hook_ctx, &agent, &tool, &output).await;
                }
                Ok::<_, AgentsError>((tool, call_id, result))
            }
        })
        .collect();
    // Python starts every call of the turn unless `max_function_tool_concurrency` is set; a slot
    // frees as soon as a call finishes, and the first failure cancels the rest.
    // The window is driven by hand over concrete futures: boxing them would need a `Send` proof
    // that the recursive agent-as-tool call chain makes impossible to infer.
    let limit = max_concurrency.unwrap_or(usize::MAX).max(1);
    let mut waiting = invoke_futs.into_iter().enumerate();
    let mut running = futures::stream::FuturesUnordered::new();
    let mut invoked = Vec::new();
    loop {
        while running.len() < limit {
            match waiting.next() {
                Some((index, fut)) => running.push(async move { (index, fut.await) }),
                None => break,
            }
        }
        match running.next().await {
            Some((index, Ok(done))) => invoked.push((index, done)),
            Some((_, Err(error))) => return Err(error),
            None => break,
        }
    }
    invoked.sort_by_key(|(index, _)| *index);
    for (_, (tool, call_id, result)) in invoked {
        if !result.interruptions.is_empty() {
            if let Some(state) = result.nested_state {
                nested_states.insert(call_id.clone(), *state);
            }
            nested_interruptions.extend(result.interruptions);
            continue;
        }
        let output = result.output.unwrap_or(Value::Null);
        results.push((tool, output, call_id));
    }

    let index: HashMap<&str, usize> = order
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    results.sort_by_key(|(_, _, id)| index.get(id.as_str()).copied().unwrap_or(usize::MAX));
    Ok((results, nested_interruptions, nested_states))
}

pub(crate) fn has_tool_output(items: &[RunItem], call_id: &str) -> bool {
    items.iter().any(|gi| {
        matches!(
            gi,
            RunItem::ToolCallOutput(o)
                if o.raw_item.get("call_id").and_then(|c| c.as_str()) == Some(call_id)
        )
    })
}

pub(crate) fn existing_tool_output(items: &[RunItem], call_id: &str) -> Option<Value> {
    items.iter().find_map(|gi| match gi {
        RunItem::ToolCallOutput(o)
            if o.raw_item.get("call_id").and_then(|c| c.as_str()) == Some(call_id) =>
        {
            Some(o.output.clone())
        }
        _ => None,
    })
}

/// Build an `Agent.as_tool` FunctionTool (Python: `Agent.as_tool`).
pub(crate) fn build_agent_as_tool(agent: &Agent, config: AsToolConfig) -> FunctionTool {
    let nested_agent = agent.clone();
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
    let needs = config.needs_approval;

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
            let nested_agent = nested_agent.clone();
            async move {
                let input_text = extract_as_tool_input(&args)?;
                let mut opts = RunOptions::default();
                if let Some(mt) = max_turns {
                    opts.max_turns = Some(mt);
                }

                let result = if let Some(state) = take_nested_resume_state(&ctx.tool_call_id) {
                    Runner::run_state(&nested_agent, state, opts).await?
                } else {
                    Runner::run(&nested_agent, input_text, opts).await?
                };

                if result.is_interrupted() {
                    let nested_state = result.to_state()?;
                    return Ok(ToolResult::interrupted(
                        result.interruptions.clone(),
                        nested_state,
                    ));
                }

                Ok(ToolResult::output(result.final_output))
            }
        },
    )
    .with_needs_approval(needs)
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

pub(crate) fn tool_output_item(agent_name: &str, call_id: &str, output: &Value) -> RunItem {
    let raw = ItemHelpers::function_call_output(call_id, value_to_tool_string(output));
    RunItem::ToolCallOutput(ToolCallOutputItem {
        agent_name: agent_name.to_string(),
        raw_item: raw,
        output: output.clone(),
    })
}

pub(crate) fn value_to_tool_string(output: &Value) -> String {
    match output {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}
