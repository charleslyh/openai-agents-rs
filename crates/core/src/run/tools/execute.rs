//! Executing the planned tool calls (Python: `run_internal/tool_execution.py`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use serde_json::{json, Value};

use crate::agent::Agent;
use crate::error::{AgentsError, ToolTimeoutError};
use crate::items::ToolInterruptionItem;
use crate::lifecycle::RunHooks;
use crate::run_context::RunContextWrapper;
use crate::run_state::RunState;
use crate::tool::{
    default_tool_timeout_error_message, FunctionTool, ToolContext, ToolOutput, ToolTimeoutBehavior,
};
use crate::tool_guardrails::{run_tool_input_guardrails, run_tool_output_guardrails};
use crate::tracing::function_span;

use crate::run::agent_loop::{SharedToolGuardrailLog, NESTED_RESUME_STATES};
use crate::run::panic;

use super::plan::ToolPlan;

/// The output of a call the run was cancelled before starting.
pub(crate) const CANCELLED_BEFORE_START: &str =
    "The run was cancelled before this tool call started, so it did not run.";

pub(crate) async fn execute_planned_tools(
    plan: &ToolPlan,
    nested_resume: &HashMap<String, RunState>,
    agent: &Arc<Agent>,
    context: &RunContextWrapper,
    hooks: Option<&Arc<dyn RunHooks>>,
    max_concurrency: Option<usize>,
    guardrail_log: &SharedToolGuardrailLog,
) -> Result<
    (
        Vec<(FunctionTool, Value, String)>,
        Vec<ToolInterruptionItem>,
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
            let agent = Arc::clone(agent);
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
                let mut input_results = Vec::new();
                let rejection = panic::guard(
                    "a tool input guardrail",
                    run_tool_input_guardrails(
                        &tool.tool_input_guardrails,
                        &ctx,
                        &agent,
                        &mut input_results,
                    ),
                )
                .await
                .and_then(|checked| Ok(checked?));
                guardrail_log
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .input
                    .extend(input_results);
                if let Some(message) = rejection? {
                    return Ok::<_, AgentsError>((
                        tool,
                        call_id,
                        ToolOutput::output(Value::String(message)),
                    ));
                }
                if let Some(h) = &hooks {
                    panic::guard(
                        "RunHooks::on_tool_start",
                        h.on_tool_start(hook_ctx.clone(), &agent, &tool),
                    )
                    .await?;
                }
                if let Some(ah) = &agent_hooks {
                    panic::guard(
                        "AgentHooks::on_tool_start",
                        ah.on_tool_start(hook_ctx.clone(), &agent, &tool),
                    )
                    .await?;
                }
                // Python (`failure_error_function`): a failing tool is reported to the model so it
                // can retry, instead of aborting the run.
                // A run the tool starts (an agent used as a tool) nests under this function span.
                let span_id = _fs.span().span_id.clone();
                let call = crate::tracing::with_current_span(
                    &span_id,
                    NESTED_RESUME_STATES.scope(RefCell::new(resume_map), async {
                        // A panic in the tool body is a failed call, like an `Err`: the model is
                        // told and can retry. The text it gets is fixed; the panic message goes to
                        // the log only, since it may carry internals the model should not see.
                        match panic::catch((tool.on_invoke_tool)(ctx, arguments)).await {
                            Ok(outcome) => outcome,
                            Err(message) => {
                                tracing::error!(tool = %tool.name, %message, "tool panicked");
                                Err(AgentsError::tool("the tool panicked"))
                            }
                        }
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
                                Ok(ToolOutput::output(Value::String(message)))
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
                        ToolOutput::output(Value::String(message))
                    }
                };
                let mut result = result;
                // Python: output guardrails see the result (including an error message) unless the
                // call paused for a nested resolution.
                if result.interruptions.is_empty() {
                    let mut output_results = Vec::new();
                    let guarded = panic::guard(
                        "a tool output guardrail",
                        run_tool_output_guardrails(
                            &tool.tool_output_guardrails,
                            &hook_ctx,
                            &agent,
                            result.output.clone().unwrap_or(Value::Null),
                            &mut output_results,
                        ),
                    )
                    .await
                    .and_then(|checked| Ok(checked?));
                    guardrail_log
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .output
                        .extend(output_results);
                    result.output = Some(guarded?);
                }
                let output = result.output.clone().unwrap_or(Value::Null);
                if let Some(h) = &hooks {
                    panic::guard(
                        "RunHooks::on_tool_end",
                        h.on_tool_end(hook_ctx.clone(), &agent, &tool, &output),
                    )
                    .await?;
                }
                if let Some(ah) = &agent_hooks {
                    panic::guard(
                        "AgentHooks::on_tool_end",
                        ah.on_tool_end(hook_ctx, &agent, &tool, &output),
                    )
                    .await?;
                }
                Ok::<_, AgentsError>((tool, call_id, result))
            }
        })
        .collect();
    // Python starts every call of the turn unless `max_function_tool_concurrency` is set; a slot
    // frees as soon as a call finishes, and the first failure cancels the rest.
    // The window is driven by hand over concrete futures: boxing them would need a `Send` proof
    // that the recursive agent-as-tool call chain makes impossible to infer.
    //
    // Cancellation is cooperative. Once the run's token fires, calls that have not started are
    // not started (each gets a "cancelled" output so every call still has one), and calls that
    // are running are awaited to the end: dropping them would lose their last output and skip
    // the cleanup they do on their own token. The token is raced against the wait for the next
    // call, so a cancel that arrives while every slot is busy is seen at once instead of after
    // the next call happens to finish.
    let limit = max_concurrency.unwrap_or(usize::MAX).max(1);
    let token = context.cancellation().clone();
    let mut waiting = invoke_futs.into_iter().enumerate();
    let mut running = futures::stream::FuturesUnordered::new();
    let mut invoked = Vec::new();
    let mut not_started = Vec::new();
    let mut cancelled = false;
    loop {
        cancelled = cancelled || token.is_cancelled();
        if cancelled {
            not_started.extend(waiting.by_ref().map(|(index, _)| index));
        } else {
            while running.len() < limit {
                match waiting.next() {
                    Some((index, fut)) => running.push(async move { (index, fut.await) }),
                    None => break,
                }
            }
        }
        let next = if cancelled {
            running.next().await
        } else {
            tokio::select! {
                biased;
                () = token.cancelled() => continue,
                next = running.next() => next,
            }
        };
        match next {
            Some((index, Ok(done))) => invoked.push((index, done)),
            Some((_, Err(error))) => return Err(error),
            None => break,
        }
    }
    for index in not_started {
        let (tool, _, call_id) = &plan.to_invoke[index];
        results.push((
            tool.clone(),
            Value::String(CANCELLED_BEFORE_START.to_owned()),
            call_id.clone(),
        ));
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
