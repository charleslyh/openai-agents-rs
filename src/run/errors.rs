//! Run error handlers: turning a run-ending error into a final output
//! (Python: `run_error_handlers.py` and the `error_handlers` runner options).

use std::sync::Arc;

use serde_json::{json, Value};

use crate::agent::Agent;
use crate::error::{AgentsError, MaxTurnsExceeded, ModelError, ModelRefusalError, UserError};
use crate::items::{apply_reasoning_item_id_policy, ReasoningItemIdPolicy};
use crate::items::{InputLike, ItemHelpers, MessageOutputItem, ModelResponse, RunItem};
use crate::model::wire_events::FAKE_RESPONSES_ID;
use crate::run_context::RunContextWrapper;

use super::tools::value_to_tool_string;

/// Snapshot of the run handed to a run error handler (Python: `RunErrorData`).
#[derive(Debug, Clone)]
pub struct RunErrorData {
    /// The run input.
    pub input: InputLike,
    /// Run items generated so far.
    pub new_items: Vec<RunItem>,
    /// The input followed by every model-visible generated item.
    pub history: Vec<Value>,
    /// The model-visible generated items.
    pub output: Vec<Value>,
    /// Raw model responses so far.
    pub raw_responses: Vec<ModelResponse>,
    /// The agent that was running.
    pub last_agent: Arc<Agent>,
}

/// The error a run error handler is asked to turn into a final output
/// (Python: `MaxTurnsExceeded | ModelRefusalError | ModelBehaviorError`).
#[derive(Debug, Clone)]
pub enum RunHandledError {
    /// `max_turns` was exceeded.
    MaxTurns(MaxTurnsExceeded),
    /// The model refused to answer.
    ModelRefusal(ModelRefusalError),
    /// The final message did not match the structured `output_type`.
    InvalidFinalOutput(ModelError),
}

impl RunHandledError {
    /// The error the run raises when no handler produces an output.
    pub fn into_error(self) -> AgentsError {
        match self {
            Self::MaxTurns(e) => e.into(),
            Self::ModelRefusal(e) => e.into(),
            Self::InvalidFinalOutput(e) => e.into(),
        }
    }
}

/// Input of a run error handler (Python: `RunErrorHandlerInput`).
#[derive(Clone)]
pub struct RunErrorHandlerInput {
    /// The error that stopped the run.
    pub error: RunHandledError,
    /// The run context.
    pub context: RunContextWrapper,
    /// The run so far.
    pub run_data: RunErrorData,
}

/// What a run error handler produces (Python: `RunErrorHandlerResult`).
#[derive(Debug, Clone)]
pub struct RunErrorHandlerResult {
    /// The final output to finish the run with.
    pub final_output: Value,
    /// Whether the synthesized assistant message joins the run's items (default true).
    pub include_in_history: bool,
}

impl RunErrorHandlerResult {
    /// Finish the run with `final_output`, recorded in the history.
    pub fn new(final_output: impl Into<Value>) -> Self {
        Self {
            final_output: final_output.into(),
            include_in_history: true,
        }
    }
}

/// Turns an error into a final output; `None` re-raises the error (Python: `RunErrorHandler`).
pub type RunErrorHandler = Arc<
    dyn Fn(
            RunErrorHandlerInput,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<Option<RunErrorHandlerResult>, AgentsError>>
                    + Send,
            >,
        > + Send
        + Sync,
>;

/// Error handlers keyed by error kind (Python: `RunErrorHandlers`).
#[derive(Clone, Default)]
pub struct RunErrorHandlers {
    /// Called when `max_turns` is exceeded.
    pub max_turns: Option<RunErrorHandler>,
    /// Called when the model refuses to answer.
    pub model_refusal: Option<RunErrorHandler>,
    /// Called when the final message does not match the structured `output_type`.
    pub invalid_final_output: Option<RunErrorHandler>,
}

impl RunErrorHandlers {
    /// Handle `MaxTurnsExceeded` with an async closure.
    pub fn on_max_turns<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunErrorHandlerInput) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Option<RunErrorHandlerResult>, AgentsError>>
            + Send
            + 'static,
    {
        self.max_turns = Some(Arc::new(move |input| Box::pin(f(input))));
        self
    }

    /// Handle `ModelRefusalError` with an async closure.
    pub fn on_model_refusal<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunErrorHandlerInput) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Option<RunErrorHandlerResult>, AgentsError>>
            + Send
            + 'static,
    {
        self.model_refusal = Some(Arc::new(move |input| Box::pin(f(input))));
        self
    }

    /// Handle an invalid structured final output with an async closure.
    pub fn on_invalid_final_output<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunErrorHandlerInput) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Option<RunErrorHandlerResult>, AgentsError>>
            + Send
            + 'static,
    {
        self.invalid_final_output = Some(Arc::new(move |input| Box::pin(f(input))));
        self
    }
}

/// Run `handler` (if any) for `error`; `None` means the error should be raised.
pub(crate) async fn invoke_run_error_handler(
    handler: Option<RunErrorHandler>,
    error: RunHandledError,
    context: &RunContextWrapper,
    run_data: RunErrorData,
) -> Result<Option<RunErrorHandlerResult>, AgentsError> {
    match handler {
        Some(handler) => {
            handler(RunErrorHandlerInput {
                error,
                context: context.clone(),
                run_data,
            })
            .await
        }
        None => Ok(None),
    }
}

/// Validate a handler's output against the agent's `output_type` and, unless the handler opted
/// out, record it as an assistant message (Python: `finalize_*_handler_output`).
pub(crate) fn accept_handler_output(
    agent: &Agent,
    handled: RunErrorHandlerResult,
    generated_items: &mut Vec<RunItem>,
) -> Result<Value, AgentsError> {
    let (final_output, text) = validate_handler_final_output(agent, handled.final_output)?;
    if handled.include_in_history {
        let mut message = ItemHelpers::text_message(text);
        message["id"] = Value::String(FAKE_RESPONSES_ID.to_string());
        generated_items.push(RunItem::Message(MessageOutputItem {
            agent_name: agent.name.clone(),
            raw_item: message,
        }));
    }
    Ok(final_output)
}

pub(crate) fn build_run_error_data(
    input: &InputLike,
    items: &[RunItem],
    raw_responses: &[ModelResponse],
    agent: &Agent,
    policy: Option<ReasoningItemIdPolicy>,
) -> RunErrorData {
    let output: Vec<Value> = items
        .iter()
        .filter(|i| i.is_model_input())
        .map(|i| apply_reasoning_item_id_policy(i.raw_item(), policy))
        .collect();
    let mut history = ItemHelpers::input_to_new_input_list(input);
    history.extend(output.iter().cloned());
    RunErrorData {
        input: input.clone(),
        new_items: items.to_vec(),
        history,
        output,
        raw_responses: raw_responses.to_vec(),
        last_agent: Arc::new(agent.clone()),
    }
}

/// Python (`validate_handler_final_output` + `format_final_output_text`): a structured agent's
/// handler output must validate against its schema. Returns the output and its message text.
pub(crate) fn validate_handler_final_output(
    agent: &Agent,
    output: Value,
) -> Result<(Value, String), AgentsError> {
    let invalid =
        || UserError::new("Invalid run error handler final_output for structured output.");
    let Some(schema) = agent.output_type.as_deref().filter(|s| !s.is_plain_text()) else {
        let text = value_to_tool_string(&output);
        return Ok((output, text));
    };
    // Python wraps non-object outputs under `response`; try the value as given, then wrapped.
    let wrapped = json!({ crate::agent_output::WRAPPER_DICT_KEY: output.clone() });
    for payload in [&output, &wrapped] {
        let text = serde_json::to_string(payload).map_err(|_| invalid())?;
        if let Ok(validated) = schema.validate_json(&text) {
            return Ok((validated, text));
        }
    }
    Err(invalid().into())
}
