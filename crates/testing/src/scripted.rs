//! Scripted model for deterministic tests (Python: `agents.testing.ScriptedModel`).
//!
//! Mirrors `agents/testing/model.py`: steps are consumed in order, every call is recorded as a
//! [`ModelCall`], and script misuse raises a typed [`ModelScriptError`].

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;

use opai_agents_core::error::ModelError;
use opai_agents_core::items::{ModelResponse, ResponseOutputItem};
use opai_agents_core::model::{replay_response, Model, ModelRequest, ModelTracing};
use opai_agents_core::model_settings::ModelSettings;
use opai_agents_core::retry::{ModelRetryAdvice, ModelRetryAdviceRequest};
use opai_agents_core::usage::Usage;

/// Response id used when a step declares none (Python: `"scripted-response"`).
const SCRIPTED_RESPONSE_ID: &str = "scripted-response";

/// Model name reported in the `response` object (Python: `"scripted-model"`).
const SCRIPTED_MODEL: &str = "scripted-model";

/// Why a scripted step was rejected (Python: `ModelStepReason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStepReason {
    /// The step payload is not shaped correctly.
    InvalidInput,
    /// The step sets an unsupported field.
    UnsupportedField,
    /// The step error is not a valid error value.
    InvalidError,
    /// The step combines mutually exclusive outcomes.
    ConflictingOutcomes,
}

impl ModelStepReason {
    /// Wire name used in error messages, matching Python's literals.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidInput => "invalid_input",
            Self::UnsupportedField => "unsupported_field",
            Self::InvalidError => "invalid_error",
            Self::ConflictingOutcomes => "conflicting_outcomes",
        }
    }
}

/// Base error for an invalid or incompletely consumed model script
/// (Python: `ModelScriptError` and its subclasses).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelScriptError {
    /// A step was invalid before it entered the script queue.
    InvalidModelStep {
        /// Human readable detail.
        message: String,
        /// Why the step was rejected.
        reason: ModelStepReason,
        /// Index of the offending step in the input sequence.
        input_index: usize,
    },
    /// The model was called after every configured step was consumed.
    UnexpectedModelCall {
        /// Human readable detail.
        message: String,
        /// Index of the call that had no step.
        call_index: usize,
    },
    /// A test finished before consuming every configured step.
    UnconsumedModelSteps {
        /// Human readable detail.
        message: String,
        /// Steps left in the queue.
        remaining_steps: usize,
    },
}

impl std::fmt::Display for ModelScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidModelStep { message, .. } => write!(f, "{message}"),
            Self::UnexpectedModelCall { message, .. } => write!(f, "{message}"),
            Self::UnconsumedModelSteps { message, .. } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ModelScriptError {}

/// A recorded call at the `Model` boundary (Python: `ModelCall`).
#[derive(Debug, Clone)]
pub struct ModelCall {
    /// System instructions.
    pub system_instructions: Option<String>,
    /// Input snapshot.
    pub input: Value,
    /// Model settings snapshot.
    pub model_settings: ModelSettings,
    /// Tool names presented to the model.
    pub tool_names: Vec<String>,
    /// Tracing mode.
    pub tracing: ModelTracing,
    /// Previous response id.
    pub previous_response_id: Option<String>,
    /// Conversation id.
    pub conversation_id: Option<String>,
    /// Whether the call was made through `stream_response`.
    pub streamed: bool,
    /// Name of the structured output schema in effect, if any.
    pub output_schema_name: Option<String>,
}

/// A way a step can misbehave instead of answering (Rust-only; Python has no equivalent).
///
/// For testing how the code around a model copes with a model that crashes or never answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepFault {
    /// The model call panics with this message.
    Panic(String),
    /// The model call never returns; only dropping or cancelling the call ends it.
    Hang,
}

/// One deterministic step (Python: `ModelStep`).
#[derive(Debug, Clone)]
pub struct ModelStep {
    /// Output items for this call.
    pub output: Vec<ResponseOutputItem>,
    /// Usage for this call.
    pub usage: Usage,
    /// Response id.
    pub response_id: Option<String>,
    /// Transport request id when available.
    pub request_id: Option<String>,
    /// If set, the model call fails with this message.
    pub error: Option<String>,
    /// If set, the model call fails with this typed error (for example an HTTP status).
    pub model_error: Option<ModelError>,
    /// Provider retry guidance attached to the error this step raises
    /// (Python: `ModelStep.retry_advice`).
    pub retry_advice: Option<ModelRetryAdvice>,
    /// If set, the call panics or never returns instead of answering (see [`StepFault`]).
    pub fault: Option<StepFault>,
}

impl ModelStep {
    /// Create a successful step from output items (Python: `ModelStep(output=...)`).
    pub fn output(items: impl IntoIterator<Item = ResponseOutputItem>) -> Self {
        Self {
            output: items.into_iter().collect(),
            usage: Usage::default(),
            response_id: Some("resp-789".into()),
            request_id: None,
            error: None,
            model_error: None,
            retry_advice: None,
            fault: None,
        }
    }

    /// Create a step whose model call panics with `message`.
    pub fn panic(message: impl Into<String>) -> Self {
        Self {
            fault: Some(StepFault::Panic(message.into())),
            ..Self::raise_error("").without_message()
        }
    }

    /// Create a step whose model call never returns.
    pub fn hang() -> Self {
        Self {
            fault: Some(StepFault::Hang),
            ..Self::raise_error("").without_message()
        }
    }

    /// Create a step that raises an error (Python: `ModelStep.raise_error`).
    pub fn raise_error(message: impl Into<String>) -> Self {
        Self {
            output: vec![],
            usage: Usage::default(),
            response_id: None,
            request_id: None,
            error: Some(message.into()),
            model_error: None,
            retry_advice: None,
            fault: None,
        }
    }

    /// Create a step that fails with a typed [`ModelError`], e.g. a 429 or a dropped connection.
    pub fn raise_model_error(error: ModelError) -> Self {
        Self {
            model_error: Some(error),
            ..Self::raise_error("")
        }
        .without_message()
    }

    fn without_message(mut self) -> Self {
        self.error = None;
        self
    }

    /// Attach provider retry guidance to the error this step raises.
    pub fn with_retry_advice(mut self, advice: ModelRetryAdvice) -> Self {
        self.retry_advice = Some(advice);
        self
    }

    /// Python rejects steps that combine an error with output items.
    fn validate(&self, input_index: usize) -> Result<(), ModelScriptError> {
        if (self.error.is_some() || self.model_error.is_some()) && !self.output.is_empty() {
            return Err(ModelScriptError::InvalidModelStep {
                message: format!(
                    "Scripted model step #{} cannot combine error and output outcomes.",
                    input_index + 1
                ),
                reason: ModelStepReason::ConflictingOutcomes,
                input_index,
            });
        }
        Ok(())
    }
}

impl From<Vec<ResponseOutputItem>> for ModelStep {
    fn from(output: Vec<ResponseOutputItem>) -> Self {
        Self::output(output)
    }
}

impl From<ResponseOutputItem> for ModelStep {
    fn from(item: ResponseOutputItem) -> Self {
        Self::output([item])
    }
}

/// Deterministic fake model that consumes scripted steps in order
/// (Python: `agents.testing.ScriptedModel`).
#[derive(Debug, Default)]
pub struct ScriptedModel {
    steps: Mutex<VecDeque<ModelStep>>,
    calls: Mutex<Vec<ModelCall>>,
    default_usage: Mutex<Option<Usage>>,
    /// Advice attached to scripted errors, matched by error text when asked for.
    advice_by_error: Mutex<Vec<(String, ModelRetryAdvice)>>,
}

impl ScriptedModel {
    /// Create a model with the given steps.
    pub fn new(steps: impl IntoIterator<Item = ModelStep>) -> Self {
        let steps: VecDeque<ModelStep> = steps
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                s.validate(i).expect("invalid scripted model step");
                s
            })
            .collect();
        Self {
            steps: Mutex::new(steps),
            calls: Mutex::new(Vec::new()),
            default_usage: Mutex::new(None),
            advice_by_error: Mutex::new(Vec::new()),
        }
    }

    /// Enqueue one more step.
    pub fn enqueue(&self, step: ModelStep) {
        step.validate(0).expect("invalid scripted model step");
        self.steps
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back(step);
    }

    /// Extend with more steps.
    pub fn extend(&self, steps: impl IntoIterator<Item = ModelStep>) {
        let mut q = self
            .steps
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (i, step) in steps.into_iter().enumerate() {
            step.validate(i).expect("invalid scripted model step");
            q.push_back(step);
        }
    }

    /// Recorded calls in order (Python: `ScriptedModel.calls`).
    pub fn calls(&self) -> Vec<ModelCall> {
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Number of configured steps that have not run yet (Python: `remaining_steps`).
    pub fn remaining_steps(&self) -> usize {
        self.steps
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// First recorded call, if any (Python: `first_call`).
    pub fn first_call(&self) -> Option<ModelCall> {
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .first()
            .cloned()
    }

    /// Most recent recorded call, if any (Python: `last_call`).
    pub fn last_call(&self) -> Option<ModelCall> {
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .last()
            .cloned()
    }

    /// Usage applied to steps that do not provide their own (Python: `set_default_usage`).
    pub fn set_default_usage(&self, usage: Option<Usage>) {
        *self
            .default_usage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = usage;
    }

    /// Raise when configured steps remain unconsumed (Python: `assert_complete`).
    pub fn assert_complete(&self) {
        let remaining = self.remaining_steps();
        assert!(
            remaining == 0,
            "{}",
            ModelScriptError::UnconsumedModelSteps {
                message: format!(
                    "UnconsumedModelSteps: {remaining} scripted model step(s) were not consumed."
                ),
                remaining_steps: remaining,
            }
        );
    }
}

#[async_trait]
impl Model for ScriptedModel {
    /// The advice attached to the exact scripted error that was raised
    /// (Python: `ScriptedModel.get_retry_advice`).
    fn get_retry_advice(&self, request: &ModelRetryAdviceRequest) -> Option<ModelRetryAdvice> {
        let text = request.error.to_string();
        self.advice_by_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .find(|(error, _)| *error == text)
            .map(|(_, advice)| advice.clone())
    }

    async fn get_response(&self, request: ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        self.record(request, false);

        let call_index = self
            .calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
            - 1;
        let step = self.steps.lock().unwrap_or_else(std::sync::PoisonError::into_inner).pop_front().ok_or_else(|| {
            ModelError::Script(
                ModelScriptError::UnexpectedModelCall {
                    message: format!(
                        "UnexpectedModelCall: unexpected non-streaming model call #{}: no scripted steps remain.",
                        call_index + 1
                    ),
                    call_index,
                }
                .to_string(),
            )
        })?;

        apply_fault(&step).await;
        if let Some(err) = self.step_error(&step) {
            return Err(err);
        }
        let usage = self.resolve_usage(&step);

        Ok(ModelResponse {
            output: step.output,
            usage,
            response_id: step.response_id,
            request_id: step.request_id,
            raw_usage: None,
        })
    }

    async fn stream_response(
        &self,
        request: ModelRequest<'_>,
        raw_tx: tokio::sync::mpsc::Sender<Value>,
    ) -> Result<ModelResponse, ModelError> {
        self.record(request, true);

        let call_index = self
            .calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
            - 1;
        let step = self.steps.lock().unwrap_or_else(std::sync::PoisonError::into_inner).pop_front().ok_or_else(|| {
            ModelError::Script(
                ModelScriptError::UnexpectedModelCall {
                    message: format!(
                        "UnexpectedModelCall: unexpected streaming model call #{}: no scripted steps remain.",
                        call_index + 1
                    ),
                    call_index,
                }
                .to_string(),
            )
        })?;

        apply_fault(&step).await;
        if let Some(err) = self.step_error(&step) {
            return Err(err);
        }

        let usage = self.resolve_usage(&step);
        let response = ModelResponse {
            output: step.output,
            usage,
            response_id: step.response_id,
            request_id: step.request_id,
            raw_usage: None,
        };

        // Python (`testing/model.py`) expands a step into the standard Responses event sequence;
        // Rust replays the same sequence (D-011).
        replay_response(
            &raw_tx,
            &response,
            SCRIPTED_RESPONSE_ID,
            SCRIPTED_MODEL,
            0.0,
        )
        .await;

        Ok(response)
    }
}

/// Act out a step's [`StepFault`]: panic, or wait forever. Returns at once for a normal step.
async fn apply_fault(step: &ModelStep) {
    match &step.fault {
        Some(StepFault::Panic(message)) => panic!("{message}"),
        Some(StepFault::Hang) => std::future::pending::<()>().await,
        None => {}
    }
}

impl ScriptedModel {
    /// The error a failing step raises, remembering its retry advice.
    fn step_error(&self, step: &ModelStep) -> Option<ModelError> {
        let error = match (&step.model_error, &step.error) {
            (Some(error), _) => error.clone(),
            (None, Some(message)) => ModelError::Script(message.clone()),
            (None, None) => return None,
        };
        if let Some(advice) = &step.retry_advice {
            self.advice_by_error
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push((error.to_string(), advice.clone()));
        }
        Some(error)
    }

    /// Python: an empty usage falls back to the configured default, else `Usage(requests=1)`.
    fn resolve_usage(&self, step: &ModelStep) -> Usage {
        if step.usage == Usage::default() {
            self.default_usage
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
                .unwrap_or_else(|| Usage::from_tokens(0, 0))
        } else {
            step.usage.clone()
        }
    }

    fn record(&self, request: ModelRequest<'_>, streamed: bool) {
        let call = ModelCall {
            system_instructions: request.system_instructions.map(str::to_string),
            input: match &request.input {
                opai_agents_core::model::ModelInput::Text(s) => Value::String((*s).to_string()),
                opai_agents_core::model::ModelInput::Items(items) => Value::Array(items.to_vec()),
            },
            model_settings: request.model_settings.clone(),
            tool_names: request.tools.iter().map(|t| t.name.clone()).collect(),
            tracing: request.tracing,
            previous_response_id: request.previous_response_id.map(str::to_string),
            conversation_id: request.conversation_id.map(str::to_string),
            streamed,
            output_schema_name: request.output_schema.map(|s| s.name().to_string()),
        };
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(call);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use opai_agents_core::items::ItemHelpers;
    use opai_agents_core::{Agent, RunOptions, Runner};

    use super::*;

    fn agent(model: &Arc<ScriptedModel>) -> Agent {
        Agent::new("a").model(Arc::clone(model) as Arc<dyn Model>)
    }

    #[tokio::test]
    async fn steps_play_back_in_order_and_are_recorded() {
        let model = Arc::new(ScriptedModel::new([ModelStep::from(
            ItemHelpers::text_message("hello"),
        )]));
        let result = Runner::run(&agent(&model), "hi", RunOptions::default())
            .await
            .expect("run");
        assert_eq!(result.final_output_as_str(), Some("hello"));
        assert_eq!(model.calls().len(), 1);
        assert_eq!(model.remaining_steps(), 0);
        model.assert_complete();
    }

    #[tokio::test]
    async fn a_failing_step_fails_the_call() {
        let model = Arc::new(ScriptedModel::new([ModelStep::raise_error("boom")]));
        let err = Runner::run(&agent(&model), "hi", RunOptions::default())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("boom"), "{err}");
    }

    #[tokio::test]
    async fn running_out_of_steps_is_an_error() {
        let model = Arc::new(ScriptedModel::new([]));
        let err = Runner::run(&agent(&model), "hi", RunOptions::default())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("UnexpectedModelCall"), "{err}");
    }

    #[test]
    #[should_panic(expected = "cannot combine error and output")]
    fn a_step_cannot_mix_an_error_with_output() {
        let mut step = ModelStep::from(ItemHelpers::text_message("x"));
        step.error = Some("e".into());
        let _ = ScriptedModel::new([step]);
    }

    #[test]
    fn fault_constructors_carry_no_output_or_error() {
        let p = ModelStep::panic("bang");
        assert_eq!(p.fault, Some(StepFault::Panic("bang".into())));
        assert!(p.output.is_empty() && p.error.is_none() && p.model_error.is_none());
        let h = ModelStep::hang();
        assert_eq!(h.fault, Some(StepFault::Hang));
        assert!(h.output.is_empty() && h.error.is_none());
    }

    #[tokio::test]
    async fn a_panic_step_panics_in_the_model_call() {
        let model = Arc::new(ScriptedModel::new([ModelStep::panic("scripted panic")]));
        let handle = tokio::spawn({
            let model = Arc::clone(&model);
            async move { Runner::run(&agent(&model), "hi", RunOptions::default()).await }
        });
        let joined = handle.await;
        let err = joined.expect_err("the task must panic");
        assert!(err.is_panic());
    }

    #[tokio::test]
    async fn a_hang_step_never_returns() {
        let model = Arc::new(ScriptedModel::new([ModelStep::hang()]));
        let hanging = agent(&model);
        let run = Runner::run(&hanging, "hi", RunOptions::default());
        let waited = tokio::time::timeout(Duration::from_millis(100), run).await;
        assert!(waited.is_err(), "a hanging step must not complete");
        assert_eq!(model.calls().len(), 1);
    }
}
