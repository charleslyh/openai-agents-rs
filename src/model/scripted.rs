//! Scripted model for deterministic tests (Python: `agents.testing.ScriptedModel`).
//!
//! Mirrors `agents/testing/model.py`: steps are consumed in order, every call is recorded as a
//! [`ModelCall`], and script misuse raises a typed [`ModelScriptError`].

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::ModelError;
use crate::items::{ModelResponse, ResponseOutputItem};
use crate::model::wire_events::{emit_response_stream, response_object};
use crate::model::wire_events::{SCRIPTED_MODEL, SCRIPTED_RESPONSE_ID};
use crate::model::{Model, ModelRequest};
use crate::model_settings::ModelSettings;
use crate::retry::{ModelRetryAdvice, ModelRetryAdviceRequest};
use crate::usage::Usage;

use super::ModelTracing;

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
                s.validate(i)
                    .expect("invalid scripted model step");
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
        self.steps.lock().expect("scripted lock").push_back(step);
    }

    /// Extend with more steps.
    pub fn extend(&self, steps: impl IntoIterator<Item = ModelStep>) {
        let mut q = self.steps.lock().expect("scripted lock");
        for (i, step) in steps.into_iter().enumerate() {
            step.validate(i).expect("invalid scripted model step");
            q.push_back(step);
        }
    }

    /// Recorded calls in order (Python: `ScriptedModel.calls`).
    pub fn calls(&self) -> Vec<ModelCall> {
        self.calls.lock().expect("scripted lock").clone()
    }

    /// Number of configured steps that have not run yet (Python: `remaining_steps`).
    pub fn remaining_steps(&self) -> usize {
        self.steps.lock().expect("scripted lock").len()
    }

    /// First recorded call, if any (Python: `first_call`).
    pub fn first_call(&self) -> Option<ModelCall> {
        self.calls.lock().expect("scripted lock").first().cloned()
    }

    /// Most recent recorded call, if any (Python: `last_call`).
    pub fn last_call(&self) -> Option<ModelCall> {
        self.calls.lock().expect("scripted lock").last().cloned()
    }

    /// Usage applied to steps that do not provide their own (Python: `set_default_usage`).
    pub fn set_default_usage(&self, usage: Option<Usage>) {
        *self.default_usage.lock().expect("scripted lock") = usage;
    }

    /// Raise when configured steps remain unconsumed (Python: `assert_complete`).
    pub fn assert_complete(&self) {
        let remaining = self.remaining_steps();
        assert!(
            remaining == 0,
            "{}",
            ModelScriptError::UnconsumedModelSteps {
                message: format!("UnconsumedModelSteps: {remaining} scripted model step(s) were not consumed."),
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
            .expect("scripted lock")
            .iter()
            .find(|(error, _)| *error == text)
            .map(|(_, advice)| advice.clone())
    }

    async fn get_response(&self, request: ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        self.record(request, false);

        let call_index = self.calls.lock().expect("scripted lock").len() - 1;
        let step = self.steps.lock().expect("scripted lock").pop_front().ok_or_else(|| {
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

        let call_index = self.calls.lock().expect("scripted lock").len() - 1;
        let step = self.steps.lock().expect("scripted lock").pop_front().ok_or_else(|| {
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
        let wire = response_object(
            response
                .response_id
                .as_deref()
                .unwrap_or(SCRIPTED_RESPONSE_ID),
            SCRIPTED_MODEL,
            "completed",
            0.0,
            &response.output,
            &response.usage,
            "none",
        );
        emit_response_stream(&raw_tx, wire).await;

        Ok(response)
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
                .expect("scripted lock")
                .push((error.to_string(), advice.clone()));
        }
        Some(error)
    }

    /// Python: an empty usage falls back to the configured default, else `Usage(requests=1)`.
    fn resolve_usage(&self, step: &ModelStep) -> Usage {
        if step.usage == Usage::default() {
            self.default_usage
                .lock()
                .expect("scripted lock")
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
                crate::model::ModelInput::Text(s) => Value::String((*s).to_string()),
                crate::model::ModelInput::Items(items) => Value::Array(items.to_vec()),
            },
            model_settings: request.model_settings.clone(),
            tool_names: request.tools.iter().map(|t| t.name.clone()).collect(),
            tracing: request.tracing,
            previous_response_id: request.previous_response_id.map(str::to_string),
            conversation_id: request.conversation_id.map(str::to_string),
            streamed,
            output_schema_name: request.output_schema.map(|s| s.name().to_string()),
        };
        self.calls.lock().expect("scripted lock").push(call);
    }
}
