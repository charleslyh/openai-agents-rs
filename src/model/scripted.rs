//! Scripted model for deterministic tests (Python: `agents.testing.ScriptedModel`).

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::ModelError;
use crate::items::{ModelResponse, ResponseOutputItem};
use crate::model::{Model, ModelRequest};
use crate::model_settings::ModelSettings;
use crate::tool::FunctionTool;
use crate::usage::Usage;

use super::ModelTracing;

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
}

/// One deterministic step (Python: `ModelStep` simplified).
#[derive(Debug, Clone)]
pub struct ModelStep {
    /// Output items for this call.
    pub output: Vec<ResponseOutputItem>,
    /// Usage for this call.
    pub usage: Usage,
    /// Response id.
    pub response_id: Option<String>,
    /// If set, the model call fails with this message.
    pub error: Option<String>,
}

impl ModelStep {
    /// Create a successful step from output items.
    pub fn respond(output: Vec<ResponseOutputItem>) -> Self {
        Self {
            output,
            usage: Usage::default(),
            response_id: Some("resp-789".into()),
            error: None,
        }
    }

    /// Create a step that raises an error.
    pub fn raise_error(message: impl Into<String>) -> Self {
        Self {
            output: vec![],
            usage: Usage::default(),
            response_id: None,
            error: Some(message.into()),
        }
    }
}

impl From<Vec<ResponseOutputItem>> for ModelStep {
    fn from(output: Vec<ResponseOutputItem>) -> Self {
        Self::respond(output)
    }
}

impl From<ResponseOutputItem> for ModelStep {
    fn from(item: ResponseOutputItem) -> Self {
        Self::respond(vec![item])
    }
}

/// Deterministic fake model that consumes scripted steps in order.
#[derive(Debug, Default)]
pub struct ScriptedModel {
    steps: Mutex<VecDeque<ModelStep>>,
    calls: Mutex<Vec<ModelCall>>,
}

impl ScriptedModel {
    /// Create a model with the given steps.
    pub fn new(steps: impl IntoIterator<Item = ModelStep>) -> Self {
        Self {
            steps: Mutex::new(steps.into_iter().collect()),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Enqueue one more step.
    pub fn enqueue(&self, step: ModelStep) {
        self.steps.lock().expect("scripted lock").push_back(step);
    }

    /// Extend with more steps.
    pub fn extend(&self, steps: impl IntoIterator<Item = ModelStep>) {
        let mut q = self.steps.lock().expect("scripted lock");
        q.extend(steps);
    }

    /// Recorded calls in order.
    pub fn calls(&self) -> Vec<ModelCall> {
        self.calls.lock().expect("scripted lock").clone()
    }

    /// Assert every configured step was consumed.
    pub fn assert_complete(&self) {
        let remaining = self.steps.lock().expect("scripted lock").len();
        assert!(
            remaining == 0,
            "UnconsumedModelSteps: {remaining} step(s) remaining"
        );
    }
}

#[async_trait]
impl Model for ScriptedModel {
    async fn get_response(&self, request: ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
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
        };
        self.calls.lock().expect("scripted lock").push(call);

        let step = self
            .steps
            .lock()
            .expect("scripted lock")
            .pop_front()
            .ok_or_else(|| {
                ModelError::Script(
                    "UnexpectedModelCall: no remaining scripted steps".to_string(),
                )
            })?;

        if let Some(err) = step.error {
            return Err(ModelError::Script(err));
        }

        let _tools: &[FunctionTool] = request.tools;

        Ok(ModelResponse {
            output: step.output,
            usage: step.usage,
            response_id: step.response_id,
            request_id: None,
        })
    }
}
