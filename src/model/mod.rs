//! Model interface (Python: `agents.models.interface`).

use async_trait::async_trait;

use crate::error::ModelError;
use crate::items::{ModelResponse, ResponseInputItem};
use crate::model_settings::ModelSettings;
use crate::tool::FunctionTool;

#[cfg(feature = "openai")]
pub mod openai;
pub mod scripted;

pub use scripted::ScriptedModel;

/// Tracing mode for a model call (Python: `ModelTracing`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelTracing {
    /// Tracing disabled.
    Disabled = 0,
    /// Tracing enabled with data.
    #[default]
    Enabled = 1,
    /// Tracing enabled without input/output payloads.
    EnabledWithoutData = 2,
}

impl ModelTracing {
    /// Whether tracing is fully disabled.
    pub fn is_disabled(self) -> bool {
        matches!(self, Self::Disabled)
    }

    /// Whether sensitive payloads should be recorded.
    pub fn include_data(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

/// Bundled arguments for [`Model::get_response`].
#[derive(Debug)]
pub struct ModelRequest<'a> {
    /// System instructions.
    pub system_instructions: Option<&'a str>,
    /// Input string or Responses input items.
    pub input: ModelInput<'a>,
    /// Model settings.
    pub model_settings: &'a ModelSettings,
    /// Available function tools.
    pub tools: &'a [FunctionTool],
    /// Tracing mode.
    pub tracing: ModelTracing,
    /// Previous Responses API response id.
    pub previous_response_id: Option<&'a str>,
    /// Conversation id when using stored conversations.
    pub conversation_id: Option<&'a str>,
}

/// Input to a model call.
#[derive(Debug, Clone)]
pub enum ModelInput<'a> {
    /// Plain text.
    Text(&'a str),
    /// Responses input item list.
    Items(&'a [ResponseInputItem]),
}

impl<'a> ModelInput<'a> {
    /// Borrow as owned items when needed.
    pub fn to_owned_items(&self) -> Vec<ResponseInputItem> {
        match self {
            Self::Text(s) => vec![serde_json::json!({"role": "user", "content": s})],
            Self::Items(items) => items.to_vec(),
        }
    }
}

/// Provider-neutral model interface.
#[async_trait]
pub trait Model: Send + Sync {
    /// Get a complete model response (non-streaming).
    async fn get_response(&self, request: ModelRequest<'_>) -> Result<ModelResponse, ModelError>;

    /// Stream raw JSON events (e.g. text deltas), then return the assembled response.
    ///
    /// Default: call [`Self::get_response`] and emit a single synthetic `response.completed`.
    /// Chat Completions overrides this for token-level `output_text.delta` events.
    async fn stream_response(
        &self,
        request: ModelRequest<'_>,
        raw_tx: tokio::sync::mpsc::Sender<serde_json::Value>,
    ) -> Result<ModelResponse, ModelError> {
        let response = self.get_response(request).await?;
        let _ = raw_tx
            .send(serde_json::json!({
                "type": "response.completed",
                "response_id": response.response_id,
                "output": response.output,
            }))
            .await;
        Ok(response)
    }
}
