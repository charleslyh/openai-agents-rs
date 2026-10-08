//! Model interface (Python: `agents.models.interface`).

use std::sync::Arc;

use async_trait::async_trait;

use crate::agent_output::AgentOutputSchemaBase;
use crate::error::ModelError;
use crate::items::{ModelResponse, ResponseInputItem};
use crate::model_settings::ModelSettings;
use crate::tool::FunctionTool;

#[cfg(feature = "openai")]
pub mod openai;
pub mod provider;
pub mod scripted;
pub(crate) mod wire_events;

pub use provider::{MissingProvider, ModelProvider, MultiProvider};
pub use scripted::ScriptedModel;

/// A model reference: either a ready instance or a name resolved by a [`ModelProvider`].
#[derive(Clone)]
pub enum ModelRef {
    /// A pre-built model instance.
    Instance(Arc<dyn Model>),
    /// A model name resolved through `RunConfig.model_provider`.
    Name(String),
}

impl std::fmt::Debug for ModelRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Instance(_) => f.write_str("ModelRef::Instance(..)"),
            Self::Name(n) => f.debug_tuple("ModelRef::Name").field(n).finish(),
        }
    }
}

impl From<Arc<dyn Model>> for ModelRef {
    fn from(value: Arc<dyn Model>) -> Self {
        Self::Instance(value)
    }
}

impl From<&str> for ModelRef {
    fn from(value: &str) -> Self {
        Self::Name(value.to_string())
    }
}

impl From<String> for ModelRef {
    fn from(value: String) -> Self {
        Self::Name(value)
    }
}

/// The provider used when `RunConfig.model_provider` is not set.
///
/// With the `openai` feature this routes `openai/...` (and unprefixed) names to the OpenAI
/// protocol (Chat Completions unless changed), `openai_responses/...` to Responses, honouring [`crate::get_default_openai_api`]. Without it, name resolution fails with an
/// actionable error.
pub fn default_model_provider() -> Arc<dyn ModelProvider> {
    #[cfg(feature = "openai")]
    {
        match openai::OpenAIProvider::from_env() {
            Ok(provider) => {
                let router = MultiProvider::new()
                    .register("openai", Arc::new(provider.clone()))
                    .register(
                        "openai_chat_completions",
                        Arc::new(provider.clone().always_chat_completions()),
                    )
                    .register(
                        "openai_responses",
                        Arc::new(provider.clone().api(crate::run::DefaultOpenAiApi::Responses)),
                    );
                Arc::new(router)
            }
            Err(_) => Arc::new(MissingProvider),
        }
    }
    #[cfg(not(feature = "openai"))]
    {
        Arc::new(MissingProvider)
    }
}

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
    /// Structured output schema declared by the agent (Python: `ModelRequest.output_schema`).
    ///
    /// `None` or a plain-text schema means the model should answer in free-form text.
    pub output_schema: Option<&'a dyn AgentOutputSchemaBase>,
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

    /// Provider guidance about a failed call, used by retry policies
    /// (Python: `Model.get_retry_advice`). `None` means the adapter has no opinion.
    fn get_retry_advice(
        &self,
        _request: &crate::retry::ModelRetryAdviceRequest,
    ) -> Option<crate::retry::ModelRetryAdvice> {
        None
    }

    /// Stream standard Responses API wire events, then return the assembled response.
    ///
    /// Default: call [`Self::get_response`] and replay the result as the standard event sequence
    /// (`response.created` → per-item events → `response.completed`). Both OpenAI adapters
    /// override it to stream real token deltas (D-011).
    async fn stream_response(
        &self,
        request: ModelRequest<'_>,
        raw_tx: tokio::sync::mpsc::Sender<serde_json::Value>,
    ) -> Result<ModelResponse, ModelError> {
        let response = self.get_response(request).await?;
        let wire = wire_events::response_object(
            response
                .response_id
                .as_deref()
                .unwrap_or(wire_events::FAKE_RESPONSES_ID),
            "unknown",
            "completed",
            wire_events::now_seconds(),
            &response.output,
            &response.usage,
            "none",
        );
        wire_events::emit_response_stream(&raw_tx, wire).await;
        Ok(response)
    }
}
