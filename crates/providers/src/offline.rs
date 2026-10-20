//! A model that answers without a network (Rust-only, X-015).
//!
//! With `AGENTS_OFFLINE=1` a run goes through every step — instructions, turns, tools if the
//! script asks for none, sessions, tracing — and stops at the model call, which answers locally.
//! That is enough to see whether a project is wired up before any key exists, and to keep a demo
//! or a test from spending money.
//!
//! The answer says what it is, so nobody mistakes it for a real one.

use opai_agents_core::error::ModelError;
use std::sync::Arc;

use opai_agents_core::error::UserError;
use opai_agents_core::model::{Model, ModelProvider, ModelRequest};
use opai_agents_core::{ItemHelpers, ModelResponse, Usage};

/// A model that answers locally with a fixed message.
#[derive(Debug, Clone, Copy, Default)]
pub struct OfflineModel;

impl OfflineModel {
    /// The answer this model gives.
    pub fn answer() -> String {
        "[offline] No provider was called: AGENTS_OFFLINE=1 is set. \
         Set OPENAI_API_KEY (or a preset's variable such as DEEPSEEK_API_KEY) and run again to \
         talk to a real model."
            .to_string()
    }
}

#[async_trait::async_trait]
impl Model for OfflineModel {
    async fn get_response(&self, _request: ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        tracing::info!("offline model answering; no request was sent");
        Ok(ModelResponse {
            output: vec![ItemHelpers::text_message(Self::answer())],
            usage: Usage::default(),
            response_id: None,
            request_id: None,
            raw_usage: None,
        })
    }
}

/// Hands out [`OfflineModel`] whatever name is asked for.
#[derive(Debug, Clone, Copy, Default)]
pub struct OfflineProvider;

#[async_trait::async_trait]
impl ModelProvider for OfflineProvider {
    async fn get_model(&self, _model_name: Option<&str>) -> Result<Arc<dyn Model>, UserError> {
        Ok(Arc::new(OfflineModel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opai_agents_core::model::{ModelInput, ModelRequest};
    use opai_agents_core::{ModelSettings, ModelTracing};

    #[tokio::test]
    async fn the_offline_answer_says_it_is_offline() {
        let settings = ModelSettings::default();
        let request = ModelRequest {
            system_instructions: None,
            input: ModelInput::Text("hello"),
            model_settings: &settings,
            tools: &[],
            tracing: ModelTracing::Disabled,
            previous_response_id: None,
            conversation_id: None,
            output_schema: None,
        };
        let response = OfflineModel.get_response(request).await.expect("answers");
        let text = serde_json::to_string(&response.output).unwrap_or_default();
        assert!(text.contains("offline"), "{text}");
        assert!(text.contains("AGENTS_OFFLINE"), "{text}");
    }
}
