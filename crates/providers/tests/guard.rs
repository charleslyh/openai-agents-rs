//! A request that needs something the provider lacks is refused before it is sent.

use std::sync::Arc;

use openai_agents_core::error::ModelError;
use openai_agents_core::model::{Model, ModelInput, ModelRequest};
use openai_agents_core::{
    AgentOutputSchemaBase, AgentsError, FunctionTool, ModelResponse, ModelSettings, ModelTracing,
    Reasoning, ReasoningEffort,
};
use openai_agents_providers::{Capabilities, Capability, GuardedModel};

/// Answers with a marker, so a test can tell "the request went through" from "it was refused".
struct StubModel;

#[async_trait::async_trait]
impl Model for StubModel {
    async fn get_response(&self, _request: ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        Err(ModelError::Behavior("stub was called".into()))
    }
}

/// A structured (non plain-text) output type, to ask for `StructuredOutput`.
struct JsonSchema;

impl std::fmt::Debug for JsonSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("JsonSchema")
    }
}

impl AgentOutputSchemaBase for JsonSchema {
    fn is_plain_text(&self) -> bool {
        false
    }
    fn name(&self) -> &str {
        "answer"
    }
    fn json_schema(&self) -> Result<&serde_json::Value, AgentsError> {
        Err(openai_agents_core::UserError::new("not needed here").into())
    }
    fn is_strict_json_schema(&self) -> bool {
        false
    }
    fn validate_json(&self, _json_str: &str) -> Result<serde_json::Value, ModelError> {
        Err(ModelError::Behavior("not needed here".into()))
    }
}

struct Case<'a> {
    settings: ModelSettings,
    tools: Vec<FunctionTool>,
    input: Vec<serde_json::Value>,
    schema: Option<&'a dyn AgentOutputSchemaBase>,
}

impl Case<'_> {
    fn new() -> Self {
        Self {
            settings: ModelSettings::default(),
            tools: Vec::new(),
            input: Vec::new(),
            schema: None,
        }
    }

    async fn against(self, capabilities: Capabilities) -> String {
        let model = GuardedModel::new(Arc::new(StubModel), "acme", capabilities);
        let request = ModelRequest {
            system_instructions: None,
            input: ModelInput::Items(&self.input),
            model_settings: &self.settings,
            tools: &self.tools,
            tracing: ModelTracing::Disabled,
            previous_response_id: None,
            conversation_id: None,
            output_schema: self.schema,
        };
        match model.get_response(request).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("StubModel always answers with an error"),
        }
    }
}

fn refused(message: &str) -> bool {
    message.contains("does not support")
}

#[tokio::test]
async fn a_request_the_provider_can_honour_goes_through() {
    let mut settings = ModelSettings::default();
    settings.reasoning = Some(Reasoning::effort(ReasoningEffort::Low));
    let message = Case {
        settings,
        tools: vec![FunctionTool::constant("t", "t", "ok")],
        ..Case::new()
    }
    .against(Capabilities::full())
    .await;
    assert!(!refused(&message), "{message}");
    assert!(message.contains("stub was called"), "{message}");
}

#[tokio::test]
async fn reasoning_is_refused_where_the_provider_has_no_reasoning_field() {
    let mut settings = ModelSettings::default();
    settings.reasoning = Some(Reasoning::effort(ReasoningEffort::High));
    let message = Case {
        settings,
        ..Case::new()
    }
    .against(Capabilities::chat())
    .await;
    assert!(refused(&message), "{message}");
    assert!(message.contains("reasoning"), "{message}");
    assert!(message.contains("acme"), "{message}");
}

#[tokio::test]
async fn a_json_schema_is_refused_where_the_provider_cannot_constrain_output() {
    let message = Case {
        schema: Some(&JsonSchema),
        ..Case::new()
    }
    .against(Capabilities::none())
    .await;
    assert!(refused(&message), "{message}");
    assert!(message.contains("structured_output"), "{message}");
}

#[tokio::test]
async fn tools_are_refused_where_the_provider_cannot_call_them() {
    let message = Case {
        tools: vec![FunctionTool::constant("t", "t", "ok")],
        ..Case::new()
    }
    .against(Capabilities::none())
    .await;
    assert!(refused(&message), "{message}");
    assert!(message.contains("tool_calls"), "{message}");
}

#[tokio::test]
async fn parallel_tool_calls_are_refused_where_the_provider_runs_them_one_at_a_time() {
    let mut settings = ModelSettings::default();
    settings.parallel_tool_calls = Some(true);
    let message = Case {
        settings,
        tools: vec![
            FunctionTool::constant("a", "a", "a"),
            FunctionTool::constant("b", "b", "b"),
        ],
        ..Case::new()
    }
    .against(Capabilities::chat().without(Capability::ParallelToolCalls))
    .await;
    assert!(refused(&message), "{message}");
    assert!(message.contains("parallel_tool_calls"), "{message}");
}

#[tokio::test]
async fn one_tool_call_is_not_a_parallel_request() {
    let mut settings = ModelSettings::default();
    settings.parallel_tool_calls = Some(true);
    let message = Case {
        settings,
        tools: vec![FunctionTool::constant("a", "a", "a")],
        ..Case::new()
    }
    .against(Capabilities::chat().without(Capability::ParallelToolCalls))
    .await;
    assert!(!refused(&message), "{message}");
}

#[tokio::test]
async fn server_side_storage_is_refused_outside_the_responses_api() {
    let mut settings = ModelSettings::default();
    settings.store = Some(true);
    let message = Case {
        settings,
        ..Case::new()
    }
    .against(Capabilities::chat())
    .await;
    assert!(refused(&message), "{message}");
    assert!(message.contains("store"), "{message}");
}

#[tokio::test]
async fn an_image_is_refused_where_the_provider_has_no_vision() {
    let message = Case {
        input: vec![serde_json::json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "image_url": "https://x/y.png"}]
        })],
        ..Case::new()
    }
    .against(Capabilities::chat())
    .await;
    assert!(refused(&message), "{message}");
    assert!(message.contains("vision"), "{message}");
}

#[tokio::test]
async fn a_pdf_is_refused_where_the_provider_takes_no_documents() {
    let message = Case {
        input: vec![serde_json::json!({
            "type": "input_file",
            "filename": "paper.pdf",
            "file_data": "..."
        })],
        ..Case::new()
    }
    .against(Capabilities::chat())
    .await;
    assert!(refused(&message), "{message}");
    assert!(message.contains("pdf_input"), "{message}");
}

/// The input is only scanned when the provider lacks the capability.
#[tokio::test]
async fn an_image_is_not_scanned_when_the_provider_has_vision() {
    let message = Case {
        input: vec![serde_json::json!({"type": "input_image"})],
        ..Case::new()
    }
    .against(Capabilities::full())
    .await;
    assert!(!refused(&message), "{message}");
}
