//! Refuse a request the provider cannot honour, instead of sending a field it will drop.
//!
//! [`GuardedModel`] wraps whatever a provider built and looks at what the request actually needs
//! — reasoning, a JSON schema, tool calls, an image — before the request goes out. `ModelError` has
//! no variant for "the provider cannot do this", so the refusal travels as
//! [`ModelError::Unsupported`] with [`UnsupportedCapability`]'s text, which names the provider,
//! the capability and, when there is one, the alternative.
//!
//! Checks that would cost something (scanning the input for images and PDFs) only run when the
//! provider lacks that capability, so the usual path pays nothing.

use std::sync::Arc;

use openai_agents_core::error::ModelError;
use openai_agents_core::model::{Model, ModelInput, ModelRequest};
use openai_agents_core::ModelResponse;

use super::spec::{Capabilities, Capability};
use super::UnsupportedCapability;

/// A model whose requests are checked against what its provider supports.
pub struct GuardedModel {
    inner: Arc<dyn Model>,
    provider: String,
    capabilities: Capabilities,
}

impl std::fmt::Debug for GuardedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardedModel")
            .field("provider", &self.provider)
            .field("model", &std::sync::Arc::as_ptr(&self.inner))
            .finish()
    }
}

impl GuardedModel {
    /// Wrap `inner`, refusing any request `capabilities` does not cover.
    pub fn new(
        inner: Arc<dyn Model>,
        provider: impl Into<String>,
        capabilities: Capabilities,
    ) -> Self {
        Self {
            inner,
            provider: provider.into(),
            capabilities,
        }
    }

    /// The capabilities of the wrapped provider.
    pub fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    /// What `request` needs that this provider lacks, plus a hint for the first one.
    fn unsupported(&self, request: &ModelRequest<'_>) -> Option<UnsupportedCapability> {
        let needs = |capability: Capability| {
            (!self.capabilities.supports(capability)).then_some(capability)
        };
        let settings = request.model_settings;

        if let Some(capability) = settings
            .reasoning
            .as_ref()
            .and_then(|_| needs(Capability::Reasoning))
        {
            return Some(
                UnsupportedCapability::new(&self.provider, capability).with_hint(
                    "drop `reasoning`, or name a model whose provider reports reasoning",
                ),
            );
        }
        if let Some(schema) = request.output_schema {
            if !schema.is_plain_text() {
                if let Some(capability) = needs(Capability::StructuredOutput) {
                    return Some(
                        UnsupportedCapability::new(&self.provider, capability).with_hint(
                            "use a plain-text output type and parse it yourself, or a provider \
                             that constrains output to a schema",
                        ),
                    );
                }
            }
        }
        if !request.tools.is_empty() {
            if let Some(capability) = needs(Capability::ToolCalls) {
                return Some(UnsupportedCapability::new(&self.provider, capability));
            }
            let parallel = settings.parallel_tool_calls == Some(true) && request.tools.len() > 1;
            if parallel {
                if let Some(capability) = needs(Capability::ParallelToolCalls) {
                    return Some(
                        UnsupportedCapability::new(&self.provider, capability)
                            .with_hint("set `parallel_tool_calls` to false"),
                    );
                }
            }
        }
        if settings.store == Some(true) {
            if let Some(capability) = needs(Capability::Store) {
                return Some(
                    UnsupportedCapability::new(&self.provider, capability)
                        .with_hint("server-side storage is a Responses-only feature"),
                );
            }
        }
        // Only pay for the scan when the provider cannot do it.
        if !self.capabilities.supports(Capability::Vision)
            && input_mentions(&request.input, |item| item_type_is(item, "input_image"))
        {
            return Some(
                UnsupportedCapability::new(&self.provider, Capability::Vision)
                    .with_hint("the provider does not accept images"),
            );
        }
        if !self.capabilities.supports(Capability::PdfInput)
            && input_mentions(&request.input, is_pdf)
        {
            return Some(
                UnsupportedCapability::new(&self.provider, Capability::PdfInput)
                    .with_hint("extract the text first: the provider does not accept documents"),
            );
        }
        None
    }

    fn check(&self, request: &ModelRequest<'_>) -> Result<(), ModelError> {
        match self.unsupported(request) {
            Some(unsupported) => {
                tracing::warn!(
                    provider = %self.provider,
                    capability = %unsupported.capability,
                    "request refused before it was sent"
                );
                Err(ModelError::Unsupported(unsupported.to_string()))
            }
            None => Ok(()),
        }
    }
}

#[async_trait::async_trait]
impl Model for GuardedModel {
    async fn get_response(&self, request: ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        self.check(&request)?;
        self.inner.get_response(request).await
    }

    async fn stream_response(
        &self,
        request: ModelRequest<'_>,
        raw_tx: tokio::sync::mpsc::Sender<serde_json::Value>,
    ) -> Result<ModelResponse, ModelError> {
        // Streaming itself is not checked: every adapter can stream or fall back to a replayed
        // response, so refusing it here would only remove a working path.
        self.check(&request)?;
        self.inner.stream_response(request, raw_tx).await
    }

    fn get_retry_advice(
        &self,
        request: &openai_agents_core::retry::ModelRetryAdviceRequest,
    ) -> Option<openai_agents_core::retry::ModelRetryAdvice> {
        self.inner.get_retry_advice(request)
    }
}

/// Walk the input items until `wanted` says yes.
fn input_mentions(input: &ModelInput<'_>, wanted: impl Fn(&serde_json::Value) -> bool) -> bool {
    match input {
        ModelInput::Items(items) => items.iter().any(wanted),
        ModelInput::Text(_) => false,
    }
}

fn item_type_is(item: &serde_json::Value, kind: &str) -> bool {
    has_kind(item, kind)
}

fn is_pdf(item: &serde_json::Value) -> bool {
    has_kind(item, "input_file")
        && item
            .pointer("/filename")
            .or_else(|| item.pointer("/file_data"))
            .and_then(|v| v.as_str())
            .is_some_and(|name| name.to_ascii_lowercase().ends_with(".pdf"))
}

/// Whether `item`, or any nested content part, has `"type": kind`.
fn has_kind(item: &serde_json::Value, kind: &str) -> bool {
    match item {
        serde_json::Value::Object(map) => {
            if map.get("type").and_then(serde_json::Value::as_str) == Some(kind) {
                return true;
            }
            map.values().any(|value| has_kind(value, kind))
        }
        serde_json::Value::Array(items) => items.iter().any(|value| has_kind(value, kind)),
        _ => false,
    }
}
