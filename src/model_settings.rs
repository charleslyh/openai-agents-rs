//! Model settings (Python: `agents.model_settings.ModelSettings` subset).

use serde::{Deserialize, Serialize};

/// Tunable parameters forwarded to the model provider.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelSettings {
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling.
    pub top_p: Option<f32>,
    /// Maximum output tokens.
    pub max_tokens: Option<u32>,
    /// Tool choice hint (`"auto"`, `"none"`, `"required"`, or a tool name).
    pub tool_choice: Option<String>,
    /// Parallel tool calls.
    pub parallel_tool_calls: Option<bool>,
    /// Extra JSON fields merged into the provider request (advanced).
    pub extra_body: Option<serde_json::Value>,
}

/// Default model settings used when an agent does not override them.
pub fn get_default_model_settings() -> ModelSettings {
    ModelSettings::default()
}
