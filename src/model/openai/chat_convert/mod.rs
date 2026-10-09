//! Responses items <-> Chat Completions messages (Python: `chatcmpl_converter`).
//!
//! Chat Completions has no item list: an assistant turn is one message that carries its text, its
//! refusal and *all* of its tool calls, and tool results follow as `tool` messages. Servers are
//! strict about that shape, so these two functions are the core of talking to third-party
//! providers:
//!
//! * [`items_to_chat_messages`] turns the run's input items into `messages`.
//! * [`chat_message_to_output_items`] turns a reply `message` back into output items.
//!
//! Both are pure JSON transforms and are checked against Python's converter on shared inputs.
//!
//! Not ported (OpenAI- or LiteLLM-specific): Claude thinking blocks, Gemini thought signatures,
//! `provider_data`, `file_search_call` items, audio and video parts.

use std::sync::Arc;

use serde_json::Value;

use crate::error::ModelError;

/// Stand-in for a tool output that has no text (Python: `_OMITTED_TOOL_OUTPUT_PLACEHOLDER`).
pub const OMITTED_TOOL_OUTPUT_PLACEHOLDER: &str = "[tool output omitted]";

/// Decides whether an earlier reasoning item is sent back with the next request, given the target
/// model name (Python: `should_replay_reasoning_content`).
pub type ReplayReasoningFn = Arc<dyn Fn(&str, &Value) -> bool + Send + Sync>;

/// Key under which an item remembers that the server sent its chain of thought as the
/// `reasoning` field (Python: `_CHAT_COMPLETIONS_REASONING_FIELD_KEY`).
pub const REASONING_FIELD_KEY: &str = "_chat_completions_reasoning_field";

/// Default replay rule: only DeepSeek models take `reasoning_content` back, and only reasoning
/// that came from DeepSeek or predates provider tracking (Python:
/// `default_should_replay_reasoning_content`). That keeps another model family's reasoning out
/// of a DeepSeek assistant message.
pub fn default_should_replay_reasoning(model: &str, reasoning: &Value) -> bool {
    if !model.to_lowercase().contains("deepseek") {
        return false;
    }
    let provider_data = reasoning.get("provider_data").and_then(Value::as_object);
    let origin_is_deepseek = provider_data
        .and_then(|d| d.get("model"))
        .and_then(Value::as_str)
        .is_some_and(|m| m.to_lowercase().contains("deepseek"));
    let tracked = provider_data.is_some_and(|d| d.keys().any(|k| k != "thinking_blocks"));
    origin_is_deepseek || !tracked
}

/// Settings of [`items_to_chat_messages`].
#[derive(Clone, Default)]
pub struct ChatConvertOptions {
    /// Model that will receive the messages.
    pub model: String,
    /// Replaces [`default_should_replay_reasoning`] when set.
    pub replay_reasoning: Option<ReplayReasoningFn>,
    /// Send Claude thinking blocks back with the assistant message that follows them
    /// (Python: `preserve_thinking_blocks`, which only its LiteLLM adapter turns on). Needed by
    /// Claude models with extended thinking behind a gateway that returns `thinking_blocks`.
    pub preserve_thinking_blocks: bool,
}

mod from_chat;
mod to_chat;

pub use from_chat::{chat_message_to_output_items, chat_message_to_output_items_with};
pub use to_chat::items_to_chat_messages;

pub(super) fn unsupported(message: impl Into<String>) -> ModelError {
    ModelError::Unsupported(message.into())
}

pub(super) fn str_field<'a>(item: &'a Value, key: &str) -> Option<&'a str> {
    item.get(key).and_then(Value::as_str)
}
