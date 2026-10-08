//! Summarizing compaction of a stored conversation (provider-neutral).
//!
//! [`CompactingSession`] wraps any [`Session`]. After each run is saved, when the stored history
//! has grown past a trigger, it replaces the oldest turns by a short summary written by a
//! [`Summarizer`] and keeps the most recent turns word for word. The model only ever sees an
//! ordinary user message that starts with `<conversation_summary>`, so any provider can read it.
//!
//! Python's counterpart, `OpenAIResponsesCompactionSession`, calls OpenAI's `responses.compact`
//! endpoint and is not portable; this is a new, provider-neutral feature. It never splits a tool
//! call from its output (turns are replaced as a whole).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;

use super::{Session, SessionSettings};
use crate::context::{
    default_token_counter, is_summary_item, summary_item, summary_text, Conversation, TokenCounter,
};
use crate::error::{AgentsError, ModelError};
use crate::items::extract_message_text;
use crate::model::{Model, ModelInput, ModelRequest, ModelTracing};
use crate::model_settings::ModelSettings;
use crate::usage::Usage;

/// Estimated tokens of stored history above which compaction starts, when no trigger is set.
pub const DEFAULT_TRIGGER_TOKENS: usize = 8_000;

/// Writes the summary that replaces old turns.
#[async_trait]
pub trait Summarizer: Send + Sync {
    /// Condense `items` into notes. `previous_summary` holds the notes from an earlier
    /// compaction, which the result must carry forward. Returns the new notes as plain text.
    async fn summarize(
        &self,
        previous_summary: Option<&str>,
        items: &[Value],
    ) -> Result<String, AgentsError>;

    /// Token usage of every summary written so far. Summaries are model calls the runner does
    /// not see, so they are not part of `RunResult.usage`; read this to account for them.
    fn usage(&self) -> Usage {
        Usage::default()
    }
}

const DEFAULT_SUMMARY_INSTRUCTIONS: &str = "You condense a conversation between a user and an AI \
assistant into notes that let the assistant continue seamlessly. Write plain text, no preamble. \
Keep everything needed later: the user's goals and preferences, decisions and their reasons, \
concrete facts (names, ids, numbers, file paths, URLs), results of tool calls that still matter, \
and unfinished tasks. Drop pleasantries and anything already resolved. If existing notes are \
given, merge them with the new conversation into one set of notes. Be concise.";

/// A [`Summarizer`] that asks a model (any [`Model`], so any provider) to write the notes.
#[derive(Clone)]
pub struct ModelSummarizer {
    model: Arc<dyn Model>,
    model_settings: ModelSettings,
    instructions: String,
    max_tool_chars: usize,
    usage: Arc<Mutex<Usage>>,
}

impl ModelSummarizer {
    /// Summarize with `model` and the default instructions.
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            model_settings: ModelSettings::default(),
            instructions: DEFAULT_SUMMARY_INSTRUCTIONS.to_string(),
            max_tool_chars: 1_500,
            usage: Arc::default(),
        }
    }

    /// Settings of the summarization call (for example a low temperature).
    pub fn model_settings(mut self, settings: ModelSettings) -> Self {
        self.model_settings = settings;
        self
    }

    /// Replace the system prompt that tells the model how to write the notes.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// Tool arguments and results longer than this many characters are cut in the transcript.
    pub fn max_tool_chars(mut self, chars: usize) -> Self {
        self.max_tool_chars = chars;
        self
    }
}

/// `text` cut to `max` characters, with a note of how much was dropped.
fn truncate_chars(text: &str, max: usize) -> String {
    let total = text.chars().count();
    if total <= max {
        return text.to_string();
    }
    let head: String = text.chars().take(max).collect();
    format!("{head}...[truncated {} chars]", total - max)
}

/// The text of a message-like item, whatever shape its content has.
fn message_text(item: &Value) -> Option<String> {
    if let Some(text) = extract_message_text(item) {
        return Some(text);
    }
    match item.get("content")? {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let text: Vec<&str> = parts
                .iter()
                .filter_map(|part| {
                    part.get("text")
                        .or_else(|| part.get("refusal"))
                        .and_then(Value::as_str)
                })
                .collect();
            (!text.is_empty()).then(|| text.join("\n"))
        }
        _ => None,
    }
}

/// One line-ish entry of the transcript, or `None` for items that carry no content.
fn render_item(item: &Value, max_tool_chars: usize) -> Option<String> {
    match item.get("type").and_then(Value::as_str) {
        Some("function_call") => {
            let name = item.get("name").and_then(Value::as_str).unwrap_or("?");
            let arguments = item.get("arguments").and_then(Value::as_str).unwrap_or("");
            Some(format!("[tool call] {name}({})", truncate_chars(arguments, max_tool_chars)))
        }
        Some("function_call_output") => {
            let output = match item.get("output") {
                Some(Value::String(text)) => text.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            };
            Some(format!("[tool result] {}", truncate_chars(&output, max_tool_chars)))
        }
        Some("reasoning") => None,
        _ => {
            let role = item.get("role").and_then(Value::as_str).unwrap_or("assistant");
            let text = message_text(item)?;
            Some(format!("{role}: {text}"))
        }
    }
}

fn render_transcript(items: &[Value], max_tool_chars: usize) -> String {
    items
        .iter()
        .filter_map(|item| render_item(item, max_tool_chars))
        .collect::<Vec<_>>()
        .join("\n")
}

#[async_trait]
impl Summarizer for ModelSummarizer {
    async fn summarize(
        &self,
        previous_summary: Option<&str>,
        items: &[Value],
    ) -> Result<String, AgentsError> {
        let mut prompt = String::new();
        if let Some(previous) = previous_summary {
            prompt.push_str("Existing notes:\n");
            prompt.push_str(previous);
            prompt.push_str("\n\n");
        }
        prompt.push_str("New conversation to fold into the notes:\n");
        prompt.push_str(&render_transcript(items, self.max_tool_chars));

        // Shows up in the trace of the run whose save triggered the compaction.
        let _span = crate::tracing::custom_span("conversation_summary");
        let response = self
            .model
            .get_response(ModelRequest {
                system_instructions: Some(&self.instructions),
                input: ModelInput::Text(&prompt),
                model_settings: &self.model_settings,
                tools: &[],
                tracing: ModelTracing::Disabled,
                previous_response_id: None,
                conversation_id: None,
                output_schema: None,
            })
            .await?;
        self.usage.lock().expect("summarizer usage").add(&response.usage);
        let text: String = response.output.iter().filter_map(extract_message_text).collect();
        if text.trim().is_empty() {
            return Err(ModelError::Behavior("the summarizer model returned no text".into()).into());
        }
        Ok(text)
    }

    fn usage(&self) -> Usage {
        self.usage.lock().expect("summarizer usage").clone()
    }
}

/// A [`Session`] that keeps the stored history short by summarizing old turns.
///
/// ```ignore
/// let session = CompactingSession::new(
///     SqliteSession::open("chat-1", "chats.db")?.shared(),
///     Arc::new(ModelSummarizer::new(model)),
/// )
/// .trigger_tokens(6_000)
/// .keep_recent_turns(3)
/// .shared();
/// ```
///
/// Compaction runs inside `add_items`, that is when a run finishes. If it fails (for example the
/// summarizer model is down) a warning is logged and the history stays as it was, so a completed
/// run is never lost. Writes through one `CompactingSession` are serialized; two sessions over
/// the same store can still race, so use one per conversation.
///
/// A compaction does not count toward `Usage`, because it happens outside the run.
pub struct CompactingSession {
    inner: Arc<dyn Session>,
    summarizer: Arc<dyn Summarizer>,
    trigger_tokens: Option<usize>,
    trigger_items: Option<usize>,
    keep_recent_turns: usize,
    counter: TokenCounter,
    writes: AsyncMutex<()>,
}

impl std::fmt::Debug for CompactingSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactingSession")
            .field("session_id", &self.inner.session_id())
            .field("trigger_tokens", &self.trigger_tokens)
            .field("trigger_items", &self.trigger_items)
            .field("keep_recent_turns", &self.keep_recent_turns)
            .finish()
    }
}

impl CompactingSession {
    /// Wrap `inner`. Until a trigger is set, compaction starts above
    /// [`DEFAULT_TRIGGER_TOKENS`] estimated tokens; the two most recent turns stay verbatim.
    pub fn new(inner: Arc<dyn Session>, summarizer: Arc<dyn Summarizer>) -> Self {
        Self {
            inner,
            summarizer,
            trigger_tokens: None,
            trigger_items: None,
            keep_recent_turns: 2,
            counter: default_token_counter(),
            writes: AsyncMutex::new(()),
        }
    }

    /// Compact when the stored history exceeds this many (estimated) tokens.
    pub fn trigger_tokens(mut self, tokens: usize) -> Self {
        self.trigger_tokens = Some(tokens);
        self
    }

    /// Compact when the stored history exceeds this many items. With
    /// [`trigger_tokens`](Self::trigger_tokens) also set, either one starts a compaction.
    pub fn trigger_items(mut self, items: usize) -> Self {
        self.trigger_items = Some(items);
        self
    }

    /// Number of most recent turns kept word for word (default 2).
    pub fn keep_recent_turns(mut self, turns: usize) -> Self {
        self.keep_recent_turns = turns;
        self
    }

    /// Count tokens with `counter` instead of the built-in estimate.
    pub fn token_counter<F>(mut self, counter: F) -> Self
    where
        F: Fn(&Value) -> usize + Send + Sync + 'static,
    {
        self.counter = Arc::new(counter);
        self
    }

    /// Usage of the summaries written so far (see [`Summarizer::usage`]).
    pub fn summarizer_usage(&self) -> Usage {
        self.summarizer.usage()
    }

    /// Wrap in an `Arc`, ready for `RunOptions::session`.
    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Compact now, ignoring the triggers. Returns whether anything was summarized: `false`
    /// when there are no more turns than `keep_recent_turns`.
    pub async fn compact(&self) -> Result<bool, AgentsError> {
        let _writes = self.writes.lock().await;
        self.compact_locked(true).await
    }

    fn should_compact(&self, items: &[Value]) -> bool {
        let tokens = self
            .trigger_tokens
            .or_else(|| self.trigger_items.is_none().then_some(DEFAULT_TRIGGER_TOKENS));
        if let Some(max) = tokens {
            if items.iter().map(|item| (self.counter)(item)).sum::<usize>() > max {
                return true;
            }
        }
        self.trigger_items.is_some_and(|max| items.len() > max)
    }

    async fn compact_locked(&self, force: bool) -> Result<bool, AgentsError> {
        let items = self.inner.get_items(None).await?;
        if !force && !self.should_compact(&items) {
            return Ok(false);
        }
        let Conversation { pinned, turns } = Conversation::split(&items);
        if turns.len() <= self.keep_recent_turns {
            return Ok(false);
        }
        let (old, recent) = turns.split_at(turns.len() - self.keep_recent_turns);

        let earlier: Vec<String> = pinned.iter().filter_map(summary_text).collect();
        let previous = (!earlier.is_empty()).then(|| earlier.join("\n\n"));
        let old_items: Vec<Value> = old.iter().flatten().cloned().collect();
        let notes = self.summarizer.summarize(previous.as_deref(), &old_items).await?;

        let mut compacted: Vec<Value> =
            pinned.into_iter().filter(|item| !is_summary_item(item)).collect();
        compacted.push(summary_item(&notes));
        compacted.extend(recent.iter().flatten().cloned());
        self.inner.replace_items(compacted).await?;
        Ok(true)
    }
}

#[async_trait]
impl Session for CompactingSession {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        self.inner.session_settings()
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<Value>, AgentsError> {
        self.inner.get_items(limit).await
    }

    async fn add_items(&self, items: Vec<Value>) -> Result<(), AgentsError> {
        let _writes = self.writes.lock().await;
        self.inner.add_items(items).await?;
        if let Err(error) = self.compact_locked(false).await {
            ::tracing::warn!(
                session = self.inner.session_id(),
                %error,
                "context compaction failed; the stored history was left unchanged"
            );
        }
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<Value>, AgentsError> {
        let _writes = self.writes.lock().await;
        self.inner.pop_item().await
    }

    async fn clear_session(&self) -> Result<(), AgentsError> {
        let _writes = self.writes.lock().await;
        self.inner.clear_session().await
    }

    async fn replace_items(&self, items: Vec<Value>) -> Result<(), AgentsError> {
        let _writes = self.writes.lock().await;
        self.inner.replace_items(items).await
    }
}
