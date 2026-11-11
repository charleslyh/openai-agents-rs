//! Summarizing compaction of a stored history (provider-neutral).
//!
//! [`CompactingSession`] wraps any [`Session`]. After each run is saved, when the stored history
//! has grown past a trigger, it replaces the oldest exchanges by a short summary written by a
//! [`Summarizer`] and keeps the most recent exchanges word for word. The model only ever sees an
//! ordinary user message that starts with `<conversation_summary>`, so any provider can read it.
//!
//! Python's counterpart, `OpenAIResponsesCompactionSession`, calls OpenAI's `responses.compact`
//! endpoint and is not portable; this is a new, provider-neutral feature. It never splits a tool
//! call from its output (exchanges are replaced as a whole).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;

use super::{Session, SessionSettings};
use crate::context::{default_token_counter, TokenCounter};
use crate::error::{AgentsError, ModelError};
use crate::items::extract_message_text;
use crate::model::{Model, ModelInput, ModelRequest, ModelTracing};
use crate::model_settings::ModelSettings;
use crate::reducer::{ContextReducer, ReduceInput};
use crate::usage::Usage;

/// Estimated tokens of stored history above which compaction starts, when no trigger is set.
pub const DEFAULT_TRIGGER_TOKENS: usize = 8_000;

/// How long one summarization call may take before it counts as failed.
pub const DEFAULT_SUMMARIZE_TIMEOUT: Duration = Duration::from_secs(60);

/// After an automatic compaction fails, how long later writes skip trying again.
///
/// Without it a summarizer that is down or slow would be called, and waited for, at the end of
/// every single run.
pub const DEFAULT_FAILURE_COOLDOWN: Duration = Duration::from_secs(60);

/// When a stored history is long enough to compact.
///
/// With neither limit set the token limit is [`DEFAULT_TRIGGER_TOKENS`]. With both set, either one
/// fires.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactionTrigger {
    tokens: Option<usize>,
    items: Option<usize>,
}

impl CompactionTrigger {
    /// A trigger with the default token limit.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fire when the history exceeds this many (estimated) tokens.
    pub fn tokens(mut self, tokens: usize) -> Self {
        self.tokens = Some(tokens);
        self
    }

    /// Fire when the history exceeds this many items.
    pub fn items(mut self, items: usize) -> Self {
        self.items = Some(items);
        self
    }

    /// Whether `items` is long enough to compact, counting tokens with `counter`.
    pub fn fires(&self, items: &[Value], counter: &TokenCounter) -> bool {
        let tokens = self
            .tokens
            .or_else(|| self.items.is_none().then_some(DEFAULT_TRIGGER_TOKENS));
        if let Some(max) = tokens {
            if items.iter().map(|item| counter(item)).sum::<usize>() > max {
                return true;
            }
        }
        self.items.is_some_and(|max| items.len() > max)
    }
}

/// Writes the summary that replaces old exchanges.
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

const STRUCTURED_SUMMARY_INSTRUCTIONS: &str = "You condense a conversation between a user and an \
AI assistant into structured notes that let the assistant continue seamlessly. Write plain text, \
no preamble, using exactly these sections, in this order, leaving a section out only when it is \
empty:\n\
Goal: what the user is trying to achieve, and their stated preferences and constraints.\n\
Done: what has been completed, including results of tool calls that still matter.\n\
Decisions: choices made and the reasons for them.\n\
Facts: concrete details to keep verbatim (names, ids, numbers, file paths, URLs).\n\
Open: unfinished tasks and unanswered questions.\n\
If existing notes are given, merge them with the new conversation into one set of notes. Be \
concise.";

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

    /// Summarize with `model` into fixed sections (goal, done, decisions, facts, open items)
    /// instead of free text. Easier to scan, and less likely to lose a constraint.
    pub fn structured(model: Arc<dyn Model>) -> Self {
        Self::new(model).instructions(STRUCTURED_SUMMARY_INSTRUCTIONS)
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
pub(crate) fn truncate_chars(text: &str, max: usize) -> String {
    let total = text.chars().count();
    if total <= max {
        return text.to_string();
    }
    let head: String = text.chars().take(max).collect();
    format!("{head}...[truncated {} chars]", total - max)
}

/// The text of a message-like item, whatever shape its content has.
pub(crate) fn message_text(item: &Value) -> Option<String> {
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
            Some(format!(
                "[tool call] {name}({})",
                truncate_chars(arguments, max_tool_chars)
            ))
        }
        Some("function_call_output") => {
            let output = match item.get("output") {
                Some(Value::String(text)) => text.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            };
            Some(format!(
                "[tool result] {}",
                truncate_chars(&output, max_tool_chars)
            ))
        }
        Some("reasoning") => None,
        _ => {
            let role = item
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("assistant");
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
        self.usage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .add(&response.usage);
        let text: String = response
            .output
            .iter()
            .filter_map(extract_message_text)
            .collect();
        if text.trim().is_empty() {
            return Err(
                ModelError::Behavior("the summarizer model returned no text".into()).into(),
            );
        }
        Ok(text)
    }

    fn usage(&self) -> Usage {
        self.usage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

/// A [`Session`] that keeps the stored history short by summarizing old exchanges.
///
/// ```ignore
/// let session = CompactingSession::new(
///     SqliteSession::open("chat-1", "chats.db")?.shared(),
///     Arc::new(ModelSummarizer::new(model)),
/// )
/// .trigger_tokens(6_000)
/// .keep_recent_exchanges(3)
/// .shared();
/// ```
///
/// Compaction runs inside `add_items`, that is when a run finishes. If it fails (for example the
/// summarizer model is down) a warning is logged and the history stays as it was, so a completed
/// run is never lost. Writes through one `CompactingSession` are serialized; two sessions over
/// the same store can still race, so use one per history.
///
/// A compaction does not count toward `Usage`, because it happens outside the run.
pub struct CompactingSession {
    inner: Arc<dyn Session>,
    summarizer: Option<Arc<dyn Summarizer>>,
    reducer: Option<Arc<dyn ContextReducer>>,
    trigger: CompactionTrigger,
    keep_recent_exchanges: usize,
    counter: TokenCounter,
    summarize_timeout: Duration,
    failure_cooldown: Duration,
    /// When set, automatic compaction is skipped until this instant.
    retry_at: Mutex<Option<Instant>>,
    writes: AsyncMutex<()>,
}

impl std::fmt::Debug for CompactingSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactingSession")
            .field("session_id", &self.inner.session_id())
            .field("trigger", &self.trigger)
            .field("keep_recent_exchanges", &self.keep_recent_exchanges)
            .finish()
    }
}

impl CompactingSession {
    /// Wrap `inner`. Until a trigger is set, compaction starts above
    /// [`DEFAULT_TRIGGER_TOKENS`] estimated tokens; the two most recent exchanges stay verbatim.
    pub fn new(inner: Arc<dyn Session>, summarizer: Arc<dyn Summarizer>) -> Self {
        Self {
            inner,
            summarizer: Some(summarizer),
            reducer: None,
            trigger: CompactionTrigger::new(),
            keep_recent_exchanges: 2,
            counter: default_token_counter(),
            summarize_timeout: DEFAULT_SUMMARIZE_TIMEOUT,
            failure_cooldown: DEFAULT_FAILURE_COOLDOWN,
            retry_at: Mutex::new(None),
            writes: AsyncMutex::new(()),
        }
    }

    /// Wrap `inner` and shorten the stored history with `reducer` instead of summarizing.
    ///
    /// The triggers, [`keep_recent_exchanges`](Self::keep_recent_exchanges) aside, work as with
    /// [`new`](Self::new): the reducer runs when the history is long enough, and its result
    /// replaces the stored history when it changed anything. `summarize_timeout` and the failure
    /// cooldown apply to the reducer too. `keep_recent_exchanges` has no effect; the reducer
    /// decides what to keep.
    pub fn with_reducer(inner: Arc<dyn Session>, reducer: Arc<dyn ContextReducer>) -> Self {
        let mut session = Self::new(inner, Arc::new(NoSummarizer));
        session.summarizer = None;
        session.reducer = Some(reducer);
        session
    }

    /// Give up on a summarization call after `timeout` (default
    /// `DEFAULT_SUMMARIZE_TIMEOUT`).
    pub fn summarize_timeout(mut self, timeout: Duration) -> Self {
        self.summarize_timeout = timeout;
        self
    }

    /// After a failed automatic compaction, skip further automatic attempts for `cooldown`
    /// (default `DEFAULT_FAILURE_COOLDOWN`). An explicit [`compact`](Self::compact) always
    /// tries.
    pub fn failure_cooldown(mut self, cooldown: Duration) -> Self {
        self.failure_cooldown = cooldown;
        self
    }

    fn cooling_down(&self) -> bool {
        let guard = self
            .retry_at
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.is_some_and(|at| Instant::now() < at)
    }

    fn set_cooldown(&self, failed: bool) {
        let at = failed.then(|| Instant::now() + self.failure_cooldown);
        *self
            .retry_at
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = at;
    }

    /// Compact when the stored history exceeds this many (estimated) tokens.
    pub fn trigger_tokens(mut self, tokens: usize) -> Self {
        self.trigger.tokens = Some(tokens);
        self
    }

    /// Compact when the stored history exceeds this many items. With
    /// [`trigger_tokens`](Self::trigger_tokens) also set, either one starts a compaction.
    pub fn trigger_items(mut self, items: usize) -> Self {
        self.trigger.items = Some(items);
        self
    }

    /// Number of most recent exchanges kept word for word (default 2).
    pub fn keep_recent_exchanges(mut self, exchanges: usize) -> Self {
        self.keep_recent_exchanges = exchanges;
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
        self.summarizer
            .as_ref()
            .map(|summarizer| summarizer.usage())
            .unwrap_or_default()
    }

    /// Wrap in an `Arc`, ready for `RunOptions::session`.
    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Compact now, ignoring the triggers. Returns whether anything was summarized: `false`
    /// when there are no more exchanges than `keep_recent_exchanges`.
    pub async fn compact(&self) -> Result<bool, AgentsError> {
        let _writes = self.writes.lock().await;
        self.compact_locked(true).await
    }

    async fn compact_locked(&self, force: bool) -> Result<bool, AgentsError> {
        let items = self.inner.get_items(None).await?;
        if !force && !self.trigger.fires(&items, &self.counter) {
            return Ok(false);
        }
        if let Some(reducer) = &self.reducer {
            return self.reduce_locked(reducer, &items).await;
        }
        let Some(summarizer) = &self.summarizer else {
            return Ok(false);
        };
        let Some((compacted, _)) = super::summary::summarize_history(
            &items,
            summarizer.as_ref(),
            self.keep_recent_exchanges,
            Some(self.summarize_timeout),
        )
        .await?
        else {
            return Ok(false);
        };
        self.inner.replace_items(compacted).await?;
        Ok(true)
    }
}

/// Placeholder that lets `with_reducer` reuse `new`; never called.
struct NoSummarizer;

#[async_trait]
impl Summarizer for NoSummarizer {
    async fn summarize(&self, _: Option<&str>, _: &[Value]) -> Result<String, AgentsError> {
        Err(ModelError::Behavior("no summarizer is configured".into()).into())
    }
}

impl CompactingSession {
    async fn reduce_locked(
        &self,
        reducer: &Arc<dyn ContextReducer>,
        items: &[Value],
    ) -> Result<bool, AgentsError> {
        let _span = crate::tracing::custom_span(&format!("context_reduce:{}", reducer.name()));
        let outcome = tokio::time::timeout(
            self.summarize_timeout,
            reducer.reduce(ReduceInput {
                items,
                reserved_tokens: 0,
                counter: &self.counter,
            }),
        )
        .await
        .map_err(|_| {
            AgentsError::from(ModelError::Behavior(format!(
                "the reducer did not finish within {:?}",
                self.summarize_timeout
            )))
        })??;
        if !outcome.report.changed {
            return Ok(false);
        }
        self.inner.replace_items(outcome.items).await?;
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
        if self.cooling_down() {
            return Ok(());
        }
        match self.compact_locked(false).await {
            Ok(_) => self.set_cooldown(false),
            Err(error) => {
                self.set_cooldown(true);
                ::tracing::warn!(
                    session = self.inner.session_id(),
                    %error,
                    "context compaction failed; the stored history was left unchanged and \
                     automatic compaction pauses for a while"
                );
            }
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

#[cfg(test)]
mod trigger_tests {
    use super::*;
    use serde_json::json;

    fn history(n: usize, text: &str) -> Vec<Value> {
        (0..n)
            .map(|_| json!({"role": "user", "content": text}))
            .collect()
    }

    #[test]
    fn default_limit_is_tokens_only() {
        let counter = default_token_counter();
        let trigger = CompactionTrigger::new();
        assert!(!trigger.fires(&history(100, "hi"), &counter));
        assert!(trigger.fires(&history(10, &"x".repeat(4_000)), &counter));
    }

    #[test]
    fn item_limit_alone_disables_the_default_token_limit() {
        let counter = default_token_counter();
        let trigger = CompactionTrigger::new().items(5);
        assert!(!trigger.fires(&history(5, &"x".repeat(40_000)), &counter));
        assert!(trigger.fires(&history(6, "hi"), &counter));
    }

    #[test]
    fn either_limit_fires() {
        let counter = default_token_counter();
        let trigger = CompactionTrigger::new().tokens(50).items(100);
        assert!(trigger.fires(&history(3, &"x".repeat(400)), &counter));
        let trigger = CompactionTrigger::new().tokens(1_000_000).items(100);
        assert!(!trigger.fires(&history(100, "a"), &counter));
        assert!(trigger.fires(&history(101, "a"), &counter));
    }
}

#[cfg(test)]
mod reducer_session_tests {
    use super::*;
    use crate::context::HeadTail;
    use crate::memory::InMemorySession;
    use serde_json::json;

    fn user(text: &str) -> Value {
        json!({"role": "user", "content": text})
    }

    #[tokio::test]
    async fn with_reducer_replaces_the_stored_history_once_the_trigger_fires() {
        let inner = Arc::new(InMemorySession::new("s"));
        let session = CompactingSession::with_reducer(inner.clone(), Arc::new(HeadTail::new(1, 1)))
            .trigger_items(3);

        session.add_items(vec![user("a"), user("b")]).await.unwrap();
        assert_eq!(
            inner.get_items(None).await.unwrap().len(),
            2,
            "below trigger"
        );

        session.add_items(vec![user("c"), user("d")]).await.unwrap();
        let stored = inner.get_items(None).await.unwrap();
        assert_eq!(stored, vec![user("a"), user("d")]);
        assert_eq!(session.summarizer_usage(), Usage::default());
    }

    #[tokio::test]
    async fn explicit_compact_runs_the_reducer_and_reports_whether_it_changed_anything() {
        let inner = Arc::new(InMemorySession::new("s"));
        let session = CompactingSession::with_reducer(inner.clone(), Arc::new(HeadTail::new(1, 1)));
        session.add_items(vec![user("a"), user("b")]).await.unwrap();
        assert!(!session.compact().await.unwrap());
        session.add_items(vec![user("c")]).await.unwrap();
        assert!(session.compact().await.unwrap());
        assert_eq!(
            inner.get_items(None).await.unwrap(),
            vec![user("a"), user("c")]
        );
    }
}
