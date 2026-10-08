//! Context management: keep the model input within a budget (provider-neutral).
//!
//! Long conversations and tool-heavy runs grow the input until it is slow, costly or too large for
//! the model. This module offers two kinds of tools, neither of which depends on a provider:
//!
//! * **Trimming** rewrites the input of *one model call* and leaves the stored history alone.
//!   [`ToolOutputTrimmer`] shortens bulky tool outputs of older turns (a port of Python's
//!   `ToolOutputTrimmer`); [`ContextWindowTrimmer`] drops the oldest turns to fit a token or turn
//!   budget. Both turn into a [`CallModelInputFilter`] for `RunConfig.call_model_input_filter`, and
//!   [`chain_input_filters`] runs several in order.
//! * **Compaction** rewrites the *stored* history by summarizing old turns; see
//!   [`crate::memory::CompactingSession`].
//!
//! Python only ships the first tool (and `responses.compact`, which needs OpenAI), so everything
//! else here is new to this SDK. Token counts are estimates (see [`estimate_item_tokens`]) because
//! tokenizers differ per model; pass your own [`TokenCounter`] for exact numbers.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde_json::Value;

use crate::error::{AgentsError, UserError};
use crate::run::{CallModelData, CallModelInputFilter, ModelInputData};

/// Opening marker of the item that stands in for summarized history.
pub(crate) const SUMMARY_OPEN: &str = "<conversation_summary>";
/// Closing marker of the summary item.
pub(crate) const SUMMARY_CLOSE: &str = "</conversation_summary>";
/// First line inside the summary item, telling the model what it is looking at.
pub(crate) const SUMMARY_PREAMBLE: &str =
    "The earlier part of this conversation was condensed into the notes below.";

/// Counts the tokens of one input item.
pub type TokenCounter = Arc<dyn Fn(&Value) -> usize + Send + Sync>;

/// Rough token estimate of an item: ASCII text at about four characters per token and every other
/// character as one token, plus a few tokens of per-message overhead.
///
/// This is deliberately model-agnostic. It over- and under-counts by a fair margin, so leave
/// headroom in budgets or supply an exact [`TokenCounter`].
pub fn estimate_item_tokens(item: &Value) -> usize {
    let text = item.to_string();
    let ascii = text.bytes().filter(u8::is_ascii).count();
    let other = text.chars().count() - text.chars().filter(char::is_ascii).count();
    ascii.div_ceil(4) + other + 4
}

/// The default counter, [`estimate_item_tokens`].
pub fn default_token_counter() -> TokenCounter {
    Arc::new(estimate_item_tokens)
}

fn role_of(item: &Value) -> Option<&str> {
    item.get("role").and_then(Value::as_str)
}

fn plain_content(item: &Value) -> Option<&str> {
    item.get("content").and_then(Value::as_str)
}

/// Whether `item` is the stand-in for summarized history.
pub(crate) fn is_summary_item(item: &Value) -> bool {
    role_of(item) == Some("user")
        && plain_content(item).is_some_and(|text| text.starts_with(SUMMARY_OPEN))
}

/// The summary item for `summary` (a plain user message so every provider accepts it).
pub(crate) fn summary_item(summary: &str) -> Value {
    serde_json::json!({
        "role": "user",
        "content": format!("{SUMMARY_OPEN}\n{SUMMARY_PREAMBLE}\n{}\n{SUMMARY_CLOSE}", summary.trim()),
    })
}

/// The summary text carried by a summary item.
pub(crate) fn summary_text(item: &Value) -> Option<String> {
    let text = plain_content(item)?.strip_prefix(SUMMARY_OPEN)?;
    let text = text.trim().strip_suffix(SUMMARY_CLOSE).unwrap_or(text).trim();
    Some(text.strip_prefix(SUMMARY_PREAMBLE).unwrap_or(text).trim().to_string())
}

/// System / developer messages and the summary stay in front of the conversation whatever happens.
fn is_pinned(item: &Value) -> bool {
    matches!(role_of(item), Some("system" | "developer")) || is_summary_item(item)
}

fn starts_turn(item: &Value) -> bool {
    role_of(item) == Some("user") && !is_summary_item(item)
}

/// History split into the items that always stay and the turns that can be dropped or summarized.
#[derive(Debug, Clone, Default)]
pub(crate) struct Conversation {
    /// System / developer messages and summaries, in order.
    pub pinned: Vec<Value>,
    /// Turns, oldest first. A turn begins at a user message and holds everything up to the next
    /// one, so a tool call and its output are never separated.
    pub turns: Vec<Vec<Value>>,
}

impl Conversation {
    pub(crate) fn split(items: &[Value]) -> Self {
        let mut conversation = Conversation::default();
        for item in items {
            if is_pinned(item) {
                conversation.pinned.push(item.clone());
            } else if starts_turn(item) || conversation.turns.is_empty() {
                conversation.turns.push(vec![item.clone()]);
            } else if let Some(turn) = conversation.turns.last_mut() {
                turn.push(item.clone());
            }
        }
        conversation
    }

    pub(crate) fn join(self) -> Vec<Value> {
        self.pinned.into_iter().chain(self.turns.into_iter().flatten()).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Tool output trimming

/// Shortens large tool outputs of older turns (Python: `ToolOutputTrimmer`).
///
/// The last `recent_turns` user messages and everything after them are never touched. In older
/// turns, a `function_call_output` longer than `max_output_chars` is replaced by a preview of its
/// first `preview_chars` characters. Items are copied, never mutated.
///
/// Not ported: Python also shrinks structured (multi-part) outputs and `tool_search_output`
/// items; those exist only in OpenAI-hosted flows, so such outputs are left as they are.
#[derive(Debug, Clone)]
pub struct ToolOutputTrimmer {
    recent_turns: usize,
    max_output_chars: usize,
    preview_chars: usize,
    trimmable_tools: Option<HashSet<String>>,
}

impl Default for ToolOutputTrimmer {
    fn default() -> Self {
        Self {
            recent_turns: 2,
            max_output_chars: 500,
            preview_chars: 200,
            trimmable_tools: None,
        }
    }
}

impl ToolOutputTrimmer {
    /// Defaults: 2 recent turns, 500 character limit, 200 character preview, every tool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of recent user messages (with what follows them) that stay intact. At least 1.
    pub fn recent_turns(mut self, turns: usize) -> Self {
        self.recent_turns = turns;
        self
    }

    /// Outputs longer than this many characters are candidates for trimming. At least 1.
    pub fn max_output_chars(mut self, chars: usize) -> Self {
        self.max_output_chars = chars;
        self
    }

    /// Characters of the original output kept as a preview.
    pub fn preview_chars(mut self, chars: usize) -> Self {
        self.preview_chars = chars;
        self
    }

    /// Only trim the outputs of these tools. Without this every tool is eligible.
    pub fn trimmable_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.trimmable_tools = Some(tools.into_iter().map(Into::into).collect());
        self
    }

    /// Check the settings (Python raises `ValueError` from `__post_init__`).
    pub fn validate(&self) -> Result<(), AgentsError> {
        if self.recent_turns < 1 {
            return Err(UserError::new("recent_turns must be >= 1").into());
        }
        if self.max_output_chars < 1 {
            return Err(UserError::new("max_output_chars must be >= 1").into());
        }
        Ok(())
    }

    /// Index of the `recent_turns`-th user message from the end; items before it are old.
    /// 0 when there are fewer user messages than that, meaning nothing is old.
    fn recent_boundary(&self, items: &[Value]) -> usize {
        let mut seen = 0;
        for (index, item) in items.iter().enumerate().rev() {
            if role_of(item) == Some("user") {
                seen += 1;
                if seen >= self.recent_turns {
                    return index;
                }
            }
        }
        0
    }

    /// Trim `items`, returning the new list.
    pub fn trim(&self, items: &[Value]) -> Vec<Value> {
        let boundary = self.recent_boundary(items);
        if boundary == 0 {
            return items.to_vec();
        }
        let mut tool_names: HashMap<&str, &str> = HashMap::new();
        for item in items {
            if item.get("type").and_then(Value::as_str) == Some("function_call") {
                if let (Some(call_id), Some(name)) = (
                    item.get("call_id").and_then(Value::as_str),
                    item.get("name").and_then(Value::as_str),
                ) {
                    tool_names.insert(call_id, name);
                }
            }
        }
        items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                if index >= boundary
                    || item.get("type").and_then(Value::as_str) != Some("function_call_output")
                {
                    return item.clone();
                }
                let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                let tool = tool_names.get(call_id).copied().unwrap_or("");
                if let Some(allowed) = &self.trimmable_tools {
                    if !allowed.contains(tool) {
                        return item.clone();
                    }
                }
                self.trim_output(item, tool).unwrap_or_else(|| item.clone())
            })
            .collect()
    }

    /// Replacement for an oversized string output, or `None` to keep the original.
    fn trim_output(&self, item: &Value, tool: &str) -> Option<Value> {
        let output = item.get("output")?.as_str()?;
        let length = output.chars().count();
        if length <= self.max_output_chars {
            return None;
        }
        let preview: String = output.chars().take(self.preview_chars).collect();
        let name = if tool.is_empty() { "unknown_tool" } else { tool };
        let summary = format!(
            "[Trimmed: {name} output \u{2014} {length} chars \u{2192} {} char preview]\n{preview}...",
            self.preview_chars
        );
        // Never swap in something longer than what it replaces.
        if summary.chars().count() >= length {
            return None;
        }
        let mut trimmed = item.clone();
        trimmed["output"] = Value::String(summary);
        Some(trimmed)
    }

    /// Use as `RunConfig.call_model_input_filter`.
    pub fn into_filter(self) -> CallModelInputFilter {
        input_filter(move |input, _| self.trim(input))
    }
}

// ---------------------------------------------------------------------------------------------
// Context window trimming

/// Drops the oldest turns so the input fits a budget.
///
/// Whole turns are dropped, oldest first, so a tool call is never separated from its output. Always
/// kept: system and developer messages, a conversation summary, and the latest turn (even when it
/// alone exceeds the budget; shorten tool outputs with [`ToolOutputTrimmer`] first for that case).
/// Nothing tells the model that turns were dropped, so combine with a
/// [`crate::memory::CompactingSession`] when earlier context matters.
#[derive(Clone)]
pub struct ContextWindowTrimmer {
    max_tokens: Option<usize>,
    max_turns: Option<usize>,
    counter: TokenCounter,
}

impl std::fmt::Debug for ContextWindowTrimmer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextWindowTrimmer")
            .field("max_tokens", &self.max_tokens)
            .field("max_turns", &self.max_turns)
            .finish()
    }
}

impl Default for ContextWindowTrimmer {
    fn default() -> Self {
        Self {
            max_tokens: None,
            max_turns: None,
            counter: default_token_counter(),
        }
    }
}

impl ContextWindowTrimmer {
    /// A trimmer without limits; add [`max_tokens`](Self::max_tokens) and/or
    /// [`max_turns`](Self::max_turns).
    pub fn new() -> Self {
        Self::default()
    }

    /// Keep the input within this many (estimated) tokens, instructions included.
    pub fn max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Keep at most this many turns (at least 1).
    pub fn max_turns(mut self, turns: usize) -> Self {
        self.max_turns = Some(turns.max(1));
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

    /// Trim `items`; `reserved_tokens` is what instructions and tool schemas already use.
    pub fn trim(&self, items: &[Value], reserved_tokens: usize) -> Vec<Value> {
        let conversation = Conversation::split(items);
        let Some((latest, earlier)) = conversation.turns.split_last() else {
            return items.to_vec();
        };
        let count = |turn: &[Value]| turn.iter().map(|item| (self.counter)(item)).sum::<usize>();
        let mut used = reserved_tokens
            + conversation.pinned.iter().map(|item| (self.counter)(item)).sum::<usize>()
            + count(latest);
        let mut kept = 1;
        let mut first_kept = earlier.len();
        for (index, turn) in earlier.iter().enumerate().rev() {
            let cost = count(turn);
            let fits_tokens = self.max_tokens.is_none_or(|max| used + cost <= max);
            let fits_turns = self.max_turns.is_none_or(|max| kept < max);
            if !(fits_tokens && fits_turns) {
                break;
            }
            used += cost;
            kept += 1;
            first_kept = index;
        }
        let mut result = conversation.pinned.clone();
        result.extend(earlier[first_kept..].iter().flatten().cloned());
        result.extend(latest.iter().cloned());
        result
    }

    /// Use as `RunConfig.call_model_input_filter`.
    pub fn into_filter(self) -> CallModelInputFilter {
        let counter = Arc::clone(&self.counter);
        Arc::new(move |data: CallModelData| {
            let trimmer = self.clone();
            let counter = Arc::clone(&counter);
            Box::pin(async move {
                let reserved = data
                    .model_data
                    .instructions
                    .as_ref()
                    .map(|text| counter(&Value::String(text.clone())))
                    .unwrap_or(0);
                Ok(ModelInputData {
                    input: trimmer.trim(&data.model_data.input, reserved),
                    instructions: data.model_data.instructions,
                })
            })
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Composition

/// Build a [`CallModelInputFilter`] from a synchronous rewrite of the input items. The second
/// argument is the instructions, for filters that need them.
pub fn input_filter<F>(rewrite: F) -> CallModelInputFilter
where
    F: Fn(&[Value], Option<&str>) -> Vec<Value> + Send + Sync + 'static,
{
    Arc::new(move |data: CallModelData| {
        let input = rewrite(&data.model_data.input, data.model_data.instructions.as_deref());
        Box::pin(async move {
            Ok(ModelInputData {
                input,
                instructions: data.model_data.instructions,
            })
        })
    })
}

/// Run `filters` one after another, each seeing the previous one's output.
pub fn chain_input_filters(filters: Vec<CallModelInputFilter>) -> CallModelInputFilter {
    let filters = Arc::new(filters);
    Arc::new(move |mut data: CallModelData| {
        let filters = Arc::clone(&filters);
        Box::pin(async move {
            for filter in filters.iter() {
                data.model_data = filter(data.clone()).await?;
            }
            Ok(data.model_data)
        })
    })
}
