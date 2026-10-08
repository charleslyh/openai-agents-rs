//! Session memory (Python: `agents.memory`).
//!
//! A [`Session`] stores the conversation items of earlier runs. Pass one through
//! [`crate::RunOptions::session`]: its history is prepended to the run input, and the run's
//! input and generated items are appended once the run finishes.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use crate::error::AgentsError;

mod compaction;
mod history;
pub use compaction::{CompactingSession, ModelSummarizer, Summarizer, DEFAULT_TRIGGER_TOKENS};

#[cfg(feature = "sqlite")]
mod sqlite;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteSession;

/// Settings for session reads (Python: `SessionSettings`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionSettings {
    /// Maximum number of stored items to load; `None` loads everything.
    pub limit: Option<usize>,
}

impl SessionSettings {
    /// Overlay the non-`None` fields of `over` on top of `self` (Python: `resolve`).
    pub fn resolve(self, over: Option<SessionSettings>) -> SessionSettings {
        match over {
            Some(over) => SessionSettings {
                limit: over.limit.or(self.limit),
            },
            None => self,
        }
    }
}

/// Merges stored history with the new turn input (Python: `SessionInputCallback`).
///
/// Receives `(history, new_items)` and returns the exact list sent to the model; items of the
/// result that came from `new_items` are the ones saved to the session afterwards.
pub type SessionInputCallback = Arc<
    dyn Fn(
            Vec<Value>,
            Vec<Value>,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<Value>, AgentsError>> + Send>>
        + Send
        + Sync,
>;

/// Conversation storage used across runs (Python: `Session` protocol).
#[async_trait]
pub trait Session: Send + Sync {
    /// Identifier of this conversation.
    fn session_id(&self) -> &str;

    /// Default read settings of this session (Python: `Session.session_settings`).
    fn session_settings(&self) -> Option<SessionSettings> {
        None
    }

    /// Stored items, oldest first. With `limit`, only the latest `limit` items.
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<Value>, AgentsError>;

    /// Append items to the conversation.
    async fn add_items(&self, items: Vec<Value>) -> Result<(), AgentsError>;

    /// Remove and return the most recent item.
    async fn pop_item(&self) -> Result<Option<Value>, AgentsError>;

    /// Remove every item.
    async fn clear_session(&self) -> Result<(), AgentsError>;

    /// Replace the whole conversation with `items`.
    ///
    /// Used by context compaction to swap old history for a summary. The default clears and then
    /// appends, so a failure in between loses the history; stores that can do better (the bundled
    /// ones) override it with a single atomic write.
    async fn replace_items(&self, items: Vec<Value>) -> Result<(), AgentsError> {
        self.clear_session().await?;
        self.add_items(items).await
    }
}

/// Process-local [`Session`] (Python has `SQLiteSession(":memory:")`; persistence is not ported).
#[derive(Debug)]
pub struct InMemorySession {
    id: String,
    items: Mutex<Vec<Value>>,
}

impl InMemorySession {
    /// Create an empty session.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            id: session_id.into(),
            items: Mutex::new(Vec::new()),
        }
    }

    /// Create an empty session behind an `Arc`, ready for `RunOptions::session`.
    pub fn shared(session_id: impl Into<String>) -> Arc<Self> {
        Arc::new(Self::new(session_id))
    }
}

#[async_trait]
impl Session for InMemorySession {
    fn session_id(&self) -> &str {
        &self.id
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<Value>, AgentsError> {
        let items = self.items.lock().expect("session items");
        Ok(match limit {
            Some(n) if n < items.len() => items[items.len() - n..].to_vec(),
            _ => items.clone(),
        })
    }

    async fn add_items(&self, items: Vec<Value>) -> Result<(), AgentsError> {
        self.items.lock().expect("session items").extend(items);
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<Value>, AgentsError> {
        Ok(self.items.lock().expect("session items").pop())
    }

    async fn clear_session(&self) -> Result<(), AgentsError> {
        self.items.lock().expect("session items").clear();
        Ok(())
    }

    async fn replace_items(&self, items: Vec<Value>) -> Result<(), AgentsError> {
        *self.items.lock().expect("session items") = items;
        Ok(())
    }
}

/// `json.dumps(item, sort_keys=True)` of an item without SDK-only keys, so equal items compare
/// equal whatever their key order (Python: `_session_item_key`).
fn session_item_key(item: &Value) -> String {
    fn canonical(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                Value::Object(keys.into_iter().map(|k| (k.clone(), canonical(&map[k]))).collect())
            }
            Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
            other => other.clone(),
        }
    }
    let mut item = item.clone();
    if let Some(map) = item.as_object_mut() {
        for key in ["_agents_tool_description", "_agents_tool_title", "created_by"] {
            map.remove(key);
        }
    }
    canonical(&item).to_string()
}

/// Key the runner adds to the items it hands a [`SessionInputCallback`], so the items it returns
/// can be traced back to history or to the new input even when the callback rewrites them.
const ORIGIN_KEY: &str = "_agents_session_origin";

/// Build the model input for a fresh run from the session
/// (Python: `prepare_input_with_session`).
///
/// Returns `(prepared, to_save)`: the items sent to the model, and the items of this turn that
/// belong to the new input and must be saved. Without a callback that is history followed by the
/// new input, and the new input alone.
///
/// With a callback, a returned item counts as history or as new by where it came from, even if
/// the callback edited it (Python tracks object identity; here each item handed to the callback
/// carries a hidden [`ORIGIN_KEY`] entry, removed again from everything it returns). That keeps a
/// callback that rewrites old items (shortening tool outputs, say) from having those edited items
/// saved a second time as new. Items without the key (built by the callback) are matched by
/// content against what is left of the history and the new input, then saved.
///
/// The model input is then cleaned: stored function calls without an output are dropped (with the
/// reasoning that led to them), and duplicate items are merged, as Python does.
pub(crate) async fn prepare_input_with_session(
    session: &dyn Session,
    run_settings: Option<SessionSettings>,
    callback: Option<&SessionInputCallback>,
    new_items: Vec<Value>,
) -> Result<(Vec<Value>, Vec<Value>), AgentsError> {
    let settings = session.session_settings().unwrap_or_default().resolve(run_settings);
    let history = session.get_items(settings.limit).await?;

    let (combined, from_history, to_save) = match callback {
        None => {
            let from_history: Vec<bool> = history
                .iter()
                .map(|_| true)
                .chain(new_items.iter().map(|_| false))
                .collect();
            let combined: Vec<Value> = history.into_iter().chain(new_items.iter().cloned()).collect();
            (combined, from_history, new_items)
        }
        Some(callback) => {
            let tag = |items: &[Value], prefix: &str| -> Vec<Value> {
                items
                    .iter()
                    .enumerate()
                    .map(|(i, item)| {
                        let mut item = item.clone();
                        if let Some(map) = item.as_object_mut() {
                            map.insert(ORIGIN_KEY.into(), Value::String(format!("{prefix}:{i}")));
                        }
                        item
                    })
                    .collect()
            };
            let returned = callback(tag(&history, "h"), tag(&new_items, "n")).await?;
            attribute_callback_result(returned, &history, &new_items)
        }
    };

    let prune_outputs = callback.is_none() && settings.limit.is_some();
    let prepared = history::drop_orphan_function_calls(combined, &from_history, prune_outputs);
    Ok((history::deduplicate_input_items_preferring_latest(prepared), to_save))
}

/// Split the items a callback returned into `(items, is_history, to_save)`.
fn attribute_callback_result(
    returned: Vec<Value>,
    history: &[Value],
    new_items: &[Value],
) -> (Vec<Value>, Vec<bool>, Vec<Value>) {
    let count = |items: &[Value]| {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for item in items {
            *counts.entry(session_item_key(item)).or_default() += 1;
        }
        counts
    };
    let mut history_counts = count(history);
    let mut new_counts = count(new_items);
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut combined = Vec::with_capacity(returned.len());
    let mut from_history = Vec::with_capacity(returned.len());
    let mut to_save = Vec::new();
    for mut item in returned {
        let origin = item
            .as_object_mut()
            .and_then(|map| map.remove(ORIGIN_KEY))
            .and_then(|v| v.as_str().map(str::to_string))
            .filter(|origin| seen.insert(origin.clone()));
        let source = origin.as_deref().and_then(|origin| {
            let (kind, index) = origin.split_once(':')?;
            let index: usize = index.parse().ok()?;
            match kind {
                "h" => history.get(index).map(|original| (true, original)),
                "n" => new_items.get(index).map(|original| (false, original)),
                _ => None,
            }
        });
        let is_history = match source {
            Some((true, original)) => {
                decrement(&mut history_counts, original);
                true
            }
            Some((false, original)) => {
                decrement(&mut new_counts, original);
                false
            }
            None => {
                let key = session_item_key(&item);
                if take_one(&mut history_counts, &key) {
                    true
                } else {
                    take_one(&mut new_counts, &key);
                    false
                }
            }
        };
        if !is_history {
            to_save.push(item.clone());
        }
        from_history.push(is_history);
        combined.push(item);
    }
    (combined, from_history, to_save)
}

fn decrement(counts: &mut HashMap<String, usize>, original: &Value) {
    take_one(counts, &session_item_key(original));
}

/// Use up one occurrence of `key`; false when none was left.
fn take_one(counts: &mut HashMap<String, usize>, key: &str) -> bool {
    match counts.get_mut(key) {
        Some(n) if *n > 0 => {
            *n -= 1;
            true
        }
        _ => false,
    }
}
