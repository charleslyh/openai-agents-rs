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

/// Build the model input for a fresh run from the session
/// (Python: `prepare_input_with_session`).
///
/// Returns `(prepared, to_save)`: the items sent to the model, and the items of this turn that
/// belong to the new input and must be saved. Without a callback that is history followed by the
/// new input, and the new input alone. With one, items are attributed by content: a result item
/// equal to a stored one counts as history, otherwise it is new and gets saved.
pub(crate) async fn prepare_input_with_session(
    session: &dyn Session,
    run_settings: Option<SessionSettings>,
    callback: Option<&SessionInputCallback>,
    new_items: Vec<Value>,
) -> Result<(Vec<Value>, Vec<Value>), AgentsError> {
    let settings = session.session_settings().unwrap_or_default().resolve(run_settings);
    let history = session.get_items(settings.limit).await?;
    let Some(callback) = callback else {
        let mut prepared = history;
        prepared.extend(new_items.iter().cloned());
        return Ok((prepared, new_items));
    };

    let mut history_counts: HashMap<String, usize> = HashMap::new();
    for item in &history {
        *history_counts.entry(session_item_key(item)).or_default() += 1;
    }
    let combined = callback(history, new_items).await?;
    let mut to_save = Vec::new();
    for item in &combined {
        match history_counts.get_mut(&session_item_key(item)) {
            Some(count) if *count > 0 => *count -= 1,
            _ => to_save.push(item.clone()),
        }
    }
    Ok((combined, to_save))
}
