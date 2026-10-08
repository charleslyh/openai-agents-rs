//! Session memory (Python: `agents.memory`).
//!
//! A [`Session`] stores the conversation items of earlier runs. Pass one through
//! [`crate::RunOptions::session`]: its history is prepended to the run input, and the run's
//! input and generated items are appended once the run finishes.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use crate::error::AgentsError;

/// Conversation storage used across runs (Python: `Session` protocol).
#[async_trait]
pub trait Session: Send + Sync {
    /// Identifier of this conversation.
    fn session_id(&self) -> &str;

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
