//! Writing run items back into a [`Session`].

use std::sync::Arc;

use serde_json::Value;

use crate::error::AgentsError;
use crate::items::{apply_reasoning_item_id_policy, RunItem};
use crate::memory::Session;

pub(crate) struct SessionWriter {
    session: Arc<dyn Session>,
    policy: Option<crate::items::ReasoningItemIdPolicy>,
    /// The run's input, until it has been saved.
    pub(crate) pending_input: Option<Vec<Value>>,
    pub(crate) saved_items: usize,
}

impl SessionWriter {
    pub(crate) fn new(
        session: Arc<dyn Session>,
        policy: Option<crate::items::ReasoningItemIdPolicy>,
    ) -> Self {
        Self {
            session,
            policy,
            pending_input: None,
            saved_items: 0,
        }
    }

    /// Save the turn whose final output an output guardrail withheld.
    ///
    /// Python (`blocked_output.py`) keeps the turn's tool calls but replaces every tool output
    /// with the placeholder, and leaves out the model's own messages. A turn that holds a
    /// reasoning item is not saved at all, because reasoning cannot be replayed without the
    /// items it led up to. Earlier turns were saved when they ended.
    pub(crate) async fn save_blocked_turn(
        &mut self,
        items: &[RunItem],
        placeholder: &str,
    ) -> Result<(), AgentsError> {
        let mut out = self.pending_input.take().unwrap_or_default();
        let turn = &items[self.saved_items.min(items.len())..];
        if !turn.iter().any(|i| matches!(i, RunItem::Reasoning(_))) {
            for item in turn {
                match item {
                    RunItem::ToolCall(_) => out.push(item.raw_item().clone()),
                    RunItem::ToolCallOutput(_) => {
                        let mut raw = item.raw_item().clone();
                        raw["output"] = Value::String(placeholder.to_string());
                        out.push(raw);
                    }
                    _ => {}
                }
            }
        }
        self.saved_items = items.len();
        if !out.is_empty() {
            self.session.add_items(out).await?;
        }
        Ok(())
    }

    /// Save the input (once) and every model-visible item not saved yet.
    pub(crate) async fn flush(&mut self, items: &[RunItem]) -> Result<(), AgentsError> {
        let mut out = self.pending_input.take().unwrap_or_default();
        let start = self.saved_items.min(items.len());
        out.extend(
            items[start..]
                .iter()
                .filter(|i| i.is_model_input())
                .map(|i| apply_reasoning_item_id_policy(i.raw_item(), self.policy)),
        );
        self.saved_items = items.len();
        if !out.is_empty() {
            self.session.add_items(out).await?;
        }
        Ok(())
    }
}
