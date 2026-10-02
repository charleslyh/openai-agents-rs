//! Streaming events (Python: `agents.stream_events`).

use serde_json::Value;

use crate::items::RunItem;

/// Semantic stream event names for [`StreamEvent::RunItem`] (Python: `RunItemStreamEvent.name`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunItemStreamName {
    /// Assistant message created.
    MessageOutputCreated,
    /// Function/tool call requested.
    ToolCalled,
    /// Tool output produced.
    ToolOutput,
    /// Handoff tool call requested.
    HandoffRequested,
    /// Handoff completed (Python spelling: `handoff_occured`).
    HandoffOccured,
}

impl RunItemStreamName {
    /// Python wire / public name string.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MessageOutputCreated => "message_output_created",
            Self::ToolCalled => "tool_called",
            Self::ToolOutput => "tool_output",
            Self::HandoffRequested => "handoff_requested",
            Self::HandoffOccured => "handoff_occured",
        }
    }
}

/// A streaming event from an agent run (Python: `StreamEvent`).
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Raw model stream / response payload (JSON).
    ///
    /// Chat Completions streaming emits:
    /// - `{"type":"reasoning_text.delta","delta":"..."}` (optional; DeepSeek-style CoT)
    /// - `{"type":"output_text.delta","delta":"..."}` visible tokens
    /// - then a `response.completed` object.
    /// Scripted / Responses models may only emit completed.
    RawResponse {
        /// Payload data.
        data: Value,
    },
    /// A run item was created.
    RunItem {
        /// Event name.
        name: RunItemStreamName,
        /// The item.
        item: RunItem,
    },
    /// The active agent changed (or the run started).
    AgentUpdated {
        /// New agent name.
        agent_name: String,
    },
}
