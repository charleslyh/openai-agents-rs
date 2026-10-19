//! Strongly typed view over the `Value` carried by [`crate::StreamEvent::RawResponse`].
//!
//! `StreamEvent::RawResponse.data` stays a raw `Value` so provider events are forwarded verbatim
//! (new event types and fields are never lost). [`parse`] is an opt-in, read-only view: it never
//! mutates the source and falls back to [`WireEvent::Unknown`] for any event it cannot model.

// Variant fields mirror the wire field names one-to-one; documenting each is noise.
#![allow(missing_docs)]

use serde::Deserialize;
use serde_json::Value;

/// A Responses API wire event, covering the vocabulary this crate emits.
///
/// Unknown `type`s, and known types whose fields do not match, become [`WireEvent::Unknown`]
/// holding the original JSON. Fields not listed here are ignored; read them from the source
/// `Value` when needed.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type")]
pub enum WireEvent {
    /// `response.created`.
    #[serde(rename = "response.created")]
    Created {
        response: Value,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.output_item.added`.
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        #[serde(default)]
        output_index: usize,
        item: Value,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.output_item.done`.
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        #[serde(default)]
        output_index: usize,
        item: Value,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.content_part.added`.
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        #[serde(default)]
        item_id: String,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        content_index: usize,
        part: Value,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.content_part.done`.
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        #[serde(default)]
        item_id: String,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        content_index: usize,
        part: Value,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.output_text.delta`.
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        #[serde(default)]
        item_id: String,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        content_index: usize,
        delta: String,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.output_text.done`.
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        #[serde(default)]
        item_id: String,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        content_index: usize,
        text: String,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.refusal.delta`.
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta {
        #[serde(default)]
        item_id: String,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        content_index: usize,
        delta: String,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.refusal.done`.
    #[serde(rename = "response.refusal.done")]
    RefusalDone {
        #[serde(default)]
        item_id: String,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        content_index: usize,
        refusal: String,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.reasoning_summary_part.added`.
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded {
        #[serde(default)]
        item_id: String,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        summary_index: usize,
        part: Value,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.reasoning_summary_part.done`.
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone {
        #[serde(default)]
        item_id: String,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        summary_index: usize,
        part: Value,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.reasoning_summary_text.delta`.
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        #[serde(default)]
        item_id: String,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        summary_index: usize,
        delta: String,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.reasoning_summary_text.done`.
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone {
        #[serde(default)]
        item_id: String,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        summary_index: usize,
        text: String,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.reasoning_text.delta`.
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        #[serde(default)]
        item_id: String,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        content_index: usize,
        delta: String,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.reasoning_text.done`.
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone {
        #[serde(default)]
        item_id: String,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        content_index: usize,
        text: String,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.function_call_arguments.delta`.
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        #[serde(default)]
        item_id: String,
        #[serde(default)]
        output_index: usize,
        delta: String,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.function_call_arguments.done`.
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        #[serde(default)]
        item_id: String,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        name: String,
        arguments: String,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// `response.completed` — the terminal event.
    #[serde(rename = "response.completed")]
    Completed {
        response: Value,
        #[serde(default)]
        sequence_number: Option<u64>,
    },
    /// Any event this enum does not model; carries the original JSON.
    #[serde(skip)]
    Unknown(Value),
}

impl WireEvent {
    /// The event's `sequence_number`, when present.
    pub fn sequence_number(&self) -> Option<u64> {
        match self {
            Self::Created {
                sequence_number, ..
            }
            | Self::OutputItemAdded {
                sequence_number, ..
            }
            | Self::OutputItemDone {
                sequence_number, ..
            }
            | Self::ContentPartAdded {
                sequence_number, ..
            }
            | Self::ContentPartDone {
                sequence_number, ..
            }
            | Self::OutputTextDelta {
                sequence_number, ..
            }
            | Self::OutputTextDone {
                sequence_number, ..
            }
            | Self::RefusalDelta {
                sequence_number, ..
            }
            | Self::RefusalDone {
                sequence_number, ..
            }
            | Self::ReasoningSummaryPartAdded {
                sequence_number, ..
            }
            | Self::ReasoningSummaryPartDone {
                sequence_number, ..
            }
            | Self::ReasoningSummaryTextDelta {
                sequence_number, ..
            }
            | Self::ReasoningSummaryTextDone {
                sequence_number, ..
            }
            | Self::ReasoningTextDelta {
                sequence_number, ..
            }
            | Self::ReasoningTextDone {
                sequence_number, ..
            }
            | Self::FunctionCallArgumentsDelta {
                sequence_number, ..
            }
            | Self::FunctionCallArgumentsDone {
                sequence_number, ..
            }
            | Self::Completed {
                sequence_number, ..
            } => *sequence_number,
            Self::Unknown(value) => value.get("sequence_number").and_then(Value::as_u64),
        }
    }
}

/// Parse one `RawResponse` payload into a [`WireEvent`].
///
/// Never fails: anything unrecognised becomes [`WireEvent::Unknown`] with the original JSON.
pub fn parse(value: &Value) -> WireEvent {
    WireEvent::deserialize(value).unwrap_or_else(|_| WireEvent::Unknown(value.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_known_event() {
        let ev = parse(&json!({
            "type": "response.output_text.delta",
            "item_id": "i", "output_index": 0, "content_index": 1,
            "delta": "hi", "logprobs": [], "sequence_number": 3
        }));
        assert_eq!(
            ev,
            WireEvent::OutputTextDelta {
                item_id: "i".into(),
                output_index: 0,
                content_index: 1,
                delta: "hi".into(),
                sequence_number: Some(3),
            }
        );
        assert_eq!(ev.sequence_number(), Some(3));
    }

    #[test]
    fn unknown_type_keeps_original_json() {
        let raw = json!({"type": "response.foo.bar", "x": 1, "sequence_number": 9});
        let ev = parse(&raw);
        assert_eq!(ev, WireEvent::Unknown(raw));
        assert_eq!(ev.sequence_number(), Some(9));
    }

    #[test]
    fn malformed_known_type_falls_back() {
        let raw = json!({"type": "response.output_text.delta"});
        assert_eq!(parse(&raw), WireEvent::Unknown(raw));
    }
}
