//! The single event type the browser understands, the per-run context, and the mapping from SDK
//! events and errors onto it.

use std::fmt;
use std::sync::atomic::AtomicU32;

use openai_agents::{AgentsError, ModelError, RunItem, StreamEvent, ToolGuardrailBehavior, Usage};
use serde::Serialize;
use serde_json::Value;
use tokio::sync::mpsc;

/// Longest tool argument / output text sent to the browser.
const MAX_TEXT: usize = 2_000;

#[derive(Debug, Clone, Serialize)]
pub struct InterruptionDto {
    pub call_id: String,
    pub agent: String,
    pub tool: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpanDto {
    pub id: String,
    pub parent: Option<String>,
    pub kind: String,
    pub name: String,
    pub start_ms: u128,
    pub end_ms: u128,
    pub error: Option<String>,
}

/// Everything the page can receive, tagged by `type`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DemoEvent {
    RunStarted {
        run_id: String,
        resumed: bool,
    },
    Token {
        text: String,
    },
    Reasoning {
        text: String,
    },
    AgentUpdated {
        agent: String,
    },
    Handoff {
        from: String,
        to: String,
    },
    ToolCalled {
        name: String,
        arguments: String,
        call_id: String,
        source: String,
    },
    ToolOutput {
        call_id: Option<String>,
        output: String,
    },
    Hook {
        name: String,
        agent: String,
        detail: String,
    },
    ToolGuardrail {
        name: String,
        kind: String,
        action: String,
        tool: String,
    },
    FaultInjected {
        layer: String,
        kind: String,
        detail: String,
    },
    ModelAttempt {
        attempt: u32,
        stream: bool,
        outcome: String,
        error_kind: Option<String>,
    },
    StreamInterrupted {
        delivered_chars: usize,
    },
    Interruption {
        run_id: String,
        expires_in_secs: u64,
        items: Vec<InterruptionDto>,
    },
    GuardrailTripped {
        name: String,
        kind: String,
    },
    Session {
        count: usize,
        until_compaction: usize,
        compacted: bool,
        compaction_failed: bool,
        summary: Option<String>,
    },
    Usage {
        requests: u64,
        input_tokens: u64,
        output_tokens: u64,
        total_tokens: u64,
    },
    Final {
        text: String,
        structured: Option<Value>,
        last_agent: String,
    },
    Handled {
        kind: String,
        message: String,
    },
    Trace {
        spans: Vec<SpanDto>,
    },
    Cancelled {
        mode: String,
    },
    Error {
        kind: String,
        message: String,
        retryable: bool,
        hint: String,
        injected: bool,
    },
}

/// Sends events to the SSE stream of the current run. Cloning is cheap.
#[derive(Clone)]
pub struct Emitter {
    tx: mpsc::Sender<DemoEvent>,
}

impl fmt::Debug for Emitter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Emitter")
    }
}

impl Emitter {
    pub fn new(tx: mpsc::Sender<DemoEvent>) -> Self {
        Self { tx }
    }

    /// Queue an event. A closed stream (the browser left) is not an error here: the runner
    /// notices through [`Emitter::closed`] and stops the run.
    pub async fn emit(&self, event: DemoEvent) {
        let _ = self.tx.send(event).await;
    }

    /// Resolves once the browser side of the stream is gone.
    pub async fn closed(&self) {
        self.tx.closed().await;
    }
}

/// Run context handed to hooks, guardrails and tools through `RunOptions.context`.
pub struct DemoCtx {
    pub tenant: String,
    pub emit: Emitter,
    /// Calls of `flaky_inventory` so far in this run.
    pub flaky_calls: AtomicU32,
    /// Tool invocations so far, for the "first N calls fail" fault.
    pub tool_calls: AtomicU32,
}

pub fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max).collect();
    out.push('…');
    out
}

/// Which kind of tool this is, for the badge on the tool card.
pub fn tool_source(name: &str) -> &'static str {
    if name.starts_with("mcp_") {
        "mcp"
    } else if name.starts_with("translate_") {
        "agent_tool"
    } else {
        "function"
    }
}

pub fn usage_event(usage: &Usage) -> DemoEvent {
    DemoEvent::Usage {
        requests: usage.requests,
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
    }
}

/// Map one SDK stream event. `delivered` counts visible characters streamed so far.
pub fn map_stream_event(event: StreamEvent, delivered: &mut usize) -> Vec<DemoEvent> {
    match event {
        StreamEvent::RawResponse { data } => {
            let kind = data.get("type").and_then(Value::as_str).unwrap_or("");
            let delta = data.get("delta").and_then(Value::as_str).unwrap_or("");
            if delta.is_empty() {
                return Vec::new();
            }
            match kind {
                "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                    vec![DemoEvent::Reasoning {
                        text: delta.to_string(),
                    }]
                }
                "response.output_text.delta" => {
                    *delivered += delta.chars().count();
                    vec![DemoEvent::Token {
                        text: delta.to_string(),
                    }]
                }
                _ => Vec::new(),
            }
        }
        StreamEvent::AgentUpdated { agent_name } => {
            vec![DemoEvent::AgentUpdated { agent: agent_name }]
        }
        StreamEvent::RunItem { item, .. } => match &item {
            RunItem::ToolCall(call) => {
                let raw = &call.raw_item;
                let name = raw.get("name").and_then(Value::as_str).unwrap_or("?");
                vec![DemoEvent::ToolCalled {
                    name: name.to_string(),
                    arguments: truncate(
                        raw.get("arguments").and_then(Value::as_str).unwrap_or(""),
                        MAX_TEXT,
                    ),
                    call_id: raw
                        .get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    source: tool_source(name).to_string(),
                }]
            }
            RunItem::ToolCallOutput(out) => {
                let text = match &out.output {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                vec![DemoEvent::ToolOutput {
                    call_id: out
                        .raw_item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    output: truncate(&text, MAX_TEXT),
                }]
            }
            RunItem::HandoffOutput(h) => vec![DemoEvent::Handoff {
                from: h.source_agent_name.clone(),
                to: h.target_agent_name.clone(),
            }],
            _ => Vec::new(),
        },
    }
}

/// An error as the page shows it.
#[derive(Debug, Clone)]
pub struct ErrorInfo {
    pub kind: String,
    pub message: String,
    pub retryable: bool,
    pub hint: String,
    pub injected: bool,
}

impl ErrorInfo {
    fn new(kind: &str, retryable: bool, hint: &str, err: &AgentsError) -> Self {
        let message = err.to_string();
        Self {
            kind: kind.to_string(),
            injected: message.contains(super::faults::INJECTED_TAG),
            message: truncate(&message, 400),
            retryable,
            hint: hint.to_string(),
        }
    }

    pub fn into_event(self) -> DemoEvent {
        DemoEvent::Error {
            kind: self.kind,
            message: self.message,
            retryable: self.retryable,
            hint: self.hint,
            injected: self.injected,
        }
    }
}

/// Sort an SDK error into a kind the page can explain.
pub fn classify(err: &AgentsError) -> ErrorInfo {
    match err {
        AgentsError::Model(ModelError::Status(s)) => match s.status_code {
            429 => ErrorInfo::new(
                "rate_limit",
                true,
                "Rate limited. Choose the `Retry-After` (or `provider suggested`) retry policy in Settings to wait and retry automatically.",
                err,
            ),
            500..=599 => ErrorInfo::new(
                "server_error",
                true,
                "Provider-side failure. An `HTTP status` retry policy retries 5xx responses; otherwise resend later.",
                err,
            ),
            401 | 403 => ErrorInfo::new(
                "auth",
                false,
                "Authentication failed. Check OPENAI_API_KEY. Retrying cannot help, so no policy retries it.",
                err,
            ),
            _ => ErrorInfo::new(
                "bad_request",
                false,
                "The provider rejected the request. Retrying the same request gives the same answer; fix the request.",
                err,
            ),
        },
        AgentsError::Model(ModelError::Connection(c)) if c.is_timeout => ErrorInfo::new(
            "connect_timeout",
            true,
            "The connection timed out. The `network error` retry policy covers this.",
            err,
        ),
        AgentsError::Model(ModelError::Connection(_)) => ErrorInfo::new(
            "connection",
            true,
            "The connection dropped. The `network error` retry policy retries this; otherwise resend.",
            err,
        ),
        AgentsError::Model(ModelError::Timeout(_)) => ErrorInfo::new(
            "attempt_timeout",
            true,
            "One model attempt exceeded the per-attempt timeout. A retry policy can try again.",
            err,
        ),
        AgentsError::Model(ModelError::Behavior(_)) => ErrorInfo::new(
            "bad_response",
            false,
            "The provider returned an unusable payload. Resend, or try another model.",
            err,
        ),
        AgentsError::Model(_) => ErrorInfo::new(
            "model_error",
            false,
            "The model call failed. Resend the message.",
            err,
        ),
        AgentsError::MaxTurns(_) => ErrorInfo::new(
            "max_turns",
            false,
            "The agent used up its turns. Raise max turns in Settings or simplify the request.",
            err,
        ),
        AgentsError::ModelRefusal(_) => ErrorInfo::new(
            "model_refusal",
            false,
            "The model refused to answer. Rephrase the request.",
            err,
        ),
        AgentsError::InputGuardrailTripwire(_) => ErrorInfo::new(
            "guardrail_input",
            false,
            "An input guardrail blocked the message before the model ran.",
            err,
        ),
        AgentsError::OutputGuardrailTripwire(_) => ErrorInfo::new(
            "guardrail_output",
            false,
            "An output guardrail blocked the answer.",
            err,
        ),
        AgentsError::ToolInputGuardrailTripwire(_) => ErrorInfo::new(
            "guardrail_tool_input",
            false,
            "A tool input guardrail stopped the run (raise mode). Use reject-content mode to let the model recover.",
            err,
        ),
        AgentsError::ToolOutputGuardrailTripwire(_) => ErrorInfo::new(
            "guardrail_tool_output",
            false,
            "A tool output guardrail stopped the run (raise mode). Use reject-content mode to let the model recover.",
            err,
        ),
        AgentsError::ToolTimeout(_) => ErrorInfo::new(
            "tool_timeout",
            true,
            "A tool timed out and the run was set to abort. Switch the timeout behavior to `tell the model` to recover.",
            err,
        ),
        AgentsError::Tool { .. } => ErrorInfo::new(
            "tool_error",
            true,
            "A tool failed and aborted the run. Resend the message.",
            err,
        ),
        AgentsError::User(_) | AgentsError::Internal { .. } => ErrorInfo::new(
            "internal",
            false,
            "Unexpected failure in the run. Resend, or start a new conversation.",
            err,
        ),
    }
}

/// The guardrail (and its layer) behind a tripwire error, for the red banner.
pub fn tripwire_event(err: &AgentsError) -> Option<DemoEvent> {
    match err {
        AgentsError::InputGuardrailTripwire(t) => Some(DemoEvent::GuardrailTripped {
            name: t.result.guardrail_name.clone(),
            kind: "input".into(),
        }),
        AgentsError::OutputGuardrailTripwire(t) => Some(DemoEvent::GuardrailTripped {
            name: t.result.guardrail_name.clone(),
            kind: "output".into(),
        }),
        AgentsError::ToolInputGuardrailTripwire(t) => Some(DemoEvent::GuardrailTripped {
            name: t.guardrail_name.clone(),
            kind: "tool_input".into(),
        }),
        AgentsError::ToolOutputGuardrailTripwire(t) => Some(DemoEvent::GuardrailTripped {
            name: t.guardrail_name.clone(),
            kind: "tool_output".into(),
        }),
        _ => None,
    }
}

/// Short label of a tool guardrail decision.
pub fn guardrail_action(behavior: &ToolGuardrailBehavior) -> &'static str {
    match behavior {
        ToolGuardrailBehavior::Allow => "allow",
        ToolGuardrailBehavior::RejectContent { .. } => "reject_content",
        ToolGuardrailBehavior::RaiseException => "raise_exception",
    }
}

/// Short name of a model error, for the attempt timeline.
pub fn model_error_kind(err: &ModelError) -> &'static str {
    match err {
        ModelError::Status(s) => match s.status_code {
            429 => "http_429",
            500..=599 => "http_5xx",
            401 | 403 => "http_auth",
            _ => "http_4xx",
        },
        ModelError::Connection(c) if c.is_timeout => "connect_timeout",
        ModelError::Connection(_) => "connection",
        ModelError::Timeout(_) => "attempt_timeout",
        ModelError::Behavior(_) => "bad_response",
        ModelError::Transport(_) => "transport",
        ModelError::Script(_) => "script",
        ModelError::Unsupported(_) => "unsupported",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openai_agents::{ModelConnectionError, ModelStatusError};
    use std::collections::BTreeMap;

    fn status(code: u16) -> AgentsError {
        AgentsError::Model(ModelError::Status(ModelStatusError {
            status_code: code,
            message: format!("[injected] HTTP {code}"),
            body: Value::Null,
            headers: BTreeMap::new(),
        }))
    }

    #[test]
    fn classifies_status_codes() {
        assert_eq!(classify(&status(429)).kind, "rate_limit");
        assert!(classify(&status(503)).retryable);
        assert!(!classify(&status(401)).retryable);
        assert!(!classify(&status(400)).retryable);
        assert!(classify(&status(429)).injected);
    }

    #[test]
    fn classifies_connection_errors() {
        let timeout = AgentsError::Model(ModelError::Connection(ModelConnectionError {
            message: "x".into(),
            is_timeout: true,
        }));
        assert_eq!(classify(&timeout).kind, "connect_timeout");
        assert!(!classify(&timeout).injected);
    }

    #[test]
    fn truncates_on_char_boundaries() {
        assert_eq!(truncate("héllo", 3), "hél…");
        assert_eq!(truncate("hi", 3), "hi");
    }
}
