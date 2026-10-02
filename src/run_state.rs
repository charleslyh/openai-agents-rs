//! Run state for HITL pause/resume (Python: `agents.run_state.RunState` subset).
//!
//! Supports in-memory sticky approvals and JSON round-trip (`to_json` / `from_json`).
//! Schema id: `openai-agents-rust/1` (not the Python 1.18 wire format — see D-012).

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::error::{AgentsError, UserError};
use crate::items::{
    InputLike, MessageOutputItem, ModelResponse, ResponseInputItem, RunItem, ToolApprovalItem,
    ToolCallItem, ToolCallOutputItem,
};
use crate::model_settings::ModelSettings;
use crate::tool::DEFAULT_APPROVAL_REJECTION_MESSAGE;
use crate::usage::Usage;

/// Schema version embedded in [`RunState::to_json`].
pub const RUN_STATE_SCHEMA_VERSION: &str = "openai-agents-rust/1";

/// Decision recorded for a pending tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalDecision {
    /// Tool may execute.
    Approved,
    /// Tool must not execute; model receives `message`.
    Rejected {
        /// Rejection text sent as `function_call_output`.
        message: String,
    },
}

/// Sticky always-approve / always-reject for `(agent_name, tool_name)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StickyDecision {
    /// Future calls to this tool on this agent are auto-approved.
    AlwaysApprove,
    /// Future calls are auto-rejected with `message`.
    AlwaysReject {
        /// Sticky rejection message.
        message: String,
    },
}

/// Approval table: per-call decisions + sticky tool policies (Python: context approvals subset).
#[derive(Debug, Clone, Default)]
pub struct ApprovalStore {
    /// Exact call_id decisions (win over sticky).
    pub by_call: HashMap<String, ApprovalDecision>,
    /// Sticky decisions keyed by `"agent_name\\0tool_name"`.
    pub sticky: HashMap<String, StickyDecision>,
}

impl ApprovalStore {
    fn sticky_key(agent_name: &str, tool_name: &str) -> String {
        format!("{agent_name}\0{tool_name}")
    }

    /// Resolve approval for a call (exact call_id first, then sticky).
    pub fn status(
        &self,
        agent_name: &str,
        tool_name: &str,
        call_id: &str,
    ) -> Option<ApprovalDecision> {
        if let Some(d) = self.by_call.get(call_id) {
            return Some(d.clone());
        }
        match self.sticky.get(&Self::sticky_key(agent_name, tool_name)) {
            Some(StickyDecision::AlwaysApprove) => Some(ApprovalDecision::Approved),
            Some(StickyDecision::AlwaysReject { message }) => Some(ApprovalDecision::Rejected {
                message: message.clone(),
            }),
            None => None,
        }
    }

    /// Record an approve decision.
    pub fn approve(
        &mut self,
        agent_name: &str,
        tool_name: &str,
        call_id: &str,
        always: bool,
    ) {
        self.by_call
            .insert(call_id.to_string(), ApprovalDecision::Approved);
        if always {
            self.sticky.insert(
                Self::sticky_key(agent_name, tool_name),
                StickyDecision::AlwaysApprove,
            );
        }
    }

    /// Record a reject decision.
    pub fn reject(
        &mut self,
        agent_name: &str,
        tool_name: &str,
        call_id: &str,
        always: bool,
        message: String,
    ) {
        self.by_call.insert(
            call_id.to_string(),
            ApprovalDecision::Rejected {
                message: message.clone(),
            },
        );
        if always {
            self.sticky.insert(
                Self::sticky_key(agent_name, tool_name),
                StickyDecision::AlwaysReject { message },
            );
        }
    }

    fn to_json(&self) -> Value {
        let by_call: Value = self
            .by_call
            .iter()
            .map(|(k, v)| {
                let entry = match v {
                    ApprovalDecision::Approved => json!({"approved": true}),
                    ApprovalDecision::Rejected { message } => {
                        json!({"approved": false, "message": message})
                    }
                };
                (k.clone(), entry)
            })
            .collect::<serde_json::Map<_, _>>()
            .into();
        let sticky: Value = self
            .sticky
            .iter()
            .map(|(k, v)| {
                let parts: Vec<&str> = k.split('\0').collect();
                let (agent, tool) = (parts.first().copied().unwrap_or(""), parts.get(1).copied().unwrap_or(""));
                match v {
                    StickyDecision::AlwaysApprove => json!({
                        "agent": agent,
                        "tool": tool,
                        "approved": true
                    }),
                    StickyDecision::AlwaysReject { message } => json!({
                        "agent": agent,
                        "tool": tool,
                        "approved": false,
                        "message": message
                    }),
                }
            })
            .collect();
        json!({ "by_call": by_call, "sticky": sticky })
    }

    fn from_json(value: &Value) -> Result<Self, AgentsError> {
        let mut store = Self::default();
        if let Some(map) = value.get("by_call").and_then(|v| v.as_object()) {
            for (call_id, entry) in map {
                let approved = entry.get("approved").and_then(|v| v.as_bool()).unwrap_or(false);
                if approved {
                    store
                        .by_call
                        .insert(call_id.clone(), ApprovalDecision::Approved);
                } else {
                    let message = entry
                        .get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or(DEFAULT_APPROVAL_REJECTION_MESSAGE)
                        .to_string();
                    store.by_call.insert(
                        call_id.clone(),
                        ApprovalDecision::Rejected { message },
                    );
                }
            }
        }
        if let Some(arr) = value.get("sticky").and_then(|v| v.as_array()) {
            for entry in arr {
                let agent = entry
                    .get("agent")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| UserError::new("sticky approval missing agent"))?;
                let tool = entry
                    .get("tool")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| UserError::new("sticky approval missing tool"))?;
                let approved = entry.get("approved").and_then(|v| v.as_bool()).unwrap_or(false);
                if approved {
                    store.sticky.insert(
                        Self::sticky_key(agent, tool),
                        StickyDecision::AlwaysApprove,
                    );
                } else {
                    let message = entry
                        .get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or(DEFAULT_APPROVAL_REJECTION_MESSAGE)
                        .to_string();
                    store.sticky.insert(
                        Self::sticky_key(agent, tool),
                        StickyDecision::AlwaysReject { message },
                    );
                }
            }
        }
        Ok(store)
    }
}

/// Snapshot of a paused (or resumable) agent run (Python: `RunState`).
#[derive(Debug, Clone)]
pub struct RunState {
    /// Original user input for this run.
    pub input: InputLike,
    /// Agent that started the run (name).
    pub starting_agent_name: String,
    /// Agent active when paused.
    pub current_agent_name: String,
    /// Conversation items already sent / to send on the next model turn.
    pub current_input_items: Vec<ResponseInputItem>,
    /// Items generated so far (includes pending [`ToolApprovalItem`]s).
    pub generated_items: Vec<RunItem>,
    /// Raw model responses so far.
    pub raw_responses: Vec<ModelResponse>,
    /// Aggregated usage.
    pub usage: Usage,
    /// Max turns for the run.
    pub max_turns: usize,
    /// Turn index when interrupted (1-based, matches run loop).
    pub turn: usize,
    /// Previous Responses API response id, if any.
    pub previous_response_id: Option<String>,
    /// Model settings at the point of interruption.
    pub model_settings: ModelSettings,
    /// Pending approvals for the interrupted turn.
    pub interruptions: Vec<ToolApprovalItem>,
    /// The model response that produced the pending tool calls.
    pub pending_response: ModelResponse,
    /// Approval decisions (per-call + sticky).
    pub(crate) approvals: ApprovalStore,
    /// Nested `Agent.as_tool` run states keyed by outer tool call_id.
    pub(crate) nested_agent_runs: HashMap<String, RunState>,
}

impl RunState {
    /// Pending interruptions (Python: `get_interruptions`).
    pub fn get_interruptions(&self) -> Vec<ToolApprovalItem> {
        self.interruptions.clone()
    }

    /// Approve a pending tool call (Python: `RunState.approve`).
    ///
    /// When `always_approve` is true, future calls to the same tool on the same agent
    /// are auto-approved for the rest of this run (and survive JSON round-trip).
    pub fn approve(&mut self, item: &ToolApprovalItem, always_approve: bool) {
        if let Some(nested) = self.find_nested_mut(item) {
            nested.approve(item, always_approve);
            return;
        }
        self.approvals
            .approve(&item.agent_name, &item.tool_name, &item.call_id, always_approve);
    }

    /// Reject a pending tool call (Python: `RunState.reject`).
    pub fn reject(
        &mut self,
        item: &ToolApprovalItem,
        always_reject: bool,
        rejection_message: Option<&str>,
    ) {
        if let Some(nested) = self.find_nested_mut(item) {
            nested.reject(item, always_reject, rejection_message);
            return;
        }
        let message = rejection_message
            .unwrap_or(DEFAULT_APPROVAL_REJECTION_MESSAGE)
            .to_string();
        self.approvals.reject(
            &item.agent_name,
            &item.tool_name,
            &item.call_id,
            always_reject,
            message,
        );
    }

    /// Look up a decision by call id (and sticky fallback needs agent/tool — prefer [`ApprovalStore::status`]).
    pub fn approval_status(
        &self,
        agent_name: &str,
        tool_name: &str,
        call_id: &str,
    ) -> Option<ApprovalDecision> {
        if let Some(nested) = self.find_nested_ref_by_call(call_id) {
            return nested.approval_status(agent_name, tool_name, call_id);
        }
        self.approvals.status(agent_name, tool_name, call_id)
    }

    #[allow(dead_code)]
    pub(crate) fn nested_agent_runs(&self) -> &HashMap<String, RunState> {
        &self.nested_agent_runs
    }

    fn find_nested_mut(&mut self, item: &ToolApprovalItem) -> Option<&mut RunState> {
        let owned_key = self.nested_agent_runs.iter().find_map(|(k, nested)| {
            if nested.contains_call(item.call_id.as_str()) {
                Some(k.clone())
            } else {
                None
            }
        })?;
        let nested = self.nested_agent_runs.get_mut(&owned_key)?;
        if nested
            .interruptions
            .iter()
            .any(|i| i.call_id == item.call_id)
        {
            Some(nested)
        } else {
            nested.find_nested_mut(item)
        }
    }

    fn contains_call(&self, call_id: &str) -> bool {
        self.interruptions.iter().any(|i| i.call_id == call_id)
            || self.approvals.by_call.contains_key(call_id)
            || self
                .nested_agent_runs
                .values()
                .any(|n| n.contains_call(call_id))
    }

    fn find_nested_ref_by_call(&self, call_id: &str) -> Option<&RunState> {
        for nested in self.nested_agent_runs.values() {
            if nested.contains_call(call_id) {
                if nested.interruptions.iter().any(|i| i.call_id == call_id)
                    || nested.approvals.by_call.contains_key(call_id)
                {
                    return Some(nested);
                }
                return nested.find_nested_ref_by_call(call_id);
            }
        }
        None
    }

    /// Serialize to JSON (Python: `to_json`).
    pub fn to_json(&self) -> Value {
        json!({
            "$schemaVersion": RUN_STATE_SCHEMA_VERSION,
            "starting_agent_name": self.starting_agent_name,
            "current_agent_name": self.current_agent_name,
            "input": serialize_input(&self.input),
            "current_input_items": self.current_input_items,
            "generated_items": self.generated_items.iter().map(serialize_run_item).collect::<Vec<_>>(),
            "raw_responses": self.raw_responses.iter().map(serialize_model_response).collect::<Vec<_>>(),
            "usage": self.usage,
            "max_turns": self.max_turns,
            "turn": self.turn,
            "previous_response_id": self.previous_response_id,
            "model_settings": self.model_settings,
            "interruptions": self.interruptions.iter().map(serialize_approval_item).collect::<Vec<_>>(),
            "pending_response": serialize_model_response(&self.pending_response),
            "approvals": self.approvals.to_json(),
            "nested_agent_runs": self.nested_agent_runs.iter().map(|(k, v)| (k.clone(), v.to_json())).collect::<serde_json::Map<_,_>>(),
        })
    }

    /// Serialize to a JSON string (Python: `to_string`).
    pub fn to_string(&self) -> String {
        self.to_json().to_string()
    }

    /// Restore from JSON (Python: `from_json`).
    ///
    /// `starting_agent_name` in the payload must match `starting_agent.name`.
    pub fn from_json(starting_agent_name: &str, value: Value) -> Result<Self, AgentsError> {
        let version = value
            .get("$schemaVersion")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if version != RUN_STATE_SCHEMA_VERSION {
            return Err(UserError::new(format!(
                "unsupported RunState schema `{version}` (expected {RUN_STATE_SCHEMA_VERSION})"
            ))
            .into());
        }
        let saved_start = value
            .get("starting_agent_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| UserError::new("RunState missing starting_agent_name"))?;
        if saved_start != starting_agent_name {
            return Err(UserError::new(format!(
                "RunState starting agent `{saved_start}` does not match `{starting_agent_name}`"
            ))
            .into());
        }

        let mut nested_agent_runs = HashMap::new();
        if let Some(map) = value.get("nested_agent_runs").and_then(|v| v.as_object()) {
            for (k, v) in map {
                // Nested states keep their own starting_agent_name (the nested agent).
                let nested_start = v
                    .get("starting_agent_name")
                    .and_then(|s| s.as_str())
                    .unwrap_or(starting_agent_name);
                nested_agent_runs.insert(k.clone(), Self::from_json(nested_start, v.clone())?);
            }
        }

        Ok(Self {
            input: deserialize_input(
                value
                    .get("input")
                    .ok_or_else(|| UserError::new("RunState missing input"))?,
            )?,
            starting_agent_name: saved_start.to_string(),
            current_agent_name: value
                .get("current_agent_name")
                .and_then(|v| v.as_str())
                .unwrap_or(saved_start)
                .to_string(),
            current_input_items: value
                .get("current_input_items")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default(),
            generated_items: value
                .get("generated_items")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(deserialize_run_item)
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default(),
            raw_responses: value
                .get("raw_responses")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(deserialize_model_response)
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default(),
            usage: serde_json::from_value(value.get("usage").cloned().unwrap_or(json!({})))
                .unwrap_or_default(),
            max_turns: value
                .get("max_turns")
                .and_then(|v| v.as_u64())
                .unwrap_or(10) as usize,
            turn: value.get("turn").and_then(|v| v.as_u64()).unwrap_or(1) as usize,
            previous_response_id: value
                .get("previous_response_id")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            model_settings: serde_json::from_value(
                value.get("model_settings").cloned().unwrap_or(json!({})),
            )
            .unwrap_or_default(),
            interruptions: value
                .get("interruptions")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(deserialize_approval_item)
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default(),
            pending_response: deserialize_model_response(
                value
                    .get("pending_response")
                    .ok_or_else(|| UserError::new("RunState missing pending_response"))?,
            )?,
            approvals: ApprovalStore::from_json(
                value.get("approvals").unwrap_or(&json!({})),
            )?,
            nested_agent_runs,
        })
    }

    /// Restore from a JSON string (Python: `from_string`).
    pub fn from_string(starting_agent_name: &str, s: &str) -> Result<Self, AgentsError> {
        let value: Value = serde_json::from_str(s)
            .map_err(|e| UserError::new(format!("invalid RunState JSON: {e}")))?;
        Self::from_json(starting_agent_name, value)
    }
}

fn serialize_input(input: &InputLike) -> Value {
    match input {
        InputLike::Text(s) => json!({"type": "text", "text": s}),
        InputLike::Items(items) => json!({"type": "items", "items": items}),
    }
}

fn deserialize_input(value: &Value) -> Result<InputLike, AgentsError> {
    match value.get("type").and_then(|v| v.as_str()) {
        Some("text") => Ok(InputLike::Text(
            value
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        )),
        Some("items") => Ok(InputLike::Items(
            value
                .get("items")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default(),
        )),
        _ => Err(UserError::new("invalid RunState input").into()),
    }
}

fn serialize_model_response(r: &ModelResponse) -> Value {
    json!({
        "output": r.output,
        "usage": r.usage,
        "response_id": r.response_id,
        "request_id": r.request_id,
    })
}

fn deserialize_model_response(value: &Value) -> Result<ModelResponse, AgentsError> {
    Ok(ModelResponse {
        output: value
            .get("output")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default(),
        usage: serde_json::from_value(value.get("usage").cloned().unwrap_or(json!({})))
            .unwrap_or_default(),
        response_id: value
            .get("response_id")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        request_id: value
            .get("request_id")
            .and_then(|v| v.as_str())
            .map(str::to_string),
    })
}

fn serialize_approval_item(item: &ToolApprovalItem) -> Value {
    json!({
        "type": "tool_approval_item",
        "agent_name": item.agent_name,
        "tool_name": item.tool_name,
        "call_id": item.call_id,
        "arguments": item.arguments,
        "raw_item": item.raw_item,
    })
}

fn deserialize_approval_item(value: &Value) -> Result<ToolApprovalItem, AgentsError> {
    Ok(ToolApprovalItem {
        agent_name: value
            .get("agent_name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        tool_name: value
            .get("tool_name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        call_id: value
            .get("call_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        arguments: value
            .get("arguments")
            .and_then(|v| v.as_str())
            .unwrap_or("{}")
            .to_string(),
        raw_item: value.get("raw_item").cloned().unwrap_or(Value::Null),
    })
}

fn serialize_run_item(item: &RunItem) -> Value {
    match item {
        RunItem::Message(m) => json!({
            "kind": "message",
            "agent_name": m.agent_name,
            "raw_item": m.raw_item,
        }),
        RunItem::ToolCall(t) => json!({
            "kind": "tool_call",
            "agent_name": t.agent_name,
            "raw_item": t.raw_item,
        }),
        RunItem::ToolCallOutput(o) => json!({
            "kind": "tool_call_output",
            "agent_name": o.agent_name,
            "raw_item": o.raw_item,
            "output": o.output,
        }),
        RunItem::ToolApproval(a) => json!({
            "kind": "tool_approval",
            "item": serialize_approval_item(a),
        }),
    }
}

fn deserialize_run_item(value: &Value) -> Result<RunItem, AgentsError> {
    match value.get("kind").and_then(|v| v.as_str()) {
        Some("message") => Ok(RunItem::Message(MessageOutputItem {
            agent_name: value
                .get("agent_name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            raw_item: value.get("raw_item").cloned().unwrap_or(Value::Null),
        })),
        Some("tool_call") => Ok(RunItem::ToolCall(ToolCallItem {
            agent_name: value
                .get("agent_name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            raw_item: value.get("raw_item").cloned().unwrap_or(Value::Null),
        })),
        Some("tool_call_output") => Ok(RunItem::ToolCallOutput(ToolCallOutputItem {
            agent_name: value
                .get("agent_name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            raw_item: value.get("raw_item").cloned().unwrap_or(Value::Null),
            output: value.get("output").cloned().unwrap_or(Value::Null),
        })),
        Some("tool_approval") => Ok(RunItem::ToolApproval(deserialize_approval_item(
            value
                .get("item")
                .ok_or_else(|| UserError::new("tool_approval missing item"))?,
        )?)),
        other => Err(UserError::new(format!("unknown RunItem kind: {other:?}")).into()),
    }
}
