//! Approval decisions recorded for pending tool calls, and the sticky policies on top of them.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::error::{AgentsError, UserError, DEFAULT_APPROVAL_REJECTION_MESSAGE};

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
    pub fn approve(&mut self, agent_name: &str, tool_name: &str, call_id: &str, always: bool) {
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

    pub(super) fn to_json(&self) -> Value {
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
                let (agent, tool) = (
                    parts.first().copied().unwrap_or(""),
                    parts.get(1).copied().unwrap_or(""),
                );
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

    pub(super) fn from_json(value: &Value) -> Result<Self, AgentsError> {
        let mut store = Self::default();
        if let Some(map) = value.get("by_call").and_then(|v| v.as_object()) {
            for (call_id, entry) in map {
                let approved = entry
                    .get("approved")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
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
                    store
                        .by_call
                        .insert(call_id.clone(), ApprovalDecision::Rejected { message });
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
                let approved = entry
                    .get("approved")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if approved {
                    store
                        .sticky
                        .insert(Self::sticky_key(agent, tool), StickyDecision::AlwaysApprove);
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
