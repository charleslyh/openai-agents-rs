//! How paused tool calls are resolved, and the sticky approval policies on top of them.
//!
//! A tool whose [`crate::tool::NeedsResolution`] says so pauses before it runs; the host then
//! resolves the call. Human approval is the common case (`approve` runs the tool, `reject` answers
//! the model with a message), but a host can also run the work elsewhere and hand back its result
//! ([`ToolResolution::Output`]).

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::error::{AgentsError, UserError, DEFAULT_APPROVAL_REJECTION_MESSAGE};

/// How a paused tool call is resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolResolution {
    /// Run the tool (Python: approved).
    Run,
    /// Do not run the tool; the model receives `message` (Python: rejected).
    Reject {
        /// Rejection text sent as `function_call_output`.
        message: String,
    },
    /// Do not run the tool; the model receives `output` as the call's result.
    ///
    /// For work the host carried out itself, e.g. a long job finished outside the run.
    Output {
        /// The call's result, recorded like a tool return value.
        output: Value,
    },
}

/// Python name of [`ToolResolution`] (variants: `Approved` → `Run`, `Rejected` → `Reject`).
pub type ApprovalDecision = ToolResolution;

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

/// Resolution table: per-call resolutions + sticky approval policies.
#[derive(Debug, Clone, Default)]
pub struct ResolutionStore {
    /// Exact call_id resolutions (win over sticky).
    pub by_call: HashMap<String, ToolResolution>,
    /// Sticky decisions keyed by `"agent_name\\0tool_name"`.
    pub sticky: HashMap<String, StickyDecision>,
}

/// Python name of [`ResolutionStore`].
pub type ApprovalStore = ResolutionStore;

impl ResolutionStore {
    fn sticky_key(agent_name: &str, tool_name: &str) -> String {
        format!("{agent_name}\0{tool_name}")
    }

    /// Resolution of a call (exact call_id first, then sticky).
    pub fn status(
        &self,
        agent_name: &str,
        tool_name: &str,
        call_id: &str,
    ) -> Option<ToolResolution> {
        if let Some(d) = self.by_call.get(call_id) {
            return Some(d.clone());
        }
        match self.sticky.get(&Self::sticky_key(agent_name, tool_name)) {
            Some(StickyDecision::AlwaysApprove) => Some(ToolResolution::Run),
            Some(StickyDecision::AlwaysReject { message }) => Some(ToolResolution::Reject {
                message: message.clone(),
            }),
            None => None,
        }
    }

    /// Record how a call is resolved.
    pub fn resolve(&mut self, call_id: &str, resolution: ToolResolution) {
        self.by_call.insert(call_id.to_string(), resolution);
    }

    /// Record an approve decision.
    pub fn approve(&mut self, agent_name: &str, tool_name: &str, call_id: &str, always: bool) {
        self.resolve(call_id, ToolResolution::Run);
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
        self.resolve(
            call_id,
            ToolResolution::Reject {
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
                    ToolResolution::Run => json!({"approved": true}),
                    ToolResolution::Reject { message } => {
                        json!({"approved": false, "message": message})
                    }
                    ToolResolution::Output { output } => json!({"output": output}),
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
                store
                    .by_call
                    .insert(call_id.clone(), resolution_from_json(entry));
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
                let decision = match resolution_from_json(entry) {
                    ToolResolution::Run => StickyDecision::AlwaysApprove,
                    ToolResolution::Reject { message } => StickyDecision::AlwaysReject { message },
                    ToolResolution::Output { .. } => {
                        return Err(UserError::new("sticky approval cannot carry an output").into())
                    }
                };
                store.sticky.insert(Self::sticky_key(agent, tool), decision);
            }
        }
        Ok(store)
    }
}

fn resolution_from_json(entry: &Value) -> ToolResolution {
    if let Some(output) = entry.get("output") {
        return ToolResolution::Output {
            output: output.clone(),
        };
    }
    if entry.get("approved").and_then(|v| v.as_bool()) == Some(true) {
        return ToolResolution::Run;
    }
    ToolResolution::Reject {
        message: entry
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_APPROVAL_REJECTION_MESSAGE)
            .to_string(),
    }
}
