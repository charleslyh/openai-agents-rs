//! How paused tool calls are resolved, and the sticky approval policies on top of them.
//!
//! A tool whose [`crate::tool::NeedsResolution`] says so pauses before it runs; the host then
//! resolves the call. Human approval is the common case (`approve` runs the tool, `reject` answers
//! the model with a message), but a host can also run the work elsewhere and hand back its result
//! ([`ToolResolution::Output`]).

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::error::{AgentsError, UserError, DEFAULT_APPROVAL_REJECTION_MESSAGE};
use crate::items::ToolInterruptionItem;

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

impl ToolResolution {
    /// Do not run the tool; the model receives `message`.
    pub fn reject(message: impl Into<String>) -> Self {
        Self::Reject {
            message: message.into(),
        }
    }

    /// Do not run the tool; the model receives `output` as the call's result.
    pub fn output(output: impl Into<Value>) -> Self {
        Self::Output {
            output: output.into(),
        }
    }
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

/// The call a per-call resolution was given for (Rust-only, X-010).
///
/// A resolution made through [`crate::run_state::RunState`] (`resolve`, `approve`, `reject`,
/// `provide_output`) remembers the agent, tool and exact arguments of the paused call it answered.
/// It is applied only to a call that matches all three, so an approval cannot be carried over to a
/// different call that happens to reuse the call id (some servers number ids per response), nor to
/// arguments edited after the decision. A call that does not match pauses again.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CallBinding {
    agent_name: String,
    tool_name: String,
    arguments: String,
}

impl CallBinding {
    fn of(item: &ToolInterruptionItem) -> Self {
        Self {
            agent_name: item.agent_name.clone(),
            tool_name: item.tool_name.clone(),
            arguments: item.arguments.clone(),
        }
    }

    fn matches(&self, agent_name: &str, tool_name: &str, arguments: &str) -> bool {
        self.agent_name == agent_name && self.tool_name == tool_name && self.arguments == arguments
    }
}

/// Resolution table: per-call resolutions + sticky approval policies.
#[derive(Debug, Clone, Default)]
pub struct ResolutionStore {
    /// Exact call_id resolutions (win over sticky).
    pub by_call: HashMap<String, ToolResolution>,
    /// Sticky decisions keyed by `"agent_name\\0tool_name"`.
    pub sticky: HashMap<String, StickyDecision>,
    /// The call each bound `by_call` entry answered; entries set by call id alone have none.
    bindings: HashMap<String, CallBinding>,
}

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
        self.sticky_status(agent_name, tool_name)
    }

    /// Resolution of `call_id` for a call of `tool_name` on `agent_name` with `arguments`.
    ///
    /// Like [`Self::status`], except that a per-call resolution bound to a different agent, tool
    /// or arguments is ignored (with a warning), so the call falls back to the sticky policy or
    /// pauses again.
    pub(crate) fn status_for_call(
        &self,
        agent_name: &str,
        tool_name: &str,
        call_id: &str,
        arguments: &str,
    ) -> Option<ToolResolution> {
        if let Some(resolution) = self.by_call.get(call_id) {
            match self.bindings.get(call_id) {
                Some(bound) if !bound.matches(agent_name, tool_name, arguments) => {
                    tracing::warn!(
                        call_id,
                        agent = agent_name,
                        tool = tool_name,
                        bound_agent = %bound.agent_name,
                        bound_tool = %bound.tool_name,
                        "ignoring a resolution given for a different call with the same id"
                    );
                }
                _ => return Some(resolution.clone()),
            }
        }
        self.sticky_status(agent_name, tool_name)
    }

    fn sticky_status(&self, agent_name: &str, tool_name: &str) -> Option<ToolResolution> {
        match self.sticky.get(&Self::sticky_key(agent_name, tool_name)) {
            Some(StickyDecision::AlwaysApprove) => Some(ToolResolution::Run),
            Some(StickyDecision::AlwaysReject { message }) => Some(ToolResolution::Reject {
                message: message.clone(),
            }),
            None => None,
        }
    }

    /// Record how a call is resolved, by call id alone (applies to any call with that id).
    pub fn resolve(&mut self, call_id: &str, resolution: ToolResolution) {
        self.bindings.remove(call_id);
        self.by_call.insert(call_id.to_string(), resolution);
    }

    /// Record how the paused call `item` is resolved, bound to its agent, tool and arguments.
    pub(crate) fn resolve_item(&mut self, item: &ToolInterruptionItem, resolution: ToolResolution) {
        self.resolve(&item.call_id, resolution);
        self.bind(item);
    }

    /// [`Self::approve`] bound to the paused call `item`.
    pub(crate) fn approve_item(&mut self, item: &ToolInterruptionItem, always: bool) {
        self.approve(&item.agent_name, &item.tool_name, &item.call_id, always);
        self.bind(item);
    }

    /// [`Self::reject`] bound to the paused call `item`.
    pub(crate) fn reject_item(
        &mut self,
        item: &ToolInterruptionItem,
        always: bool,
        message: String,
    ) {
        self.reject(
            &item.agent_name,
            &item.tool_name,
            &item.call_id,
            always,
            message,
        );
        self.bind(item);
    }

    fn bind(&mut self, item: &ToolInterruptionItem) {
        self.bindings
            .insert(item.call_id.clone(), CallBinding::of(item));
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
                let mut entry = match v {
                    ToolResolution::Run => json!({"approved": true}),
                    ToolResolution::Reject { message } => {
                        json!({"approved": false, "message": message})
                    }
                    ToolResolution::Output { output } => json!({"output": output}),
                };
                if let Some(bound) = self.bindings.get(k) {
                    entry["bound"] = json!({
                        "agent": bound.agent_name,
                        "tool": bound.tool_name,
                        "arguments": bound.arguments,
                    });
                }
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
                // Absent in states written before X-010: those entries stay unbound.
                if let Some(bound) = entry.get("bound") {
                    let field = |name: &str| {
                        bound
                            .get(name)
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                            .ok_or_else(|| {
                                UserError::new(format!(
                                    "resolution for call `{call_id}` has a `bound` without `{name}`"
                                ))
                            })
                    };
                    let binding = CallBinding {
                        agent_name: field("agent")?,
                        tool_name: field("tool")?,
                        arguments: field("arguments")?,
                    };
                    store.bindings.insert(call_id.clone(), binding);
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
