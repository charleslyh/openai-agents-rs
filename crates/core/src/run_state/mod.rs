//! Run state for HITL pause/resume (Python: `agents.run_state.RunState` subset).
//!
//! A run pauses when a tool call needs resolution ([`crate::tool::NeedsResolution`]); the host
//! resolves each paused call ([`RunState::resolve`], or the Python-named `approve` / `reject`) and
//! resumes the run. Supports sticky approvals and JSON round-trip (`to_json` / `from_json`).
//! Schema id: `openai-agents-rs/4` (not the Python 1.18 wire format — see D-012).

use std::collections::HashMap;

mod codec;
mod resolutions;

use codec::{
    deserialize_input, deserialize_interruption_item, deserialize_model_response,
    deserialize_run_item, deserialize_tool_guardrail_output, guardrail_results, serialize_input,
    serialize_interruption_item, serialize_model_response, serialize_run_item,
    serialize_tool_guardrail_output,
};

use serde_json::{json, Value};

use crate::error::{
    AgentsError, GuardrailFunctionOutput, InputGuardrailResult, ToolInputGuardrailResult,
    ToolOutputGuardrailResult, UserError, DEFAULT_APPROVAL_REJECTION_MESSAGE,
};
use crate::items::{InputLike, ModelResponse, ResponseInputItem, RunItem, ToolInterruptionItem};
use crate::model_settings::ModelSettings;
use crate::usage::Usage;

/// Schema version embedded in [`RunState::to_json`].
pub const RUN_STATE_SCHEMA_VERSION: &str = "openai-agents-rs/4";

/// Older schema ids that [`RunState::from_json`] still accepts.
///
/// The `openai-agents-rust/*` ids were emitted before the project was renamed to
/// `openai-agents-rs`; snapshots persisted by earlier versions carry them and must stay
/// loadable. They are accepted but never re-emitted by [`RunState::to_json`].
///
/// `openai-agents-rs/2` is the shape where `max_turns` was always a number. It loads as
/// `Some(n)`; only `/3` and later can carry `null` for an unlimited run (D-042).
///
/// `openai-agents-rs/3` keeps call resolutions under `approvals` and has no output resolutions;
/// `/4` keeps them under `resolutions`, where an entry may carry an `output` (D-048).
pub const SUPPORTED_RUN_STATE_SCHEMAS: &[&str] = &[
    "openai-agents-rust/1",
    "openai-agents-rust/2",
    "openai-agents-rs/2",
    "openai-agents-rs/3",
    RUN_STATE_SCHEMA_VERSION,
];

pub use resolutions::{ResolutionStore, StickyDecision, ToolResolution};

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
    /// Items generated so far (includes pending [`ToolInterruptionItem`]s).
    pub generated_items: Vec<RunItem>,
    /// Raw model responses so far.
    pub raw_responses: Vec<ModelResponse>,
    /// Aggregated usage.
    pub usage: Usage,
    /// Max turns for the run; `None` disables the limit (Python: `RunOptions.max_turns = None`).
    pub max_turns: Option<usize>,
    /// Turn index when interrupted (1-based, matches run loop).
    pub turn: usize,
    /// Previous Responses API response id, if any.
    pub previous_response_id: Option<String>,
    /// Model settings at the point of interruption.
    pub model_settings: ModelSettings,
    /// Tool calls of the interrupted turn that wait for a resolution.
    pub interruptions: Vec<ToolInterruptionItem>,
    /// The model response that produced the pending tool calls.
    pub pending_response: ModelResponse,
    /// Call resolutions (per-call + sticky approvals).
    pub(crate) resolutions: ResolutionStore,
    /// Nested `Agent.as_tool` run states keyed by outer tool call_id.
    pub(crate) nested_agent_runs: HashMap<String, RunState>,
    /// How many of `generated_items` a session already holds. A run with a session saves its
    /// items at the end of every turn, so a paused run has saved the turns before
    /// the paused one, and resuming with the same session must not save them again.
    pub session_saved_items: usize,
    /// Whether the run's input was already saved to the session.
    pub session_input_saved: bool,
    /// Input guardrail results of the run so far.
    pub input_guardrail_results: Vec<InputGuardrailResult>,
    /// Tool input guardrail results of the run so far.
    pub tool_input_guardrail_results: Vec<ToolInputGuardrailResult>,
    /// Tool output guardrail results of the run so far.
    pub tool_output_guardrail_results: Vec<ToolOutputGuardrailResult>,
}

impl RunState {
    /// Pending interruptions (Python: `get_interruptions`).
    pub fn get_interruptions(&self) -> Vec<ToolInterruptionItem> {
        self.interruptions.clone()
    }

    /// Resolve a pending tool call: run it, reject it, or hand back its result.
    pub fn resolve(&mut self, item: &ToolInterruptionItem, resolution: ToolResolution) {
        if let Some(nested) = self.find_nested_mut(item) {
            nested.resolve(item, resolution);
            return;
        }
        self.resolutions.resolve(&item.call_id, resolution);
    }

    /// Resolve a pending tool call with a result produced outside the run; the tool is not run
    /// and the model receives `output` as the call's result.
    pub fn provide_output(&mut self, item: &ToolInterruptionItem, output: Value) {
        self.resolve(item, ToolResolution::Output { output });
    }

    /// Approve a pending tool call (Python: `RunState.approve`).
    ///
    /// When `always_approve` is true, future calls to the same tool on the same agent
    /// are auto-approved for the rest of this run (and survive JSON round-trip).
    pub fn approve(&mut self, item: &ToolInterruptionItem, always_approve: bool) {
        if let Some(nested) = self.find_nested_mut(item) {
            nested.approve(item, always_approve);
            return;
        }
        self.resolutions.approve(
            &item.agent_name,
            &item.tool_name,
            &item.call_id,
            always_approve,
        );
    }

    /// Reject a pending tool call (Python: `RunState.reject`).
    pub fn reject(
        &mut self,
        item: &ToolInterruptionItem,
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
        self.resolutions.reject(
            &item.agent_name,
            &item.tool_name,
            &item.call_id,
            always_reject,
            message,
        );
    }

    /// Look up how a call is resolved (exact call id, then the sticky approval of agent/tool).
    pub fn resolution_status(
        &self,
        agent_name: &str,
        tool_name: &str,
        call_id: &str,
    ) -> Option<ToolResolution> {
        if let Some(nested) = self.find_nested_ref_by_call(call_id) {
            return nested.resolution_status(agent_name, tool_name, call_id);
        }
        self.resolutions.status(agent_name, tool_name, call_id)
    }

    /// Python name of [`Self::resolution_status`].
    pub fn approval_status(
        &self,
        agent_name: &str,
        tool_name: &str,
        call_id: &str,
    ) -> Option<ToolResolution> {
        self.resolution_status(agent_name, tool_name, call_id)
    }

    fn find_nested_mut(&mut self, item: &ToolInterruptionItem) -> Option<&mut RunState> {
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
            || self.resolutions.by_call.contains_key(call_id)
            || self
                .nested_agent_runs
                .values()
                .any(|n| n.contains_call(call_id))
    }

    fn find_nested_ref_by_call(&self, call_id: &str) -> Option<&RunState> {
        for nested in self.nested_agent_runs.values() {
            if nested.contains_call(call_id) {
                if nested.interruptions.iter().any(|i| i.call_id == call_id)
                    || nested.resolutions.by_call.contains_key(call_id)
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
            "interruptions": self.interruptions.iter().map(serialize_interruption_item).collect::<Vec<_>>(),
            "pending_response": serialize_model_response(&self.pending_response),
            "resolutions": self.resolutions.to_json(),
            "nested_agent_runs": self.nested_agent_runs.iter().map(|(k, v)| (k.clone(), v.to_json())).collect::<serde_json::Map<_,_>>(),
            "session_saved_items": self.session_saved_items,
            "session_input_saved": self.session_input_saved,
            "input_guardrail_results": self.input_guardrail_results.iter().map(|r| json!({"guardrail_name": r.guardrail_name, "output": {"output_info": r.output.output_info, "tripwire_triggered": r.output.tripwire_triggered}})).collect::<Vec<_>>(),
            "tool_input_guardrail_results": self.tool_input_guardrail_results.iter().map(|r| json!({"guardrail_name": r.guardrail_name, "output": serialize_tool_guardrail_output(&r.output)})).collect::<Vec<_>>(),
            "tool_output_guardrail_results": self.tool_output_guardrail_results.iter().map(|r| json!({"guardrail_name": r.guardrail_name, "output": serialize_tool_guardrail_output(&r.output)})).collect::<Vec<_>>(),
        })
    }

    /// Serialize to a JSON string (Python: `to_string`).
    ///
    /// Deliberately an inherent method rather than `impl Display`: the name mirrors Python's
    /// `RunState.to_string()` and a `Display` impl would also bring `ToString::to_string`, which
    /// would make the two spellings mean different things.
    #[allow(clippy::inherent_to_string)]
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
        // v1 payloads stay readable: they simply never carry handoff/reasoning items.
        if version != RUN_STATE_SCHEMA_VERSION && !SUPPORTED_RUN_STATE_SCHEMAS.contains(&version) {
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
            // `None` (schema `/3`) disables the limit; a schema `/2` payload always carries a
            // number, which loads as `Some(n)` — that is the value a `/2` snapshot meant.
            max_turns: value
                .get("max_turns")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize),
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
                        .map(deserialize_interruption_item)
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default(),
            pending_response: deserialize_model_response(
                value
                    .get("pending_response")
                    .ok_or_else(|| UserError::new("RunState missing pending_response"))?,
            )?,
            resolutions: ResolutionStore::from_json(
                value
                    .get("resolutions")
                    .or_else(|| value.get("approvals"))
                    .unwrap_or(&json!({})),
            )?,
            nested_agent_runs,
            // Absent in states written before turn-by-turn session saving: nothing was saved.
            session_saved_items: value
                .get("session_saved_items")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
            session_input_saved: value
                .get("session_input_saved")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            input_guardrail_results: guardrail_results(&value, "input_guardrail_results")
                .map(|(name, output)| InputGuardrailResult {
                    guardrail_name: name,
                    output: GuardrailFunctionOutput {
                        output_info: output.get("output_info").cloned().unwrap_or(Value::Null),
                        tripwire_triggered: output
                            .get("tripwire_triggered")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                    },
                })
                .collect(),
            tool_input_guardrail_results: guardrail_results(&value, "tool_input_guardrail_results")
                .map(|(guardrail_name, output)| ToolInputGuardrailResult {
                    guardrail_name,
                    output: deserialize_tool_guardrail_output(&output),
                })
                .collect(),
            tool_output_guardrail_results: guardrail_results(
                &value,
                "tool_output_guardrail_results",
            )
            .map(|(guardrail_name, output)| ToolOutputGuardrailResult {
                guardrail_name,
                output: deserialize_tool_guardrail_output(&output),
            })
            .collect(),
        })
    }

    /// Restore from a JSON string (Python: `from_string`).
    pub fn from_string(starting_agent_name: &str, s: &str) -> Result<Self, AgentsError> {
        let value: Value = serde_json::from_str(s)
            .map_err(|e| UserError::new(format!("invalid RunState JSON: {e}")))?;
        Self::from_json(starting_agent_name, value)
    }
}
