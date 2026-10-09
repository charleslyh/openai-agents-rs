//! Plain data shared by the error types and the run state, plus the default texts that go with it.
//!
//! They live next to [`super::AgentsError`] so the error module stays a leaf: these types only
//! need `serde_json`, while the modules that *produce* them (`guardrail`, `tool_guardrails`,
//! `tool`) are higher up and re-export them for backwards compatibility. `run_state` reads them
//! from here too, which keeps it out of the agent/tool dependency cycle.

use serde_json::Value;

/// What a tool guardrail decided to do (Python: `ToolGuardrailBehavior`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ToolGuardrailBehavior {
    /// Carry on normally (default).
    #[default]
    Allow,
    /// Skip the call (input) or discard the result (output) and send `message` to the model.
    RejectContent {
        /// Text the model sees instead of the tool result.
        message: String,
    },
    /// Stop the run with a tripwire error.
    RaiseException,
}

/// Result of a tool guardrail function (Python: `ToolGuardrailFunctionOutput`).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolGuardrailFunctionOutput {
    /// Free-form details about the checks that ran.
    pub output_info: Value,
    /// What to do about it.
    pub behavior: ToolGuardrailBehavior,
}

impl ToolGuardrailFunctionOutput {
    /// Allow the call to continue.
    pub fn allow(output_info: Value) -> Self {
        Self {
            output_info,
            behavior: ToolGuardrailBehavior::Allow,
        }
    }

    /// Reject the call or result and tell the model `message` instead.
    pub fn reject_content(message: impl Into<String>, output_info: Value) -> Self {
        Self {
            output_info,
            behavior: ToolGuardrailBehavior::RejectContent {
                message: message.into(),
            },
        }
    }

    /// Stop the run.
    pub fn raise_exception(output_info: Value) -> Self {
        Self {
            output_info,
            behavior: ToolGuardrailBehavior::RaiseException,
        }
    }
}

/// Result of a guardrail function (Python: `GuardrailFunctionOutput`).
#[derive(Debug, Clone)]
pub struct GuardrailFunctionOutput {
    /// Optional details about the checks that were performed.
    pub output_info: Value,
    /// Whether the guardrail halted the run.
    pub tripwire_triggered: bool,
}

impl GuardrailFunctionOutput {
    /// A guardrail that did not trigger the tripwire.
    pub fn pass(output_info: Value) -> Self {
        Self {
            output_info,
            tripwire_triggered: false,
        }
    }

    /// A guardrail that triggered the tripwire.
    pub fn trip(output_info: Value) -> Self {
        Self {
            output_info,
            tripwire_triggered: true,
        }
    }
}

/// Result of running an input guardrail (Python: `InputGuardrailResult`).
#[derive(Debug, Clone)]
pub struct InputGuardrailResult {
    /// Guardrail name.
    pub guardrail_name: String,
    /// Guardrail output.
    pub output: GuardrailFunctionOutput,
}

/// Result of running an output guardrail (Python: `OutputGuardrailResult`).
#[derive(Debug, Clone)]
pub struct OutputGuardrailResult {
    /// Guardrail name.
    pub guardrail_name: String,
    /// Agent whose output was checked.
    pub agent_name: String,
    /// The checked output.
    pub agent_output: Value,
    /// Guardrail output.
    pub output: GuardrailFunctionOutput,
}

/// `Tool '<name>' timed out after <n> seconds.` (Python: `default_tool_timeout_error_message`,
/// formatted with `:g`, so `2.0` prints as `2`).
pub fn default_tool_timeout_error_message(tool_name: &str, timeout_seconds: f64) -> String {
    format!("Tool '{tool_name}' timed out after {timeout_seconds} seconds.")
}

/// Result of one tool input guardrail run (Python: `ToolInputGuardrailResult`).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolInputGuardrailResult {
    /// Name of the guardrail that ran.
    pub guardrail_name: String,
    /// What it returned.
    pub output: ToolGuardrailFunctionOutput,
}

/// Result of one tool output guardrail run (Python: `ToolOutputGuardrailResult`).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutputGuardrailResult {
    /// Name of the guardrail that ran.
    pub guardrail_name: String,
    /// What it returned.
    pub output: ToolGuardrailFunctionOutput,
}

/// Default rejection text when a tool call is rejected (Python: `DEFAULT_APPROVAL_REJECTION_MESSAGE`).
pub const DEFAULT_APPROVAL_REJECTION_MESSAGE: &str = "Tool execution was not approved.";
