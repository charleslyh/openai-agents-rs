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

/// How serious a guardrail failure is (Rust-only, X-012).
///
/// Every failure stops the run. `High` and `Critical` also stop the guardrails still running in
/// the same group, as Python does for any tripwire; `Low` and `Medium` let them finish so their
/// results are reported too.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Severity {
    /// Worth reporting; the other guardrails finish.
    Low,
    /// Clearly unwanted; the other guardrails finish.
    Medium,
    /// Stop now (default, and what [`GuardrailFunctionOutput::trip`] uses).
    #[default]
    High,
    /// Stop now.
    Critical,
}

impl Severity {
    /// Whether a failure of this severity cancels the rest of its group.
    pub fn stops_early(self) -> bool {
        self >= Self::High
    }

    /// Lower-case name, as stored in a `RunState`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }

    /// Parse [`Self::as_str`]; anything else is `None`.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "critical" => Self::Critical,
            _ => return None,
        })
    }
}

/// What a guardrail decided (Rust-only, X-012; Python has only the tripwire flag).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Verdict {
    /// Let the run continue unchanged.
    Pass,
    /// Stop the run (Python: `tripwire_triggered = True`).
    Fail {
        /// Why; may be empty (Python's tripwire carries no reason).
        reason: String,
        /// How serious; decides whether the other guardrails are cancelled.
        severity: Severity,
    },
    /// Let the run continue with this content instead (e.g. personal data redacted).
    ///
    /// For an input guardrail, a JSON string or an array of input items that replaces the run's
    /// new input before the model sees it; only guardrails built with `run_in_parallel(false)`
    /// may transform. For an output guardrail, the new final output.
    Transform(Value),
}

/// Result of a guardrail function (Python: `GuardrailFunctionOutput`).
#[derive(Debug, Clone)]
pub struct GuardrailFunctionOutput {
    /// Optional details about the checks that were performed.
    pub output_info: Value,
    /// The decision.
    pub verdict: Verdict,
}

impl GuardrailFunctionOutput {
    /// A guardrail that did not trigger the tripwire.
    pub fn pass(output_info: Value) -> Self {
        Self {
            output_info,
            verdict: Verdict::Pass,
        }
    }

    /// A guardrail that triggered the tripwire (Python): a [`Verdict::Fail`] with no reason and
    /// [`Severity::High`].
    pub fn trip(output_info: Value) -> Self {
        Self {
            output_info,
            verdict: Verdict::Fail {
                reason: String::new(),
                severity: Severity::High,
            },
        }
    }

    /// Stop the run, saying why and how serious it is.
    pub fn fail(reason: impl Into<String>, severity: Severity) -> Self {
        Self {
            output_info: Value::Null,
            verdict: Verdict::Fail {
                reason: reason.into(),
                severity,
            },
        }
    }

    /// Continue with `content` instead of the checked input or output.
    pub fn transform(content: impl Into<Value>) -> Self {
        Self {
            output_info: Value::Null,
            verdict: Verdict::Transform(content.into()),
        }
    }

    /// Attach details about the checks that ran.
    #[must_use]
    pub fn with_info(mut self, output_info: Value) -> Self {
        self.output_info = output_info;
        self
    }

    /// Whether the guardrail stops the run (Python: `tripwire_triggered`).
    pub fn tripwire_triggered(&self) -> bool {
        matches!(self.verdict, Verdict::Fail { .. })
    }

    /// Whether a failure here cancels the other guardrails of its group.
    pub(crate) fn stops_group(&self) -> bool {
        matches!(&self.verdict, Verdict::Fail { severity, .. } if severity.stops_early())
    }

    /// The replacement content of a [`Verdict::Transform`].
    pub fn transformed(&self) -> Option<&Value> {
        match &self.verdict {
            Verdict::Transform(content) => Some(content),
            _ => None,
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
