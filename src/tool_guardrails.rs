//! Guardrails around a single function tool call (Python: `agents.tool_guardrails`).
//!
//! An input guardrail runs before the tool body, an output guardrail after it. Each one decides
//! with a [`ToolGuardrailBehavior`]: let the call through, replace the call or its result with a
//! message for the model, or stop the run.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::agent::Agent;
use crate::tool::ToolContext;

/// How a tool guardrail result is acted on (Python: `Allow` / `RejectContent` / `RaiseException`
/// behavior).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
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

/// Data given to a tool input guardrail (Python: `ToolInputGuardrailData`).
#[derive(Debug, Clone)]
pub struct ToolInputGuardrailData {
    /// The tool call being checked.
    pub context: ToolContext,
    /// The agent executing the tool.
    pub agent: Arc<Agent>,
}

/// Data given to a tool output guardrail (Python: `ToolOutputGuardrailData`).
#[derive(Debug, Clone)]
pub struct ToolOutputGuardrailData {
    /// The tool call that produced the output.
    pub context: ToolContext,
    /// The agent executing the tool.
    pub agent: Arc<Agent>,
    /// What the tool returned.
    pub output: Value,
}

type GuardrailFuture = Pin<Box<dyn Future<Output = ToolGuardrailFunctionOutput> + Send>>;

/// A check that runs before a function tool is invoked (Python: `ToolInputGuardrail`).
#[derive(Clone)]
pub struct ToolInputGuardrail {
    name: String,
    guardrail_function: Arc<dyn Fn(ToolInputGuardrailData) -> GuardrailFuture + Send + Sync>,
}

impl std::fmt::Debug for ToolInputGuardrail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolInputGuardrail")
            .field("name", &self.name)
            .finish()
    }
}

impl ToolInputGuardrail {
    /// Build a guardrail from an async closure.
    pub fn new<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(ToolInputGuardrailData) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ToolGuardrailFunctionOutput> + Send + 'static,
    {
        Self {
            name: name.into(),
            guardrail_function: Arc::new(move |data| Box::pin(f(data))),
        }
    }

    /// The guardrail's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Run the check.
    pub async fn run(&self, data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput {
        (self.guardrail_function)(data).await
    }
}

/// A check that runs after a function tool returned (Python: `ToolOutputGuardrail`).
#[derive(Clone)]
pub struct ToolOutputGuardrail {
    name: String,
    guardrail_function: Arc<dyn Fn(ToolOutputGuardrailData) -> GuardrailFuture + Send + Sync>,
}

impl std::fmt::Debug for ToolOutputGuardrail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolOutputGuardrail")
            .field("name", &self.name)
            .finish()
    }
}

impl ToolOutputGuardrail {
    /// Build a guardrail from an async closure.
    pub fn new<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(ToolOutputGuardrailData) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ToolGuardrailFunctionOutput> + Send + 'static,
    {
        Self {
            name: name.into(),
            guardrail_function: Arc::new(move |data| Box::pin(f(data))),
        }
    }

    /// The guardrail's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Run the check.
    pub async fn run(&self, data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput {
        (self.guardrail_function)(data).await
    }
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

/// Run `guardrails` before a tool call. Returns the message that replaces the call, if one
/// rejected it; the first rejection ends the check, like Python.
pub(crate) async fn run_tool_input_guardrails(
    guardrails: &[ToolInputGuardrail],
    context: &ToolContext,
    agent: &Arc<Agent>,
    results: &mut Vec<ToolInputGuardrailResult>,
) -> Result<Option<String>, crate::error::ToolInputGuardrailTripwireTriggered> {
    for guardrail in guardrails {
        let output = guardrail
            .run(ToolInputGuardrailData {
                context: context.clone(),
                agent: Arc::clone(agent),
            })
            .await;
        let result = ToolInputGuardrailResult {
            guardrail_name: guardrail.name.clone(),
            output,
        };
        results.push(result.clone());
        match &result.output.behavior {
            ToolGuardrailBehavior::RaiseException => {
                return Err(crate::error::ToolInputGuardrailTripwireTriggered {
                    guardrail_name: result.guardrail_name,
                    output: result.output,
                })
            }
            ToolGuardrailBehavior::RejectContent { message } => return Ok(Some(message.clone())),
            ToolGuardrailBehavior::Allow => {}
        }
    }
    Ok(None)
}

/// Run `guardrails` on a tool result. Returns the (possibly replaced) output.
pub(crate) async fn run_tool_output_guardrails(
    guardrails: &[ToolOutputGuardrail],
    context: &ToolContext,
    agent: &Arc<Agent>,
    output: Value,
    results: &mut Vec<ToolOutputGuardrailResult>,
) -> Result<Value, crate::error::ToolOutputGuardrailTripwireTriggered> {
    for guardrail in guardrails {
        let function_output = guardrail
            .run(ToolOutputGuardrailData {
                context: context.clone(),
                agent: Arc::clone(agent),
                output: output.clone(),
            })
            .await;
        let result = ToolOutputGuardrailResult {
            guardrail_name: guardrail.name.clone(),
            output: function_output,
        };
        results.push(result.clone());
        match &result.output.behavior {
            ToolGuardrailBehavior::RaiseException => {
                return Err(crate::error::ToolOutputGuardrailTripwireTriggered {
                    guardrail_name: result.guardrail_name,
                    output: result.output,
                })
            }
            ToolGuardrailBehavior::RejectContent { message } => {
                return Ok(Value::String(message.clone()))
            }
            ToolGuardrailBehavior::Allow => {}
        }
    }
    Ok(output)
}
