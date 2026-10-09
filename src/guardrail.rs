//! Input / output guardrails (Python: `agents.guardrail`).
//!
//! Guardrails run alongside the agent and can halt it by setting `tripwire_triggered`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::agent::Agent;
use crate::items::InputLike;
use crate::run_context::RunContextWrapper;

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

/// Guardrail body: `(context, agent, input) -> GuardrailFunctionOutput`.
pub type InputGuardrailFn = Arc<
    dyn Fn(
            RunContextWrapper,
            Arc<Agent>,
            InputLike,
        ) -> Pin<Box<dyn Future<Output = GuardrailFunctionOutput> + Send>>
        + Send
        + Sync,
>;

/// Guardrail body: `(context, agent, agent_output) -> GuardrailFunctionOutput`.
pub type OutputGuardrailFn = Arc<
    dyn Fn(
            RunContextWrapper,
            Arc<Agent>,
            Value,
        ) -> Pin<Box<dyn Future<Output = GuardrailFunctionOutput> + Send>>
        + Send
        + Sync,
>;

/// A check that runs against the run input (Python: `InputGuardrail`).
#[derive(Clone)]
pub struct InputGuardrail {
    guardrail_function: InputGuardrailFn,
    name: String,
    /// Whether the guardrail runs concurrently with the agent (Python default: true).
    pub run_in_parallel: bool,
}

impl std::fmt::Debug for InputGuardrail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InputGuardrail")
            .field("name", &self.name)
            .field("run_in_parallel", &self.run_in_parallel)
            .finish()
    }
}

impl InputGuardrail {
    /// Build a guardrail from a closure.
    pub fn new<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(RunContextWrapper, Arc<Agent>, InputLike) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = GuardrailFunctionOutput> + Send + 'static,
    {
        let f = Arc::new(f);
        Self {
            guardrail_function: Arc::new(move |ctx, agent, input| {
                let f = Arc::clone(&f);
                Box::pin(async move { f(ctx, agent, input).await })
            }),
            name: name.into(),
            run_in_parallel: true,
        }
    }

    /// Whether to run concurrently with the agent rather than before it starts.
    pub fn run_in_parallel(mut self, parallel: bool) -> Self {
        self.run_in_parallel = parallel;
        self
    }

    /// Guardrail name used for tracing and error messages.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Run the guardrail.
    ///
    /// Python wraps the guardrail body in `guardrail_span(name)`, so the check is visible in the
    /// trace and inherits the run's tracing switch (`RunConfig.tracing_disabled`).
    pub async fn run(
        &self,
        agent: Arc<Agent>,
        input: InputLike,
        context: RunContextWrapper,
    ) -> InputGuardrailResult {
        let _span = crate::tracing::guardrail_span(&self.name);
        let output = (self.guardrail_function)(context, agent, input).await;
        InputGuardrailResult {
            guardrail_name: self.name.clone(),
            output,
        }
    }
}

/// Convenience constructor mirroring Python's `@input_guardrail`.
pub fn input_guardrail<F, Fut>(name: impl Into<String>, f: F) -> InputGuardrail
where
    F: Fn(RunContextWrapper, Arc<Agent>, InputLike) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = GuardrailFunctionOutput> + Send + 'static,
{
    InputGuardrail::new(name, f)
}

/// A check that runs against the final output (Python: `OutputGuardrail`).
#[derive(Clone)]
pub struct OutputGuardrail {
    guardrail_function: OutputGuardrailFn,
    name: String,
}

impl std::fmt::Debug for OutputGuardrail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputGuardrail")
            .field("name", &self.name)
            .finish()
    }
}

impl OutputGuardrail {
    /// Build a guardrail from a closure.
    pub fn new<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(RunContextWrapper, Arc<Agent>, Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = GuardrailFunctionOutput> + Send + 'static,
    {
        let f = Arc::new(f);
        Self {
            guardrail_function: Arc::new(move |ctx, agent, output| {
                let f = Arc::clone(&f);
                Box::pin(async move { f(ctx, agent, output).await })
            }),
            name: name.into(),
        }
    }

    /// Guardrail name used for tracing and error messages.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Run the guardrail.
    ///
    /// Python wraps the guardrail body in `guardrail_span(name)`, so the check is visible in the
    /// trace and inherits the run's tracing switch (`RunConfig.tracing_disabled`).
    pub async fn run(
        &self,
        agent: Arc<Agent>,
        agent_output: Value,
        context: RunContextWrapper,
    ) -> OutputGuardrailResult {
        let _span = crate::tracing::guardrail_span(&self.name);
        let output =
            (self.guardrail_function)(context, Arc::clone(&agent), agent_output.clone()).await;
        OutputGuardrailResult {
            guardrail_name: self.name.clone(),
            agent_name: agent.name.clone(),
            agent_output,
            output,
        }
    }
}

/// Convenience constructor mirroring Python's `@output_guardrail`.
pub fn output_guardrail<F, Fut>(name: impl Into<String>, f: F) -> OutputGuardrail
where
    F: Fn(RunContextWrapper, Arc<Agent>, Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = GuardrailFunctionOutput> + Send + 'static,
{
    OutputGuardrail::new(name, f)
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
