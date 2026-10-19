//! Input / output guardrails (Python: `agents.guardrail`).
//!
//! Guardrails run alongside the agent. Each returns a [`Verdict`]: pass, fail (halt the run;
//! Python's `tripwire_triggered`) or transform (continue with rewritten content, Rust-only).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::agent::Agent;
use crate::error::payload as guardrail_payload;
use crate::items::InputLike;
use crate::run_context::RunContextWrapper;

pub use guardrail_payload::{
    GuardrailFunctionOutput, InputGuardrailResult, OutputGuardrailResult, Severity, Verdict,
};

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

/// Run a group of guardrails side by side and return their results in declaration order.
///
/// A [`Verdict::Fail`] whose [`Severity`] stops early ends the group at once: the guardrails
/// still running are dropped, which cancels them (Python cancels the rest on any tripwire, and
/// [`GuardrailFunctionOutput::trip`] is `High`). Their results are then missing, as in Python.
pub(crate) async fn run_group<R, F>(
    guardrails: impl IntoIterator<Item = F>,
    output_of: fn(&R) -> &GuardrailFunctionOutput,
) -> Vec<R>
where
    F: Future<Output = R>,
{
    use futures::stream::{FuturesUnordered, StreamExt};

    let mut running: FuturesUnordered<_> = guardrails
        .into_iter()
        .enumerate()
        .map(|(index, fut)| async move { (index, fut.await) })
        .collect();
    let mut done = Vec::with_capacity(running.len());
    while let Some((index, result)) = running.next().await {
        let stop = output_of(&result).stops_group();
        done.push((index, result));
        if stop {
            break;
        }
    }
    done.sort_by_key(|(index, _)| *index);
    done.into_iter().map(|(_, result)| result).collect()
}

/// The single transform among `results`, if any; two or more are ambiguous and fail the run.
pub(crate) fn single_transform<R>(
    results: &[R],
    output_of: fn(&R) -> &GuardrailFunctionOutput,
    name_of: fn(&R) -> &str,
) -> Result<Option<(&str, &Value)>, crate::error::UserError> {
    let mut found = results.iter().filter_map(|r| {
        output_of(r)
            .transformed()
            .map(|content| (name_of(r), content))
    });
    let first = found.next();
    if let Some((second, _)) = found.next() {
        let first_name = first.map_or("", |(name, _)| name);
        return Err(crate::error::UserError::new(format!(
            "guardrails `{first_name}` and `{second}` both returned Verdict::Transform for the \
             same content; combine them into one guardrail so the order is explicit"
        )));
    }
    Ok(first)
}
