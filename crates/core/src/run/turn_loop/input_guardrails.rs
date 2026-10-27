//! The input guardrails of a run: which ones block the first model call, which run beside it, and
//! how their results change the input.

use super::*;

/// The input guardrails of a run.
///
/// Python splits them by `run_in_parallel`: blocking ones finish before the first model call, the
/// rest run concurrently with it.
pub(super) struct InputGuardrails {
    blocking: Option<Vec<InputGuardrail>>,
    parallel: Option<Vec<InputGuardrail>>,
    pub(super) pending: Option<AbortOnDrop<Vec<InputGuardrailResult>>>,
    agent: Arc<Agent>,
    input: InputLike,
}

impl InputGuardrails {
    /// A resumed run already has the results of its guardrails, so it gets none.
    pub(super) fn new(
        options: &RunOptions,
        agent: &Arc<Agent>,
        input: &InputLike,
        resuming: bool,
    ) -> Self {
        let (parallel, blocking): (Vec<InputGuardrail>, Vec<InputGuardrail>) = options
            .run_config
            .input_guardrails
            .iter()
            .chain(agent.all_input_guardrails())
            .filter(|_| !resuming)
            .cloned()
            .partition(|g| g.run_in_parallel);
        Self {
            blocking: Some(blocking),
            parallel: Some(parallel),
            pending: None,
            agent: Arc::clone(agent),
            input: input.clone(),
        }
    }

    /// A future that runs `guardrails` side by side. It owns its inputs so it can be spawned.
    pub(super) fn runner(
        &self,
        guardrails: Vec<InputGuardrail>,
        context: &RunContextWrapper,
    ) -> impl Future<Output = Vec<InputGuardrailResult>> + Send + 'static {
        let agent = Arc::clone(&self.agent);
        let run_input = self.input.clone();
        let ctx = context.clone();
        async move {
            let futs = guardrails.into_iter().map(|g| {
                let agent = Arc::clone(&agent);
                let run_input = run_input.clone();
                let ctx = ctx.clone();
                async move { g.run(agent, run_input, ctx).await }
            });
            run_group(futs, |r: &InputGuardrailResult| &r.output).await
        }
    }
}

/// The first tripped guardrail of `results`, as the error that aborts the run.
pub(super) fn tripwire(results: &[InputGuardrailResult]) -> Result<(), AgentsError> {
    match results.iter().find(|r| r.output.tripwire_triggered()) {
        Some(r) => Err(InputGuardrailTripwireTriggered { result: r.clone() }.into()),
        None => Ok(()),
    }
}

/// A guardrail that runs alongside the model cannot change what the model already received.
pub(super) fn reject_parallel_transform(
    results: &[InputGuardrailResult],
) -> Result<(), AgentsError> {
    match results.iter().find(|r| r.output.transformed().is_some()) {
        Some(r) => Err(UserError::new(format!(
            "input guardrail `{}` returned Verdict::Transform but runs in parallel with the model, \
             which has already received the original input; build it with \
             `.run_in_parallel(false)`",
            r.guardrail_name
        ))
        .into()),
        None => Ok(()),
    }
}

impl TurnLoop<'_> {
    /// Both groups start inside the first turn (Python runs them under the turn span): blocking
    /// ones finish first, then the parallel ones are spawned alongside the model call.
    pub(super) async fn start_input_guardrails(&mut self) -> Result<(), AgentsError> {
        if let Some(blocking) = self.guardrails.blocking.take().filter(|g| !g.is_empty()) {
            let results = self.guardrails.runner(blocking, &self.context).await;
            tripwire(&results)?;
            let transform = single_transform(
                &results,
                |r: &InputGuardrailResult| &r.output,
                |r: &InputGuardrailResult| r.guardrail_name.as_str(),
            )?
            .map(|(name, content)| (name.to_owned(), content.clone()));
            if let Some((name, content)) = transform {
                self.apply_input_transform(&name, content)?;
            }
            self.input_guardrail_results.extend(results);
        }
        if let Some(parallel) = self.guardrails.parallel.take().filter(|g| !g.is_empty()) {
            // `tokio::spawn` does not inherit task-locals, so the run's tracing switch is
            // re-established inside the task; otherwise spans opened by a guardrail escape
            // `RunConfig.tracing_disabled`.
            let disabled = self.options.run_config.tracing.disabled;
            let fut = self.guardrails.runner(parallel, &self.context);
            self.guardrails.pending = Some(AbortOnDrop(tokio::spawn(
                crate::tracing::with_run_tracing_disabled(disabled, fut),
            )));
        }
        Ok(())
    }

    /// Input guardrails ran alongside the first turn; a tripwire aborts the run.
    pub(super) async fn settle_input_guardrails(&mut self) -> Result<(), AgentsError> {
        if let Some(task) = self.guardrails.pending.take() {
            let results = task
                .await
                .map_err(|e| AgentsError::internal(format!("input guardrail task failed: {e}")))?;
            tripwire(&results)?;
            reject_parallel_transform(&results)?;
            self.input_guardrail_results.extend(results);
        }
        Ok(())
    }

    /// Replace the run's new input with what a blocking input guardrail returned, everywhere the
    /// original would otherwise go: the model input, the parallel guardrails, the session and the
    /// result. History loaded from a session is left alone.
    pub(super) fn apply_input_transform(
        &mut self,
        guardrail: &str,
        content: Value,
    ) -> Result<(), AgentsError> {
        let new_input = match content {
            Value::String(text) => InputLike::Text(text),
            Value::Array(items) => InputLike::Items(items),
            other => {
                return Err(UserError::new(format!(
                    "input guardrail `{guardrail}` returned a transform that is neither a string \
                     nor an array of input items: {other}"
                ))
                .into())
            }
        };
        let old_items = ItemHelpers::input_to_new_input_list(&self.input);
        let new_items = ItemHelpers::input_to_new_input_list(&new_input);
        let start = self.current_input_items.len().checked_sub(old_items.len());
        let in_place = start.filter(|&i| self.current_input_items[i..] == old_items[..]);
        let Some(start) = in_place else {
            return Err(UserError::new(format!(
                "input guardrail `{guardrail}` returned Verdict::Transform, but the run's input was \
                 already changed by the session input callback, so it cannot be replaced safely"
            ))
            .into());
        };
        tracing::debug!(guardrail, "input replaced by an input guardrail transform");
        self.current_input_items.truncate(start);
        self.current_input_items.extend(new_items.iter().cloned());
        self.original_input_len = self.original_input_len + new_items.len() - old_items.len();
        // A `pending_input` of `None` means the input was already saved; leave it alone.
        if let Some(writer) = self.session_writer.as_deref_mut() {
            if writer.pending_input.is_some() {
                writer.pending_input = Some(new_items);
            }
        }
        self.guardrails.input = new_input.clone();
        self.input = new_input;
        Ok(())
    }

    /// Whether this turn leaves the run's input unsaved because blocking input guardrails have
    /// not run yet: they may rewrite it with a `Verdict::Transform`, and what the session already
    /// holds cannot be rewritten. Later turns have no blocking guardrails left.
    pub(super) fn input_waits_for_guardrails(&self) -> bool {
        self.guardrails
            .blocking
            .as_ref()
            .is_some_and(|g| !g.is_empty())
            && self
                .session_writer
                .as_ref()
                .is_some_and(|w| w.pending_input.is_some())
    }
}
