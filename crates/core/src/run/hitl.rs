//! Resolve paused tool calls in-process with a callback (Rust-only, X-009).
//!
//! Without a resolver a run that reaches a tool call needing resolution returns early with
//! [`RunResult::interruptions`]; the host turns it into a
//! [`RunState`](crate::run_state::RunState), resolves each item and
//! resumes with [`Runner::run_state`](super::Runner::run_state). [`RunOptions::resolve_with`]
//! runs that loop for the host: the callback is asked about each paused call, and the run
//! resumes until it finishes or fails. The state machine underneath is the same one, so a run
//! that needs to outlive the process should keep using the explicit `RunState` flow.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::agent::Agent;
use crate::error::AgentsError;
use crate::items::ToolInterruptionItem;
use crate::result::RunResult;
use crate::run_state::ToolResolution;

use super::agent_loop::{run_loop, LoopStart};
use super::config::RunOptions;

type ResolveFuture = Pin<Box<dyn Future<Output = ToolResolution> + Send>>;
type ResolveFn = dyn Fn(ToolInterruptionItem) -> ResolveFuture + Send + Sync;

/// Decides how a paused tool call is resolved; set with [`RunOptions::resolve_with`].
///
/// The callback receives the paused call (agent, tool, call id and arguments) and returns a
/// [`ToolResolution`]: [`Run`](ToolResolution::Run) the tool, [`reject`](ToolResolution::reject)
/// it with a message for the model, or hand back its [`output`](ToolResolution::output).
#[derive(Clone)]
pub struct ToolResolver(Arc<ResolveFn>);

impl ToolResolver {
    /// Wrap an async callback.
    pub fn new<F, Fut>(resolve: F) -> Self
    where
        F: Fn(ToolInterruptionItem) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ToolResolution> + Send + 'static,
    {
        Self(Arc::new(move |item| Box::pin(resolve(item))))
    }

    async fn resolve(&self, item: ToolInterruptionItem) -> ToolResolution {
        (self.0)(item).await
    }
}

impl std::fmt::Debug for ToolResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ToolResolver(..)")
    }
}

/// A paused result turned into a cancelled one: the calls that were waiting for an answer will
/// not get one, so the result no longer lists them or offers a state to resume from.
fn cancelled_while_paused(mut result: RunResult) -> RunResult {
    result.interruptions.clear();
    result.interrupt_state = None;
    result.is_cancelled = true;
    result
}

/// Run from `start`, then, while the result is paused and `options` has a resolver, resolve every
/// pending call through it and resume. Without a resolver this is a single `run_loop`.
pub(super) async fn run_resolving(
    agent: Arc<Agent>,
    start: LoopStart,
    options: RunOptions,
) -> Result<RunResult, AgentsError> {
    let Some(resolver) = options.resolver.clone() else {
        return run_loop(agent, start, options, None, None).await;
    };
    let mut result = run_loop(Arc::clone(&agent), start, options.clone(), None, None).await?;
    let token = options.cancellation.clone().unwrap_or_default();
    while result.is_interrupted() {
        let mut state = result.to_state()?;
        for item in &result.interruptions {
            tracing::debug!(
                agent = %item.agent_name,
                tool = %item.tool_name,
                call_id = %item.call_id,
                "resolving paused tool call through RunOptions::resolve_with"
            );
            // The callback may wait on a person. Cancelling stops the wait and drops the
            // callback's future, so whatever it holds (a pending prompt, a lock) is released by
            // its own `Drop` instead of staying behind for as long as nobody answers.
            let resolution = tokio::select! {
                biased;
                () = token.cancelled() => {
                    return Ok(cancelled_while_paused(result));
                }
                resolution = resolver.resolve(item.clone()) => resolution,
            };
            state.resolve(item, resolution);
        }
        result = run_loop(
            Arc::clone(&agent),
            LoopStart::Resume {
                state: Box::new(state),
            },
            options.clone(),
            None,
            None,
        )
        .await?;
    }
    Ok(result)
}
