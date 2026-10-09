//! Runner and run configuration (Python: `agents.run` / `run_config`).
//!
//! The module is split by concern, and everything public is re-exported here so
//! `crate::run::Thing` keeps working:
//!
//! - [`config`] — `RunConfig`, `RunOptions` and the settings hanging off them
//! - [`errors`] — the `error_handlers` machinery that turns a run-ending error into an output
//! - `tools` — planning and executing the function tools of one turn
//! - `agent_loop` — the agent loop itself

mod agent_loop;
pub mod config;
pub mod errors;
pub(crate) mod tools;

pub use agent_loop::default_provider;
pub use config::{
    default_trace_include_sensitive_data, get_default_openai_api, set_default_openai_api,
    CallModelData, CallModelInputFilter, DefaultOpenAiApi, ModelInputData,
    OutputGuardrailBlockedMessage, OutputGuardrailBlockedMessageArgs, RunConfig, RunOptions,
    ToolErrorFormatter, ToolErrorFormatterArgs, ToolErrorKind, ToolExecutionConfig,
    ToolNameCollisionPolicy, ToolNotFoundBehavior, DEFAULT_MAX_TURNS,
    OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT,
};
pub use errors::{
    RunErrorData, RunErrorHandler, RunErrorHandlerInput, RunErrorHandlerResult, RunErrorHandlers,
    RunHandledError,
};

pub(crate) use config::explicit_default_openai_api;
pub(crate) use tools::build_agent_as_tool;

use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::error::{AgentsError, UserError};
use crate::items::InputLike;
use crate::result::{RunResult, RunResultStreaming, StreamingSnapshot};
use crate::run_state::RunState;

use agent_loop::{run_loop, LoopStart};

/// Facade entry point (Python: `Runner`).
pub struct Runner;

impl Runner {
    /// Run an agent asynchronously (Python: `Runner.run` with string/list input).
    pub async fn run(
        starting_agent: &Agent,
        input: impl Into<InputLike>,
        options: RunOptions,
    ) -> Result<RunResult, AgentsError> {
        run_loop(
            starting_agent.clone(),
            LoopStart::Fresh {
                input: input.into(),
                prepared: None,
            },
            options,
            None,
            None,
        )
        .await
    }

    /// Resume a paused run after approve/reject (Python: `Runner.run(agent, state)`).
    pub async fn run_state(
        starting_agent: &Agent,
        state: RunState,
        options: RunOptions,
    ) -> Result<RunResult, AgentsError> {
        if state.starting_agent_name != starting_agent.name {
            return Err(UserError::new(format!(
                "RunState starting agent `{}` does not match `{}`",
                state.starting_agent_name, starting_agent.name
            ))
            .into());
        }
        run_loop(
            starting_agent.clone(),
            LoopStart::Resume {
                state: Box::new(state),
            },
            options,
            None,
            None,
        )
        .await
    }

    /// Run in streaming mode (Python: `Runner.run_streamed`).
    pub fn run_streamed(
        starting_agent: Agent,
        input: impl Into<InputLike>,
        options: RunOptions,
    ) -> RunResultStreaming {
        let input = input.into();
        let max_turns = options.max_turns;
        let (tx, rx) = mpsc::channel(64);
        let snapshot = Arc::new(Mutex::new(StreamingSnapshot {
            current_agent_name: starting_agent.name.clone(),
            ..Default::default()
        }));
        let snap = Arc::clone(&snapshot);
        let task = tokio::spawn(async move {
            let result = run_loop(
                starting_agent,
                LoopStart::Fresh {
                    input,
                    prepared: None,
                },
                options,
                Some(tx.clone()),
                Some(Arc::clone(&snap)),
            )
            .await;
            match result {
                Ok(r) => {
                    let mut s = snap.lock().expect("snapshot");
                    s.is_complete = true;
                    // A graceful cancel returns a result without a real final output.
                    let stopped_early = s.cancel_after_turn;
                    if stopped_early {
                        s.is_cancelled = true;
                    }
                    s.final_output = if r.interruptions.is_empty() && !stopped_early {
                        Some(r.final_output.clone())
                    } else {
                        None
                    };
                    s.interruptions = r.interruptions.clone();
                    s.new_items = r.new_items;
                    s.raw_responses = r.raw_responses;
                    s.usage = r.usage;
                    s.current_agent_name = r.last_agent_name;
                }
                Err(e) => {
                    {
                        let mut s = snap.lock().expect("snapshot");
                        s.error = Some(e.to_string());
                        s.is_complete = true;
                    }
                    let _ = tx.send(Err(e)).await;
                }
            }
        });
        RunResultStreaming::new(snapshot, max_turns, rx).with_task(task)
    }

    /// Blocking wrapper (Python: `run_sync` → Rust `run_blocking`, see D-001).
    ///
    /// Works from any thread, with or without a Tokio runtime around it:
    /// - **No runtime** (a plain `fn main`, a worker thread): the run executes on a shared,
    ///   lazily started multi-thread runtime, so no runtime is built per call and HTTP
    ///   connection pools survive between calls.
    /// - **Inside a multi-thread runtime** (for example from a `spawn_blocking` closure or a
    ///   synchronous callback): the run executes on that runtime and the calling worker is
    ///   handed over with `block_in_place`, so it does not panic with "cannot start a runtime
    ///   from within a runtime".
    /// - **Inside a current-thread runtime** (`#[tokio::test]`, `Runtime::new_current_thread`):
    ///   the calling thread cannot be lent out, so the run executes on the shared runtime from a
    ///   helper thread while the caller waits. The caller's runtime is stalled meanwhile; await
    ///   [`Runner::run`] instead when you can.
    ///
    /// Use [`Runner::run_blocking_on`] to choose the runtime yourself.
    pub fn run_blocking(
        starting_agent: &Agent,
        input: impl Into<InputLike>,
        options: RunOptions,
    ) -> Result<RunResult, AgentsError> {
        let input = input.into();
        block_on_any(None, move || Self::run(starting_agent, input, options))?
    }

    /// [`Runner::run_blocking`] on a runtime you provide (it must be a multi-thread runtime,
    /// otherwise nothing would drive it while the caller blocks).
    pub fn run_blocking_on(
        runtime: &tokio::runtime::Handle,
        starting_agent: &Agent,
        input: impl Into<InputLike>,
        options: RunOptions,
    ) -> Result<RunResult, AgentsError> {
        let input = input.into();
        block_on_any(Some(runtime.clone()), move || {
            Self::run(starting_agent, input, options)
        })?
    }
}

/// The runtime behind `run_blocking` when the caller has none of its own.
fn shared_runtime() -> Result<&'static tokio::runtime::Runtime, AgentsError> {
    static RUNTIME: std::sync::OnceLock<Result<tokio::runtime::Runtime, String>> =
        std::sync::OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("openai-agents-blocking")
                .build()
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| AgentsError::internal(format!("could not start a Tokio runtime: {e}")))
}

/// Run a future to completion on the calling thread's behalf, whatever the thread is doing.
///
/// `make` builds the future on the thread that drives it, so the future itself need not be `Send`.
fn block_on_any<R, F, Fut>(
    target: Option<tokio::runtime::Handle>,
    make: F,
) -> Result<R, AgentsError>
where
    R: Send,
    F: FnOnce() -> Fut + Send,
    Fut: std::future::Future<Output = R>,
{
    use tokio::runtime::{Handle, RuntimeFlavor};
    let inside = Handle::try_current().ok();
    let target = match target {
        Some(handle) => {
            if handle.runtime_flavor() != RuntimeFlavor::MultiThread {
                return Err(UserError::new(
                    "run_blocking_on needs a handle to a multi-thread runtime: nothing would \
                     drive a current-thread runtime while this thread blocks",
                )
                .into());
            }
            handle
        }
        None => match &inside {
            Some(current) if current.runtime_flavor() == RuntimeFlavor::MultiThread => {
                current.clone()
            }
            _ => shared_runtime()?.handle().clone(),
        },
    };
    match inside {
        None => Ok(target.block_on(make())),
        Some(current) if current.runtime_flavor() == RuntimeFlavor::MultiThread => {
            Ok(tokio::task::block_in_place(|| target.block_on(make())))
        }
        Some(_) => {
            std::thread::scope(
                |scope| match scope.spawn(|| target.block_on(make())).join() {
                    Ok(value) => Ok(value),
                    Err(panic) => std::panic::resume_unwind(panic),
                },
            )
        }
    }
}
