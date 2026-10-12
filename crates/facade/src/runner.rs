//! The facade `Runner`: the core runner, plus the default model provider.
//!
//! Core knows no concrete model provider, so a name such as `gpt-4o` cannot resolve until one is
//! installed. This crate has the OpenAI providers, so [`Runner`] installs OpenAI as the process
//! default the first time a run starts, which keeps the "set `OPENAI_API_KEY` and go" behavior of
//! a single-crate build. Everything else is forwarded to the core runner unchanged.
//!
//! Every public method of `openai_agents_core::Runner` needs a twin here;
//! `scripts/check_runner_api.py` (run in CI) fails when the two lists differ.

use openai_agents_core::{
    Agent, AgentsError, InputLike, RunOptions, RunResult, RunResultStreaming, RunState,
};

/// Entry point for running agents (Python: `Runner`).
///
/// The same methods as `openai_agents_core::Runner`, which documents them. The first call in a
/// process also installs the default OpenAI model provider (see the module docs).
pub struct Runner;

impl Runner {
    /// See `openai_agents_core::Runner::run`.
    pub async fn run(
        starting_agent: &Agent,
        input: impl Into<InputLike>,
        options: RunOptions,
    ) -> Result<RunResult, AgentsError> {
        install_default_provider();
        openai_agents_core::Runner::run(starting_agent, input, options).await
    }

    /// See `openai_agents_core::Runner::run_state`.
    pub async fn run_state(
        starting_agent: &Agent,
        state: RunState,
        options: RunOptions,
    ) -> Result<RunResult, AgentsError> {
        install_default_provider();
        openai_agents_core::Runner::run_state(starting_agent, state, options).await
    }

    /// See `openai_agents_core::Runner::run_streamed`.
    pub fn run_streamed(
        starting_agent: Agent,
        input: impl Into<InputLike>,
        options: RunOptions,
    ) -> RunResultStreaming {
        install_default_provider();
        openai_agents_core::Runner::run_streamed(starting_agent, input, options)
    }

    /// See `openai_agents_core::Runner::run_state_streamed`.
    pub fn run_state_streamed(
        starting_agent: Agent,
        state: RunState,
        options: RunOptions,
    ) -> Result<RunResultStreaming, AgentsError> {
        install_default_provider();
        openai_agents_core::Runner::run_state_streamed(starting_agent, state, options)
    }

    /// See `openai_agents_core::Runner::run_blocking`.
    pub fn run_blocking(
        starting_agent: &Agent,
        input: impl Into<InputLike>,
        options: RunOptions,
    ) -> Result<RunResult, AgentsError> {
        install_default_provider();
        openai_agents_core::Runner::run_blocking(starting_agent, input, options)
    }

    /// See `openai_agents_core::Runner::run_blocking_on`.
    pub fn run_blocking_on(
        runtime: &tokio::runtime::Handle,
        starting_agent: &Agent,
        input: impl Into<InputLike>,
        options: RunOptions,
    ) -> Result<RunResult, AgentsError> {
        install_default_provider();
        openai_agents_core::Runner::run_blocking_on(runtime, starting_agent, input, options)
    }
}

/// Register OpenAI as the default model provider, once per process.
#[cfg(feature = "openai")]
fn install_default_provider() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        openai_agents_core::set_default_model_provider_factory(default_model_provider);
    });
}

/// Without the `openai` feature there is nothing to install: core falls back to a provider that
/// fails with an actionable message.
#[cfg(not(feature = "openai"))]
fn install_default_provider() {}

/// The provider that resolves OpenAI model names from the environment
/// (Python: the default provider of `agents`).
///
/// Built from `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`, it routes the `openai`,
/// `openai_chat_completions` and `openai_responses` name prefixes. When neither the key nor a
/// base URL is set it fails with an actionable error.
#[cfg(feature = "openai")]
pub fn default_model_provider() -> std::sync::Arc<dyn openai_agents_core::ModelProvider> {
    use std::sync::Arc;

    use openai_agents_core::{MissingProvider, MultiProvider};
    use openai_agents_openai::{DefaultOpenAiApi, OpenAIProvider};

    match OpenAIProvider::from_env() {
        Ok(provider) => {
            let router = MultiProvider::new()
                .register("openai", Arc::new(provider.clone()))
                .register(
                    "openai_chat_completions",
                    Arc::new(provider.clone().always_chat_completions()),
                )
                .register(
                    "openai_responses",
                    Arc::new(provider.api(DefaultOpenAiApi::Responses)),
                );
            Arc::new(router)
        }
        Err(_) => Arc::new(MissingProvider),
    }
}
