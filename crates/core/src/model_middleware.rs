//! Model middleware: rewrite a turn's model input before the request leaves the process
//! (Rust-only; Python's equivalent is the single `call_model_input_filter`, kept here as
//! [`ModelOptions::input_filter`](crate::run::ModelOptions::input_filter)).
//!
//! A [`ModelMiddleware`] composes several named steps instead of one closure, so unrelated
//! concerns (redacting a field, injecting a system note, enforcing a token budget) stay in
//! separate types and can be added or removed independently. Every middleware runs, in the
//! order it was added, *after* `input_filter` — Python parity keeps that single hook; this is
//! the additional Rust-idiomatic composition on top of it (see `docs/DEVIATIONS.md`).
//!
//! Only request rewriting is covered so far. Wrapping the response (metrics, caching) would need
//! to reach into the retry/streaming call path and is tracked as a follow-up, not part of this
//! trait yet.
//!
//! ```
//! use async_trait::async_trait;
//! use opai_agents_core::agent::Agent;
//! use opai_agents_core::error::AgentsError;
//! use opai_agents_core::model::ModelInputData;
//! use opai_agents_core::model_middleware::ModelMiddleware;
//! use opai_agents_core::run::ModelOptions;
//! use opai_agents_core::run_context::RunContextWrapper;
//!
//! struct AppendNote;
//!
//! #[async_trait]
//! impl ModelMiddleware for AppendNote {
//!     async fn prepare(
//!         &self,
//!         mut data: ModelInputData,
//!         _agent: &Agent,
//!         _context: &RunContextWrapper,
//!     ) -> Result<ModelInputData, AgentsError> {
//!         data.instructions = data.instructions.map(|i| format!("{i}\n(middleware note)"));
//!         Ok(data)
//!     }
//! }
//!
//! let options = ModelOptions::new().middleware(AppendNote);
//! ```

use async_trait::async_trait;

use crate::agent::Agent;
use crate::error::AgentsError;
use crate::model::ModelInputData;
use crate::run_context::RunContextWrapper;

/// Rewrites a turn's model input before it is sent.
#[async_trait]
pub trait ModelMiddleware: Send + Sync {
    /// Name used in error messages when this middleware fails.
    fn name(&self) -> &str {
        "model_middleware"
    }

    /// Rewrite `data`. Called after
    /// [`ModelOptions::input_filter`](crate::run::ModelOptions::input_filter), in the order the
    /// middleware was added to [`ModelOptions`](crate::run::ModelOptions).
    async fn prepare(
        &self,
        data: ModelInputData,
        agent: &Agent,
        context: &RunContextWrapper,
    ) -> Result<ModelInputData, AgentsError>;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::run_context::RunContextWrapper;

    struct Prefix(&'static str);

    #[async_trait]
    impl ModelMiddleware for Prefix {
        async fn prepare(
            &self,
            mut data: ModelInputData,
            _agent: &Agent,
            _context: &RunContextWrapper,
        ) -> Result<ModelInputData, AgentsError> {
            data.instructions = Some(format!(
                "{}{}",
                self.0,
                data.instructions.unwrap_or_default()
            ));
            Ok(data)
        }
    }

    // `ModelOptions`/`RunConfig` (crate::run, tier 3) carrying this field is covered by
    // `run::options::tests`; this module (tier 2) only tests the trait itself.
    #[tokio::test]
    async fn middleware_runs_in_the_order_it_was_added() {
        let chain: Vec<Arc<dyn ModelMiddleware>> =
            vec![Arc::new(Prefix("a-")), Arc::new(Prefix("b-"))];
        let agent = Agent::new("probe");
        let context = RunContextWrapper::default();
        let mut data = ModelInputData {
            input: Vec::new(),
            instructions: Some("base".into()),
        };
        for mw in &chain {
            data = mw.prepare(data, &agent, &context).await.unwrap();
        }
        assert_eq!(data.instructions.as_deref(), Some("b-a-base"));
    }
}
