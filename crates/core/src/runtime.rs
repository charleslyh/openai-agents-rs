//! Explicit run-time environment: the default model provider and the Tokio runtime behind
//! blocking calls.
//!
//! These were process-wide `static`s (three `OnceLock`s in `run::resolve`, plus the one behind
//! `Runner::run_blocking`, plus a `std::sync::Once` in the facade crate that guarded installing
//! the default OpenAI provider). They now live on one `Runtime` value instead, so an application
//! can hold it explicitly and a test can build a private one instead of mutating process state.
//! [`Runtime::global()`] keeps the old "process default" behavior for callers that want it; the
//! facade `Runner` installs the default OpenAI provider there exactly once (`OnceLock::set` is
//! already idempotent, so the separate `Once` the facade used to need is gone).
//!
//! Tracing's global switch and processor, and the handoff history wrapper markers, are still
//! `static`s in their own modules (`tracing`, `handoffs::history`): dozens of call sites read
//! them with no `Runtime` in scope (every `agent_span()` / `turn_span()` call in the run loop).
//! Moving them onto `Runtime` needs threading it through those call sites, which this step does
//! not do to keep the change mechanical and low-risk; [`Runtime::tracing_disabled`] /
//! [`Runtime::set_tracing_disabled`] forward to that still-global state so `Runtime` is at least
//! the one place to read and change it, and the follow-up is tracked in
//! `docs/refactor/08-calibration-inventory.md`.
//!
//! A `Runtime` is cheap to clone: everything is behind an `Arc`.

use std::sync::{Arc, OnceLock};

use crate::error::AgentsError;
use crate::model::{MissingProvider, ModelProvider};

type ProviderFactory = Box<dyn Fn() -> Arc<dyn ModelProvider> + Send + Sync>;

struct ProviderSlot {
    factory: OnceLock<ProviderFactory>,
    built: OnceLock<Arc<dyn ModelProvider>>,
    /// Used when nothing is configured, so the error names the fix instead of the symptom.
    missing: Arc<dyn ModelProvider>,
}

impl ProviderSlot {
    fn new() -> Self {
        Self {
            factory: OnceLock::new(),
            built: OnceLock::new(),
            missing: Arc::new(MissingProvider),
        }
    }

    fn provider(&self) -> Arc<dyn ModelProvider> {
        if let Some(provider) = self.built.get() {
            return Arc::clone(provider);
        }
        match self.factory.get() {
            Some(factory) => Arc::clone(self.built.get_or_init(|| factory())),
            None => Arc::clone(&self.missing),
        }
    }
}

/// Builder for a [`Runtime`].
///
/// ```
/// use openai_agents_core::Runtime;
///
/// let runtime = Runtime::builder().build();
/// assert!(!runtime.tracing_disabled());
/// ```
#[derive(Default)]
pub struct RuntimeBuilder {
    provider: Option<Arc<dyn ModelProvider>>,
    provider_factory: Option<Arc<dyn Fn() -> Arc<dyn ModelProvider> + Send + Sync>>,
}

impl RuntimeBuilder {
    /// Use `provider` for every model name that has no explicit `RunConfig.model.provider`.
    #[must_use]
    pub fn provider(mut self, provider: Arc<dyn ModelProvider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Build the default provider lazily, the first time a name has to be resolved without one.
    #[must_use]
    pub fn provider_factory(
        mut self,
        factory: impl Fn() -> Arc<dyn ModelProvider> + Send + Sync + 'static,
    ) -> Self {
        self.provider_factory = Some(Arc::new(factory));
        self
    }

    /// Finish the runtime.
    pub fn build(self) -> Runtime {
        let slot = ProviderSlot::new();
        if let Some(factory) = self.provider_factory {
            let _ = slot.factory.set(Box::new(move || factory()));
        }
        if let Some(provider) = self.provider {
            let _ = slot.built.set(provider);
        }
        Runtime {
            inner: Arc::new(RuntimeInner {
                provider: slot,
                blocking: OnceLock::new(),
            }),
        }
    }
}

struct RuntimeInner {
    provider: ProviderSlot,
    /// The Tokio runtime behind `Runner::run_blocking` when the caller supplies none.
    blocking: OnceLock<Result<tokio::runtime::Runtime, String>>,
}

/// The environment a run executes in (see the module docs).
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field(
                "has_explicit_provider",
                &self.inner.provider.built.get().is_some(),
            )
            .field("tracing_disabled", &self.tracing_disabled())
            .finish()
    }
}

impl Runtime {
    /// A runtime with no model provider configured yet.
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// The process-wide runtime behind the `set_default_model_provider_factory` free function
    /// and the facade `Runner`'s default OpenAI provider installation.
    ///
    /// Prefer building and passing an explicit [`Runtime`] in new code; this exists so
    /// application-level "configure the SDK once" code, and single-crate users who never think
    /// about a `Runtime` at all, keep working.
    pub fn global() -> &'static Runtime {
        static GLOBAL: OnceLock<Runtime> = OnceLock::new();
        GLOBAL.get_or_init(Runtime::new)
    }

    /// Start building a runtime.
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    /// The provider used for names without an explicit `RunConfig.model.provider`.
    pub fn provider(&self) -> Arc<dyn ModelProvider> {
        self.inner.provider.provider()
    }

    /// Register how to build the default provider once, lazily. Returns `false` (keeping the
    /// earlier registration) when one is already set.
    pub fn set_model_provider_factory(
        &self,
        factory: impl Fn() -> Arc<dyn ModelProvider> + Send + Sync + 'static,
    ) -> bool {
        self.inner.provider.factory.set(Box::new(factory)).is_ok()
    }

    /// Install a ready default provider. Returns `false` when one is already set.
    pub fn set_model_provider(&self, provider: Arc<dyn ModelProvider>) -> bool {
        self.inner.provider.built.set(provider).is_ok()
    }

    /// Whether tracing is off (forwards to the process-wide switch; see the module docs).
    pub fn tracing_disabled(&self) -> bool {
        crate::tracing::is_tracing_disabled()
    }

    /// Turn tracing off or on (forwards to the process-wide switch; see the module docs).
    pub fn set_tracing_disabled(&self, disabled: bool) {
        crate::tracing::set_tracing_disabled(disabled);
    }

    /// The Tokio runtime handle used by blocking calls made without one of their own.
    pub(crate) fn blocking_handle(&self) -> Result<&tokio::runtime::Runtime, AgentsError> {
        self.inner
            .blocking
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
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Marker;
    impl ModelProvider for Marker {
        fn get_model(
            &self,
            _name: Option<&str>,
        ) -> Result<Arc<dyn crate::model::Model>, crate::error::UserError> {
            unimplemented!()
        }
    }

    #[test]
    fn two_runtimes_do_not_share_their_provider() {
        let a = Runtime::new();
        let b = Runtime::new();
        assert!(a.set_model_provider(Arc::new(Marker)));
        // `b` still has no provider of its own: it reports the actionable missing-provider error.
        let err = b.provider().get_model(None).err().expect("no provider set");
        assert!(err.to_string().contains("set_model_provider"), "{err}");
    }

    #[test]
    fn a_provider_can_only_be_set_once() {
        let runtime = Runtime::new();
        assert!(runtime.set_model_provider(Arc::new(Marker)));
        assert!(!runtime.set_model_provider(Arc::new(Marker)));
    }

    #[test]
    fn blocking_handle_is_reused_across_calls() {
        let runtime = Runtime::new();
        let a = runtime.blocking_handle().expect("runtime").handle().id();
        let b = runtime.blocking_handle().expect("runtime").handle().id();
        assert_eq!(a, b);
    }
}
