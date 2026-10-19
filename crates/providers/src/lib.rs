//! Named provider presets and a capability matrix (Rust-only, X-013).
//!
//! Python names a provider by handing `ModelProvider` objects to `MultiProvider`; there is no
//! table of third-party servers. This crate adds one, so connecting a provider is "set its
//! environment variable" rather than "know its base URL, protocol and limits":
//!
//! ```no_run
//! use openai_agents_providers::ProviderRegistry;
//!
//! // Registers every preset whose key variable is set (DEEPSEEK_API_KEY, GEMINI_API_KEY, ...),
//! // plus AGENTS_PROVIDER_1_ENDPOINT and friends.
//! let provider = ProviderRegistry::from_env()?.into_provider()?;
//! # Ok::<(), openai_agents_core::UserError>(())
//! ```
//!
//! Three things live here:
//!
//! - [`PRESETS`]: static data — base URL, key variable, protocol, capabilities, model prefixes.
//! - [`Provider`]: a preset (or a URL) turned into a
//!   [`ModelProvider`](openai_agents_core::ModelProvider).
//! - [`ProviderRegistry`] / [`ProviderRouter`]: discovery from the environment, and `prefix/model`
//!   routing with inference for names that carry no prefix.
//!
//! The table goes stale the way any list of third-party servers does; [`LAST_CHECKED`] says when
//! it was last reviewed, and unknown model ids are always allowed through (private deployments
//! serve models no catalogue knows).

#![deny(missing_docs)]

mod guard;
mod provider;
mod registry;
mod spec;

pub use guard::GuardedModel;
pub use provider::{validate_endpoint, Provider};
pub use registry::{Entry, ProviderRegistry, ProviderRouter};
pub use spec::{
    infer_preset, preset, Capabilities, Capability, ProviderApi, ProviderSpec, PRESETS,
};

pub use spec::LAST_CHECKED;

use openai_agents_core::error::UserError;

/// A request needs something the provider does not support (Rust-only, X-013).
///
/// Raised instead of quietly dropping the field: a run that asked for reasoning on a server with
/// no reasoning field would otherwise answer as if it had thought about it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct UnsupportedCapability {
    /// The provider that cannot do it.
    pub provider: String,
    /// What it cannot do.
    pub capability: Capability,
    /// What to do instead, when there is a sensible answer.
    pub hint: Option<String>,
}

impl UnsupportedCapability {
    /// Build with no hint.
    pub fn new(provider: impl Into<String>, capability: Capability) -> Self {
        Self {
            provider: provider.into(),
            capability,
            hint: None,
        }
    }

    /// Attach advice: another provider, another protocol, another way to ask.
    #[must_use]
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
}

impl std::fmt::Display for UnsupportedCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "provider `{}` does not support {}",
            self.provider, self.capability
        )?;
        if let Some(hint) = &self.hint {
            write!(f, "; {hint}")?;
        }
        Ok(())
    }
}

impl std::error::Error for UnsupportedCapability {}

impl From<UnsupportedCapability> for UserError {
    fn from(error: UnsupportedCapability) -> Self {
        UserError::new(error.to_string())
    }
}

impl From<UnsupportedCapability> for openai_agents_core::AgentsError {
    fn from(error: UnsupportedCapability) -> Self {
        openai_agents_core::AgentsError::User(error.into())
    }
}
