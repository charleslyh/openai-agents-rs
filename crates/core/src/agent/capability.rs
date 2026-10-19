//! Named bundles of agent parts (Rust-only; `docs/DEVIATIONS.md` X-007).
//!
//! A [`Capability`] is a named set of things an agent can be given: tools, an instruction
//! paragraph, input and output guardrails, default model settings and the
//! [`Extensions`](crate::Extensions) types it needs. Mount one with [`Agent::capability`].
//!
//! Mounting is equivalent to handing the agent the same parts yourself, and the parts are kept
//! apart from the agent's own fields, so the order of builder calls does not matter:
//!
//! | Part | Merged with the agent's own | Precedence |
//! | --- | --- | --- |
//! | tools | after `Agent::tools`, before MCP tools | name clashes follow `ToolOptions::name_collision_policy` |
//! | instructions | appended after the agent's instructions, separated by a blank line | — |
//! | input / output guardrails | after the agent's own | all run |
//! | model settings | below the agent's settings, which are below the run's | agent and run win |
//! | requires | checked before the first turn, like `Tool::requires` | — |
//!
//! ```
//! use openai_agents_core::{Agent, CapabilityParts, FunctionTool};
//!
//! let clock = CapabilityParts::new()
//!     .tool(FunctionTool::constant("now", "Current time", "12:00"))
//!     .instructions("Use `now` when the user asks for the time.")
//!     .named("clock");
//! let agent = Agent::new("assistant").capability(clock);
//! assert_eq!(agent.capabilities()[0].name(), "clock");
//! ```

use std::sync::Arc;

use crate::extensions::TypeRequirement;
use crate::guardrail::{InputGuardrail, OutputGuardrail};
use crate::model_settings::ModelSettings;
use crate::tool::FunctionTool;

use super::Agent;

/// What a [`Capability`] contributes to an agent.
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct CapabilityParts {
    /// Tools added after the agent's own.
    pub tools: Vec<FunctionTool>,
    /// A paragraph appended to the agent's instructions.
    pub instructions: Option<String>,
    /// Input guardrails added after the agent's own.
    pub input_guardrails: Vec<InputGuardrail>,
    /// Output guardrails added after the agent's own.
    pub output_guardrails: Vec<OutputGuardrail>,
    /// Defaults below the agent's own model settings.
    pub model_settings: Option<ModelSettings>,
    /// Extension types the capability needs, checked before the first turn.
    pub requires: Vec<TypeRequirement>,
}

impl CapabilityParts {
    /// No parts.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tool.
    #[must_use]
    pub fn tool(mut self, tool: FunctionTool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add several tools.
    #[must_use]
    pub fn tools(mut self, tools: impl IntoIterator<Item = FunctionTool>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Set the instruction paragraph.
    #[must_use]
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Add an input guardrail.
    #[must_use]
    pub fn input_guardrail(mut self, guardrail: InputGuardrail) -> Self {
        self.input_guardrails.push(guardrail);
        self
    }

    /// Add an output guardrail.
    #[must_use]
    pub fn output_guardrail(mut self, guardrail: OutputGuardrail) -> Self {
        self.output_guardrails.push(guardrail);
        self
    }

    /// Set default model settings.
    #[must_use]
    pub fn model_settings(mut self, settings: ModelSettings) -> Self {
        self.model_settings = Some(settings);
        self
    }

    /// Declare that the capability needs a value of type `T` in the run's extensions.
    #[must_use]
    pub fn require<T: std::any::Any>(mut self) -> Self {
        self.requires.push(TypeRequirement::of::<T>());
        self
    }

    /// Give the parts a name, making them a [`Capability`].
    pub fn named(self, name: impl Into<String>) -> NamedCapability {
        NamedCapability {
            name: name.into(),
            parts: self,
        }
    }
}

/// A named bundle of agent parts.
///
/// Implement it for a reusable, configurable bundle (a database toolset that takes a connection
/// string, say); for a one-off, [`CapabilityParts::named`] is enough.
pub trait Capability: Send + Sync {
    /// Name used in error messages and `Debug` output.
    fn name(&self) -> &str;

    /// The parts. Called once, when the capability is mounted.
    fn parts(&self) -> CapabilityParts;
}

/// A [`Capability`] made from [`CapabilityParts`] and a name.
#[derive(Clone)]
pub struct NamedCapability {
    name: String,
    parts: CapabilityParts,
}

impl Capability for NamedCapability {
    fn name(&self) -> &str {
        &self.name
    }

    fn parts(&self) -> CapabilityParts {
        self.parts.clone()
    }
}

/// A capability as mounted on an agent: its name and parts, computed once.
#[derive(Clone)]
pub struct MountedCapability {
    name: Arc<str>,
    parts: Arc<CapabilityParts>,
}

impl MountedCapability {
    fn mount(capability: &dyn Capability) -> Self {
        Self {
            name: capability.name().into(),
            parts: Arc::new(capability.parts()),
        }
    }

    /// The capability's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The parts it contributes.
    pub fn parts(&self) -> &CapabilityParts {
        &self.parts
    }
}

impl std::fmt::Debug for MountedCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = &self.parts;
        f.debug_struct("MountedCapability")
            .field("name", &self.name)
            .field(
                "tools",
                &p.tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            )
            .field("instructions", &p.instructions.is_some())
            .field("input_guardrails", &p.input_guardrails.len())
            .field("output_guardrails", &p.output_guardrails.len())
            .field("model_settings", &p.model_settings.is_some())
            .field(
                "requires",
                &p.requires.iter().map(|r| r.type_name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Agent {
    /// Mount a [`Capability`]: its parts are merged with the agent's own wherever they are
    /// read (see the [module docs](crate::agent::capability) for the rules). Mounting order is kept.
    #[must_use]
    pub fn capability(mut self, capability: impl Capability) -> Self {
        self.capabilities
            .push(MountedCapability::mount(&capability));
        self
    }

    /// The mounted capabilities, in mounting order.
    pub fn capabilities(&self) -> &[MountedCapability] {
        &self.capabilities
    }

    /// The agent's own tools followed by those of its capabilities.
    pub fn local_tools(&self) -> impl Iterator<Item = &FunctionTool> {
        self.tools
            .iter()
            .chain(self.capabilities.iter().flat_map(|c| c.parts.tools.iter()))
    }

    /// The agent's own input guardrails followed by those of its capabilities.
    pub fn all_input_guardrails(&self) -> impl Iterator<Item = &InputGuardrail> {
        self.input_guardrails.iter().chain(
            self.capabilities
                .iter()
                .flat_map(|c| c.parts.input_guardrails.iter()),
        )
    }

    /// The agent's own output guardrails followed by those of its capabilities.
    pub fn all_output_guardrails(&self) -> impl Iterator<Item = &OutputGuardrail> {
        self.output_guardrails.iter().chain(
            self.capabilities
                .iter()
                .flat_map(|c| c.parts.output_guardrails.iter()),
        )
    }

    /// The capabilities' model settings (in mounting order) overlaid by the agent's own.
    pub fn effective_model_settings(&self) -> std::borrow::Cow<'_, ModelSettings> {
        let mut layers = self
            .capabilities
            .iter()
            .filter_map(|c| c.parts.model_settings.as_ref())
            .peekable();
        if layers.peek().is_none() {
            return std::borrow::Cow::Borrowed(&self.model_settings);
        }
        let mut merged = ModelSettings::default();
        for layer in layers {
            merged = merged.resolve(Some(layer));
        }
        std::borrow::Cow::Owned(merged.resolve(Some(&self.model_settings)))
    }

    /// The capabilities' instruction paragraphs, in mounting order.
    pub(crate) fn capability_instructions(&self) -> impl Iterator<Item = &str> {
        self.capabilities
            .iter()
            .filter_map(|c| c.parts.instructions.as_deref())
    }
}
