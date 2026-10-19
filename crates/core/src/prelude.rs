//! The types most programs need, for a single glob import.
//!
//! ```
//! use openai_agents_core::prelude::*;
//!
//! let agent = Agent::new("Assistant").instructions("Be brief.");
//! let options = RunOptions::new().max_turns(3);
//! # let _ = (agent, options);
//! ```
//!
//! The set is chosen by how often the examples and the web demo name each type: building an agent
//! and its tools, running it, reading the result, pausing for a resolution, and the traits a
//! custom tool, model, provider or session implements. Everything else stays at the crate root.
//! Local items and explicit imports take precedence over a glob, so a name here never shadows yours.
//!
//! The `openai-agents` crate has its own `prelude` that adds the `#[function_tool]` macro and
//! uses its `Runner`, which installs the default OpenAI provider.

pub use crate::agent::{Agent, Capability, CapabilityParts};
pub use crate::error::AgentsError;
pub use crate::extensions::Extensions;
pub use crate::guardrail::{
    input_guardrail, output_guardrail, GuardrailFunctionOutput, InputGuardrail, OutputGuardrail,
};
pub use crate::handoffs::{handoff, Handoff};
pub use crate::items::{InputLike, RunItem, ToolInterruptionItem};
pub use crate::memory::{InMemorySession, Session};
pub use crate::model::{Model, ModelProvider};
pub use crate::model_settings::{ModelSettings, Reasoning, ReasoningEffort};
pub use crate::result::RunResult;
pub use crate::run::{RunOptions, Runner};
pub use crate::run_context::RunContextWrapper;
pub use crate::run_state::{RunState, ToolResolution};
pub use crate::runtime::Runtime;
pub use crate::stream_events::StreamEvent;
pub use crate::tool::{FunctionTool, NeedsResolution, Tool, ToolContext, ToolOutput};
