//! Capabilities loaded on demand (Rust-only; `docs/DEVIATIONS.md` X-021).
//!
//! A capability marked [`CapabilityParts::deferred`] stays out of the prompt: the model sees one
//! catalog line per unloaded capability and a framework tool, `load_capability`, that makes the
//! capability's tools and instructions visible from the next step on. Which capabilities are
//! loaded lives in the [`RunContextWrapper`], so it is per run, and a paused run carries it in
//! its [`RunState`](crate::RunState).

use std::sync::Arc;

use serde_json::{json, Value};

use crate::run_context::RunContextWrapper;
use crate::tool::FunctionTool;

use super::capability::MountedCapability;
use super::Agent;

/// Name of the framework tool that loads a deferred capability.
pub const LOAD_CAPABILITY_TOOL: &str = "load_capability";

impl Agent {
    /// Whether `capability`'s tools and instructions are visible now.
    fn is_capability_visible(
        &self,
        capability: &MountedCapability,
        context: &RunContextWrapper,
    ) -> bool {
        capability.parts().deferred.is_none()
            || context.is_capability_loaded(&self.name, capability.name())
    }

    /// Deferred capabilities not loaded yet, as `(name, description)`, one per name.
    fn unloaded_capabilities(&self, context: &RunContextWrapper) -> Vec<(&str, &str)> {
        let mut out: Vec<(&str, &str)> = Vec::new();
        for capability in &self.capabilities {
            let Some(description) = capability.parts().deferred.as_deref() else {
                continue;
            };
            let name = capability.name();
            if !context.is_capability_loaded(&self.name, name)
                && !out.iter().any(|(seen, _)| *seen == name)
            {
                out.push((name, description));
            }
        }
        out
    }

    /// The agent's own tools followed by those of its capabilities that are visible now.
    pub(crate) fn visible_local_tools<'a>(
        &'a self,
        context: &'a RunContextWrapper,
    ) -> impl Iterator<Item = &'a FunctionTool> {
        self.tools.iter().chain(
            self.capabilities
                .iter()
                .filter(move |c| self.is_capability_visible(c, context))
                .flat_map(|c| c.parts().tools.iter()),
        )
    }

    /// Paragraphs the capabilities add to the instructions, in mounting order: those of the
    /// visible capabilities, then the catalog of the deferred ones still unloaded.
    pub(crate) fn capability_instructions(&self, context: &RunContextWrapper) -> Vec<String> {
        let mut out: Vec<String> = self
            .capabilities
            .iter()
            .filter(|c| self.is_capability_visible(c, context))
            .filter_map(|c| c.parts().instructions.clone())
            .collect();
        let unloaded = self.unloaded_capabilities(context);
        if !unloaded.is_empty() {
            let mut catalog = format!(
                "More capabilities can be loaded on demand. Call `{LOAD_CAPABILITY_TOOL}` with a \
                 name to get its tools:"
            );
            for (name, description) in unloaded {
                catalog.push_str(&format!("\n- {name}: {description}"));
            }
            out.push(catalog);
        }
        out
    }

    /// The `load_capability` tool for this turn; `None` when nothing is left to load.
    ///
    /// The schema lists the unloaded names as an enum, and the call checks the name again: a
    /// name that is not a deferred capability of this agent is answered with the loadable names,
    /// it never loads anything.
    pub(crate) fn load_capability_tool(
        self: &Arc<Self>,
        context: &RunContextWrapper,
    ) -> Option<FunctionTool> {
        let loadable: Vec<String> = self
            .unloaded_capabilities(context)
            .into_iter()
            .map(|(name, _)| name.to_owned())
            .collect();
        if loadable.is_empty() {
            return None;
        }
        let schema = json!({
            "type": "object",
            "properties": {"name": {"type": "string", "enum": loadable}},
            "required": ["name"],
            "additionalProperties": false
        });
        let agent = Arc::clone(self);
        Some(FunctionTool::new(
            LOAD_CAPABILITY_TOOL,
            "Load a capability from the catalog in the instructions; its tools become available \
             on your next step.",
            schema,
            move |tool_context, arguments| {
                let agent = Arc::clone(&agent);
                async move {
                    let run_context = &tool_context.run_context;
                    let requested = serde_json::from_str::<Value>(&arguments)
                        .ok()
                        .and_then(|v| v.get("name").and_then(Value::as_str).map(str::to_owned));
                    let capability = requested.as_deref().and_then(|name| {
                        agent
                            .capabilities
                            .iter()
                            .find(|c| c.name() == name && c.parts().deferred.is_some())
                    });
                    let Some(capability) = capability else {
                        let names: Vec<&str> = agent
                            .unloaded_capabilities(run_context)
                            .into_iter()
                            .map(|(name, _)| name)
                            .collect();
                        return Ok(Value::String(format!(
                            "No loadable capability with that name. Loadable: {}",
                            if names.is_empty() {
                                "none".to_owned()
                            } else {
                                names.join(", ")
                            }
                        )));
                    };
                    let name = capability.name();
                    if !run_context.load_capability(&agent.name, name) {
                        return Ok(Value::String(format!("`{name}` is already loaded.")));
                    }
                    let tools: Vec<&str> = capability
                        .parts()
                        .tools
                        .iter()
                        .map(|t| t.name.as_str())
                        .collect();
                    Ok(Value::String(format!(
                        "Loaded `{name}`. Available on your next step: {}",
                        if tools.is_empty() {
                            "(instructions only)".to_owned()
                        } else {
                            tools.join(", ")
                        }
                    )))
                }
            },
        ))
    }
}
