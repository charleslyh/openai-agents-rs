//! `Agent.as_tool`: run an agent as a function tool (Python: `Agent.as_tool`).

use std::sync::Arc;

use serde_json::{json, Value};

use crate::agent::Agent;
use crate::error::AgentsError;
use crate::tool::{FunctionTool, ToolOutput};

use super::agent_loop::take_nested_resume_state;
use super::config::RunOptions;

/// Options for [`Agent::as_tool`] (Python: `Agent.as_tool` kwargs subset).
///
/// The Python keyword list is much longer (D-043), so fields will be added here: build one with
/// `AsToolConfig { field, ..Default::default() }` and the tail keeps you compiling.
#[derive(Debug, Clone, Default)]
pub struct AsToolConfig {
    /// Override tool name (default: sanitized agent name).
    pub name: Option<String>,
    /// Tool description (default: handoff_description or agent name).
    pub description: Option<String>,
    /// Whether a call of the agent-tool pauses for resolution before the nested run starts
    /// (Python: `needs_approval`).
    pub needs_resolution: bool,
    /// Max turns for the nested run.
    pub max_turns: Option<usize>,
}

impl Agent {
    /// Expose this agent as a function tool (Python: `Agent.as_tool`).
    ///
    /// Nested tool interruptions surface on the outer run's `interruptions`; resolve them on the
    /// outer [`crate::RunState`], then resume with [`crate::Runner::run_state`].
    ///
    /// Lives next to the runner it calls, so `agent` stays independent of `run`.
    pub fn as_tool(&self, config: AsToolConfig) -> FunctionTool {
        build_agent_as_tool(self, config)
    }
}

/// Build an `Agent.as_tool` FunctionTool (Python: `Agent.as_tool`).
fn build_agent_as_tool(agent: &Agent, config: AsToolConfig) -> FunctionTool {
    // Shared by every call of the tool; each call starts a run on the same `Arc`.
    let nested_agent = Arc::new(agent.clone());
    let tool_name = config
        .name
        .unwrap_or_else(|| crate::handoffs::transform_string_function_style(&nested_agent.name));
    let tool_description = config.description.unwrap_or_else(|| {
        nested_agent
            .handoff_description
            .clone()
            .unwrap_or_else(|| format!("Agent tool: {}", nested_agent.name))
    });
    let max_turns = config.max_turns;
    let needs = config.needs_resolution;

    FunctionTool::new_with_result(
        tool_name,
        tool_description,
        json!({
            "type": "object",
            "properties": {
                "input": { "type": "string", "description": "Input for the agent" }
            },
            "required": ["input"],
            "additionalProperties": false
        }),
        move |ctx, args| {
            let nested_agent = Arc::clone(&nested_agent);
            async move {
                let input_text = extract_as_tool_input(&args)?;
                let mut opts = RunOptions::default();
                if let Some(mt) = max_turns {
                    opts.max_turns = Some(mt);
                }

                let result = if let Some(state) = take_nested_resume_state(&ctx.tool_call_id) {
                    super::run_state_shared(nested_agent, state, opts).await?
                } else {
                    super::run_shared(nested_agent, input_text.into(), opts).await?
                };

                if result.is_interrupted() {
                    let nested_state = result.to_state()?;
                    return Ok(ToolOutput::interrupted(
                        result.interruptions.clone(),
                        nested_state,
                    ));
                }

                Ok(ToolOutput::output(result.final_output))
            }
        },
    )
    .with_needs_resolution(needs)
}

fn extract_as_tool_input(args: &str) -> Result<String, AgentsError> {
    let v: Value = serde_json::from_str(args)
        .map_err(|e| AgentsError::tool_with_source("invalid agent tool args", e))?;
    if let Some(s) = v.get("input").and_then(|x| x.as_str()) {
        return Ok(s.to_string());
    }
    if let Some(s) = v.as_str() {
        return Ok(s.to_string());
    }
    Err(AgentsError::tool(
        "agent tool requires string field `input`",
    ))
}
