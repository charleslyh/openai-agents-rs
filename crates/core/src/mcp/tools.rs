//! Turning MCP tools into function tools (Python: `MCPUtil`).

use std::collections::HashSet;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::agent::Agent;
use crate::error::{AgentsError, ModelError, UserError};
use crate::run_context::RunContextWrapper;
use crate::tool::FunctionTool;

use super::{render_tool_result, McpServer, McpTool};

/// How an agent turns MCP tools into function tools (Python: `MCPConfig`).
#[derive(Debug, Clone, Copy, Default)]
pub struct McpConfig {
    /// Rewrite each tool's input schema into OpenAI's strict form. A schema that cannot be made
    /// strict is used as it is.
    pub convert_schemas_to_strict: bool,
}

/// A tool call must provide every property the schema marks as required (Python:
/// `_validate_required_parameters`), which gives the model a clearer error than the server's.
fn check_required_arguments(
    tool: &McpTool,
    arguments: &serde_json::Map<String, Value>,
) -> Result<(), AgentsError> {
    let missing: Vec<&str> = tool
        .input_schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|name| !arguments.contains_key(*name))
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(UserError::new(format!(
            "Tool `{}` is missing required parameters: {}",
            tool.name,
            missing.join(", ")
        ))
        .into())
    }
}

/// Make a function tool that calls `tool` on `server` (Python: `MCPUtil.to_function_tool`).
fn to_function_tool(
    server: &Arc<dyn McpServer>,
    tool: McpTool,
    config: &McpConfig,
) -> FunctionTool {
    let mut schema = tool.input_schema.clone();
    if schema.get("properties").is_none() {
        // MCP does not require `properties`, but model APIs do.
        schema["properties"] = json!({});
    }
    let mut strict = false;
    if config.convert_schemas_to_strict {
        if let Ok(converted) = crate::strict_schema::ensure_strict_json_schema(&schema) {
            schema = converted;
            strict = true;
        }
    }
    let description = tool
        .description
        .clone()
        .or_else(|| tool.title.clone())
        .unwrap_or_default();

    let needs_resolution = server.needs_resolution_for(&tool);
    let invoker = {
        let server = Arc::clone(server);
        let tool = tool.clone();
        move |_ctx, arguments: String| {
            let server = Arc::clone(&server);
            let tool = tool.clone();
            async move {
                let parsed: Value = if arguments.trim().is_empty() {
                    json!({})
                } else {
                    serde_json::from_str(&arguments).map_err(|_| {
                        ModelError::Behavior(format!(
                            "Invalid JSON input for tool {}: {arguments}",
                            tool.name
                        ))
                    })?
                };
                let Value::Object(object) = &parsed else {
                    return Err(ModelError::Behavior(format!(
                        "Invalid JSON input for tool {}: expected a JSON object",
                        tool.name
                    ))
                    .into());
                };
                check_required_arguments(&tool, object)?;
                let result = server.call_tool(&tool.name, Some(parsed)).await?;
                Ok(Value::String(render_tool_result(
                    &result,
                    server.use_structured_content(),
                )))
            }
        }
    };
    let mut function_tool = FunctionTool::new(tool.name.clone(), description, schema, invoker);
    function_tool.strict_json_schema = strict;
    function_tool.needs_resolution = needs_resolution;
    function_tool
}

/// Function tools for everything `servers` offer to `agent` (Python: `get_all_function_tools`).
///
/// Two servers offering the same tool name is an error: the model could not tell them apart.
pub async fn mcp_function_tools(
    servers: &[Arc<dyn McpServer>],
    config: &McpConfig,
    context: &RunContextWrapper,
    agent: &Agent,
) -> Result<Vec<FunctionTool>, AgentsError> {
    let mut tools: Vec<FunctionTool> = Vec::new();
    let mut names: HashSet<String> = HashSet::new();
    for server in servers {
        let offered = server.list_tools(context, agent).await?;
        let batch: Vec<FunctionTool> = offered
            .into_iter()
            .map(|tool| to_function_tool(server, tool, config))
            .collect();
        let mut duplicates: Vec<String> = batch
            .iter()
            .filter(|t| names.contains(&t.name))
            .map(|t| format!("{:?}", t.name))
            .collect();
        if !duplicates.is_empty() {
            duplicates.sort();
            return Err(UserError::new(format!(
                "Duplicate tool names found across MCP servers: {}",
                duplicates.join(", ")
            ))
            .into());
        }
        names.extend(batch.iter().map(|t| t.name.clone()));
        tools.extend(batch);
    }
    Ok(tools)
}
