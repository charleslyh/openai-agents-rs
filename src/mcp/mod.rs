//! Model Context Protocol client (Python: `agents.mcp`).
//!
//! An MCP server offers tools over a standard protocol, so the same tool works with any model:
//! the SDK lists the server's tools, turns each into a function tool the model can call, and
//! forwards the call. Give servers to an agent with [`Agent::mcp_servers`](crate::Agent::mcp_servers).
//!
//! ```ignore
//! let server = McpClient::stdio(StdioParams::new("npx").args(["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]))
//!     .shared();
//! let agent = Agent::new("files").mcp_servers(vec![server.clone()]);
//! let result = Runner::run(&agent, "list the files", RunOptions::default()).await?;
//! server.cleanup().await?;
//! ```
//!
//! Supported: stdio and streamable HTTP transports, tool listing with pagination and optional
//! caching, static and dynamic tool filters, approval policies, `structuredContent`, and
//! strict-schema conversion. A server connects on first use (`connect()` does it eagerly) and the
//! child process of a stdio server is killed when the client is dropped.
//!
//! Not ported: the legacy SSE transport, prompts and resources, tool name prefixing
//! (`include_server_in_tool_names`), `tool_meta_resolver`, per-server retries and tool guardrails.
//! The OpenAI-hosted MCP tool (`HostedMCPTool`) is out of scope.

mod rpc;

#[cfg(feature = "mcp")]
mod http;
#[cfg(feature = "mcp")]
mod stdio;

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::Mutex as AsyncMutex;

use crate::agent::Agent;
use crate::error::{AgentsError, ModelError, UserError};
use crate::run_context::RunContextWrapper;
use crate::tool::{FunctionTool, NeedsApproval};

#[cfg(feature = "mcp")]
pub use http::StreamableHttpParams;
pub use rpc::McpError;
#[cfg(feature = "mcp")]
pub use stdio::StdioParams;

use rpc::{Connector, Transport};

/// Upper bound on `tools/list` pages followed, so a misbehaving server cannot loop forever.
const MAX_TOOL_PAGES: usize = 100;

/// A tool a server offers (Python: `mcp.types.Tool`).
#[derive(Debug, Clone, PartialEq)]
pub struct McpTool {
    /// Tool name, unique within the server.
    pub name: String,
    /// What the tool does.
    pub description: Option<String>,
    /// Human readable title.
    pub title: Option<String>,
    /// JSON Schema of the arguments.
    pub input_schema: Value,
}

impl McpTool {
    fn parse(value: &Value) -> Option<Self> {
        let text = |key: &str| value.get(key).and_then(Value::as_str).map(str::to_string);
        Some(Self {
            name: text("name")?,
            description: text("description"),
            title: text("title"),
            input_schema: value
                .get("inputSchema")
                .filter(|s| s.is_object())
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
        })
    }
}

/// Answer of a tool call (Python: `mcp.types.CallToolResult`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct McpCallToolResult {
    /// Content blocks (`text`, `image`, ...), as the server sent them.
    pub content: Vec<Value>,
    /// Structured result, when the tool declares an output schema.
    pub structured_content: Option<Value>,
    /// The server flags the call as failed. The content then explains why and is shown to the
    /// model like any other result.
    pub is_error: bool,
}

impl McpCallToolResult {
    fn parse(value: &Value) -> Self {
        Self {
            content: value
                .get("content")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            structured_content: value.get("structuredContent").filter(|s| !s.is_null()).cloned(),
            is_error: value.get("isError").and_then(Value::as_bool).unwrap_or(false),
        }
    }
}

/// The text the model sees for a tool result.
///
/// Text blocks are joined with newlines. With `use_structured_content` and a successful call, the
/// structured result is sent instead, as JSON. Other blocks (images, resources) cannot travel in a
/// Chat Completions tool message, so an image becomes a short note and anything else its JSON.
/// (Python returns the content blocks as a list of tool output parts; a plain string works with
/// every provider.)
pub fn render_tool_result(result: &McpCallToolResult, use_structured_content: bool) -> String {
    if use_structured_content && !result.is_error {
        if let Some(structured) = &result.structured_content {
            return structured.to_string();
        }
    }
    result
        .content
        .iter()
        .map(|item| match item.get("type").and_then(Value::as_str) {
            Some("text") => item.get("text").and_then(Value::as_str).unwrap_or_default().to_string(),
            Some("image") => format!(
                "[image: {}]",
                item.get("mimeType").and_then(Value::as_str).unwrap_or("unknown type")
            ),
            _ => item.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------------------------
// Tool filtering and approval

/// What a dynamic tool filter is told (Python: `ToolFilterContext`).
#[derive(Clone)]
pub struct ToolFilterContext {
    /// The run context.
    pub run_context: RunContextWrapper,
    /// The agent listing the tools.
    pub agent: Arc<Agent>,
    /// Name of the server.
    pub server_name: String,
}

type DynamicFilter =
    Arc<dyn Fn(ToolFilterContext, McpTool) -> Pin<Box<dyn Future<Output = bool> + Send>> + Send + Sync>;

/// Which of a server's tools the agent may use (Python: `ToolFilter`).
#[derive(Clone)]
pub enum ToolFilter {
    /// Fixed lists. `allowed` is applied first, then `blocked`.
    Static {
        /// If set, only these tools pass.
        allowed: Option<Vec<String>>,
        /// These tools never pass.
        blocked: Option<Vec<String>>,
    },
    /// Decide per tool, per listing.
    Dynamic(DynamicFilter),
}

impl std::fmt::Debug for ToolFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static { allowed, blocked } => f
                .debug_struct("Static")
                .field("allowed", allowed)
                .field("blocked", blocked)
                .finish(),
            Self::Dynamic(_) => f.write_str("Dynamic(..)"),
        }
    }
}

impl ToolFilter {
    /// Only these tools.
    pub fn allow<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Static {
            allowed: Some(names.into_iter().map(Into::into).collect()),
            blocked: None,
        }
    }

    /// Everything except these tools.
    pub fn block<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Static {
            allowed: None,
            blocked: Some(names.into_iter().map(Into::into).collect()),
        }
    }

    /// Decide with an async closure.
    pub fn dynamic<F, Fut>(f: F) -> Self
    where
        F: Fn(ToolFilterContext, McpTool) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        Self::Dynamic(Arc::new(move |ctx, tool| Box::pin(f(ctx, tool))))
    }
}

/// Which tool calls need human approval (Python: `require_approval`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum RequireApproval {
    /// No call needs approval.
    #[default]
    Never,
    /// Every call needs approval.
    Always,
    /// Per tool; a tool in neither list needs no approval.
    PerTool {
        /// Tools that always need approval.
        always: Vec<String>,
        /// Tools that never do.
        never: Vec<String>,
    },
}

impl RequireApproval {
    fn for_tool(&self, name: &str) -> NeedsApproval {
        NeedsApproval::Fixed(match self {
            Self::Never => false,
            Self::Always => true,
            Self::PerTool { always, .. } => always.iter().any(|n| n == name),
        })
    }
}

/// How an agent turns MCP tools into function tools (Python: `MCPConfig`).
#[derive(Debug, Clone, Copy, Default)]
pub struct McpConfig {
    /// Rewrite each tool's input schema into OpenAI's strict form. A schema that cannot be made
    /// strict is used as it is.
    pub convert_schemas_to_strict: bool,
}

// ---------------------------------------------------------------------------------------------
// The server abstraction

/// A source of tools (Python: `MCPServer`).
///
/// [`McpClient`] implements it for real servers; implement it yourself for an in-process source.
#[async_trait]
pub trait McpServer: std::fmt::Debug + Send + Sync {
    /// Name used in messages and traces.
    fn name(&self) -> &str;

    /// Connect now. Optional: the first use connects on its own.
    async fn connect(&self) -> Result<(), AgentsError>;

    /// Disconnect and release resources. A later call reconnects.
    async fn cleanup(&self) -> Result<(), AgentsError>;

    /// The tools this agent may use, after filtering.
    async fn list_tools(
        &self,
        context: &RunContextWrapper,
        agent: &Agent,
    ) -> Result<Vec<McpTool>, AgentsError>;

    /// Call a tool.
    async fn call_tool(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> Result<McpCallToolResult, AgentsError>;

    /// Forget a cached tool list.
    fn invalidate_tools_cache(&self) {}

    /// Whether calls of `tool` need approval.
    fn needs_approval_for(&self, _tool: &McpTool) -> NeedsApproval {
        NeedsApproval::Fixed(false)
    }

    /// Send `structuredContent` instead of the content blocks when a result has it.
    fn use_structured_content(&self) -> bool {
        false
    }
}

/// An MCP server reached over a [`Transport`] (Python: `MCPServerStdio`,
/// `MCPServerStreamableHttp`).
pub struct McpClient {
    name: String,
    connector: Arc<dyn Connector>,
    connection: AsyncMutex<Option<Arc<dyn Transport>>>,
    request_timeout: Duration,
    cache_tools_list: bool,
    cached_tools: Mutex<Option<Vec<McpTool>>>,
    tool_filter: Option<ToolFilter>,
    require_approval: RequireApproval,
    use_structured_content: bool,
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpClient")
            .field("name", &self.name)
            .field("cache_tools_list", &self.cache_tools_list)
            .field("tool_filter", &self.tool_filter)
            .field("require_approval", &self.require_approval)
            .finish()
    }
}

impl McpClient {
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    fn with_connector(name: String, connector: Arc<dyn Connector>) -> Self {
        Self {
            name,
            connector,
            connection: AsyncMutex::new(None),
            request_timeout: Duration::from_secs(30),
            cache_tools_list: false,
            cached_tools: Mutex::new(None),
            tool_filter: None,
            require_approval: RequireApproval::Never,
            use_structured_content: false,
        }
    }

    /// A server run as a child process (Python: `MCPServerStdio`).
    #[cfg(feature = "mcp")]
    pub fn stdio(params: StdioParams) -> Self {
        let name = format!("stdio: {}", params.command);
        Self::with_connector(name, Arc::new(stdio::StdioConnector(params)))
    }

    /// A server reached over streamable HTTP (Python: `MCPServerStreamableHttp`).
    #[cfg(feature = "mcp")]
    pub fn streamable_http(params: StreamableHttpParams) -> Self {
        let name = format!("streamable_http: {}", params.url);
        Self::with_connector(name, Arc::new(http::HttpConnector(params)))
    }

    /// Name the server (the default describes the transport).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// How long one request may take (default 30 seconds; Python: `client_session_timeout_seconds`
    /// defaults to 5, which is short for tools that do real work).
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Keep the tool list after the first fetch (Python: `cache_tools_list`). Only turn this on
    /// for servers whose tools do not change; see [`McpServer::invalidate_tools_cache`].
    pub fn cache_tools_list(mut self, cache: bool) -> Self {
        self.cache_tools_list = cache;
        self
    }

    /// Restrict the tools the agent sees.
    pub fn tool_filter(mut self, filter: ToolFilter) -> Self {
        self.tool_filter = Some(filter);
        self
    }

    /// Require approval for tool calls.
    pub fn require_approval(mut self, policy: RequireApproval) -> Self {
        self.require_approval = policy;
        self
    }

    /// Prefer `structuredContent` over content blocks.
    pub fn with_structured_content(mut self, enabled: bool) -> Self {
        self.use_structured_content = enabled;
        self
    }

    /// Wrap in an `Arc`, ready for [`Agent::mcp_servers`](crate::Agent::mcp_servers).
    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    fn error(&self, error: McpError) -> AgentsError {
        AgentsError::tool_with_source(format!("MCP server `{}`: {error}", self.name), error)
    }

    async fn transport(&self) -> Result<Arc<dyn Transport>, AgentsError> {
        let mut connection = self.connection.lock().await;
        if let Some(transport) = connection.as_ref() {
            return Ok(Arc::clone(transport));
        }
        let transport = self
            .connector
            .connect(self.request_timeout)
            .await
            .map_err(|e| self.error(e))?;
        *connection = Some(Arc::clone(&transport));
        Ok(transport)
    }

    async fn request(&self, method: &str, params: Option<Value>) -> Result<Value, AgentsError> {
        let transport = self.transport().await?;
        match transport.request(method, params, self.request_timeout).await {
            Ok(value) => Ok(value),
            Err(error) => {
                // A dead connection is dropped so the next call starts a fresh one.
                if matches!(error, McpError::Closed(_)) {
                    if let Some(dead) = self.connection.lock().await.take() {
                        dead.close().await;
                    }
                }
                Err(self.error(error))
            }
        }
    }

    async fn fetch_tools(&self) -> Result<Vec<McpTool>, AgentsError> {
        if self.cache_tools_list {
            if let Some(tools) = self.cached_tools.lock().expect("cache").clone() {
                return Ok(tools);
            }
        }
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        let mut seen_cursors = HashSet::new();
        for _ in 0..MAX_TOOL_PAGES {
            let params = cursor.as_ref().map(|c| json!({"cursor": c}));
            let page = self.request("tools/list", params).await?;
            tools.extend(
                page.get("tools")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(McpTool::parse),
            );
            match page.get("nextCursor").and_then(Value::as_str).filter(|c| !c.is_empty()) {
                Some(next) if seen_cursors.insert(next.to_string()) => cursor = Some(next.to_string()),
                _ => break,
            }
        }
        if self.cache_tools_list {
            *self.cached_tools.lock().expect("cache") = Some(tools.clone());
        }
        Ok(tools)
    }
}

#[async_trait]
impl McpServer for McpClient {
    fn name(&self) -> &str {
        &self.name
    }

    async fn connect(&self) -> Result<(), AgentsError> {
        self.transport().await.map(|_| ())
    }

    async fn cleanup(&self) -> Result<(), AgentsError> {
        *self.cached_tools.lock().expect("cache") = None;
        if let Some(transport) = self.connection.lock().await.take() {
            transport.close().await;
        }
        Ok(())
    }

    async fn list_tools(
        &self,
        context: &RunContextWrapper,
        agent: &Agent,
    ) -> Result<Vec<McpTool>, AgentsError> {
        let tools = self.fetch_tools().await?;
        Ok(match &self.tool_filter {
            None => tools,
            Some(ToolFilter::Static { allowed, blocked }) => tools
                .into_iter()
                .filter(|t| allowed.as_ref().is_none_or(|a| a.contains(&t.name)))
                .filter(|t| !blocked.as_ref().is_some_and(|b| b.contains(&t.name)))
                .collect(),
            Some(ToolFilter::Dynamic(decide)) => {
                let agent = Arc::new(agent.clone());
                let mut kept = Vec::new();
                for tool in tools {
                    let filter_context = ToolFilterContext {
                        run_context: context.clone(),
                        agent: Arc::clone(&agent),
                        server_name: self.name.clone(),
                    };
                    if decide(filter_context, tool.clone()).await {
                        kept.push(tool);
                    }
                }
                kept
            }
        })
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> Result<McpCallToolResult, AgentsError> {
        let params = json!({"name": name, "arguments": arguments.unwrap_or_else(|| json!({}))});
        let result = self.request("tools/call", Some(params)).await?;
        Ok(McpCallToolResult::parse(&result))
    }

    fn invalidate_tools_cache(&self) {
        *self.cached_tools.lock().expect("cache") = None;
    }

    fn needs_approval_for(&self, tool: &McpTool) -> NeedsApproval {
        self.require_approval.for_tool(&tool.name)
    }

    fn use_structured_content(&self) -> bool {
        self.use_structured_content
    }
}

// ---------------------------------------------------------------------------------------------
// Turning MCP tools into function tools

/// A tool call must provide every property the schema marks as required (Python:
/// `_validate_required_parameters`), which gives the model a clearer error than the server's.
fn check_required_arguments(tool: &McpTool, arguments: &serde_json::Map<String, Value>) -> Result<(), AgentsError> {
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
fn to_function_tool(server: &Arc<dyn McpServer>, tool: McpTool, config: &McpConfig) -> FunctionTool {
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

    let needs_approval = server.needs_approval_for(&tool);
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
                Ok(Value::String(render_tool_result(&result, server.use_structured_content())))
            }
        }
    };
    let mut function_tool = FunctionTool::new(tool.name.clone(), description, schema, invoker);
    function_tool.strict_json_schema = strict;
    function_tool.needs_approval = needs_approval;
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::tool::ToolContext;

    /// A scripted server: `tools/list` pages in order, `tools/call` answers with its arguments.
    #[derive(Default)]
    struct FakeTransport {
        pages: Vec<Value>,
        requests: Mutex<Vec<(String, Option<Value>)>>,
        /// Fail the next request as if the connection died.
        die_once: std::sync::atomic::AtomicBool,
    }

    #[async_trait]
    impl Transport for FakeTransport {
        async fn request(
            &self,
            method: &str,
            params: Option<Value>,
            _timeout: Duration,
        ) -> Result<Value, McpError> {
            self.requests.lock().unwrap().push((method.to_string(), params.clone()));
            if self.die_once.swap(false, Ordering::SeqCst) {
                return Err(McpError::Closed("gone".into()));
            }
            match method {
                "tools/list" => {
                    let index = params
                        .as_ref()
                        .and_then(|p| p.get("cursor"))
                        .and_then(Value::as_str)
                        .map(|c| c.parse::<usize>().unwrap())
                        .unwrap_or(0);
                    Ok(self.pages.get(index).cloned().unwrap_or_else(|| json!({"tools": []})))
                }
                "tools/call" => Ok(json!({
                    "content": [{"type": "text", "text": params.unwrap()["arguments"].to_string()}],
                    "structuredContent": {"ok": true},
                })),
                other => Err(McpError::Protocol(format!("unexpected {other}"))),
            }
        }

        async fn notify(&self, _: &str, _: Option<Value>) -> Result<(), McpError> {
            Ok(())
        }

        async fn close(&self) {}
    }

    struct FakeConnector {
        transport: Arc<FakeTransport>,
        connects: AtomicUsize,
    }

    #[async_trait]
    impl Connector for FakeConnector {
        async fn connect(&self, _: Duration) -> Result<Arc<dyn Transport>, McpError> {
            self.connects.fetch_add(1, Ordering::SeqCst);
            Ok(self.transport.clone())
        }
    }

    fn tool_json(name: &str) -> Value {
        json!({"name": name, "description": format!("{name} tool"),
               "inputSchema": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}})
    }

    fn fake(pages: Vec<Value>) -> (McpClient, Arc<FakeTransport>, Arc<FakeConnector>) {
        let transport = Arc::new(FakeTransport { pages, ..Default::default() });
        let connector = Arc::new(FakeConnector { transport: transport.clone(), connects: AtomicUsize::new(0) });
        (McpClient::with_connector("fake".into(), connector.clone()), transport, connector)
    }

    fn listed(transport: &FakeTransport) -> usize {
        transport.requests.lock().unwrap().iter().filter(|(m, _)| m == "tools/list").count()
    }

    async fn names(server: &dyn McpServer) -> Vec<String> {
        let context = RunContextWrapper::new(None);
        server
            .list_tools(&context, &Agent::new("a"))
            .await
            .unwrap()
            .into_iter()
            .map(|t| t.name)
            .collect()
    }

    #[tokio::test]
    async fn tool_lists_are_paginated_and_a_repeated_cursor_stops() {
        let (client, transport, _) = fake(vec![
            json!({"tools": [tool_json("a"), tool_json("b")], "nextCursor": "1"}),
            json!({"tools": [tool_json("c")], "nextCursor": "1"}),
        ]);
        assert_eq!(names(&client).await, ["a", "b", "c"]);
        assert_eq!(listed(&transport), 2, "the repeated cursor ends the walk");
    }

    #[tokio::test]
    async fn the_tool_list_is_cached_only_on_request() {
        let (client, transport, _) = fake(vec![json!({"tools": [tool_json("a")]})]);
        names(&client).await;
        names(&client).await;
        assert_eq!(listed(&transport), 2, "no caching by default");

        let (client, transport, _) = fake(vec![json!({"tools": [tool_json("a")]})]);
        let client = client.cache_tools_list(true);
        names(&client).await;
        names(&client).await;
        assert_eq!(listed(&transport), 1);
        client.invalidate_tools_cache();
        names(&client).await;
        assert_eq!(listed(&transport), 2, "invalidating forces a refetch");
    }

    #[tokio::test]
    async fn filters_select_tools() {
        let page = || vec![json!({"tools": [tool_json("a"), tool_json("b"), tool_json("c")]})];
        let (client, ..) = fake(page());
        assert_eq!(names(&client.tool_filter(ToolFilter::allow(["a", "c"]))).await, ["a", "c"]);
        let (client, ..) = fake(page());
        assert_eq!(names(&client.tool_filter(ToolFilter::block(["b"]))).await, ["a", "c"]);
        let (client, ..) = fake(page());
        let both = ToolFilter::Static {
            allowed: Some(vec!["a".into(), "b".into()]),
            blocked: Some(vec!["b".into()]),
        };
        assert_eq!(names(&client.tool_filter(both)).await, ["a"], "allowed first, then blocked");
        let (client, ..) = fake(page());
        let dynamic = ToolFilter::dynamic(|ctx, tool| async move {
            ctx.server_name == "fake" && ctx.agent.name == "a" && tool.name != "a"
        });
        assert_eq!(names(&client.tool_filter(dynamic)).await, ["b", "c"]);
    }

    #[tokio::test]
    async fn a_dead_connection_is_replaced_on_the_next_call() {
        let (client, transport, connector) = fake(vec![json!({"tools": [tool_json("a")]})]);
        names(&client).await;
        transport.die_once.store(true, Ordering::SeqCst);
        assert!(client.call_tool("a", None).await.is_err());
        assert!(client.call_tool("a", Some(json!({"x": 1}))).await.is_ok());
        assert_eq!(connector.connects.load(Ordering::SeqCst), 2, "reconnected once");
        client.cleanup().await.unwrap();
        client.connect().await.unwrap();
        assert_eq!(connector.connects.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn approval_policies_map_to_needs_approval() {
        let check = |policy: &RequireApproval, name: &str| match policy.for_tool(name) {
            NeedsApproval::Fixed(value) => value,
            NeedsApproval::Dynamic(_) => panic!("fixed expected"),
        };
        assert!(!check(&RequireApproval::Never, "a"));
        assert!(check(&RequireApproval::Always, "a"));
        let per_tool = RequireApproval::PerTool { always: vec!["rm".into()], never: vec!["ls".into()] };
        assert!(check(&per_tool, "rm"));
        assert!(!check(&per_tool, "ls"));
        assert!(!check(&per_tool, "other"), "unlisted tools need no approval");
    }

    async fn invoke(tool: &FunctionTool, arguments: &str) -> Result<Value, AgentsError> {
        let context = ToolContext::new(tool.name.clone(), "c1", arguments, RunContextWrapper::new(None));
        (tool.on_invoke_tool)(context, arguments.to_string()).await.map(|r| r.output.unwrap())
    }

    async fn convert(client: McpClient, config: McpConfig) -> Result<Vec<FunctionTool>, AgentsError> {
        let server: Arc<dyn McpServer> = client.shared();
        mcp_function_tools(&[server], &config, &RunContextWrapper::new(None), &Agent::new("a")).await
    }

    #[tokio::test]
    async fn function_tools_validate_arguments_and_render_results() {
        let (client, transport, _) = fake(vec![json!({"tools": [
            tool_json("a"),
            {"name": "bare", "title": "Bare title", "inputSchema": {"type": "object"}},
        ]})]);
        let client = client.require_approval(RequireApproval::Always);
        let tools = convert(client, McpConfig::default()).await.unwrap();
        let (a, bare) = (&tools[0], &tools[1]);
        assert_eq!(a.description, "a tool");
        assert_eq!(bare.description, "Bare title", "falls back to the title");
        assert_eq!(bare.params_json_schema["properties"], json!({}), "`properties` is added");
        assert!(matches!(a.needs_approval, NeedsApproval::Fixed(true)));

        assert_eq!(invoke(a, r#"{"x": 1}"#).await.unwrap(), json!("{\"x\":1}"));
        let missing = invoke(a, "{}").await.unwrap_err();
        assert!(missing.to_string().contains("missing required parameters: x"), "{missing}");
        assert!(invoke(a, "not json").await.is_err());
        assert!(invoke(a, "[1]").await.is_err(), "arguments must be an object");
        assert_eq!(invoke(bare, "").await.unwrap(), json!("{}"), "empty input means no arguments");
        let calls = transport.requests.lock().unwrap().iter().filter(|(m, _)| m == "tools/call").count();
        assert_eq!(calls, 2, "invalid calls never reach the server");
    }

    #[tokio::test]
    async fn strict_conversion_is_opt_in() {
        let page = || vec![json!({"tools": [tool_json("a")]})];
        let strict = McpConfig { convert_schemas_to_strict: true };
        let tools = convert(fake(page()).0, strict).await.unwrap();
        assert!(tools[0].strict_json_schema);
        assert_eq!(tools[0].params_json_schema["additionalProperties"], false);
        let tools = convert(fake(page()).0, McpConfig::default()).await.unwrap();
        assert!(!tools[0].strict_json_schema);
    }

    #[tokio::test]
    async fn duplicate_names_across_servers_are_rejected() {
        let one: Arc<dyn McpServer> = fake(vec![json!({"tools": [tool_json("dup"), tool_json("x")]})]).0.shared();
        let two: Arc<dyn McpServer> = fake(vec![json!({"tools": [tool_json("dup")]})]).0.shared();
        let error = mcp_function_tools(
            &[one, two],
            &McpConfig::default(),
            &RunContextWrapper::new(None),
            &Agent::new("a"),
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("Duplicate tool names found across MCP servers: \"dup\""),
            "{error}"
        );
    }

    #[test]
    fn results_render_as_text() {
        let result = McpCallToolResult {
            content: vec![
                json!({"type": "text", "text": "one"}),
                json!({"type": "image", "data": "aGk=", "mimeType": "image/png"}),
                json!({"type": "resource_link", "uri": "file:///x"}),
                json!({"type": "text", "text": "two"}),
            ],
            structured_content: Some(json!({"n": 1})),
            is_error: false,
        };
        assert_eq!(
            render_tool_result(&result, false),
            "one\n[image: image/png]\n{\"type\":\"resource_link\",\"uri\":\"file:///x\"}\ntwo"
        );
        assert_eq!(render_tool_result(&result, true), "{\"n\":1}");
        let failed = McpCallToolResult { is_error: true, ..result };
        assert!(render_tool_result(&failed, true).starts_with("one"), "errors keep their content");
    }
}
