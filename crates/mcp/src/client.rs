//! [`McpClient`]: an MCP server reached over a private [`Transport`].

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::Mutex as AsyncMutex;

use openai_agents_core::agent::Agent;
use openai_agents_core::error::AgentsError;
use openai_agents_core::run_context::RunContextWrapper;
use openai_agents_core::tool::NeedsApproval;

use super::http::StreamableHttpParams;
use super::rpc::{Connector, McpError, Transport};
use super::stdio::StdioParams;
use super::{RequireApproval, ToolFilter, ToolFilterContext};
use openai_agents_core::{McpCallToolResult, McpServer, McpTool};

/// Upper bound on `tools/list` pages followed, so a misbehaving server cannot loop forever.
const MAX_TOOL_PAGES: usize = 100;

/// An MCP server reached over a private `Transport` (Python: `MCPServerStdio`,
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
    max_retry_attempts: i32,
    retry_backoff_base: Duration,
    retry_backoff_max: Option<Duration>,
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
    pub(crate) fn with_connector(name: String, connector: Arc<dyn Connector>) -> Self {
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
            max_retry_attempts: 0,
            retry_backoff_base: Duration::from_secs(1),
            retry_backoff_max: None,
        }
    }

    /// A server run as a child process (Python: `MCPServerStdio`).
    pub fn stdio(params: StdioParams) -> Self {
        let name = format!("stdio: {}", params.command);
        Self::with_connector(name, Arc::new(super::stdio::StdioConnector(params)))
    }

    /// A server reached over streamable HTTP (Python: `MCPServerStreamableHttp`).
    pub fn streamable_http(params: StreamableHttpParams) -> Self {
        let name = format!("streamable_http: {}", params.url);
        Self::with_connector(name, Arc::new(super::http::HttpConnector(params)))
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

    /// Retry a failed request this many times (Python: `max_retry_attempts`; default 0, `-1`
    /// retries forever). The wait before retry `n` is `backoff_base * 2^(n-1)`, see
    /// [`retry_backoff`](Self::retry_backoff). Every request is retried, including
    /// `tools/call`, so a tool that is not idempotent may run more than once.
    pub fn max_retry_attempts(mut self, attempts: i32) -> Self {
        self.max_retry_attempts = attempts;
        self
    }

    /// Wait `base * 2^(n-1)` before retry `n`, at most `max` when given
    /// (Python: `retry_backoff_seconds_base` / `retry_backoff_seconds_max`; base defaults to 1s).
    pub fn retry_backoff(mut self, base: Duration, max: Option<Duration>) -> Self {
        self.retry_backoff_base = base;
        self.retry_backoff_max = max;
        self
    }

    /// Wrap in an `Arc`, ready for [`Agent::mcp_servers`](openai_agents_core::Agent::mcp_servers).
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

    /// One request, retried as configured (Python: `_run_with_retries`).
    async fn request(&self, method: &str, params: Option<Value>) -> Result<Value, AgentsError> {
        let mut attempts: i32 = 0;
        loop {
            match self.request_once(method, params.clone()).await {
                Ok(value) => return Ok(value),
                Err(error) => {
                    attempts = attempts.saturating_add(1);
                    if self.max_retry_attempts != -1 && attempts > self.max_retry_attempts {
                        return Err(error);
                    }
                    tokio::time::sleep(self.backoff_after(attempts - 1)).await;
                }
            }
        }
    }

    /// The wait after `taken` earlier retries: `base * 2^taken`, capped.
    pub(crate) fn backoff_after(&self, taken: i32) -> Duration {
        let factor = 2f64.powi(taken.clamp(0, 62));
        let wait = self.retry_backoff_base.as_secs_f64() * factor;
        let wait = Duration::try_from_secs_f64(wait).unwrap_or(Duration::MAX);
        self.retry_backoff_max.map_or(wait, |max| wait.min(max))
    }

    async fn request_once(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, AgentsError> {
        let transport = self.transport().await?;
        match transport
            .request(method, params, self.request_timeout)
            .await
        {
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
                    .filter_map(parse_tool),
            );
            match page
                .get("nextCursor")
                .and_then(Value::as_str)
                .filter(|c| !c.is_empty())
            {
                Some(next) if seen_cursors.insert(next.to_string()) => {
                    cursor = Some(next.to_string())
                }
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
        Ok(parse_call_result(&result))
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

/// A `tools/list` entry as a [`McpTool`]; an entry without a name is skipped.
fn parse_tool(value: &Value) -> Option<McpTool> {
    let text = |key: &str| value.get(key).and_then(Value::as_str).map(str::to_string);
    Some(McpTool {
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

/// The `result` of a `tools/call` as a [`McpCallToolResult`].
fn parse_call_result(value: &Value) -> McpCallToolResult {
    McpCallToolResult {
        content: value
            .get("content")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        structured_content: value
            .get("structuredContent")
            .filter(|s| !s.is_null())
            .cloned(),
        is_error: value
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    }
}
