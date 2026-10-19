//! Unit tests for the MCP client and tool conversion.

#![cfg(test)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

use openai_agents_core::agent::Agent;
use openai_agents_core::error::AgentsError;
use openai_agents_core::run_context::RunContextWrapper;
use openai_agents_core::tool::{FunctionTool, NeedsResolution, ToolContext};
use openai_agents_core::{
    mcp_function_tools, render_tool_result, McpCallToolResult, McpConfig, McpServer,
};

use super::rpc::{Connector, Transport};
use super::{McpClient, McpError, RequireApproval, ToolFilter};

/// A scripted server: `tools/list` pages in order, `tools/call` answers with its arguments.
#[derive(Default)]
struct FakeTransport {
    pages: Vec<Value>,
    requests: Mutex<Vec<(String, Option<Value>)>>,
    /// Fail the next request as if the connection died.
    die_once: std::sync::atomic::AtomicBool,
    /// Fail this many more requests the same way.
    deaths: AtomicUsize,
}

#[async_trait]
impl Transport for FakeTransport {
    async fn request(
        &self,
        method: &str,
        params: Option<Value>,
        _timeout: Duration,
    ) -> Result<Value, McpError> {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((method.to_string(), params.clone()));
        if self.die_once.swap(false, Ordering::SeqCst)
            || self
                .deaths
                .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
        {
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
                Ok(self
                    .pages
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| json!({"tools": []})))
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
    let transport = Arc::new(FakeTransport {
        pages,
        ..Default::default()
    });
    let connector = Arc::new(FakeConnector {
        transport: transport.clone(),
        connects: AtomicUsize::new(0),
    });
    (
        McpClient::with_connector("fake".into(), connector.clone()),
        transport,
        connector,
    )
}

fn listed(transport: &FakeTransport) -> usize {
    transport
        .requests
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|(m, _)| m == "tools/list")
        .count()
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
    assert_eq!(
        names(&client.tool_filter(ToolFilter::allow(["a", "c"]))).await,
        ["a", "c"]
    );
    let (client, ..) = fake(page());
    assert_eq!(
        names(&client.tool_filter(ToolFilter::block(["b"]))).await,
        ["a", "c"]
    );
    let (client, ..) = fake(page());
    let both = ToolFilter::Static {
        allowed: Some(vec!["a".into(), "b".into()]),
        blocked: Some(vec!["b".into()]),
    };
    assert_eq!(
        names(&client.tool_filter(both)).await,
        ["a"],
        "allowed first, then blocked"
    );
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
    assert_eq!(
        connector.connects.load(Ordering::SeqCst),
        2,
        "reconnected once"
    );
    client.cleanup().await.unwrap();
    client.connect().await.unwrap();
    assert_eq!(connector.connects.load(Ordering::SeqCst), 3);
}

/// Fail the next `n` requests as if the connection died.
impl FakeTransport {
    fn die_times(&self, n: usize) {
        self.deaths.store(n, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn failed_requests_are_retried_with_backoff() {
    let ms = Duration::from_millis;
    let (client, transport, connector) = fake(vec![json!({"tools": [tool_json("a")]})]);
    let client = client
        .max_retry_attempts(2)
        .retry_backoff(ms(1), Some(ms(2)));
    transport.die_times(2);
    assert!(
        client.call_tool("a", None).await.is_ok(),
        "two failures fit in two retries"
    );
    assert_eq!(
        connector.connects.load(Ordering::SeqCst),
        3,
        "each retry reconnected"
    );

    transport.die_times(3);
    assert!(
        client.call_tool("a", None).await.is_err(),
        "a third failure is reported"
    );

    // The default is no retry, and -1 never gives up.
    let (client, transport, _) = fake(vec![json!({"tools": [tool_json("a")]})]);
    transport.die_times(1);
    assert!(client.call_tool("a", None).await.is_err());
    let client = client.max_retry_attempts(-1).retry_backoff(ms(1), None);
    transport.die_times(5);
    assert!(client.call_tool("a", None).await.is_ok());

    // base * 2^n, capped.
    let client = client.retry_backoff(Duration::from_secs(1), Some(Duration::from_secs(5)));
    let waits: Vec<_> = (0..5).map(|n| client.backoff_after(n).as_secs()).collect();
    assert_eq!(waits, [1, 2, 4, 5, 5]);
    assert_eq!(
        client.backoff_after(i32::MAX),
        Duration::from_secs(5),
        "no overflow"
    );
}

#[test]
fn approval_policies_map_to_needs_approval() {
    let check = |policy: &RequireApproval, name: &str| match policy.for_tool(name) {
        NeedsResolution::Fixed(value) => value,
        NeedsResolution::Dynamic(_) => panic!("fixed expected"),
    };
    assert!(!check(&RequireApproval::Never, "a"));
    assert!(check(&RequireApproval::Always, "a"));
    let per_tool = RequireApproval::PerTool {
        always: vec!["rm".into()],
        never: vec!["ls".into()],
    };
    assert!(check(&per_tool, "rm"));
    assert!(!check(&per_tool, "ls"));
    assert!(
        !check(&per_tool, "other"),
        "unlisted tools need no approval"
    );
}

async fn invoke(tool: &FunctionTool, arguments: &str) -> Result<Value, AgentsError> {
    let context = ToolContext::new(
        tool.name.clone(),
        "c1",
        arguments,
        RunContextWrapper::new(None),
    );
    (tool.on_invoke_tool)(context, arguments.to_string())
        .await
        .map(|r| r.output.unwrap())
}

async fn convert(client: McpClient, config: McpConfig) -> Result<Vec<FunctionTool>, AgentsError> {
    let server: Arc<dyn McpServer> = client.shared();
    mcp_function_tools(
        &[server],
        &config,
        &RunContextWrapper::new(None),
        &Agent::new("a"),
    )
    .await
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
    assert_eq!(
        bare.params_json_schema["properties"],
        json!({}),
        "`properties` is added"
    );
    assert!(matches!(a.needs_resolution, NeedsResolution::Fixed(true)));

    assert_eq!(invoke(a, r#"{"x": 1}"#).await.unwrap(), json!("{\"x\":1}"));
    let missing = invoke(a, "{}").await.unwrap_err();
    assert!(
        missing
            .to_string()
            .contains("missing required parameters: x"),
        "{missing}"
    );
    assert!(invoke(a, "not json").await.is_err());
    assert!(
        invoke(a, "[1]").await.is_err(),
        "arguments must be an object"
    );
    assert_eq!(
        invoke(bare, "").await.unwrap(),
        json!("{}"),
        "empty input means no arguments"
    );
    let calls = transport
        .requests
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|(m, _)| m == "tools/call")
        .count();
    assert_eq!(calls, 2, "invalid calls never reach the server");
}

#[tokio::test]
async fn strict_conversion_is_opt_in() {
    let page = || vec![json!({"tools": [tool_json("a")]})];
    let strict = McpConfig {
        convert_schemas_to_strict: true,
    };
    let tools = convert(fake(page()).0, strict).await.unwrap();
    assert!(tools[0].strict_json_schema);
    assert_eq!(tools[0].params_json_schema["additionalProperties"], false);
    let tools = convert(fake(page()).0, McpConfig::default()).await.unwrap();
    assert!(!tools[0].strict_json_schema);
}

#[tokio::test]
async fn duplicate_names_across_servers_are_rejected() {
    let one: Arc<dyn McpServer> = fake(vec![json!({"tools": [tool_json("dup"), tool_json("x")]})])
        .0
        .shared();
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
        error
            .to_string()
            .contains("Duplicate tool names found across MCP servers: \"dup\""),
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
    let failed = McpCallToolResult {
        is_error: true,
        ..result
    };
    assert!(
        render_tool_result(&failed, true).starts_with("one"),
        "errors keep their content"
    );
}
