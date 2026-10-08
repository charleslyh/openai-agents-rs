//! The MCP client against a real server: the official Python `mcp` package, run from the oracle
//! venv (`scripts/setup_venv.sh`). Tests skip themselves when the venv is missing.
//!
//! Two independent implementations talking to each other is the point: it checks the wire format
//! (framing, handshake, SSE, sessions), not just our reading of the spec.
#![cfg(feature = "mcp")]

use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    mcp_function_tools, Agent, AgentsError, McpClient, McpConfig, McpServer, RequireApproval,
    RunContextWrapper, RunOptions, Runner, StdioParams, StreamableHttpParams, ToolFilter,
};
use serde_json::{json, Value};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn python() -> Option<PathBuf> {
    let python = root().join(".venv/bin/python");
    if python.exists() {
        Some(python)
    } else {
        eprintln!("skipping: no .venv (run scripts/setup_venv.sh)");
        None
    }
}

fn server_script() -> PathBuf {
    root().join("tests/mcp/server.py")
}

fn stdio_client(python: &PathBuf) -> McpClient {
    McpClient::stdio(
        StdioParams::new(python.display().to_string()).args([server_script().display().to_string(), "stdio".into()]),
    )
}

fn context() -> RunContextWrapper {
    RunContextWrapper::new(None)
}

async fn tool_names(server: &dyn McpServer) -> Vec<String> {
    server
        .list_tools(&context(), &Agent::new("a"))
        .await
        .expect("list tools")
        .into_iter()
        .map(|t| t.name)
        .collect()
}

/// An HTTP MCP server process that is killed when dropped.
struct HttpServer {
    child: Child,
    port: u16,
}

impl HttpServer {
    fn start(python: &PathBuf, json_replies: bool) -> Self {
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let mut command = Command::new(python);
        command
            .arg(server_script())
            .args(["http", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if json_replies {
            command.arg("json");
        }
        let child = command.spawn().expect("start server");
        let deadline = Instant::now() + Duration::from_secs(20);
        while TcpStream::connect(("127.0.0.1", port)).is_err() {
            assert!(Instant::now() < deadline, "the server did not come up");
            std::thread::sleep(Duration::from_millis(100));
        }
        Self { child, port }
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}/mcp", self.port)
    }
}

impl Drop for HttpServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[tokio::test]
async fn stdio_lists_tools_and_calls_them() {
    let Some(python) = python() else { return };
    let client = stdio_client(&python);
    let names = tool_names(&client).await;
    for expected in ["add", "echo", "greet", "two_parts", "picture", "boom", "env_var", "slow"] {
        assert!(names.contains(&expected.to_string()), "{names:?}");
    }

    let sum = client.call_tool("add", Some(json!({"a": 2, "b": 3}))).await.unwrap();
    assert!(!sum.is_error);
    assert_eq!(sum.content[0]["text"], "5");
    assert_eq!(sum.structured_content, Some(json!({"result": 5})));
    assert_eq!(openai_agents::render_tool_result(&sum, false), "5");
    assert_eq!(openai_agents::render_tool_result(&sum, true), "{\"result\":5}");

    let parts = client.call_tool("two_parts", None).await.unwrap();
    assert_eq!(openai_agents::render_tool_result(&parts, false), "first\nsecond");
    let picture = client.call_tool("picture", None).await.unwrap();
    assert_eq!(openai_agents::render_tool_result(&picture, false), "[image: image/png]\na caption");

    // A tool that raises comes back as an error result (or an error), never as a silent success.
    match client.call_tool("boom", None).await {
        Ok(result) => {
            assert!(result.is_error, "{result:?}");
            assert!(openai_agents::render_tool_result(&result, false).contains("boom"));
        }
        Err(error) => assert!(error.to_string().contains("boom"), "{error}"),
    }
    // The connection is still good afterwards.
    assert!(client.call_tool("echo", Some(json!({"text": "x"}))).await.is_ok());
    client.cleanup().await.unwrap();
}

#[tokio::test]
async fn function_tools_match_what_the_python_sdk_builds() {
    let Some(python) = python() else { return };
    let output = Command::new(&python)
        .arg(root().join("scripts/mcp_oracle.py"))
        .arg(&python)
        .arg(server_script())
        .output()
        .expect("python");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let expected: Vec<Value> = serde_json::from_str(stdout.lines().last().unwrap()).expect("oracle json");

    let client: Arc<dyn McpServer> = stdio_client(&python).shared();
    let tools = mcp_function_tools(
        std::slice::from_ref(&client),
        &McpConfig::default(),
        &context(),
        &Agent::new("oracle"),
    )
    .await
    .unwrap();
    let actual: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "params_json_schema": t.params_json_schema,
                "strict": t.strict_json_schema,
            })
        })
        .collect();
    assert_eq!(actual, expected);
    client.cleanup().await.unwrap();
}

fn call(name: &str, arguments: &str, id: &str) -> ModelStep {
    ModelStep::from(ItemHelpers::function_tool_call(name, arguments, id))
}

#[tokio::test]
async fn an_agent_calls_mcp_tools_in_a_run() {
    let Some(python) = python() else { return };
    let model = Arc::new(ScriptedModel::new([
        call("add", r#"{"a": 2, "b": 3}"#, "c1"),
        ModelStep::from(ItemHelpers::text_message("the sum is 5")),
    ]));
    let client = stdio_client(&python).shared();
    let agent = Agent::new("calc").model(model.clone()).mcp_servers(vec![client.clone()]);
    let result = Runner::run(&agent, "2+3?", RunOptions::default()).await.expect("run");
    assert_eq!(result.final_output_as_str(), Some("the sum is 5"));

    let calls = model.calls();
    assert!(calls[0].tool_names.contains(&"add".to_string()), "{:?}", calls[0].tool_names);
    let second = calls[1].input.to_string();
    assert!(second.contains("function_call_output") && second.contains("\"output\":\"5\""), "{second}");
    client.cleanup().await.unwrap();
}

/// A failing tool, a call with a missing argument and a call with broken JSON are all reported to
/// the model, which then recovers; none of them aborts the run.
#[tokio::test]
async fn failing_mcp_tool_calls_do_not_abort_the_run() {
    let Some(python) = python() else { return };
    let model = Arc::new(ScriptedModel::new([
        call("boom", "{}", "c1"),
        call("add", r#"{"a": 1}"#, "c2"),
        call("add", "not json", "c3"),
        ModelStep::from(ItemHelpers::text_message("recovered")),
    ]));
    let client = stdio_client(&python).shared();
    let agent = Agent::new("calc").model(model.clone()).mcp_servers(vec![client.clone()]);
    let result = Runner::run(&agent, "go", RunOptions::default()).await.expect("run");
    assert_eq!(result.final_output_as_str(), Some("recovered"));
    client.cleanup().await.unwrap();
}

#[tokio::test]
async fn filters_and_approval_apply_to_agent_tools() {
    let Some(python) = python() else { return };
    let client = stdio_client(&python)
        .tool_filter(ToolFilter::allow(["add", "echo"]))
        .require_approval(RequireApproval::PerTool { always: vec!["add".into()], never: vec![] })
        .shared();
    assert_eq!(tool_names(client.as_ref()).await, ["add", "echo"]);

    let model = Arc::new(ScriptedModel::new([call("add", r#"{"a": 2, "b": 3}"#, "c1")]));
    let agent = Agent::new("calc").model(model).mcp_servers(vec![client.clone()]);
    let result = Runner::run(&agent, "go", RunOptions::default()).await.expect("run");
    assert_eq!(result.interruptions.len(), 1, "the call waits for approval");
    client.cleanup().await.unwrap();
}

#[tokio::test]
async fn a_slow_call_times_out_and_the_connection_stays_usable() {
    let Some(python) = python() else { return };
    let client = stdio_client(&python).request_timeout(Duration::from_millis(800));
    client.connect().await.unwrap();
    let error = client.call_tool("slow", Some(json!({"seconds": 3}))).await.unwrap_err();
    assert!(matches!(error, AgentsError::Tool { .. }) && error.to_string().contains("timed out"), "{error}");
    let fine = client.call_tool("echo", Some(json!({"text": "still here"}))).await.unwrap();
    assert_eq!(fine.content[0]["text"], "echo: still here");
    client.cleanup().await.unwrap();
}

/// A spawned server sees only a safe subset of this process's environment.
#[tokio::test]
async fn stdio_servers_do_not_inherit_the_environment() {
    let Some(python) = python() else { return };
    std::env::set_var("MCP_TEST_PARENT_SECRET", "leak");
    let plain = stdio_client(&python);
    let hidden = plain.call_tool("env_var", Some(json!({"name": "MCP_TEST_PARENT_SECRET"}))).await.unwrap();
    assert_eq!(hidden.content[0]["text"], "<unset>");
    let path = plain.call_tool("env_var", Some(json!({"name": "PATH"}))).await.unwrap();
    assert_ne!(path.content[0]["text"], "<unset>", "PATH is passed on");
    plain.cleanup().await.unwrap();

    let explicit = McpClient::stdio(
        StdioParams::new(python.display().to_string())
            .args([server_script().display().to_string(), "stdio".into()])
            .env("MCP_TEST_EXPLICIT", "given"),
    );
    let given = explicit.call_tool("env_var", Some(json!({"name": "MCP_TEST_EXPLICIT"}))).await.unwrap();
    assert_eq!(given.content[0]["text"], "given");
    explicit.cleanup().await.unwrap();
}

/// Streamable HTTP, in both reply styles a server may use: an event stream per request (the
/// default of the Python server) and plain JSON.
#[tokio::test]
async fn streamable_http_works_with_event_streams_and_plain_json() {
    let Some(python) = python() else { return };
    for json_replies in [false, true] {
        let server = HttpServer::start(&python, json_replies);
        let client = McpClient::streamable_http(StreamableHttpParams::new(server.url()));
        let names = tool_names(&client).await;
        assert!(names.contains(&"add".to_string()), "json_replies={json_replies}: {names:?}");
        let sum = client.call_tool("add", Some(json!({"a": 40, "b": 2}))).await.unwrap();
        assert_eq!(sum.content[0]["text"], "42", "json_replies={json_replies}");
        // Several calls reuse one session.
        for n in 0..3 {
            let reply = client.call_tool("echo", Some(json!({"text": n.to_string()}))).await.unwrap();
            assert_eq!(reply.content[0]["text"], format!("echo: {n}"));
        }
        client.cleanup().await.unwrap();
    }
}

#[tokio::test]
async fn an_agent_uses_tools_from_an_http_server() {
    let Some(python) = python() else { return };
    let server = HttpServer::start(&python, false);
    let client = McpClient::streamable_http(StreamableHttpParams::new(server.url())).shared();
    let model = Arc::new(ScriptedModel::new([
        call("greet", r#"{"name": "Ada", "excited": true}"#, "c1"),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("a").model(model.clone()).mcp_servers(vec![client.clone()]);
    Runner::run(&agent, "hi", RunOptions::default()).await.expect("run");
    assert!(model.calls()[1].input.to_string().contains("Hello, Ada!"));
    client.cleanup().await.unwrap();
}

#[tokio::test]
async fn connection_failures_are_clear_errors() {
    let missing = McpClient::stdio(StdioParams::new("/definitely/not/a/program"));
    let error = missing.connect().await.unwrap_err();
    assert!(error.to_string().contains("could not start"), "{error}");

    let port = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    };
    let refused = McpClient::streamable_http(StreamableHttpParams::new(format!("http://127.0.0.1:{port}/mcp")));
    let error = refused.connect().await.unwrap_err();
    assert!(error.to_string().contains("failed"), "{error}");
}
