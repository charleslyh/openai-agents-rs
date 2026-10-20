//! The console, driven the way an application drives it: a real agent run sent with
//! `ConsoleLink`, and the HTTP surface a browser uses.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::{Duration, Instant};

use opai_agents::observability::{ConsoleLink, ConversationHub, LinkConfig};
use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::tracing::set_trace_processors;
use opai_agents::{Agent, FunctionTool, RunOptions, Runner};
use opai_agents_console::server::{router, AppState};
use opai_agents_console::store::{Store, StoreLimits, UNASSIGNED};
use serde_json::{json, Value};

struct Console {
    addr: String,
    store: Arc<Store>,
}

async fn start(token: Option<&str>) -> Console {
    let store = Arc::new(Store::new(StoreLimits::default()));
    let app = AppState::new(Arc::clone(&store), token.map(str::to_owned));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr").to_string();
    let hosts = vec![addr.clone()];
    tokio::spawn(async move {
        let _ = axum::serve(listener, router(app, Some(hosts))).await;
    });
    Console { addr, store }
}

struct Reply {
    status: u16,
    head: String,
    body: String,
}

/// One blocking HTTP/1.1 request, answered in full.
fn http(addr: &str, method: &str, path: &str, headers: &[(&str, &str)], body: &str) -> Reply {
    let mut stream = TcpStream::connect(addr).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .expect("timeout");
    let mut request = format!("{method} {path} HTTP/1.1\r\nConnection: close\r\n");
    if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("host")) {
        request.push_str(&format!("Host: {addr}\r\n"));
    }
    for (k, v) in headers {
        request.push_str(&format!("{k}: {v}\r\n"));
    }
    request.push_str(&format!("Content-Length: {}\r\n\r\n{body}", body.len()));
    stream.write_all(request.as_bytes()).expect("send");
    let mut raw = String::new();
    let _ = stream.read_to_string(&mut raw);
    let (head, body) = raw.split_once("\r\n\r\n").unwrap_or((&raw, ""));
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    Reply {
        status,
        head: head.to_string(),
        body: body.to_string(),
    }
}

async fn call(
    addr: &str,
    method: &'static str,
    path: &'static str,
    headers: Vec<(&'static str, String)>,
    body: String,
) -> Reply {
    let addr = addr.to_string();
    tokio::task::spawn_blocking(move || {
        let headers: Vec<(&str, &str)> = headers.iter().map(|(k, v)| (*k, v.as_str())).collect();
        http(&addr, method, path, &headers, &body)
    })
    .await
    .expect("join")
}

const NDJSON: (&str, &str) = ("Content-Type", "application/x-ndjson");

fn lookup_tool() -> FunctionTool {
    FunctionTool::new(
        "lookup",
        "look a city up",
        json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false
        }),
        |_ctx, _args| async move { Ok(json!("1200 orders")) },
    )
}

#[tokio::test]
async fn a_real_run_arrives_as_one_run_with_its_spans_and_conversation() {
    let console = start(Some("s3cret")).await;
    let hub = ConversationHub::new();
    let link = ConsoleLink::start(
        LinkConfig::new(console.addr.clone()).token(Some("s3cret".into())),
        &hub,
    )
    .expect("link");
    set_trace_processors(vec![link.processor()]);

    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(
            "lookup",
            r#"{"city":"Berlin"}"#,
            "c1",
        )),
        ModelStep::from(ItemHelpers::text_message("Berlin: 1200 orders")),
    ]));
    let agent = Agent::new("Ops")
        .instructions("Use lookup.")
        .model(model)
        .tools(vec![lookup_tool()]);
    let options = RunOptions::new()
        .workflow_name("run-e2e")
        .hooks(hub.recorder("run-e2e"));
    Runner::run(&agent, "metrics for Berlin?", options)
        .await
        .expect("run");
    let flush = tokio::task::spawn_blocking(move || {
        link.flush();
        link.dropped()
    });
    assert_eq!(flush.await.expect("flush"), 0, "nothing may be lost");

    let view = console.store.run("run-e2e").expect("the run");
    let items = view["items"].as_array().expect("items");
    let kinds: Vec<&str> = items.iter().filter_map(|i| i["kind"].as_str()).collect();
    for expected in [
        "span_start",
        "span_end",
        "agent_start",
        "llm_start",
        "llm_end",
        "tool_start",
        "tool_end",
        "agent_end",
    ] {
        assert!(kinds.contains(&expected), "missing {expected}: {kinds:?}");
    }
    let span_kinds: Vec<&str> = items
        .iter()
        .filter_map(|i| i["span_kind"].as_str())
        .collect();
    for expected in ["task", "agent", "turn", "function"] {
        assert!(
            span_kinds.contains(&expected),
            "missing {expected}: {span_kinds:?}"
        );
    }

    // Every span found its run through the root task, and conversation named its own.
    assert!(console.store.run(UNASSIGNED).is_none());
    assert!(items.iter().all(|i| i["run_id"] == "run-e2e"));

    let summary = &console.store.runs()[0];
    assert_eq!(summary["status"], "done");
    assert!(summary["turns"].as_u64().unwrap() >= 2);
    assert_eq!(summary["preview"], "metrics for Berlin?");

    // The content is what the model and the tool saw.
    let call = items.iter().find(|i| i["kind"] == "tool_start").unwrap();
    assert_eq!(call["data"]["arguments"]["city"], "Berlin");
    let result = items.iter().find(|i| i["kind"] == "tool_end").unwrap();
    assert_eq!(result["data"]["result"], "1200 orders");
}

#[tokio::test]
async fn ingest_needs_the_token_and_a_content_type_a_web_page_cannot_send() {
    let console = start(Some("s3cret")).await;
    let line = r#"{"kind":"agent_start","run_id":"r"}"#.to_string();
    let auth = ("Authorization", "Bearer s3cret".to_string());
    let ndjson = (NDJSON.0, NDJSON.1.to_string());

    let none = call(
        &console.addr,
        "POST",
        "/ingest",
        vec![ndjson.clone()],
        line.clone(),
    )
    .await;
    assert_eq!(none.status, 401, "{}", none.body);

    let wrong = call(
        &console.addr,
        "POST",
        "/ingest",
        vec![ndjson.clone(), ("Authorization", "Bearer nope".into())],
        line.clone(),
    )
    .await;
    assert_eq!(wrong.status, 401);

    let simple = call(
        &console.addr,
        "POST",
        "/ingest",
        vec![auth.clone(), ("Content-Type", "text/plain".into())],
        line.clone(),
    )
    .await;
    assert_eq!(simple.status, 415);
    assert!(console.store.runs().is_empty(), "nothing was stored yet");

    let ok = call(
        &console.addr,
        "POST",
        "/ingest",
        vec![auth, ndjson],
        format!("{line}\nnot json\n"),
    )
    .await;
    assert_eq!(ok.status, 200, "{}", ok.body);
    let counts: Value = serde_json::from_str(&ok.body).expect("json");
    assert_eq!(counts, json!({"accepted": 1, "rejected": 1}));
    assert_eq!(console.store.runs().len(), 1);
}

#[tokio::test]
async fn a_request_for_another_host_is_refused() {
    let console = start(None).await;
    let reply = call(
        &console.addr,
        "GET",
        "/api/runs",
        vec![("Host", "evil.example".into())],
        String::new(),
    )
    .await;
    assert_eq!(reply.status, 403);
}

#[tokio::test]
async fn the_page_is_locked_down_and_the_api_lists_runs() {
    let console = start(None).await;
    console
        .store
        .ingest(json!({"kind": "agent_start", "run_id": "r", "agent": "Ops"}));

    let page = call(&console.addr, "GET", "/", vec![], String::new()).await;
    assert_eq!(page.status, 200);
    let csp = page
        .head
        .lines()
        .find(|l| {
            l.to_ascii_lowercase()
                .starts_with("content-security-policy:")
        })
        .expect("a content security policy");
    assert!(
        csp.contains("default-src 'none'") && csp.contains("script-src 'nonce-"),
        "{csp}"
    );
    assert!(!page.body.contains("__NONCE__"), "the nonce is filled in");
    let nonce = csp
        .split("script-src 'nonce-")
        .nth(1)
        .unwrap()
        .split('\'')
        .next()
        .unwrap();
    assert!(page.body.contains(&format!("nonce=\"{nonce}\"")));

    let runs = call(&console.addr, "GET", "/api/runs", vec![], String::new()).await;
    let runs: Value = serde_json::from_str(&runs.body).expect("json");
    assert_eq!(runs["runs"][0]["id"], "r");

    let one = call(&console.addr, "GET", "/api/runs/r", vec![], String::new()).await;
    assert_eq!(one.status, 200);
    let missing = call(
        &console.addr,
        "GET",
        "/api/runs/nope",
        vec![],
        String::new(),
    )
    .await;
    assert_eq!(missing.status, 404);
}

#[tokio::test]
async fn a_watching_browser_is_sent_what_arrives() {
    let console = start(None).await;
    let addr = console.addr.clone();
    let watcher = tokio::task::spawn_blocking(move || {
        let mut stream = TcpStream::connect(&addr).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_millis(300)))
            .expect("timeout");
        write!(
            stream,
            "GET /api/stream HTTP/1.1\r\nHost: {addr}\r\nAccept: text/event-stream\r\n\r\n"
        )
        .expect("send");
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut seen = String::new();
        let mut chunk = [0u8; 4096];
        while Instant::now() < deadline && !seen.contains("\"run_id\":\"watched\"") {
            match stream.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => seen.push_str(&String::from_utf8_lossy(&chunk[..n])),
                Err(_) => {}
            }
        }
        seen
    });

    // Give the watcher time to subscribe, then deliver.
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && !watcher.is_finished() {
        tokio::time::sleep(Duration::from_millis(100)).await;
        console
            .store
            .ingest(json!({"kind": "agent_start", "run_id": "watched"}));
    }
    let seen = watcher.await.expect("join");
    assert!(seen.contains("text/event-stream"), "{seen}");
    assert!(seen.contains("data: "), "{seen}");
    assert!(seen.contains("\"run_id\":\"watched\""), "{seen}");
}
