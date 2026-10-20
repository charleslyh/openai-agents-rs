//! The link sends spans and conversation to a console, and survives one that is not there.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, SystemTime};

use opai_agents_core::tracing::{Span, SpanData, Trace};
use opai_agents_core::{Agent, RunContextWrapper, RunHooks};
use opai_agents_observability::{ConsoleLink, ConversationHub, LinkConfig};
use serde_json::Value;

/// What a request carried.
struct Request {
    head: String,
    body: String,
}

/// A console that answers every request with `status` and reports what it received.
fn fake_console(status: &'static str) -> (String, Receiver<Request>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr").to_string();
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            if let Some(request) = read_request(&mut stream) {
                let _ = tx.send(request);
            }
            let _ = write!(stream, "HTTP/1.1 {status}\r\nContent-Length: 0\r\n\r\n");
        }
    });
    (addr, rx)
}

fn read_request(stream: &mut TcpStream) -> Option<Request> {
    stream.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
    let mut data = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        data.extend_from_slice(&chunk[..n]);
        let text = String::from_utf8_lossy(&data).into_owned();
        if let Some((head, body)) = text.split_once("\r\n\r\n") {
            let length = head
                .lines()
                .find_map(|l| l.strip_prefix("Content-Length: "))
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
            if body.len() >= length {
                return Some(Request {
                    head: head.to_string(),
                    body: body.to_string(),
                });
            }
        }
    }
}

fn span() -> Span {
    Span {
        span_id: "span_1".into(),
        parent_id: None,
        data: SpanData::Task {
            name: "run-1".into(),
            usage: None,
        },
        started_at: SystemTime::now(),
        ended_at: None,
        error: None,
    }
}

fn lines(request: &Request) -> Vec<Value> {
    request
        .body
        .lines()
        .map(|l| serde_json::from_str(l).expect("ndjson"))
        .collect()
}

#[tokio::test]
async fn spans_and_conversation_arrive_as_ndjson_with_the_token() {
    let (addr, received) = fake_console("204 No Content");
    let hub = ConversationHub::new();
    let link = ConsoleLink::start(LinkConfig::new(addr).token(Some("s3cret".into())), &hub)
        .expect("start");

    let processor = link.processor();
    processor.on_trace_start(&Trace {
        trace_id: "t".into(),
        workflow_name: "run-1".into(),
        group_id: None,
        metadata: None,
    });
    processor.on_span_start(&span());
    hub.recorder("run-1")
        .on_agent_start(RunContextWrapper::default(), &Agent::new("Ops"))
        .await;
    link.flush();

    let mut kinds = Vec::new();
    while let Ok(request) = received.recv_timeout(Duration::from_millis(500)) {
        assert!(
            request.head.starts_with("POST /ingest HTTP/1.1"),
            "{}",
            request.head
        );
        assert!(
            request.head.contains("Authorization: Bearer s3cret"),
            "{}",
            request.head
        );
        assert!(
            request.head.contains("application/x-ndjson"),
            "{}",
            request.head
        );
        for line in lines(&request) {
            kinds.push(line["kind"].as_str().unwrap().to_string());
            if line["kind"] == "span_start" {
                assert_eq!(line["name"], "run-1");
            }
            if line["kind"] == "agent_start" {
                assert_eq!(line["run_id"], "run-1");
                assert_eq!(line["agent"], "Ops");
            }
        }
    }
    kinds.sort();
    assert_eq!(kinds, ["agent_start", "span_start", "trace_start"]);
    assert_eq!(link.dropped(), 0);
}

#[test]
fn a_console_that_is_not_there_costs_events_not_the_run() {
    // Bind and drop to get a port nothing listens on.
    let addr = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().to_string()
    };
    let hub = ConversationHub::new();
    let link = ConsoleLink::start(LinkConfig::new(addr), &hub).expect("start");
    let processor = link.processor();

    let started = std::time::Instant::now();
    processor.on_span_start(&span());
    assert!(
        started.elapsed() < Duration::from_millis(100),
        "the callback must not wait"
    );
    link.flush();

    assert_eq!(link.dropped(), 1);
}

#[test]
fn a_console_that_refuses_the_batch_is_counted() {
    let (addr, _received) = fake_console("401 Unauthorized");
    let hub = ConversationHub::new();
    let link = ConsoleLink::start(LinkConfig::new(addr), &hub).expect("start");
    link.processor().on_span_start(&span());
    link.flush();
    assert_eq!(link.dropped(), 1);
}

#[test]
fn a_bad_address_is_refused_up_front() {
    let hub = ConversationHub::new();
    assert!(ConsoleLink::start(LinkConfig::new("no-port"), &hub).is_err());
}
