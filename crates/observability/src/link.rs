//! Send a run's spans and conversation to a console process (Rust-only, X-018).
//!
//! The console is a separate program (`opai-console`) that takes newline-delimited JSON on
//! `POST /ingest` and draws it live. [`ConsoleLink`] is the client side: it installs nothing by
//! itself, it hands back a trace processor and attaches to a [`ConversationHub`], and both write to
//! the console from threads of their own, so a slow or absent console costs the run nothing.
//!
//! ```rust,no_run
//! use opai_agents_observability::{ConsoleLink, ConversationHub, LinkConfig};
//!
//! let hub = ConversationHub::new();
//! let link = ConsoleLink::start(LinkConfig::new("127.0.0.1:7077"), &hub).expect("address");
//! // opai_agents::tracing::set_trace_processors(vec![link.processor()]);
//! // Per run: RunOptions::new().workflow_name("run-1").hooks(hub.recorder("run-1"))
//! link.flush();
//! ```
//!
//! The console groups spans by the name of the run's root `task` span, which is the workflow name,
//! and conversation events by their `run_id`. Use the same string for both.
//!
//! The transport is plain HTTP/1.1 over a TCP socket from the standard library: no TLS, no
//! redirects, one short connection per batch. It is meant for a console on the same machine or a
//! trusted network; for anything else put it behind a tunnel or a reverse proxy. The address is
//! configuration the application chooses, never something taken from a run's input.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::time::Duration;

use opai_agents_core::tracing::TracingProcessor;

use super::batched::{BatchedProcessor, EventSink};
use super::conversation::{ConversationEvent, ConversationHub, QueuedConversationSink};
use super::export::ExportEvent;

/// Longest body of one request. The console accepts more; staying well under keeps a batch of large
/// events from becoming one enormous request.
const MAX_BODY_BYTES: usize = 1024 * 1024;

/// How long to wait to connect, and for each read and write.
const IO_TIMEOUT: Duration = Duration::from_secs(2);

/// Where the console is and how to talk to it.
#[derive(Debug, Clone)]
pub struct LinkConfig {
    addr: String,
    token: Option<String>,
    capacity: usize,
    interval: Duration,
}

impl LinkConfig {
    /// A console at `host:port`, with a 1024-event queue and a 200 ms flush interval.
    pub fn new(addr: impl Into<String>) -> Self {
        Self {
            addr: addr.into(),
            token: None,
            capacity: 1024,
            interval: Duration::from_millis(200),
        }
    }

    /// The bearer token the console expects. Read it from the environment, not from source.
    #[must_use]
    pub fn token(mut self, token: Option<String>) -> Self {
        self.token = token.filter(|t| !t.is_empty());
        self
    }

    /// How many events may wait for each of the two writers, and how long a lone event may sit.
    #[must_use]
    pub fn queue(mut self, capacity: usize, interval: Duration) -> Self {
        self.capacity = capacity;
        self.interval = interval;
        self
    }

    fn validate(&self) -> Result<(), String> {
        let (host, port) = self
            .addr
            .rsplit_once(':')
            .ok_or_else(|| format!("console address `{}` must be host:port", self.addr))?;
        if host.is_empty() || port.parse::<u16>().is_err() {
            return Err(format!("console address `{}` must be host:port", self.addr));
        }
        // Both end up in request headers: a line break would let a value add headers of its own.
        let unsafe_char = |s: &str| s.chars().any(|c| c.is_control() || c.is_whitespace());
        if unsafe_char(&self.addr) {
            return Err("console address must not contain whitespace".into());
        }
        if self.token.as_deref().is_some_and(unsafe_char) {
            return Err("console token must not contain whitespace".into());
        }
        Ok(())
    }
}

/// A running connection to a console: two background writers and what is needed to stop waiting
/// for them.
pub struct ConsoleLink {
    processor: Arc<BatchedProcessor>,
    conversation: Arc<QueuedConversationSink>,
}

impl std::fmt::Debug for ConsoleLink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsoleLink")
            .field("dropped", &self.dropped())
            .finish()
    }
}

impl ConsoleLink {
    /// Start sending to the console in `config`, and attach to `hub` so its recorders' events
    /// follow. Does not need the console to be up: a batch it cannot deliver is counted as dropped.
    ///
    /// Errors when the address or token is malformed.
    pub fn start(config: LinkConfig, hub: &ConversationHub) -> Result<Self, String> {
        config.validate()?;
        let target = Arc::new(Target {
            addr: config.addr.clone(),
            token: config.token.clone(),
        });

        let processor = BatchedProcessor::spawn(
            TraceWriter(Arc::clone(&target)),
            config.capacity,
            config.interval,
        );
        let writer = Arc::clone(&target);
        let conversation = QueuedConversationSink::spawn(
            "console",
            move |events: &[ConversationEvent]| {
                let lines: Vec<String> = events.iter().map(|e| e.to_json().to_string()).collect();
                writer.post(&lines)
            },
            config.capacity,
            config.interval,
        );
        hub.attach(conversation.clone());
        Ok(Self {
            processor,
            conversation,
        })
    }

    /// The trace processor to install with `set_trace_processors` or `add_trace_processor`.
    pub fn processor(&self) -> Arc<dyn TracingProcessor> {
        self.processor.clone()
    }

    /// Events lost, spans and conversation together.
    pub fn dropped(&self) -> u64 {
        self.processor.dropped() + self.conversation.dropped()
    }

    /// Write what is queued, waiting a little for the writers. Call before the process ends.
    pub fn flush(&self) {
        self.processor.force_flush();
        self.conversation.flush();
    }
}

/// Where a request goes.
struct Target {
    addr: String,
    token: Option<String>,
}

impl Target {
    /// Send `lines` as NDJSON, in requests of at most [`MAX_BODY_BYTES`].
    fn post(&self, lines: &[String]) -> Result<(), String> {
        for body in chunks(lines, MAX_BODY_BYTES)? {
            self.request(&body)?;
        }
        Ok(())
    }

    fn request(&self, body: &str) -> Result<(), String> {
        let socket = self
            .addr
            .to_socket_addrs()
            .map_err(|e| format!("cannot resolve {}: {e}", self.addr))?
            .find_map(|a| TcpStream::connect_timeout(&a, IO_TIMEOUT).ok())
            .ok_or_else(|| format!("cannot connect to the console at {}", self.addr))?;
        socket
            .set_read_timeout(Some(IO_TIMEOUT))
            .and_then(|()| socket.set_write_timeout(Some(IO_TIMEOUT)))
            .map_err(|e| format!("cannot set a timeout: {e}"))?;

        let auth = self
            .token
            .as_ref()
            .map(|t| format!("Authorization: Bearer {t}\r\n"))
            .unwrap_or_default();
        let head = format!(
            "POST /ingest HTTP/1.1\r\nHost: {}\r\nContent-Type: application/x-ndjson\r\n\
             Content-Length: {}\r\nConnection: close\r\n{auth}\r\n",
            self.addr,
            body.len()
        );
        let mut socket = socket;
        socket
            .write_all(head.as_bytes())
            .and_then(|()| socket.write_all(body.as_bytes()))
            .map_err(|e| format!("cannot send to the console: {e}"))?;

        // The status line may arrive in pieces; read until it is complete.
        let mut reply = Vec::with_capacity(64);
        let mut chunk = [0u8; 64];
        while !reply.windows(2).any(|w| w == b"\r\n") && reply.len() < 256 {
            match socket.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => reply.extend_from_slice(&chunk[..n]),
                Err(e) => return Err(format!("no reply from the console: {e}")),
            }
        }
        let status = std::str::from_utf8(&reply)
            .ok()
            .and_then(|s| s.split_whitespace().nth(1))
            .and_then(|s| s.parse::<u16>().ok())
            .ok_or("the console sent something that is not HTTP")?;
        if (200..300).contains(&status) {
            Ok(())
        } else {
            Err(format!("the console refused the batch: HTTP {status}"))
        }
    }
}

/// Group lines into bodies of at most `max` bytes. A line that is alone over the limit is an error
/// for the whole batch, which is then counted as lost.
fn chunks(lines: &[String], max: usize) -> Result<Vec<String>, String> {
    let mut bodies = Vec::new();
    let mut current = String::new();
    for line in lines {
        if line.len() + 1 > max {
            return Err(format!(
                "an event of {} bytes is over the limit",
                line.len()
            ));
        }
        if current.len() + line.len() + 1 > max {
            bodies.push(std::mem::take(&mut current));
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.is_empty() {
        bodies.push(current);
    }
    Ok(bodies)
}

/// Spans and trace events, for [`BatchedProcessor`].
struct TraceWriter(Arc<Target>);

impl EventSink for TraceWriter {
    fn write(&mut self, events: &[ExportEvent]) -> Result<(), String> {
        let lines: Vec<String> = events.iter().map(|e| e.to_json().to_string()).collect();
        self.0.post(&lines)
    }

    fn name(&self) -> &'static str {
        "console"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_must_be_host_and_port_without_whitespace() {
        let ok = |a: &str| LinkConfig::new(a).validate().is_ok();
        assert!(ok("127.0.0.1:7077"));
        assert!(ok("localhost:80"));
        assert!(!ok("localhost"));
        assert!(!ok(":7077"));
        assert!(!ok("host:notaport"));
        assert!(!ok("host:80\r\nX-Evil: 1"));
        assert!(LinkConfig::new("h:1")
            .token(Some("a\r\nb".into()))
            .validate()
            .is_err());
    }

    #[test]
    fn an_empty_token_means_no_token() {
        assert!(LinkConfig::new("h:1")
            .token(Some(String::new()))
            .token
            .is_none());
    }

    #[test]
    fn lines_are_grouped_under_the_limit_and_an_oversized_line_is_an_error() {
        let lines: Vec<String> = (0..5).map(|_| "x".repeat(9)).collect();
        let bodies = chunks(&lines, 25).unwrap();
        assert_eq!(bodies.len(), 3);
        assert!(bodies.iter().all(|b| b.len() <= 25 && b.ends_with('\n')));
        assert_eq!(bodies.concat().lines().count(), 5);
        assert!(chunks(&["y".repeat(30)], 25).is_err());
    }
}
