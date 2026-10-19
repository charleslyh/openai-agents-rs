//! MCP over streamable HTTP (Python: `MCPServerStreamableHttp`).
//!
//! Every message is a `POST`. The server answers either with one JSON document or with an event
//! stream (SSE) that ends with the response. A session id the server hands out on `initialize`
//! is sent back on later requests.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
use serde_json::Value;

use super::rpc::{
    handshake, notification_message, reply_to_server_request, request_message, response_outcome,
    Connector, McpError, Transport,
};

const SESSION_HEADER: &str = "mcp-session-id";
const VERSION_HEADER: &str = "mcp-protocol-version";

/// How to reach a streamable HTTP server (Python: `MCPServerStreamableHttpParams`).
#[derive(Debug, Clone, Default)]
pub struct StreamableHttpParams {
    /// The MCP endpoint, for example `http://127.0.0.1:8000/mcp`.
    pub url: String,
    /// Extra request headers (authorization and the like).
    pub headers: HashMap<String, String>,
}

impl StreamableHttpParams {
    /// Connect to `url` without extra headers.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            headers: HashMap::new(),
        }
    }

    /// Add a request header.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }
}

/// Incremental parser of a server-sent event stream: feed bytes, take complete `data` payloads.
#[derive(Default)]
struct SseEvents {
    buffer: String,
    decoder: openai_agents_core::utf8_stream::Utf8Stream,
}

impl SseEvents {
    fn feed(&mut self, chunk: &[u8]) {
        self.buffer.push_str(&self.decoder.push(chunk));
        // A `\r` at the very end may be the first half of `\r\n` split across chunks, so it is
        // normalised only once the next byte shows what follows it.
        let held_cr = self.buffer.ends_with('\r');
        if held_cr {
            self.buffer.pop();
        }
        if self.buffer.contains('\r') {
            self.buffer = self.buffer.replace("\r\n", "\n").replace('\r', "\n");
        }
        if held_cr {
            self.buffer.push('\r');
        }
    }

    /// The `data` of the next complete event that has any.
    fn next_data(&mut self) -> Option<String> {
        while let Some(end) = self.buffer.find("\n\n") {
            let block: String = self.buffer[..end].to_string();
            self.buffer.drain(..end + 2);
            let data: Vec<&str> = block
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|d| d.strip_prefix(' ').unwrap_or(d))
                .collect();
            if !data.is_empty() {
                return Some(data.join("\n"));
            }
        }
        None
    }
}

/// How long establishing a connection may take.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Upper bound for the fire-and-forget calls that carry no caller-supplied timeout.
const AUX_TIMEOUT: Duration = Duration::from_secs(30);
/// Redirects followed per request, same origin only.
const MAX_REDIRECTS: usize = 5;

/// The shared HTTP client.
///
/// Redirects are followed only within the original origin: the configured headers are
/// credentials (API keys, bearer tokens) that must not travel to another host, and a 307/308
/// would otherwise replay the POST body there. Connecting is bounded so a dead host cannot hang
/// the caller; request bodies stay governed by the per-request timeout.
fn build_client() -> reqwest::Client {
    let redirect = reqwest::redirect::Policy::custom(|attempt| {
        let same_origin = attempt.previous().last().is_some_and(|prev| {
            prev.scheme() == attempt.url().scheme()
                && prev.host_str() == attempt.url().host_str()
                && prev.port_or_known_default() == attempt.url().port_or_known_default()
        });
        if attempt.previous().len() >= MAX_REDIRECTS {
            attempt.error("too many redirects")
        } else if same_origin {
            attempt.follow()
        } else {
            attempt.stop()
        }
    });
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .redirect(redirect)
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

pub(crate) struct HttpTransport {
    client: reqwest::Client,
    url: String,
    headers: HashMap<String, String>,
    session_id: Mutex<Option<String>>,
    protocol_version: Mutex<Option<String>>,
    next_id: AtomicU64,
}

impl HttpTransport {
    fn new(params: &StreamableHttpParams) -> Self {
        Self {
            client: build_client(),
            url: params.url.clone(),
            headers: params.headers.clone(),
            session_id: Mutex::new(None),
            protocol_version: Mutex::new(None),
            next_id: AtomicU64::new(1),
        }
    }

    fn header_map(&self) -> Result<HeaderMap, McpError> {
        let mut map = HeaderMap::new();
        let bad = |what: &str, name: &str| McpError::Transport(format!("invalid {what} `{name}`"));
        for (name, value) in &self.headers {
            map.insert(
                HeaderName::try_from(name.as_str()).map_err(|_| bad("header name", name))?,
                HeaderValue::from_str(value).map_err(|_| bad("header value for", name))?,
            );
        }
        map.insert(
            ACCEPT,
            HeaderValue::from_static("application/json, text/event-stream"),
        );
        map.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(id) = self
            .session_id
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_deref()
        {
            map.insert(
                HeaderName::from_static(SESSION_HEADER),
                HeaderValue::from_str(id).map_err(|_| bad("session id", id))?,
            );
        }
        if let Some(version) = self
            .protocol_version
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_deref()
        {
            map.insert(
                HeaderName::from_static(VERSION_HEADER),
                HeaderValue::from_str(version).map_err(|_| bad("protocol version", version))?,
            );
        }
        Ok(map)
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response, McpError> {
        let response = self
            .client
            .post(&self.url)
            .headers(self.header_map()?)
            .body(body.to_string())
            .send()
            .await
            .map_err(|e| McpError::Transport(format!("POST {} failed: {e}", self.url)))?;
        if let Some(id) = response
            .headers()
            .get(SESSION_HEADER)
            .and_then(|v| v.to_str().ok())
        {
            *self
                .session_id
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(id.to_string());
        }
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let hint = if status.as_u16() == 404
                && self
                    .session_id
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .is_some()
            {
                " (the session may have expired)"
            } else {
                ""
            };
            return Err(McpError::Transport(format!("HTTP {status}{hint}: {text}")));
        }
        Ok(response)
    }

    /// Look through `message` (one JSON-RPC message or a batch) for the response to `id`;
    /// requests the server sends along the way are answered.
    async fn scan(&self, message: Value, id: u64) -> Option<Result<Value, McpError>> {
        let messages = match message {
            Value::Array(items) => items,
            single => vec![single],
        };
        let mut found = None;
        for message in messages {
            if message.get("method").is_some() {
                if message.get("id").is_some() {
                    let _ = self.post(&reply_to_server_request(&message)).await;
                }
            } else if message.get("id").and_then(Value::as_u64) == Some(id) {
                found = found.or_else(|| response_outcome(&message));
            }
        }
        found
    }

    async fn exchange(&self, body: &Value, id: u64) -> Result<Value, McpError> {
        let response = self.post(body).await?;
        let streamed = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|t| t.contains("text/event-stream"));
        if !streamed {
            let message: Value = response
                .json()
                .await
                .map_err(|e| McpError::Protocol(format!("the reply is not JSON: {e}")))?;
            return self
                .scan(message, id)
                .await
                .unwrap_or_else(|| Err(McpError::Protocol("the reply holds no response".into())));
        }
        let mut events = SseEvents::default();
        let mut bytes = response.bytes_stream();
        while let Some(chunk) = bytes.next().await {
            let chunk = chunk
                .map_err(|e| McpError::Transport(format!("reading the stream failed: {e}")))?;
            events.feed(&chunk);
            while let Some(data) = events.next_data() {
                let Ok(message) = serde_json::from_str::<Value>(&data) else {
                    continue;
                };
                if let Some(outcome) = self.scan(message, id).await {
                    return outcome;
                }
            }
        }
        Err(McpError::Closed(
            "the event stream ended without a response".into(),
        ))
    }
}

#[async_trait]
impl Transport for HttpTransport {
    async fn request(
        &self,
        method: &str,
        params: Option<Value>,
        timeout: Duration,
    ) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let body = request_message(id, method, params);
        tokio::time::timeout(timeout, self.exchange(&body, id))
            .await
            .map_err(|_| McpError::Timeout(timeout))?
    }

    async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), McpError> {
        tokio::time::timeout(
            AUX_TIMEOUT,
            self.post(&notification_message(method, params)),
        )
        .await
        .map_err(|_| McpError::Timeout(AUX_TIMEOUT))?
        .map(|_| ())
    }

    fn set_protocol_version(&self, version: &str) {
        *self
            .protocol_version
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(version.to_string());
    }

    async fn close(&self) {
        let session = self
            .session_id
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if session.is_none() {
            return;
        }
        // Ending the session is polite and optional: a server may answer 405, which is fine.
        let mut headers = self.header_map().unwrap_or_default();
        if let Some(id) = session
            .as_deref()
            .and_then(|s| HeaderValue::from_str(s).ok())
        {
            headers.insert(HeaderName::from_static(SESSION_HEADER), id);
        }
        let delete = self.client.delete(&self.url).headers(headers).send();
        let _ = tokio::time::timeout(Duration::from_secs(5), delete).await;
    }
}

/// Builds the HTTP transport and runs the handshake.
pub(crate) struct HttpConnector(pub StreamableHttpParams);

#[async_trait]
impl Connector for HttpConnector {
    async fn connect(&self, timeout: Duration) -> Result<Arc<dyn Transport>, McpError> {
        let transport = HttpTransport::new(&self.0);
        if let Err(error) = handshake(&transport, timeout).await {
            transport.close().await;
            return Err(error);
        }
        Ok(Arc::new(transport))
    }
}

#[cfg(test)]
mod tests {
    use super::SseEvents;

    #[test]
    fn sse_events_handle_chunking_crlf_and_multiline_data() {
        let mut events = SseEvents::default();
        events.feed(b"event: message\r\nda");
        assert!(events.next_data().is_none());
        events.feed(b"ta: {\"a\":1}\r\n\r\n: comment\n\ndata: x\ndata: y\n\n");
        assert_eq!(events.next_data().as_deref(), Some("{\"a\":1}"));
        assert_eq!(
            events.next_data().as_deref(),
            Some("x\ny"),
            "multi-line data is joined"
        );
        assert!(
            events.next_data().is_none(),
            "comment-only events carry no data"
        );
    }

    #[test]
    fn sse_events_keep_multibyte_text_and_crlf_split_across_chunks() {
        let payload = "data: {\"t\":\"你好🙂\"}\r\n\r\n".as_bytes().to_vec();
        for split in 1..payload.len() {
            let mut events = SseEvents::default();
            events.feed(&payload[..split]);
            events.feed(&payload[split..]);
            assert_eq!(
                events.next_data().as_deref(),
                Some("{\"t\":\"你好🙂\"}"),
                "split at {split}"
            );
        }
    }

    /// Serve one canned HTTP response per connection; returns the base URL and a hit counter.
    async fn serve(response: String) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let hits = std::sync::Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                counter.fetch_add(1, Ordering::SeqCst);
                let mut buf = [0u8; 4096];
                let _ = socket.read(&mut buf).await;
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        (url, hits)
    }

    #[tokio::test]
    async fn redirects_to_another_origin_are_not_followed() {
        use std::sync::atomic::Ordering;
        let (other_url, other_hits) =
            serve("HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}".into()).await;
        let (url, _) = serve(format!(
            "HTTP/1.1 307 Temporary Redirect\r\nlocation: {other_url}/steal\r\ncontent-length: 0\r\n\r\n"
        ))
        .await;
        let response = super::build_client()
            .post(&url)
            .header("x-api-key", "secret")
            .body("{}")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), 307);
        assert_eq!(
            other_hits.load(Ordering::SeqCst),
            0,
            "credentials and body must not reach another origin"
        );
    }
}
