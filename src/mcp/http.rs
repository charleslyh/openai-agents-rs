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
}

impl SseEvents {
    fn feed(&mut self, chunk: &[u8]) {
        self.buffer.push_str(&String::from_utf8_lossy(chunk));
        if self.buffer.contains('\r') {
            self.buffer = self.buffer.replace("\r\n", "\n").replace('\r', "\n");
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
            client: reqwest::Client::new(),
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
        map.insert(ACCEPT, HeaderValue::from_static("application/json, text/event-stream"));
        map.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(id) = self.session_id.lock().expect("session").as_deref() {
            map.insert(
                HeaderName::from_static(SESSION_HEADER),
                HeaderValue::from_str(id).map_err(|_| bad("session id", id))?,
            );
        }
        if let Some(version) = self.protocol_version.lock().expect("version").as_deref() {
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
        if let Some(id) = response.headers().get(SESSION_HEADER).and_then(|v| v.to_str().ok()) {
            *self.session_id.lock().expect("session") = Some(id.to_string());
        }
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let hint = if status.as_u16() == 404 && self.session_id.lock().expect("session").is_some() {
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
            let chunk = chunk.map_err(|e| McpError::Transport(format!("reading the stream failed: {e}")))?;
            events.feed(&chunk);
            while let Some(data) = events.next_data() {
                let Ok(message) = serde_json::from_str::<Value>(&data) else { continue };
                if let Some(outcome) = self.scan(message, id).await {
                    return outcome;
                }
            }
        }
        Err(McpError::Closed("the event stream ended without a response".into()))
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
        self.post(&notification_message(method, params)).await.map(|_| ())
    }

    fn set_protocol_version(&self, version: &str) {
        *self.protocol_version.lock().expect("version") = Some(version.to_string());
    }

    async fn close(&self) {
        let session = self.session_id.lock().expect("session").take();
        if session.is_none() {
            return;
        }
        // Ending the session is polite and optional: a server may answer 405, which is fine.
        let mut headers = self.header_map().unwrap_or_default();
        if let Some(id) = session.as_deref().and_then(|s| HeaderValue::from_str(s).ok()) {
            headers.insert(HeaderName::from_static(SESSION_HEADER), id);
        }
        let _ = self.client.delete(&self.url).headers(headers).send().await;
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
        assert_eq!(events.next_data().as_deref(), Some("x\ny"), "multi-line data is joined");
        assert!(events.next_data().is_none(), "comment-only events carry no data");
    }
}
