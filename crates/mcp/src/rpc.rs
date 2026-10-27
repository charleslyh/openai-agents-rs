//! JSON-RPC plumbing shared by the MCP transports.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

/// Protocol revision asked for in the legacy `initialize` handshake.
pub(crate) const PROTOCOL_VERSION: &str = "2025-06-18";
/// Revisions the legacy handshake accepts; a server answering with another one is rejected.
const SUPPORTED_PROTOCOL_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];
/// The stateless revision: no handshake, version and capabilities in every request's `_meta`.
pub(crate) const MODERN_VERSION: &str = "2026-07-28";
/// JSON-RPC code of `UnsupportedProtocolVersionError` in the stateless revision.
const UNSUPPORTED_VERSION_CODE: i64 = -32022;
const META_VERSION: &str = "io.modelcontextprotocol/protocolVersion";
const META_CAPABILITIES: &str = "io.modelcontextprotocol/clientCapabilities";
const META_CLIENT_INFO: &str = "io.modelcontextprotocol/clientInfo";

/// What can go wrong talking to an MCP server.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum McpError {
    /// The server answered with a JSON-RPC error object.
    #[error("MCP error {code}: {message}")]
    Rpc {
        /// JSON-RPC error code.
        code: i64,
        /// Human readable message from the server.
        message: String,
        /// Optional structured details.
        data: Option<Value>,
    },
    /// The bytes could not be exchanged (spawn failure, HTTP error, write failure).
    #[error("MCP transport error: {0}")]
    Transport(String),
    /// No answer arrived in time.
    #[error("MCP request timed out after {0:?}")]
    Timeout(Duration),
    /// The connection ended before an answer arrived.
    #[error("MCP connection closed: {0}")]
    Closed(String),
    /// The server broke the protocol (bad handshake, unusable reply).
    #[error("MCP protocol error: {0}")]
    Protocol(String),
}

/// One live connection to a server. A transport numbers its own requests.
#[async_trait]
pub(crate) trait Transport: Send + Sync {
    /// Send a request and wait for its `result`.
    async fn request(
        &self,
        method: &str,
        params: Option<Value>,
        timeout: Duration,
    ) -> Result<Value, McpError>;

    /// Send a notification (no answer expected).
    async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), McpError>;

    /// Record the protocol revision in use (HTTP sends it on every request); `None` forgets it.
    fn set_protocol_version(&self, _version: Option<&str>) {}

    /// End the connection and release its resources.
    async fn close(&self);
}

/// Opens a [`Transport`] and runs the initialize handshake on it.
#[async_trait]
pub(crate) trait Connector: Send + Sync {
    async fn connect(&self, timeout: Duration) -> Result<std::sync::Arc<dyn Transport>, McpError>;
}

pub(crate) fn request_message(id: u64, method: &str, params: Option<Value>) -> Value {
    let mut message = json!({"jsonrpc": "2.0", "id": id, "method": method});
    if let Some(params) = params {
        message["params"] = params;
    }
    message
}

pub(crate) fn notification_message(method: &str, params: Option<Value>) -> Value {
    let mut message = json!({"jsonrpc": "2.0", "method": method});
    if let Some(params) = params {
        message["params"] = params;
    }
    message
}

/// Reply to a request the *server* sent us. Only `ping` is supported.
pub(crate) fn reply_to_server_request(message: &Value) -> Value {
    let id = message.get("id").cloned().unwrap_or(Value::Null);
    if message.get("method").and_then(Value::as_str) == Some("ping") {
        json!({"jsonrpc": "2.0", "id": id, "result": {}})
    } else {
        json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "Method not found"}})
    }
}

/// The outcome carried by a response message, or `None` when it is not a response.
pub(crate) fn response_outcome(message: &Value) -> Option<Result<Value, McpError>> {
    if let Some(error) = message.get("error").filter(|e| !e.is_null()) {
        return Some(Err(McpError::Rpc {
            code: error.get("code").and_then(Value::as_i64).unwrap_or(0),
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
                .to_string(),
            data: error.get("data").cloned(),
        }));
    }
    message.get("result").map(|result| Ok(result.clone()))
}

/// Connect as well as the server allows: the stateless revision when it speaks it, the
/// `initialize` handshake otherwise.
///
/// `server/discover` goes first, as the specification recommends for a client that must also talk
/// to older servers; the old handshake is the fallback, so servers deployed today keep working.
pub(crate) async fn negotiate(
    transport: Arc<dyn Transport>,
    timeout: Duration,
) -> Result<Arc<dyn Transport>, McpError> {
    transport.set_protocol_version(Some(MODERN_VERSION));
    match discover(transport.as_ref(), timeout).await {
        Ok(()) => Ok(Arc::new(StatelessTransport { inner: transport })),
        // A timeout or a closed pipe says nothing about the dialect, and a second attempt would
        // only double the wait.
        Err(error @ (McpError::Timeout(_) | McpError::Closed(_))) => Err(error),
        // A modern server that does not speak our version: retrying the old way cannot help.
        Err(error) if is_modern_refusal(&error) => Err(error),
        Err(_) => {
            transport.set_protocol_version(None);
            handshake(transport.as_ref(), timeout).await?;
            Ok(transport)
        }
    }
}

fn is_modern_refusal(error: &McpError) -> bool {
    matches!(error, McpError::Rpc { code, .. } if *code == UNSUPPORTED_VERSION_CODE)
        || matches!(error, McpError::Protocol(_))
}

/// `_meta` every stateless request carries.
fn request_meta() -> Value {
    json!({
        META_VERSION: MODERN_VERSION,
        META_CAPABILITIES: {},
        META_CLIENT_INFO: {"name": "openai-agents-rs", "version": env!("CARGO_PKG_VERSION")},
    })
}

/// `params` with the stateless `_meta` merged in; keys the caller set are kept.
fn with_meta(params: Option<Value>) -> Value {
    let mut params = match params {
        Some(Value::Object(map)) => Value::Object(map),
        _ => json!({}),
    };
    let mut meta = request_meta();
    if let Some(existing) = params.get("_meta").and_then(Value::as_object) {
        for (key, value) in existing {
            meta[key] = value.clone();
        }
    }
    params["_meta"] = meta;
    params
}

/// `server/discover`: ask which versions the server speaks and whether ours is one of them.
async fn discover(transport: &dyn Transport, timeout: Duration) -> Result<(), McpError> {
    let result = transport
        .request("server/discover", Some(with_meta(None)), timeout)
        .await?;
    let supported = result
        .get("supportedVersions")
        .and_then(Value::as_array)
        .ok_or_else(|| McpError::Rpc {
            code: 0,
            message: "the discover reply has no supportedVersions".into(),
            data: None,
        })?;
    if supported.iter().any(|v| v.as_str() == Some(MODERN_VERSION)) {
        Ok(())
    } else {
        Err(McpError::Protocol(format!(
            "the server supports protocol versions {supported:?} and not {MODERN_VERSION}"
        )))
    }
}

/// A connection that speaks the stateless revision: every request carries its own `_meta`.
struct StatelessTransport {
    inner: Arc<dyn Transport>,
}

#[async_trait]
impl Transport for StatelessTransport {
    async fn request(
        &self,
        method: &str,
        params: Option<Value>,
        timeout: Duration,
    ) -> Result<Value, McpError> {
        let result = self
            .inner
            .request(method, Some(with_meta(params)), timeout)
            .await?;
        check_result_type(&result)?;
        Ok(result)
    }

    async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), McpError> {
        self.inner.notify(method, params).await
    }

    async fn close(&self) {
        self.inner.close().await;
    }
}

/// A `resultType: "input_required"` reply asks the client for more input before it can answer
/// (multi round-trip requests). This client declares no capability that could supply it, so such a
/// reply is an error and not a result to read as if it were complete. A reply without the field
/// comes from an older server and is complete.
fn check_result_type(result: &Value) -> Result<(), McpError> {
    match result.get("resultType").and_then(Value::as_str) {
        Some("input_required") => Err(McpError::Protocol(
            "the server needs more input to answer (resultType input_required), which this client \
             cannot provide"
                .into(),
        )),
        _ => Ok(()),
    }
}

/// `initialize` followed by `notifications/initialized` (revisions before the stateless one).
async fn handshake(transport: &dyn Transport, timeout: Duration) -> Result<(), McpError> {
    let result = transport
        .request(
            "initialize",
            Some(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "openai-agents-rs", "version": env!("CARGO_PKG_VERSION")},
            })),
            timeout,
        )
        .await?;
    let version = result
        .get("protocolVersion")
        .and_then(Value::as_str)
        .ok_or_else(|| McpError::Protocol("the initialize reply has no protocolVersion".into()))?;
    if !SUPPORTED_PROTOCOL_VERSIONS.contains(&version) {
        return Err(McpError::Protocol(format!(
            "the server speaks protocol version {version}, which this client does not support"
        )));
    }
    transport.set_protocol_version(Some(version));
    transport.notify("notifications/initialized", None).await
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// Answers by method with a canned outcome and records every request and notification.
    struct Script {
        discover: Result<Value, McpError>,
        log: Mutex<Vec<(String, Option<Value>)>>,
        version: Mutex<Option<String>>,
    }

    impl Script {
        fn new(discover: Result<Value, McpError>) -> Arc<Self> {
            Arc::new(Self {
                discover,
                log: Mutex::default(),
                version: Mutex::default(),
            })
        }

        fn methods(&self) -> Vec<String> {
            self.log
                .lock()
                .unwrap()
                .iter()
                .map(|(m, _)| m.clone())
                .collect()
        }
    }

    #[async_trait]
    impl Transport for Script {
        async fn request(
            &self,
            method: &str,
            params: Option<Value>,
            _: Duration,
        ) -> Result<Value, McpError> {
            self.log.lock().unwrap().push((method.into(), params));
            match method {
                "server/discover" => self.discover.clone(),
                "initialize" => Ok(json!({"protocolVersion": "2025-06-18"})),
                _ => Ok(json!({"resultType": "complete", "tools": []})),
            }
        }

        async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), McpError> {
            self.log.lock().unwrap().push((method.into(), params));
            Ok(())
        }

        fn set_protocol_version(&self, version: Option<&str>) {
            *self.version.lock().unwrap() = version.map(str::to_string);
        }

        async fn close(&self) {}
    }

    fn rpc_error(code: i64) -> McpError {
        McpError::Rpc {
            code,
            message: "no".into(),
            data: None,
        }
    }

    #[tokio::test]
    async fn a_stateless_server_is_used_without_a_handshake_and_every_request_says_who_asks() {
        let script = Script::new(Ok(json!({"supportedVersions": [MODERN_VERSION]})));
        let transport = negotiate(script.clone(), Duration::from_secs(1))
            .await
            .unwrap();
        transport
            .request(
                "tools/call",
                Some(json!({"name": "t", "_meta": {"x": 1}})),
                Duration::from_secs(1),
            )
            .await
            .unwrap();

        assert_eq!(script.methods(), ["server/discover", "tools/call"]);
        assert_eq!(
            script.version.lock().unwrap().as_deref(),
            Some(MODERN_VERSION)
        );
        let log = script.log.lock().unwrap();
        let meta = &log[1].1.as_ref().unwrap()["_meta"];
        assert_eq!(meta[META_VERSION], MODERN_VERSION);
        assert_eq!(meta[META_CAPABILITIES], json!({}));
        assert_eq!(meta[META_CLIENT_INFO]["name"], "openai-agents-rs");
        assert_eq!(meta["x"], 1, "a key the caller set is kept");
        assert_eq!(log[1].1.as_ref().unwrap()["name"], "t");
    }

    #[tokio::test]
    async fn an_older_server_falls_back_to_the_initialize_handshake() {
        let script = Script::new(Err(rpc_error(-32601)));
        let transport = negotiate(script.clone(), Duration::from_secs(1))
            .await
            .unwrap();
        transport
            .request("tools/list", None, Duration::from_secs(1))
            .await
            .unwrap();

        assert_eq!(
            script.methods(),
            [
                "server/discover",
                "initialize",
                "notifications/initialized",
                "tools/list"
            ]
        );
        assert_eq!(
            script.version.lock().unwrap().as_deref(),
            Some("2025-06-18")
        );
        let log = script.log.lock().unwrap();
        assert!(
            log[3].1.is_none(),
            "an older server gets no stateless _meta"
        );
    }

    #[tokio::test]
    async fn a_stateless_server_that_lacks_our_version_is_an_error_not_a_fallback() {
        let refused = Script::new(Err(McpError::Rpc {
            code: UNSUPPORTED_VERSION_CODE,
            message: "Unsupported protocol version".into(),
            data: Some(json!({"supported": ["2030-01-01"], "requested": MODERN_VERSION})),
        }));
        assert!(negotiate(refused.clone(), Duration::from_secs(1))
            .await
            .is_err());
        assert_eq!(refused.methods(), ["server/discover"]);

        let other = Script::new(Ok(json!({"supportedVersions": ["2030-01-01"]})));
        let error = negotiate(other.clone(), Duration::from_secs(1))
            .await
            .err()
            .unwrap();
        assert!(matches!(error, McpError::Protocol(_)), "{error}");
        assert_eq!(other.methods(), ["server/discover"]);
    }

    #[tokio::test]
    async fn a_timeout_while_discovering_is_not_retried_the_old_way() {
        let script = Script::new(Err(McpError::Timeout(Duration::from_secs(1))));
        assert!(negotiate(script.clone(), Duration::from_secs(1))
            .await
            .is_err());
        assert_eq!(script.methods(), ["server/discover"]);
    }

    #[tokio::test]
    async fn a_reply_that_asks_for_input_is_an_error_and_one_without_a_type_is_complete() {
        let script = Script::new(Ok(json!({"supportedVersions": [MODERN_VERSION]})));
        let transport = negotiate(script, Duration::from_secs(1)).await.unwrap();
        // The scripted server answers `complete`; check the two other shapes directly.
        assert!(transport
            .request("tools/list", None, Duration::from_secs(1))
            .await
            .is_ok());
        assert!(
            check_result_type(&json!({"resultType": "input_required", "inputRequests": {}}))
                .is_err()
        );
        assert!(check_result_type(&json!({"tools": []})).is_ok());
    }
}
