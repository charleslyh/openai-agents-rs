//! JSON-RPC plumbing shared by the MCP transports.

// Only the transports (feature `mcp`) use most of this.
#![cfg_attr(not(feature = "mcp"), allow(dead_code))]

use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

/// Protocol revision this client asks for.
pub(crate) const PROTOCOL_VERSION: &str = "2025-06-18";
/// Revisions this client can speak; a server answering with another one is rejected.
const SUPPORTED_PROTOCOL_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

/// What can go wrong talking to an MCP server.
#[derive(Debug, Clone, thiserror::Error)]
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

    /// Record the protocol revision the server picked (HTTP sends it on every request).
    fn set_protocol_version(&self, _version: &str) {}

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

/// `initialize` followed by `notifications/initialized`.
pub(crate) async fn handshake(
    transport: &dyn Transport,
    timeout: Duration,
) -> Result<(), McpError> {
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
    transport.set_protocol_version(version);
    transport.notify("notifications/initialized", None).await
}
