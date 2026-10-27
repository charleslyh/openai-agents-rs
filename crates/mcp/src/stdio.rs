//! MCP over a child process's stdin / stdout (Python: `MCPServerStdio`).
//!
//! Messages are single-line JSON documents separated by newlines.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Mutex as AsyncMutex};

use super::rpc::{
    negotiate, notification_message, reply_to_server_request, request_message, response_outcome,
    Connector, McpError, Transport,
};

/// Environment variables a spawned server inherits. Everything else is withheld on purpose, so a
/// server never sees API keys that happen to be in the parent's environment; pass what it needs
/// through [`StdioParams::env`] (Python: `get_default_environment`).
const INHERITED_ENV: [&str; 12] = [
    "HOME",
    "LOGNAME",
    "PATH",
    "SHELL",
    "TERM",
    "USER",
    "APPDATA",
    "HOMEDRIVE",
    "HOMEPATH",
    "SYSTEMROOT",
    "TEMP",
    "USERNAME",
];

/// How to start a stdio server (Python: `MCPServerStdioParams`).
#[derive(Debug, Clone, Default)]
pub struct StdioParams {
    /// Program to run.
    pub command: String,
    /// Arguments.
    pub args: Vec<String>,
    /// Extra environment variables, on top of the safe subset inherited from this process.
    pub env: HashMap<String, String>,
    /// Working directory.
    pub cwd: Option<PathBuf>,
}

impl StdioParams {
    /// Run `command` with no arguments.
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            ..Self::default()
        }
    }

    /// Set the arguments.
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    /// Add an environment variable.
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Set the working directory.
    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }
}

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, McpError>>>>>;
type SharedStdin = Arc<AsyncMutex<Option<ChildStdin>>>;

/// Longest a single message may take to reach the server's stdin.
///
/// A server that stops reading lets the pipe buffer fill, and an unbounded `write_all` would then
/// block forever while holding the stdin lock, which also stalls `close()`.
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);

async fn write_message(stdin: &SharedStdin, message: &Value) -> Result<(), McpError> {
    let mut guard = tokio::time::timeout(WRITE_TIMEOUT, stdin.lock())
        .await
        .map_err(|_| McpError::Timeout(WRITE_TIMEOUT))?;
    let pipe = guard
        .as_mut()
        .ok_or_else(|| McpError::Closed("the server's input is closed".into()))?;
    let mut line = message.to_string();
    line.push('\n');
    let written = tokio::time::timeout(WRITE_TIMEOUT, async {
        pipe.write_all(line.as_bytes()).await?;
        pipe.flush().await
    })
    .await;
    match written {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(McpError::Transport(format!(
            "writing to the server failed: {e}"
        ))),
        Err(_) => {
            // A half-written line would corrupt the framing, so the pipe cannot be reused.
            guard.take();
            Err(McpError::Timeout(WRITE_TIMEOUT))
        }
    }
}

pub(crate) struct StdioTransport {
    stdin: SharedStdin,
    pending: Pending,
    next_id: AtomicU64,
    closed: Arc<AtomicBool>,
    child: AsyncMutex<Child>,
}

impl StdioTransport {
    fn spawn(params: &StdioParams) -> Result<Self, McpError> {
        let mut command = Command::new(&params.command);
        command
            .args(&params.args)
            .env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // A server's diagnostics go to our stderr, like Python's default `errlog`.
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        for key in INHERITED_ENV {
            if let Ok(value) = std::env::var(key) {
                command.env(key, value);
            }
        }
        command.envs(&params.env);
        if let Some(cwd) = &params.cwd {
            command.current_dir(cwd);
        }
        let mut child = command.spawn().map_err(|e| {
            McpError::Transport(format!("could not start `{}`: {e}", params.command))
        })?;
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            return Err(McpError::Transport(format!(
                "could not open the pipes of `{}`",
                params.command
            )));
        };

        let stdin: SharedStdin = Arc::new(AsyncMutex::new(Some(stdin)));
        let pending: Pending = Arc::default();
        let closed = Arc::new(AtomicBool::new(false));

        let reader = (
            Arc::clone(&stdin),
            Arc::clone(&pending),
            Arc::clone(&closed),
        );
        tokio::spawn(async move {
            let (stdin, pending, closed) = reader;
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(message) = serde_json::from_str::<Value>(&line) else {
                    continue; // not JSON-RPC (a stray log line); ignore it
                };
                if message.get("method").is_some() {
                    // The server asks us something. Answer so it does not wait forever.
                    if message.get("id").is_some() {
                        let _ = write_message(&stdin, &reply_to_server_request(&message)).await;
                    }
                } else if let Some(id) = message.get("id").and_then(Value::as_u64) {
                    let sender = pending
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .remove(&id);
                    if let (Some(sender), Some(outcome)) = (sender, response_outcome(&message)) {
                        let _ = sender.send(outcome);
                    }
                }
            }
            closed.store(true, Ordering::SeqCst);
            for (_, sender) in pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .drain()
            {
                let _ = sender.send(Err(McpError::Closed("the server closed its output".into())));
            }
        });

        Ok(Self {
            stdin,
            pending,
            next_id: AtomicU64::new(1),
            closed,
            child: AsyncMutex::new(child),
        })
    }
}

#[async_trait]
impl Transport for StdioTransport {
    async fn request(
        &self,
        method: &str,
        params: Option<Value>,
        timeout: Duration,
    ) -> Result<Value, McpError> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(McpError::Closed("the server has exited".into()));
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (sender, receiver) = oneshot::channel();
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, sender);
        if let Err(error) = write_message(&self.stdin, &request_message(id, method, params)).await {
            self.pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&id);
            return Err(error);
        }
        match tokio::time::timeout(timeout, receiver).await {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(_)) => Err(McpError::Closed("the server closed its output".into())),
            Err(_) => {
                self.pending
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .remove(&id);
                // Tell the server to stop working on it (best effort).
                let _ = self
                    .notify(
                        "notifications/cancelled",
                        Some(serde_json::json!({"requestId": id, "reason": "timeout"})),
                    )
                    .await;
                Err(McpError::Timeout(timeout))
            }
        }
    }

    async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), McpError> {
        write_message(&self.stdin, &notification_message(method, params)).await
    }

    async fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        // Closing stdin is the polite way to stop a stdio server.
        // If a stuck write holds the lock, skip the polite step and go straight to the kill below.
        if let Ok(mut stdin) = tokio::time::timeout(Duration::from_secs(1), self.stdin.lock()).await
        {
            stdin.take();
        }
        let mut child = self.child.lock().await;
        if tokio::time::timeout(Duration::from_secs(2), child.wait())
            .await
            .is_err()
        {
            let _ = child.kill().await;
        }
    }
}

/// Starts the process and runs the handshake.
pub(crate) struct StdioConnector(pub StdioParams);

#[async_trait]
impl Connector for StdioConnector {
    async fn connect(&self, timeout: Duration) -> Result<Arc<dyn Transport>, McpError> {
        let transport: Arc<dyn Transport> = Arc::new(StdioTransport::spawn(&self.0)?);
        match negotiate(Arc::clone(&transport), timeout).await {
            Ok(transport) => Ok(transport),
            Err(error) => {
                transport.close().await;
                Err(error)
            }
        }
    }
}
