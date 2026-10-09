//! Connecting a group of MCP servers together (Python: `MCPServerManager`).
//!
//! A server connects on first use, so an unreachable one only fails the run when its tools are
//! first listed. The manager connects the whole group up front, remembers which servers failed,
//! and hands the agent only the ones that work, so one bad server does not take the agent down.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::McpServer;
use crate::error::AgentsError;

#[derive(Default)]
struct State {
    /// Indexes into `servers` of the servers that failed their last connection attempt.
    failed: Vec<usize>,
    /// The error text of each failed server, by index.
    errors: Vec<(usize, String)>,
}

/// Connects MCP servers as a group and tracks which of them work.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use openai_agents::{Agent, McpServer, McpServerManager};
/// # async fn demo(servers: Vec<Arc<dyn McpServer>>) -> Result<(), openai_agents::AgentsError> {
/// let manager = McpServerManager::new(servers);
/// let active = manager.connect_all().await?;
/// let agent = Agent::new("assistant").mcp_servers(active);
/// // ... run the agent ...
/// manager.cleanup_all().await;
/// # Ok(()) }
/// ```
pub struct McpServerManager {
    servers: Vec<Arc<dyn McpServer>>,
    connect_timeout: Option<Duration>,
    cleanup_timeout: Option<Duration>,
    drop_failed_servers: bool,
    strict: bool,
    connect_in_parallel: bool,
    state: Mutex<State>,
}

impl std::fmt::Debug for McpServerManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServerManager")
            .field(
                "servers",
                &self.servers.iter().map(|s| s.name()).collect::<Vec<_>>(),
            )
            .field("drop_failed_servers", &self.drop_failed_servers)
            .field("strict", &self.strict)
            .finish()
    }
}

impl McpServerManager {
    /// Manage `servers`. By default each connection gets 10 seconds, failures are dropped from
    /// [`active_servers`](Self::active_servers) and do not stop the others, and servers connect
    /// one after the other (Python's defaults).
    pub fn new(servers: Vec<Arc<dyn McpServer>>) -> Self {
        Self {
            servers,
            connect_timeout: Some(Duration::from_secs(10)),
            cleanup_timeout: Some(Duration::from_secs(10)),
            drop_failed_servers: true,
            strict: false,
            connect_in_parallel: false,
            state: Mutex::new(State::default()),
        }
    }

    /// How long one connection attempt may take; `None` waits as long as the server does.
    pub fn connect_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// How long one cleanup may take; `None` waits as long as the server does.
    pub fn cleanup_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.cleanup_timeout = timeout;
        self
    }

    /// Leave failed servers out of [`active_servers`](Self::active_servers) (default `true`).
    /// When `false` the agent still gets them and fails when it first uses one.
    pub fn drop_failed_servers(mut self, drop: bool) -> Self {
        self.drop_failed_servers = drop;
        self
    }

    /// Make [`connect_all`](Self::connect_all) return the first connection error instead of
    /// carrying on without the failed server (default `false`).
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Connect all servers at the same time instead of one by one (default `false`).
    pub fn connect_in_parallel(mut self, parallel: bool) -> Self {
        self.connect_in_parallel = parallel;
        self
    }

    /// Connect every server and return the ones to give the agent.
    ///
    /// A failure is recorded (see [`errors`](Self::errors)) and, unless the manager is
    /// [`strict`](Self::strict), the rest still connect. When strict, the first failure is
    /// returned and the servers that did connect are cleaned up again.
    pub async fn connect_all(&self) -> Result<Vec<Arc<dyn McpServer>>, AgentsError> {
        let all: Vec<usize> = (0..self.servers.len()).collect();
        self.connect_indexes(&all).await?;
        Ok(self.active_servers())
    }

    /// Try again: only the servers that failed (`failed_only`), or all of them.
    pub async fn reconnect(
        &self,
        failed_only: bool,
    ) -> Result<Vec<Arc<dyn McpServer>>, AgentsError> {
        let targets: Vec<usize> = if failed_only {
            self.state.lock().expect("manager state").failed.clone()
        } else {
            (0..self.servers.len()).collect()
        };
        self.connect_indexes(&targets).await?;
        Ok(self.active_servers())
    }

    /// Clean up every server. A server that fails or takes too long is logged, not reported, so
    /// the others still get cleaned up.
    pub async fn cleanup_all(&self) {
        for server in &self.servers {
            match self.within(self.cleanup_timeout, server.cleanup()).await {
                Ok(()) => {}
                Err(error) => {
                    ::tracing::warn!("cleaning up MCP server `{}` failed: {error}", server.name())
                }
            }
        }
    }

    /// The servers to give the agent: all of them, minus the failed ones when they are dropped.
    pub fn active_servers(&self) -> Vec<Arc<dyn McpServer>> {
        let state = self.state.lock().expect("manager state");
        self.servers
            .iter()
            .enumerate()
            .filter(|(i, _)| !(self.drop_failed_servers && state.failed.contains(i)))
            .map(|(_, server)| Arc::clone(server))
            .collect()
    }

    /// Every managed server.
    pub fn all_servers(&self) -> Vec<Arc<dyn McpServer>> {
        self.servers.clone()
    }

    /// The servers that failed their last connection attempt.
    pub fn failed_servers(&self) -> Vec<Arc<dyn McpServer>> {
        let state = self.state.lock().expect("manager state");
        state
            .failed
            .iter()
            .map(|i| Arc::clone(&self.servers[*i]))
            .collect()
    }

    /// `(server name, error)` of each failed server.
    pub fn errors(&self) -> Vec<(String, String)> {
        let state = self.state.lock().expect("manager state");
        state
            .errors
            .iter()
            .map(|(i, error)| (self.servers[*i].name().to_string(), error.clone()))
            .collect()
    }

    async fn within<T>(
        &self,
        timeout: Option<Duration>,
        future: impl std::future::Future<Output = Result<T, AgentsError>>,
    ) -> Result<T, AgentsError> {
        match timeout {
            None => future.await,
            Some(limit) => tokio::time::timeout(limit, future)
                .await
                .unwrap_or_else(|_| {
                    Err(AgentsError::tool(format!(
                        "timed out after {} seconds",
                        limit.as_secs_f64()
                    )))
                }),
        }
    }

    async fn attempt(&self, index: usize) -> Result<(), AgentsError> {
        self.within(self.connect_timeout, self.servers[index].connect())
            .await
    }

    async fn connect_indexes(&self, indexes: &[usize]) -> Result<(), AgentsError> {
        let results: Vec<(usize, Result<(), AgentsError>)> = if self.connect_in_parallel {
            futures::future::join_all(
                indexes
                    .iter()
                    .map(|&i| async move { (i, self.attempt(i).await) }),
            )
            .await
        } else {
            let mut done = Vec::new();
            for &i in indexes {
                let result = self.attempt(i).await;
                let stop = self.strict && result.is_err();
                done.push((i, result));
                if stop {
                    break;
                }
            }
            done
        };

        {
            let mut state = self.state.lock().expect("manager state");
            for (index, result) in &results {
                state.failed.retain(|i| i != index);
                state.errors.retain(|(i, _)| i != index);
                if let Err(error) = result {
                    state.failed.push(*index);
                    state.errors.push((*index, error.to_string()));
                }
            }
            state.failed.sort_unstable();
            state.errors.sort_by_key(|(i, _)| *i);
        }
        if self.strict {
            if let Some(error) = results.into_iter().find_map(|(_, r)| r.err()) {
                // Do not leave half a group connected.
                self.cleanup_all().await;
                return Err(error);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use serde_json::Value;

    use super::*;
    use crate::agent::Agent;
    use crate::mcp::{McpCallToolResult, McpTool};
    use crate::run_context::RunContextWrapper;

    /// A server whose `connect` fails the first `fail_times` times, or hangs when `hang`.
    #[derive(Debug)]
    struct Fake {
        name: &'static str,
        fail_times: AtomicUsize,
        hang: bool,
        connects: AtomicUsize,
        cleanups: AtomicUsize,
    }

    impl Fake {
        fn new(name: &'static str, fail_times: usize, hang: bool) -> Arc<Self> {
            Arc::new(Self {
                name,
                fail_times: AtomicUsize::new(fail_times),
                hang,
                connects: AtomicUsize::new(0),
                cleanups: AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl McpServer for Fake {
        fn name(&self) -> &str {
            self.name
        }
        async fn connect(&self) -> Result<(), AgentsError> {
            self.connects.fetch_add(1, Ordering::SeqCst);
            if self.hang {
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
            if self
                .fail_times
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
            {
                return Err(AgentsError::tool("refused"));
            }
            Ok(())
        }
        async fn cleanup(&self) -> Result<(), AgentsError> {
            self.cleanups.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn list_tools(
            &self,
            _: &RunContextWrapper,
            _: &Agent,
        ) -> Result<Vec<McpTool>, AgentsError> {
            Ok(Vec::new())
        }
        async fn call_tool(
            &self,
            _: &str,
            _: Option<Value>,
        ) -> Result<McpCallToolResult, AgentsError> {
            unreachable!()
        }
    }

    fn names(servers: &[Arc<dyn McpServer>]) -> Vec<String> {
        servers.iter().map(|s| s.name().to_string()).collect()
    }

    fn group(fakes: &[Arc<Fake>]) -> Vec<Arc<dyn McpServer>> {
        fakes
            .iter()
            .map(|f| Arc::clone(f) as Arc<dyn McpServer>)
            .collect()
    }

    #[tokio::test]
    async fn failed_servers_are_dropped_and_can_be_retried() {
        let fakes = [
            Fake::new("a", 0, false),
            Fake::new("b", 1, false),
            Fake::new("c", 0, false),
        ];
        let manager = McpServerManager::new(group(&fakes));
        assert_eq!(names(&manager.connect_all().await.unwrap()), ["a", "c"]);
        assert_eq!(names(&manager.failed_servers()), ["b"]);
        assert_eq!(manager.errors().len(), 1);
        assert!(manager.errors()[0].1.contains("refused"));
        assert_eq!(names(&manager.all_servers()), ["a", "b", "c"]);

        // Only the failed one is tried again, and it works now.
        assert_eq!(
            names(&manager.reconnect(true).await.unwrap()),
            ["a", "b", "c"]
        );
        assert_eq!(
            fakes[0].connects.load(Ordering::SeqCst),
            1,
            "healthy servers were not reconnected"
        );
        assert_eq!(fakes[1].connects.load(Ordering::SeqCst), 2);
        assert!(manager.failed_servers().is_empty() && manager.errors().is_empty());
    }

    #[tokio::test]
    async fn failed_servers_can_be_kept() {
        let fakes = [Fake::new("a", 1, false), Fake::new("b", 0, false)];
        let manager = McpServerManager::new(group(&fakes)).drop_failed_servers(false);
        assert_eq!(names(&manager.connect_all().await.unwrap()), ["a", "b"]);
        assert_eq!(names(&manager.failed_servers()), ["a"]);
    }

    #[tokio::test]
    async fn strict_stops_at_the_first_failure_and_cleans_up() {
        let fakes = [
            Fake::new("a", 0, false),
            Fake::new("b", 1, false),
            Fake::new("c", 0, false),
        ];
        let manager = McpServerManager::new(group(&fakes)).strict(true);
        let error = manager.connect_all().await.unwrap_err();
        assert!(error.to_string().contains("refused"), "{error}");
        assert_eq!(
            fakes[2].connects.load(Ordering::SeqCst),
            0,
            "later servers are not tried"
        );
        assert_eq!(
            fakes[0].cleanups.load(Ordering::SeqCst),
            1,
            "what connected is cleaned up"
        );
    }

    #[tokio::test]
    async fn a_hanging_server_times_out_and_parallel_connects_all() {
        let fakes = [Fake::new("slow", 0, true), Fake::new("ok", 0, false)];
        let manager = McpServerManager::new(group(&fakes))
            .connect_timeout(Some(Duration::from_millis(50)))
            .connect_in_parallel(true);
        assert_eq!(names(&manager.connect_all().await.unwrap()), ["ok"]);
        assert!(
            manager.errors()[0].1.contains("timed out"),
            "{:?}",
            manager.errors()
        );
        manager.cleanup_all().await;
        assert_eq!(fakes[1].cleanups.load(Ordering::SeqCst), 1);
    }
}
