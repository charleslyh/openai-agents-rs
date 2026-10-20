//! Web demo of the Rust Agents SDK: one chat box, one agent network, a capability checklist and a
//! chaos drawer that injects faults to show how the SDK behaves at its edges.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`, `OPENAI_API`,
//! `DEMO_ADDR` (default `127.0.0.1:3000`), `DEMO_DEBUG` (`1`/`0`; on only for loopback by default).
//!
//! ```bash
//! cargo run -p opai-agents-web-demo
//! ```

mod agents;
mod events;
mod faults;
mod hooks;
mod mcp_server;
mod model;
mod runner;
mod server;
mod state;
mod tools;
mod trace;

use std::net::SocketAddr;
use std::sync::Arc;

use opai_agents::{McpClient, McpServer, StdioParams, ToolFilter};
use state::AppState;
use trace::TraceCollector;

fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Server mode must be decided before anything reads the model environment: the child process
    // gets a scrubbed environment and has no API key.
    if std::env::args().nth(1).as_deref() == Some(mcp_server::MCP_SERVER_FLAG) {
        mcp_server::run()?;
        return Ok(());
    }

    let model = model::model()?;
    let model_name = std::env::var("OPENAI_MODEL").unwrap_or_default();
    let api = std::env::var("OPENAI_API").unwrap_or_else(|_| "chat_completions".into());

    let addr: SocketAddr = std::env::var("DEMO_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3000".into())
        .parse()
        .map_err(|e| format!("DEMO_ADDR is not a socket address: {e}"))?;
    let loopback = addr.ip().is_loopback();
    let debug = env_flag("DEMO_DEBUG").unwrap_or(loopback);
    if !loopback {
        eprintln!(
            "warning: listening on {addr}, which is not a loopback address. This demo has no \
             authentication; anyone who can reach it can spend your model quota."
        );
        if debug {
            eprintln!("warning: DEMO_DEBUG=1 on a non-loopback address enables fault injection for everyone.");
        }
    }

    let trace = TraceCollector::install();

    // The MCP server is this same executable, started with a fixed argument.
    let exe = std::env::current_exe()?;
    let mcp: Arc<dyn McpServer> = McpClient::stdio(
        StdioParams::new(exe.to_string_lossy()).args([mcp_server::MCP_SERVER_FLAG]),
    )
    .tool_filter(ToolFilter::block(["mcp_admin_reset"]))
    .shared();

    let app = AppState::new(
        model,
        model_name.clone(),
        api,
        Some(Arc::clone(&mcp)),
        trace,
        debug,
    );

    let sweeper = Arc::clone(&app);
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(30));
        loop {
            tick.tick().await;
            sweeper.sweep();
        }
    });

    let allowed_hosts = loopback.then(|| {
        vec![
            addr.to_string(),
            format!("localhost:{}", addr.port()),
            format!("127.0.0.1:{}", addr.port()),
            format!("[::1]:{}", addr.port()),
        ]
    });
    let live = server::live_reload(&app);
    let router = server::router(app, allowed_hosts);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("web demo: http://{addr}  model={model_name}  debug={debug}");
    if live {
        eprintln!("live reload: index.html is read from disk; edits refresh the page, no rebuild");
    }

    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;

    let _ = mcp.cleanup().await;
    Ok(())
}
