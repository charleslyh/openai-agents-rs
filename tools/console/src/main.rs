//! `opai-console`: a live console for agent runs.
//!
//! ```bash
//! cargo run -p opai-agents-console
//! # then, in the application: ConsoleLink::start(LinkConfig::new("127.0.0.1:7077"), &hub)
//! ```
//!
//! Settings come from the environment:
//!
//! * `CONSOLE_ADDR` — where to listen; loopback only. Default `127.0.0.1:7077`.
//! * `CONSOLE_TOKEN` — when set, `POST /ingest` must carry it as a bearer token.
//! * `CONSOLE_MAX_RUNS` — runs kept in memory. Default 200.

use std::net::ToSocketAddrs;
use std::sync::Arc;

use opai_agents_console::server::{router, AppState};
use opai_agents_console::store::{Store, StoreLimits};

const DEFAULT_ADDR: &str = "127.0.0.1:7077";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let wanted = std::env::var("CONSOLE_ADDR").unwrap_or_else(|_| DEFAULT_ADDR.to_string());
    let addrs: Vec<_> = wanted.to_socket_addrs()?.collect();
    if addrs.is_empty() || addrs.iter().any(|a| !a.ip().is_loopback()) {
        return Err(format!(
            "CONSOLE_ADDR={wanted} is not a loopback address. The console shows conversation text \
             and has no login; to reach it from another machine, tunnel to it (ssh -L)."
        )
        .into());
    }

    let mut limits = StoreLimits::default();
    if let Some(max) = std::env::var("CONSOLE_MAX_RUNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
    {
        limits.max_runs = max;
    }
    let token = std::env::var("CONSOLE_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());
    let app = AppState::new(Arc::new(Store::new(limits)), token.clone());

    let listener = tokio::net::TcpListener::bind(addrs[0]).await?;
    let port = listener.local_addr()?.port();
    let hosts = ["127.0.0.1", "localhost", "[::1]"]
        .iter()
        .map(|h| format!("{h}:{port}"))
        .collect();
    println!("console: http://127.0.0.1:{port}/");
    println!(
        "ingest:  POST http://127.0.0.1:{port}/ingest ({})",
        if token.is_some() {
            "bearer token required"
        } else {
            "no token"
        }
    );

    axum::serve(listener, router(app, Some(hosts)))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
