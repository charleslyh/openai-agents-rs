//! Minimal Responses API agent example.
//!
//! Required env:
//! - `OPENAI_API_KEY`
//! - `OPENAI_MODEL` (e.g. tuvvi agent-runner `deepseek-v4-pro`)
//!
//! Optional:
//! - `OPENAI_BASE_URL` (OpenAI-compatible `/v1` base, e.g. TokenHub)
//!
//! ```bash
//! # load key/url from tuvvi; model comes from agent-runner settings (not .env)
//! python3 - <<'PY'
//! import os, subprocess
//! from pathlib import Path
//! for line in Path.home().joinpath("Projects/tuvvi/.dev/.env").read_text().splitlines():
//!     s=line.strip()
//!     if s and not s.startswith("#") and "=" in s:
//!         k,v=s.split("=",1); os.environ[k.strip()]=v.strip().strip('"')
//! os.environ["OPENAI_MODEL"]="deepseek-v4-pro"
//! raise SystemExit(subprocess.call(["cargo","run","--example","hello_agent","--features","openai"], env=os.environ))
//! PY
//! ```

use std::sync::Arc;

use openai_agents::{Agent, OpenAIResponsesModel, RunOptions, Runner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY is required")?;
    let model_name = std::env::var("OPENAI_MODEL")
        .map_err(|_| "OPENAI_MODEL is required (e.g. deepseek-v4-pro for TokenHub)")?;
    let base_url = std::env::var("OPENAI_BASE_URL").ok();

    eprintln!(
        "using model={model_name} base={}",
        base_url.as_deref().unwrap_or("(openai default)")
    );

    let model = Arc::new(OpenAIResponsesModel::new(
        model_name,
        api_key,
        base_url.as_deref(),
    ));
    let agent = Agent::new("Assistant")
        .instructions("You are a concise assistant.")
        .model(model);

    let result =
        Runner::run(&agent, "Say hello in one short sentence.", RunOptions::default()).await?;
    println!("{}", result.final_output);
    Ok(())
}
