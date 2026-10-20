//! A live console for runs (Rust-only, X-018)
//!
//! Spans say what happened and how long it took; they never hold a prompt, a reply or a tool
//! argument. A console that shows a conversation needs those, so they travel on a channel of their
//! own: a `ConversationHub` that anything can attach to, filled by a `RunHooks` recorder.
//!
//! `ConsoleLink` is one such attachment. It sends the spans and the conversation to `opai-console`,
//! a separate program that draws them in a browser as they happen.
//!
//! ```bash
//! # terminal 1: the console (listens on 127.0.0.1:7077, loopback only)
//! cargo run -p opai-agents-console
//! # terminal 2: a few runs; open http://127.0.0.1:7077/
//! cargo run -p opai-agents-examples --example console
//! ```
//!
//! No key is needed: without `OPENAI_API_KEY` the offline model answers.
//!
//! * `CONSOLE_ADDR` — where the console is. Default `127.0.0.1:7077`.
//! * `CONSOLE_TOKEN` — the bearer token, when the console was started with one.
//! * `CONSOLE_CONTENT=0` — send names, ids and counts only; no prompt, reply or tool payload.
//!
//! To read the conversation without the console, attach your own sink instead:
//! `hub.attach(Arc::new(|event: &ConversationEvent| println!("{}", event.to_json())))`.

mod common;

use std::time::Duration;

use opai_agents::observability::{ConsoleLink, ConversationHub, LinkConfig};
use opai_agents::tracing::set_trace_processors;
use opai_agents::{function_tool, Agent, RunOptions, Runner};

#[function_tool(description = "Get operations metrics for a city")]
async fn get_city_metrics(city: String) -> String {
    tokio::time::sleep(Duration::from_millis(300)).await;
    format!("{city}: 1200 orders today, 98% on time")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = std::env::var("CONSOLE_ADDR").unwrap_or_else(|_| "127.0.0.1:7077".into());
    let token = std::env::var("CONSOLE_TOKEN").ok();
    let content = std::env::var("CONSOLE_CONTENT").map_or(true, |v| v != "0");

    // One hub per process; recorders are per run. `metadata_only` is the switch for text that
    // must not leave the process.
    let hub = if content {
        ConversationHub::new()
    } else {
        ConversationHub::new().metadata_only()
    };
    let link = ConsoleLink::start(LinkConfig::new(addr.clone()).token(token), &hub)?;
    set_trace_processors(vec![link.processor()]);
    println!("sending to the console at http://{addr}/");

    let agent = Agent::new("Ops")
        .instructions("Use get_city_metrics for every city you are asked about.")
        .model(common::model_or_offline()?)
        .tools(vec![get_city_metrics()]);

    for (n, city) in ["Berlin", "Paris"].iter().enumerate() {
        // The run id is both the workflow name (it names the root span, which is how the console
        // groups spans) and the recorder's id (which is how it groups the conversation).
        let run_id = format!("example-{}-{n}", std::process::id());
        let options = RunOptions::new()
            .workflow_name(run_id.clone())
            .hooks(hub.recorder(run_id.clone()));
        let result = Runner::run(
            &agent,
            format!("Give me the city metrics for {city}."),
            options,
        )
        .await?;
        println!(
            "{run_id}> {}",
            result.final_output_as_str().unwrap_or_default()
        );
    }

    // The writers are behind: ask them to catch up before the process ends.
    link.flush();
    println!("sent; {} events could not be delivered", link.dropped());
    Ok(())
}
