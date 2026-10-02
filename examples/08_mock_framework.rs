//! 08 — HTTP mock framework (testing)
//!
//! ```bash
//! cargo run --example 08_mock_framework --features testing
//! ```

use std::sync::Arc;
use std::time::Duration;

use openai_agents::testing::{ConcurrentProbe, MockCompletions, MockResponses, MockToolCall};
use openai_agents::{Agent, FunctionTool, RunItemStreamName, RunOptions, Runner, StreamEvent};

fn lookup() -> FunctionTool {
    FunctionTool::new(
        "lookup",
        "look up a key",
        serde_json::json!({
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"]
        }),
        |_ctx, args| async move {
            let v: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
            let key = v.get("key").and_then(|k| k.as_str()).unwrap_or("?");
            Ok(serde_json::Value::String(format!("{key}=42")))
        },
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    demo_responses_loop().await?;
    demo_completions_parallel().await?;
    demo_responses_stream().await?;
    Ok(())
}

async fn demo_responses_loop() -> Result<(), Box<dyn std::error::Error>> {
    let mock = MockResponses::start().await;
    mock.enqueue_tool_calls([MockToolCall::new(
        "lookup",
        r#"{"key":"alpha"}"#,
        "c1",
    )]);
    mock.enqueue_text("alpha is 42");

    let agent = Agent::new("researcher")
        .instructions("Call lookup, then answer.")
        .model(Arc::new(mock.model("mock-responses")))
        .tools(vec![lookup()]);

    let result = Runner::run(&agent, "what is alpha?", RunOptions::default()).await?;
    println!("[responses loop] final={}", result.final_output);
    println!("[responses loop] turns={}", result.raw_responses.len());
    Ok(())
}

async fn demo_completions_parallel() -> Result<(), Box<dyn std::error::Error>> {
    let mock = MockCompletions::start().await;
    mock.enqueue_tool_calls([
        MockToolCall::new("slow", "{}", "s"),
        MockToolCall::new("fast", "{}", "f"),
    ]);
    mock.enqueue_text("merged");

    let probe = ConcurrentProbe::new();
    let agent = Agent::new("worker")
        .model(Arc::new(mock.model("mock-chat")))
        .tools(vec![
            probe.delayed_tool("slow", Duration::from_millis(60), "slow"),
            probe.delayed_tool("fast", Duration::from_millis(5), "fast"),
        ]);

    let started = std::time::Instant::now();
    let result = Runner::run(&agent, "run both", RunOptions::default()).await?;
    println!(
        "[completions parallel] final={} elapsed={:?} max_inflight={}",
        result.final_output,
        started.elapsed(),
        probe.max_inflight()
    );
    Ok(())
}

async fn demo_responses_stream() -> Result<(), Box<dyn std::error::Error>> {
    let mock = MockResponses::start().await;
    mock.enqueue_tool_calls([MockToolCall::new(
        "lookup",
        r#"{"key":"beta"}"#,
        "c1",
    )]);
    mock.enqueue_text("beta is 42");

    let agent = Agent::new("streamer")
        .model(Arc::new(mock.model("mock-responses")))
        .tools(vec![lookup()]);

    let mut streamed = Runner::run_streamed(agent, "beta?", RunOptions::default());
    while let Some(ev) = streamed.next_event().await {
        match ev? {
            StreamEvent::AgentUpdated { agent_name } => {
                println!("[stream] agent={agent_name}");
            }
            StreamEvent::RunItem { name, .. } => {
                println!("[stream] item={}", name.as_str());
            }
            StreamEvent::RawResponse { .. } => {
                println!("[stream] raw_response");
            }
        }
    }
    println!(
        "[stream] complete final={:?} saw_tool_output={}",
        streamed.final_output(),
        true
    );
    let _ = RunItemStreamName::ToolOutput;
    Ok(())
}
