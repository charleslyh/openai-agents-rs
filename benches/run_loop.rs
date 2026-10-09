//! Benchmarks for the pieces a real agent run spends its time on.
//!
//! Run with `cargo bench`. These are not assertions; they exist so a change that doubles the
//! per-turn overhead, the cost of a parallel tool batch or the price of a session write shows up
//! as a number before it shows up as a latency regression in production.
//!
//! Everything here is offline: `ScriptedModel` answers, `InMemorySession` stores, and no HTTP
//! client is involved.

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
use openai_agents::testing::{assistant_message, function_call, ModelStep, ScriptedModel};
use openai_agents::{Agent, FunctionTool, RunOptions, Runner, ToolUseBehavior};
use serde_json::json;
use tokio::runtime::Runtime;

/// One turn, no tools: the floor the runner adds on top of a model call.
fn bench_single_turn(c: &mut Criterion) {
    let rt = Runtime::new().expect("runtime");
    c.bench_function("single_turn_no_tools", |b| {
        b.iter(|| {
            let model = Arc::new(ScriptedModel::new([ModelStep::output(vec![
                assistant_message("done"),
            ])]));
            let agent = Agent::new("a").instructions("be brief").model(model);
            rt.block_on(Runner::run(&agent, "ping", RunOptions::default()))
                .expect("run")
        })
    });
}

/// One tool call, then the answer: tool dispatch, guardrail slots and history bookkeeping.
fn bench_one_tool_then_answer(c: &mut Criterion) {
    let rt = Runtime::new().expect("runtime");
    c.bench_function("one_tool_then_answer", |b| {
        b.iter(|| {
            let model = Arc::new(ScriptedModel::new([
                ModelStep::output(vec![function_call("echo", json!({"n": 1}), "c1")]),
                ModelStep::output(vec![assistant_message("done")]),
            ]));
            let agent = Agent::new("a")
                .model(model)
                .tools(vec![FunctionTool::constant("echo", "echo", "ok")]);
            rt.block_on(Runner::run(&agent, "ping", RunOptions::default()))
                .expect("run")
        })
    });
}

/// A batch of tool calls in one turn: the parallel scheduling path.
fn bench_parallel_tools(c: &mut Criterion) {
    let rt = Runtime::new().expect("runtime");
    let calls: Vec<_> = (0..8)
        .map(|i| function_call("echo", json!({"n": i}), format!("c{i}")))
        .collect();
    c.bench_function("eight_parallel_tools", |b| {
        b.iter(|| {
            let model = Arc::new(ScriptedModel::new([
                ModelStep::output(calls.clone()),
                ModelStep::output(vec![assistant_message("done")]),
            ]));
            let agent = Agent::new("a")
                .model(model)
                .tools(vec![FunctionTool::constant("echo", "echo", "ok")]);
            rt.block_on(Runner::run(&agent, "ping", RunOptions::default()))
                .expect("run")
        })
    });
}

/// `stop_on_first_tool`: how cheap the "stop" paths are compared with `run_llm_again`.
fn bench_stop_on_first_tool(c: &mut Criterion) {
    let rt = Runtime::new().expect("runtime");
    c.bench_function("stop_on_first_tool", |b| {
        b.iter(|| {
            let model = Arc::new(ScriptedModel::new([ModelStep::output(vec![
                function_call("echo", json!({}), "c1"),
            ])]));
            let agent = Agent::new("a")
                .model(model)
                .tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
                .tools(vec![FunctionTool::constant("echo", "echo", "ok")]);
            rt.block_on(Runner::run(&agent, "ping", RunOptions::default()))
                .expect("run")
        })
    });
}

/// Session write + read: what a long conversation pays per turn.
fn bench_session_round_trip(c: &mut Criterion) {
    use openai_agents::{InMemorySession, Session};

    let rt = Runtime::new().expect("runtime");
    let session = InMemorySession::shared("bench");
    c.bench_function("session_add_then_get", |b| {
        b.iter(|| {
            rt.block_on(session.add_items(vec![json!({"role": "user", "content": "hi"})]))
                .expect("add");
            rt.block_on(session.get_items(None)).expect("get")
        })
    });
}

/// Strict schema conversion: runs once per tool, per turn, for every agent.
fn bench_strict_schema(c: &mut Criterion) {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "nested": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}}
            }
        },
        "required": ["name"]
    });
    c.bench_function("ensure_strict_json_schema", |b| {
        b.iter(|| openai_agents::strict_schema::ensure_strict_json_schema(&schema).expect("strict"))
    });
}

criterion_group!(
    benches,
    bench_single_turn,
    bench_one_tool_then_answer,
    bench_parallel_tools,
    bench_stop_on_first_tool,
    bench_session_round_trip,
    bench_strict_schema
);
criterion_main!(benches);
