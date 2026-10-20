//! Rust-only (X-022): a run on its own `Runtime` shares no tracing or handoff state with others.
//!
//! No test here takes a lock or resets global state: runs on private runtimes cannot disturb each
//! other, which is the point. The one global processor installed is only read to prove it stays
//! empty.

use std::sync::Arc;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::tracing::{InMemoryProcessor, SpanData};
use opai_agents::{
    handoff, input_guardrail, Agent, GuardrailFunctionOutput, RunConfig, RunOptions, Runner,
    Runtime,
};
use serde_json::json;

fn install_on(runtime: &Runtime) -> Arc<InMemoryProcessor> {
    let proc = Arc::new(InMemoryProcessor::default());
    runtime.set_trace_processors(vec![proc.clone()]);
    proc
}

fn agent(name: &str) -> Agent {
    Agent::new(name).model(Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )])))
}

fn agent_span_names(proc: &InMemoryProcessor) -> Vec<String> {
    proc.spans
        .lock()
        .unwrap()
        .iter()
        .filter_map(|s| match &s.data {
            SpanData::Agent { name } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn each_runtime_sees_only_its_own_runs() {
    let global = InMemoryProcessor::install();
    let (rt_a, rt_b) = (Runtime::new(), Runtime::new());
    let (proc_a, proc_b) = (install_on(&rt_a), install_on(&rt_b));

    let (agent_one, agent_two) = (agent("one"), agent("two"));
    let (one, two) = tokio::join!(
        Runner::run(&agent_one, "hi", RunOptions::new().runtime(rt_a.clone())),
        Runner::run(&agent_two, "hi", RunOptions::new().runtime(rt_b.clone())),
    );
    one.expect("one");
    two.expect("two");

    assert_eq!(agent_span_names(&proc_a), ["one"]);
    assert_eq!(agent_span_names(&proc_b), ["two"]);
    assert!(
        global.spans.lock().unwrap().is_empty() && global.started_traces.lock().unwrap().is_empty(),
        "a run on a private runtime must not reach the global processor"
    );
}

#[tokio::test]
async fn disabling_tracing_on_one_runtime_leaves_the_other() {
    let (off, on) = (Runtime::new(), Runtime::new());
    let (proc_off, proc_on) = (install_on(&off), install_on(&on));
    off.set_tracing_disabled(true);

    for (rt, name) in [(&off, "off"), (&on, "on")] {
        Runner::run(&agent(name), "hi", RunOptions::new().runtime(rt.clone()))
            .await
            .expect("run");
    }
    assert!(agent_span_names(&proc_off).is_empty());
    assert_eq!(agent_span_names(&proc_on), ["on"]);
    assert!(!on.tracing_disabled());
}

#[tokio::test]
async fn a_streamed_run_keeps_its_runtime_across_the_spawn() {
    let rt = Runtime::new();
    let proc = install_on(&rt);
    let mut streamed = Runner::run_streamed(
        agent("streamed"),
        "hi",
        RunOptions::new().runtime(rt.clone()),
    );
    streamed.collect_events().await.expect("events");

    assert_eq!(agent_span_names(&proc), ["streamed"]);
}

/// A parallel input guardrail runs in a task of its own; its span still goes to the run's runtime.
#[tokio::test]
async fn a_spawned_guardrail_reports_to_the_runs_runtime() {
    let global = InMemoryProcessor::install();
    let rt = Runtime::new();
    let proc = install_on(&rt);
    let guarded = agent("guarded").input_guardrails(vec![input_guardrail(
        "always-ok",
        |_ctx, _agent, _input| async { GuardrailFunctionOutput::pass(json!({})) },
    )]);

    Runner::run(&guarded, "hi", RunOptions::new().runtime(rt))
        .await
        .expect("run");

    let guardrail_spans = |p: &InMemoryProcessor| {
        p.spans
            .lock()
            .unwrap()
            .iter()
            .filter(|s| matches!(&s.data, SpanData::Guardrail { .. }))
            .count()
    };
    assert_eq!(guardrail_spans(&proc), 1);
    assert_eq!(guardrail_spans(&global), 0);
}

/// The markers around a nested handoff summary belong to the runtime, not the process.
#[tokio::test]
async fn handoff_history_markers_are_per_runtime() {
    async fn run_with(markers: Option<(&'static str, &'static str)>) -> String {
        let b_model = Arc::new(ScriptedModel::new([ModelStep::from(
            ItemHelpers::text_message("ok"),
        )]));
        let b = Agent::new("B").model(b_model.clone());
        let a = Agent::new("A")
            .model(Arc::new(ScriptedModel::new([ModelStep::from(
                ItemHelpers::function_tool_call("transfer_to_b", "{}", "h1"),
            )])))
            .handoffs(vec![handoff(b)]);
        let runtime = Runtime::new();
        if let Some((start, end)) = markers {
            runtime.set_conversation_history_wrappers(Some(start), Some(end));
        }
        let mut config = RunConfig::default();
        config.handoff.nest_history = true;
        Runner::run(
            &a,
            "hello",
            RunOptions::new().config(config).runtime(runtime),
        )
        .await
        .expect("run");
        b_model.calls()[0].input.to_string()
    }

    let (custom, default) = tokio::join!(run_with(Some(("<<S>>", "<<E>>"))), run_with(None));
    assert!(
        custom.contains("<<S>>") && custom.contains("<<E>>"),
        "{custom}"
    );
    assert!(!custom.contains("<CONVERSATION HISTORY>"), "{custom}");
    assert!(default.contains("<CONVERSATION HISTORY>"), "{default}");
    assert!(!default.contains("<<S>>"), "{default}");
}

#[test]
fn a_runtime_starts_with_the_defaults() {
    let rt = Runtime::new();
    assert!(!rt.tracing_disabled());
    assert_eq!(
        rt.conversation_history_wrappers(),
        (
            "<CONVERSATION HISTORY>".to_string(),
            "</CONVERSATION HISTORY>".to_string()
        )
    );
    rt.set_conversation_history_wrappers(Some("<a>"), None);
    assert_eq!(
        rt.conversation_history_wrappers().1,
        "</CONVERSATION HISTORY>"
    );
    rt.reset_conversation_history_wrappers();
    assert_eq!(
        rt.conversation_history_wrappers().0,
        "<CONVERSATION HISTORY>"
    );
}
