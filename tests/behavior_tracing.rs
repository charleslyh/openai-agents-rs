//! Tracing smoke tests.

use std::sync::{Arc, Mutex, OnceLock};

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::tracing::{self, InMemoryProcessor, SpanData};
use openai_agents::{Agent, RunOptions, Runner};

fn tracing_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[tokio::test]
async fn run_emits_agent_and_generation_spans() {
    let _guard = tracing_test_lock().lock().unwrap();
    tracing::set_tracing_disabled(false);
    let proc = InMemoryProcessor::install();

    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let agent = Agent::new("tracer").model(model);
    let _ = Runner::run(&agent, "hi", RunOptions::default())
        .await
        .expect("run");

    let spans = proc.spans.lock().unwrap().clone();
    assert!(
        spans.iter().any(|s| matches!(
            &s.data,
            SpanData::Agent { name } if name == "tracer"
        )),
        "spans={spans:?}"
    );
    assert!(spans
        .iter()
        .any(|s| matches!(&s.data, SpanData::Generation { .. })));
}

/// D-G: processors observe span/trace *start* events, not only completions.
#[tokio::test]
async fn processor_records_start_events() {
    let _guard = tracing_test_lock().lock().unwrap();
    tracing::set_tracing_disabled(false);
    let proc = InMemoryProcessor::install();

    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let agent = Agent::new("starter").model(model);
    let _ = Runner::run(&agent, "hi", RunOptions::default())
        .await
        .expect("run");

    assert!(!proc.started_spans.lock().unwrap().is_empty());
    assert!(!proc.started_traces.lock().unwrap().is_empty());
    // Every finished span must have been started first.
    assert_eq!(
        proc.started_spans.lock().unwrap().len(),
        proc.spans.lock().unwrap().len()
    );
}

/// D-H: `RunConfig.trace_id` / `group_id` reach the trace (Python: `RunConfig.trace_id`).
#[tokio::test]
async fn run_config_injects_trace_identity() {
    let _guard = tracing_test_lock().lock().unwrap();
    tracing::set_tracing_disabled(false);
    let proc = InMemoryProcessor::install();

    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let agent = Agent::new("identified").model(model);
    let mut opts = RunOptions::default();
    opts.run_config.trace_id = Some("trace-custom".into());
    opts.run_config.group_id = Some("thread-7".into());
    opts.run_config.trace_metadata = Some(serde_json::json!({"tenant": "acme"}));
    let _ = Runner::run(&agent, "hi", opts).await.expect("run");

    let traces = proc.traces.lock().unwrap().clone();
    assert_eq!(traces.len(), 1, "traces={traces:?}");
    assert_eq!(traces[0].trace_id, "trace-custom");
    assert_eq!(traces[0].group_id.as_deref(), Some("thread-7"));
    assert_eq!(
        traces[0].metadata,
        Some(serde_json::json!({"tenant": "acme"}))
    );
}

/// D-F: a run that disables tracing records nothing, even when the global switch is on.
#[tokio::test]
async fn run_config_tracing_disabled_is_authoritative() {
    let _guard = tracing_test_lock().lock().unwrap();
    tracing::set_tracing_disabled(false);
    let proc = InMemoryProcessor::install();

    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let agent = Agent::new("private").model(model);
    let mut opts = RunOptions::default();
    opts.run_config.tracing_disabled = true;
    let _ = Runner::run(&agent, "hi", opts).await.expect("run");

    assert!(proc.spans.lock().unwrap().is_empty());
    assert!(proc.started_spans.lock().unwrap().is_empty());
    assert!(proc.traces.lock().unwrap().is_empty());
}

/// D-B: a handoff emits a `Handoff` span carrying both agent names.
#[tokio::test]
async fn handoff_emits_handoff_span() {
    let _guard = tracing_test_lock().lock().unwrap();
    tracing::set_tracing_disabled(false);
    let proc = InMemoryProcessor::install();

    let specialist_model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let specialist = Agent::new("SpanTarget").model(specialist_model);
    let triage_model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call(
            openai_agents::Handoff::default_tool_name("SpanTarget"),
            "{}",
            "h1",
        ),
    )]));
    let triage = Agent::new("SpanSource")
        .model(triage_model)
        .handoffs(vec![openai_agents::handoff(specialist)]);
    let _ = Runner::run(&triage, "go", RunOptions::default())
        .await
        .expect("run");

    let spans = proc.spans.lock().unwrap().clone();
    assert!(
        spans.iter().any(|s| matches!(
            &s.data,
            SpanData::Handoff { from_agent, to_agent }
                if from_agent == "SpanSource" && to_agent.as_deref() == Some("SpanTarget")
        )),
        "spans={spans:?}"
    );
}

/// D-E: `flush_traces` is a real call into the installed processors.
#[tokio::test]
async fn flush_traces_invokes_processors() {
    let _guard = tracing_test_lock().lock().unwrap();
    let proc = Arc::new(CountingProcessor::default());
    tracing::set_trace_processors(vec![proc.clone()]);
    tracing::set_tracing_disabled(false);
    tracing::flush_traces();
    assert_eq!(*proc.flushes.lock().unwrap(), 1);
}

#[derive(Default)]
struct CountingProcessor {
    flushes: Mutex<usize>,
}

impl tracing::TracingProcessor for CountingProcessor {
    fn on_trace_start(&self, _trace: &tracing::Trace) {}
    fn on_trace_end(&self, _trace: &tracing::Trace) {}
    fn on_span_start(&self, _span: &openai_agents::tracing::Span) {}
    fn on_span_end(&self, _span: &openai_agents::tracing::Span) {}
    fn force_flush(&self) {
        *self.flushes.lock().unwrap() += 1;
    }
}

#[tokio::test]
async fn tracing_disabled_skips_processor() {
    let _guard = tracing_test_lock().lock().unwrap();
    let proc = InMemoryProcessor::install();
    tracing::set_tracing_disabled(true);

    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let agent = Agent::new("quiet").model(model);
    let mut opts = RunOptions::default();
    opts.run_config.tracing_disabled = true;
    let _ = Runner::run(&agent, "hi", opts).await.expect("run");

    assert!(proc.spans.lock().unwrap().is_empty());
    tracing::set_tracing_disabled(false);
}
