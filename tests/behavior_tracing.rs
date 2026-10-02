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
