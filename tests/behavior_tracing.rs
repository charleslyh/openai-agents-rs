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

/// Guardrails run on a spawned task, so spans they open must still honor `RunConfig.tracing_disabled`
/// (task-locals are not inherited by `tokio::spawn`).
#[tokio::test]
async fn spawned_input_guardrail_respects_run_tracing_disabled() {
    let _guard = tracing_test_lock().lock().unwrap();
    tracing::set_tracing_disabled(false);
    let proc = InMemoryProcessor::install();

    let spanned = |_ctx, _agent, _input| async {
        let _span = tracing::guardrail_span("spanned");
        openai_agents::GuardrailFunctionOutput::pass(serde_json::json!({}))
    };

    // Tracing on: the span opened inside the spawned guardrail task is recorded.
    let on_model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let on_agent = Agent::new("guarded")
        .model(on_model)
        .input_guardrails(vec![openai_agents::input_guardrail("spanned", spanned)]);
    let _ = Runner::run(&on_agent, "hi", RunOptions::default())
        .await
        .expect("run");
    assert!(
        proc.spans
            .lock()
            .unwrap()
            .iter()
            .any(|s| matches!(&s.data, SpanData::Guardrail { name } if name == "spanned")),
        "guardrail span should be recorded when tracing is enabled"
    );

    // Tracing off for this run: the same span must not reach the processor.
    let off_model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let off_agent = Agent::new("guarded")
        .model(off_model)
        .input_guardrails(vec![openai_agents::input_guardrail("spanned", spanned)]);
    let mut opts = RunOptions::default();
    opts.run_config.tracing_disabled = true;
    let before_spans = proc.spans.lock().unwrap().len();
    let before_started = proc.started_spans.lock().unwrap().len();
    let _ = Runner::run(&off_agent, "hi", opts).await.expect("run");

    assert_eq!(
        proc.spans.lock().unwrap().len(),
        before_spans,
        "guardrail span bypassed RunConfig.tracing_disabled"
    );
    assert_eq!(
        proc.started_spans.lock().unwrap().len(),
        before_started,
        "guardrail span start bypassed RunConfig.tracing_disabled"
    );
}

/// Python wraps guardrail bodies in `guardrail_span`; both directions must appear in the trace
/// (input guardrails run on a spawned task, output guardrails inline at finalize time).
#[tokio::test]
async fn guardrails_emit_guardrail_spans() {
    let _guard = tracing_test_lock().lock().unwrap();
    tracing::set_tracing_disabled(false);
    let proc = InMemoryProcessor::install();

    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let agent = Agent::new("guarded")
        .model(model)
        .input_guardrails(vec![openai_agents::input_guardrail(
            "in-check",
            |_ctx, _agent, _input| async {
                openai_agents::GuardrailFunctionOutput::pass(serde_json::json!({}))
            },
        )])
        .output_guardrails(vec![openai_agents::output_guardrail(
            "out-check",
            |_ctx, _agent, _out| async {
                openai_agents::GuardrailFunctionOutput::pass(serde_json::json!({}))
            },
        )]);
    let _ = Runner::run(&agent, "hi", RunOptions::default())
        .await
        .expect("run");

    let spans = proc.spans.lock().unwrap().clone();
    assert!(
        spans
            .iter()
            .any(|s| matches!(&s.data, SpanData::Guardrail { name } if name == "in-check")),
        "spans={spans:?}"
    );
    assert!(
        spans
            .iter()
            .any(|s| matches!(&s.data, SpanData::Guardrail { name } if name == "out-check")),
        "spans={spans:?}"
    );
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

/// Describe the span tree in start order as `kind(detail)<-parent_index` (`-` for the trace root),
/// the same shape the Python SDK produces for the same scenarios.
fn span_tree(proc: &InMemoryProcessor) -> Vec<String> {
    // The scripted model emits no generation span in Python, so leave them out of the shape.
    let spans: Vec<_> = proc
        .started_spans
        .lock()
        .unwrap()
        .iter()
        .filter(|s| !matches!(s.data, SpanData::Generation { .. }))
        .cloned()
        .collect();
    let index = |id: &Option<String>| match id {
        None => "-".to_string(),
        Some(id) => spans.iter().position(|s| &s.span_id == id).map_or("?".into(), |i| i.to_string()),
    };
    spans
        .iter()
        .map(|s| {
            let what = match &s.data {
                SpanData::Task { .. } => "task".to_string(),
                SpanData::Agent { name } => format!("agent({name})"),
                SpanData::Turn { turn, agent_name, .. } => format!("turn({turn},{agent_name})"),
                SpanData::Function { name } => format!("function({name})"),
                SpanData::Handoff { .. } => "handoff".to_string(),
                SpanData::Guardrail { name } => format!("guardrail({name})"),
                SpanData::Generation { .. } => "generation".to_string(),
                other => format!("{other:?}"),
            };
            format!("{what}<-{}", index(&s.parent_id))
        })
        .collect()
}

/// D-004: task / turn spans nest agent, function, handoff and guardrail spans like Python; the
/// agent span covers all turns of one agent and is replaced on handoff.
#[tokio::test]
async fn task_and_turn_spans_nest_like_python() {
    use openai_agents::{GuardrailFunctionOutput, InputGuardrail, OutputGuardrail, TracingConfig};
    let _guard = tracing_test_lock().lock().unwrap();
    tracing::set_tracing_disabled(false);

    let build = || {
        let b = Agent::new("B").model(Arc::new(ScriptedModel::new([ModelStep::from(
            ItemHelpers::text_message("b done"),
        )])));
        Agent::new("A")
            .model(Arc::new(ScriptedModel::new([
                ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c1")),
                ModelStep::from(ItemHelpers::function_tool_call(
                    openai_agents::Handoff::default_tool_name("B"),
                    "{}",
                    "h1",
                )),
            ])))
            .tools(vec![openai_agents::FunctionTool::constant("echo", "e", "x")])
            .handoffs(vec![openai_agents::handoff(b)])
    };

    let proc = InMemoryProcessor::install();
    Runner::run(&build(), "go", RunOptions::default()).await.expect("run");
    // Python: task, agent A under task, turn 1 with its function, turn 2 with the handoff,
    // agent B back under the task.
    assert_eq!(
        span_tree(&proc),
        [
            "task<--",
            "agent(A)<-0",
            "turn(1,A)<-1",
            "function(echo)<-2",
            "turn(2,A)<-1",
            "handoff<-4",
            "agent(B)<-0",
            "turn(3,B)<-6",
        ]
    );
    let finished = proc.spans.lock().unwrap().clone();
    let task = finished.iter().find(|s| matches!(s.data, SpanData::Task { .. })).unwrap();
    match &task.data {
        SpanData::Task { name, usage } => {
            assert_eq!(name, "Agent workflow");
            assert_eq!(usage.as_ref().unwrap()["requests"], 3);
        }
        _ => unreachable!(),
    }
    let turn = finished.iter().find(|s| matches!(s.data, SpanData::Turn { .. })).unwrap();
    assert!(matches!(&turn.data, SpanData::Turn { usage: Some(u), .. } if u.get("requests").is_none()));

    // `include_task_and_turn_spans: false` drops both layers and re-parents the rest.
    let proc = InMemoryProcessor::install();
    let mut options = RunOptions::default();
    options.run_config.tracing = Some(TracingConfig {
        include_task_and_turn_spans: Some(false),
        ..TracingConfig::default()
    });
    Runner::run(&build(), "go", options).await.expect("run");
    assert_eq!(
        span_tree(&proc),
        ["agent(A)<--", "function(echo)<-0", "handoff<-0", "agent(B)<--"]
    );

    // Input guardrails run inside the first turn, output guardrails after it, under the agent.
    let proc = InMemoryProcessor::install();
    let ok = || GuardrailFunctionOutput::pass(serde_json::Value::Null);
    let agent = Agent::new("A")
        .model(Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message("hi"))])))
        .input_guardrails(vec![
            InputGuardrail::new("par", move |_c, _a, _i| async move { ok() }),
            InputGuardrail::new("blk", move |_c, _a, _i| async move { ok() }).run_in_parallel(false),
        ])
        .output_guardrails(vec![OutputGuardrail::new("out", move |_c, _a, _o| async move { ok() })]);
    Runner::run(&agent, "go", RunOptions::default()).await.expect("run");
    assert_eq!(
        span_tree(&proc),
        [
            "task<--",
            "agent(A)<-0",
            "turn(1,A)<-1",
            "guardrail(blk)<-2",
            "guardrail(par)<-2",
            "guardrail(out)<-1",
        ]
    );
}

/// D-039: the summary call a `CompactingSession` makes while a run saves its turn is traced
/// under that run, and its usage is readable from the session.
#[tokio::test]
async fn compaction_summary_is_a_span_of_the_run_that_triggered_it() {
    use openai_agents::{CompactingSession, InMemorySession, ModelSummarizer, Session};
    let _guard = tracing_test_lock().lock().unwrap();
    tracing::set_tracing_disabled(false);

    let summary_model = Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message("notes"))]));
    let session = Arc::new(
        CompactingSession::new(
            InMemorySession::shared("s"),
            Arc::new(ModelSummarizer::new(summary_model)),
        )
        .trigger_items(3)
        .keep_recent_turns(1),
    );
    session
        .add_items(vec![
            serde_json::json!({"role": "user", "content": "t1"}),
            serde_json::json!({"role": "assistant", "content": "a1"}),
            serde_json::json!({"role": "user", "content": "t2"}),
        ])
        .await
        .unwrap();

    let proc = InMemoryProcessor::install();
    let agent = Agent::new("a")
        .model(Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message("hi"))])));
    let mut options = RunOptions::default();
    options.session = Some(session.clone());
    Runner::run(&agent, "t3", options).await.expect("run");

    let started = proc.started_spans.lock().unwrap().clone();
    let summary = started
        .iter()
        .find(|s| matches!(&s.data, SpanData::Custom { name } if name == "conversation_summary"))
        .expect("summary span");
    assert!(summary.parent_id.is_some(), "it hangs under the run, not at the trace root");
    assert_eq!(session.summarizer_usage().requests, 1);
}

/// D-037: a run started by an agent used as a tool is a child of that tool call's function span,
/// with its own task, agent and turn spans (same tree as Python).
#[tokio::test]
async fn agent_as_tool_run_nests_under_the_function_span() {
    let _guard = tracing_test_lock().lock().unwrap();
    tracing::set_tracing_disabled(false);
    let inner = Agent::new("Inner")
        .model(Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message("inner done"))])));
    let outer = Agent::new("Outer")
        .model(Arc::new(ScriptedModel::new([
            ModelStep::from(ItemHelpers::function_tool_call("ask_inner", r#"{"input":"hi"}"#, "c1")),
            ModelStep::from(ItemHelpers::text_message("outer done")),
        ])))
        .tools(vec![inner.as_tool(openai_agents::AsToolConfig {
            name: Some("ask_inner".into()),
            description: Some("d".into()),
            ..Default::default()
        })]);

    let proc = InMemoryProcessor::install();
    Runner::run(&outer, "go", RunOptions::default()).await.expect("run");
    assert_eq!(
        span_tree(&proc),
        [
            "task<--",
            "agent(Outer)<-0",
            "turn(1,Outer)<-1",
            "function(ask_inner)<-2",
            "task<-3",
            "agent(Inner)<-4",
            "turn(1,Inner)<-5",
            "turn(2,Outer)<-1",
        ]
    );
}

/// Tools of one turn run at the same time; each nested run must hang under its own function span.
#[tokio::test]
async fn parallel_agent_tools_keep_their_own_parents() {
    let _guard = tracing_test_lock().lock().unwrap();
    tracing::set_tracing_disabled(false);
    let helper = |name: &str| {
        Agent::new(name).model(Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message("ok"))])))
    };
    let tool = |agent: Agent, name: &str| {
        agent.as_tool(openai_agents::AsToolConfig {
            name: Some(name.into()),
            description: Some("d".into()),
            ..Default::default()
        })
    };
    let outer = Agent::new("Outer")
        .model(Arc::new(ScriptedModel::new([
            ModelStep::output([
                ItemHelpers::function_tool_call("ask_a", r#"{"input":"1"}"#, "c1"),
                ItemHelpers::function_tool_call("ask_b", r#"{"input":"2"}"#, "c2"),
            ]),
            ModelStep::from(ItemHelpers::text_message("done")),
        ])))
        .tools(vec![tool(helper("A"), "ask_a"), tool(helper("B"), "ask_b")]);

    let proc = InMemoryProcessor::install();
    Runner::run(&outer, "go", RunOptions::default()).await.expect("run");
    let spans = proc.started_spans.lock().unwrap().clone();
    // agent span -> its task span -> the function span the task hangs under.
    let nested_parent = |agent: &str| {
        spans
            .iter()
            .position(|s| matches!(&s.data, SpanData::Agent { name } if name == agent))
            .and_then(|i| spans[i].parent_id.clone())
            .and_then(|id| spans.iter().find(|s| s.span_id == id))
            .and_then(|task| spans.iter().find(|s| Some(&s.span_id) == task.parent_id.as_ref()))
            .map(|p| p.data.clone())
    };
    assert!(matches!(nested_parent("A"), Some(SpanData::Function { name }) if name == "ask_a"));
    assert!(matches!(nested_parent("B"), Some(SpanData::Function { name }) if name == "ask_b"));
}
