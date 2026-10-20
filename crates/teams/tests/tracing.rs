mod support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use opai_agents_core::tracing::{Span, Trace, TracingProcessor};
use opai_agents_core::{Runner, Runtime};
use opai_agents_teams::{Role, Team};
use serde_json::json;
use support::{agent, Script, Step};

#[derive(Default)]
struct Recorder {
    traces: Mutex<Vec<Trace>>,
}

impl TracingProcessor for Recorder {
    fn on_trace_start(&self, trace: &Trace) {
        self.traces.lock().unwrap().push(trace.clone());
    }
    fn on_trace_end(&self, _trace: &Trace) {}
    fn on_span_start(&self, _span: &Span) {}
    fn on_span_end(&self, _span: &Span) {}
}

#[tokio::test(flavor = "multi_thread")]
async fn every_run_of_the_team_is_grouped_under_the_team_id() {
    let recorder = Arc::new(Recorder::default());
    let runtime = Runtime::builder().build();
    runtime.set_trace_processors(vec![recorder.clone() as Arc<dyn TracingProcessor>]);

    let worker = Script::new(vec![Step::say("done")]);
    let team = Team::builder("release-crew")
        .role(Role::new("worker", agent("worker", &worker)))
        .runtime(runtime)
        .build()
        .await
        .unwrap();
    let lead_script = Script::new(vec![
        Step::call(
            "team_spawn",
            json!({"role": "worker", "name": "w", "brief": "go"}),
        ),
        Step::call("team_wait", json!({"timeout_ms": 5000})),
        Step::say("finished"),
    ]);
    let lead = agent("lead", &lead_script).tools(team.tools());
    Runner::run(&lead, "start", team.lead_options())
        .await
        .unwrap();
    assert!(team.wait_idle(Duration::from_secs(5)).await);

    let traces = recorder.traces.lock().unwrap().clone();
    assert_eq!(
        traces.len(),
        2,
        "one trace for the lead and one for the teammate run"
    );
    for trace in &traces {
        assert_eq!(trace.workflow_name, "release-crew");
        assert_eq!(trace.group_id.as_deref(), Some(team.id()));
    }
    let members: Vec<_> = traces
        .iter()
        .map(|t| {
            t.metadata.as_ref().unwrap()["member"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    assert!(members.contains(&"lead".to_string()), "{members:?}");
    assert!(members.contains(&"w".to_string()), "{members:?}");
    let teammate = traces
        .iter()
        .find(|t| t.metadata.as_ref().unwrap()["member"] == "w")
        .unwrap();
    assert_eq!(teammate.metadata.as_ref().unwrap()["role"], "worker");
}
