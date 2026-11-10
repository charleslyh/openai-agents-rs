//! `AsToolConfig` context: a nested `Agent::as_tool` run inherits the outer run's user context by
//! default, can be isolated with `inherit_context(false)`, or given its own with `context(..)`.

use std::sync::{Arc, Mutex};

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{Agent, AsToolConfig, FunctionTool, RunOptions, RunState, Runner};

struct Tenant(&'static str);
struct Region(&'static str);

/// What the nested agent's tool saw: tenant id and the address of the `Tenant` value.
type Seen = Arc<Mutex<Vec<Option<(&'static str, usize)>>>>;

fn probe_tool(seen: &Seen, needs_resolution: bool) -> FunctionTool {
    let seen = Arc::clone(seen);
    FunctionTool::new(
        "probe",
        "probe",
        serde_json::json!({"type":"object","properties":{},"additionalProperties":false}),
        move |ctx, _args| {
            let seen = Arc::clone(&seen);
            async move {
                let found = ctx
                    .run_context
                    .context::<Tenant>()
                    .map(|t| (t.0, t as *const Tenant as usize));
                seen.lock().unwrap().push(found);
                Ok(serde_json::json!("probed"))
            }
        },
    )
    .with_needs_resolution(needs_resolution)
}

fn helper(seen: &Seen, needs_resolution: bool) -> Agent {
    Agent::new("helper")
        .model(Arc::new(ScriptedModel::new([
            ModelStep::from(ItemHelpers::function_tool_call("probe", "{}", "n1")),
            ModelStep::from(ItemHelpers::text_message("nested done")),
        ])))
        .tools(vec![probe_tool(seen, needs_resolution)])
}

fn outer(helper: &Agent, config: AsToolConfig) -> Agent {
    let config = AsToolConfig {
        name: Some("call_helper".into()),
        ..config
    };
    Agent::new("boss")
        .model(Arc::new(ScriptedModel::new([
            ModelStep::from(ItemHelpers::function_tool_call(
                "call_helper",
                r#"{"input":"hi"}"#,
                "outer-1",
            )),
            ModelStep::from(ItemHelpers::text_message("outer done")),
        ])))
        .tools(vec![helper.as_tool(config)])
}

fn streaming() -> AsToolConfig {
    AsToolConfig::default().on_stream(|_event| async {})
}

fn tenant_options() -> (RunOptions, usize) {
    let tenant = Arc::new(Tenant("acme"));
    let addr = Arc::as_ptr(&tenant) as usize;
    let mut options = RunOptions::default();
    options.context = Some(tenant);
    (options, addr)
}

async fn run(config: AsToolConfig, options: RunOptions) -> Vec<Option<(&'static str, usize)>> {
    let seen: Seen = Arc::default();
    let boss = outer(&helper(&seen, false), config);
    let result = Runner::run(&boss, "go", options).await.expect("run");
    assert_eq!(result.final_output_as_str(), Some("outer done"));
    let seen = seen.lock().unwrap().clone();
    seen
}

#[tokio::test]
async fn nested_run_inherits_the_outer_context_by_default() {
    let (options, addr) = tenant_options();
    let seen = run(AsToolConfig::default(), options).await;
    assert_eq!(seen, vec![Some(("acme", addr))], "the same value");
}

#[tokio::test]
async fn inherit_context_false_isolates_the_nested_run() {
    let (options, _) = tenant_options();
    let seen = run(AsToolConfig::default().inherit_context(false), options).await;
    assert_eq!(seen, vec![None]);
}

#[tokio::test]
async fn explicit_context_replaces_the_outer_one() {
    let (options, outer_addr) = tenant_options();
    let seen = run(AsToolConfig::default().context(Tenant("other")), options).await;
    assert_eq!(seen.len(), 1);
    let (id, addr) = seen[0].expect("the nested run has a context");
    assert_eq!(id, "other");
    assert_ne!(addr, outer_addr);
}

#[tokio::test]
async fn explicit_context_wins_over_inherit_false_and_hides_outer_values() {
    // Explicit context takes precedence even with inheritance switched off.
    let seen = run(
        AsToolConfig::default()
            .inherit_context(false)
            .context(Tenant("other")),
        RunOptions::default(),
    )
    .await;
    assert_eq!(seen.iter().flatten().next().map(|s| s.0), Some("other"));

    // And nothing from the outer run leaks in next to it.
    let seen: Arc<Mutex<Option<bool>>> = Arc::default();
    let region_seen = Arc::clone(&seen);
    let tool = FunctionTool::new(
        "probe",
        "probe",
        serde_json::json!({"type":"object","properties":{},"additionalProperties":false}),
        move |ctx, _args| {
            let region_seen = Arc::clone(&region_seen);
            async move {
                *region_seen.lock().unwrap() = Some(ctx.run_context.context::<Region>().is_some());
                Ok(serde_json::json!("ok"))
            }
        },
    );
    let helper = Agent::new("helper")
        .model(Arc::new(ScriptedModel::new([
            ModelStep::from(ItemHelpers::function_tool_call("probe", "{}", "n1")),
            ModelStep::from(ItemHelpers::text_message("nested done")),
        ])))
        .tools(vec![tool]);
    let boss = outer(&helper, AsToolConfig::default().context(Tenant("other")));
    Runner::run(&boss, "go", RunOptions::new().with(Region("eu")))
        .await
        .expect("run");
    assert_eq!(*seen.lock().unwrap(), Some(false));
}

#[tokio::test]
async fn every_typed_extension_is_inherited() {
    type Pair = (Option<&'static str>, Option<&'static str>);
    let regions: Arc<Mutex<Vec<Pair>>> = Arc::default();
    let sink = Arc::clone(&regions);
    let tool = FunctionTool::new(
        "probe",
        "probe",
        serde_json::json!({"type":"object","properties":{},"additionalProperties":false}),
        move |ctx, _args| {
            let sink = Arc::clone(&sink);
            async move {
                sink.lock().unwrap().push((
                    ctx.run_context.context::<Tenant>().map(|t| t.0),
                    ctx.run_context.context::<Region>().map(|r| r.0),
                ));
                Ok(serde_json::json!("ok"))
            }
        },
    );
    let helper = Agent::new("helper")
        .model(Arc::new(ScriptedModel::new([
            ModelStep::from(ItemHelpers::function_tool_call("probe", "{}", "n1")),
            ModelStep::from(ItemHelpers::text_message("nested done")),
        ])))
        .tools(vec![tool]);
    let boss = outer(&helper, AsToolConfig::default());
    Runner::run(
        &boss,
        "go",
        RunOptions::new().with(Tenant("acme")).with(Region("eu")),
    )
    .await
    .expect("run");
    assert_eq!(*regions.lock().unwrap(), vec![(Some("acme"), Some("eu"))]);
}

#[tokio::test]
async fn streaming_path_follows_the_same_rules() {
    let (options, addr) = tenant_options();
    assert_eq!(run(streaming(), options).await, vec![Some(("acme", addr))]);

    let (options, _) = tenant_options();
    assert_eq!(
        run(streaming().inherit_context(false), options).await,
        vec![None]
    );

    let (options, _) = tenant_options();
    let seen = run(streaming().context(Tenant("other")), options).await;
    assert_eq!(seen.iter().flatten().next().map(|s| s.0), Some("other"));
}

/// After a nested approval, resuming the outer run (given the same context) still shows the nested
/// tool the inherited context.
#[tokio::test]
async fn resumed_nested_run_still_sees_the_context() {
    for config in [AsToolConfig::default(), streaming()] {
        let seen: Seen = Arc::default();
        let boss = outer(&helper(&seen, true), config);
        let (options, addr) = tenant_options();

        let paused = Runner::run(&boss, "go", options.clone())
            .await
            .expect("run");
        assert!(paused.is_interrupted());
        assert!(seen.lock().unwrap().is_empty(), "not run before approval");

        let mut state = RunState::from_string(&boss.name, &paused.to_state().unwrap().to_string())
            .expect("round trip");
        state.approve(&paused.interruptions[0], false);
        let result = Runner::run_state(&boss, state, options)
            .await
            .expect("resume");
        assert_eq!(result.final_output_as_str(), Some("outer done"));
        assert_eq!(*seen.lock().unwrap(), vec![Some(("acme", addr))]);
    }
}
