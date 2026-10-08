//! `Runner::run_blocking` must work whatever the calling thread is doing (D-001).

use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{Agent, AgentsError, RunOptions, Runner};

fn agent(answer: &str) -> Agent {
    Agent::new("a").model(Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message(answer),
    )])))
}

fn run(answer: &str) -> String {
    Runner::run_blocking(&agent(answer), "go", RunOptions::default())
        .expect("run")
        .final_output_as_str()
        .expect("text")
        .to_string()
}

#[test]
fn works_without_any_runtime_and_repeatedly() {
    assert_eq!(run("one"), "one");
    assert_eq!(run("two"), "two");
}

#[test]
fn works_from_many_plain_threads_at_once() {
    let handles: Vec<_> = (0..8)
        .map(|i| std::thread::spawn(move || run(&format!("t{i}"))))
        .collect();
    for (i, handle) in handles.into_iter().enumerate() {
        assert_eq!(handle.join().expect("thread"), format!("t{i}"));
    }
}

/// Used to panic with "Cannot start a runtime from within a runtime".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn works_inside_a_multi_thread_runtime() {
    assert_eq!(run("direct"), "direct", "called straight from async code");
    let from_blocking = tokio::task::spawn_blocking(|| run("blocking")).await.expect("join");
    assert_eq!(from_blocking, "blocking");
}

#[tokio::test]
async fn works_inside_a_current_thread_runtime() {
    assert_eq!(run("current"), "current");
}

#[test]
fn run_blocking_on_uses_the_given_runtime() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let result = Runner::run_blocking_on(
        runtime.handle(),
        &agent("mine"),
        "go",
        RunOptions::default(),
    )
    .expect("run");
    assert_eq!(result.final_output_as_str(), Some("mine"));

    let current = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let err = Runner::run_blocking_on(current.handle(), &agent("x"), "go", RunOptions::default())
        .unwrap_err();
    assert!(matches!(err, AgentsError::User(_)), "{err}");
}

#[test]
fn a_panic_in_the_run_reaches_the_caller() {
    use openai_agents::FunctionTool;
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call("boom", "{}", "c1"),
    )]));
    let agent = Agent::new("a")
        .model(model)
        .tools(vec![FunctionTool::new(
            "boom",
            "panics",
            serde_json::json!({"type": "object", "properties": {}}),
            |_ctx, _args| async move { panic!("tool bug") },
        )]);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = Runner::run_blocking(&agent, "go", RunOptions::default());
    }));
    assert!(caught.is_err());
}
