//! Cooperative cancellation: a cancelled run stops at the next safe point, keeps the outputs of
//! tool calls that were already running, and marks calls that never started as cancelled.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{
    Agent, CancellationToken, FunctionTool, RunConfig, RunItem, RunOptions, Runner,
    ToolExecutionConfig,
};
use serde_json::{json, Value};

fn empty_schema() -> Value {
    json!({"type": "object", "properties": {}, "additionalProperties": false})
}

fn output_texts(items: &[RunItem]) -> Vec<String> {
    items
        .iter()
        .filter_map(|item| match item {
            RunItem::ToolCallOutput(o) => o.raw_item["output"].as_str().map(str::to_string),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_token_cancelled_before_the_run_ends_it_without_calling_the_model() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("never"),
    )]));
    let agent = Agent::new("a").model(model.clone());
    let token = CancellationToken::new();
    token.cancel();

    let result = Runner::run(&agent, "go", RunOptions::new().cancellation_token(token))
        .await
        .expect("a cancelled run is not an error");

    assert!(result.is_cancelled);
    assert_eq!(result.final_output, Value::Null);
    assert_eq!(model.calls().len(), 0);
}

#[tokio::test]
async fn cancelling_during_a_model_call_drops_the_call() {
    let model = Arc::new(ScriptedModel::new([ModelStep::hang()]));
    let agent = Agent::new("a").model(model.clone());
    let token = CancellationToken::new();

    let canceller = {
        let token = token.clone();
        let model = Arc::clone(&model);
        tokio::spawn(async move {
            while model.calls().is_empty() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            token.cancel();
        })
    };
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        Runner::run(&agent, "go", RunOptions::new().cancellation_token(token)),
    )
    .await
    .expect("cancelling must free a run stuck in the model call")
    .expect("a cancelled run is not an error");
    canceller.await.unwrap();

    assert!(result.is_cancelled);
}

#[tokio::test]
async fn a_running_tool_finishes_and_its_output_is_recorded() {
    let token = CancellationToken::new();
    let saw_cancel = Arc::new(AtomicBool::new(false));
    let tool = {
        let saw_cancel = Arc::clone(&saw_cancel);
        FunctionTool::new("slow", "slow", empty_schema(), move |ctx, _args| {
            let saw_cancel = Arc::clone(&saw_cancel);
            async move {
                // The tool is told about the cancellation and chooses how to react to it.
                ctx.cancellation().cancelled().await;
                saw_cancel.store(true, Ordering::SeqCst);
                Ok(json!("partial result"))
            }
        })
    };
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("slow", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("never")),
    ]));
    let agent = Agent::new("a").model(model.clone()).tools(vec![tool]);

    let canceller = {
        let token = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            token.cancel();
        })
    };
    let result = Runner::run(&agent, "go", RunOptions::new().cancellation_token(token))
        .await
        .expect("a cancelled run is not an error");
    canceller.await.unwrap();

    assert!(result.is_cancelled);
    assert!(saw_cancel.load(Ordering::SeqCst));
    assert_eq!(output_texts(&result.new_items), vec!["partial result"]);
    // The turn after the tool never started.
    assert_eq!(model.calls().len(), 1);
}

#[tokio::test]
async fn calls_that_never_started_get_a_cancelled_output() {
    let token = CancellationToken::new();
    let started = Arc::new(AtomicUsize::new(0));
    let tool = |name: &'static str| {
        let (token, started) = (token.clone(), Arc::clone(&started));
        FunctionTool::new(name, name, empty_schema(), move |_ctx, _args| {
            let (token, started) = (token.clone(), Arc::clone(&started));
            async move {
                started.fetch_add(1, Ordering::SeqCst);
                // The first call to run cancels the run; the cap lets only one run at a time.
                token.cancel();
                Ok(json!(name))
            }
        })
    };
    let model = Arc::new(ScriptedModel::new([ModelStep::from(vec![
        ItemHelpers::function_tool_call("a", "{}", "c1"),
        ItemHelpers::function_tool_call("b", "{}", "c2"),
    ])]));
    let agent = Agent::new("x")
        .model(model)
        .tools(vec![tool("a"), tool("b")]);
    let mut config = RunConfig::default();
    config.tools.execution = Some(ToolExecutionConfig {
        max_function_tool_concurrency: Some(1),
    });
    let mut options = RunOptions::new().cancellation_token(token);
    options.run_config = config;

    let result = Runner::run(&agent, "go", options)
        .await
        .expect("a cancelled run is not an error");

    assert!(result.is_cancelled);
    assert_eq!(started.load(Ordering::SeqCst), 1);
    let texts = output_texts(&result.new_items);
    assert_eq!(texts.len(), 2, "every call has an output: {texts:?}");
    assert_eq!(texts[0], "a");
    assert!(texts[1].to_lowercase().contains("cancel"), "{texts:?}");
}

#[tokio::test]
async fn a_run_without_a_token_is_not_cancelled() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("done"),
    )]));
    let agent = Agent::new("a").model(model);

    let result = Runner::run(&agent, "go", RunOptions::default())
        .await
        .unwrap();

    assert!(!result.is_cancelled);
    assert_eq!(result.final_output_as_str(), Some("done"));
}

/// Held by a resolver while it waits; dropping it is how a pending prompt would be released.
struct Prompt(Arc<AtomicBool>);

impl Drop for Prompt {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn cancelling_while_a_resolver_waits_releases_what_it_holds() {
    let released = Arc::new(AtomicBool::new(false));
    let token = CancellationToken::new();
    let tool = FunctionTool::new(
        "danger",
        "needs a person",
        empty_schema(),
        |_ctx, _args| async { Ok(json!("ran")) },
    )
    .with_needs_resolution(true);
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call("danger", "{}", "c1"),
    )]));
    let agent = Agent::new("a").model(model).tools(vec![tool]);

    let options = {
        let (released, token) = (Arc::clone(&released), token.clone());
        RunOptions::new()
            .cancellation_token(token.clone())
            .resolve_with(move |_call| {
                let (prompt, token) = (Prompt(Arc::clone(&released)), token.clone());
                async move {
                    let _prompt = prompt;
                    // Nobody ever answers; the cancel is what ends the wait.
                    token.cancel();
                    std::future::pending().await
                }
            })
    };
    let result = tokio::time::timeout(Duration::from_secs(5), Runner::run(&agent, "go", options))
        .await
        .expect("a cancel must end the wait for an answer")
        .expect("a cancelled run is not an error");

    assert!(result.is_cancelled);
    assert!(result.interruptions.is_empty());
    assert!(
        released.load(Ordering::SeqCst),
        "the prompt was left behind"
    );
}

#[tokio::test]
async fn an_immediate_cancel_of_a_streamed_run_reaches_work_a_tool_started_elsewhere() {
    let told = Arc::new(AtomicBool::new(false));
    let tool = {
        let told = Arc::clone(&told);
        FunctionTool::new(
            "spawn",
            "detaches work",
            empty_schema(),
            move |ctx, _args| {
                let (told, token) = (Arc::clone(&told), ctx.cancellation().clone());
                async move {
                    // Work on another task survives the abort of the run's own task; only the token
                    // can stop it.
                    tokio::spawn(async move {
                        token.cancelled().await;
                        told.store(true, Ordering::SeqCst);
                    });
                    std::future::pending::<()>().await;
                    Ok(json!("unreachable"))
                }
            },
        )
    };
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call("spawn", "{}", "c1"),
    )]));
    let agent = Agent::new("a").model(model.clone()).tools(vec![tool]);

    let mut stream = Runner::run_streamed(agent, "go", RunOptions::default());
    while model.calls().is_empty() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    tokio::time::sleep(Duration::from_millis(30)).await;
    stream.cancel(opai_agents::CancelMode::Immediate);

    assert!(stream.is_cancelled());
    for _ in 0..200 {
        if told.load(Ordering::SeqCst) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("the detached work never learned of the cancel");
}
