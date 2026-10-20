//! Rust-only (X-012): guardrail verdicts. `Transform` rewrites the checked content instead of
//! stopping the run; `Fail` carries a reason and a severity that decides whether the rest of the
//! group is cancelled.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{
    input_guardrail, output_guardrail, Agent, AgentsError, FunctionTool, GuardrailFunctionOutput,
    InMemorySession, InputLike, RunOptions, RunState, Runner, Session, Severity, Verdict,
};

fn redact(text: &str) -> String {
    text.replace("4111-1111-1111-1111", "[card]")
}

fn input_text(input: &InputLike) -> String {
    match input {
        InputLike::Text(text) => text.clone(),
        InputLike::Items(items) => serde_json::to_string(items).unwrap_or_default(),
    }
}

fn redactor() -> opai_agents::InputGuardrail {
    input_guardrail("redact", |_ctx, _agent, input| async move {
        GuardrailFunctionOutput::transform(redact(&input_text(&input)))
    })
    .run_in_parallel(false)
}

fn reply(text: &str) -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message(text),
    )]))
}

#[tokio::test]
async fn a_blocking_input_transform_is_what_the_model_and_the_session_see() {
    let model = reply("ok");
    let session = Arc::new(InMemorySession::new("s"));
    let agent = Agent::new("a")
        .model(model.clone())
        .input_guardrails(vec![redactor()]);
    let result = Runner::run(
        &agent,
        "my card is 4111-1111-1111-1111",
        RunOptions::new().session(session.clone()),
    )
    .await
    .expect("run");

    let sent = model.last_call().expect("called").input.to_string();
    assert!(sent.contains("my card is [card]"), "{sent}");
    assert!(!sent.contains("4111"), "the model never sees the original");
    assert_eq!(input_text(&result.input), "my card is [card]");
    let stored = serde_json::to_string(&session.get_items(None).await.unwrap()).unwrap();
    assert!(
        !stored.contains("4111"),
        "the session never stores it: {stored}"
    );
    assert!(matches!(
        result.input_guardrail_results[0].output.verdict,
        Verdict::Transform(_)
    ));
}

#[tokio::test]
async fn a_parallel_input_transform_fails_closed() {
    let agent = Agent::new("a")
        .model(reply("ok"))
        .input_guardrails(vec![redactor().run_in_parallel(true)]);
    let err = Runner::run(&agent, "4111-1111-1111-1111", RunOptions::new())
        .await
        .expect_err("cannot transform what the model already received");
    assert!(matches!(err, AgentsError::User(_)), "{err:?}");
    assert!(err.to_string().contains("run_in_parallel(false)"), "{err}");
}

#[tokio::test]
async fn two_transforms_of_the_same_input_are_ambiguous() {
    let second = input_guardrail("upper", |_ctx, _agent, input| async move {
        GuardrailFunctionOutput::transform(input_text(&input).to_uppercase())
    })
    .run_in_parallel(false);
    let agent = Agent::new("a")
        .model(reply("ok"))
        .input_guardrails(vec![redactor(), second]);
    let err = Runner::run(&agent, "x", RunOptions::new())
        .await
        .expect_err("ambiguous order");
    assert!(err.to_string().contains("combine them"), "{err}");
}

#[tokio::test]
async fn an_output_transform_replaces_the_final_output_only() {
    let guard = output_guardrail("redact", |_ctx, _agent, output| async move {
        GuardrailFunctionOutput::transform(redact(output.as_str().unwrap_or_default()))
    });
    let agent = Agent::new("a")
        .model(reply("charged 4111-1111-1111-1111"))
        .output_guardrails(vec![guard]);
    let result = Runner::run(&agent, "go", RunOptions::new())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("charged [card]"));
    let items = serde_json::to_string(&result.to_input_list()).unwrap();
    assert!(
        items.contains("4111"),
        "the model's own message is kept as written"
    );
}

/// A `High` failure cancels a sibling that would never finish; `trip` is `High`.
#[tokio::test]
async fn a_severe_failure_cancels_the_rest_of_its_group() {
    let never = input_guardrail("never", |_ctx, _agent, _input| async move {
        std::future::pending::<GuardrailFunctionOutput>().await
    })
    .run_in_parallel(false);
    let block = input_guardrail("block", |_ctx, _agent, _input| async move {
        GuardrailFunctionOutput::fail("off-topic", Severity::High)
    })
    .run_in_parallel(false);
    let agent = Agent::new("a")
        .model(reply("ok"))
        .input_guardrails(vec![never, block]);
    let err = tokio::time::timeout(
        Duration::from_secs(5),
        Runner::run(&agent, "go", RunOptions::new()),
    )
    .await
    .expect("the hanging guardrail is cancelled")
    .expect_err("tripped");
    let AgentsError::InputGuardrailTripwire(tripped) = err else {
        panic!("{err:?}");
    };
    assert_eq!(tripped.result.guardrail_name, "block");
    assert_eq!(
        tripped.result.output.verdict,
        Verdict::Fail {
            reason: "off-topic".into(),
            severity: Severity::High
        }
    );
}

/// A `Low` failure still stops the run, but lets the other guardrails finish first.
#[tokio::test]
async fn a_low_failure_lets_the_group_finish() {
    let finished = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&finished);
    let slow = input_guardrail("slow", move |_ctx, _agent, _input| {
        let flag = Arc::clone(&flag);
        async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            flag.store(true, Ordering::SeqCst);
            GuardrailFunctionOutput::pass(serde_json::Value::Null)
        }
    })
    .run_in_parallel(false);
    let nit = input_guardrail("nit", |_ctx, _agent, _input| async move {
        GuardrailFunctionOutput::fail("style", Severity::Low)
    })
    .run_in_parallel(false);
    let agent = Agent::new("a")
        .model(reply("ok"))
        .input_guardrails(vec![slow, nit]);
    let err = Runner::run(&agent, "go", RunOptions::new())
        .await
        .expect_err("a low failure still stops the run");
    assert!(matches!(err, AgentsError::InputGuardrailTripwire(_)));
    assert!(
        finished.load(Ordering::SeqCst),
        "the slow guardrail finished"
    );
}

/// A blocking guardrail that only passes changes nothing, including what the session holds and
/// when it holds it.
#[tokio::test]
async fn a_blocking_guardrail_that_passes_leaves_the_input_save_in_place() {
    let session = Arc::new(InMemorySession::new("s"));
    let agent = Agent::new("a").model(reply("ok")).input_guardrails(vec![
        input_guardrail("noop", |_ctx, _agent, _input| async move {
            GuardrailFunctionOutput::pass(serde_json::Value::Null)
        })
        .run_in_parallel(false),
    ]);
    let result = Runner::run(&agent, "hello", RunOptions::new().session(session.clone()))
        .await
        .expect("run");
    assert_eq!(input_text(&result.input), "hello");
    let stored = serde_json::to_string(&session.get_items(None).await.unwrap()).unwrap();
    assert!(stored.contains("hello"), "{stored}");
}

/// A transform verdict survives a pause and a JSON round trip of the run state.
#[tokio::test]
async fn verdicts_survive_a_run_state_round_trip() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("pay", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("paid")),
    ]));
    let agent = Agent::new("a")
        .model(model)
        .input_guardrails(vec![redactor()])
        .tools(vec![
            FunctionTool::constant("pay", "pay", "ok").with_needs_resolution(true)
        ]);
    let result = Runner::run(&agent, "card 4111-1111-1111-1111", RunOptions::new())
        .await
        .expect("run");
    let json = result.to_state().expect("state").to_json();
    assert_eq!(
        json["input_guardrail_results"][0]["output"]["verdict"]["type"],
        "transform"
    );
    let state = RunState::from_json("a", json).expect("load");
    assert_eq!(
        state.input_guardrail_results[0].output.verdict,
        Verdict::Transform(serde_json::json!("card [card]"))
    );
}
