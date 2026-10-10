//! Runner-managed model retries (Python: `agents.retry`, `run_internal.model_retry`).
//!
//! Waiting is asserted on tokio's paused clock, so these tests are exact and instant.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    retry_policies, Agent, AgentsError, Model, ModelConnectionError, ModelError, ModelResponse,
    ModelRetryAdvice, ModelRetryBackoffSettings, ModelRetrySettings, ModelSettings,
    ModelStatusError, ReplaySafety, RetryDecision, RetryPolicy, RunOptions, Runner,
};
use serde_json::{json, Value};
use tokio::time::Instant;

fn status(code: u16, headers: &[(&str, &str)], body: Value) -> ModelError {
    ModelError::Status(ModelStatusError {
        status_code: code,
        message: format!("status={code} body={body}"),
        body,
        headers: headers
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<BTreeMap<_, _>>(),
    })
}

fn connection(is_timeout: bool) -> ModelError {
    ModelError::Connection(ModelConnectionError {
        message: "connection error".into(),
        is_timeout,
    })
}

fn ok_step() -> ModelStep {
    ModelStep::from(ItemHelpers::text_message("ok"))
}

/// Backoff of 1s, 2s, 4s without jitter.
fn backoff() -> ModelRetryBackoffSettings {
    ModelRetryBackoffSettings {
        initial_delay: Some(1.0),
        max_delay: Some(10.0),
        multiplier: Some(2.0),
        jitter: Some(false),
    }
}

fn agent_with(model: Arc<ScriptedModel>, max_retries: u32, policy: RetryPolicy) -> Agent {
    Agent::new("a").model(model).model_settings(ModelSettings {
        retry: Some(ModelRetrySettings::new(max_retries, policy).with_backoff(backoff())),
        ..Default::default()
    })
}

#[tokio::test(start_paused = true)]
async fn network_errors_are_retried_with_backoff_and_counted_in_usage() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::raise_model_error(connection(false)),
        ModelStep::raise_model_error(connection(true)),
        ok_step(),
    ]));
    let agent = agent_with(model.clone(), 2, retry_policies::network_error());

    let start = Instant::now();
    let result = Runner::run(&agent, "hi", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(
        start.elapsed(),
        Duration::from_secs(3),
        "1s then 2s of backoff"
    );
    assert_eq!(model.calls().len(), 3);
    assert_eq!(result.final_output_as_str(), Some("ok"));
    // Python: every failed attempt is a request with a zero-token entry.
    assert_eq!(result.usage.requests, 3);
    assert_eq!(result.usage.request_usage_entries.len(), 3);
    assert_eq!(result.usage.request_usage_entries[0].total_tokens, 0);
}

#[tokio::test(start_paused = true)]
async fn the_error_is_returned_when_retries_run_out() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::raise_model_error(connection(false)),
        ModelStep::raise_model_error(connection(false)),
        ModelStep::raise_model_error(connection(false)),
    ]));
    let agent = agent_with(model.clone(), 1, retry_policies::network_error());
    let err = Runner::run(&agent, "hi", RunOptions::default())
        .await
        .unwrap_err();
    assert!(
        matches!(err, AgentsError::Model(ModelError::Connection(_))),
        "{err}"
    );
    assert_eq!(model.calls().len(), 2, "one retry, then give up");
}

#[tokio::test(start_paused = true)]
async fn nothing_is_retried_without_settings_or_policy() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::raise_model_error(connection(false)),
        ok_step(),
    ]));
    let agent = Agent::new("a").model(model.clone());
    assert!(Runner::run(&agent, "hi", RunOptions::default())
        .await
        .is_err());
    assert_eq!(model.calls().len(), 1);

    let model = Arc::new(ScriptedModel::new([
        ModelStep::raise_model_error(connection(false)),
        ok_step(),
    ]));
    let agent = Agent::new("a")
        .model(model.clone())
        .model_settings(ModelSettings {
            retry: Some(ModelRetrySettings {
                max_retries: Some(3),
                ..Default::default()
            }),
            ..Default::default()
        });
    assert!(Runner::run(&agent, "hi", RunOptions::default())
        .await
        .is_err());
    assert_eq!(model.calls().len(), 1, "no policy, no retry");
}

#[tokio::test(start_paused = true)]
async fn retry_after_from_the_provider_beats_the_backoff() {
    let policy = retry_policies::any(vec![
        retry_policies::http_status([429]),
        retry_policies::retry_after(),
    ]);
    let model = Arc::new(ScriptedModel::new([
        ModelStep::raise_model_error(status(
            429,
            &[("retry-after", "7")],
            json!({"error": {"code": "rate_limit_exceeded"}}),
        )),
        ModelStep::raise_model_error(status(429, &[("retry-after-ms", "1500")], json!({}))),
        ok_step(),
    ]));
    let agent = agent_with(model, 2, policy);
    let start = Instant::now();
    Runner::run(&agent, "hi", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(start.elapsed(), Duration::from_millis(8500), "7s then 1.5s");
}

#[tokio::test(start_paused = true)]
async fn a_policy_can_choose_the_delay_and_see_the_normalized_error() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let policy = RetryPolicy::new(move |ctx| {
        sink.lock().unwrap().push((
            ctx.attempt,
            ctx.max_retries,
            ctx.normalized.status_code,
            ctx.normalized.error_code.clone(),
            ctx.stream,
        ));
        async { RetryDecision::yes().with_delay(0.5) }
    });
    let model = Arc::new(ScriptedModel::new([
        ModelStep::raise_model_error(status(503, &[], json!({"error": {"code": "overloaded"}}))),
        ok_step(),
    ]));
    let agent = agent_with(model, 3, policy);
    let start = Instant::now();
    Runner::run(&agent, "hi", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(start.elapsed(), Duration::from_millis(500));
    assert_eq!(
        *seen.lock().unwrap(),
        vec![(1, 3, Some(503), Some("overloaded".to_string()), false)]
    );
}

#[tokio::test(start_paused = true)]
async fn stateful_requests_fail_closed_unless_the_provider_says_replay_is_safe() {
    let always = || RetryPolicy::new(|_| async { true });
    let run = |step: ModelStep| async move {
        let model = Arc::new(ScriptedModel::new([step, ok_step()]));
        let agent = agent_with(model.clone(), 2, always());
        let mut options = RunOptions::default();
        options.previous_response_id = Some("resp_prev".into());
        let outcome = Runner::run(&agent, "hi", options).await;
        (outcome.is_ok(), model.calls().len())
    };

    let (ok, calls) = run(ModelStep::raise_model_error(connection(false))).await;
    assert!(
        !ok && calls == 1,
        "a stateful request is not replayed on a plain retry=true"
    );

    let safe =
        ModelStep::raise_model_error(connection(false)).with_retry_advice(ModelRetryAdvice {
            replay_safety: Some(ReplaySafety::Safe),
            ..Default::default()
        });
    let (ok, calls) = run(safe).await;
    assert!(ok && calls == 2, "provider-marked safe replay is allowed");
}

#[tokio::test(start_paused = true)]
async fn provider_unsafe_replay_needs_an_explicit_approval() {
    let unsafe_step = || {
        ModelStep::raise_model_error(connection(false)).with_retry_advice(ModelRetryAdvice {
            suggested: Some(true),
            replay_safety: Some(ReplaySafety::Unsafe),
            ..Default::default()
        })
    };
    let run = |policy: RetryPolicy| async move {
        let model = Arc::new(ScriptedModel::new([unsafe_step(), ok_step()]));
        let agent = agent_with(model.clone(), 2, policy);
        Runner::run(&agent, "hi", RunOptions::default())
            .await
            .is_ok()
    };
    assert!(
        !run(RetryPolicy::new(|_| async { true })).await,
        "plain retry=true is vetoed"
    );
    assert!(
        run(RetryPolicy::new(|_| async {
            RetryDecision::yes().with_approve_unsafe_replay()
        }))
        .await,
        "approve_unsafe_replay lifts the veto"
    );
}

#[tokio::test(start_paused = true)]
async fn provider_suggested_follows_the_advice() {
    let run = |suggested: Option<bool>| async move {
        let step =
            ModelStep::raise_model_error(connection(false)).with_retry_advice(ModelRetryAdvice {
                suggested,
                ..Default::default()
            });
        let model = Arc::new(ScriptedModel::new([step, ok_step()]));
        let agent = agent_with(model, 1, retry_policies::provider_suggested());
        Runner::run(&agent, "hi", RunOptions::default())
            .await
            .is_ok()
    };
    assert!(run(Some(true)).await);
    assert!(!run(Some(false)).await);
    assert!(!run(None).await);
}

#[tokio::test(start_paused = true)]
async fn all_and_any_combine_policies() {
    let run = |policy: RetryPolicy| async move {
        let model = Arc::new(ScriptedModel::new([
            ModelStep::raise_model_error(status(429, &[], json!({}))),
            ok_step(),
        ]));
        Runner::run(&agent_with(model, 1, policy), "hi", RunOptions::default())
            .await
            .is_ok()
    };
    let rate_limited = || retry_policies::http_status([429]);
    let network = || retry_policies::network_error();
    assert!(run(retry_policies::any(vec![network(), rate_limited()])).await);
    assert!(!run(retry_policies::all(vec![network(), rate_limited()])).await);
    assert!(run(retry_policies::all(vec![rate_limited(), rate_limited()])).await);
    assert!(!run(retry_policies::never()).await);
    assert!(!run(retry_policies::any(vec![])).await);
}

#[tokio::test(start_paused = true)]
async fn conversation_locked_is_replayed_without_a_policy_unless_disabled() {
    let locked = || {
        status(
            400,
            &[],
            json!({"error": {"code": "conversation_locked", "message": "busy"}}),
        )
    };
    let model = Arc::new(ScriptedModel::new([
        ModelStep::raise_model_error(locked()),
        ModelStep::raise_model_error(locked()),
        ok_step(),
    ]));
    let start = Instant::now();
    let result = Runner::run(
        &Agent::new("a").model(model.clone()),
        "hi",
        RunOptions::default(),
    )
    .await
    .expect("replayed");
    assert_eq!(start.elapsed(), Duration::from_secs(3), "1s then 2s");
    assert_eq!(result.usage.requests, 3);

    let model = Arc::new(ScriptedModel::new([
        ModelStep::raise_model_error(locked()),
        ok_step(),
    ]));
    let agent = Agent::new("a")
        .model(model.clone())
        .model_settings(ModelSettings {
            retry: Some(ModelRetrySettings {
                max_retries: Some(0),
                ..Default::default()
            }),
            ..Default::default()
        });
    assert!(Runner::run(&agent, "hi", RunOptions::default())
        .await
        .is_err());
    assert_eq!(model.calls().len(), 1, "max_retries=0 opts out");

    // Python gives up after 3 compatibility replays (1s, 2s, 4s).
    let model = Arc::new(ScriptedModel::new(
        (0..5).map(|_| ModelStep::raise_model_error(locked())),
    ));
    let start = Instant::now();
    assert!(Runner::run(
        &Agent::new("a").model(model.clone()),
        "hi",
        RunOptions::default()
    )
    .await
    .is_err());
    assert_eq!(model.calls().len(), 4);
    assert_eq!(start.elapsed(), Duration::from_secs(7));
}

#[tokio::test(start_paused = true)]
async fn streamed_runs_retry_before_any_output_reached_the_consumer() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::raise_model_error(connection(false)),
        ok_step(),
    ]));
    let agent = agent_with(model.clone(), 1, retry_policies::network_error());
    let mut streamed = Runner::run_streamed(agent, "hi", RunOptions::default());
    streamed.collect_events().await.expect("events");
    assert_eq!(model.calls().len(), 2);
    assert_eq!(streamed.final_output(), Some(json!("ok")));
}

/// A model that emits a streamed text delta and then fails.
struct FailsAfterOutput(AtomicUsize);

#[async_trait]
impl Model for FailsAfterOutput {
    async fn get_response(
        &self,
        _request: openai_agents::ModelRequest<'_>,
    ) -> Result<ModelResponse, ModelError> {
        Err(connection(false))
    }

    async fn stream_response(
        &self,
        _request: openai_agents::ModelRequest<'_>,
        raw_tx: tokio::sync::mpsc::Sender<Value>,
    ) -> Result<ModelResponse, ModelError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        let _ = raw_tx.send(json!({"type": "response.created"})).await;
        let _ = raw_tx
            .send(json!({"type": "response.output_text.delta", "delta": "he"}))
            .await;
        Err(connection(false))
    }
}

#[tokio::test(start_paused = true)]
async fn a_stream_that_already_showed_output_is_not_replayed() {
    let model = Arc::new(FailsAfterOutput(AtomicUsize::new(0)));
    let agent = Agent::new("a")
        .model(model.clone())
        .model_settings(ModelSettings {
            retry: Some(ModelRetrySettings::new(
                3,
                RetryPolicy::new(|_| async { true }),
            )),
            ..Default::default()
        });
    let mut streamed = Runner::run_streamed(agent, "hi", RunOptions::default());
    assert!(streamed.collect_events().await.is_err());
    assert_eq!(
        model.0.load(Ordering::SeqCst),
        1,
        "replaying would duplicate the delta"
    );
}

/// A model that never answers.
struct Hangs(AtomicUsize);

#[async_trait]
impl Model for Hangs {
    async fn get_response(
        &self,
        _request: openai_agents::ModelRequest<'_>,
    ) -> Result<ModelResponse, ModelError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        std::future::pending().await
    }
}

#[tokio::test(start_paused = true)]
async fn each_attempt_has_its_own_timeout_and_a_timeout_is_retryable() {
    let model = Arc::new(Hangs(AtomicUsize::new(0)));
    let agent = Agent::new("a")
        .model(model.clone())
        .model_settings(ModelSettings {
            timeout: Some(2.0),
            retry: Some(
                ModelRetrySettings::new(1, retry_policies::network_error()).with_backoff(backoff()),
            ),
            ..Default::default()
        });
    let start = Instant::now();
    let err = Runner::run(&agent, "hi", RunOptions::default())
        .await
        .unwrap_err();
    assert!(
        matches!(err, AgentsError::Model(ModelError::Timeout(ref t)) if t.timeout_seconds == 2.0),
        "{err}"
    );
    assert_eq!(model.0.load(Ordering::SeqCst), 2);
    assert_eq!(
        start.elapsed(),
        Duration::from_secs(2 + 1 + 2),
        "attempt, backoff, attempt"
    );
}

#[test]
fn retry_settings_merge_field_by_field_and_serialize_without_the_policy() {
    let base = ModelSettings {
        retry: Some(ModelRetrySettings {
            max_retries: Some(3),
            backoff: Some(ModelRetryBackoffSettings {
                initial_delay: Some(1.0),
                max_delay: Some(8.0),
                ..Default::default()
            }),
            policy: Some(retry_policies::network_error()),
        }),
        ..Default::default()
    };
    let over = ModelSettings {
        retry: Some(ModelRetrySettings {
            max_retries: None,
            backoff: Some(ModelRetryBackoffSettings {
                max_delay: Some(30.0),
                jitter: Some(false),
                ..Default::default()
            }),
            policy: None,
        }),
        ..Default::default()
    };
    let merged = base.resolve(Some(&over)).retry.expect("retry");
    assert_eq!(merged.max_retries, Some(3));
    assert!(merged.policy.is_some(), "the inherited policy survives");
    let b = merged.backoff.expect("backoff");
    assert_eq!(
        (b.initial_delay, b.max_delay, b.jitter),
        (Some(1.0), Some(30.0), Some(false))
    );

    let json = serde_json::to_value(&merged).unwrap();
    assert_eq!(json["max_retries"], 3);
    assert!(json.get("policy").is_none(), "the policy is runtime-only");
    let back: ModelRetrySettings = serde_json::from_value(json).unwrap();
    assert!(back.policy.is_none());
}
