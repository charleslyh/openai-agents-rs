//! Retrying a failed model call (Rust-only walkthrough of `ModelRetrySettings`)
//!
//! `settings` shows the one-line form (`retry_policies::network_error()`). This example is about
//! the three decisions behind it:
//!
//! * **Whether** to retry: a `RetryPolicy`. The ready-made ones (`network_error`, `http_status`,
//!   `retry_after`, `provider_suggested`, `never`) combine with `any` / `all`. Nothing is retried
//!   unless you give a policy: a bare `max_retries` does not retry on its own.
//! * **How long** to wait: exponential backoff from `ModelRetryBackoffSettings`, unless the
//!   server asked for a delay (`retry-after` / `retry-after-ms`), which wins.
//! * **How often**: `max_retries`; when they run out, the last error is returned.
//!
//! A retried attempt counts as a request in `result.usage` (with zero tokens), so a flaky
//! provider shows up in the numbers.
//!
//! The model is scripted to fail in specific ways, so each case is repeatable and needs no key:
//!
//! 1. A 429 with a `retry-after-ms` header, then a dropped connection, then success.
//! 2. A 400 (a bad request): the policy does not match, so the run fails at once.
//! 3. A dropped connection that never recovers: retries run out and the error comes back.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example model_retry
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{
    retry_policies, Agent, ModelConnectionError, ModelError, ModelRetryBackoffSettings,
    ModelRetrySettings, ModelSettings, ModelStatusError, RunOptions, Runner,
};
use serde_json::json;

// The two helpers below build the errors a provider would raise, so the scenarios are
// repeatable. Retry decisions look at exactly these fields: the status code, the headers
// and the error kind.
fn status(code: u16, headers: &[(&str, &str)]) -> ModelError {
    ModelError::Status(Box::new(ModelStatusError {
        status_code: code,
        message: format!("HTTP {code}"),
        body: json!({"error": {"code": format!("http_{code}")}}),
        headers: headers
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<BTreeMap<_, _>>(),
    }))
}

fn dropped_connection() -> ModelError {
    ModelError::Connection(ModelConnectionError {
        message: "connection reset by peer".into(),
        is_timeout: false,
    })
}

/// Retry rate limits, server errors and network failures, honouring `retry-after`.
fn settings() -> ModelSettings {
    // Policies are small predicates. `any` retries when at least one matches; `all` needs every
    // one. Anything not matched (a 400, say) fails at once, as it should: retrying cannot fix it.
    let policy = retry_policies::any(vec![
        retry_policies::http_status([429, 500, 502, 503]),
        retry_policies::network_error(),
        retry_policies::retry_after(),
    ]);
    // The wait before retry n grows geometrically. A server's `retry-after` always wins over this.
    let backoff = ModelRetryBackoffSettings {
        initial_delay: Some(0.1), // seconds; 0.1, 0.2, 0.4 ... up to `max_delay`
        max_delay: Some(2.0),
        multiplier: Some(2.0),
        jitter: Some(false), // keep the demo's timings readable; leave jitter on in production
    };
    ModelSettings {
        // 3 = retries after the first attempt, so a call is made at most 4 times.
        retry: Some(ModelRetrySettings::new(3, policy).with_backoff(backoff)),
        ..Default::default()
    }
}

async fn scenario(title: &str, steps: Vec<ModelStep>) {
    println!("\n== {title}");
    let model = Arc::new(ScriptedModel::new(steps));
    let agent = Agent::new("Assistant")
        .model(Arc::clone(&model) as Arc<dyn opai_agents::Model>)
        .model_settings(settings());

    let started = Instant::now();
    let outcome = Runner::run(&agent, "Hello!", RunOptions::new()).await;
    let elapsed = started.elapsed();

    println!("model calls : {}", model.calls().len());
    match outcome {
        Ok(result) => {
            println!(
                "answer      : {}",
                result.final_output_as_str().unwrap_or_default()
            );
            println!("usage       : {} requests", result.usage.requests);
        }
        Err(error) => println!("failed with : {error}"),
    }
    println!("took        : {} ms", elapsed.as_millis());
}

#[tokio::main]
async fn main() {
    scenario(
        "429 (asks to wait 300 ms), dropped connection, then success",
        vec![
            ModelStep::raise_model_error(status(429, &[("retry-after-ms", "300")])),
            ModelStep::raise_model_error(dropped_connection()),
            ModelStep::from(ItemHelpers::text_message("Hello! (third attempt)")),
        ],
    )
    .await;

    scenario(
        "400 bad request: not in the policy, so no retry",
        vec![
            ModelStep::raise_model_error(status(400, &[])),
            ModelStep::from(ItemHelpers::text_message("never reached")),
        ],
    )
    .await;

    scenario(
        "the connection never recovers: 3 retries, then the error",
        (0..4)
            .map(|_| ModelStep::raise_model_error(dropped_connection()))
            .collect(),
    )
    .await;
}
