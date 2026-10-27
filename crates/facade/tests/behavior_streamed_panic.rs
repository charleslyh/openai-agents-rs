//! A panic inside a streamed run reaches the consumer as an error.
//!
//! The run loop executes in a spawned task. If it panicked and nobody looked, the task would end,
//! the event channel would close, and `collect_events` would return `Ok` for a run that never
//! finished.

use std::sync::Arc;

use async_trait::async_trait;
use opai_agents::{
    Agent, AgentsError, Model, ModelError, ModelRequest, ModelResponse, RunOptions, Runner,
};

/// A model whose call panics.
struct Panics;

#[async_trait]
impl Model for Panics {
    async fn get_response(&self, _request: ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        panic!("model exploded");
    }
}

#[tokio::test]
async fn a_panic_in_a_streamed_run_is_an_error_not_a_clean_end() {
    let agent = Agent::new("assistant").model(Arc::new(Panics));
    let mut streamed = Runner::run_streamed(agent, "hi", RunOptions::default());

    let err = streamed
        .collect_events()
        .await
        .expect_err("a panicking run must not look like a normal end");

    match err {
        AgentsError::Internal { message, .. } => {
            assert!(message.contains("run panicked"), "got: {message}");
            assert!(message.contains("model exploded"), "got: {message}");
        }
        other => panic!("expected an internal error, got {other:?}"),
    }
}

#[tokio::test]
async fn the_snapshot_records_the_panic_and_the_run_is_marked_complete() {
    let agent = Agent::new("assistant").model(Arc::new(Panics));
    let mut streamed = Runner::run_streamed(agent, "hi", RunOptions::default());

    let _ = streamed.collect_events().await;

    assert!(streamed.is_complete());
    assert!(
        streamed.final_output().is_none(),
        "a panicked run has no final output"
    );
    let snapshot = streamed.snapshot.lock().unwrap();
    assert!(
        snapshot
            .error
            .as_deref()
            .is_some_and(|e| e.contains("model exploded")),
        "snapshot.error: {:?}",
        snapshot.error
    );
}
