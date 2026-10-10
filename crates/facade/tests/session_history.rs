//! What the runner sends to the model and saves when a session has history (D-027). The
//! expected values were produced by the Python SDK's `SQLiteSession` runs of the same scenarios.

use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    Agent, InMemorySession, RunConfig, RunOptions, Runner, Session, SessionInputCallback,
    SessionSettings,
};
use serde_json::{json, Value};

/// `(type or role, string content)` of each item, the shape the Python runs were printed in.
fn short(items: &Value) -> Vec<(String, Option<String>)> {
    items
        .as_array()
        .unwrap()
        .iter()
        .map(|i| {
            (
                i["type"]
                    .as_str()
                    .or(i["role"].as_str())
                    .unwrap()
                    .to_string(),
                i["content"].as_str().map(str::to_string),
            )
        })
        .collect()
}

fn item(kind: &str, content: Option<&str>) -> (String, Option<String>) {
    (kind.to_string(), content.map(str::to_string))
}

fn call() -> Value {
    json!({"id": "fc1", "type": "function_call", "name": "t", "arguments": "{}", "call_id": "c1"})
}

fn output() -> Value {
    json!({"type": "function_call_output", "call_id": "c1", "output": "x"})
}

/// Run "next" over `history` and return the model's input and the session afterwards.
async fn run(history: Vec<Value>, config: RunConfig) -> (Value, Vec<Value>) {
    let session = InMemorySession::shared("s");
    session.add_items(history).await.unwrap();
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("reply"),
    )]));
    let agent = Agent::new("a").model(model.clone());
    let mut options = RunOptions::default();
    options.session = Some(session.clone());
    options.run_config = config;
    Runner::run(&agent, "next", options).await.expect("run");
    let input = model.calls()[0].input.clone();
    (input, session.get_items(None).await.unwrap())
}

#[tokio::test]
async fn a_callback_that_edits_history_in_place_does_not_resave_it() {
    let callback: SessionInputCallback = Arc::new(|mut history, new| {
        Box::pin(async move {
            for item in history.iter_mut() {
                item["content"] = json!("short");
            }
            history.extend(new);
            Ok(history)
        })
    });
    let config = RunConfig {
        session_input_callback: Some(callback),
        ..RunConfig::default()
    };
    let history = vec![
        json!({"role": "user", "content": "long long long"}),
        json!({"role": "assistant", "content": "answer answer"}),
    ];
    let (input, stored) = run(history, config).await;
    assert_eq!(
        short(&input),
        [
            item("user", Some("short")),
            item("assistant", Some("short")),
            item("user", Some("next"))
        ]
    );
    assert_eq!(
        short(&json!(stored)),
        [
            item("user", Some("long long long")),
            item("assistant", Some("answer answer")),
            item("user", Some("next")),
            item("message", None),
        ],
        "the edited history items were not saved again"
    );
}

#[tokio::test]
async fn stored_calls_without_an_output_are_not_sent_to_the_model() {
    let history = vec![json!({"role": "user", "content": "u1"}), call()];
    let (input, _) = run(history, RunConfig::default()).await;
    assert_eq!(
        short(&input),
        [item("user", Some("u1")), item("user", Some("next"))]
    );
}

#[tokio::test]
async fn a_limit_that_cuts_a_call_from_its_output_drops_the_output() {
    let history = vec![
        call(),
        output(),
        json!({"role": "user", "content": "a"}),
        json!({"role": "assistant", "content": "b"}),
    ];
    let config = RunConfig {
        session_settings: Some(SessionSettings { limit: Some(3) }),
        ..RunConfig::default()
    };
    let (input, _) = run(history, config).await;
    assert_eq!(
        short(&input),
        [
            item("user", Some("a")),
            item("assistant", Some("b")),
            item("user", Some("next"))
        ]
    );
}

#[tokio::test]
async fn duplicate_stored_items_are_merged() {
    let history = vec![
        json!({"role": "user", "content": "u"}),
        call(),
        output(),
        call(),
        output(),
    ];
    let (input, _) = run(history, RunConfig::default()).await;
    assert_eq!(
        short(&input),
        [
            item("user", Some("u")),
            item("function_call", None),
            item("function_call_output", None),
            item("user", Some("next")),
        ]
    );
}

#[tokio::test]
async fn the_hidden_origin_key_never_reaches_the_model_or_the_session() {
    let callback: SessionInputCallback = Arc::new(|history, new| {
        Box::pin(async move {
            let mut all = new;
            all.extend(history);
            Ok(all)
        })
    });
    let config = RunConfig {
        session_input_callback: Some(callback),
        ..RunConfig::default()
    };
    let (input, stored) = run(vec![json!({"role": "user", "content": "old"})], config).await;
    for item in input.as_array().unwrap().iter().chain(stored.iter()) {
        assert!(item.get("_agents_session_origin").is_none(), "{item}");
    }
    // Reordered: new first, history second; only the new input and the reply were saved.
    assert_eq!(stored.len(), 3);
}
