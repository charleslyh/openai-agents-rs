//! `CompactingSession`: summarizing old turns of a stored conversation.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    Agent, AgentsError, CompactingSession, InMemorySession, ModelSummarizer, RunOptions, Runner,
    Session, Summarizer,
};
use serde_json::{json, Value};

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

fn assistant(text: &str) -> Value {
    json!({"type": "message", "role": "assistant",
           "content": [{"type": "output_text", "text": text, "annotations": []}]})
}

/// Records what it was asked to summarize and returns canned notes.
#[derive(Default)]
struct RecordingSummarizer {
    calls: Mutex<Vec<(Option<String>, Vec<Value>)>>,
    fail: bool,
}

#[async_trait]
impl Summarizer for RecordingSummarizer {
    async fn summarize(
        &self,
        previous: Option<&str>,
        items: &[Value],
    ) -> Result<String, AgentsError> {
        let mut calls = self.calls.lock().unwrap();
        calls.push((previous.map(str::to_string), items.to_vec()));
        if self.fail {
            return Err(AgentsError::internal("summarizer down"));
        }
        Ok(format!("notes #{}", calls.len()))
    }
}

fn conversation() -> Vec<Value> {
    vec![
        json!({"role": "system", "content": "be brief"}),
        user("t1"),
        json!({"type": "function_call", "name": "search", "arguments": "{}", "call_id": "c1"}),
        json!({"type": "function_call_output", "call_id": "c1", "output": "r1"}),
        assistant("a1"),
        user("t2"),
        assistant("a2"),
        user("t3"),
        assistant("a3"),
        user("t4"),
        assistant("a4"),
    ]
}

async fn session_with(
    items: Vec<Value>,
    summarizer: Arc<RecordingSummarizer>,
) -> (Arc<InMemorySession>, CompactingSession) {
    let store = InMemorySession::shared("s");
    store.add_items(items).await.unwrap();
    let session = CompactingSession::new(store.clone(), summarizer).keep_recent_turns(2);
    (store, session)
}

#[tokio::test]
async fn compact_summarizes_old_turns_and_keeps_recent_ones() {
    let summarizer = Arc::new(RecordingSummarizer::default());
    let (store, session) = session_with(conversation(), summarizer.clone()).await;

    assert!(session.compact().await.unwrap());
    let stored = store.get_items(None).await.unwrap();
    // System message, the summary, then the two most recent turns verbatim.
    assert_eq!(stored[0]["role"], "system");
    let notes = stored[1]["content"].as_str().unwrap();
    assert!(notes.starts_with("<conversation_summary>") && notes.contains("notes #1"), "{notes}");
    assert_eq!(&stored[2..], &conversation()[7..], "t3 and t4 stay word for word");

    // The summarizer saw whole turns: t1 with its tool call and output, and t2.
    let calls = summarizer.calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].0.is_none());
    assert_eq!(calls[0].1, conversation()[1..7].to_vec());
}

#[tokio::test]
async fn a_second_compaction_carries_the_earlier_notes_forward() {
    let summarizer = Arc::new(RecordingSummarizer::default());
    let (store, session) = session_with(conversation(), summarizer.clone()).await;
    session.compact().await.unwrap();
    store.add_items(vec![user("t5"), assistant("a5")]).await.unwrap();

    assert!(session.compact().await.unwrap());
    let calls = summarizer.calls.lock().unwrap();
    assert_eq!(calls[1].0.as_deref(), Some("notes #1"), "previous notes passed in");
    assert_eq!(calls[1].1.len(), 2, "only t3 is newly summarized");
    let stored = store.get_items(None).await.unwrap();
    let summaries = stored.iter().filter(|i| i["content"].as_str().is_some_and(|c| c.starts_with("<conversation_summary>"))).count();
    assert_eq!(summaries, 1, "summaries merge instead of piling up");
    assert!(stored[1]["content"].as_str().unwrap().contains("notes #2"));
}

#[tokio::test]
async fn nothing_happens_without_enough_turns_or_below_the_trigger() {
    let summarizer = Arc::new(RecordingSummarizer::default());
    let store = InMemorySession::shared("s");
    store.add_items(vec![user("t1"), assistant("a1"), user("t2"), assistant("a2")]).await.unwrap();
    let session = CompactingSession::new(store.clone(), summarizer.clone()).keep_recent_turns(2);
    assert!(!session.compact().await.unwrap(), "only two turns, both kept");

    // Triggers are honoured on add_items: far below 8000 tokens, so nothing is summarized.
    session.add_items(vec![user("t3"), assistant("a3")]).await.unwrap();
    assert!(summarizer.calls.lock().unwrap().is_empty());
    assert_eq!(store.get_items(None).await.unwrap().len(), 6);
}

#[tokio::test]
async fn add_items_compacts_when_a_trigger_fires() {
    let summarizer = Arc::new(RecordingSummarizer::default());
    let store = InMemorySession::shared("s");
    let session = CompactingSession::new(store.clone(), summarizer.clone())
        .trigger_items(6)
        .keep_recent_turns(1);
    for turn in 1..=4 {
        session
            .add_items(vec![user(&format!("t{turn}")), assistant(&format!("a{turn}"))])
            .await
            .unwrap();
    }
    // Compaction fires once the store holds more than six items, and again as it regrows.
    assert!(!summarizer.calls.lock().unwrap().is_empty());
    let stored = store.get_items(None).await.unwrap();
    assert!(stored.len() <= 6, "{stored:?}");
    assert_eq!(stored.last().unwrap()["content"][0]["text"], "a4", "latest turn untouched");

    // A token trigger works the same way, with a custom counter.
    let by_tokens = Arc::new(RecordingSummarizer::default());
    let store = InMemorySession::shared("s2");
    let session = CompactingSession::new(store.clone(), by_tokens.clone())
        .trigger_tokens(5)
        .token_counter(|_| 1)
        .keep_recent_turns(1);
    session.add_items(vec![user("a"), assistant("b"), user("c"), assistant("d")]).await.unwrap();
    session.add_items(vec![user("e"), assistant("f")]).await.unwrap();
    assert_eq!(by_tokens.calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn a_failing_summarizer_never_loses_history() {
    let summarizer = Arc::new(RecordingSummarizer { fail: true, ..Default::default() });
    let store = InMemorySession::shared("s");
    let session = CompactingSession::new(store.clone(), summarizer.clone())
        .trigger_items(2)
        .keep_recent_turns(1);
    // The run's own write succeeds even though compaction fails behind it.
    session
        .add_items(vec![user("t1"), assistant("a1"), user("t2"), assistant("a2")])
        .await
        .expect("add_items must not fail");
    assert_eq!(store.get_items(None).await.unwrap().len(), 4);
    // An explicit compact() does report the failure.
    assert!(session.compact().await.is_err());
}

#[tokio::test]
async fn model_summarizer_sends_a_transcript_and_returns_the_text() {
    let summary_model = Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message(
        "User wants X; decided Y.",
    ))]));
    let summarizer = ModelSummarizer::new(summary_model.clone());
    let items = vec![
        user("please do X"),
        json!({"type": "function_call", "name": "search", "arguments": "{\"q\":1}", "call_id": "c1"}),
        json!({"type": "function_call_output", "call_id": "c1", "output": "o".repeat(5_000)}),
        json!({"type": "reasoning", "summary": []}),
        assistant("ok, Y"),
    ];
    let notes = summarizer.summarize(Some("old notes"), &items).await.unwrap();
    assert_eq!(notes, "User wants X; decided Y.");
    assert_eq!(summarizer.usage().requests, 1, "the summary call is counted");

    let call = &summary_model.calls()[0];
    assert!(call.system_instructions.as_deref().unwrap().contains("notes"));
    let prompt = call.input.as_str().expect("plain text prompt").to_string();
    assert!(prompt.contains("Existing notes:\nold notes"));
    assert!(prompt.contains("user: please do X"));
    assert!(prompt.contains("[tool call] search({\"q\":1})"));
    assert!(prompt.contains("[truncated 3500 chars]"), "long tool output is cut");
    assert!(prompt.contains("assistant: ok, Y"));
    assert!(!prompt.contains("reasoning"));
}

/// End to end: runs keep appending to a compacting session; the summary reaches the model.
#[tokio::test]
async fn a_long_conversation_stays_short_and_the_model_sees_the_summary() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    struct Counting(Arc<AtomicUsize>);
    #[async_trait]
    impl Summarizer for Counting {
        async fn summarize(&self, _: Option<&str>, _: &[Value]) -> Result<String, AgentsError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok("the user is planning a trip".to_string())
        }
    }

    let model = Arc::new(ScriptedModel::new(
        (1..=5).map(|n| ModelStep::from(ItemHelpers::text_message(&format!("answer {n}")))),
    ));
    let agent = Agent::new("a").model(model.clone());
    let store = InMemorySession::shared("chat");
    let session = CompactingSession::new(store.clone(), Arc::new(Counting(counter)))
        .trigger_items(5)
        .keep_recent_turns(1)
        .shared();
    for n in 1..=5 {
        let mut options = RunOptions::default();
        options.session = Some(session.clone());
        Runner::run(&agent, format!("question {n}"), options).await.expect("run");
    }

    assert!(calls.load(Ordering::SeqCst) >= 1);
    assert!(store.get_items(None).await.unwrap().len() <= 5, "history stays bounded");
    // The last call carried the summary instead of the early turns.
    let last_input = model.calls()[4].input.to_string();
    assert!(last_input.contains("<conversation_summary>"), "{last_input}");
    assert!(last_input.contains("the user is planning a trip"));
    assert!(!last_input.contains("question 1"), "{last_input}");
    assert!(last_input.contains("question 5"));
}
