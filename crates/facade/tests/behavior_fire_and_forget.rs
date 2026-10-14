//! Fire-and-forget runs plus cross-device subscription to one session.
//!
//! The SDK gives one `RunResultStreaming` per run (a single consumer). This test shows how to
//! build the multi-device layer on top of it: a small hub spawns the run in the background,
//! pumps every `StreamEvent` into a per-session log + broadcast channel, and any device can
//! subscribe by session id, before, during or after the run (late joiners replay the log).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    Agent, FunctionTool, InMemorySession, RunItemStreamName, RunOptions, Runner, Session,
    StreamEvent,
};
use tokio::sync::broadcast;

struct SessionChannel {
    state: Mutex<ChannelState>,
}

struct ChannelState {
    log: Vec<StreamEvent>,
    /// `None` once the run has finished, which closes every subscriber's receiver.
    tx: Option<broadcast::Sender<StreamEvent>>,
}

#[derive(Default, Clone)]
struct SessionHub {
    channels: Arc<Mutex<HashMap<String, Arc<SessionChannel>>>>,
}

impl SessionHub {
    /// Fire and forget: returns immediately; the run continues in the background.
    fn submit(&self, session: Arc<dyn Session>, agent: Agent, input: &str) {
        let (tx, _) = broadcast::channel(256);
        let channel = Arc::new(SessionChannel {
            state: Mutex::new(ChannelState {
                log: Vec::new(),
                tx: Some(tx),
            }),
        });
        self.channels
            .lock()
            .unwrap()
            .insert(session.session_id().to_string(), channel.clone());

        let mut options = RunOptions::default();
        options.session = Some(session);
        let mut streamed = Runner::run_streamed(agent, input, options);
        tokio::spawn(async move {
            while let Some(event) = streamed.next_event().await {
                let Ok(event) = event else { break };
                let mut state = channel.state.lock().unwrap();
                state.log.push(event.clone());
                if let Some(tx) = &state.tx {
                    let _ = tx.send(event);
                }
            }
            channel.state.lock().unwrap().tx = None;
        });
    }

    /// Attach a device to a session; yields every event from the start of the run until it ends.
    async fn subscribe(&self, session_id: &str) -> Vec<StreamEvent> {
        let channel = self
            .channels
            .lock()
            .unwrap()
            .get(session_id)
            .cloned()
            .expect("unknown session");
        // Snapshot the log and subscribe under one lock so no event is missed or duplicated.
        let (mut events, rx) = {
            let state = channel.state.lock().unwrap();
            (
                state.log.clone(),
                state.tx.as_ref().map(|tx| tx.subscribe()),
            )
        };
        if let Some(mut rx) = rx {
            loop {
                match rx.recv().await {
                    Ok(event) => events.push(event),
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
        events
    }
}

fn agent() -> Agent {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    Agent::new("assistant")
        .model(model)
        .tools(vec![FunctionTool::constant("echo", "echo", "ok")])
}

fn is_final_message(e: &StreamEvent) -> bool {
    matches!(
        e,
        StreamEvent::RunItem {
            name: RunItemStreamName::MessageOutputCreated,
            ..
        }
    )
}

#[tokio::test]
async fn fire_and_forget_run_completes_without_a_consumer() {
    let hub = SessionHub::default();
    let session = InMemorySession::shared("s-forget");

    // The caller submits and walks away: no stream handle is kept or awaited.
    hub.submit(session.clone(), agent(), "go");

    // Observe the outcome only afterwards, through the session and a late subscriber.
    let events = hub.subscribe("s-forget").await;
    assert!(events.iter().any(is_final_message));
    let items = session.get_items(None).await.unwrap();
    assert!(
        items.len() >= 2,
        "run must persist history even though nobody consumed the stream: {items:?}"
    );
}

#[tokio::test]
async fn two_devices_subscribe_to_the_same_session() {
    let hub = SessionHub::default();
    let session = InMemorySession::shared("s-shared");
    hub.submit(session, agent(), "go");

    // Device A (phone) and device B (laptop) attach concurrently to the same session id.
    let (a, b) = tokio::join!(hub.subscribe("s-shared"), hub.subscribe("s-shared"));

    assert!(!a.is_empty());
    assert_eq!(a.len(), b.len(), "both devices see the full event stream");
    assert!(matches!(
        &a[0],
        StreamEvent::AgentUpdated { agent_name } if agent_name == "assistant"
    ));
    assert!(a.iter().any(is_final_message) && b.iter().any(is_final_message));
}

#[tokio::test]
async fn late_device_replays_history_after_run_finished() {
    let hub = SessionHub::default();
    hub.submit(InMemorySession::shared("s-late"), agent(), "go");

    let early = hub.subscribe("s-late").await;
    // The run is over; a device that connects now still gets the whole run from the log.
    let late = hub.subscribe("s-late").await;

    assert_eq!(early.len(), late.len());
    assert!(late.iter().any(is_final_message));
}
