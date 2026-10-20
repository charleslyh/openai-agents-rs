//! What was said in a run: prompts, replies, tool arguments and results (Rust-only, X-018).
//!
//! Spans say *what happened and how long it took*; they never hold a request body, a tool argument
//! or a model output (see [`crate::export`]). A console that shows a conversation needs exactly
//! those, so they travel on a channel of their own, and only when the application asks for it.
//!
//! * [`ConversationHub`] is the attachment point. Anything that wants the content calls
//!   [`ConversationHub::attach`] with a [`ConversationSink`] — a closure is enough — and receives
//!   every [`ConversationEvent`] of every run that records into the hub.
//! * [`ConversationRecorder`] is a [`RunHooks`] for one run. It turns the hook callbacks
//!   (`on_llm_start`, `on_tool_end`, ...) into events and gives them to the hub.
//!
//! A sink is called from inside the run loop, so it must not block. For anything slower than a
//! memory write — a socket, a file — wrap the writer in a [`QueuedConversationSink`], which hands
//! events to a thread of its own and drops them, counted, rather than make the run wait.
//!
//! Content is bounded before it leaves the hook: every string is clipped to
//! [`Limits::max_chars`], and an event that is still larger than [`Limits::max_event_bytes`] is
//! replaced by a marker. [`ConversationHub::metadata_only`] drops the content altogether and keeps
//! names, ids and counts, for runs whose text must not be recorded.
//!
//! ```rust
//! use std::sync::Arc;
//! use opai_agents_observability::conversation::{ConversationEvent, ConversationHub};
//!
//! let hub = ConversationHub::new();
//! hub.attach(Arc::new(|event: &ConversationEvent| println!("{}", event.to_json())));
//! // Per run: `RunOptions::new().hooks(hub.recorder("run-1"))`.
//! let hooks = hub.recorder("run-1");
//! # let _ = hooks;
//! ```

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use opai_agents_core::{
    Agent, FunctionTool, ModelResponse, ResponseInputItem, RunContextWrapper, RunHooks, ToolContext,
};
use serde_json::{json, Value};

use super::queue::Queue;

/// One thing said or done in a run, with its content.
#[derive(Debug, Clone, PartialEq)]
pub struct ConversationEvent {
    /// The run this belongs to; the console groups by it.
    pub run_id: String,
    /// What happened: `agent_start`, `agent_end`, `llm_start`, `llm_end`, `tool_start`,
    /// `tool_end` or `handoff`.
    pub kind: &'static str,
    /// The agent the event is about.
    pub agent: String,
    /// Seconds since the epoch.
    pub at: f64,
    /// The span that was current when the event happened, to place it on a trace.
    pub span_id: Option<String>,
    /// The content; its shape depends on `kind`.
    pub data: Value,
}

impl ConversationEvent {
    /// The JSON the console takes in.
    pub fn to_json(&self) -> Value {
        json!({
            "kind": self.kind,
            "run_id": self.run_id,
            "agent": self.agent,
            "at": self.at,
            "span_id": self.span_id,
            "data": self.data,
        })
    }
}

/// Receives the events of every run that records into a hub.
///
/// Called from inside the run loop: keep it quick, and never block.
pub trait ConversationSink: Send + Sync {
    /// One event, in the order it happened within its run.
    fn emit(&self, event: &ConversationEvent);
}

impl<F> ConversationSink for F
where
    F: Fn(&ConversationEvent) + Send + Sync,
{
    fn emit(&self, event: &ConversationEvent) {
        self(event)
    }
}

/// How much of a payload is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Longest string kept, in characters; the rest is replaced by a count.
    pub max_chars: usize,
    /// Largest event, as serialized JSON; past it the content is replaced by a marker.
    pub max_event_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_chars: 8 * 1024,
            max_event_bytes: 256 * 1024,
        }
    }
}

/// Where the events of a process go. Cheap to clone; clones share their sinks.
#[derive(Clone)]
pub struct ConversationHub {
    sinks: Arc<Mutex<Vec<Arc<dyn ConversationSink>>>>,
    content: bool,
    limits: Limits,
}

impl std::fmt::Debug for ConversationHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationHub")
            .field("sinks", &self.sink_count())
            .field("content", &self.content)
            .field("limits", &self.limits)
            .finish()
    }
}

impl Default for ConversationHub {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationHub {
    /// A hub with no sinks, recording content within the default [`Limits`].
    pub fn new() -> Self {
        Self {
            sinks: Arc::new(Mutex::new(Vec::new())),
            content: true,
            limits: Limits::default(),
        }
    }

    /// Keep names, ids and counts, and drop every prompt, reply, argument and result.
    #[must_use]
    pub fn metadata_only(mut self) -> Self {
        self.content = false;
        self
    }

    /// Replace the size limits.
    #[must_use]
    pub fn limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Start receiving events. Sinks attached to a hub are shared by its clones and recorders.
    pub fn attach(&self, sink: Arc<dyn ConversationSink>) {
        self.lock().push(sink);
    }

    /// How many sinks are attached.
    pub fn sink_count(&self) -> usize {
        self.lock().len()
    }

    /// The hooks for one run. Pass them as `RunOptions::hooks`.
    pub fn recorder(&self, run_id: impl Into<String>) -> Arc<ConversationRecorder> {
        Arc::new(ConversationRecorder {
            run_id: run_id.into(),
            hub: self.clone(),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Arc<dyn ConversationSink>>> {
        self.sinks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn record(&self, run_id: &str, kind: &'static str, agent: &str, data: Value) {
        // Cloned out so a sink that attaches another one from its callback cannot deadlock.
        let sinks: Vec<_> = self.lock().clone();
        if sinks.is_empty() {
            return;
        }
        let event = ConversationEvent {
            run_id: run_id.to_string(),
            kind,
            agent: agent.to_string(),
            at: now_secs(),
            span_id: opai_agents_core::tracing::current_span_id(),
            data: bound(data, self.limits),
        };
        for sink in sinks {
            sink.emit(&event);
        }
    }
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Clip every string in `value` to `max_chars` characters.
fn clip(value: &Value, max_chars: usize) -> Value {
    match value {
        Value::String(s) => {
            let total = s.chars().count();
            if total <= max_chars {
                return value.clone();
            }
            let head: String = s.chars().take(max_chars).collect();
            Value::String(format!("{head}…[{} more characters]", total - max_chars))
        }
        Value::Array(items) => Value::Array(items.iter().map(|v| clip(v, max_chars)).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), clip(v, max_chars)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Apply both limits to an event's content.
fn bound(data: Value, limits: Limits) -> Value {
    let clipped = clip(&data, limits.max_chars);
    let bytes = serde_json::to_vec(&clipped).map_or(0, |b| b.len());
    if bytes > limits.max_event_bytes {
        return json!({"truncated": true, "bytes": bytes});
    }
    clipped
}

/// A tool's JSON arguments are a string on the wire; show them as JSON when they parse.
fn parse_arguments(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// [`RunHooks`] that record one run into a [`ConversationHub`].
#[derive(Debug)]
pub struct ConversationRecorder {
    run_id: String,
    hub: ConversationHub,
}

impl ConversationRecorder {
    /// The run this records.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    fn record(&self, kind: &'static str, agent: &str, data: Value) {
        self.hub.record(&self.run_id, kind, agent, data);
    }

    /// `full` when content is recorded, else `bare`.
    fn pick(&self, full: impl FnOnce() -> Value, bare: Value) -> Value {
        if self.hub.content {
            full()
        } else {
            bare
        }
    }
}

#[async_trait]
impl RunHooks for ConversationRecorder {
    async fn on_agent_start(&self, _context: RunContextWrapper, agent: &Agent) {
        self.record("agent_start", &agent.name, json!({}));
    }

    async fn on_agent_end(&self, _context: RunContextWrapper, agent: &Agent, output: &Value) {
        let data = self.pick(|| json!({"output": output}), json!({}));
        self.record("agent_end", &agent.name, data);
    }

    async fn on_handoff(&self, _context: RunContextWrapper, from_agent: &Agent, to_agent: &Agent) {
        self.record(
            "handoff",
            &from_agent.name,
            json!({"from": from_agent.name, "to": to_agent.name}),
        );
    }

    async fn on_tool_start(&self, context: ToolContext, agent: &Agent, tool: &FunctionTool) {
        let data = self.pick(
            || {
                json!({
                    "tool": tool.name,
                    "call_id": context.tool_call_id,
                    "arguments": parse_arguments(&context.tool_arguments),
                })
            },
            json!({"tool": tool.name, "call_id": context.tool_call_id}),
        );
        self.record("tool_start", &agent.name, data);
    }

    async fn on_tool_end(
        &self,
        context: ToolContext,
        agent: &Agent,
        tool: &FunctionTool,
        result: &Value,
    ) {
        let data = self.pick(
            || {
                json!({
                    "tool": tool.name,
                    "call_id": context.tool_call_id,
                    "result": result,
                })
            },
            json!({"tool": tool.name, "call_id": context.tool_call_id}),
        );
        self.record("tool_end", &agent.name, data);
    }

    async fn on_llm_start(
        &self,
        _context: RunContextWrapper,
        agent: &Agent,
        system_prompt: Option<&str>,
        input_items: &[ResponseInputItem],
    ) {
        let data = self.pick(
            || json!({"system_prompt": system_prompt, "input": input_items}),
            json!({"input_items": input_items.len()}),
        );
        self.record("llm_start", &agent.name, data);
    }

    async fn on_llm_end(
        &self,
        _context: RunContextWrapper,
        agent: &Agent,
        response: &ModelResponse,
    ) {
        let usage = json!({
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        });
        let data = self.pick(
            || {
                json!({
                    "output": response.output,
                    "usage": usage,
                    "response_id": response.response_id,
                })
            },
            json!({
                "output_items": response.output.len(),
                "usage": usage,
                "response_id": response.response_id,
            }),
        );
        self.record("llm_end", &agent.name, data);
    }
}

/// A sink that hands events to a writer on a thread of its own.
///
/// The run never waits on it: a full queue drops the event and counts it
/// ([`QueuedConversationSink::dropped`]).
pub struct QueuedConversationSink {
    queue: Queue<ConversationEvent>,
}

impl std::fmt::Debug for QueuedConversationSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueuedConversationSink")
            .field("capacity", &self.queue.capacity())
            .field("dropped", &self.dropped())
            .finish()
    }
}

impl QueuedConversationSink {
    /// Write batches with `write` on a background thread. `capacity` is how many events may be
    /// waiting; `interval` is how long a lone event may sit in the queue.
    pub fn spawn(
        name: &'static str,
        write: impl FnMut(&[ConversationEvent]) -> Result<(), String> + Send + 'static,
        capacity: usize,
        interval: Duration,
    ) -> Arc<Self> {
        Arc::new(Self {
            queue: Queue::spawn(name, write, capacity, interval),
        })
    }

    /// Events lost: dropped from a full queue, or lost to a failed write.
    pub fn dropped(&self) -> u64 {
        self.queue.dropped()
    }

    /// Ask the writer to write what it has, and wait up to [`super::FLUSH_TIMEOUT`].
    pub fn flush(&self) {
        self.queue.flush();
    }
}

impl ConversationSink for QueuedConversationSink {
    fn emit(&self, event: &ConversationEvent) {
        self.queue.send(event.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(hub: &ConversationHub) -> Arc<Mutex<Vec<ConversationEvent>>> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        hub.attach(Arc::new(move |e: &ConversationEvent| {
            sink.lock().unwrap().push(e.clone());
        }));
        seen
    }

    #[test]
    fn long_strings_are_clipped_at_a_character_boundary() {
        let clipped = clip(&json!({"a": ["héllo wörld"]}), 4);
        assert_eq!(clipped["a"][0], "héll…[7 more characters]");
        assert_eq!(clip(&json!("short"), 10), json!("short"));
        assert_eq!(clip(&json!(42), 1), json!(42));
    }

    #[test]
    fn an_event_over_the_byte_limit_becomes_a_marker() {
        let limits = Limits {
            max_chars: 1000,
            max_event_bytes: 50,
        };
        let bounded = bound(json!({"text": "x".repeat(200)}), limits);
        assert_eq!(bounded["truncated"], true);
        assert!(bounded["bytes"].as_u64().unwrap() > 50);
        assert_eq!(bound(json!({"t": "ok"}), limits), json!({"t": "ok"}));
    }

    #[test]
    fn arguments_are_json_when_they_parse_and_text_when_they_do_not() {
        assert_eq!(parse_arguments(r#"{"city":"Berlin"}"#)["city"], "Berlin");
        assert_eq!(parse_arguments("not json"), json!("not json"));
    }

    #[tokio::test]
    async fn a_recorder_gives_the_hub_what_the_hooks_see() {
        let hub = ConversationHub::new();
        let seen = collect(&hub);
        let hooks = hub.recorder("run-1");
        let agent = Agent::new("Ops");
        let context = RunContextWrapper::new(None);

        hooks.on_agent_start(context.clone(), &agent).await;
        hooks
            .on_llm_start(
                context.clone(),
                &agent,
                Some("be brief"),
                &[json!({"role": "user", "content": "hi"})],
            )
            .await;
        hooks.on_agent_end(context, &agent, &json!("done")).await;

        let events = seen.lock().unwrap().clone();
        let kinds: Vec<_> = events.iter().map(|e| e.kind).collect();
        assert_eq!(kinds, ["agent_start", "llm_start", "agent_end"]);
        assert!(events
            .iter()
            .all(|e| e.run_id == "run-1" && e.agent == "Ops"));
        assert_eq!(events[1].data["system_prompt"], "be brief");
        assert_eq!(events[1].data["input"][0]["content"], "hi");
        assert_eq!(events[2].data["output"], "done");
    }

    #[tokio::test]
    async fn metadata_only_keeps_counts_and_drops_the_text() {
        let hub = ConversationHub::new().metadata_only();
        let seen = collect(&hub);
        let hooks = hub.recorder("run-1");
        let agent = Agent::new("Ops");
        let context = RunContextWrapper::new(None);

        hooks
            .on_llm_start(
                context.clone(),
                &agent,
                Some("secret prompt"),
                &[json!({"role": "user", "content": "secret input"})],
            )
            .await;
        hooks
            .on_agent_end(context, &agent, &json!("secret output"))
            .await;

        let events = seen.lock().unwrap().clone();
        assert_eq!(events[0].data, json!({"input_items": 1}));
        let wire =
            serde_json::to_string(&events.iter().map(|e| e.to_json()).collect::<Vec<_>>()).unwrap();
        assert!(!wire.contains("secret"), "{wire}");
    }

    #[test]
    fn a_hub_without_sinks_records_nothing_and_clones_share_sinks() {
        let hub = ConversationHub::new();
        hub.record("r", "agent_start", "a", json!({}));
        let clone = hub.clone();
        let seen = collect(&clone);
        assert_eq!(hub.sink_count(), 1);
        hub.record("r", "agent_start", "a", json!({}));
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[test]
    fn a_queued_sink_writes_in_order_off_the_calling_thread() {
        let written = Arc::new(Mutex::new(Vec::new()));
        let out = Arc::clone(&written);
        let sink = QueuedConversationSink::spawn(
            "test",
            move |events| {
                out.lock()
                    .unwrap()
                    .extend(events.iter().map(|e| e.kind.to_string()));
                Ok(())
            },
            16,
            Duration::from_millis(20),
        );
        let hub = ConversationHub::new();
        hub.attach(sink.clone());
        hub.record("r", "agent_start", "a", json!({}));
        hub.record("r", "agent_end", "a", json!({}));
        sink.flush();
        assert_eq!(*written.lock().unwrap(), ["agent_start", "agent_end"]);
        assert_eq!(sink.dropped(), 0);
    }
}
