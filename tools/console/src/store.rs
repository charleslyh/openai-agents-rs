//! What the console remembers: runs, each with the spans and conversation that belong to it.
//!
//! Everything is bounded, because a console is left running: a limited number of runs (the oldest
//! goes first), a limited number of items per run, and a limited number of span ids tracked. What
//! does not fit is counted, not hidden.
//!
//! Spans do not carry a run id, so one is worked out. A run's root span is a `task` span with no
//! parent, named after the workflow, and the application uses the run id as the workflow name. Every
//! other span belongs to the run of its parent, which is known because a parent starts before its
//! children. Conversation events name their run themselves.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};
use tokio::sync::broadcast;

/// Where spans go when no run can be worked out for them.
pub const UNASSIGNED: &str = "_unassigned";

/// How much the store keeps.
#[derive(Debug, Clone, Copy)]
pub struct StoreLimits {
    /// Runs kept; past it the one that has been quiet longest is dropped.
    pub max_runs: usize,
    /// Items kept per run; past it new ones are counted in `dropped` but not stored.
    pub max_items_per_run: usize,
    /// Span ids remembered for the run they belong to.
    pub max_tracked_spans: usize,
}

impl Default for StoreLimits {
    fn default() -> Self {
        Self {
            max_runs: 200,
            max_items_per_run: 5_000,
            max_tracked_spans: 100_000,
        }
    }
}

/// What an ingest call did with a body.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Report {
    /// Lines stored.
    pub accepted: usize,
    /// Lines that were not JSON objects of a known shape.
    pub rejected: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    /// Items are still arriving, and no root span has ended.
    Running,
    /// The root span ended.
    Done,
    /// The root span ended with an error.
    Failed,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Done => "done",
            Self::Failed => "error",
        }
    }
}

struct Run {
    id: String,
    started_at: f64,
    last_at: f64,
    status: Status,
    preview: Option<String>,
    root_span: Option<String>,
    items: Vec<Value>,
    dropped: u64,
    turns: u64,
    input_tokens: u64,
    output_tokens: u64,
    errors: u64,
}

impl Run {
    fn new(id: &str, now: f64) -> Self {
        Self {
            id: id.to_string(),
            started_at: now,
            last_at: now,
            status: Status::Running,
            preview: None,
            root_span: None,
            items: Vec::new(),
            dropped: 0,
            turns: 0,
            input_tokens: 0,
            output_tokens: 0,
            errors: 0,
        }
    }

    fn summary(&self) -> Value {
        json!({
            "id": self.id,
            "status": self.status.as_str(),
            "started_at": self.started_at,
            "last_at": self.last_at,
            "preview": self.preview,
            "turns": self.turns,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "errors": self.errors,
            "items": self.items.len(),
            "dropped": self.dropped,
        })
    }
}

#[derive(Default)]
struct Inner {
    runs: HashMap<String, Run>,
    /// span id -> run id
    owner: HashMap<String, String>,
}

/// The runs seen so far, and a feed of what arrives.
pub struct Store {
    inner: Mutex<Inner>,
    limits: StoreLimits,
    seq: AtomicU64,
    feed: broadcast::Sender<Arc<Value>>,
}

impl Store {
    /// An empty store.
    pub fn new(limits: StoreLimits) -> Self {
        let (feed, _) = broadcast::channel(1024);
        Self {
            inner: Mutex::new(Inner::default()),
            limits,
            seq: AtomicU64::new(0),
            feed,
        }
    }

    /// Every ingested item, as `{"run": <summary>, "item": <item>}`.
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<Value>> {
        self.feed.subscribe()
    }

    /// Run summaries, most recently active first.
    pub fn runs(&self) -> Vec<Value> {
        let inner = self.lock();
        let mut runs: Vec<&Run> = inner.runs.values().collect();
        runs.sort_by(|a, b| b.last_at.total_cmp(&a.last_at));
        runs.iter().map(|r| r.summary()).collect()
    }

    /// One run: its summary and every stored item, in arrival order.
    pub fn run(&self, id: &str) -> Option<Value> {
        let inner = self.lock();
        let run = inner.runs.get(id)?;
        Some(json!({"run": run.summary(), "items": run.items}))
    }

    /// Store each line of an NDJSON body.
    pub fn ingest_ndjson(&self, body: &str) -> Report {
        let mut report = Report::default();
        for line in body.lines().filter(|l| !l.trim().is_empty()) {
            let stored = serde_json::from_str::<Value>(line).is_ok_and(|v| self.ingest(v));
            if stored {
                report.accepted += 1;
            } else {
                report.rejected += 1;
            }
        }
        report
    }

    /// Store one item. `false` when it is not an object with a `kind`, or names no run.
    pub fn ingest(&self, item: Value) -> bool {
        let Value::Object(mut item) = item else {
            return false;
        };
        let Some(kind) = item.get("kind").and_then(Value::as_str).map(str::to_owned) else {
            return false;
        };
        let now = now_secs();
        let message = {
            let mut inner = self.lock();
            let Some(run_id) = self.assign(&mut inner, &kind, &item) else {
                return false;
            };
            item.insert("run_id".into(), json!(run_id));
            item.insert(
                "seq".into(),
                json!(self.seq.fetch_add(1, Ordering::Relaxed)),
            );
            self.store(&mut inner, &run_id, &kind, item.clone(), now);
            let run = &inner.runs[&run_id];
            json!({"run": run.summary(), "item": Value::Object(item)})
        };
        // Nobody listening is fine: the feed is for live viewers only.
        let _ = self.feed.send(Arc::new(message));
        true
    }

    /// The run an item belongs to, remembering which run a starting span is in.
    fn assign(&self, inner: &mut Inner, kind: &str, item: &Map<String, Value>) -> Option<String> {
        let text = |key: &str| item.get(key).and_then(Value::as_str);
        match kind {
            "trace_start" | "trace_end" => {
                Some(text("workflow_name").unwrap_or(UNASSIGNED).to_string())
            }
            "span_start" | "span_end" => {
                let span_id = text("span_id")?;
                let parent = text("parent_id");
                let run = if let Some(known) = inner.owner.get(span_id) {
                    known.clone()
                } else if text("span_kind") == Some("task") && parent.is_none() {
                    text("name").unwrap_or(UNASSIGNED).to_string()
                } else {
                    parent
                        .and_then(|p| inner.owner.get(p))
                        .cloned()
                        .unwrap_or_else(|| UNASSIGNED.to_string())
                };
                if kind == "span_start" && inner.owner.len() < self.limits.max_tracked_spans {
                    inner.owner.insert(span_id.to_string(), run.clone());
                }
                Some(run)
            }
            // Anything else is conversation, which says whose it is.
            _ => text("run_id").filter(|r| !r.is_empty()).map(str::to_owned),
        }
    }

    fn store(
        &self,
        inner: &mut Inner,
        run_id: &str,
        kind: &str,
        item: Map<String, Value>,
        now: f64,
    ) {
        if !inner.runs.contains_key(run_id) {
            inner.runs.insert(run_id.to_string(), Run::new(run_id, now));
            self.evict(inner);
        }
        let Some(run) = inner.runs.get_mut(run_id) else {
            return;
        };
        run.last_at = now;
        observe(run, kind, &item);
        if run.items.len() < self.limits.max_items_per_run {
            run.items.push(Value::Object(item));
        } else {
            run.dropped += 1;
        }
    }

    /// Drop the runs that have been quiet longest until the limit holds.
    fn evict(&self, inner: &mut Inner) {
        while inner.runs.len() > self.limits.max_runs {
            let Some(oldest) = inner
                .runs
                .values()
                .min_by(|a, b| a.last_at.total_cmp(&b.last_at))
                .map(|r| r.id.clone())
            else {
                return;
            };
            inner.runs.remove(&oldest);
            inner.owner.retain(|_, run| *run != oldest);
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Update a run's summary from one item.
fn observe(run: &mut Run, kind: &str, item: &Map<String, Value>) {
    let text = |key: &str| item.get(key).and_then(Value::as_str);
    match kind {
        "span_start" => {
            if text("span_kind") == Some("turn") {
                run.turns += 1;
            }
            if text("span_kind") == Some("task") && item.get("parent_id").is_none_or(Value::is_null)
            {
                run.root_span = text("span_id").map(str::to_owned);
            }
        }
        "span_end" => {
            if item.get("error").is_some_and(|e| !e.is_null()) {
                run.errors += 1;
            }
            if text("span_kind") == Some("task") && run.root_span.as_deref() == text("span_id") {
                run.status = if item.get("error").is_some_and(|e| !e.is_null()) {
                    Status::Failed
                } else {
                    Status::Done
                };
                let usage = item.get("usage");
                let count = |key: &str| usage.and_then(|u| u.get(key)).and_then(Value::as_u64);
                run.input_tokens = count("input_tokens").unwrap_or(run.input_tokens);
                run.output_tokens = count("output_tokens").unwrap_or(run.output_tokens);
            }
        }
        "llm_start" if run.preview.is_none() => {
            run.preview = item
                .get("data")
                .and_then(|d| d.get("input"))
                .and_then(last_user_text);
        }
        _ => {}
    }
}

/// The text of the last user message in a model request, shortened for a list.
fn last_user_text(input: &Value) -> Option<String> {
    let items = input.as_array()?;
    let message = items
        .iter()
        .rev()
        .find(|i| i.get("role").and_then(Value::as_str) == Some("user"))?;
    let content = message.get("content")?;
    let text = match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(" "),
        _ => return None,
    };
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let short: String = text.chars().take(120).collect();
    Some(short).filter(|s| !s.is_empty())
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(kind: &str, span_kind: &str, id: &str, parent: Option<&str>, name: &str) -> Value {
        json!({"kind": kind, "span_kind": span_kind, "span_id": id, "parent_id": parent, "name": name})
    }

    fn items(store: &Store, run: &str) -> Vec<Value> {
        store.run(run).expect("run")["items"]
            .as_array()
            .expect("items")
            .clone()
    }

    #[test]
    fn spans_follow_their_parents_to_the_run_named_by_the_root_task() {
        let store = Store::new(StoreLimits::default());
        assert!(store.ingest(span("span_start", "task", "t", None, "run-1")));
        assert!(store.ingest(span("span_start", "agent", "a", Some("t"), "Ops")));
        assert!(store.ingest(span("span_start", "turn", "u", Some("a"), "Ops")));
        // Children end before parents; the end still finds the run.
        assert!(store.ingest(span("span_end", "turn", "u", Some("a"), "Ops")));

        assert_eq!(items(&store, "run-1").len(), 4);
        assert!(store.run(UNASSIGNED).is_none());
        let runs = store.runs();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0]["turns"], 1);
        assert_eq!(runs[0]["status"], "running");
    }

    #[test]
    fn a_span_with_no_known_parent_is_unassigned_rather_than_lost() {
        let store = Store::new(StoreLimits::default());
        assert!(store.ingest(span("span_end", "function", "f", Some("gone"), "tool")));
        assert_eq!(items(&store, UNASSIGNED).len(), 1);
    }

    #[test]
    fn the_root_task_ending_finishes_the_run_and_reports_usage() {
        let store = Store::new(StoreLimits::default());
        store.ingest(span("span_start", "task", "t", None, "run-1"));
        let mut end = span("span_end", "task", "t", None, "run-1");
        end["usage"] = json!({"input_tokens": 12, "output_tokens": 5});
        store.ingest(end);

        let run = &store.runs()[0];
        assert_eq!(run["status"], "done");
        assert_eq!(run["input_tokens"], 12);
        assert_eq!(run["output_tokens"], 5);
    }

    #[test]
    fn an_error_on_the_root_task_fails_the_run() {
        let store = Store::new(StoreLimits::default());
        store.ingest(span("span_start", "task", "t", None, "run-1"));
        let mut end = span("span_end", "task", "t", None, "run-1");
        end["error"] = json!({"message": "boom"});
        store.ingest(end);
        assert_eq!(store.runs()[0]["status"], "error");
        assert_eq!(store.runs()[0]["errors"], 1);
    }

    #[test]
    fn conversation_names_its_own_run_and_the_first_request_becomes_the_preview() {
        let store = Store::new(StoreLimits::default());
        assert!(!store.ingest(json!({"kind": "llm_start"})), "no run id");
        assert!(store.ingest(json!({
            "kind": "llm_start", "run_id": "run-1", "at": 1.0,
            "data": {"input": [
                {"role": "user", "content": "first   question"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": [{"type": "input_text", "text": "follow up"}]}
            ]}
        })));
        assert_eq!(store.runs()[0]["preview"], "follow up");
    }

    #[test]
    fn lines_that_are_not_objects_with_a_kind_are_rejected() {
        let store = Store::new(StoreLimits::default());
        let report = store.ingest_ndjson(
            "{\"kind\":\"agent_start\",\"run_id\":\"r\"}\nnot json\n[1]\n\n{\"no\":\"kind\"}\n",
        );
        assert_eq!(
            report,
            Report {
                accepted: 1,
                rejected: 3
            }
        );
    }

    #[test]
    fn items_are_numbered_so_a_viewer_can_tell_what_it_has_seen() {
        let store = Store::new(StoreLimits::default());
        for _ in 0..3 {
            store.ingest(json!({"kind": "agent_start", "run_id": "r"}));
        }
        let seqs: Vec<_> = items(&store, "r")
            .iter()
            .map(|i| i["seq"].as_u64().unwrap())
            .collect();
        assert_eq!(seqs, [0, 1, 2]);
    }

    #[test]
    fn a_run_keeps_a_bounded_number_of_items_and_counts_the_rest() {
        let store = Store::new(StoreLimits {
            max_items_per_run: 2,
            ..StoreLimits::default()
        });
        for _ in 0..5 {
            store.ingest(json!({"kind": "agent_start", "run_id": "r"}));
        }
        assert_eq!(items(&store, "r").len(), 2);
        assert_eq!(store.runs()[0]["dropped"], 3);
    }

    #[test]
    fn the_run_that_has_been_quiet_longest_is_dropped_with_its_span_ids() {
        let store = Store::new(StoreLimits {
            max_runs: 2,
            ..StoreLimits::default()
        });
        store.ingest(span("span_start", "task", "t1", None, "old"));
        std::thread::sleep(std::time::Duration::from_millis(5));
        store.ingest(span("span_start", "task", "t2", None, "mid"));
        std::thread::sleep(std::time::Duration::from_millis(5));
        store.ingest(span("span_start", "task", "t3", None, "new"));

        assert!(store.run("old").is_none());
        assert!(store.run("mid").is_some() && store.run("new").is_some());
        assert!(!store.lock().owner.contains_key("t1"));
    }

    #[tokio::test]
    async fn live_viewers_receive_each_item_with_its_run_summary() {
        let store = Store::new(StoreLimits::default());
        let mut feed = store.subscribe();
        store.ingest(json!({"kind": "agent_start", "run_id": "r", "agent": "Ops"}));
        let message = feed.recv().await.expect("message");
        assert_eq!(message["run"]["id"], "r");
        assert_eq!(message["item"]["agent"], "Ops");
        assert_eq!(message["item"]["seq"], 0);
    }
}
