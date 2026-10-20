//! Server state: conversations, runs paused for approval, the run table, and per-run settings.
//!
//! Everything is in memory (a `RunState` is a process-local snapshot), so every table has a TTL
//! and a size cap, and a background task sweeps them.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use opai_agents::{
    retry_policies, CancelMode, InMemorySession, McpServer, Model, ModelRetryBackoffSettings,
    ModelRetrySettings, ModelSettings, ModelSummarizer, RunState, Session, ToolInterruptionItem,
};
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;

use crate::faults::FaultPlan;
use crate::trace::TraceCollector;

pub const MAX_CONVERSATIONS: usize = 64;
pub const MAX_CONCURRENT_RUNS: usize = 8;
pub const DEFAULT_APPROVAL_TTL_SECS: u64 = 10 * 60;
/// Compact the stored history once it holds more items than this (kept low so it is easy to see).
pub const COMPACTION_TRIGGER_ITEMS: usize = 8;
const MAX_RUN_RECORDS: usize = 200;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RetryChoice {
    /// `never`, `network_error`, `http_status`, `retry_after` or `provider_suggested`.
    pub policy: Option<String>,
    pub max_retries: Option<u32>,
    pub initial_delay_secs: Option<f64>,
    pub max_delay_secs: Option<f64>,
    /// Per-attempt model timeout.
    pub attempt_timeout_secs: Option<f64>,
}

/// The settings drawer, validated.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunSettings {
    pub temperature: Option<f32>,
    pub max_turns: Option<usize>,
    pub parallel_tool_calls: Option<bool>,
    pub tool_concurrency: Option<usize>,
    /// Ask the provider for token usage in streamed answers. On: `include_usage = true`. Off: the
    /// SDK default is left alone (which omits it for third-party servers, so a server that rejects
    /// `stream_options` still works); the page then hides the token numbers.
    pub include_usage: Option<bool>,
    pub retry: RetryChoice,
}

fn clamp_f64(v: &mut Option<f64>, lo: f64, hi: f64) {
    if let Some(x) = v {
        *x = if x.is_finite() { x.clamp(lo, hi) } else { lo };
    }
}

impl RunSettings {
    pub fn sanitize(&mut self) {
        if let Some(t) = &mut self.temperature {
            *t = if t.is_finite() {
                t.clamp(0.0, 2.0)
            } else {
                1.0
            };
        }
        if let Some(n) = &mut self.max_turns {
            *n = (*n).clamp(1, 20);
        }
        if let Some(n) = &mut self.tool_concurrency {
            *n = (*n).clamp(1, 8);
        }
        if let Some(n) = &mut self.retry.max_retries {
            *n = (*n).min(5);
        }
        clamp_f64(&mut self.retry.initial_delay_secs, 0.0, 10.0);
        clamp_f64(&mut self.retry.max_delay_secs, 0.0, 10.0);
        clamp_f64(&mut self.retry.attempt_timeout_secs, 0.5, 60.0);
    }

    pub fn model_settings(&self) -> ModelSettings {
        let mut ms = ModelSettings {
            temperature: self.temperature,
            parallel_tool_calls: self.parallel_tool_calls,
            include_usage: self.include_usage.filter(|on| *on),
            timeout: self.retry.attempt_timeout_secs.map(|t| t as f32),
            ..Default::default()
        };
        let policy = match self.retry.policy.as_deref() {
            Some("network_error") => Some(retry_policies::network_error()),
            Some("http_status") => Some(retry_policies::http_status([429, 500, 502, 503, 504])),
            Some("retry_after") => Some(retry_policies::retry_after()),
            Some("provider_suggested") => Some(retry_policies::provider_suggested()),
            Some("never") => Some(retry_policies::never()),
            _ => None,
        };
        if let Some(policy) = policy {
            let mut retry = ModelRetrySettings::new(self.retry.max_retries.unwrap_or(2), policy);
            if self.retry.initial_delay_secs.is_some() || self.retry.max_delay_secs.is_some() {
                retry = retry.with_backoff(ModelRetryBackoffSettings {
                    initial_delay: self.retry.initial_delay_secs,
                    max_delay: self.retry.max_delay_secs,
                    ..Default::default()
                });
            }
            ms.retry = Some(retry);
        }
        ms
    }
}

/// One chat. The browser holds its id; the id is the only credential.
pub struct Conversation {
    pub id: String,
    pub session: Arc<InMemorySession>,
    pub summarizer: Arc<ModelSummarizer>,
    busy: AtomicBool,
    last_used: Mutex<Instant>,
}

impl Conversation {
    fn touch(&self) {
        *lock(&self.last_used) = Instant::now();
    }

    /// The text of the summary item currently at the head of the history, if any.
    pub async fn summary(&self) -> Option<String> {
        let items = self.session.get_items(None).await.ok()?;
        items.iter().find_map(summary_of)
    }
}

/// Mirrors the SDK's (private) rule: a user message whose text opens with `<conversation_summary>`.
pub fn summary_of(item: &Value) -> Option<String> {
    let is_user = item.get("role").and_then(Value::as_str) == Some("user");
    let text = item.get("content").and_then(Value::as_str)?;
    (is_user && text.starts_with("<conversation_summary>")).then(|| text.to_string())
}

/// Keeps a conversation to one run at a time; releases it on drop.
pub struct BusyGuard(Arc<Conversation>);

impl Drop for BusyGuard {
    fn drop(&mut self) {
        self.0.busy.store(false, Ordering::SeqCst);
        self.0.touch();
    }
}

/// Counts a running run against [`MAX_CONCURRENT_RUNS`]; releases it on drop.
pub struct RunSlot(Arc<AppState>);

impl Drop for RunSlot {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct PendingRun {
    pub state: RunState,
    pub interruptions: Vec<ToolInterruptionItem>,
    pub conversation_id: String,
    pub settings: RunSettings,
    pub faults: Arc<FaultPlan>,
    pub expires: Instant,
}

pub struct RunInfo {
    pub status: String,
    pub conversation_id: String,
    pub started: Instant,
    pub finished: Option<Instant>,
    pub cancel: Option<UnboundedSender<CancelMode>>,
}

pub enum TakeError {
    Unknown,
    Expired,
    BadCall(String),
}

pub struct AppState {
    pub model: Arc<dyn Model>,
    pub model_name: String,
    pub api: String,
    pub mcp: Option<Arc<dyn McpServer>>,
    pub trace: Arc<TraceCollector>,
    pub debug: bool,
    pub nonce: String,
    pub active: AtomicUsize,
    conversations: Mutex<HashMap<String, Arc<Conversation>>>,
    pending: Mutex<HashMap<String, PendingRun>>,
    runs: Mutex<HashMap<String, RunInfo>>,
}

pub fn new_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Arc<dyn Model>,
        model_name: String,
        api: String,
        mcp: Option<Arc<dyn McpServer>>,
        trace: Arc<TraceCollector>,
        debug: bool,
    ) -> Arc<Self> {
        Arc::new(Self {
            model,
            model_name,
            api,
            mcp,
            trace,
            debug,
            nonce: new_id(),
            active: AtomicUsize::new(0),
            conversations: Mutex::new(HashMap::new()),
            pending: Mutex::new(HashMap::new()),
            runs: Mutex::new(HashMap::new()),
        })
    }

    /// Reserve one of the run slots, or `None` when the server is busy.
    pub fn acquire_slot(self: &Arc<Self>) -> Option<RunSlot> {
        let previous = self.active.fetch_add(1, Ordering::SeqCst);
        if previous >= MAX_CONCURRENT_RUNS {
            self.active.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(RunSlot(Arc::clone(self)))
    }

    // ---- conversations ----

    /// Create a conversation; `None` when the table is full of busy conversations.
    pub fn create_conversation(&self) -> Option<Arc<Conversation>> {
        let mut table = lock(&self.conversations);
        if table.len() >= MAX_CONVERSATIONS {
            let oldest = table
                .values()
                .filter(|c| !c.busy.load(Ordering::SeqCst))
                .min_by_key(|c| *lock(&c.last_used))
                .map(|c| c.id.clone())?;
            table.remove(&oldest);
        }
        let id = new_id();
        let conversation = Arc::new(Conversation {
            id: id.clone(),
            session: InMemorySession::shared(id.clone()),
            summarizer: Arc::new(ModelSummarizer::new(Arc::clone(&self.model))),
            busy: AtomicBool::new(false),
            last_used: Mutex::new(Instant::now()),
        });
        table.insert(id, Arc::clone(&conversation));
        Some(conversation)
    }

    pub fn conversation(&self, id: &str) -> Option<Arc<Conversation>> {
        let found = lock(&self.conversations).get(id).cloned();
        if let Some(c) = &found {
            c.touch();
        }
        found
    }

    /// Mark the conversation busy. `None` when it already has a run.
    pub fn lock_conversation(&self, conversation: &Arc<Conversation>) -> Option<BusyGuard> {
        conversation
            .busy
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .ok()
            .map(|_| BusyGuard(Arc::clone(conversation)))
    }

    /// Remove a conversation and the approvals waiting on it. Returns whether it existed.
    pub fn remove_conversation(&self, id: &str) -> bool {
        let existed = lock(&self.conversations).remove(id).is_some();
        lock(&self.pending).retain(|_, p| p.conversation_id != id);
        existed
    }

    // ---- approvals ----

    pub fn store_pending(&self, run_id: String, pending: PendingRun) {
        lock(&self.pending).insert(run_id, pending);
    }

    /// Take a paused run for resuming. The decisions must name exactly the paused calls; a bad
    /// request leaves the run waiting, a good one consumes it, so it can only be resumed once.
    pub fn take_pending(
        &self,
        run_id: &str,
        conversation_id: &str,
        call_ids: &[String],
    ) -> Result<PendingRun, TakeError> {
        let mut table = lock(&self.pending);
        let Some(pending) = table
            .get(run_id)
            .filter(|p| p.conversation_id == conversation_id)
        else {
            return Err(TakeError::Unknown);
        };
        if pending.expires <= Instant::now() {
            table.remove(run_id);
            return Err(TakeError::Expired);
        }
        for id in call_ids {
            if !pending.interruptions.iter().any(|i| &i.call_id == id) {
                return Err(TakeError::BadCall(id.clone()));
            }
        }
        for item in &pending.interruptions {
            if !call_ids.contains(&item.call_id) {
                return Err(TakeError::BadCall(format!(
                    "missing decision for {}",
                    item.call_id
                )));
            }
        }
        table.remove(run_id).ok_or(TakeError::Unknown)
    }

    // ---- run table ----

    pub fn register_run(
        &self,
        run_id: &str,
        conversation_id: &str,
        cancel: UnboundedSender<CancelMode>,
    ) {
        let mut runs = lock(&self.runs);
        if runs.len() >= MAX_RUN_RECORDS {
            let oldest = runs
                .iter()
                .filter(|(_, r)| r.finished.is_some())
                .min_by_key(|(_, r)| r.started)
                .map(|(id, _)| id.clone());
            if let Some(id) = oldest {
                runs.remove(&id);
            }
        }
        runs.insert(
            run_id.to_string(),
            RunInfo {
                status: "running".into(),
                conversation_id: conversation_id.to_string(),
                started: Instant::now(),
                finished: None,
                cancel: Some(cancel),
            },
        );
    }

    pub fn finish_run(&self, run_id: &str, status: &str) {
        if let Some(info) = lock(&self.runs).get_mut(run_id) {
            info.status = status.to_string();
            info.finished = Some(Instant::now());
            info.cancel = None;
        }
    }

    /// Ask a running run to stop. `None` when the run is unknown or already over.
    pub fn cancel_run(&self, run_id: &str, conversation_id: &str, mode: CancelMode) -> Option<()> {
        let runs = lock(&self.runs);
        let info = runs.get(run_id)?;
        if info.conversation_id != conversation_id {
            return None;
        }
        info.cancel.as_ref()?.send(mode).ok()
    }

    pub fn runs_for(&self, conversation_id: &str) -> Vec<(String, String, u64)> {
        let runs = lock(&self.runs);
        let mut out: Vec<_> = runs
            .iter()
            .filter(|(_, r)| r.conversation_id == conversation_id)
            .map(|(id, r)| (id.clone(), r.status.clone(), r.started.elapsed().as_secs()))
            .collect();
        out.sort_by_key(|(_, _, age)| *age);
        out
    }

    /// Drop expired approvals and old run records. Conversations are not swept (local dev demo);
    /// they are only bounded by `MAX_CONVERSATIONS`.
    pub fn sweep(&self) {
        let now = Instant::now();
        lock(&self.pending).retain(|_, p| p.expires > now);
        lock(&self.runs).retain(|_, r| {
            r.finished
                .is_none_or(|t| now.duration_since(t) < Duration::from_secs(3600))
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_are_clamped() {
        let mut s: RunSettings = serde_json::from_value(serde_json::json!({
            "temperature": 99.0,
            "max_turns": 1000,
            "tool_concurrency": 0,
            "retry": {"policy": "retry_after", "max_retries": 50, "attempt_timeout_secs": 0.0}
        }))
        .expect("valid");
        s.sanitize();
        assert_eq!(s.temperature, Some(2.0));
        assert_eq!(s.max_turns, Some(20));
        assert_eq!(s.tool_concurrency, Some(1));
        assert_eq!(s.retry.max_retries, Some(5));
        let ms = s.model_settings();
        assert_eq!(ms.timeout, Some(0.5));
        assert!(ms.retry.is_some());
    }

    #[test]
    fn include_usage_is_only_forced_when_on() {
        let on: RunSettings =
            serde_json::from_value(serde_json::json!({"include_usage": true})).expect("valid");
        assert_eq!(on.model_settings().include_usage, Some(true));
        let off: RunSettings =
            serde_json::from_value(serde_json::json!({"include_usage": false})).expect("valid");
        assert_eq!(off.model_settings().include_usage, None);
        assert_eq!(RunSettings::default().model_settings().include_usage, None);
    }

    #[test]
    fn summary_item_is_recognized() {
        let item = serde_json::json!({"role": "user", "content": "<conversation_summary>\nx"});
        assert!(summary_of(&item).is_some());
        let plain = serde_json::json!({"role": "user", "content": "hello"});
        assert!(summary_of(&plain).is_none());
    }
}
