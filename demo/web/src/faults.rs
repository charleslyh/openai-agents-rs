//! Fault injection: a per-run [`FaultPlan`] plus decorators that apply it to the model, the
//! session, the summarizer and the MCP server. Nothing here touches the SDK itself: every
//! decorator implements a public trait and wraps the real implementation.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use opai_agents::{
    Agent, AgentsError, InMemorySession, McpCallToolResult, McpServer, McpTool, Model,
    ModelConnectionError, ModelError, ModelRequest, ModelResponse, ModelStatusError,
    ModelSummarizer, RunContextWrapper, Session, Summarizer, Usage,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::events::{model_error_kind, DemoEvent, Emitter};

/// Every injected error message carries this tag, so the page can tell it from a real failure.
pub const INJECTED_TAG: &str = "[injected]";

const MAX_DELAY_MS: u64 = 15_000;
const MAX_FAIL_ATTEMPTS: u32 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum LlmFault {
    #[serde(rename = "rate_limit_429")]
    RateLimit429,
    #[serde(rename = "server_500")]
    Server500,
    #[serde(rename = "server_503")]
    Server503,
    #[serde(rename = "bad_request_400")]
    BadRequest400,
    #[serde(rename = "unauthorized_401")]
    Unauthorized401,
    #[serde(rename = "connection_reset")]
    ConnectionReset,
    #[serde(rename = "connect_timeout")]
    ConnectTimeout,
    #[serde(rename = "hang")]
    Hang,
    #[serde(rename = "empty_response")]
    EmptyResponse,
    #[serde(rename = "malformed_response")]
    MalformedResponse,
    #[serde(rename = "slow_first_token")]
    SlowFirstToken,
    #[serde(rename = "slow_stream")]
    SlowStream,
    #[serde(rename = "stream_cut")]
    StreamCut,
    #[serde(rename = "refusal")]
    Refusal,
    #[serde(rename = "invalid_structured")]
    InvalidStructured,
    #[serde(rename = "max_turns")]
    MaxTurns,
}

impl LlmFault {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RateLimit429 => "rate_limit_429",
            Self::Server500 => "server_500",
            Self::Server503 => "server_503",
            Self::BadRequest400 => "bad_request_400",
            Self::Unauthorized401 => "unauthorized_401",
            Self::ConnectionReset => "connection_reset",
            Self::ConnectTimeout => "connect_timeout",
            Self::Hang => "hang",
            Self::EmptyResponse => "empty_response",
            Self::MalformedResponse => "malformed_response",
            Self::SlowFirstToken => "slow_first_token",
            Self::SlowStream => "slow_stream",
            Self::StreamCut => "stream_cut",
            Self::Refusal => "refusal",
            Self::InvalidStructured => "invalid_structured",
            Self::MaxTurns => "max_turns",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionFault {
    Summarizer,
    Read,
    Write,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpFault {
    ListFail,
    CallFail,
    CallHang,
    Disconnect,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolFaults {
    /// Extra latency before every demo tool runs.
    pub delay_ms: Option<u64>,
    /// The first N tool invocations of the run fail.
    pub fail_times: Option<u32>,
    /// Every demo tool hangs (until its timeout).
    pub hang: bool,
    /// Overrides every tool's timeout.
    pub timeout_secs: Option<f64>,
    /// A timeout aborts the run instead of telling the model.
    pub timeout_raises: bool,
    /// A tool guardrail stops the run instead of replacing the content.
    pub guardrail_raises: bool,
}

/// The faults to apply to one run. Comes from the request body, so every value is validated and
/// clamped by [`FaultPlan::sanitize`] before use.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FaultPlan {
    pub llm: Option<LlmFault>,
    /// How many model calls (attempts) the LLM fault hits. Default 1.
    pub llm_fail_attempts: Option<u32>,
    pub llm_delay_ms: Option<u64>,
    pub retry_after_secs: Option<u32>,
    /// Text deltas forwarded before a `stream_cut`.
    pub stream_cut_after: Option<u32>,
    pub tool: ToolFaults,
    pub session: Option<SessionFault>,
    pub mcp: Option<McpFault>,
    pub sse_delay_ms: Option<u64>,
    pub sse_drop_after: Option<u32>,
    pub approval_ttl_secs: Option<u64>,
}

impl FaultPlan {
    /// Clamp every number to a safe range.
    pub fn sanitize(&mut self) {
        let clamp_ms = |v: &mut Option<u64>, max: u64| {
            if let Some(x) = v {
                *x = (*x).min(max);
            }
        };
        clamp_ms(&mut self.llm_delay_ms, MAX_DELAY_MS);
        clamp_ms(&mut self.tool.delay_ms, MAX_DELAY_MS);
        clamp_ms(&mut self.sse_delay_ms, 5_000);
        if let Some(n) = &mut self.llm_fail_attempts {
            *n = (*n).clamp(1, MAX_FAIL_ATTEMPTS);
        }
        if let Some(n) = &mut self.tool.fail_times {
            *n = (*n).min(MAX_FAIL_ATTEMPTS);
        }
        if let Some(n) = &mut self.retry_after_secs {
            *n = (*n).clamp(0, 10);
        }
        if let Some(n) = &mut self.stream_cut_after {
            *n = (*n).clamp(1, 200);
        }
        if let Some(n) = &mut self.sse_drop_after {
            *n = (*n).clamp(1, 10_000);
        }
        if let Some(n) = &mut self.approval_ttl_secs {
            *n = (*n).clamp(1, 600);
        }
        if let Some(t) = &mut self.tool.timeout_secs {
            *t = if t.is_finite() {
                t.clamp(0.2, 30.0)
            } else {
                10.0
            };
        }
    }

    pub fn is_empty(&self) -> bool {
        self.llm.is_none()
            && self.session.is_none()
            && self.mcp.is_none()
            && self.sse_delay_ms.is_none()
            && self.sse_drop_after.is_none()
            && self.approval_ttl_secs.is_none()
            && self.tool.delay_ms.is_none()
            && self.tool.fail_times.is_none()
            && !self.tool.hang
            && self.tool.timeout_secs.is_none()
    }

    pub fn fail_attempts(&self) -> u32 {
        self.llm_fail_attempts.unwrap_or(1)
    }
}

fn now_seconds() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn connection_error(message: &str, is_timeout: bool) -> ModelError {
    ModelError::Connection(ModelConnectionError {
        message: format!("{INJECTED_TAG} {message}"),
        is_timeout,
    })
}

fn status_error(code: u16, reason: &str, retry_after: Option<u32>) -> ModelError {
    let mut headers = BTreeMap::new();
    headers.insert("x-request-id".to_string(), "req_injected_demo".to_string());
    if let Some(secs) = retry_after {
        headers.insert("retry-after".to_string(), secs.to_string());
    }
    ModelError::Status(Box::new(ModelStatusError {
        status_code: code,
        message: format!("{INJECTED_TAG} HTTP {code}: {reason}"),
        body: json!({"error": {"message": reason, "type": "injected"}}),
        headers,
    }))
}

enum Step {
    Fail(ModelError),
    Respond(ModelResponse),
    DelayThenInner(u64),
}

/// Wraps the real model and breaks it on purpose, call by call.
pub struct FaultyModel {
    inner: Arc<dyn Model>,
    plan: Arc<FaultPlan>,
    emit: Emitter,
    calls: AtomicU32,
}

impl FaultyModel {
    pub fn new(inner: Arc<dyn Model>, plan: Arc<FaultPlan>, emit: Emitter) -> Self {
        Self {
            inner,
            plan,
            emit,
            calls: AtomicU32::new(0),
        }
    }

    /// The fault hitting call number `n`, if any. Structured-output faults only hit calls that
    /// ask for a structured answer.
    fn fault_for(&self, n: u32, structured: bool, stream: bool) -> Option<LlmFault> {
        let kind = self.plan.llm?;
        if n > self.plan.fail_attempts() {
            return None;
        }
        match kind {
            LlmFault::MaxTurns => None,
            LlmFault::InvalidStructured if !structured => None,
            LlmFault::SlowStream | LlmFault::StreamCut if !stream => None,
            other => Some(other),
        }
    }

    fn step(&self, kind: LlmFault) -> Option<Step> {
        let delay = self.plan.llm_delay_ms;
        let message_item = |content: Value| json!({"type": "message", "role": "assistant", "status": "completed", "content": content});
        let respond = |item: Value| {
            Step::Respond(ModelResponse {
                output: vec![item],
                usage: Usage::default(),
                response_id: None,
                request_id: None,
                raw_usage: None,
            })
        };
        Some(match kind {
            LlmFault::RateLimit429 => Step::Fail(status_error(
                429,
                "rate limit exceeded",
                Some(self.plan.retry_after_secs.unwrap_or(1)),
            )),
            LlmFault::Server500 => Step::Fail(status_error(500, "internal server error", None)),
            LlmFault::Server503 => Step::Fail(status_error(503, "service unavailable", None)),
            LlmFault::BadRequest400 => Step::Fail(status_error(400, "invalid request", None)),
            LlmFault::Unauthorized401 => Step::Fail(status_error(401, "invalid api key", None)),
            LlmFault::ConnectionReset => {
                Step::Fail(connection_error("connection reset by peer", false))
            }
            LlmFault::ConnectTimeout => Step::Fail(connection_error("connect timed out", true)),
            LlmFault::Hang => Step::DelayThenInner(delay.unwrap_or(MAX_DELAY_MS)),
            LlmFault::SlowFirstToken => Step::DelayThenInner(delay.unwrap_or(4_000)),
            LlmFault::EmptyResponse => Step::Fail(ModelError::Behavior(format!(
                "{INJECTED_TAG} empty response body"
            ))),
            LlmFault::MalformedResponse => Step::Fail(ModelError::Behavior(format!(
                "{INJECTED_TAG} malformed response: unexpected end of JSON input"
            ))),
            LlmFault::Refusal => respond(message_item(json!([
                {"type": "refusal", "refusal": "I can't help with that request."}
            ]))),
            LlmFault::InvalidStructured => respond(message_item(json!([
                {"type": "output_text", "text": "this is not json {", "annotations": []}
            ]))),
            LlmFault::SlowStream | LlmFault::StreamCut | LlmFault::MaxTurns => return None,
        })
    }

    async fn announce(&self, kind: LlmFault, n: u32) {
        let detail = match kind {
            LlmFault::RateLimit429 => format!(
                "model call {n} answers HTTP 429 with Retry-After: {}",
                self.plan.retry_after_secs.unwrap_or(1)
            ),
            LlmFault::Server500 => format!("model call {n} answers HTTP 500"),
            LlmFault::Server503 => format!("model call {n} answers HTTP 503"),
            LlmFault::BadRequest400 => format!("model call {n} answers HTTP 400"),
            LlmFault::Unauthorized401 => format!("model call {n} answers HTTP 401"),
            LlmFault::ConnectionReset => format!("model call {n} loses the connection"),
            LlmFault::ConnectTimeout => format!("model call {n} cannot connect (timeout)"),
            LlmFault::Hang => format!(
                "model call {n} hangs for up to {} ms",
                self.plan.llm_delay_ms.unwrap_or(MAX_DELAY_MS)
            ),
            LlmFault::SlowFirstToken => format!(
                "model call {n} waits {} ms before the first token",
                self.plan.llm_delay_ms.unwrap_or(4_000)
            ),
            LlmFault::SlowStream => format!(
                "model call {n} streams slowly ({} ms per chunk)",
                self.plan.llm_delay_ms.unwrap_or(150).min(1_000)
            ),
            LlmFault::StreamCut => format!(
                "model call {n} drops after {} text chunks",
                self.plan.stream_cut_after.unwrap_or(3)
            ),
            LlmFault::EmptyResponse => format!("model call {n} returns an empty body"),
            LlmFault::MalformedResponse => format!("model call {n} returns malformed JSON"),
            LlmFault::Refusal => format!("model call {n} refuses to answer"),
            LlmFault::InvalidStructured => {
                format!("model call {n} answers with text that is not the requested JSON")
            }
            LlmFault::MaxTurns => String::new(),
        };
        self.emit
            .emit(DemoEvent::FaultInjected {
                layer: "llm".into(),
                kind: kind.as_str().into(),
                detail,
            })
            .await;
    }

    async fn report(&self, n: u32, stream: bool, result: &Result<ModelResponse, ModelError>) {
        let (outcome, error_kind) = match result {
            Ok(_) => ("ok", None),
            Err(e) => ("error", Some(model_error_kind(e).to_string())),
        };
        self.emit
            .emit(DemoEvent::ModelAttempt {
                attempt: n,
                stream,
                outcome: outcome.into(),
                error_kind,
            })
            .await;
    }

    /// Forward the inner stream, one text delta at a time, until `limit` deltas went through.
    /// With `cut` the stream then breaks like a dropped connection; with `delay_ms` every delta
    /// is slowed down.
    async fn relay(
        &self,
        request: ModelRequest<'_>,
        raw_tx: mpsc::Sender<Value>,
        cut_after: Option<u32>,
        delay_ms: u64,
    ) -> Result<ModelResponse, ModelError> {
        let (itx, mut irx) = mpsc::channel::<Value>(64);
        let mut inner = Box::pin(self.inner.stream_response(request, itx));
        let mut deltas = 0u32;
        let forward = |ev: Value, deltas: &mut u32| {
            let is_text =
                ev.get("type").and_then(Value::as_str) == Some("response.output_text.delta");
            if is_text {
                *deltas += 1;
            }
            (ev, is_text)
        };
        loop {
            tokio::select! {
                result = &mut inner => {
                    while let Ok(ev) = irx.try_recv() {
                        let (ev, _) = forward(ev, &mut deltas);
                        if raw_tx.send(ev).await.is_err() {
                            break;
                        }
                    }
                    return result;
                }
                Some(ev) = irx.recv() => {
                    let (ev, is_text) = forward(ev, &mut deltas);
                    if is_text && delay_ms > 0 {
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }
                    if raw_tx.send(ev).await.is_err() {
                        return Err(connection_error("consumer went away", false));
                    }
                    if let Some(limit) = cut_after {
                        if deltas >= limit {
                            return Err(connection_error("stream reset mid-response", false));
                        }
                    }
                }
            }
        }
    }
}

#[async_trait]
impl Model for FaultyModel {
    async fn get_response(&self, request: ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let structured = request.output_schema.is_some_and(|s| !s.is_plain_text());
        let fault = self.fault_for(n, structured, false);
        if let Some(kind) = fault {
            self.announce(kind, n).await;
        }
        let result = match fault.and_then(|k| self.step(k)) {
            Some(Step::Fail(e)) => Err(e),
            Some(Step::Respond(r)) => Ok(r),
            Some(Step::DelayThenInner(ms)) => {
                tokio::time::sleep(Duration::from_millis(ms)).await;
                if fault == Some(LlmFault::Hang) {
                    Err(connection_error("model never answered", true))
                } else {
                    self.inner.get_response(request).await
                }
            }
            None => self.inner.get_response(request).await,
        };
        self.report(n, false, &result).await;
        result
    }

    fn get_retry_advice(
        &self,
        request: &opai_agents::ModelRetryAdviceRequest,
    ) -> Option<opai_agents::ModelRetryAdvice> {
        self.inner.get_retry_advice(request)
    }

    async fn stream_response(
        &self,
        request: ModelRequest<'_>,
        raw_tx: mpsc::Sender<Value>,
    ) -> Result<ModelResponse, ModelError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let structured = request.output_schema.is_some_and(|s| !s.is_plain_text());
        let fault = self.fault_for(n, structured, true);
        if let Some(kind) = fault {
            self.announce(kind, n).await;
        }
        let result = match fault {
            Some(LlmFault::StreamCut) => {
                let after = self.plan.stream_cut_after.unwrap_or(3);
                self.relay(request, raw_tx, Some(after), 0).await
            }
            Some(LlmFault::SlowStream) => {
                let delay = self.plan.llm_delay_ms.unwrap_or(150).min(1_000);
                self.relay(request, raw_tx, None, delay).await
            }
            Some(kind) => match self.step(kind) {
                Some(Step::Fail(e)) => Err(e),
                Some(Step::Respond(r)) => {
                    opai_agents::model::replay_response(
                        &raw_tx,
                        &r,
                        "resp_injected",
                        "faulty-model",
                        now_seconds(),
                    )
                    .await;
                    Ok(r)
                }
                Some(Step::DelayThenInner(ms)) => {
                    tokio::time::sleep(Duration::from_millis(ms)).await;
                    if kind == LlmFault::Hang {
                        Err(connection_error("model never answered", true))
                    } else {
                        self.inner.stream_response(request, raw_tx).await
                    }
                }
                None => self.inner.stream_response(request, raw_tx).await,
            },
            None => self.inner.stream_response(request, raw_tx).await,
        };
        self.report(n, true, &result).await;
        result
    }
}

/// A summarizer that can be told to fail, to show that compaction falls back safely.
pub struct FaultySummarizer {
    inner: Arc<ModelSummarizer>,
    fail: bool,
    emit: Emitter,
}

impl FaultySummarizer {
    pub fn new(inner: Arc<ModelSummarizer>, fail: bool, emit: Emitter) -> Self {
        Self { inner, fail, emit }
    }
}

#[async_trait]
impl Summarizer for FaultySummarizer {
    async fn summarize(
        &self,
        previous_summary: Option<&str>,
        items: &[Value],
    ) -> Result<String, AgentsError> {
        if self.fail {
            self.emit
                .emit(DemoEvent::FaultInjected {
                    layer: "session".into(),
                    kind: "summarizer_fail".into(),
                    detail: "the summarizer fails; history stays as it was".into(),
                })
                .await;
            return Err(AgentsError::internal(format!(
                "{INJECTED_TAG} summarizer unavailable"
            )));
        }
        self.inner.summarize(previous_summary, items).await
    }

    fn usage(&self) -> Usage {
        self.inner.usage()
    }
}

/// A session whose reads or writes can fail.
pub struct FaultySession {
    inner: Arc<InMemorySession>,
    mode: Option<SessionFault>,
    emit: Emitter,
}

impl FaultySession {
    pub fn new(inner: Arc<InMemorySession>, mode: Option<SessionFault>, emit: Emitter) -> Self {
        Self { inner, mode, emit }
    }

    async fn fail(&self, kind: &str, what: &str) -> AgentsError {
        self.emit
            .emit(DemoEvent::FaultInjected {
                layer: "session".into(),
                kind: kind.into(),
                detail: format!("the session store fails on {what}"),
            })
            .await;
        AgentsError::internal(format!("{INJECTED_TAG} session store unavailable ({what})"))
    }
}

#[async_trait]
impl Session for FaultySession {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<Value>, AgentsError> {
        if self.mode == Some(SessionFault::Read) {
            return Err(self.fail("read_fail", "read").await);
        }
        self.inner.get_items(limit).await
    }

    async fn add_items(&self, items: Vec<Value>) -> Result<(), AgentsError> {
        if self.mode == Some(SessionFault::Write) {
            return Err(self.fail("write_fail", "write").await);
        }
        self.inner.add_items(items).await
    }

    async fn pop_item(&self) -> Result<Option<Value>, AgentsError> {
        self.inner.pop_item().await
    }

    async fn clear_session(&self) -> Result<(), AgentsError> {
        self.inner.clear_session().await
    }

    async fn replace_items(&self, items: Vec<Value>) -> Result<(), AgentsError> {
        self.inner.replace_items(items).await
    }
}

/// An MCP server decorator: list and call failures, a hanging call, or a dropped connection.
#[derive(Debug)]
pub struct FaultyMcpServer {
    inner: Arc<dyn McpServer>,
    fault: Option<McpFault>,
    emit: Emitter,
}

impl FaultyMcpServer {
    pub fn new(inner: Arc<dyn McpServer>, fault: Option<McpFault>, emit: Emitter) -> Self {
        Self { inner, fault, emit }
    }

    async fn injected(&self, kind: &str, detail: &str) {
        self.emit
            .emit(DemoEvent::FaultInjected {
                layer: "mcp".into(),
                kind: kind.into(),
                detail: detail.into(),
            })
            .await;
    }
}

#[async_trait]
impl McpServer for FaultyMcpServer {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn connect(&self) -> Result<(), AgentsError> {
        self.inner.connect().await
    }

    async fn cleanup(&self) -> Result<(), AgentsError> {
        self.inner.cleanup().await
    }

    async fn list_tools(
        &self,
        context: &RunContextWrapper,
        agent: &Agent,
    ) -> Result<Vec<McpTool>, AgentsError> {
        if self.fault == Some(McpFault::ListFail) {
            self.injected("list_fail", "tools/list fails").await;
            return Err(AgentsError::internal(format!(
                "{INJECTED_TAG} MCP tools/list failed"
            )));
        }
        self.inner.list_tools(context, agent).await
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> Result<McpCallToolResult, AgentsError> {
        match self.fault {
            Some(McpFault::CallFail) => {
                self.injected("call_fail", "tools/call fails").await;
                Err(AgentsError::tool(format!(
                    "{INJECTED_TAG} MCP tools/call failed"
                )))
            }
            Some(McpFault::CallHang) => {
                self.injected("call_hang", "tools/call hangs for 15 s")
                    .await;
                tokio::time::sleep(Duration::from_millis(MAX_DELAY_MS)).await;
                Err(AgentsError::tool(format!(
                    "{INJECTED_TAG} MCP tools/call never answered"
                )))
            }
            Some(McpFault::Disconnect) => {
                self.injected(
                    "disconnect",
                    "the stdio connection was closed before this call; the client reconnects on demand",
                )
                .await;
                let _ = self.inner.cleanup().await;
                self.inner.call_tool(name, arguments).await
            }
            _ => self.inner.call_tool(name, arguments).await,
        }
    }

    fn invalidate_tools_cache(&self) {
        self.inner.invalidate_tools_cache();
    }

    fn needs_resolution_for(&self, tool: &McpTool) -> opai_agents::NeedsResolution {
        self.inner.needs_resolution_for(tool)
    }

    fn use_structured_content(&self) -> bool {
        self.inner.use_structured_content()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opai_agents::testing::{assistant_message, ModelStep, ScriptedModel};
    use opai_agents::{
        retry_policies, Agent, ModelRetrySettings, ModelSettings, RunConfig, RunOptions, Runner,
    };

    fn plan(json: Value) -> FaultPlan {
        let mut p: FaultPlan = serde_json::from_value(json).expect("valid plan");
        p.sanitize();
        p
    }

    #[test]
    fn sanitize_clamps_numbers() {
        let p = plan(json!({
            "llm_delay_ms": 999999,
            "llm_fail_attempts": 99,
            "tool": {"fail_times": 50, "timeout_secs": 0.0},
            "sse_drop_after": 0,
            "approval_ttl_secs": 100000
        }));
        assert_eq!(p.llm_delay_ms, Some(MAX_DELAY_MS));
        assert_eq!(p.llm_fail_attempts, Some(MAX_FAIL_ATTEMPTS));
        assert_eq!(p.tool.fail_times, Some(MAX_FAIL_ATTEMPTS));
        assert_eq!(p.tool.timeout_secs, Some(0.2));
        assert_eq!(p.sse_drop_after, Some(1));
        assert_eq!(p.approval_ttl_secs, Some(600));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let r: Result<FaultPlan, _> = serde_json::from_value(json!({"shell": "rm -rf /"}));
        assert!(r.is_err());
        let r: Result<FaultPlan, _> = serde_json::from_value(json!({"llm": "explode"}));
        assert!(r.is_err());
    }

    fn agent_with(
        plan_json: Value,
        retry: Option<ModelRetrySettings>,
    ) -> (Agent, mpsc::Receiver<DemoEvent>) {
        let (tx, rx) = mpsc::channel(256);
        let inner: Arc<dyn Model> = Arc::new(ScriptedModel::new([
            ModelStep::output([assistant_message("hello")]),
            ModelStep::output([assistant_message("hello")]),
            ModelStep::output([assistant_message("hello")]),
        ]));
        let faulty = FaultyModel::new(inner, Arc::new(plan(plan_json)), Emitter::new(tx));
        let mut agent = Agent::new("t").model(Arc::new(faulty));
        agent.model_settings = ModelSettings {
            retry,
            ..Default::default()
        };
        (agent, rx)
    }

    async fn drain(mut streamed: opai_agents::RunResultStreaming) -> Result<(), AgentsError> {
        while let Some(ev) = streamed.next_event().await {
            ev?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn rate_limit_fails_without_a_policy() {
        let (agent, mut rx) = agent_with(json!({"llm": "rate_limit_429"}), None);
        let streamed = Runner::run_streamed(agent, "hi", RunOptions::default());
        let err = drain(streamed).await.expect_err("429 must surface");
        assert!(err.to_string().contains("429"));
        let mut attempts = 0;
        while let Ok(ev) = rx.try_recv() {
            if matches!(ev, DemoEvent::ModelAttempt { .. }) {
                attempts += 1;
            }
        }
        assert_eq!(attempts, 1);
    }

    #[tokio::test]
    async fn rate_limit_recovers_with_a_policy() {
        let retry = ModelRetrySettings::new(2, retry_policies::http_status([429]));
        let (agent, mut rx) = agent_with(json!({"llm": "rate_limit_429"}), Some(retry));
        let mut options = RunOptions::default();
        options.run_config = RunConfig::default();
        let streamed = Runner::run_streamed(agent, "hi", options);
        drain(streamed).await.expect("second attempt succeeds");
        let mut outcomes = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            if let DemoEvent::ModelAttempt { outcome, .. } = ev {
                outcomes.push(outcome);
            }
        }
        assert_eq!(outcomes, vec!["error", "ok"]);
    }
}
