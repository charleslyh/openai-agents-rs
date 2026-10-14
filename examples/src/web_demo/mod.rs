//! The web demo: an axum server with one chat endpoint that streams a unified agent network as
//! server-sent events, plus the small control endpoints around it (approve, cancel, conversations,
//! debug). Fault injection and the debug endpoints are only on when `AppState.debug` says so.

pub mod agents;
pub mod events;
pub mod faults;
pub mod hooks;
pub mod mcp_server;
pub mod runner;
pub mod state;
pub mod tools;
pub mod trace;

use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Path, Query, Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use openai_agents::CancelMode;
use serde::Deserialize;
use serde_json::{json, Value};

use faults::FaultPlan;
use runner::{spawn_run, RunRequest, Start};
use state::{new_id, AppState, RunSettings, TakeError};

const INDEX_HTML: &str = include_str!("index.html");
const MAX_BODY_BYTES: usize = 16 * 1024;
const MAX_INPUT_CHARS: usize = 2_000;

fn error(status: StatusCode, kind: &str, message: &str) -> Response {
    (status, Json(json!({"kind": kind, "message": message}))).into_response()
}

fn not_found_conversation() -> Response {
    error(
        StatusCode::NOT_FOUND,
        "unknown_conversation",
        "This conversation no longer exists (it expired, or the server state was reset). Start a new one.",
    )
}

fn require_debug(app: &AppState) -> Result<(), Response> {
    if app.debug {
        Ok(())
    } else {
        Err(error(
            StatusCode::FORBIDDEN,
            "debug_disabled",
            "Debug features are off. Set DEMO_DEBUG=1 to enable them.",
        ))
    }
}

/// Build the router. `allowed_hosts` (when set) pins the `Host` header, which defeats DNS
/// rebinding against a server that only listens on loopback.
pub fn router(app: Arc<AppState>, allowed_hosts: Option<Vec<String>>) -> Router {
    let hosts = Arc::new(allowed_hosts);
    Router::new()
        .route("/", get(index))
        .route("/api/health", get(health))
        .route("/api/conversations", post(create_conversation))
        .route(
            "/api/conversations/{id}",
            get(show_conversation).delete(delete_conversation),
        )
        .route("/api/chat", post(chat))
        .route("/api/approve", post(approve))
        .route("/api/cancel", post(cancel))
        .route("/api/debug/runs", get(debug_runs))
        .route("/api/debug/reset", post(debug_reset))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(middleware::from_fn_with_state(hosts, guard_headers))
        .with_state(app)
}

/// Host check on the way in, hardening headers on the way out.
async fn guard_headers(
    State(hosts): State<Arc<Option<Vec<String>>>>,
    request: Request,
    next: Next,
) -> Response {
    if let Some(allowed) = hosts.as_ref() {
        let host = request
            .headers()
            .get(header::HOST)
            .and_then(|h| h.to_str().ok())
            .unwrap_or("");
        if !allowed.iter().any(|a| a.eq_ignore_ascii_case(host)) {
            return error(StatusCode::FORBIDDEN, "bad_host", "Unexpected Host header.");
        }
    }
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    response
}

async fn index(State(app): State<Arc<AppState>>) -> Response {
    let csp = format!(
        "default-src 'none'; script-src 'nonce-{n}'; style-src 'nonce-{n}'; connect-src 'self'; \
         img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        n = app.nonce
    );
    let mut response = Html(INDEX_HTML.replace("__NONCE__", &app.nonce)).into_response();
    if let Ok(value) = HeaderValue::from_str(&csp) {
        response
            .headers_mut()
            .insert(header::CONTENT_SECURITY_POLICY, value);
    }
    response
}

async fn health(State(app): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "model": app.model_name,
        "api": app.api,
        "debug": app.debug,
        "mcp": app.mcp.is_some(),
        "compaction_trigger_items": state::COMPACTION_TRIGGER_ITEMS,
        "input_limit": MAX_INPUT_CHARS,
    }))
}

async fn create_conversation(State(app): State<Arc<AppState>>) -> Response {
    match app.create_conversation() {
        Some(c) => Json(json!({"conversation_id": c.id})).into_response(),
        None => error(
            StatusCode::TOO_MANY_REQUESTS,
            "too_many_conversations",
            "The server is full of active conversations. Try again in a moment.",
        ),
    }
}

async fn conversation_view(app: &AppState, id: &str) -> Option<Value> {
    use openai_agents::Session;
    let conversation = app.conversation(id)?;
    let items = conversation
        .session
        .get_items(None)
        .await
        .unwrap_or_default();
    let count = items.len();
    Some(json!({
        "conversation_id": id,
        "count": count,
        "until_compaction": (state::COMPACTION_TRIGGER_ITEMS + 1).saturating_sub(count),
        "summary": items.iter().find_map(state::summary_of),
    }))
}

async fn show_conversation(State(app): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match conversation_view(&app, &id).await {
        Some(view) => Json(view).into_response(),
        None => not_found_conversation(),
    }
}

async fn delete_conversation(State(app): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    if app.remove_conversation(&id) {
        StatusCode::NO_CONTENT.into_response()
    } else {
        not_found_conversation()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatBody {
    conversation_id: String,
    input: String,
    #[serde(default)]
    settings: RunSettings,
    #[serde(default)]
    debug: FaultPlan,
}

fn effective_faults(app: &AppState, mut plan: FaultPlan) -> Arc<FaultPlan> {
    if !app.debug {
        plan = FaultPlan::default();
    }
    plan.sanitize();
    Arc::new(plan)
}

async fn chat(State(app): State<Arc<AppState>>, Json(mut body): Json<ChatBody>) -> Response {
    let input = body.input.trim().to_string();
    if input.is_empty() || input.chars().count() > MAX_INPUT_CHARS {
        return error(
            StatusCode::BAD_REQUEST,
            "bad_input",
            &format!("The message must be 1 to {MAX_INPUT_CHARS} characters."),
        );
    }
    let Some(conversation) = app.conversation(&body.conversation_id) else {
        return not_found_conversation();
    };
    let Some(guard) = app.lock_conversation(&conversation) else {
        return error(
            StatusCode::CONFLICT,
            "conversation_busy",
            "This conversation is still answering the previous message.",
        );
    };
    let Some(slot) = app.acquire_slot() else {
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "too_many_runs",
            "Too many runs in progress. Try again in a moment.",
        );
    };
    body.settings.sanitize();
    let faults = effective_faults(&app, body.debug);
    let tenant = format!("acme-{}", &conversation.id[..6]);
    let request = RunRequest {
        run_id: new_id(),
        tenant,
        conversation,
        guard,
        slot,
        start: Start::Fresh(input),
        settings: body.settings,
        faults,
    };
    sse_response(spawn_run(app, request).into_response())
}

fn sse_response(mut response: Response) -> Response {
    response
        .headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
    response
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Decision {
    call_id: String,
    approve: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ApproveBody {
    conversation_id: String,
    run_id: String,
    decisions: Vec<Decision>,
}

async fn approve(State(app): State<Arc<AppState>>, Json(body): Json<ApproveBody>) -> Response {
    let Some(conversation) = app.conversation(&body.conversation_id) else {
        return not_found_conversation();
    };
    let Some(guard) = app.lock_conversation(&conversation) else {
        return error(
            StatusCode::CONFLICT,
            "conversation_busy",
            "This conversation is still busy. Try the decision again in a moment.",
        );
    };
    let call_ids: Vec<String> = body.decisions.iter().map(|d| d.call_id.clone()).collect();
    let pending = match app.take_pending(&body.run_id, &conversation.id, &call_ids) {
        Ok(p) => p,
        Err(TakeError::Unknown) => {
            return error(
                StatusCode::NOT_FOUND,
                "unknown_approval",
                "No pending approval with this id. It was already decided, or the server state was reset.",
            )
        }
        Err(TakeError::Expired) => {
            return error(
                StatusCode::GONE,
                "approval_expired",
                "This approval expired. Ask the question again.",
            )
        }
        Err(TakeError::BadCall(call)) => {
            return error(
                StatusCode::BAD_REQUEST,
                "unknown_call_id",
                &format!("The decision does not match the paused calls ({call})."),
            )
        }
    };
    let Some(slot) = app.acquire_slot() else {
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "too_many_runs",
            "Too many runs in progress. Try again in a moment.",
        );
    };
    let mut run_state = pending.state;
    for decision in &body.decisions {
        if let Some(item) = pending
            .interruptions
            .iter()
            .find(|i| i.call_id == decision.call_id)
        {
            if decision.approve {
                run_state.approve(item, false);
            } else {
                run_state.reject(item, false, None);
            }
        }
    }
    let request = RunRequest {
        run_id: new_id(),
        tenant: format!("acme-{}", &conversation.id[..6]),
        conversation,
        guard,
        slot,
        start: Start::Resume(Box::new(run_state)),
        settings: pending.settings,
        faults: pending.faults,
    };
    sse_response(spawn_run(app, request).into_response())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CancelBody {
    conversation_id: String,
    run_id: String,
    mode: String,
}

async fn cancel(State(app): State<Arc<AppState>>, Json(body): Json<CancelBody>) -> Response {
    let mode = match body.mode.as_str() {
        "immediate" => CancelMode::Immediate,
        "after_turn" => CancelMode::AfterTurn,
        _ => {
            return error(
                StatusCode::BAD_REQUEST,
                "bad_mode",
                "mode must be `immediate` or `after_turn`.",
            )
        }
    };
    match app.cancel_run(&body.run_id, &body.conversation_id, mode) {
        Some(()) => StatusCode::ACCEPTED.into_response(),
        None => error(
            StatusCode::NOT_FOUND,
            "unknown_run",
            "That run is unknown or already finished.",
        ),
    }
}

#[derive(Deserialize)]
struct DebugQuery {
    conversation_id: String,
}

async fn debug_runs(State(app): State<Arc<AppState>>, Query(q): Query<DebugQuery>) -> Response {
    if let Err(response) = require_debug(&app) {
        return response;
    }
    let runs: Vec<Value> = app
        .runs_for(&q.conversation_id)
        .into_iter()
        .map(|(id, status, age)| json!({"run_id": id, "status": status, "age_secs": age}))
        .collect();
    Json(json!({"runs": runs, "active": app.active.load(std::sync::atomic::Ordering::SeqCst)}))
        .into_response()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResetBody {
    conversation_id: String,
}

/// Simulate a server restart for one conversation: its history and pending approvals vanish.
async fn debug_reset(State(app): State<Arc<AppState>>, Json(body): Json<ResetBody>) -> Response {
    if let Err(response) = require_debug(&app) {
        return response;
    }
    let existed = app.remove_conversation(&body.conversation_id);
    Json(json!({"reset": existed})).into_response()
}
