//! The HTTP side: ingest, a live feed, and the page.
//!
//! * `POST /ingest` — NDJSON from the SDK (or anything else).
//! * `GET /api/stream` — server-sent events, one per ingested item.
//! * `GET /api/runs`, `GET /api/runs/{id}` — what is stored.
//! * `GET /` — the page.

use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tokio::sync::broadcast::error::RecvError;

use crate::store::Store;

const INDEX_HTML: &str = include_str!("../assets/index.html");

/// Largest body `POST /ingest` takes.
pub const MAX_INGEST_BYTES: usize = 4 * 1024 * 1024;

/// Most browsers watching the live feed at once.
const MAX_VIEWERS: usize = 32;

/// What the handlers share.
pub struct AppState {
    store: Arc<Store>,
    token: Option<String>,
    nonce: String,
    viewers: AtomicUsize,
}

impl AppState {
    /// State over `store`. With a `token`, `POST /ingest` must carry it as a bearer token.
    pub fn new(store: Arc<Store>, token: Option<String>) -> Arc<Self> {
        Arc::new(Self {
            store,
            token: token.filter(|t| !t.is_empty()),
            nonce: uuid::Uuid::new_v4().simple().to_string(),
            viewers: AtomicUsize::new(0),
        })
    }
}

fn error(status: StatusCode, kind: &str, message: &str) -> Response {
    (status, Json(json!({"kind": kind, "message": message}))).into_response()
}

/// Build the router. `allowed_hosts` pins the `Host` header, which defeats DNS rebinding against a
/// server that only listens on loopback.
pub fn router(app: Arc<AppState>, allowed_hosts: Option<Vec<String>>) -> Router {
    let hosts = Arc::new(allowed_hosts);
    Router::new()
        .route("/", get(index))
        .route("/api/health", get(health))
        .route("/api/runs", get(runs))
        .route("/api/runs/{id}", get(run))
        .route("/api/stream", get(stream))
        .route(
            "/ingest",
            post(ingest).layer(DefaultBodyLimit::max(MAX_INGEST_BYTES)),
        )
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
        "ok": true,
        "runs": app.store.runs().len(),
        "ingest_requires_token": app.token.is_some(),
    }))
}

async fn runs(State(app): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({"runs": app.store.runs()}))
}

async fn run(State(app): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match app.store.run(&id) {
        Some(view) => Json(view).into_response(),
        None => error(StatusCode::NOT_FOUND, "unknown_run", "No run with this id."),
    }
}

/// Compare without stopping at the first difference, so timing does not reveal the token.
fn same_token(given: &str, expected: &str) -> bool {
    let (a, b) = (given.as_bytes(), expected.as_bytes());
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        diff |= usize::from(a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0));
    }
    diff == 0
}

fn authorized(app: &AppState, headers: &HeaderMap) -> bool {
    let Some(expected) = app.token.as_deref() else {
        return true;
    };
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|given| same_token(given, expected))
}

/// A page on the web can send a form-style POST to a loopback address without asking first; it
/// cannot send these types, which make the browser ask (and be refused). Requiring one keeps a
/// stranger's page from filling the console.
fn json_content_type(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            v.split(';')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase()
        })
        .is_some_and(|t| t == "application/x-ndjson" || t == "application/json")
}

async fn ingest(State(app): State<Arc<AppState>>, headers: HeaderMap, body: String) -> Response {
    if !authorized(&app, &headers) {
        return error(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "Send the console's token as `Authorization: Bearer <token>`.",
        );
    }
    if !json_content_type(&headers) {
        return error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "bad_content_type",
            "Send `Content-Type: application/x-ndjson`.",
        );
    }
    let report = app.store.ingest_ndjson(&body);
    Json(json!({"accepted": report.accepted, "rejected": report.rejected})).into_response()
}

/// Gives a viewer's slot back when its connection ends.
struct Viewer(Arc<AppState>);

impl Drop for Viewer {
    fn drop(&mut self) {
        self.0.viewers.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn stream(State(app): State<Arc<AppState>>) -> Response {
    if app.viewers.fetch_add(1, Ordering::SeqCst) >= MAX_VIEWERS {
        app.viewers.fetch_sub(1, Ordering::SeqCst);
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "too_many_viewers",
            "Too many browsers are watching. Close one and try again.",
        );
    }
    let viewer = Viewer(Arc::clone(&app));
    let feed = app.store.subscribe();
    let events = futures::stream::unfold((feed, viewer), |(mut feed, viewer)| async move {
        match feed.recv().await {
            Ok(message) => {
                let event = Event::default().data(message.to_string());
                Some((Ok::<_, Infallible>(event), (feed, viewer)))
            }
            // The viewer fell behind: tell it, so it reloads what it missed.
            Err(RecvError::Lagged(_)) => {
                let event = Event::default().event("lagged").data("{}");
                Some((Ok(event), (feed, viewer)))
            }
            Err(RecvError::Closed) => None,
        }
    });
    let mut response = Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response();
    response
        .headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_compare_equal_only_when_identical() {
        assert!(same_token("abc", "abc"));
        assert!(!same_token("abc", "abd"));
        assert!(!same_token("abc", "abcd"));
        assert!(!same_token("", "abc"));
    }

    #[test]
    fn only_non_simple_content_types_are_accepted() {
        let with = |value: &str| {
            let mut h = HeaderMap::new();
            h.insert(header::CONTENT_TYPE, HeaderValue::from_str(value).unwrap());
            h
        };
        assert!(json_content_type(&with("application/x-ndjson")));
        assert!(json_content_type(&with("application/json; charset=utf-8")));
        assert!(!json_content_type(&with("text/plain")));
        assert!(!json_content_type(&with(
            "application/x-www-form-urlencoded"
        )));
        assert!(!json_content_type(&HeaderMap::new()));
    }
}
