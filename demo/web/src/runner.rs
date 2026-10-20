//! One run, start to finish: build the agents, stream the SDK events into the SSE channel, watch
//! for cancellation and for the browser leaving, pause for approvals, and report what happened
//! after the stream ends (session, usage, trace).

use std::convert::Infallible;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use opai_agents::teams::Team;
use opai_agents::{
    AgentsError, CancelMode, CompactingSession, ContextValue, McpServer, Model, RunOptions,
    RunState, Runner, Session, StreamEvent, ToolExecutionConfig,
};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::agents;
use crate::events::{
    classify, map_stream_event, tripwire_event, usage_event, DemoCtx, DemoEvent, Emitter,
    InterruptionDto,
};
use crate::faults::{
    FaultPlan, FaultyMcpServer, FaultyModel, FaultySession, FaultySummarizer, LlmFault,
    SessionFault,
};
use crate::hooks::TimelineHooks;
use crate::state::{
    AppState, BusyGuard, Conversation, PendingRun, RunSettings, RunSlot, COMPACTION_TRIGGER_ITEMS,
    DEFAULT_APPROVAL_TTL_SECS,
};
use crate::team;

/// The whole run is cut off after this long, whatever the faults. Longer than a solo run: a team
/// adds a teammate run per helper, each with model turns of its own.
const RUN_DEADLINE: Duration = Duration::from_secs(120);
const EVENT_BUFFER: usize = 256;

/// How long a finishing run waits for its teammates to leave before dropping them.
const TEAM_GRACE: Duration = Duration::from_millis(300);

/// Forwards the Lead's team events into the run's SSE stream, and stops with the run.
///
/// The task holds an `Emitter` clone: without aborting it the stream would never close after the
/// run finished, because the channel would still have a sender.
struct TeamFeed(JoinHandle<()>);

impl Drop for TeamFeed {
    fn drop(&mut self) {
        self.0.abort();
    }
}

fn feed(team: &Team, emit: &Emitter) -> TeamFeed {
    let mut events = team.subscribe();
    let emit = emit.clone();
    TeamFeed(tokio::spawn(async move {
        loop {
            match events.recv().await {
                Ok(event) => {
                    let (kind, detail) = team::describe(&event);
                    emit.emit(DemoEvent::Team {
                        kind: kind.to_string(),
                        detail: crate::events::truncate(&detail, 200),
                    })
                    .await;
                }
                // A lagging subscriber skips ahead; the timeline is best effort.
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    }))
}

/// Take a finished run's team down: ask the teammates to leave, wait a moment, then drop them.
async fn close(team: Option<Team>, emit: &Emitter) {
    let Some(team) = team else { return };
    team.shutdown(TEAM_GRACE).await;
    let usage = team.handle().usage();
    emit.emit(DemoEvent::Team {
        kind: "team usage".into(),
        detail: format!("{} runs, {} tokens", usage.runs, usage.tokens),
    })
    .await;
}

/// What the run starts from.
pub enum Start {
    Fresh(String),
    Resume(Box<RunState>),
}

pub struct RunRequest {
    pub run_id: String,
    pub tenant: String,
    pub conversation: Arc<Conversation>,
    pub guard: BusyGuard,
    pub slot: RunSlot,
    pub start: Start,
    pub settings: RunSettings,
    pub faults: Arc<FaultPlan>,
}

/// Start the run in the background and return its event stream as the HTTP response.
pub fn spawn_run(app: Arc<AppState>, request: RunRequest) -> impl IntoResponse {
    let (tx, rx) = mpsc::channel::<DemoEvent>(EVENT_BUFFER);
    let sse_delay = request.faults.sse_delay_ms.map(Duration::from_millis);
    let drop_after = request.faults.sse_drop_after;
    tokio::spawn(drive(app, request, Emitter::new(tx)));

    let body = futures::stream::unfold((rx, 0u32), move |(mut rx, sent)| async move {
        if drop_after.is_some_and(|limit| sent >= limit) {
            return None;
        }
        let event = rx.recv().await?;
        if let Some(delay) = sse_delay {
            tokio::time::sleep(delay).await;
        }
        let sse = Event::default()
            .json_data(&event)
            .unwrap_or_else(|_| Event::default().data("{}"));
        Some((Ok::<_, Infallible>(sse), (rx, sent + 1)))
    });
    Sse::new(body).keep_alive(KeepAlive::default())
}

enum Tick {
    Closed,
    Cancel(CancelMode),
    Deadline,
    // Boxed: `AgentsError` carries a guardrail result, which makes it far larger than the other
    // variants.
    Event(Box<Option<Result<StreamEvent, AgentsError>>>),
}

async fn drive(app: Arc<AppState>, request: RunRequest, emit: Emitter) {
    let RunRequest {
        run_id,
        tenant,
        conversation,
        guard: _guard,
        slot: _slot,
        start,
        settings,
        faults,
    } = request;

    let (cancel_tx, mut cancel_rx) = mpsc::unbounded_channel::<CancelMode>();
    app.register_run(&run_id, &conversation.id, cancel_tx);
    app.trace.register(&run_id);
    emit.emit(DemoEvent::RunStarted {
        run_id: run_id.clone(),
        resumed: matches!(start, Start::Resume(_)),
    })
    .await;

    // ---- build the per-run world ----
    let model: Arc<dyn Model> = Arc::new(FaultyModel::new(
        Arc::clone(&app.model),
        Arc::clone(&faults),
        emit.clone(),
    ));
    let mcp = app.mcp.as_ref().map(|server| {
        Arc::new(FaultyMcpServer::new(
            Arc::clone(server),
            faults.mcp,
            emit.clone(),
        )) as Arc<dyn McpServer>
    });
    // The team is per run, like the agent network: the Lead's tools read its handle from the run
    // options, and dropping it at the end of `drive` stops whatever is left of the teammates.
    let team = match team::build(Arc::clone(&model), &run_id, &faults.tool).await {
        Ok(team) => Some(team),
        Err(error) => {
            // A run without a team still works: the Lead is simply not offered.
            emit.emit(DemoEvent::Team {
                kind: "team unavailable".into(),
                detail: crate::events::truncate(&error.to_string(), 200),
            })
            .await;
            None
        }
    };
    let _team_feed = team.as_ref().map(|team| feed(team, &emit));

    let agent = match agents::build_network(model, &faults.tool, mcp, team.as_ref()) {
        Ok(agent) => agent,
        Err(e) => {
            emit.emit(classify(&e).into_event()).await;
            app.finish_run(&run_id, "failed");
            let _ = app.trace.take(&run_id);
            return;
        }
    };

    let summarizer = Arc::new(FaultySummarizer::new(
        Arc::clone(&conversation.summarizer),
        faults.session == Some(SessionFault::Summarizer),
        emit.clone(),
    ));
    let store = Arc::new(FaultySession::new(
        Arc::clone(&conversation.session),
        faults.session,
        emit.clone(),
    ));
    let session: Arc<dyn Session> = CompactingSession::new(store, summarizer)
        .trigger_items(COMPACTION_TRIGGER_ITEMS)
        .keep_recent_turns(2)
        .shared();
    let summary_before = conversation.summary().await;

    let ctx = Arc::new(DemoCtx {
        tenant,
        emit: emit.clone(),
        flaky_calls: AtomicU32::new(0),
        tool_calls: AtomicU32::new(0),
    });
    // The Lead's options must carry the `TeamHandle`; every other field is the demo's own. The
    // team's base options start from `RunOptions::default`, so nothing else comes along.
    let mut options = match &team {
        Some(team) => team.lead_options(),
        None => RunOptions::default(),
    };
    options.max_turns = Some(if faults.llm == Some(LlmFault::MaxTurns) {
        1
    } else {
        settings.max_turns.unwrap_or(10)
    });
    options.context = Some(Arc::clone(&ctx) as ContextValue);
    options.hooks = Some(Arc::new(TimelineHooks));
    options.session = Some(session);
    options.error_handlers = agents::error_handlers();
    // Set the fields in place: replacing the whole `RunConfig` would drop the trace workflow name
    // and group id the team put there.
    options.run_config.tracing.workflow_name = Some(run_id.clone());
    options.run_config.model.settings = Some(settings.model_settings());
    if let Some(n) = settings.tool_concurrency {
        options.run_config.tools.execution = Some(ToolExecutionConfig {
            max_function_tool_concurrency: Some(n),
        });
    }

    let mut streamed = match start {
        Start::Fresh(input) => Runner::run_streamed(agent, input, options),
        Start::Resume(state) => match Runner::run_state_streamed(agent, *state, options) {
            Ok(s) => s,
            Err(e) => {
                emit.emit(classify(&e).into_event()).await;
                app.finish_run(&run_id, "failed");
                let _ = app.trace.take(&run_id);
                return;
            }
        },
    }
    .cancel_on_drop(true);

    // ---- stream ----
    let deadline = tokio::time::sleep(RUN_DEADLINE);
    tokio::pin!(deadline);
    let mut delivered = 0usize;
    let mut cancelled: Option<CancelMode> = None;
    let mut errored = false;
    let mut timed_out = false;
    let mut disconnected = false;

    loop {
        let tick = tokio::select! {
            biased;
            _ = emit.closed() => Tick::Closed,
            Some(mode) = cancel_rx.recv() => Tick::Cancel(mode),
            _ = &mut deadline, if !timed_out => Tick::Deadline,
            event = streamed.next_event() => Tick::Event(Box::new(event)),
        };
        match tick {
            Tick::Closed => {
                disconnected = true;
                break;
            }
            Tick::Cancel(mode) => {
                streamed.cancel(mode);
                cancelled = Some(mode);
            }
            Tick::Deadline => {
                timed_out = true;
                streamed.cancel(CancelMode::Immediate);
                emit.emit(DemoEvent::Error {
                    kind: "run_timeout".into(),
                    message: format!(
                        "The run exceeded {} seconds and was stopped.",
                        RUN_DEADLINE.as_secs()
                    ),
                    retryable: true,
                    hint: "Shorten the injected delays or hangs, or lower the per-attempt timeout."
                        .into(),
                    injected: !faults.is_empty(),
                })
                .await;
                errored = true;
            }
            Tick::Event(inner) => match *inner {
                None => break,
                Some(Ok(event)) => {
                    for out in map_stream_event(event, &mut delivered) {
                        emit.emit(out).await;
                    }
                }
                Some(Err(err)) => {
                    report_error(&emit, &err, delivered).await;
                    errored = true;
                    break;
                }
            },
        }
    }

    if disconnected {
        // Dropping `streamed` below cancels the model and tool calls (`cancel_on_drop`).
        drop(streamed);
        close(team, &emit).await;
        app.finish_run(&run_id, "cancelled_on_disconnect");
        let _ = app.trace.take(&run_id);
        return;
    }

    // ---- what the run ended with ----
    let (interruptions, final_output, usage, last_agent, is_cancelled, snapshot_error) = {
        let snap = streamed
            .snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (
            snap.interruptions.clone(),
            snap.final_output.clone(),
            snap.usage.clone(),
            snap.current_agent_name.clone(),
            snap.is_cancelled,
            snap.error.clone(),
        )
    };

    let mut status = "completed";
    if !interruptions.is_empty() && !errored && cancelled.is_none() {
        status = "interrupted";
        match streamed.to_state() {
            Ok(state) => {
                let ttl = faults
                    .approval_ttl_secs
                    .unwrap_or(DEFAULT_APPROVAL_TTL_SECS);
                let items: Vec<InterruptionDto> = interruptions
                    .iter()
                    .map(|i| InterruptionDto {
                        call_id: i.call_id.clone(),
                        agent: i.agent_name.clone(),
                        tool: i.tool_name.clone(),
                        arguments: crate::events::truncate(&i.arguments, 500),
                    })
                    .collect();
                app.store_pending(
                    run_id.clone(),
                    PendingRun {
                        state,
                        interruptions,
                        conversation_id: conversation.id.clone(),
                        settings: settings.clone(),
                        faults: Arc::clone(&faults),
                        expires: Instant::now() + Duration::from_secs(ttl),
                    },
                );
                emit.emit(DemoEvent::Interruption {
                    run_id: run_id.clone(),
                    expires_in_secs: ttl,
                    items,
                })
                .await;
            }
            Err(e) => {
                report_error(&emit, &e, delivered).await;
                status = "failed";
            }
        }
    } else if let Some(mode) = cancelled {
        status = "cancelled";
        emit.emit(DemoEvent::Cancelled {
            mode: match mode {
                CancelMode::Immediate => "immediate".into(),
                CancelMode::AfterTurn => "after_turn".into(),
            },
        })
        .await;
    } else if errored {
        status = "failed";
    } else if let Some(output) = final_output.filter(|_| !is_cancelled) {
        let (text, structured) = match &output {
            serde_json::Value::String(s) => (s.clone(), None),
            other => (other.to_string(), Some(other.clone())),
        };
        emit.emit(DemoEvent::Final {
            text,
            structured,
            last_agent,
        })
        .await;
    } else if let Some(message) = snapshot_error {
        emit.emit(DemoEvent::Error {
            kind: "internal".into(),
            message: crate::events::truncate(&message, 400),
            retryable: false,
            hint: "Unexpected failure in the run. Resend, or start a new conversation.".into(),
            injected: false,
        })
        .await;
        status = "failed";
    }

    // The Lead has finished (or stopped). Its teammates are independent tasks: they would keep
    // calling the model, so they are taken down before the run is reported over.
    close(team, &emit).await;

    emit.emit(usage_event(&usage)).await;
    emit_session(&emit, &conversation, summary_before).await;
    let spans = app.trace.take(&run_id);
    if !spans.is_empty() {
        emit.emit(DemoEvent::Trace { spans }).await;
    }
    app.finish_run(&run_id, status);
}

async fn report_error(emit: &Emitter, err: &AgentsError, delivered: usize) {
    if let Some(event) = tripwire_event(err) {
        emit.emit(event).await;
    }
    let mut info = classify(err);
    let cut_mid_stream = delivered > 0
        && matches!(
            err,
            AgentsError::Model(opai_agents::ModelError::Connection(_))
        );
    if cut_mid_stream {
        emit.emit(DemoEvent::StreamInterrupted {
            delivered_chars: delivered,
        })
        .await;
        info.kind = "stream_interrupted".into();
        info.retryable = true;
        info.hint = "The connection broke after part of the answer was delivered. The SDK never retries a stream that already emitted output; the partial text is kept, resend to get a full answer.".into();
    }
    emit.emit(info.into_event()).await;
}

async fn emit_session(emit: &Emitter, conversation: &Conversation, before: Option<String>) {
    let items = conversation
        .session
        .get_items(None)
        .await
        .unwrap_or_default();
    let count = items.len();
    let summary = items.iter().find_map(crate::state::summary_of);
    let compacted = summary.is_some() && summary != before;
    emit.emit(DemoEvent::Session {
        count,
        until_compaction: (COMPACTION_TRIGGER_ITEMS + 1).saturating_sub(count),
        compacted,
        compaction_failed: !compacted && count > COMPACTION_TRIGGER_ITEMS,
        summary,
    })
    .await;
}
