//! The team: shared state, the handle every member and tool works through, and the builder.
//!
//! [`Team`] is the owner. Dropping it stops every teammate. [`TeamHandle`] is what code running
//! inside the team holds (a tool, a teammate's loop): it carries the caller's identity, so a sender
//! can never be forged from tool arguments, and it can be cloned freely.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use opai_agents_core::{RunOptions, Runtime, Session, ToolResolution};
use serde_json::json;
use tokio::sync::{broadcast, watch, Notify};
use tokio::task::JoinSet;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::budget::{Budget, Meter};
use crate::error::TeamError;
use crate::events::{AllowAll, EventBus, HookDecision, TeamEvent, TeamHooks};
use crate::mailbox::{InMemoryMailbox, Mailbox};
use crate::role::{valid_name, Role};
use crate::taskboard::{InMemoryTaskBoard, TaskBoard};
use crate::teammate;
use crate::types::{
    ApprovalInfo, ClaimOutcome, MemberId, MemberInfo, MemberState, Message, MessageKind, NewTask,
    Task, TaskId, TaskStatus, TaskUpdate,
};

/// Longest task title in bytes.
const MAX_TITLE_BYTES: usize = 200;

/// Poison-tolerant lock: every critical section here is a plain read or a single update.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn check_size(what: &'static str, len: usize, max: usize) -> Result<(), TeamError> {
    if len > max {
        Err(TeamError::TooLarge { what, len, max })
    } else {
        Ok(())
    }
}

/// Tool calls of one teammate that wait for the lead.
#[derive(Default)]
pub(crate) struct Approvals {
    pub(crate) pending: BTreeMap<String, ApprovalInfo>,
    pub(crate) decided: HashMap<String, ToolResolution>,
}

/// Runtime record of one teammate.
pub(crate) struct MemberEntry {
    pub(crate) id: MemberId,
    pub(crate) role: String,
    state: Mutex<MemberState>,
    /// Messages delivered to this member whose batch is not finished yet. Raised before a message
    /// is enqueued and lowered after the member has handled it, so "idle and zero" really means
    /// nothing is on its way.
    pub(crate) unprocessed: AtomicUsize,
    pub(crate) shutdown_requested: AtomicBool,
    pub(crate) token: CancellationToken,
    pub(crate) approvals: Mutex<Approvals>,
    /// Wakes a teammate that waits for a decision or a shutdown request.
    pub(crate) wake: Notify,
    /// Set once the lead approved this teammate's plan (roles with plan approval).
    pub(crate) plan_approved: Arc<AtomicBool>,
}

impl MemberEntry {
    fn new(id: MemberId, role: String, token: CancellationToken) -> Self {
        Self {
            id,
            role,
            // A new member starts with its brief on the way, so it is not idle.
            state: Mutex::new(MemberState::Working),
            unprocessed: AtomicUsize::new(0),
            shutdown_requested: AtomicBool::new(false),
            token,
            approvals: Mutex::new(Approvals::default()),
            wake: Notify::new(),
            plan_approved: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn state(&self) -> MemberState {
        *lock(&self.state)
    }

    fn info(&self) -> MemberInfo {
        MemberInfo {
            id: self.id.clone(),
            role: self.role.clone(),
            state: self.state(),
            unprocessed: self.unprocessed.load(Ordering::SeqCst),
        }
    }
}

/// State shared by the team, its handles and every teammate task.
pub(crate) struct TeamShared {
    pub(crate) team_id: String,
    pub(crate) name: String,
    pub(crate) mailbox: Arc<dyn Mailbox>,
    pub(crate) board: Arc<dyn TaskBoard>,
    pub(crate) meter: Meter,
    events: EventBus,
    pub(crate) hooks: Arc<dyn TeamHooks>,
    roles: BTreeMap<String, Role>,
    members: Mutex<BTreeMap<MemberId, Arc<MemberEntry>>>,
    tasks: Mutex<JoinSet<()>>,
    pub(crate) token: CancellationToken,
    /// Bumped on every change anyone may be waiting for. A counter in a `watch` channel, so a
    /// change that happens between "look" and "wait" is never missed.
    activity: watch::Sender<u64>,
    base_options: RunOptions,
    runtime: Option<Runtime>,
    next_auto_name: AtomicU64,
    /// Set when shutdown starts: no more spawns.
    closing: AtomicBool,
    /// Set when shutdown is over: nothing is delivered any more.
    closed: AtomicBool,
}

impl TeamShared {
    pub(crate) fn notify(&self) {
        self.activity.send_modify(|v| *v = v.wrapping_add(1));
    }

    pub(crate) fn emit(&self, event: TeamEvent) {
        self.events.emit(event);
    }

    pub(crate) fn member(&self, id: &MemberId) -> Option<Arc<MemberEntry>> {
        lock(&self.members).get(id).cloned()
    }

    fn live_members(&self) -> Vec<Arc<MemberEntry>> {
        lock(&self.members)
            .values()
            .filter(|m| !m.state().is_terminal())
            .cloned()
            .collect()
    }

    /// Move `entry` to `state`. A member that has ended stays ended.
    pub(crate) fn set_state(&self, entry: &MemberEntry, state: MemberState) {
        self.transition(entry, None, state);
    }

    /// Move `entry` to `to`, but only from one of the states in `from` (any when `None`).
    pub(crate) fn transition(
        &self,
        entry: &MemberEntry,
        from: Option<&[MemberState]>,
        to: MemberState,
    ) -> bool {
        {
            let mut current = lock(&entry.state);
            if current.is_terminal() || *current == to {
                return false;
            }
            if from.is_some_and(|allowed| !allowed.contains(&*current)) {
                return false;
            }
            *current = to;
        }
        tracing::debug!(member = %entry.id, state = ?to, "member state changed");
        self.emit(TeamEvent::MemberStateChanged {
            member: entry.id.clone(),
            state: to,
        });
        self.notify();
        true
    }

    /// Every member is finished, or idle with nothing on its way.
    pub(crate) fn quiescent(&self) -> bool {
        lock(&self.members).values().all(|m| {
            let state = m.state();
            state.is_terminal()
                || (state == MemberState::Idle && m.unprocessed.load(Ordering::SeqCst) == 0)
        })
    }

    /// Put `message` in its recipient's inbox, keeping the recipient's counters right.
    pub(crate) async fn deliver(&self, message: Message) -> Result<(), TeamError> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(TeamError::Closed);
        }
        check_size(
            "message",
            message.body.len(),
            self.meter.budget().max_message_bytes,
        )?;
        let recipient = if message.to.is_lead() {
            None
        } else {
            let entry = self
                .member(&message.to)
                .ok_or_else(|| TeamError::UnknownMember(message.to.clone()))?;
            if entry.state().is_terminal() {
                return Err(TeamError::UnknownMember(message.to.clone()));
            }
            Some(entry)
        };
        if let Some(entry) = &recipient {
            entry.unprocessed.fetch_add(1, Ordering::SeqCst);
        }
        let (from, to, kind) = (message.from.clone(), message.to.clone(), message.kind);
        match self.mailbox.send(message).await {
            Ok(()) => {
                tracing::debug!(%from, %to, ?kind, "message delivered");
                self.emit(TeamEvent::MessageSent { from, to, kind });
                self.notify();
                Ok(())
            }
            Err(error) => {
                if let Some(entry) = &recipient {
                    entry.unprocessed.fetch_sub(1, Ordering::SeqCst);
                }
                Err(error)
            }
        }
    }

    /// A message from the team itself to the lead. Failures are not interesting: if the lead
    /// cannot be told (inbox full, team closed), the event stream still carries the news.
    pub(crate) async fn alert_lead(&self, from: &MemberId, kind: MessageKind, body: String) {
        let message = Message::new(from.clone(), MemberId::lead(), kind, body);
        if let Err(error) = self.deliver(message).await {
            tracing::debug!(%error, "could not deliver a report to the lead");
        }
    }

    /// A member ended because of an error.
    pub(crate) async fn fail_member(&self, entry: &MemberEntry, reason: &str) {
        self.set_state(entry, MemberState::Failed);
        tracing::warn!(member = %entry.id, "member failed");
        self.emit(TeamEvent::MemberFailed {
            member: entry.id.clone(),
            reason: reason.to_string(),
        });
        self.alert_lead(
            &entry.id,
            MessageKind::Alert,
            format!("Teammate `{}` stopped: {reason}.", entry.id),
        )
        .await;
    }

    /// Options for one run of `member` (or the lead when `session` is `None`).
    pub(crate) fn run_options(
        &self,
        handle: TeamHandle,
        role: Option<&str>,
        session: Option<Arc<dyn Session>>,
        max_turns: Option<usize>,
    ) -> RunOptions {
        let mut options = self.base_options.clone();
        if let Some(turns) = max_turns {
            options.max_turns = Some(turns);
        }
        if let Some(session) = session {
            options.session = Some(session);
        }
        if let Some(runtime) = &self.runtime {
            options.runtime = Some(runtime.clone());
        }
        // `tokio::spawn` does not carry the tracing scope over, so every run is told explicitly
        // which team it belongs to.
        let tracing = &mut options.run_config.tracing;
        tracing.workflow_name = Some(self.name.clone());
        tracing.group_id = Some(self.team_id.clone());
        tracing.metadata = Some(json!({
            "team": self.name,
            "team_id": self.team_id,
            "member": handle.me.as_str(),
            "role": role,
        }));
        options.with(handle)
    }

    fn role_names(&self) -> String {
        if self.roles.is_empty() {
            return "(none registered)".to_string();
        }
        self.roles.keys().cloned().collect::<Vec<_>>().join(", ")
    }

    pub(crate) fn spawn_task(
        &self,
        future: impl std::future::Future<Output = ()> + Send + 'static,
    ) {
        lock(&self.tasks).spawn(future);
    }
}

/// What [`TeamHandle::wait`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaitOutcome {
    /// Messages that arrived for the waiting member, oldest first.
    pub messages: Vec<Message>,
    /// Every teammate is idle or finished, with nothing on its way.
    pub quiescent: bool,
    /// The wait ended because the time was up.
    pub timed_out: bool,
}

/// A member's way into the team. Cheap to clone.
#[derive(Clone)]
pub struct TeamHandle {
    pub(crate) shared: Arc<TeamShared>,
    pub(crate) me: MemberId,
}

impl std::fmt::Debug for TeamHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TeamHandle")
            .field("team", &self.shared.name)
            .field("me", &self.me)
            .finish()
    }
}

impl TeamHandle {
    pub(crate) fn new(shared: Arc<TeamShared>, me: MemberId) -> Self {
        Self { shared, me }
    }

    /// Who is calling.
    pub fn me(&self) -> &MemberId {
        &self.me
    }

    /// The team's id (also the trace group id).
    pub fn team_id(&self) -> &str {
        &self.shared.team_id
    }

    fn require_lead(&self, what: &str) -> Result<(), TeamError> {
        if self.me.is_lead() {
            Ok(())
        } else {
            Err(TeamError::Invalid(format!("only the lead can {what}")))
        }
    }

    /// The roles a lead can create teammates from, with their descriptions.
    pub fn roles(&self) -> Vec<(String, String)> {
        self.shared
            .roles
            .values()
            .map(|r| (r.name.clone(), r.description.clone()))
            .collect()
    }

    /// Every teammate (not the lead), by name.
    pub fn members(&self) -> Vec<MemberInfo> {
        lock(&self.shared.members)
            .values()
            .map(|m| m.info())
            .collect()
    }

    /// Pick the name and record the member, all under one lock so two spawns cannot take the same
    /// name or both squeeze under the size limit.
    fn register_member(
        &self,
        role: &str,
        name: Option<&str>,
    ) -> Result<Arc<MemberEntry>, TeamError> {
        let shared = &self.shared;
        let mut members = lock(&shared.members);
        let live = members
            .values()
            .filter(|m| !m.state().is_terminal())
            .count();
        let max = shared.meter.budget().max_members;
        if live >= max {
            return Err(TeamError::BudgetExceeded(format!(
                "at most {max} teammates can be active at once"
            )));
        }
        let id = match name {
            Some(name) => {
                valid_name(name).map_err(|reason| TeamError::InvalidName {
                    name: name.to_string(),
                    reason,
                })?;
                if name == MemberId::LEAD || members.contains_key(&MemberId::new(name)) {
                    return Err(TeamError::InvalidName {
                        name: name.to_string(),
                        reason: "the name is reserved or already used".to_string(),
                    });
                }
                MemberId::new(name)
            }
            None => loop {
                let n = shared.next_auto_name.fetch_add(1, Ordering::SeqCst) + 1;
                let candidate = MemberId::new(format!("{role}-{n}"));
                if !members.contains_key(&candidate) {
                    break candidate;
                }
            },
        };
        let entry = Arc::new(MemberEntry::new(
            id.clone(),
            role.to_string(),
            shared.token.child_token(),
        ));
        members.insert(id, Arc::clone(&entry));
        Ok(entry)
    }

    /// Create a teammate from `role` and hand it `brief`. Lead only.
    ///
    /// `name` must be unused and follow the naming rule of [`Role::new`]; without one the team
    /// picks `<role>-<n>`.
    pub async fn spawn_member(
        &self,
        role: &str,
        name: Option<&str>,
        brief: &str,
    ) -> Result<MemberId, TeamError> {
        self.require_lead("create teammates")?;
        let shared = &self.shared;
        if shared.closing.load(Ordering::SeqCst) {
            return Err(TeamError::Closed);
        }
        check_size(
            "brief",
            brief.len(),
            shared.meter.budget().max_message_bytes,
        )?;
        let role_def = shared
            .roles
            .get(role)
            .ok_or_else(|| TeamError::UnknownRole {
                role: role.to_string(),
                available: shared.role_names(),
            })?;

        let entry = match self.register_member(role, name) {
            Ok(entry) => entry,
            Err(error) => {
                if let TeamError::BudgetExceeded(reason) = &error {
                    self.report_budget(reason.clone()).await;
                }
                return Err(error);
            }
        };

        let id = entry.id.clone();
        shared.mailbox.register(&id).await;
        let agent = teammate::prepare_agent(role_def, &id, &shared.name, &entry.plan_approved);
        let session = role_def.make_session(&id, &shared.team_id);
        let max_turns = role_def
            .max_turns
            .unwrap_or(shared.meter.budget().max_turns_per_run)
            .min(shared.meter.budget().max_turns_per_run)
            .max(1);
        shared.spawn_task(teammate::run_member(
            Arc::clone(shared),
            Arc::clone(&entry),
            agent,
            session,
            max_turns,
        ));
        tracing::debug!(member = %id, role, "teammate spawned");
        shared.emit(TeamEvent::MemberSpawned {
            member: id.clone(),
            role: role.to_string(),
        });
        shared.notify();

        let first = Message::new(MemberId::lead(), id.clone(), MessageKind::Text, brief);
        if let Err(error) = shared.deliver(first).await {
            entry.token.cancel();
            shared.set_state(&entry, MemberState::Failed);
            return Err(error);
        }
        Ok(id)
    }

    async fn report_budget(&self, reason: String) {
        if self.shared.meter.first_report() {
            tracing::warn!("team budget exceeded");
            self.shared.emit(TeamEvent::BudgetExceeded { reason });
        }
    }

    /// Send `body` to `to`.
    pub async fn send(&self, to: &MemberId, body: impl Into<String>) -> Result<(), TeamError> {
        if to == &self.me {
            return Err(TeamError::Invalid(
                "you cannot message yourself".to_string(),
            ));
        }
        let message = Message::new(self.me.clone(), to.clone(), MessageKind::Text, body);
        self.shared.deliver(message).await
    }

    /// Send `body` to every other member and to the lead. Returns how many inboxes took it;
    /// recipients whose inbox is full are skipped.
    pub async fn broadcast(&self, body: impl Into<String>) -> Result<usize, TeamError> {
        let body = body.into();
        check_size(
            "message",
            body.len(),
            self.shared.meter.budget().max_message_bytes,
        )?;
        let mut recipients: Vec<MemberId> = self
            .shared
            .live_members()
            .iter()
            .map(|m| m.id.clone())
            .collect();
        recipients.push(MemberId::lead());
        let mut delivered = 0;
        for to in recipients.into_iter().filter(|to| to != &self.me) {
            let message = Message::new(self.me.clone(), to, MessageKind::Text, body.clone());
            match self.shared.deliver(message).await {
                Ok(()) => delivered += 1,
                Err(error) => tracing::debug!(%error, "broadcast skipped a recipient"),
            }
        }
        Ok(delivered)
    }

    /// Take the messages waiting for the caller, without waiting.
    pub async fn take_messages(&self) -> Vec<Message> {
        self.shared.mailbox.drain(&self.me).await
    }

    /// Wait until the caller has mail, the whole team is idle, or `timeout` passes.
    ///
    /// This is how a lead waits for asynchronous teammates without polling.
    pub async fn wait(&self, timeout: Duration) -> WaitOutcome {
        // Subscribe first, look second: a change after the look wakes the wait.
        let mut changes = self.shared.activity.subscribe();
        let deadline = Instant::now() + timeout;
        loop {
            let messages = self.shared.mailbox.drain(&self.me).await;
            let quiescent = self.shared.quiescent();
            if !messages.is_empty() || quiescent {
                return WaitOutcome {
                    messages,
                    quiescent,
                    timed_out: false,
                };
            }
            match tokio::time::timeout_at(deadline, changes.changed()).await {
                Ok(Ok(())) => {}
                // The sender is gone only when the team is dropped; report what is there.
                Ok(Err(_)) | Err(_) => {
                    return WaitOutcome {
                        messages: self.shared.mailbox.drain(&self.me).await,
                        quiescent: self.shared.quiescent(),
                        timed_out: true,
                    }
                }
            }
        }
    }

    /// Ask `member` to finish its current work and leave. Lead only.
    pub async fn request_shutdown(&self, member: &MemberId) -> Result<(), TeamError> {
        self.require_lead("ask a teammate to leave")?;
        let entry = self
            .shared
            .member(member)
            .ok_or_else(|| TeamError::UnknownMember(member.clone()))?;
        if entry.state().is_terminal() {
            return Ok(());
        }
        entry.shutdown_requested.store(true, Ordering::SeqCst);
        entry.wake.notify_one();
        self.shared.transition(
            &entry,
            Some(&[
                MemberState::Idle,
                MemberState::Working,
                MemberState::Blocked,
            ]),
            MemberState::ShuttingDown,
        );
        let message = Message::new(
            self.me.clone(),
            member.clone(),
            MessageKind::ShutdownRequest,
            "Please finish your current work and leave the team.",
        );
        self.shared.deliver(message).await
    }

    // ---- approvals ----

    /// Tool calls of teammates that wait for the lead, oldest member first.
    pub fn pending_approvals(&self) -> Vec<ApprovalInfo> {
        lock(&self.shared.members)
            .values()
            .flat_map(|m| {
                lock(&m.approvals)
                    .pending
                    .values()
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Decide a tool call that `member` waits on. Lead only.
    ///
    /// `call_id` comes from the approval request ([`TeamHandle::pending_approvals`]). The
    /// teammate resumes right away: it runs the tool, tells the model why it was refused, or hands
    /// the model an output, depending on `resolution`.
    pub async fn resolve(
        &self,
        member: &MemberId,
        call_id: &str,
        resolution: ToolResolution,
    ) -> Result<(), TeamError> {
        self.require_lead("decide tool calls")?;
        let entry = self
            .shared
            .member(member)
            .ok_or_else(|| TeamError::UnknownMember(member.clone()))?;
        {
            let mut approvals = lock(&entry.approvals);
            if approvals.pending.remove(call_id).is_none() {
                return Err(TeamError::UnknownApproval {
                    member: member.clone(),
                    call_id: call_id.to_string(),
                });
            }
            approvals.decided.insert(call_id.to_string(), resolution);
        }
        entry.wake.notify_one();
        self.shared.emit(TeamEvent::ApprovalResolved {
            member: member.clone(),
            call_id: call_id.to_string(),
        });
        self.shared.notify();
        Ok(())
    }

    /// Record that the lead approved the caller's plan (used by `submit_plan`).
    pub(crate) fn mark_plan_approved(&self) {
        if let Some(entry) = self.shared.member(&self.me) {
            entry.plan_approved.store(true, Ordering::SeqCst);
        }
    }

    // ---- tasks ----

    /// Add a task to the shared list.
    pub async fn create_task(
        &self,
        title: &str,
        description: &str,
        blocked_by: Vec<TaskId>,
    ) -> Result<Task, TeamError> {
        check_size("task title", title.len(), MAX_TITLE_BYTES)?;
        check_size(
            "task description",
            description.len(),
            self.shared.meter.budget().max_task_bytes,
        )?;
        if title.trim().is_empty() {
            return Err(TeamError::Invalid("a task needs a title".to_string()));
        }
        let task = self
            .shared
            .board
            .create(NewTask {
                title: title.to_string(),
                description: description.to_string(),
                blocked_by,
                created_by: self.me.clone(),
            })
            .await?;
        self.shared.emit(TeamEvent::TaskCreated {
            task: task.id.clone(),
            by: self.me.clone(),
        });
        self.shared.notify();
        Ok(task)
    }

    /// Take a pending task. Teammates only.
    pub async fn claim_task(&self, id: &TaskId) -> Result<ClaimOutcome, TeamError> {
        if self.me.is_lead() {
            return Err(TeamError::Invalid(
                "the lead assigns work; only teammates claim tasks".to_string(),
            ));
        }
        let outcome = self.shared.board.claim(id, &self.me).await?;
        if matches!(outcome, ClaimOutcome::Claimed { .. }) {
            self.shared.emit(TeamEvent::TaskClaimed {
                task: id.clone(),
                by: self.me.clone(),
            });
            self.shared.notify();
        }
        Ok(outcome)
    }

    /// Change a task the caller owns. Completing goes through
    /// [`TeamHooks::on_task_completed`].
    pub async fn update_task(&self, id: &TaskId, update: TaskUpdate) -> Result<Task, TeamError> {
        if let Some(result) = &update.result {
            check_size(
                "task result",
                result.len(),
                self.shared.meter.budget().max_task_bytes,
            )?;
        }
        if update.status == Some(TaskStatus::Completed) {
            let current = self
                .shared
                .board
                .get(id)
                .await?
                .ok_or_else(|| TeamError::UnknownTask(id.clone()))?;
            if current.owner.as_ref() == Some(&self.me) && current.status == TaskStatus::InProgress
            {
                let mut proposed = current;
                proposed.status = TaskStatus::Completed;
                proposed.result = update.result.clone();
                if let HookDecision::Reject(reason) = self
                    .shared
                    .hooks
                    .on_task_completed(&proposed, &self.me)
                    .await
                {
                    return Err(TeamError::Rejected(reason));
                }
            }
        }
        let task = self.shared.board.update(id, &self.me, update).await?;
        if task.status == TaskStatus::Completed {
            self.shared.emit(TeamEvent::TaskCompleted {
                task: task.id.clone(),
                by: self.me.clone(),
            });
        }
        self.shared.notify();
        Ok(task)
    }

    /// All tasks.
    pub async fn list_tasks(&self) -> Result<Vec<Task>, TeamError> {
        self.shared.board.list().await
    }

    /// Watch what happens in the team.
    pub fn subscribe(&self) -> broadcast::Receiver<TeamEvent> {
        self.shared.events.subscribe()
    }

    /// What the team has used so far.
    pub fn usage(&self) -> TeamUsage {
        TeamUsage {
            runs: self.shared.meter.runs(),
            tokens: self.shared.meter.tokens(),
        }
    }
}

/// What a team has used, to compare with its [`Budget`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TeamUsage {
    /// Teammate runs started.
    pub runs: u64,
    /// Tokens used by finished teammate runs.
    pub tokens: u64,
}

/// A team. Dropping it stops all teammates.
pub struct Team {
    shared: Arc<TeamShared>,
}

impl Team {
    /// Start describing a team called `name`.
    pub fn builder(name: impl Into<String>) -> TeamBuilder {
        TeamBuilder::new(name)
    }

    /// The lead's handle.
    pub fn handle(&self) -> TeamHandle {
        TeamHandle::new(Arc::clone(&self.shared), MemberId::lead())
    }

    /// The team's id.
    pub fn id(&self) -> &str {
        &self.shared.team_id
    }

    /// Run options for the lead's own run: the template given to the builder plus the lead's
    /// [`TeamHandle`], which the team tools read.
    pub fn lead_options(&self) -> RunOptions {
        self.shared.run_options(self.handle(), None, None, None)
    }

    /// Watch what happens in the team.
    pub fn subscribe(&self) -> broadcast::Receiver<TeamEvent> {
        self.shared.events.subscribe()
    }

    /// Wait until every teammate is idle or finished. `false` when `timeout` passed first.
    pub async fn wait_idle(&self, timeout: Duration) -> bool {
        let mut changes = self.shared.activity.subscribe();
        let deadline = Instant::now() + timeout;
        loop {
            if self.shared.quiescent() {
                return true;
            }
            if tokio::time::timeout_at(deadline, changes.changed())
                .await
                .is_err()
            {
                return self.shared.quiescent();
            }
        }
    }

    /// Stop the team: ask every teammate to finish and leave, wait up to `grace` for them, then
    /// cancel whoever is left. Safe to call twice.
    pub async fn shutdown(&self, grace: Duration) {
        let shared = &self.shared;
        if shared.closing.swap(true, Ordering::SeqCst) {
            return;
        }
        let lead = self.handle();
        for member in shared.live_members() {
            if let Err(error) = lead.request_shutdown(&member.id).await {
                tracing::debug!(%error, "shutdown request not delivered");
            }
        }
        let mut changes = shared.activity.subscribe();
        let deadline = Instant::now() + grace;
        while !shared.live_members().is_empty() {
            if tokio::time::timeout_at(deadline, changes.changed())
                .await
                .is_err()
            {
                break;
            }
        }

        shared.closed.store(true, Ordering::SeqCst);
        shared.token.cancel();
        let mut tasks = std::mem::take(&mut *lock(&shared.tasks));
        let drain = async { while tasks.join_next().await.is_some() {} };
        if tokio::time::timeout(Duration::from_secs(2), drain)
            .await
            .is_err()
        {
            tasks.abort_all();
        }
        shared.mailbox.close(&MemberId::lead()).await;
        shared.emit(TeamEvent::TeamShutdown);
        shared.notify();
    }
}

impl Drop for Team {
    fn drop(&mut self) {
        self.shared.token.cancel();
        lock(&self.shared.tasks).abort_all();
    }
}

impl std::fmt::Debug for Team {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Team")
            .field("name", &self.shared.name)
            .field("id", &self.shared.team_id)
            .finish()
    }
}

/// Describes a team before it exists.
pub struct TeamBuilder {
    name: String,
    roles: Vec<Role>,
    budget: Budget,
    hooks: Option<Arc<dyn TeamHooks>>,
    mailbox: Option<Arc<dyn Mailbox>>,
    board: Option<Arc<dyn TaskBoard>>,
    options: RunOptions,
    runtime: Option<Runtime>,
}

impl TeamBuilder {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            roles: Vec::new(),
            budget: Budget::default(),
            hooks: None,
            mailbox: None,
            board: None,
            options: RunOptions::default(),
            runtime: None,
        }
    }

    /// Register a role teammates can be created from.
    #[must_use]
    pub fn role(mut self, role: Role) -> Self {
        self.roles.push(role);
        self
    }

    /// Set the limits. The default is [`Budget::default`].
    #[must_use]
    pub fn budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// Install quality gates.
    #[must_use]
    pub fn hooks(mut self, hooks: impl TeamHooks + 'static) -> Self {
        self.hooks = Some(Arc::new(hooks));
        self
    }

    /// Use another mailbox than [`InMemoryMailbox`].
    #[must_use]
    pub fn mailbox(mut self, mailbox: Arc<dyn Mailbox>) -> Self {
        self.mailbox = Some(mailbox);
        self
    }

    /// Use another task board than [`InMemoryTaskBoard`].
    #[must_use]
    pub fn task_board(mut self, board: Arc<dyn TaskBoard>) -> Self {
        self.board = Some(board);
        self
    }

    /// Options every run starts from: provider, hooks, guardrail and tracing settings, extensions.
    ///
    /// The team overrides the turn limit of teammates, their session, and the trace workflow name,
    /// group id and metadata.
    #[must_use]
    pub fn run_options(mut self, options: RunOptions) -> Self {
        self.options = options;
        self
    }

    /// The runtime every run of the team executes in (provider and tracing processors).
    #[must_use]
    pub fn runtime(mut self, runtime: Runtime) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// Create the team.
    pub async fn build(self) -> Result<Team, TeamError> {
        let mut roles = BTreeMap::new();
        for role in self.roles {
            valid_name(&role.name).map_err(|reason| TeamError::InvalidName {
                name: role.name.clone(),
                reason,
            })?;
            if roles.contains_key(&role.name) {
                return Err(TeamError::InvalidName {
                    name: role.name,
                    reason: "the role is registered twice".to_string(),
                });
            }
            // A typo here would leave a tool ungated, which is the unsafe way to fail.
            for tool in role.approval_tools.iter().chain(&role.plan_tools) {
                if !role.agent.tools.iter().any(|t| &t.name == tool) {
                    return Err(TeamError::Invalid(format!(
                        "role `{}` lists the tool `{tool}` for approval, but its agent has no \
                         such tool",
                        role.name
                    )));
                }
            }
            roles.insert(role.name.clone(), role);
        }
        let mailbox: Arc<dyn Mailbox> = self
            .mailbox
            .unwrap_or_else(|| Arc::new(InMemoryMailbox::new()));
        let board: Arc<dyn TaskBoard> = self
            .board
            .unwrap_or_else(|| Arc::new(InMemoryTaskBoard::new()));
        mailbox.register(&MemberId::lead()).await;

        let shared = Arc::new(TeamShared {
            team_id: uuid::Uuid::new_v4().to_string(),
            name: self.name,
            mailbox,
            board,
            meter: Meter::new(self.budget),
            events: EventBus::new(),
            hooks: self.hooks.unwrap_or_else(|| Arc::new(AllowAll)),
            roles,
            members: Mutex::new(BTreeMap::new()),
            tasks: Mutex::new(JoinSet::new()),
            token: CancellationToken::new(),
            activity: watch::channel(0).0,
            base_options: self.options,
            runtime: self.runtime,
            next_auto_name: AtomicU64::new(0),
            closing: AtomicBool::new(false),
            closed: AtomicBool::new(false),
        });
        Ok(Team { shared })
    }
}
