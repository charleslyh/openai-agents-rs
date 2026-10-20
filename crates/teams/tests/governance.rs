mod support;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use opai_agents_core::{FunctionTool, Runner, ToolResolution};
use opai_agents_teams::{
    HookDecision, MemberId, MemberState, MessageKind, Role, TaskStatus, Team, TeamError, TeamEvent,
    TeamHandle, TeamHooks,
};
use serde_json::json;
use support::{agent, wait_until, Script, Step};

const LONG: Duration = Duration::from_secs(5);

/// A tool that counts how often it really ran.
fn counted(name: &str, runs: &Arc<AtomicUsize>) -> FunctionTool {
    let runs = Arc::clone(runs);
    FunctionTool::new(
        name,
        "changes something",
        json!({"type": "object", "properties": {}, "additionalProperties": false}),
        move |_ctx, _args| {
            let runs = Arc::clone(&runs);
            async move {
                runs.fetch_add(1, Ordering::SeqCst);
                Ok(json!("changed"))
            }
        },
    )
}

fn state_of(team: &Team, name: &str) -> MemberState {
    team.handle()
        .members()
        .into_iter()
        .find(|m| m.id.as_str() == name)
        .unwrap()
        .state
}

// ---- approval of tool calls ----

#[tokio::test(flavor = "multi_thread")]
async fn a_tool_that_needs_approval_waits_for_the_lead_and_then_runs() {
    let runs = Arc::new(AtomicUsize::new(0));
    let script = Script::new(vec![Step::call("deploy", json!({})), Step::say("deployed")]);
    let worker = agent("worker", &script).tools(vec![counted("deploy", &runs)]);
    let team = Team::builder("t")
        .role(Role::new("worker", worker).require_approval_for(["deploy"]))
        .build()
        .await
        .unwrap();
    let mut events = team.subscribe();
    let lead = team.handle();
    lead.spawn_member("worker", Some("w"), "ship it")
        .await
        .unwrap();

    wait_until(|| state_of(&team, "w") == MemberState::Blocked).await;
    assert_eq!(
        runs.load(Ordering::SeqCst),
        0,
        "nothing runs before the lead decides"
    );
    assert!(
        !team.wait_idle(Duration::from_millis(50)).await,
        "a blocked member is not idle"
    );

    let waiting = lead.pending_approvals();
    assert_eq!(waiting.len(), 1);
    assert_eq!(waiting[0].tool, "deploy");
    assert_eq!(waiting[0].member, MemberId::new("w"));
    let request = lead.wait(LONG).await.messages;
    assert_eq!(request[0].kind, MessageKind::ApprovalRequest);
    assert!(request[0].body.contains("deploy") && request[0].body.contains(&waiting[0].call_id));

    lead.resolve(
        &MemberId::new("w"),
        &waiting[0].call_id,
        ToolResolution::Run,
    )
    .await
    .unwrap();
    assert!(team.wait_idle(LONG).await);

    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert!(lead.pending_approvals().is_empty());
    let report = lead.take_messages().await;
    assert_eq!(report.last().unwrap().body, "deployed");
    assert_eq!(lead.usage().runs, 1, "resuming is not a new run");
    assert_eq!(
        lead.usage().tokens,
        30,
        "two model calls of 15 tokens each, counted once"
    );

    let mut seen = Vec::new();
    while let Ok(event) = events.try_recv() {
        seen.push(event);
    }
    assert!(seen
        .iter()
        .any(|e| matches!(e, TeamEvent::ApprovalRequested { tool, .. } if tool == "deploy")));
    assert!(seen
        .iter()
        .any(|e| matches!(e, TeamEvent::ApprovalResolved { .. })));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rejected_call_does_not_run_and_the_model_hears_why() {
    let runs = Arc::new(AtomicUsize::new(0));
    let script = Script::new(vec![
        Step::call("deploy", json!({})),
        Step::say("understood"),
    ]);
    let worker = agent("worker", &script).tools(vec![counted("deploy", &runs)]);
    let team = Team::builder("t")
        .role(Role::new("worker", worker).require_approval_for(["deploy"]))
        .build()
        .await
        .unwrap();
    let lead = team.handle();
    lead.spawn_member("worker", Some("w"), "ship it")
        .await
        .unwrap();
    wait_until(|| !lead.pending_approvals().is_empty()).await;

    let call_id = lead.pending_approvals()[0].call_id.clone();
    lead.resolve(
        &MemberId::new("w"),
        &call_id,
        ToolResolution::reject("not on a Friday"),
    )
    .await
    .unwrap();
    assert!(team.wait_idle(LONG).await);

    assert_eq!(runs.load(Ordering::SeqCst), 0);
    assert!(script.last_input().contains("not on a Friday"));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_lead_can_decide_through_the_tool() {
    let runs = Arc::new(AtomicUsize::new(0));
    let worker_script = Script::new(vec![Step::call("deploy", json!({})), Step::say("done")]);
    let worker = agent("worker", &worker_script).tools(vec![counted("deploy", &runs)]);
    let team = Team::builder("t")
        .role(Role::new("worker", worker).require_approval_for(["deploy"]))
        .build()
        .await
        .unwrap();
    let lead = team.handle();
    lead.spawn_member("worker", Some("w"), "ship it")
        .await
        .unwrap();
    wait_until(|| !lead.pending_approvals().is_empty()).await;
    let call_id = lead.pending_approvals()[0].call_id.clone();

    let lead_script = Script::new(vec![
        Step::call(
            "team_resolve",
            json!({"member": "w", "call_id": call_id, "decision": "approve", "message": null}),
        ),
        Step::say("approved it"),
    ]);
    let lead_agent = agent("lead", &lead_script).tools(team.tools());
    Runner::run(&lead_agent, "decide", team.lead_options())
        .await
        .unwrap();
    assert!(team.wait_idle(LONG).await);

    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn resolving_something_that_is_not_waiting_fails() {
    let team = Team::builder("t")
        .role(Role::new(
            "worker",
            agent("worker", &Script::new(vec![Step::say("x")])),
        ))
        .build()
        .await
        .unwrap();
    let lead = team.handle();
    lead.spawn_member("worker", Some("w"), "go").await.unwrap();
    assert!(team.wait_idle(LONG).await);

    assert!(matches!(
        lead.resolve(&MemberId::new("w"), "nope", ToolResolution::Run)
            .await,
        Err(TeamError::UnknownApproval { .. })
    ));
    assert!(matches!(
        lead.resolve(&MemberId::new("ghost"), "nope", ToolResolution::Run)
            .await,
        Err(TeamError::UnknownMember(_))
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_teammate_cannot_decide_tool_calls() {
    // A role tool that tries to use the handle the team gave the teammate.
    let sneaky = FunctionTool::new(
        "sneaky",
        "tries to approve a call",
        json!({"type": "object", "properties": {}, "additionalProperties": false}),
        |ctx, _args| async move {
            let team = ctx.require::<TeamHandle>()?.clone();
            let outcome = team
                .resolve(&MemberId::new("other"), "any", ToolResolution::Run)
                .await;
            Ok(json!(match outcome {
                Ok(()) => "it worked".to_string(),
                Err(error) => error.to_string(),
            }))
        },
    )
    .require::<TeamHandle>();
    let script = Script::new(vec![Step::call("sneaky", json!({})), Step::say("done")]);
    let worker = agent("worker", &script).tools(vec![sneaky]);
    let team = Team::builder("t")
        .role(Role::new("worker", worker))
        .build()
        .await
        .unwrap();
    team.handle()
        .spawn_member("worker", Some("w"), "go")
        .await
        .unwrap();
    assert!(team.wait_idle(LONG).await);

    let seen = script.last_input();
    assert!(
        seen.contains("only the lead can decide tool calls"),
        "{seen}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn shutting_down_a_blocked_teammate_does_not_wait_for_the_lead() {
    let runs = Arc::new(AtomicUsize::new(0));
    let script = Script::new(vec![Step::call("deploy", json!({}))]);
    let worker = agent("worker", &script).tools(vec![counted("deploy", &runs)]);
    let team = Team::builder("t")
        .role(Role::new("worker", worker).require_approval_for(["deploy"]))
        .build()
        .await
        .unwrap();
    let lead = team.handle();
    lead.spawn_member("worker", Some("w"), "go").await.unwrap();
    wait_until(|| state_of(&team, "w") == MemberState::Blocked).await;

    let started = std::time::Instant::now();
    team.shutdown(LONG).await;

    assert!(started.elapsed() < Duration::from_secs(2));
    assert_eq!(state_of(&team, "w"), MemberState::Stopped);
    assert_eq!(runs.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_role_cannot_gate_a_tool_it_does_not_have() {
    let script = Script::new(vec![]);
    let result = Team::builder("t")
        .role(Role::new("worker", agent("worker", &script)).require_approval_for(["deploy"]))
        .build()
        .await;
    assert!(matches!(result, Err(TeamError::Invalid(text)) if text.contains("deploy")));
}

// ---- plan approval ----

#[tokio::test(flavor = "multi_thread")]
async fn write_tools_stay_gated_until_the_plan_is_approved() {
    let writes = Arc::new(AtomicUsize::new(0));
    let reads = Arc::new(AtomicUsize::new(0));
    let script = Script::new(vec![
        Step::call("read_repo", json!({})),
        Step::call("submit_plan", json!({"plan": "rewrite foo"})),
        Step::call("write_file", json!({})),
        Step::say("rewritten"),
    ]);
    let worker = agent("worker", &script).tools(vec![
        counted("read_repo", &reads),
        counted("write_file", &writes),
    ]);
    let team = Team::builder("t")
        .role(Role::new("worker", worker).require_plan_approval(["write_file"]))
        .build()
        .await
        .unwrap();
    let lead = team.handle();
    lead.spawn_member("worker", Some("w"), "improve foo")
        .await
        .unwrap();

    // The plan itself is what waits first.
    wait_until(|| !lead.pending_approvals().is_empty()).await;
    let plan = lead.pending_approvals().remove(0);
    assert_eq!(plan.tool, "submit_plan");
    assert!(plan.arguments.contains("rewrite foo"));
    assert_eq!(writes.load(Ordering::SeqCst), 0);
    assert_eq!(
        reads.load(Ordering::SeqCst),
        1,
        "reading needs no plan when it is not gated"
    );

    lead.resolve(&MemberId::new("w"), &plan.call_id, ToolResolution::Run)
        .await
        .unwrap();
    assert!(team.wait_idle(LONG).await);

    // After approval the write goes through without a second request.
    assert_eq!(writes.load(Ordering::SeqCst), 1);
    assert!(lead.pending_approvals().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rejected_plan_keeps_the_write_tool_gated() {
    let writes = Arc::new(AtomicUsize::new(0));
    let script = Script::new(vec![
        Step::call("submit_plan", json!({"plan": "delete everything"})),
        Step::call("write_file", json!({})),
        Step::say("gave up"),
    ]);
    let worker = agent("worker", &script).tools(vec![counted("write_file", &writes)]);
    let team = Team::builder("t")
        .role(Role::new("worker", worker).require_plan_approval(["write_file"]))
        .build()
        .await
        .unwrap();
    let lead = team.handle();
    lead.spawn_member("worker", Some("w"), "go").await.unwrap();

    wait_until(|| !lead.pending_approvals().is_empty()).await;
    let plan = lead.pending_approvals().remove(0);
    lead.resolve(
        &MemberId::new("w"),
        &plan.call_id,
        ToolResolution::reject("too destructive"),
    )
    .await
    .unwrap();

    // The model tried the write anyway: it has to ask again.
    wait_until(|| {
        lead.pending_approvals()
            .iter()
            .any(|a| a.tool == "write_file")
    })
    .await;
    assert_eq!(writes.load(Ordering::SeqCst), 0);
    let ask = lead.pending_approvals().remove(0);
    lead.resolve(
        &MemberId::new("w"),
        &ask.call_id,
        ToolResolution::reject("no"),
    )
    .await
    .unwrap();
    assert!(team.wait_idle(LONG).await);
    assert_eq!(writes.load(Ordering::SeqCst), 0);
}

// ---- task hooks ----

struct NeedsResult;

#[async_trait]
impl TeamHooks for NeedsResult {
    async fn on_task_completed(
        &self,
        task: &opai_agents_teams::Task,
        _by: &MemberId,
    ) -> HookDecision {
        if task.result.as_deref().unwrap_or_default().is_empty() {
            HookDecision::Reject("record what you did in `result`".into())
        } else {
            HookDecision::Allow
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_hook_can_refuse_to_complete_a_task() {
    let script = Script::new(vec![
        Step::call("task_claim", json!({"task_id": "t1"})),
        Step::call(
            "task_update",
            json!({"task_id": "t1", "status": "completed", "result": null}),
        ),
        Step::call(
            "task_update",
            json!({"task_id": "t1", "status": "completed", "result": "fixed the bug"}),
        ),
        Step::say("done"),
    ]);
    let team = Team::builder("t")
        .role(Role::new("worker", agent("worker", &script)))
        .hooks(NeedsResult)
        .build()
        .await
        .unwrap();
    let mut events = team.subscribe();
    let lead = team.handle();
    lead.create_task("bug", "fix it", vec![]).await.unwrap();
    lead.spawn_member("worker", Some("w"), "take t1")
        .await
        .unwrap();
    assert!(team.wait_idle(LONG).await);

    // The first attempt was refused with the reason, the second one went through.
    assert!(script.input(2).contains("record what you did in `result`"));
    let tasks = lead.list_tasks().await.unwrap();
    assert_eq!(tasks[0].status, TaskStatus::Completed);
    assert_eq!(tasks[0].result.as_deref(), Some("fixed the bug"));
    let mut completed = 0;
    while let Ok(event) = events.try_recv() {
        completed += usize::from(matches!(event, TeamEvent::TaskCompleted { .. }));
    }
    assert_eq!(
        completed, 1,
        "the refused attempt is not reported as a completion"
    );
}
