//! The team behind the Lead agent: two long-lived teammates that work in parallel.
//!
//! A team is built per run, like the agent network: the roles capture this run's model (the fault
//! wrapper included), so nothing survives into the next request. The team is named after the run
//! id, which also makes the span of each teammate run land in the run's bucket of the waterfall
//! (`trace.rs` attributes a root `Task` span by `workflow_name`).

use std::sync::Arc;

use opai_agents::teams::{Budget, Role, Team, TeamError, TeamEvent};
use opai_agents::{Agent, Model};

use crate::faults::ToolFaults;
use crate::tools;

/// Teammates a run may create. The demo needs two.
const MAX_MEMBERS: usize = 2;
/// Model turns of one teammate run: the brief in, one tool turn, the report out.
const TURNS_PER_RUN: usize = 4;
/// Teammate runs in total: one assignment each, plus room for a retry.
const TOTAL_RUNS: u64 = 6;
/// Tokens across the whole team, so a confused model cannot spend the request's budget.
const TOTAL_TOKENS: u64 = 40_000;

/// Build the team of one run. `run_id` names it, so its spans join the run's trace.
pub async fn build(
    model: Arc<dyn Model>,
    run_id: &str,
    faults: &ToolFaults,
) -> Result<Team, TeamError> {
    let analyst = Agent::new("TeamAnalyst")
        .instructions(
            "You answer weather questions for a team. Every figure must come from a tool: use \
             get_weather and get_temperature, one call per city, all in the same turn. Reply \
             with at most three sentences, naming the numbers you used.",
        )
        .model(Arc::clone(&model))
        .tools(vec![
            tools::get_weather(faults),
            tools::get_temperature(faults),
        ]);

    let ops = Agent::new("TeamOps")
        .instructions(
            "You answer questions about city metrics for a team. Every figure must come from a \
             tool: use get_city_metrics. Reply with at most three sentences, naming the numbers \
             you used.",
        )
        .model(model)
        .tools(vec![tools::get_city_metrics(faults)]);

    Team::builder(run_id.to_string())
        .role(Role::new("analyst", analyst).description("weather and temperature, from tools"))
        .role(Role::new("ops", ops).description("city metrics, from tools"))
        .budget(
            Budget::new()
                .max_members(MAX_MEMBERS)
                .max_turns_per_run(TURNS_PER_RUN)
                .max_total_runs(TOTAL_RUNS)
                .max_total_tokens(TOTAL_TOKENS),
        )
        .build()
        .await
}

/// A short `(kind, detail)` line for the timeline. Message bodies are never included: the event
/// carries who, to whom and which kind, and nothing the teammate wrote.
pub fn describe(event: &TeamEvent) -> (&'static str, String) {
    match event {
        TeamEvent::MemberSpawned { member, role } => {
            ("team member", format!("{member} created from `{role}`"))
        }
        TeamEvent::MemberStateChanged { member, state } => {
            ("team state", format!("{member} → {state:?}").to_lowercase())
        }
        TeamEvent::MessageSent { from, to, kind } => (
            "team message",
            format!("{from} → {to} ({kind:?})").to_lowercase(),
        ),
        TeamEvent::TaskCreated { task, by } => ("team task", format!("{task} created by {by}")),
        TeamEvent::TaskClaimed { task, by } => ("team task", format!("{task} claimed by {by}")),
        TeamEvent::TaskCompleted { task, by } => ("team task", format!("{task} completed by {by}")),
        TeamEvent::ApprovalRequested { member, tool, .. } => (
            "team approval",
            format!("{member} waits for a decision on `{tool}`"),
        ),
        TeamEvent::ApprovalResolved { member, .. } => {
            ("team approval", format!("{member} may continue"))
        }
        TeamEvent::BudgetExceeded { reason } => ("team budget", reason.clone()),
        TeamEvent::MemberFailed { member, reason } => {
            ("team failure", format!("{member}: {reason}"))
        }
        TeamEvent::TeamShutdown => ("team", "every teammate left".to_string()),
        // `TeamEvent` is non-exhaustive: a new variant shows up as a plain row, not a build error.
        _ => ("team", String::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faults::ToolFaults;
    use opai_agents::teams::{MemberId, MessageKind};
    use opai_agents::testing::{assistant_message, ModelStep, ScriptedModel};

    fn model() -> Arc<dyn Model> {
        Arc::new(ScriptedModel::new([ModelStep::output([
            assistant_message("ok"),
        ])]))
    }

    #[tokio::test]
    async fn a_team_registers_the_two_roles_the_lead_may_pick() {
        let team = build(model(), "run-1", &ToolFaults::default())
            .await
            .expect("the team builds");
        let roles: Vec<String> = team
            .handle()
            .roles()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(roles, ["analyst", "ops"]);
    }

    #[test]
    fn events_describe_who_and_what_without_message_bodies() {
        let event = TeamEvent::MessageSent {
            from: MemberId::new("analyst"),
            to: MemberId::lead(),
            kind: MessageKind::IdleNotice,
        };
        let (kind, detail) = describe(&event);
        assert_eq!(kind, "team message");
        assert!(detail.contains("analyst"));
        assert!(detail.contains("lead"));
    }
}
