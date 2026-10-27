# opai-agents-teams

Agent teams for the Rust OpenAI Agents SDK (Rust-only, experimental). A lead creates long-lived
teammates that work in parallel, message each other and share a task list.

Handoffs swap the active agent and agents-as-tools nest a synchronous run. A team adds the third
shape: several agents running at the same time, each with its own context and tools.

```toml
opai-agents = { version = "0.2", features = ["teams"] }
```

```rust,ignore
use opai_agents::teams::{Role, Team};

let team = Team::builder("research")
    .role(Role::new("researcher", researcher_agent).description("looks things up"))
    .build()
    .await?;

let lead = Agent::new("lead").model(model).tools(team.tools());
let result = Runner::run(&lead, "Compare A and B", team.lead_options()).await?;
team.shutdown(std::time::Duration::from_secs(5)).await;
```

The lead gets the tools `team_spawn`, `team_send`, `team_broadcast`, `team_wait`, `team_shutdown`,
`team_resolve`, `team_status` and `task_create`. Teammates get messaging, status and `task_create`,
`task_claim` and `task_update`. Run the lead with `Team::lead_options()`: the tools find the team
through a `TeamHandle` in the run options.

## Security model

- Teammates are created from `Role` templates the application registered. A lead picks a role and
  writes a brief; it cannot choose a model, add a tool or lift a restriction.
- The sender of a message comes from the caller's `TeamHandle`, never from tool arguments.
- Messages are injected between runs, quoted as data, and cannot close their own quote block.
- `Role::require_approval_for` and `Role::require_plan_approval` make tool calls wait for the lead
  (`team_resolve` or `TeamHandle::resolve`).
- `Budget` bounds the number of teammates, turns per run, runs, tokens and message sizes.

## Limits

In-process only; nothing is persisted. `Mailbox` and `TaskBoard` are traits so a durable store can be
added later. Messages reach a teammate between runs, not in the middle of one. The design and the
reasoning behind it are in `docs/AGENT-TEAMS.md`.
