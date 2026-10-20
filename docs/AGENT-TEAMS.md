# Agent Teams

Status: design and implementation notes for `opai-agents-teams` (Rust-only extension, experimental).

This document has two parts: how the industry builds asynchronous multi-agent "teams" (part 1), and
how this SDK adds them (parts 2 to 5).

## 1. How agent teams work in the industry

### 1.1 What a team is, compared with what the SDK already has

| Pattern | Control flow | Communication | Lifetime | Examples |
|---|---|---|---|---|
| Handoff | Serial; control moves to another agent | Shared conversation history | One run | OpenAI Agents SDK, Swarm |
| Agent-as-tool / subagent | Caller blocks until the callee returns | Input and return value only | One call | OpenAI Agents SDK, subagents in coding assistants |
| Orchestration graph / supervisor | A graph or manager schedules nodes | Shared state | One graph execution | LangGraph, CrewAI |
| Agent team | Several long-lived agents advance in parallel | Mailbox plus shared task list | Many turns; suspended and woken | Claude Code agent teams, AutoGen actors, MetaGPT |

A subagent is a function call in a tree. A teammate is a peer process. Teammates can message each
other without the lead relaying, and keep working while the lead waits.

### 1.2 The five components

```
        +------------- Lead (coordinator) -------------+
        | split work, assign, summarise, approve, stop |
        +----+------------------+------------------+---+
             |                  |                  |
        Teammate A         Teammate B         Teammate C     own context,
        (run loop)         (run loop)         (run loop)     tools, history
             +----------- Mailbox (direct / broadcast) ----------+
             +----------- Shared task list (deps, claim) --------+
```

1. **Registry.** Names, roles, states (idle, working, blocked, shutting down, stopped). Names are
   the addressing scheme.
2. **Independent execution loop.** Each teammate owns a context window and runs
   `wait for mail -> run agent turns -> report idle`. An idle teammate costs no model call.
3. **Communication.** Three shapes exist: a mailbox (point to point and broadcast; in-process
   channels, files or a database), publish/subscribe by topic (AutoGen runtime, MetaGPT message
   pool), and a shared blackboard (LangGraph state). Coupling and write-conflict risk grow in that
   order.
4. **Shared task list.** Tasks carry status, dependencies and an owner. Claiming must be atomic
   (file lock, compare-and-swap or a transaction). A task unblocks when its dependencies complete.
   The list is also the progress view for the lead and the recovery record after a crash.
5. **Lifecycle and governance.** Spawn by role; cooperative shutdown (request, then response, then
   exit) with forced cancel as a fallback; plan approval (read-only planning, then lead approval
   before writes); permission inheritance with dangerous actions escalated; quality gates (hooks
   such as "teammate idle" and "task completed" that can veto); whole-team cleanup.

### 1.3 Recurring engineering problems and the usual answers

| Problem | Usual answer |
|---|---|
| Two agents edit the same file | Split work by file ownership; one worktree per agent; parallel reads, serial writes |
| Duplicate claim of a task | Atomic claim |
| When to deliver a message | At a turn boundary, never inside a running tool call |
| Deadlock and idle spinning | Idle notifications with timeouts; turn and token budgets |
| Context growth | Isolated contexts; pass summaries; session compaction |
| Cost runaway | Team size cap; cheaper models for teammates; per-member budgets |
| Observability | One trace per team, one span tree per member; one merged event stream |
| Crash recovery | Persist the task list and mailboxes; serialisable member state |
| Prompt injection through messages | Treat mail as untrusted data; messages never raise privileges |
| Interop | MCP for tools, A2A for agent-to-agent across processes or vendors |

### 1.4 Framework comparison

Descriptions of products other than this SDK are summaries of public material; check each project's
own documentation for detail.

- **Claude Code agent teams**: lead plus independent sessions, on-disk mailboxes, shared task list,
  plan approval, hooks. A full team shape.
- **AutoGen**: actor model; asynchronous message delivery routed by a runtime; distributable.
- **LangGraph**: supervisor and swarm are graph nodes; fan-out with shared state and checkpoints.
  Deterministic control flow, less autonomy.
- **CrewAI**: crew plus tasks, sequential or hierarchical. Workflow-shaped, not peer messaging.
- **MetaGPT / ChatDev**: role-based software company with a shared message pool and SOPs.
- **OpenAI Agents SDK / Swarm**: handoffs and agent-as-tool only.

## 2. Options considered for this SDK

| | A: parallel agent-as-tool | B: new crate, in-process actors | C: persistent, cross-process | D: make `Runner` an actor |
|---|---|---|---|---|
| Real async team | no | yes | yes | yes |
| Intrusion into core | minimal | none | none | large |
| Crash recovery | none | weak, extensible | strong | depends |
| Risk | low | medium | high | high |

**Chosen: B.** A is not a team (no identity, no peer messages). C adds process management, IPC and
authentication and is deferred. D breaks the stability promise of core and contradicts the
direction in `docs/ANALYSIS-2026-10.md`.

## 3. Design (option B)

`opai-agents-teams` depends on `opai-agents-core` only (see `scripts/check_module_deps.py`) and
changes no core contract.

- **`Team`** owns shared state: member registry, `Mailbox`, `TaskBoard`, `Budget`, a cancellation
  token and an event channel.
- **Roles, not free-form agents.** `TeamBuilder::role(name, Agent, RoleConfig)` registers a
  template. The spawn tool can only pick a template name and add instructions. A lead that has been
  prompt-injected cannot pick a model, widen a tool set or lift an approval requirement.
- **Teammate loop.** One tokio task per member:
  `select { cancel, inbox } -> drain inbox -> Runner::run(agent, input, options.with(TeamHandle))
  -> report idle / blocked / failed`. Runs are non-streamed, because streamed runs do not support
  tool resolution.
- **Shared state reaches tools through `Extensions`.** A nested run does not inherit the outer
  extensions, so every teammate run is given the `TeamHandle` explicitly.
- **Mailbox** and **TaskBoard** are traits with in-memory implementations. Mailboxes are bounded;
  a full inbox returns an error instead of growing without limit.
- **Atomic claim.** Check status, dependencies and owner and write the owner under one lock.
  Outcomes: `Claimed`, `AlreadyTaken`, `Blocked`, `NotFound`.
- **Lead waiting.** `team_wait(timeout_ms)` blocks until new mail, all members idle, or timeout, and
  returns a delta summary. It is how a lead run waits without spinning.
- **Message injection** happens before each teammate run. Injection at turn boundaries inside a run
  is a P1 evaluation item.
- **Shutdown.** `shutdown_request`, then the member finishes its run and answers; the cancellation
  token is the fallback after a timeout. Members live in a `JoinSet`; dropping the `Team` aborts them.
- **Panic isolation.** A panicking member becomes `failed`; tasks it owned return to `pending`.
- **Approval.** A paused member run saves its `RunState`, posts an approval request to the lead and
  enters `blocked`. A response (from the `team_resolve` tool or `TeamHandle::resolve`) resumes it
  with `Runner::run_state`. Plan approval uses the same path.
- **Budget.** Max members (default 3), turns per run, total runs, total tokens.
- **Events and hooks.** A bounded `broadcast` channel; slow consumers lose events and never block
  the team. `TeamHooks` can veto "idle" and "task completed".
- **Tracing.** `tokio::spawn` does not inherit task-local scope, so each member run is given the
  team's runtime and trace settings explicitly, with the team id as group id.

## 4. Security model

- Messages and task text are untrusted data. They are wrapped in a delimited block and never grant
  tools or permissions.
- Body and description lengths are capped.
- Teammates hold only their role's tools. No sub-process or network code lives in this crate.
- Logs record ids and lengths, never message bodies or tool arguments.
- Write conflicts are handled by convention (split tasks by file ownership); exclusive path locks are
  future work. There is no sandbox, consistent with D-006.

## 5. Roadmap

- **P0**: in-memory mailbox and task board, runtime, tools, tests, example.
- **P1**: approval, plan approval, hooks, budgets, trace root, cooperative shutdown (this change
  set includes these).
- **P2**: SQLite or file-backed `Mailbox` and `TaskBoard`; restorable member state.
- **P3**: console and web team views, per-member models, A2A.
