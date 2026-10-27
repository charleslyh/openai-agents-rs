# Decisions on the later items (after 0.2.0)

Each item below is evaluated, not built. The entry says what was checked, what was not, the
decision, and what would change it. Nothing here is released behaviour.

What the repository itself shows is stated as fact. What comes from outside it is marked
**unverified** and must be checked before work starts.

## 1. Align with upstream Python 0.24

- **Fact.** The pinned reference is `vendor/openai-agents-python` at v0.23.1 (`docs/COMPAT.md`).
- **Unverified.** What 0.24 changed. No changelog for it was read in this round.
- **Decision.** Do it as its own task, started by diffing the two tags and sorting every changed
  public item into the existing `S` / `N` / `X` / `F` classes of `docs/COMPAT.md`. Do not start
  from memory of the release notes.
- **Order.** After 0.2.0 is on crates.io, so the breaking changes land once and the alignment
  does not mix with them.
- **Changes the decision.** A 0.24 change that fixes a behaviour we copied wrongly moves ahead of
  everything else here.

## 2. A2A (agent-to-agent protocol)

- **Fact.** `teams` is in-process only (`CHANGELOG.md` 0.1.0). There is no cross-process agent
  transport.
- **Unverified.** The current A2A specification, its stability and SDK landscape.
- **Decision.** Not before a use case exists. If built, it is a feature of an existing crate
  (`mcp`'s transport code is the closest relative), not a new crate: a new crate name costs a
  crates.io token (`docs/RELEASING.md`).
- **Changes the decision.** A user asking to put a Rust agent behind, or call one over, A2A.

## 3. Evaluation tooling

- **Fact.** `ScriptedModel` (`opai-agents-testing`) gives deterministic runs, and the exact event
  sequences are pinned in `crates/facade/tests/event_sequences.rs`. That covers "does the loop
  behave", not "is the answer good".
- **Decision.** Do not build an evaluator inside the SDK. Quality evaluation needs datasets and
  graders that belong to an application. The SDK's part is to make a run recordable and
  replayable: tracing already exports spans (`observability`), so the missing piece is a replay
  model that reads a recorded run. Scope that piece alone when someone needs it.

## 4. Safe loading of skills

- **Fact.** The repository has no skills loader. The only code-execution surfaces are tools the
  program registers itself and MCP stdio servers, whose child gets a restricted environment
  (`crates/mcp/src/stdio.rs`).
- **Decision.** If skills are added, load them as data (instructions and tool declarations),
  never as code that runs in the process; run anything executable only through an MCP server so
  it keeps the same restricted environment and the same approval path. A skill must not widen
  what the agent may do: its tools go through the agent's `ToolPolicy` and guardrails like any
  other tool.
- **Not decided.** The file format. Wait for the format Python settles on, to avoid a second
  incompatible one.

## 5. Breaking the 8-module dependency cycle in `core`

- **Fact.** `scripts/check_module_deps.py` accepts exactly one cycle: `agent`, `guardrail`,
  `handoffs`, `lifecycle`, `mcp`, `model`, `tool`, `tool_guardrails`. The cause is that every
  callback of these receives `&Agent`, which holds them in turn; the same shape as Python.
- **Decision.** Leave it. The only ways out change the signature of every user callback: an
  `AgentView` mirror of `Agent`, or generics over the agent type. The cost lands on every user
  and the gain is internal tidiness. The checker already stops it from growing.
- **Changes the decision.** Splitting `core` into several crates, which needs the cycle gone
  first. The crates.io limit (`docs/RELEASING.md`) argues against more crates anyway.

## 6. One interception pipeline

- **Fact.** There are separate hooks today: `ModelMiddleware`, `ToolMiddleware`, tool input and
  output guardrails, `RunHooks` and `AgentHooks`.
- **Decision.** Not unified now. They differ in what they may do: middleware wraps and can
  short-circuit, guardrails can only allow, reject or rewrite, hooks only observe. Merging them
  would make the type that allows the least also allow the most. Document the difference in one
  table instead, and revisit only if a fourth kind appears.

## 7. `rmcp` instead of the hand-written MCP client

- **Fact.** The client is about 2,200 lines and now covers `2026-07-28` and the older handshake
  (`DEVIATIONS.md` X-026).
- **Unverified.** `rmcp`'s minimum Rust version, dependency tree, and which protocol revisions it
  speaks.
- **Decision.** Keep the hand-written client until `rmcp` is checked against the three things
  this crate promises: MSRV 1.89, the restricted child environment, and the fallback to older
  servers. Replacing a tested transport for a dependency that may not meet them is a loss.

## Not done in this round, on purpose

- `MCP` items beyond X-026: `subscriptions/listen`, the tasks extension, `x-mcp-header`,
  `cacheScope`. Each needs a use case or a server to test against.
- Trusted Publishing for crates.io (open item in `docs/RELEASING.md`).
