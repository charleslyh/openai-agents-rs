# openai-agents-core

The agent, tool, handoff, guardrail, run loop and session contracts. This is the crate every
other crate depends on, and it depends on none of them.

## Hard rules

- **No HTTP, no database, no child processes.** Dependencies are only `async-trait`, `futures`,
  `schemars`, `serde`, `serde_json`, `thiserror`, `tokio`, `tracing`, `uuid`. Notably absent:
  `reqwest`, `async-openai`, `rusqlite`, and `tokio`'s `process` / `io-util` features.
  `scripts/check_module_deps.py` fails the build on any of these.
- **No features.** `core/Cargo.toml` defines none. Anything that needs an optional dependency
  belongs in an extension crate; only the *contract* stays here (see `src/mcp/` for the pattern:
  trait + data types + the conversion helper, no transports).
- **Module direction is enforced.** `scripts/check_module_deps.py` checks `use crate::…` against a
  tier table. `tracing` may only depend on `usage` (D3). Cycles are gated, so a new upward import
  needs a decision, not just a code change.

## Layout notes

- `src/run/agent_loop.rs` holds `run_loop_inner` (642 lines: setup, then one iteration per model
  call). `src/run/turn/` holds the phases it runs between two model calls: `mod.rs` for a turn's
  bookkeeping, `tools.rs` for everything about tools, `handoff.rs` for switching agents. A new phase belongs in
  `turn.rs` and answers one question; the loop stays the only place that decides what happens next.
  The still-large part of the loop is the model call itself (retry plus streaming), whose body
  closes over the request and the response.
- `src/model/` is the `Model` / `ModelProvider` contract. The OpenAI implementation lives in
  `crates/openai`.
- `src/context.rs` (trimmers) and `src/memory/` (`CompactingSession`) are here on purpose — see D13
  in `docs/refactor/05-open-questions.md`.

## Verification

From the workspace root:

```
cargo test -p openai-agents-core
python3 scripts/check_module_deps.py
```

Baseline for the whole workspace: 262 tests with `--all-features`, 197 (pre-P3) with
`--no-default-features`. Do not let either number drop.
