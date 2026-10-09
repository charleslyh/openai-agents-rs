# Contributing

## Commit messages (required)

Use compact single-line subjects:

```text
type(module): description
```

| Part | Meaning |
|------|---------|
| `type` | `feat` / `fix` / `docs` / `test` / `refactor` / `chore` / `ci` / `perf` |
| `module` | Area touched (`agents`, `runner`, `tool`, `tracing`, `openai`, `parity`, `vendor`, `ci`, …) |
| `description` | **What** you did and **why** (motivation), not a bare file list |

Examples:

```text
feat(runner): add max_turns enforcement to match Python DEFAULT_MAX_TURNS and fail closed on loops
fix(openai): send Responses tools in request body so wiremock parity catches schema drift
docs(deviations): record D-010 vendor tarball pin because GitHub submodule clone is unreliable here
```

This convention is also enforced for agents via [`.cursor/rules/commit-messages.mdc`](../.cursor/rules/commit-messages.mdc).

## Verification before PR

```bash
bash scripts/sync_vendor.sh --check
cargo fmt --all
cargo clippy --all-features --all-targets -- -D warnings
cargo test --no-default-features
cargo test --all-features
```

Python oracle (parity scenarios; needs Python >= 3.10 and [uv](https://docs.astral.sh/uv/)):

```bash
bash scripts/setup_venv.sh                       # .venv with the vendored SDK, Python 3.12
.venv/bin/python scripts/run_parity.py --check   # fail on drift; never overwrites
cargo test --test parity_scenarios
```

Regenerating a golden is a deliberate act, never part of `check`:

```bash
.venv/bin/python scripts/run_parity.py --write-golden
```

### Layers

| Layer | What it guards | Command |
|-------|----------------|---------|
| 0 | Format, lints, intra-doc links, vendor pin | `cargo fmt --all -- --check`, `cargo clippy --all-features --all-targets -- -D warnings`, `RUSTDOCFLAGS=-D warnings cargo doc --no-deps --all-features`, `bash scripts/sync_vendor.sh --check` |
| 1 | `ScriptedModel` behavior, no HTTP client in the build | `cargo test --no-default-features` |
| 1.5 | Invariants over generated inputs | `cargo test --test property_core` |
| 2 | OpenAI HTTP contracts via wiremock | `cargo test` / `cargo test --all-features` |
| 3 | Python oracle parity + MCP interop (two implementations over the wire) | `run_parity.py --check`, `cargo test --test parity_scenarios --test mcp_interop --test chat_convert_parity` |

`.github/workflows/ci.yml` runs all of them on Linux and macOS.

### The clippy backlog

`cargo clippy` runs with `-D warnings`, so **any new lint fails CI**. The lints already present
when the gate was added live in `[workspace.lints]` in `Cargo.toml`, each with a count and a
reason. That list must only ever shrink; when you clear one, delete its line and re-run to make
sure nothing else was hiding behind it.

A scenario is a JSON file in `tests/parity/scenarios/`. It may script handoff targets
(`agent.handoffs`), failing tools (`tools[].error`) and expect a Python exception class
(`expect.error`). Every behavior fix should add a scenario that both sides run.

See [COMPAT.md](./COMPAT.md) for the support matrix and [DEVIATIONS.md](./DEVIATIONS.md) for the
recorded differences from Python.

## Keeping the compatibility records honest

The matrix and the deviation log are what make this port trustworthy, and they go stale silently:
an audit found `COMPAT.md` claiming 19 `ModelSettings` fields when the struct had 22, and listing
`extra_query` / `preserve_raw_usage` / `ModelStep.retry_advice` as "not ported" after they had been.
So a change to behaviour or to the public API is only finished when the record moves with it:

- Changing observable behaviour of the run loop, tools, handoffs, sessions or retries → update
  `DEVIATIONS.md` (new `D-NNN`, or edit the existing row) in the same commit.
- Adding or removing a public API, or changing what a capability supports → update `COMPAT.md` in
  the same commit.
- Adding a Python-side check (an oracle script, a parity scenario) → say so in the row it pins.

A useful check before opening a PR: does anything `COMPAT.md` calls "not ported" actually exist in
`src/` now, and does every `D-NNN` referenced anywhere still have a row?
