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
cargo test --no-default-features
cargo test
```

Python oracle (parity scenarios; needs Python >= 3.10 and [uv](https://docs.astral.sh/uv/)):

```bash
bash scripts/setup_venv.sh                       # .venv with the vendored SDK, Python 3.12
.venv/bin/python scripts/run_parity.py --write-golden
cargo test --test parity_scenarios
```

A scenario is a JSON file in `tests/parity/scenarios/`. It may script handoff targets
(`agent.handoffs`), failing tools (`tools[].error`) and expect a Python exception class
(`expect.error`). Every behavior fix should add a scenario that both sides run.

See [COMPAT.md](./COMPAT.md) for the support matrix and [DEVIATIONS.md](./DEVIATIONS.md) for the
recorded differences from Python.
