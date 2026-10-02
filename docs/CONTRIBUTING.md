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
cargo test --features openai
```

Optional Python oracle:

```bash
python3 -m venv .venv && .venv/bin/pip install 'openai-agents==0.23.1'
.venv/bin/python scripts/run_parity.py --write-golden
```

See [COMPAT.md](./COMPAT.md) and [DEVIATIONS.md](./DEVIATIONS.md) for Phase-1 boundaries.
