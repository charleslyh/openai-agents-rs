# OpenAI Agents SDK (Rust)

Rust port of the [OpenAI Agents Python SDK](https://github.com/openai/openai-agents-python) **v0.23.1**.

Supported now: **Agent**, **Runner** (`run` / `run_blocking` / `run_streamed`), **FunctionTool** (+ `#[function_tool]`), **handoffs**, **local Tracing**, **Responses + Chat Completions**.

## Quick start

```rust
use std::sync::Arc;
use openai_agents::{Agent, RunOptions, Runner, testing::{ScriptedModel, ModelStep, ItemHelpers}};

#[tokio::main]
async fn main() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::text_message("hello")),
    ]));
    let agent = Agent::new("assistant").model(model);
    let result = Runner::run(&agent, "hi", RunOptions::default()).await.unwrap();
    println!("{}", result.final_output);
}
```

OpenAI (requires feature `openai`, default on):

```bash
export OPENAI_API_KEY=sk-...
cargo run --example hello_agent --features openai
cargo run --example mock_framework --features testing
```

## Standard reference

Vendored tree: [`vendor/openai-agents-python`](./vendor/openai-agents-python) pinned by [`vendor/PINNED_VERSION`](./vendor/PINNED_VERSION) (`v0.23.1`).

```bash
bash scripts/sync_vendor.sh --check
```

(Network note: preferred `git submodule` may fail behind restricted GitHub access; the sync script can download the release tarball.)

## Verification (three layers)

| Layer | Command |
|-------|---------|
| 1. ScriptedModel behavior | `cargo test --no-default-features` |
| 2. async-openai + MockResponses/MockCompletions | `cargo test --features testing` |
| 3. Python oracle parity | `.venv/bin/python scripts/run_parity.py --write-golden` then `cargo test --test parity_scenarios` |

Shared scenarios live in [`tests/parity/scenarios/`](./tests/parity/scenarios/).

## Compatibility & deviations

- [`docs/COMPAT.md`](./docs/COMPAT.md) — Phase-1 support matrix
- [`docs/DEVIATIONS.md`](./docs/DEVIATIONS.md) — intentional differences from Python (must stay current)
- [`docs/CONTRIBUTING.md`](./docs/CONTRIBUTING.md) — commit message format and verification

## Phase-1 out of scope (still deferred)

MCP, sessions, guardrails, hosted tools, sandbox, OpenAI trace cloud export, token-level model streaming (D-011).

HITL (`needs_approval` / `RunState` / `Agent.as_tool` nested approvals) is supported — Rust JSON schema is D-012 (not Python 1.18 wire-compatible).

## License

MIT
