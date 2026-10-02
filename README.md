# OpenAI Agents SDK (Rust)

Rust port of the [OpenAI Agents Python SDK](https://github.com/openai/openai-agents-python) **v0.23.1**.

Supported now: **Agent**, **Runner** (`run` / `run_blocking` / `run_streamed`), **FunctionTool** (+ `#[function_tool]`), **handoffs**, **local Tracing**, **Responses + Chat Completions**.

## Quick start

```rust
use std::sync::Arc;
use openai_agents::{Agent, OpenAIResponsesModel, RunOptions, Runner};

#[tokio::main]
async fn main() {
    let model = Arc::new(OpenAIResponsesModel::new(
        std::env::var("OPENAI_MODEL").unwrap(),
        std::env::var("OPENAI_API_KEY").unwrap(),
        std::env::var("OPENAI_BASE_URL").ok().as_deref(),
    ));
    let agent = Agent::new("Assistant")
        .instructions("You only respond in haikus.")
        .model(model);
    let result = Runner::run(&agent, "Tell me about recursion.", RunOptions::default())
        .await
        .unwrap();
    println!("{}", result.final_output);
}
```

## Examples

Numbered from core path → extended usage. Live OpenAI-compatible API (not `ScriptedModel`).

Required env: `OPENAI_API_KEY`, `OPENAI_MODEL`.  
Optional: `OPENAI_BASE_URL`, `OPENAI_API` (`chat_completions` **default** | `responses`).

| # | Example | What it shows |
|---|---------|----------------|
| 01 | `01_hello_agent` | Minimal `Agent` + `Runner::run` |
| 02 | `02_tools_agent` | `#[function_tool]` tool loop |
| 03 | `03_streamed_agent` | `Runner::run_streamed` events |
| 04 | `04_handoff_agent` | Multi-agent handoff |
| 05 | `05_agents_as_tools` | `Agent.as_tool` orchestration |
| 06 | `06_human_in_the_loop` | Approvals + `RunState` JSON resume |
| 07 | `07_always_approve` | Sticky `always_approve` |
| 08 | `08_mock_framework` | HTTP mocks (`--features testing`) |

```bash
cargo run --example 01_hello_agent
cargo run --example 02_tools_agent
cargo run --example 03_streamed_agent
cargo run --example 04_handoff_agent

OPENAI_API=responses cargo run --example 02_tools_agent

EXAMPLE_INPUT="Translate 'Hello' to French and Spanish." \
  cargo run --example 05_agents_as_tools
HITL_AUTO=approve cargo run --example 06_human_in_the_loop
HITL_AUTO=approve cargo run --example 07_always_approve

cargo run --example 08_mock_framework --features testing
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
