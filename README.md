# OpenAI Agents SDK (Rust)

Rust port of the [OpenAI Agents Python SDK](https://github.com/openai/openai-agents-python) **v0.23.1**.

Supported now: **Agent**, **Runner** (`run` / `run_blocking` / `run_streamed`), **FunctionTool** (+ `#[function_tool]` with full `schemars` schemas), **handoffs** (`HandoffCallItem` / `HandoffOutputItem` + `handoff_span`), **structured output** (`output_type`), **run context**, **input/output guardrails**, **lifecycle hooks**, **ModelProvider** name resolution, **extended `ModelSettings`**, **local Tracing**, **Responses + Chat Completions**.

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
Optional: `OPENAI_BASE_URL`, `OPENAI_API` (`chat_completions` | `responses`).

The crate default is the **Responses** API, matching `set_default_openai_api()` in Python. The
example harness below defaults `OPENAI_API` to `chat_completions` because most OpenAI-compatible
gateways implement Chat Completions SSE first (see D-011); set `OPENAI_API=responses` to use the
Responses API.

| # | Example | What it shows |
|---|---------|----------------|
| 01 | `01_hello` | Minimal `Agent` + `Runner::run` |
| 02 | `02_tools` | `#[function_tool]` tool loop |
| 03 | `03_streamed` | `Runner::run_streamed` events |
| 04 | `04_handoff` | Multi-agent handoff |
| 05 | `05_agents_as_tools` | `Agent.as_tool` orchestration |
| 06 | `06_human_in_the_loop` | Approvals + `RunState` JSON resume |
| 07 | `07_always_approve` | Sticky `always_approve` |
| 08 | `08_structured_output` | `output_type` + typed `final_output_as::<T>()` |
| 09 | `09_guardrails_hooks` | Context, guardrails, `RunHooks` / `AgentHooks` |

```bash
cargo run --example 01_hello
cargo run --example 02_tools
cargo run --example 03_streamed
cargo run --example 04_handoff

OPENAI_API=responses cargo run --example 02_tools

EXAMPLE_INPUT="Translate 'Hello' to French and Spanish." \
  cargo run --example 05_agents_as_tools
HITL_AUTO=approve cargo run --example 06_human_in_the_loop
HITL_AUTO=approve cargo run --example 07_always_approve

cargo run --example 08_structured_output
cargo run --example 09_guardrails_hooks
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

Shared scenarios live in [`tests/parity/scenarios/`](./tests/parity/scenarios/). A scenario only
needs `expect` to run; the `.golden.json` produced by the Python oracle is optional and compared
when present. Regenerate it with `.venv/bin/python scripts/run_parity.py --write-golden` on a
machine that has `openai-agents==0.23.1` installed.

## Compatibility & deviations

- [`docs/COMPAT.md`](./docs/COMPAT.md) — support matrix
- [`docs/DEVIATIONS.md`](./docs/DEVIATIONS.md) — intentional differences from Python (must stay current)
- [`docs/CONTRIBUTING.md`](./docs/CONTRIBUTING.md) — commit message format and verification

## Still out of scope

MCP, sessions, hosted tools (web search / file search / computer / shell / apply_patch), sandbox,
OpenAI trace cloud export, and token-level Responses streaming (D-011).

HITL (`needs_approval` / `RunState` / `Agent.as_tool` nested approvals) is supported — the Rust
`RunState` JSON schema `openai-agents-rust/2` is not Python 1.18 wire-compatible (D-012).

## License

MIT
