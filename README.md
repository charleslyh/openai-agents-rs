# OpenAI Agents SDK (Rust)

Rust port of the [OpenAI Agents Python SDK](https://github.com/openai/openai-agents-python) **v0.23.1**.

Supported now: **Agent**, **Runner** (`run` / `run_blocking` / `run_streamed`), **FunctionTool** (+ `#[function_tool]` with full `schemars` schemas), **handoffs** (`HandoffCallItem` / `HandoffOutputItem` + `handoff_span`), **structured output** (`output_type`), **run context**, **input/output guardrails**, **lifecycle hooks**, **ModelProvider** name resolution, **extended `ModelSettings`**, **local Tracing**, **Responses + Chat Completions**.

## Positioning

This is a **provider-neutral** agent SDK, not a client for the official OpenAI service. It runs
the Python SDK's agent loop (tools, handoffs, guardrails, sessions, tracing, HITL) on top of **any
endpoint that speaks the OpenAI Responses or Chat Completions protocol**: OpenAI itself, but also
gateways, self-hosted servers and third-party models. Point it at a different provider with a base
URL and key.

Consequently, features that exist only because OpenAI hosts them are **out of scope**: hosted tools
(web / file search, code interpreter, image generation, hosted MCP, computer, shell, apply_patch),
server-side prompts (`Agent.prompt`), the Conversations API and `responses.compact`, and OpenAI
trace export. Anything that only needs a compatible *protocol* is in scope. See
[`docs/DEVIATIONS.md`](./docs/DEVIATIONS.md#scope) for the full list.

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

The crate default is the **Responses** API, matching `set_default_openai_api()` in Python. Both
APIs stream standard Responses wire events such as `response.output_text.delta` (D-011); the
example harness below defaults `OPENAI_API` to `chat_completions` because most OpenAI-compatible
gateways implement Chat Completions SSE first. Set `OPENAI_API=responses` to use the Responses API.

| # | Example | What it shows |
|---|---------|----------------|
| 01 | `01_hello` | Minimal `Agent` + `Runner::run` |
| 02 | `02_tools` | `#[function_tool]` tool loop |
| 03 | `03_streamed` | `Runner::run_streamed` events |
| 04 | `04_handoff` | Multi-agent handoff |
| 05 | `05_as_tools` | `Agent.as_tool` orchestration |
| 06 | `06_hitl` | Approvals + `RunState` JSON resume |
| 07 | `07_always_approve` | Sticky `always_approve` |
| 08 | `08_structured_output` | `output_type` + typed `final_output_as::<T>()` |
| 09 | `09_guardrails` | Context, guardrails, `RunHooks` / `AgentHooks` |

```bash
cargo run --example 01_hello
cargo run --example 02_tools
cargo run --example 03_streamed
cargo run --example 04_handoff

OPENAI_API=responses cargo run --example 02_tools

EXAMPLE_INPUT="Translate 'Hello' to French and Spanish." \
  cargo run --example 05_as_tools
HITL_AUTO=approve cargo run --example 06_hitl
HITL_AUTO=approve cargo run --example 07_always_approve

cargo run --example 08_structured_output
cargo run --example 09_guardrails
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
| 2. OpenAI HTTP contracts (wiremock) | `cargo test` (default features; needs `openai`) |
| SQLite sessions | `cargo test --features sqlite` |
| 3. Python oracle parity | `.venv/bin/python scripts/run_parity.py --write-golden` then `cargo test --test parity_scenarios` |

Shared scenarios live in [`tests/parity/scenarios/`](./tests/parity/scenarios/). A scenario only
needs `expect` to run; the `.golden.json` produced by the Python oracle is optional and compared
when present. Regenerate it with `.venv/bin/python scripts/run_parity.py --write-golden` on a
machine that has `openai-agents==0.23.1` installed.

## Keeping context short

Long chats and tool-heavy runs outgrow the model's context. Two provider-neutral tools:

- **Per call (history untouched):** `ToolOutputTrimmer` shortens old tool outputs and
  `ContextWindowTrimmer` drops the oldest turns; set one as `RunConfig.call_model_input_filter`
  (combine with `chain_input_filters`).
- **Stored history:** wrap a session in `CompactingSession` with a `ModelSummarizer` (any model).
  Old turns become a short summary once the history passes a token or item trigger.

```rust
let session = CompactingSession::new(
    InMemorySession::shared("chat"),
    Arc::new(ModelSummarizer::new(summary_model)),
)
.trigger_tokens(6_000)
.keep_recent_turns(3)
.shared();
```

## Compatibility & deviations

- [`docs/COMPAT.md`](./docs/COMPAT.md) — support matrix
- [`docs/DEVIATIONS.md`](./docs/DEVIATIONS.md) — intentional differences from Python (must stay current)
- [`docs/CONTRIBUTING.md`](./docs/CONTRIBUTING.md) — commit message format and verification

HITL is supported: `needs_approval` (fixed or dynamic), `RunState` approve/reject, sticky
`always_approve` / `always_reject`, and `Agent.as_tool` nested approvals. The Rust `RunState` JSON
schema `openai-agents-rs/2` is a process-local snapshot format and is intentionally **not**
Python 1.18 wire-compatible (D-012) — use one language per durable state store.

## Still out of scope

OpenAI-hosted capabilities (see [Positioning](#positioning)), the sandbox runtime, and OpenAI trace
cloud export. Not yet built but in scope: local MCP servers.

## License

MIT
