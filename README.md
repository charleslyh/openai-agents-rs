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

## Crates and features

Depend on `openai-agents`; it re-exports everything below under the paths the SDK has always had.

| Crate | What it holds | Pulled in by |
| --- | --- | --- |
| `openai-agents-core` | agents, tools, handoffs, guardrails, the run loop, sessions, tracing, and the `Model` / `ModelProvider` / `McpServer` contracts. No HTTP client, database or process spawning | always |
| `openai-agents-macros` | `#[function_tool]` | always |
| `openai-agents-testing` | `ScriptedModel` and item builders (`openai_agents::testing`) | always |
| `openai-agents-openai` | Chat Completions and Responses adapters, `OpenAIProvider`, `CompatibleProvider` | feature `openai` (**default**) |
| `openai-agents-mcp` | MCP client: stdio and streamable HTTP, tool filters, `McpServerManager` | feature `mcp` |
| `openai-agents-sqlite` | `SqliteSession`, file-compatible with the Python SDK | feature `sqlite` |
| `openai-agents-providers` | provider presets (DeepSeek, Gemini, Groq, OpenRouter, Anthropic, Ollama), the capability matrix and the offline model | feature `providers` |
| `openai-agents-sessions` | `FileSession` / `FileRunStateStore`; Redis and Postgres variants | features `sessions`, `redis`, `postgres` |
| `openai-agents-observability` | console, JSONL and OpenTelemetry export, metrics | features `observability`, `otel` |
| `openai-agents-tools` | calculator, clock, read-only files, `web_fetch` with private networks refused | features `tools`, `webfetch` |
| `openai-agents-guardrails` | length, personal data, prompt injection, keywords, JSON Schema | feature `guardrails` |

```toml
openai-agents = "0.1"                                                        # core + OpenAI
openai-agents = { version = "0.1", features = ["providers", "tools", "guardrails"] }
openai-agents = { version = "0.1", features = ["mcp", "sqlite"] }            # plus MCP and SQLite
openai-agents = { version = "0.1", default-features = false }                # core only, bring your own Model
```

Features add up; `providers` is what makes `openai_agents::run` and `zero_config_options` work with
anything but OpenAI.

Contributors: each crate carries an `AGENTS.md` with its boundary rules and the traps worth
knowing before you change it (`crates/core/AGENTS.md`, `crates/mcp/AGENTS.md`, and so on).

With `default-features = false` the build contains no HTTP client. A model *name* then needs a
provider you set yourself (`RunConfig.model.provider`, or `set_default_model_provider_factory`).
With `openai`, `Runner` installs the OpenAI provider on its first call, so `OPENAI_API_KEY`
(or `OPENAI_BASE_URL`) is enough. See D-047 in [`docs/DEVIATIONS.md`](./docs/DEVIATIONS.md).

## Quick start

```rust
use openai_agents::prelude::*;

#[tokio::main]
async fn main() -> Result<(), AgentsError> {
    let agent = Agent::new("Assistant")
        .instructions("You only respond in haikus.")
        .model_name(std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".into()));
    let result = Runner::run(&agent, "Tell me about recursion.", RunOptions::new()).await?;
    println!("{}", result.final_output);
    Ok(())
}
```

`openai_agents::prelude` holds the types most programs need; everything else is at the crate root.

### Start a project

```bash
cargo install openai-agents-scaffold   # installs `cargo-agents`
cargo agents new my-agent              # templates: chat (default), tools, hitl
cd my-agent && AGENTS_OFFLINE=1 cargo run   # no key needed; set one to talk to a model
```

The generated project depends on `openai-agents` with `providers`, so a key in the environment (or
`AGENTS_OFFLINE=1`) is the whole setup. `--path` points it at a local checkout instead of the
published version.

## Examples

Numbered from core path → extended usage. Live OpenAI-compatible API (not `ScriptedModel`).

Required env: `OPENAI_API_KEY`, `OPENAI_MODEL`.  
Optional: `OPENAI_BASE_URL`, `OPENAI_API` (`chat_completions` | `responses`).

Which API a model uses (D-I): `OpenAIProvider::api(..)` / `CompatibleProvider::api(..)` first, then
`set_default_openai_api()`, then **Chat Completions** (what nearly every compatible server
implements) for every server, OpenAI included. Choose Responses explicitly with `.api(Responses)`,
`set_default_openai_api(Responses)` or the `openai_responses/<model>` name prefix.
Both APIs stream standard Responses wire events such as `response.output_text.delta` (D-011). The
example harness reads `OPENAI_API` (`chat_completions` | `responses`) to override the choice.

To talk to a server that needs no key, or to route several servers by model prefix, use
`CompatibleProvider::new(base_url)` (key optional, Chat Completions by default) with
`MultiProvider`: `"local/qwen2.5"`, `"gateway/my-model"`.

The examples live in [`examples/`](./examples) — a workspace member of their own, so they depend
on `openai-agents` the way your project would rather than reaching into the SDK's internals.

Ordered like the web demo's capability checklist, one knowledge point per example.

| Example | Checklist group | What it shows |
|---------|-----------------|----------------|
| `hello` | — | Minimal `Agent` + `Runner::run` |
| `streamed` | Conversation and memory | `Runner::run_streamed` events (streaming output) |
| `memory` | Conversation and memory | `Session` multi-turn memory |
| `compaction` | Conversation and memory | `CompactingSession` context compaction |
| `tools` | Tools | `#[function_tool]` tool loop |
| `parallel_tools` | Tools | Parallel tool calls |
| `tool_timeout` | Tools | Tool timeout (`ErrorAsResult` / `RaiseException`) |
| `tool_recovery` | Tools | Model retries after a tool failure |
| `mcp` | Tools | Local stdio MCP server with a `ToolFilter` |
| `handoff` | Multi-agent | Multi-agent handoff |
| `agents_as_tools` | Multi-agent | `Agent.as_tool` orchestration |
| `hitl` | Safety and governance | `RunOptions::resolve_with` approvals, or the explicit `RunState` JSON resume |
| `always_approve` | Safety and governance | Sticky `always_approve` |
| `input_guardrail` | Safety and governance | Input guardrail tripwire |
| `output_guardrail` | Safety and governance | Output guardrail tripwire |
| `tool_input_guardrail` | Safety and governance | Tool input guardrail (reject a call) |
| `tool_output_guardrail` | Safety and governance | Tool output guardrail (redact a result) |
| `verdict_transform` | Safety and governance | `Verdict::Transform`: redact the user's input and keep going |
| `structured_output` | Output and observability | `output_type` + typed `final_output_as::<T>()` |
| `run_context` | Output and observability | `RunOptions::context` read inside a tool |
| `hooks` | Output and observability | `RunHooks` / `AgentHooks` |
| `usage` | Output and observability | `RunResult::usage` |
| `tracing` | Output and observability | Custom `TracingProcessor` |
| `cancel` | Run control | `RunResultStreaming::cancel` |
| `settings` | Run control | `ModelSettings` temperature + retry, `max_turns` |
| `reasoning` | Run control | Typed `Reasoning` effort/summary for either API |
| `capability` | Rust-only | A named bundle of tools, instructions and guardrails |
| `zero_config` | Rust-only | `openai_agents::run`: provider and model from the environment, offline without a key |
| `structured_tool` | Rust-only | `#[derive(ToolMeta)]` + `TypedTool`: parameters, per-call approval, timeout and dependency as annotations on the argument struct; middleware |
| `deferred_capability` | Rust-only | `CapabilityParts::deferred`: tools stay out of the prompt until the model loads them |
| `builtin_guardrails` | Safety and governance | Ready-made guardrails from `openai-agents-guardrails` |
| `file_session` | Conversation and memory | `FileSession`: a conversation in a JSONL file |
| `trace_export` | Output and observability | Batched console/JSONL export and metrics |

They need `OPENAI_API_KEY` and `OPENAI_MODEL` in the environment; `OPENAI_BASE_URL` and
`OPENAI_API` (`chat_completions` | `responses`) are optional.

```bash
cargo run -p openai-agents-examples --example hello
cargo run -p openai-agents-examples --example tools

OPENAI_API=responses cargo run -p openai-agents-examples --example tools

EXAMPLE_INPUT="Translate 'Hello' to French and Spanish." \
  cargo run -p openai-agents-examples --example agents_as_tools
HITL_AUTO=approve cargo run -p openai-agents-examples --example hitl

cargo run -p openai-agents-examples --example structured_output
```

(The `-p openai-agents-examples` is needed because the workspace has more than one default member,
so a bare `cargo run --example` cannot tell which package you mean.)

### Web demo

A local web console (axum + SSE) in its own package, [`demo/web`](./demo/web): one chat box over a
multi-agent network, a capability checklist, and a chaos drawer that injects faults. The
examples above follow its checklist.

```bash
cargo run -p openai-agents-web-demo
# then open http://127.0.0.1:3000
```

Uses a real model, so `OPENAI_API_KEY` and `OPENAI_MODEL` are required (startup fails without
them). Extra variables:

- `DEMO_ADDR` — listen address, default `127.0.0.1:3000`. A non-loopback address prints a warning:
  the demo has no authentication.
- `DEMO_DEBUG` — `1`/`0`. Controls fault injection and `/api/debug/*`. On by default for loopback
  addresses, off otherwise; when off, injected faults in requests are ignored.
- `DEMO_HMR` — `1`/`0`. Live reload of `index.html`: the page is read from disk and the browser
  reloads when the file changes, so front-end edits need no rebuild. On by default in debug builds
  (plain `cargo run`) while debug features are on; release builds serve the embedded copy.

The page has a capability checklist (22 items, lit up as you trigger them) and a debug drawer with
fault presets: LLM errors and retry policies, tool delays and failures, summarizer and session
failures, MCP failures, dropped streams, expired or replayed approvals, and server state reset. The
MCP demo server is this same executable re-run with `--mcp-server`. The demo package builds the
facade's `mcp` feature because of it.

## Standard reference

Vendored tree: [`vendor/openai-agents-python`](./vendor/openai-agents-python) pinned by [`vendor/PINNED_VERSION`](./vendor/PINNED_VERSION) (`v0.23.1`).

```bash
bash scripts/sync_vendor.sh --check
```

(Network note: preferred `git submodule` may fail behind restricted GitHub access; the sync script can download the release tarball.)

## Verification

| Layer | Command |
|-------|---------|
| 0. Format, lints, docs, vendor pin | `cargo fmt --all -- --check` · `cargo clippy --all-features --all-targets -- -D warnings` · `RUSTDOCFLAGS=-D warnings cargo doc --no-deps --all-features` · `bash scripts/sync_vendor.sh --check` |
| 1. ScriptedModel behavior | `cargo test --no-default-features` |
| 1.5 Property tests (generated inputs) | `cargo test --test property_core` |
| 2. OpenAI HTTP contracts (wiremock) | `cargo test` (default features; needs `openai`) |
| 2.5 Whole request bodies (insta snapshots) | `cargo test --test openai_wiremock` |
| SQLite sessions | `cargo test --features sqlite` |
| MCP interoperability (real Python MCP server) | `cargo test --features mcp --test mcp_interop` (needs the venv, skips without it) |
| 3. Python oracle parity | `.venv/bin/python conformance/run_parity.py --check` then `cargo test --test parity_scenarios` |
| 4. Live endpoint smoke (nightly, non-blocking) | `.github/workflows/nightly.yml` |
| 5. Latency benchmarks | `cargo bench` |
| 6. Line coverage ratchet | `.github/workflows/coverage.yml` |

Shared scenarios live in [`conformance/scenarios/`](./conformance/scenarios/). A scenario only
needs `expect` to run; the `.golden.json` produced by the Python oracle is optional and compared
when present. `--check` fails on drift and never overwrites; regenerate deliberately with
`.venv/bin/python conformance/run_parity.py --write-golden` on a machine that has
`openai-agents==0.23.1` installed.

CI runs every layer on Linux and macOS — see `.github/workflows/ci.yml` and
[`docs/CONTRIBUTING.md`](./docs/CONTRIBUTING.md#layers).

## MCP tools

Enable the `mcp` feature. Any [MCP](https://modelcontextprotocol.io) server's tools become tools of
your agent, whichever model you use:

```rust
let server = McpClient::stdio(
    StdioParams::new("npx").args(["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
)
.tool_filter(ToolFilter::block(["write_file"]))
.shared();
let agent = Agent::new("files").mcp_servers(vec![server.clone()]);
// ... run the agent ...
server.cleanup().await?;
```

`McpClient::streamable_http(StreamableHttpParams::new(url).header("Authorization", "Bearer ..."))`
reaches a remote server. See D-041 in [`docs/DEVIATIONS.md`](./docs/DEVIATIONS.md).

## Other providers

Use any OpenAI-compatible server by passing its base URL and key:

```rust
let model = Arc::new(OpenAIChatCompletionsModel::new(
    "deepseek-chat",
    std::env::var("API_KEY").unwrap(),
    Some("https://api.deepseek.com/v1"),
));
```

The Chat Completions adapter is checked against the Python converter and tolerates common
deviations (tool calls without ids, list-valued `content`, error chunks in a stream). Things that
only OpenAI accepts, such as `stream_options`, are sent to other servers only when you ask
(`ModelSettings.include_usage`). See D-040 in [`docs/DEVIATIONS.md`](./docs/DEVIATIONS.md).

## Keeping context short

Long chats and tool-heavy runs outgrow the model's context. Two provider-neutral tools:

- **Per call (history untouched):** `ToolOutputTrimmer` shortens old tool outputs and
  `ContextWindowTrimmer` drops the oldest turns; set one as `RunConfig.model.input_filter`
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
- [`docs/IMPROVEMENT-PLAN.md`](./docs/IMPROVEMENT-PLAN.md) — where the SDK is heading: milestones, design decisions, and the improvement backlog
- [`STABILITY.md`](./STABILITY.md) — which parts of the API are stable, and the deprecation policy
- [`CHANGELOG.md`](./CHANGELOG.md) — what changed in each release
- [`docs/MIGRATION-FROM-PYTHON.md`](./docs/MIGRATION-FROM-PYTHON.md) — Python → Rust API mapping and what to write yourself
- [`docs/CONTRIBUTING.md`](./docs/CONTRIBUTING.md) — commit message format and verification
- [`docs/COMPARISON.md`](./docs/COMPARISON.md) — where this SDK differs from the Python one, Rig and ADK-Rust
- [`llms.txt`](./llms.txt) — the docs as one index for coding assistants ([`llms-full.txt`](./llms-full.txt) is the flat file)

HITL is supported as tool-call *resolution*: a call pauses when `needs_resolution` (Python:
`needs_approval`; fixed or dynamic) says so, and the host resolves it on the `RunState` — approve /
reject (with sticky `always_approve` / `always_reject`), or hand back a result produced elsewhere
with `provide_output` (D-048) — including calls inside `Agent.as_tool` nested runs. The Rust
`RunState` JSON schema `openai-agents-rs/4` is a process-local snapshot format and is intentionally **not**
Python 1.18 wire-compatible (D-012) — use one language per durable state store.

## Still out of scope

OpenAI-hosted capabilities (see [Positioning](#positioning)) and the sandbox runtime.

## License

MIT
