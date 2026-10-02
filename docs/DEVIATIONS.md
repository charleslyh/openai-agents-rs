# Deviations from openai-agents-python (v0.23.1)

Record every intentional or temporary divergence from the standard Python SDK.
Silent behavioral forks are bugs.

| ID | Standard location | Rust behavior | Reason | Plan to align |
|----|-------------------|---------------|--------|---------------|
| D-001 | `Runner.run_sync` | `Runner::run_blocking` | Rust has no ambient asyncio loop; blocking entry uses an owned Tokio runtime | Document only |
| D-002 | `@function_tool` docstring → JSON schema | `#[function_tool]` maps Rust types to JSON Schema; no docstring introspection | Rust has no runtime docstrings; primitives/`Option`/serde values supported | Extend schema mapping as needed |
| D-003 | `Agent.handoffs` / `handoff()` | Supported basic handoff (tool name + agent switch); no input filters / nest history | Filters deferred | Partial; filters later |
| D-004 | OpenAI BackendSpanExporter / `set_tracing_export_api_key` | Local / in-memory processors only | Phase-1 local tracing | Optional later |
| D-005 | `Runner.run_streamed` | `Runner::run_streamed` + `RunResultStreaming` | Supported; see D-011 for raw event shape | Aligned (item events) |
| D-006 | MCP / sessions / guardrails / hosted tools / sandbox | Not exported | Phase-1 subset | Later phases |
| D-007 | Package layout | Single crate `openai-agents` with modules + Cargo features | Mirrors Python single package; no empty facade | N/A (aligned intent) |
| D-008 | Default max turns `10` | Same default (`DEFAULT_MAX_TURNS = 10`) | — | Aligned |
| D-009 | `function_tool` failure_error_function defaults | Tool errors returned as string results by default | Match Python default tool error surfacing | Keep aligned |
| D-010 | Prefer git submodule for vendor | Vendored via release tarball + `vendor/PINNED_VERSION` when GitHub submodule clone is unavailable | Network / CI portability | Prefer submodule when accessible; `scripts/sync_vendor.sh` remains source of truth |
| D-011 | Token-level `RawResponsesStreamEvent` from `Model.stream_response` | Chat Completions streams `output_text.delta` (+ optional `reasoning_text.delta` for `reasoning_content`); Responses / Scripted still emit synthetic `response.completed` after `get_response` | TokenHub & most OpenAI-compatible gateways support chat SSE first; DeepSeek exposes CoT via `reasoning_content` | Add Responses SSE later |
| D-012 | Python `RunState` schema 1.18 wire format | Rust schema `openai-agents-rust/1` via `to_json`/`from_json`; sticky `always_approve`/`always_reject`; `Agent.as_tool` nested HITL | Not byte-compatible with Python snapshots; process-local + Rust JSON is enough for HITL | Optional Python-compatible exporter later |

## How to add an entry

1. Assign the next `D-NNN` ID.
2. Cite the Python path (file + symbol) under `vendor/openai-agents-python` @ `v0.23.1`.
3. State Rust behavior and whether callers can observe the difference.
4. Mark align plan: `Document only` / `Phase-N` / `Bug (fix)`.
