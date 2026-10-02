# Phase-1 Compatibility Matrix

Standard reference: `vendor/openai-agents-python` @ **v0.23.1** (`openai-agents==0.23.1`).

Legend: **S** = supported · **P** = partial · **N** = not in Phase-1 · **D** = intentional deviation (see [DEVIATIONS.md](./DEVIATIONS.md))

| Capability | Status | Notes |
|------------|--------|-------|
| `Agent` (name, instructions, tools, model, model_settings, tool_use_behavior, reset_tool_choice) | S | Dynamic instructions as `String` only in Phase-1 |
| `Runner::run` / `run_blocking` | S | D-001 naming for sync entry |
| `FunctionTool` | S | Manual schema or `#[function_tool]` |
| `@function_tool` / proc-macro | S | D-002: type→schema, not docstring |
| `ScriptedModel` | S | Parity with `agents.testing.ScriptedModel` core |
| OpenAI Responses API model | S | via `async-openai` + raw HTTP for Responses |
| OpenAI Chat Completions model | S | via `async-openai` |
| Default API = Responses | S | `set_default_openai_api` |
| Local tracing (`trace`, `agent_span`, `function_span`, `generation_span`) | S | |
| OpenAI trace cloud export | N | D-004 |
| Handoffs | S | Basic `handoff()` + tool transfer; D-003 filters deferred |
| `run_streamed` / `RunResultStreaming` | S | Item + agent events; raw events synthetic (D-011) |
| MCP / sessions / guardrails / hosted tools / sandbox / HITL | N | D-006 |
| `Agent.as_tool` | N | D-006 |

## Verification layers

1. `cargo test --no-default-features` — ScriptedModel behavior
2. `cargo test --features openai` — wiremock OpenAI request shape
3. `python scripts/run_parity.py` + Rust parity tests — Python oracle golden
