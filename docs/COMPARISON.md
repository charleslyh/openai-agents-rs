# How this SDK compares

Two Rust neighbours: [Rig](https://github.com/0xPlaygrounds/rig) and
[ADK-Rust](https://github.com/zavora-ai/adk-rust). And the Python SDK this one ports.

The rows about this SDK come from this repository's code and tests. The rows about the other two come
from their public documentation at the time of writing and are deliberately coarse — both move fast,
so check their docs before relying on a cell.

| | This SDK | [openai-agents-python](https://github.com/openai/openai-agents-python) | Rig | ADK-Rust |
| --- | --- | --- | --- | --- |
| Starting point | port of v0.23.1, pinned in `vendor/PINNED_VERSION` | the reference implementation | independent framework | a Rust agent development kit in the style of Google's ADK |
| Behaviour checked against a reference implementation | yes: 13 declarative scenarios run through both implementations (`conformance/`) | — | no shared oracle | follows ADK's model, no shared oracle |
| Scope | the Python SDK's agent loop, provider-neutral | hosted OpenAI features included | provider abstraction, agents, RAG with 10+ vector stores, WASM | agents, workflow and graph agents, realtime voice, RAG, A2A servers |
| Extensibility | one trait each: `Tool`, `Model`, `ModelProvider`, `Session`, `TracingProcessor`, `Capability` | same set, Python protocols | `Tool` trait, provider traits, hooks | `Agent` trait, `#[tool]` macro, callbacks |
| Tool arguments | `#[function_tool]` or `FunctionTool::typed` from one `schemars` struct | `@function_tool` from type hints | `Tool` trait with JSON Schema | `#[tool]` macro with `schemars` |
| Business context | typed values in an `Extensions` map, a missing one reported with its type name | `RunContext` with any user object | agent state | session state |
| Pausing for approval | `needs_resolution` + `RunState` JSON + `resolve_with` closure | `needs_approval` + `RunState` | own state machine | graph interrupts, HITL |
| Guardrail verdicts | `Pass`, `Fail`, **`Transform`** (a rewrite is a result, not a refusal) | tripwire only | — | callbacks and guardrails |
| Provider presets | a table with a capability matrix that refuses rather than degrades silently | OpenAI plus compatible endpoints | 20+ providers, feature-gated | 14+ providers, feature-gated |
| On-demand tool loading | `CapabilityParts::deferred` | — | — | — |
| Scaffolding | `cargo agents new`, 3 templates | — | examples, no scaffolding CLI | `cargo adk new`, 13 templates, 9 addons, 5 enterprise patterns |
| Offline development | `AGENTS_OFFLINE=1` | `ScriptedModel` in tests | cassette record/replay | provider-dependent |

## Where this SDK is weaker, on purpose

Being honest about the trade: Rig has a far wider integration surface (20+ providers, 10+ vector
stores as companion crates, an onchain kit, `wasm32-unknown-unknown` support). ADK-Rust has a much
larger feature set (realtime voice, graph workflows with checkpointing and human-in-the-loop
interrupts, RAG with six vector-store backends, A2A servers, declarative tool scopes, a runtime UI)
and a far more developed scaffolding CLI.

This SDK deliberately skipped all of that. `docs/IMPROVEMENT-PLAN.md` lists the non-goals: hosted
OpenAI tools, real-time voice, RAG and vector stores, a visual studio. What it spends its complexity
on instead is *behavioural fidelity* — a Python oracle it can be diffed against — plus the parts a
Rust SDK can do better than a Python one: construction-time validation, `#[non_exhaustive]` everywhere
before the first release, and a `Transform` verdict that redacts instead of refusing.

## What is different from Python on purpose

`docs/DEVIATIONS.md` is the authoritative list, with an id per row (`D-…` for behaviour differences,
`X-…` for Rust-only additions). The ones worth knowing before choosing this SDK:

- **No hosted OpenAI tools.** Web search, file search, code interpreter, image generation, hosted
  MCP, computer use, `Agent.prompt` and the Conversations API are out of scope: they exist only
  because OpenAI hosts them. Anything that needs only a compatible *protocol* is in scope.
- **`RunState` is not Python's wire format.** The JSON schema is `openai-agents-rs/4`; use one
  language per durable state store.
- **Additions with no Python counterpart**: `Capability` bundles and `deferred` loading,
  `Verdict::Transform`, the provider presets, the ready-made tools and guardrails, `openai_agents::run`.
  They are registered under "extensions" in `docs/DEVIATIONS.md` and kept out of the parity
  comparison, so the behaviour alignment stays a clean check.

## Choosing

- Coming from the Python SDK, or depending on its exact behaviour: this SDK — then read
  `docs/MIGRATION-FROM-PYTHON.md`.
- Wanting the widest provider and vector-store coverage, or WASM: Rig.
- Wanting voice, graph workflows, A2A, a runtime UI, and a template for each: ADK-Rust.
- Wanting an agent loop whose behaviour you can diff against a reference implementation: this is the
  only one of the three that can do that.
