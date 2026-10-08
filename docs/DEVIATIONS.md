# Deviations from openai-agents-python (v0.23.1)

Record every intentional or temporary divergence from the standard Python SDK.
Silent behavioral forks are bugs.

Status values: **Aligned** (resolved, kept for history) · **Accepted** (intentional, often a
Rust language difference) · **Gap** (known missing capability, tracked by COMPAT.md).

## Resolved in Phase-2

| ID | Standard location | Rust behavior | Status |
|----|-------------------|---------------|--------|
| B1 | `RunResult.last_agent` | Previously returned the *starting* agent. Now carries the agent that actually finished. | Aligned |
| B4 | `maybe_reset_tool_choice` | Reset now happens at turn start, driven by a tool-use tracker, so it also applies after a turn that ended early via `stop_on_first_tool` / `StopAtTools`. | Aligned |
| B5 | `_build_tool_result_items` | A `stop_on_first_tool` run now records a `ToolCallOutputItem` for *every* tool in the batch, not only the first. | Aligned |
| B6 | `ResponseFunctionToolCall.call_id` | A `function_call` without `call_id` raises `ModelError::Behavior` instead of falling back to `"call"`. | Aligned |
| D-A | `HandoffCallItem` / `HandoffOutputItem` / `ReasoningItem` | Implemented; handoffs and reasoning now produce their own run items. | Aligned |
| D-B | `tracing.handoff_span` | Emitted for every handoff with `from_agent` / `to_agent`. | Aligned |
| D-C | Mixed approval batches | Verified against v0.23.1: Python also executes the calls that do not need approval before pausing, so the Rust behaviour matches. | Aligned |
| D-D | Non-stream `RawResponse` emit | Dead code removed; only the streaming path emits raw events. | Aligned |
| D-E | `flush_traces` | Now calls `TracingProcessor::force_flush` on all installed processors. | Aligned |
| D-F | `RunConfig.tracing_disabled` | Per-run setting is now authoritative for spans as well as the root trace (previously only the global switch was honoured). | Aligned |
| D-G | `TracingProcessor.on_span_start` / `on_trace_start` | `InMemoryProcessor` now records start events. | Aligned |
| D-H | `RunConfig.trace_id` / `group_id` / `trace_metadata` | Supported and forwarded to `Trace`. | Aligned |
| D-I | Default OpenAI API | Crate default is Responses (Python-aligned); only the *example harness* defaults to `chat_completions`. README now says so. | Aligned |
| D-J | `ModelTracing.ENABLED_WITHOUT_DATA` | Driven by `RunConfig.trace_include_sensitive_data` (env `OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA`). | Aligned |
| D-K | Chat Completions + `conversation_id` | Returns `ModelError::Unsupported` instead of silently dropping it. | Aligned |
| D-L | Error source chain | `AgentsError::Tool` / `Internal` carry an optional `#[source]`; helpers `tool_with_source` / `internal_with_source` exist. | Aligned |

## Accepted deviations

| ID | Standard location | Rust behavior | Reason | Plan to align |
|----|-------------------|---------------|--------|---------------|
| D-001 | `Runner.run_sync` | `Runner::run_blocking` | Rust has no ambient asyncio loop; the blocking entry uses an owned Tokio runtime | Document only |
| D-002 | `@function_tool` docstring → JSON schema | `#[function_tool]` derives the schema from Rust types via `schemars`; the function doc comment becomes the tool description and `///` docs on fields become property descriptions | Rust has no runtime docstrings/annotations | Accepted |
| D-003 | `Agent.handoffs` / `handoff()` | Basic handoff supported; no `input_filter`, `nest_handoff_history`, `input_type` or `on_handoff` | Deferred | Later phase |
| D-004 | OpenAI `BackendSpanExporter` / `set_tracing_export_api_key` | Local / in-memory processors only | Phase-2 still local | Optional later |
| D-005 | `Runner.run_streamed` | `Runner::run_streamed` + `RunResultStreaming` | Naming only; the raw event vocabulary matches Python (D-011) | Aligned |
| D-006 | MCP / sessions / hosted tools / sandbox | Not exported | Out of scope | Later phases |
| D-007 | Package layout | Single crate `openai-agents` with modules + Cargo features | Mirrors the Python single package; no empty facade | N/A (aligned intent) |
| D-008 | Default max turns `10` | Same default (`DEFAULT_MAX_TURNS = 10`) | — | Aligned |
| D-009 | `function_tool` `failure_error_function` | Tool errors are surfaced as results by default; there is no per-tool error formatter hook | `RunConfig.tool_error_formatter` not ported | Later phase |
| D-010 | Prefer git submodule for vendor | Vendored via release tarball + `vendor/PINNED_VERSION` | Network / CI portability | Prefer submodule when accessible |
| D-011 | Token-level `RawResponsesStreamEvent` from `Model.stream_response` | `StreamEvent::RawResponse.data` is a standard Responses wire event, matching Python's `RawResponsesStreamEvent.data`. Responses forwards provider events verbatim (`openai_responses.py:781`); Chat Completions synthesizes them the way `chatcmpl_stream_handler.py` does; `ScriptedModel` replays a step the way `testing/model.py` does. See [Streaming wire contract](#streaming-wire-contract) | Not ported: `response.refusal.*`, `response.output_text.annotation.added` and logprob payloads (emitted empty); usage token details follow D-016 | Aligned |
| D-012 | Python `RunState` schema 1.18 wire format | Rust schema `openai-agents-rs/2` via `to_json`/`from_json`; sticky `always_approve`/`always_reject`; `Agent.as_tool` nested HITL | Not byte-compatible with Python snapshots; process-local + Rust JSON is enough for HITL. The pre-rename ids `openai-agents-rust/1` and `openai-agents-rust/2` are still accepted on load | Optional Python-compatible exporter later |
| D-013 | `MultiProvider` provider registry | Only `openai` and `openai_chat_completions` prefixes are built in; others are registered explicitly | No dynamic entry-point discovery in Rust | Accepted |
| D-022 | `ScriptedModel.set_default_usage` | Applies to steps whose usage is empty, matching Python | — | Aligned |
| D-014 | `RunContextWrapper[TContext]` | Context is type-erased (`Arc<dyn Any + Send + Sync>`) with `context::<T>()` / `try_context::<T>()` | Avoids threading a generic parameter through `Agent` / `Runner`; type errors surface at runtime, mitigated by an explicit error message | Accepted |
| D-015 | `Agent.hooks` / `RunHooks` default implementations | Provided as `async_trait` methods with empty bodies, so implementors override only what they need | Same shape as Python base classes | Accepted |
| D-016 | `Usage` token details | Only `requests` / `input_tokens` / `output_tokens` / `total_tokens` | `input_tokens_details` / `output_tokens_details` / `request_usage_entries` not ported | Later phase |
| D-017 | `ModelSettings` non-portable fields | `prompt_cache_retention`, `context_management`, `prompt_cache_options`, `preserve_raw_usage`, `extra_query` are not exposed | Rarely used | Later phase |
| D-024 | `extra_args` collision error type | A colliding `extra_args` key returns `ModelError::Behavior` (surfacing as `AgentsError::Model`) | Python raises a bare `TypeError` from the client call; Rust has no kwargs layer to raise from, so the message mirrors Python's (`… got multiple values for keyword argument 'x'`) and the failure travels through the typed error channel | Accepted |
| D-025 | `extra_args` accepts arbitrary keys | Any key is written straight into the request body | The openai SDK's `create()` has no `**kwargs`: `extra_args` keys must be *typed* SDK parameters (`service_tier`, `prompt_cache_key`, `logprobs`, …) and an unknown key raises `TypeError: … got an unexpected keyword argument`. Arbitrary body fields belong in `extra_body` ("Add additional JSON properties to the request"). Rust has no typed SDK surface to validate against, and an OpenAI-specific allowlist would reject legitimate OpenAI-compatible gateway parameters, so it stays permissive | Accepted — prefer `extra_body` for arbitrary fields |
| D-019 | `agents.testing` has no HTTP mocks | `ScriptedModel` is the only shipped test double. HTTP contract tests live in `tests/openai_wiremock.rs` and drive `wiremock` directly, so no mock layer is part of the public API | Aligns the `testing` surface with Python; the old `MockResponses` / `MockCompletions` wrappers were removed | Accepted |
| D-020 | `ModelStep.responder` / `stream_events` / `retry_advice` | Not ported; steps are a plain output/error pair | Advanced streaming and retry features are out of scope | Later phase |
| D-021 | `ModelCall` recorded fields | Records `streamed` and `output_schema_name`, but not `output_schema`, `handoffs` or `prompt` objects | `ModelRequest` does not carry handoffs; Python exposes the live objects | Later phase |
| D-018 | `function_tool` variadic / `Annotated` parameters | Not supported; the macro accepts plain owned-typed identifiers and an optional leading context parameter | Rust signatures cannot express `*args` / `**kwargs` with types | Accepted |
| D-023 | Chat Completions synthesized response id | The synthesized `response` object and `ModelResponse.response_id` carry the provider's `id` (e.g. `chatcmpl-…`) when the gateway sends one; `FAKE_RESPONSES_ID` only when it does not | Python always uses `FAKE_RESPONSES_ID` for Chat Completions. The id is only fed back as `previous_response_id`, which the Chat Completions adapter ignores | Accepted |

## Streaming wire contract

`StreamEvent::RawResponse { data }` carries **one Responses API wire event** (Python:
`RawResponsesStreamEvent.data`, `stream_events.py:11-20`). Both OpenAI adapters and
`ScriptedModel` emit the same vocabulary, so a consumer never branches on the backend.

Which emitter does what:

| Emitter | Source | Python reference |
|---------|--------|------------------|
| `OpenAIResponsesModel` | provider SSE events, forwarded verbatim | `openai_responses.py:781` (`yield chunk`) |
| `OpenAIChatCompletionsModel` | synthesized from chat chunks | `chatcmpl_stream_handler.py` (15 event types) |
| `ScriptedModel` / default `Model::stream_response` | a finished response replayed | `testing/model.py` step expansion |

Rules the three share:

- **Vocabulary**: only `response.*` wire type names. No dialect names (the pre-D-011
  `output_text.delta` / `reasoning_text.delta` are gone).
- **Shape**: `item_id` / `output_index` / `content_index` / `summary_index` / `delta` /
  `logprobs` / `sequence_number`, matching the Python event objects. `item_id` is the item's own
  `id` (or `call_id`), falling back to `FAKE_RESPONSES_ID` (`"__fake_id__"`) when absent — which
  is the case for everything the Chat Completions adapter synthesizes, matching Python.
- **Ordering**: `response.created` → `output_item.added` → `content_part.added` → deltas →
  `*_done` → `output_item.done` → `response.completed`.
- **Numbering**: `sequence_number` is 0-based and monotonic (`chatcmpl_stream_handler.py:184`).
  Forwarded events keep the provider's number and are never renumbered; a synthesized terminal
  event continues from the last number seen.
- **`output_index`**: the reasoning item occupies 0, function calls take the next slots in
  arrival order, and the assistant message is assigned lazily right after the calls known when
  its first delta arrives (`_StreamOutputLayout`).
- **Terminal event**: `{"type":"response.completed","response":{…}}` with the full `response`
  object (`id`, `object`, `created_at`, `model`, `status`, `output`, `usage`, `tools`,
  `tool_choice`, `parallel_tool_calls`, `top_p`, `temperature`).

Deliberately not ported (see D-011): `response.refusal.*` and
`response.output_text.annotation.added` are not synthesized, `logprobs` payloads are emitted as
`[]`, and `usage` carries no token details (D-016). The per-event status per emitter is tabulated
in [COMPAT.md](./COMPAT.md#streaming-wire-events).

## How to add an entry

1. Assign the next `D-NNN` ID.
2. Cite the Python path (file + symbol) under `vendor/openai-agents-python` @ `v0.23.1`.
3. State the Rust behavior and whether callers can observe the difference.
4. Mark the align plan: `Document only` / `Accepted` / `Gap` / `Later phase` / `Bug (fix)`.
