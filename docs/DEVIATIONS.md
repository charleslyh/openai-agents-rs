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
| B7 | `default_tool_error_function` / `failure_error_function` | A tool that returns `Err` no longer aborts the run: the model receives `DEFAULT_TOOL_ERROR_MESSAGE` (the error text is not exposed). `FunctionTool::with_failure_error_function` formats it, `raise_on_error()` restores abort (Python: `failure_error_function=None`). The formatter is synchronous | Aligned |
| B8 | `turn_resolution.py` handoff execution | A turn with a handoff now runs its sibling function tools first, then the first handoff; later handoffs get a `ToolCallOutputItem` with `"Multiple handoffs detected, ignoring this one."`. Every call id has an output. Handoffs take precedence over `tool_use_behavior` stops, as in Python | Aligned |
| B9 | `InputGuardrail.run_in_parallel` | `run_in_parallel(false)` guardrails finish before the first model call and trip without calling the model. Parallel guardrails race the model call: a tripwire drops the in-flight call instead of waiting for it | Aligned |
| B10 | `RunConfig.tool_not_found_behavior` | Unknown tool raises `ModelError::Behavior("Tool X not found in agent Y")`; `ToolNotFoundBehavior::ReturnErrorToModel` answers `"Tool 'X' not found."` and continues | Aligned |
| D-M | `RunConfig.call_model_input_filter` | Ported as an async closure over `CallModelData { model_data, agent, context }` returning `Result<ModelInputData, AgentsError>`; applied before the LLM hooks, for one call only. Python passes the raw context value; Rust passes the `RunContextWrapper`. Python also deduplicates input items afterwards; Rust does not | Aligned |
| D-N | Chat Completions without `usage` | The request still counts (`Usage.requests = 1`, zero tokens), as in Python; a Responses reply without `usage` stays at 0 requests, also as in Python. Previously both reported 0 | Aligned |
| D-L | Error source chain | `AgentsError::Tool` / `Internal` carry an optional `#[source]`; helpers `tool_with_source` / `internal_with_source` exist. | Aligned |

## Accepted deviations

| ID | Standard location | Rust behavior | Reason | Plan to align |
|----|-------------------|---------------|--------|---------------|
| D-001 | `Runner.run_sync` | `Runner::run_blocking` | Rust has no ambient asyncio loop; the blocking entry uses an owned Tokio runtime | Document only |
| D-002 | `@function_tool` docstring → JSON schema | `#[function_tool]` derives the schema from Rust types via `schemars`; the function doc comment becomes the tool description and `///` docs on fields become property descriptions | Rust has no runtime docstrings/annotations | Accepted |
| D-003 | `Agent.handoffs` / `handoff()` | Handoffs support `input_filter` (per handoff, or `RunConfig.handoff_input_filter`; the handoff's own wins). `HandoffInputData` is flattened to Responses input items (`input_history` / `pre_handoff_items` / `new_items`) and has no `input_items`. A filter with `conversation_id` / `previous_response_id` raises `UserError`, as in Python. The filter only changes the next agent's input; `RunResult.new_items` keeps the unfiltered items. `on_handoff` is supported (`with_on_handoff`, or `with_on_handoff_input(schema, f)` which replaces `input_type`: the schema is made strict and advertised, the arguments only have to parse as JSON and are not validated against the schema). No `nest_handoff_history` or `handoff_history_mapper` | Rest later |
| D-004 | OpenAI `BackendSpanExporter` / `set_tracing_export_api_key` | Local / in-memory processors only | Phase-2 still local | Optional later |
| D-005 | `Runner.run_streamed` | `Runner::run_streamed` + `RunResultStreaming` | Naming only; the raw event vocabulary matches Python (D-011) | Aligned |
| D-006 | MCP / sessions / hosted tools / sandbox | Not exported | Out of scope | Later phases |
| D-007 | Package layout | Single crate `openai-agents` with modules + Cargo features | Mirrors the Python single package; no empty facade | N/A (aligned intent) |
| D-008 | Default max turns `10` | Same default (`DEFAULT_MAX_TURNS = 10`) | — | Aligned |
| D-009 | `function_tool` `failure_error_function` | Per-tool `failure_error_function` is ported (B7); async formatters and `RunConfig.tool_error_formatter` (approval-rejected / tool-not-found messages) are not | `RunConfig.tool_error_formatter` not ported | Later phase |
| D-010 | Prefer git submodule for vendor | Vendored via release tarball + `vendor/PINNED_VERSION` | Network / CI portability | Prefer submodule when accessible |
| D-011 | Token-level `RawResponsesStreamEvent` from `Model.stream_response` | `StreamEvent::RawResponse.data` is a standard Responses wire event, matching Python's `RawResponsesStreamEvent.data`. Responses forwards provider events verbatim (`openai_responses.py:781`); Chat Completions synthesizes them the way `chatcmpl_stream_handler.py` does; `ScriptedModel` replays a step the way `testing/model.py` does. See [Streaming wire contract](#streaming-wire-contract) | Not ported: `response.refusal.*`, `response.output_text.annotation.added` and logprob payloads (emitted empty); usage token details are emitted (D-016) | Aligned |
| D-012 | Python `RunState` schema 1.18 wire format | Rust schema `openai-agents-rs/2` via `to_json`/`from_json`; sticky `always_approve`/`always_reject`; `Agent.as_tool` nested HITL | Not byte-compatible with Python snapshots; process-local + Rust JSON is enough for HITL. The pre-rename ids `openai-agents-rust/1` and `openai-agents-rust/2` are still accepted on load | Optional Python-compatible exporter later |
| D-013 | `MultiProvider` provider registry | Only `openai` and `openai_chat_completions` prefixes are built in; others are registered explicitly | No dynamic entry-point discovery in Rust | Accepted |
| D-022 | `ScriptedModel.set_default_usage` | Applies to steps whose usage is empty, matching Python | — | Aligned |
| D-014 | `RunContextWrapper[TContext]` | Context is type-erased (`Arc<dyn Any + Send + Sync>`) with `context::<T>()` / `try_context::<T>()` | Avoids threading a generic parameter through `Agent` / `Runner`; type errors surface at runtime, mitigated by an explicit error message | Accepted |
| D-015 | `Agent.hooks` / `RunHooks` default implementations | Provided as `async_trait` methods with empty bodies, so implementors override only what they need | Same shape as Python base classes | Accepted |
| D-016 | `Usage` token details | `Usage` carries `input_tokens_details` (`cached_tokens`, `cache_write_tokens`), `output_tokens_details` (`reasoning_tokens`) and `request_usage_entries`, aggregated by `add` like Python. Both OpenAI adapters fill them (Responses `*_tokens_details`; Chat `prompt_tokens_details` / `completion_tokens_details`), `total_tokens` is the provider's value, and `response.completed` carries the details. Not ported: `preserve_raw_usage` (D-017), per-request details from adapter-private attributes. Loading an older `RunState` without details defaults them to 0 | Aligned |
| D-017 | `ModelSettings` non-portable fields | `prompt_cache_retention`, `context_management`, `prompt_cache_options`, `preserve_raw_usage`, `extra_query` are not exposed | Rarely used | Later phase |
| D-024 | `extra_args` collision error type | A colliding `extra_args` key returns `ModelError::Behavior` (surfacing as `AgentsError::Model`) | Python raises a bare `TypeError` from the client call; Rust has no kwargs layer to raise from, so the message mirrors Python's (`… got multiple values for keyword argument 'x'`) and the failure travels through the typed error channel | Accepted |
| D-025 | `extra_args` accepts arbitrary keys | Any key is written straight into the request body | The openai SDK's `create()` has no `**kwargs`: `extra_args` keys must be *typed* SDK parameters (`service_tier`, `prompt_cache_key`, `logprobs`, …) and an unknown key raises `TypeError: … got an unexpected keyword argument`. Arbitrary body fields belong in `extra_body` ("Add additional JSON properties to the request"). Rust has no typed SDK surface to validate against, and an OpenAI-specific allowlist would reject legitimate OpenAI-compatible gateway parameters, so it stays permissive | Accepted — prefer `extra_body` for arbitrary fields |
| D-019 | `agents.testing` has no HTTP mocks | `ScriptedModel` is the only shipped test double. HTTP contract tests live in `tests/openai_wiremock.rs` and drive `wiremock` directly, so no mock layer is part of the public API | Aligns the `testing` surface with Python; the old `MockResponses` / `MockCompletions` wrappers were removed | Accepted |
| D-020 | `ModelStep.responder` / `stream_events` / `retry_advice` | Not ported; steps are a plain output/error pair | Advanced streaming and retry features are out of scope | Later phase |
| D-021 | `ModelCall` recorded fields | Records `streamed` and `output_schema_name`, but not `output_schema`, `handoffs` or `prompt` objects | `ModelRequest` does not carry handoffs; Python exposes the live objects | Later phase |
| D-018 | `function_tool` variadic / `Annotated` parameters | Not supported; the macro accepts plain owned-typed identifiers and an optional leading context parameter | Rust signatures cannot express `*args` / `**kwargs` with types | Accepted |
| D-023 | Chat Completions synthesized response id | The synthesized `response` object and `ModelResponse.response_id` carry the provider's `id` (e.g. `chatcmpl-…`) when the gateway sends one; `FAKE_RESPONSES_ID` only when it does not | Python always uses `FAKE_RESPONSES_ID` for Chat Completions. The id is only fed back as `previous_response_id`, which the Chat Completions adapter ignores | Accepted |

## Newly recorded gaps (audit 2026-10)

| ID | Standard location | Rust behavior | Plan |
|----|-------------------|---------------|------|
| D-026 | `RunConfig` | Ported: `handoff_input_filter` (D-003), `tool_not_found_behavior` (B10), `model`, `model_provider`, `model_settings`, `input_guardrails`, `output_guardrails`, `tracing_disabled`, `workflow_name`, `trace_id`, `group_id`, `trace_metadata`, `trace_include_sensitive_data`. Missing: `nest_handoff_history`, `handoff_history_mapper`, `session_input_callback`, `session_settings` (the session itself is `RunOptions.session`, D-027), `tool_error_formatter`, `reasoning_item_id_policy`, `tool_execution`, `tool_name_collision_policy`, `output_guardrail_blocked_message`, `tracing` (`TracingConfig`), `sandbox` | Prioritize `tool_error_formatter` |
| D-027 | `Runner.run(session=…)` / `agents.memory` | `Session` trait + `InMemorySession`, passed as `RunOptions.session`. History is prepended to a fresh run's input; on completion the run's input and model-visible new items are appended (also for `run_streamed`). Differences: items are saved once at the end, not per turn, so a failed run or a run paused for approval saves nothing until it is resumed with the same session; no `session_input_callback`, `SessionSettings` / `limit` on the run, compaction, SQLite / OpenAI-conversations backends or `OpenAIResponsesCompactionSession`; the `RunResult.input` stays the caller's input without history | Backends and callback later |
| D-028 | `Agent` fields | Missing: `prompt` (`Prompt` / dynamic prompt), `mcp_servers` / `mcp_config`, async `tool_use_behavior` functions (`ToolsToFinalOutputFunction`; the sync `ToolUseBehavior::Custom` is ported and sees tool name / call id / output only, not the run item), `Agent.clone(**overrides)` (use struct `Clone` + builder), `get_all_tools` | Async custom behavior can follow; the rest follow MCP / prompt phases |
| D-029 | `FunctionTool` / `tool_guardrails.py` | Only `FunctionTool` exists (no hosted, computer, shell, apply-patch, custom or MCP tools). `is_enabled` is `ToolEnabled` (fixed bool or an async closure over `(RunContextWrapper, Arc<Agent>)`; re-evaluated every turn). No `timeout_seconds` / `timeout_behavior`, no tool input/output guardrails, no `defer_loading` / namespaces | Rest as listed |
| D-030 | `Handoff` | `Agent.handoffs` takes `Handoff` only (Python also accepts a bare `Agent`). `is_enabled` is supported like tools; no `on_invoke_handoff` override (D-003 covers `input_filter` / `on_handoff`). The target is stored as `Arc<Agent>` by value, so a cyclic graph (A → B → A) cannot be built, while Python allows it by reference | Cycle support needs `Weak` / name-based resolution; consider alongside `input_filter` |
| D-031 | `RunResultStreaming.cancel(mode)` | `cancel(CancelMode::Immediate / AfterTurn)` and `is_cancelled()`. Immediate aborts the background task and closes the queue; AfterTurn stops before the next turn. A cancelled run has no `final_output`. Child tasks already spawned (parallel input guardrails, raw-event forwarder) are not aborted by Immediate. Dropping the stream does **not** cancel the run (Python behaves the same: the task keeps running). No back-pressure control: events are buffered in a 64-slot channel | Accepted |
| D-032 | Tool-name collisions | Handoff tool names and function tool names are not checked for collisions (`tool_name_collision_policy`); the first match wins silently, handoffs checked first | Add a validation pass before the model call |
| D-033 | Max-turns handling | `MaxTurnsExceeded` is always an error; there are no `run_error_handlers` (`run_error_handlers.py`) to turn it into a final output | Later phase |

Verified to **match** Python in this audit: `AgentHooks.on_handoff(context, agent=new, source=old)`
is invoked on the *source* agent's hooks; output guardrails combine `RunConfig` + agent guardrails
and trip as `OutputGuardrailTripwireTriggered`; `tool_choice` reset; `HandoffCallItem` /
`HandoffOutputItem` payload (`{"assistant": name}`); input guardrails run only on the first agent.

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
`[]`. The per-event status per emitter is tabulated
in [COMPAT.md](./COMPAT.md#streaming-wire-events).

## How to add an entry

1. Assign the next `D-NNN` ID.
2. Cite the Python path (file + symbol) under `vendor/openai-agents-python` @ `v0.23.1`.
3. State the Rust behavior and whether callers can observe the difference.
4. Mark the align plan: `Document only` / `Accepted` / `Gap` / `Later phase` / `Bug (fix)`.
