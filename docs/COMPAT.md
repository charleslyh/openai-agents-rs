# Compatibility Matrix

Standard reference: `vendor/openai-agents-python` @ **v0.23.1** (`openai-agents==0.23.1`).
Re-sync the vendored tree with `bash scripts/sync_vendor.sh`.

Legend: **S** = supported · **P** = partial · **N** = not supported · **D** = intentional deviation (see [DEVIATIONS.md](./DEVIATIONS.md))

## Core

| Capability | Status | Notes |
|------------|--------|-------|
| `Agent` (name, instructions, tools, model, model_settings, tool_use_behavior, reset_tool_choice) | S | `instructions` may be static or a dynamic closure of `(context, agent)` |
| `Agent.model_name` + provider resolution | S | Resolved via `RunConfig.model_provider` (D-013) |
| `Agent.output_type` / `AgentOutputSchema` | S | `schemars` + `ensure_strict_json_schema`; non-object types wrapped under `response` |
| `Agent.clone` | N | Use `Clone` on the struct (Rust idiom) |
| `Runner::run` / `run_blocking` / `run_state` | S | D-001 naming for the sync entry |
| `RunResultStreaming.cancel(mode)` | S | `CancelMode::Immediate` / `AfterTurn` (D-031) |
| `Runner::run_streamed` / `RunResultStreaming` | S | Item + agent events plus raw Responses wire events on both OpenAI APIs (D-011) |
| `FunctionTool` | S | Manual schema or `#[function_tool]` |
| `@function_tool` / proc-macro | S | Full `schemars` schema; `Vec<T>`, nested structs, enums and `Option<T>` supported |
| Tool context injection | S | Optional first `ToolContext` / `RunContextWrapper` parameter |
| Parallel function tools | S | `join_all`; results ordered by model tool-call order |
| `ToolExecutionConfig.max_function_tool_concurrency` | N | All tools in a batch start concurrently |
| `failure_error_function` | P | Sync formatter or `raise_on_error`; default message sent to the model (B7) |
| `tool_use_behavior` callable | P | Sync `ToolUseBehavior::Custom` |
| `RunConfig.tool_not_found_behavior` | S | B10 |
| `RunConfig.call_model_input_filter` | S | D-M |

## Model layer

| Capability | Status | Notes |
|------------|--------|-------|
| `Model` trait (`get_response` / `stream_response`) | S | `stream_response` emits Responses wire events (D-011) |
| OpenAI Responses API model | S | via `async-openai` config + raw HTTP; real SSE, events forwarded verbatim (D-011) |
| OpenAI Chat Completions model | S | real SSE; chunks are synthesized into Responses wire events (D-011) |
| Default API = Responses | S | `set_default_openai_api` |
| `ModelProvider` / `MultiProvider` | S | `prefix/model` routing, default prefix `openai` |
| `OpenAIProvider` | S | Reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`; caches per name |
| Model retry (`ModelRetrySettings`) | N | |
| Litellm / other providers | N | Register a custom `ModelProvider` |

## Streaming wire events

`StreamEvent::RawResponse.data` is one Responses API wire event on every backend
(see [DEVIATIONS.md D-011](./DEVIATIONS.md#streaming-wire-contract)). **F** = forwarded verbatim
from the provider, **S** = synthesized by the adapter, **N** = not emitted.

| Event | Responses | Chat Completions | `ScriptedModel` |
|-------|-----------|------------------|-----------------|
| `response.created` | F | S | S |
| `response.output_item.added` / `.done` | F | S | S |
| `response.content_part.added` / `.done` | F | S | S |
| `response.output_text.delta` | F | S | S |
| `response.output_text.done` | F | N (Python's chat handler does not emit it) | S |
| `response.reasoning_summary_part.added` / `.done` | F | S (from `reasoning_content`) | S |
| `response.reasoning_summary_text.delta` / `.done` | F | S (from `reasoning_content`) | S |
| `response.reasoning_text.delta` / `.done` | F | S (from `reasoning`) | S |
| `response.function_call_arguments.delta` | F | S | S |
| `response.function_call_arguments.done` | F | N | S |
| `response.completed` | F | S | S |
| `response.refusal.*` | F | N | N |
| `response.output_text.annotation.added` | F | N | N |
| MCP / hosted-tool events | F | N | N |
| `sequence_number` | forwarded as-is | synthesized, 0-based | synthesized, 0-based |
| `logprobs` payloads | forwarded as-is | `[]` | `[]` |
| `usage` token details | forwarded as-is | totals only (D-016) | totals only (D-016) |

No emitter synthesizes `response.in_progress` / `response.queued`; the Responses adapter forwards
them like any other provider event, and it reads the response id out of them when the terminal
`response.completed` never arrives.

## Model settings

`ModelSettings` covers 19 fields: `temperature`, `top_p`, `frequency_penalty`,
`presence_penalty`, `tool_choice` (typed enum), `parallel_tool_calls`, `truncation`, `max_tokens`,
`reasoning`, `verbosity`, `metadata`, `store`, `top_logprobs`, `include_usage`,
`response_include`, `extra_body`, `extra_headers`, `extra_args`, `timeout`. `resolve()` overlays
non-`None` values and merges dictionaries, matching Python.

Not ported: `extra_query`, `prompt_cache_retention`, `context_management`, `prompt_cache_options`,
`preserve_raw_usage`.

Request-body precedence, highest first (matching Python):

1. `extra_body` — Python hands it to the OpenAI SDK as a nested argument that is merged over the
   body, so it overrides both mapped settings and `extra_args`.
2. mapped settings — `temperature`, `max_output_tokens`, `metadata`, `include`, …
3. `extra_args` — fills only keys nothing else set; a key that collides with a mapped setting or
   a request field such as `model` is an error (D-024), not a silent drop. Python additionally
   requires these keys to be *typed* SDK parameters; arbitrary fields belong in `extra_body`
   (D-025).

`resolve()` replaces every mapping when the override is not `None`, except `extra_args`, whose
dictionaries are merged (`model_settings.py:273`).

## Run items

| Item | Status |
|------|--------|
| `MessageOutputItem`, `ToolCallItem`, `ToolCallOutputItem`, `ToolApprovalItem` | S |
| `HandoffCallItem`, `HandoffOutputItem` | S |
| `ReasoningItem` | S |
| `CompactionItem` | N |
| MCP / tool-search items | N |

## Handoffs

| Capability | Status | Notes |
|------------|--------|-------|
| Basic handoff (tool + agent switch) | S | Sibling tools run first; extra handoffs ignored with Python's message (B8) |
| `HandoffCallItem` / `HandoffOutputItem` | S | |
| `handoff_span` | S | |
| `input_filter` / `RunConfig.handoff_input_filter` | S | D-003 |
| `nest_handoff_history` | N | D-003 |
| `on_handoff` callback / `input_type` | P | Schema supplied explicitly, arguments parsed but not schema-validated (D-003) |

## HITL

| Capability | Status | Notes |
|------------|--------|-------|
| `needs_approval` (fixed or dynamic) | S | |
| `interruptions`, `RunState` approve/reject | S | |
| Sticky `always_approve` / `always_reject` | S | |
| `to_json` / `from_json` | S | Schema `openai-agents-rs/2` (D-012); older `openai-agents-rust/1` and `openai-agents-rust/2` payloads still load |
| `Agent.as_tool` nested approvals | S | Bubbles to the outer `RunState` |
| Custom output extractor / `on_stream` | N | |

## Context, guardrails, hooks

| Capability | Status | Notes |
|------------|--------|-------|
| `RunContextWrapper` (type-erased) | S | `context::<T>()` / `try_context::<T>()`; D-014 |
| `RunOptions.context` | S | |
| Input / output guardrails | S | `run_in_parallel` honoured; parallel tripwire cancels the model call (B9) |
| Guardrail results on `RunResult` | S | |
| `RunHooks` / `AgentHooks` | S | All methods default to no-op |
| Tool input/output guardrails | N | |
| Dynamic `is_enabled` (tools, handoffs) / `needs_approval` closures | S | `ToolEnabled::dynamic`; `NeedsApproval::Dynamic` |

## Tracing

| Capability | Status | Notes |
|------------|--------|-------|
| `trace`, `agent_span`, `function_span`, `generation_span`, `custom_span` | S | |
| `handoff_span`, `response_span`, `guardrail_span` | S | |
| Per-run `tracing_disabled` | S | Takes precedence over the global switch |
| `trace_id` / `group_id` / `trace_metadata` injection | S | |
| `trace_include_sensitive_data` | S | Maps to `ModelTracing::EnabledWithoutData` |
| Processor start events | S | `InMemoryProcessor::started_spans` / `started_traces` |
| `flush_traces` | S | Calls `TracingProcessor::force_flush` |
| `task_span` / `turn_span` / speech / transcription spans | N | |
| OpenAI cloud export | N | D-004 |

## Testing

The `testing` module mirrors `agents.testing`; HTTP-level assertions live in the crate's own
tests, which drive `wiremock` directly rather than through a shipped mock layer.

| Capability | Status | Notes |
|------------|--------|-------|
| `ScriptedModel` (`new`, `enqueue`, `extend`, `calls`, `assert_complete`) | S | |
| `ModelStep` (`output`, `raise_error`) | S | `responder` / `stream_events` / `retry_advice` not ported |
| `ModelCall` (incl. `streamed`, `output_schema_name`) | S | `output_schema` / `handoffs` / `prompt` recorded only by name, if at all |
| `remaining_steps`, `first_call`, `last_call` | S | |
| `set_default_usage` | S | |
| `ModelScriptError`, `InvalidModelStep`, `UnexpectedModelCall`, `UnconsumedModelSteps` | S | |
| `assistant_message`, `function_call` | S | |
| `ScriptedModel` streaming | S | Replays a step as standard Responses wire events (D-011) |
| `ModelStepSpec` (dict form) | N | Rust steps are typed; no dict form |
| `ScriptedSandboxSession` | N | |

## Verification layers

1. `cargo test --no-default-features` — ScriptedModel behavior
2. `cargo test` — OpenAI HTTP contracts via wiremock (requires the `openai` feature)
3. `.venv/bin/python scripts/run_parity.py --write-golden` then `cargo test --test parity_scenarios`
