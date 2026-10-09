# Compatibility Matrix

Standard reference: `vendor/openai-agents-python` @ **v0.23.1** (`openai-agents==0.23.1`).
Re-sync the vendored tree with `bash scripts/sync_vendor.sh`.

Scope: provider-neutral. The SDK targets any OpenAI Responses / Chat Completions compatible endpoint, so OpenAI-hosted features are not planned (see [DEVIATIONS.md#scope](./DEVIATIONS.md#scope)); they appear below as **X**.

Legend: **X** = out of scope (OpenAI-hosted) · **S** = supported · **P** = partial · **N** = not supported · **D** = intentional deviation (see [DEVIATIONS.md](./DEVIATIONS.md))

## Core

| Capability | Status | Notes |
|------------|--------|-------|
| `Agent` (name, instructions, tools, model, model_settings, tool_use_behavior, reset_tool_choice) | S | `instructions` may be static or a dynamic closure of `(context, agent)` |
| `Agent.model_name` + provider resolution | S | Resolved via `RunConfig.model_provider` (D-013) |
| `Agent.output_type` / `AgentOutputSchema` | S | `schemars` + `ensure_strict_json_schema`; non-object types wrapped under `response` |
| `Agent.clone` | N | Use `Clone` on the struct (Rust idiom) |
| `Runner::run` / `run_blocking` / `run_state` | S | D-001 naming for the sync entry |
| `RunResultStreaming.cancel(mode)` | S | `CancelMode::Immediate` / `AfterTurn` (D-031) |
| `RunOptions.max_turns = None` (unbounded run) | S | `None` disables the limit; `RunOptions::default()` keeps `Some(10)` (D-042) |
| `Agent.as_tool` keywords beyond `name` / `description` / `needs_approval` / `max_turns` | N | No `run_config` / `hooks` / `session` / `custom_output_extractor` / `on_stream` / structured `parameters` (D-043) |
| Tool output parts (`ToolOutputText` / `ToolOutputImage` / `ToolOutputFileContent`) | N | A tool result is one JSON value rendered as text (D-044) |
| `RunOptions.auto_previous_response_id` | N | No server conversation tracker (D-045) |
| `error_handlers={"max_turns": ...}` | S | `RunOptions.error_handlers` (D-033); other kinds N |
| `Runner::run_streamed` / `RunResultStreaming` | S | Item + agent events plus raw Responses wire events on both OpenAI APIs (D-011) |
| `FunctionTool` | S | Manual schema or `#[function_tool]` |
| `@function_tool` / proc-macro | S | Full `schemars` schema; `Vec<T>`, nested structs, enums and `Option<T>` supported |
| Tool context injection | S | Optional first `ToolContext` / `RunContextWrapper` parameter |
| Parallel function tools | S | `join_all`; results ordered by model tool-call order |
| `ToolExecutionConfig.max_function_tool_concurrency` | S | `RunConfig.tool_execution`; slots free on completion, results keep call order (D-029) |
| `FunctionTool` `timeout_seconds` / `timeout_behavior` / `timeout_error_function` | S | `ToolTimeoutError` for `RaiseException`; parity-tested message (D-029) |
| `failure_error_function` | P | Sync formatter or `raise_on_error`; default message sent to the model (B7) |
| `tool_use_behavior` callable | P | Sync `ToolUseBehavior::Custom` |
| `RunConfig.tool_not_found_behavior` | S | B10 |
| `RunConfig.tool_name_collision_policy` | S | D-032 |
| `RunConfig.reasoning_item_id_policy` | S | Checked against Python (D-035) |
| `run_error_handlers` (`max_turns`, `model_refusal`, `invalid_final_output`), `ModelRefusalError` | S | D-033 |
| `RunConfig.output_guardrail_blocked_message` | S | Tool-derived final outputs only; checked against Python (D-036) |
| `RunConfig.tool_error_formatter` | S | Rejection and not-found messages (D-009) |
| `RunConfig.call_model_input_filter` | S | D-M |

## Model layer

| Capability | Status | Notes |
|------------|--------|-------|
| `Model` trait (`get_response` / `stream_response`) | S | `stream_response` emits Responses wire events (D-011) |
| OpenAI Responses API model | S | via `async-openai` config + raw HTTP; real SSE, events forwarded verbatim (D-011) |
| OpenAI Chat Completions model | S | real SSE; chunks are synthesized into Responses wire events (D-011); converters checked against Python, tolerant of non-conforming servers (D-040) |
| Third-party servers (base URL + key) | S | `stream_options` opt-in, DeepSeek reasoning replay, generated tool call ids, `finish_reason` handling (D-040) |
| Default API = Responses | S | `set_default_openai_api` |
| `ModelProvider` / `MultiProvider` | S | `prefix/model` routing, default prefix `openai` |
| `OpenAIProvider` | S | Reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` (the key is optional with a base URL); caches per name; API per provider, global or by host (D-I) |
| `CompatibleProvider` | S | Provider-neutral entry: base URL, optional key, Chat Completions by default (D-013) |
| Model retry (`ModelSettings.retry`, `retry_policies`) | S | Checked against Python scenario by scenario; typed `ModelError::Status` / `Connection` (D-034) |
| Per-attempt `ModelSettings.timeout` | S | `ModelError::Timeout`; retryable (D-034) |
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
| `response.refusal.delta` / `.done` | F | S (from `delta.refusal`) | N |
| `response.output_text.annotation.added` | F | N | N |
| MCP / hosted-tool events | F | N | N (hosted tools not planned, D-006) |
| `sequence_number` | forwarded as-is | synthesized, 0-based | synthesized, 0-based |
| `logprobs` payloads | forwarded as-is | `[]` | `[]` |
| `usage` token details | forwarded as-is | cached / reasoning tokens (D-016) | totals only |

No emitter synthesizes `response.in_progress` / `response.queued`; the Responses adapter forwards
them like any other provider event, and it reads the response id out of them when the terminal
`response.completed` never arrives.

## Model settings

`ModelSettings` covers 22 fields: `temperature`, `top_p`, `frequency_penalty`,
`presence_penalty`, `tool_choice` (typed enum), `parallel_tool_calls`, `truncation`, `max_tokens`,
`reasoning`, `verbosity`, `metadata`, `store`, `top_logprobs`, `include_usage`,
`response_include`, `extra_body`, `extra_headers`, `extra_query`, `preserve_raw_usage`,
`extra_args`, `timeout`, `retry`. `resolve()` overlays non-`None` values and merges dictionaries,
matching Python; `retry` is merged field by field (D-034).

Not ported: `prompt_cache_retention`, `context_management`, `prompt_cache_options` — three
OpenAI-hosted server features (D-017); reach them with `extra_body` / `extra_args` when a server
does support them.

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
| Hosted-MCP / tool-search items | X |

## Handoffs

| Capability | Status | Notes |
|------------|--------|-------|
| Basic handoff (tool + agent switch) | S | Sibling tools run first; extra handoffs ignored with Python's message (B8) |
| `HandoffCallItem` / `HandoffOutputItem` | S | |
| `handoff_span` | S | |
| `input_filter` / `RunConfig.handoff_input_filter` | S | D-003 |
| `nest_handoff_history`, `handoff_history_mapper` | S | Opt-in, parity-tested against Python (D-003) |
| `on_handoff` callback / `input_type` | P | Schema supplied explicitly, arguments parsed but not schema-validated (D-003) |

## HITL

| Capability | Status | Notes |
|------------|--------|-------|
| `needs_approval` (fixed or dynamic) | S | |
| `interruptions`, `RunState` approve/reject | S | |
| Sticky `always_approve` / `always_reject` | S | |
| `to_json` / `from_json` | S | Schema `openai-agents-rs/3` (D-012); `openai-agents-rs/2`, `openai-agents-rust/1` and `openai-agents-rust/2` payloads still load |
| `Agent.as_tool` nested approvals | S | Bubbles to the outer `RunState` |
| `Agent.as_tool` config beyond `needs_approval` / `max_turns` | N | No `run_config`, `hooks`, `session`, `failure_error_function`, `is_enabled`, `parameters` / `input_builder` (D-043) |
| Custom output extractor / `on_stream` | N | D-043 |

## MCP

| Capability | Status | Notes |
|------------|--------|-------|
| `MCPServerStdio` | S | `McpClient::stdio`; restricted environment, child killed on drop (D-041) |
| `MCPServerStreamableHttp` | S | `McpClient::streamable_http`; SSE and JSON replies, session id (D-041) |
| `MCPServerSse` (legacy SSE transport) | N | Superseded by streamable HTTP |
| `Agent.mcp_servers`, `Agent.mcp_config.convert_schemas_to_strict` | S | Tools listed every turn |
| `tool_filter` (static, dynamic), `cache_tools_list`, `require_approval`, `use_structured_content` | S | |
| `include_server_in_tool_names`, `tool_meta_resolver`, retries, prompts, resources, `mcp_tools_span` | N | D-041 |
| `HostedMCPTool` | X | OpenAI-hosted |

## Sessions

| Capability | Status | Notes |
|------------|--------|-------|
| `Session` trait, `InMemorySession` | S | `RunOptions.session` (D-027) |
| `RunConfig.session_input_callback`, `SessionSettings.limit` | S | Checked against Python (D-027) |
| `SqliteSession` (feature `sqlite`) | S | Python-compatible schema; files interchangeable (D-038) |
| OpenAI-conversations session, `responses.compact` compaction | X | OpenAI-hosted |
| `ToolOutputTrimmer` | P | Checked against Python; string outputs only (D-039) |
| `ContextWindowTrimmer`, `chain_input_filters` | S | New, provider-neutral (D-039) |
| `CompactingSession`, `Summarizer`, `ModelSummarizer`, `Session::replace_items` | S | New, replaces the OpenAI-only compaction (D-039) |

## Context, guardrails, hooks

| Capability | Status | Notes |
|------------|--------|-------|
| `RunContextWrapper` (type-erased) | S | `context::<T>()` / `try_context::<T>()`; D-014 |
| `RunOptions.context` | S | |
| Input / output guardrails | S | `run_in_parallel` honoured; parallel tripwire cancels the model call (B9) |
| Guardrail results on `RunResult` | S | |
| `RunHooks` / `AgentHooks` | S | All methods default to no-op |
| Tool input/output guardrails | S | Parity-checked; not persisted in `RunState` (D-029) |
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
| `task_span` / `turn_span`, `RunConfig.tracing` (`TracingConfig`) | S | Span tree checked against Python (D-037); no `api_key` (cloud export is out of scope) |
| `Span.started_at` / `ended_at` / `SpanError` | S | Set on drop; a failing tool and an exhausted turn budget record an error |
| speech / transcription spans | N | |
| OpenAI cloud export | N | D-004, not planned |
| `SpanData` payloads (`input` / `output` / `tools`) | N | Only names and usage are recorded |

## Testing

The `testing` module mirrors `agents.testing`; HTTP-level assertions live in the crate's own
tests, which drive `wiremock` directly rather than through a shipped mock layer.

| Capability | Status | Notes |
|------------|--------|-------|
| `ScriptedModel` (`new`, `enqueue`, `extend`, `calls`, `assert_complete`) | S | |
| `ModelStep` (`output`, `raise_error`, `raise_model_error`, `with_retry_advice`) | S | `responder` / `stream_events` not ported (D-020) |
| `ModelCall` (incl. `streamed`, `output_schema_name`) | S | `output_schema` / `handoffs` / `prompt` recorded only by name, if at all |
| `remaining_steps`, `first_call`, `last_call` | S | |
| `set_default_usage` | S | |
| `ModelScriptError`, `InvalidModelStep`, `UnexpectedModelCall`, `UnconsumedModelSteps` | S | |
| `assistant_message`, `function_call` | S | |
| `ScriptedModel` streaming | S | Replays a step as standard Responses wire events (D-011) |
| `ModelStepSpec` (dict form) | N | Rust steps are typed; no dict form |
| `ScriptedSandboxSession` | N | |

## Verification layers

0. `cargo fmt --all -- --check`, `cargo clippy --all-features --all-targets -- -D warnings`,
   `RUSTDOCFLAGS=-D warnings cargo doc --no-deps --all-features`, `scripts/sync_vendor.sh --check`
1. `cargo test --no-default-features` — ScriptedModel behavior
2. `cargo test --all-features` — OpenAI HTTP contracts via wiremock, SQLite sessions
2.5 `cargo test --test property_core` — invariants over generated inputs (proptest)
3. `.venv/bin/python scripts/run_parity.py --check` then `cargo test --test parity_scenarios`;
   `cargo test --test mcp_interop` and `--test chat_convert_parity` need the venv
