# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). 

## [Unreleased]

Collects the changes for 0.2.0. Breaking changes are marked **breaking**.

### Breaking

- **breaking**: `opai-agents-sqlite` is merged into `opai-agents-sessions` behind its `sqlite`
  feature. `opai_agents::SqliteSession` and the facade feature `sqlite` are unchanged; code that
  named `opai_agents_sqlite::SqliteSession` should use `opai_agents_sessions::SqliteSession` with
  `features = ["sqlite"]`. The published `opai-agents-sqlite` 0.1.0 stays on crates.io and gets no
  new versions.
- **breaking**: `opai-agents-scaffold` lives in `tools/scaffold` and is no longer published. Install
  it with `cargo install --git https://github.com/charleslyh/openai-agents-rs opai-agents-scaffold`.
  The published SDK is now twelve crates.
- **breaking**: public enums that may grow are `#[non_exhaustive]`, and `ToolContext` is
  `#[non_exhaustive]` (build one with `ToolContext::new`).
- **breaking**: `opai_agents_core::session_contract` is behind the new `session-contract` feature
  (off by default). A backend that runs it enables the feature in its dev dependencies.
- **breaking**: large error variants are boxed (`AgentsError` 264 to 48 bytes, `ModelError` 128 to
  32), and the `result_large_err` lint allow is gone.

### Changed

- Every crate shares one version and one copy of each shared third-party dependency, declared in the
  root `Cargo.toml` (`[workspace.package]`, `[workspace.dependencies]`). `scripts/check_docs_drift.py`
  fails when the versions or the install snippets disagree. The workspace is now at `0.2.0`.

### Added

- MCP protocol 2026-07-28 (stateless): the client tries `server/discover` first and falls back to the
  `initialize` handshake for older servers. Requests carry the version, capabilities and client
  info in `_meta`; HTTP requests carry `Mcp-Method` and `Mcp-Name`; a server's `ttlMs` is honoured
  for the tool list. A reply that asks for more input (`input_required`) is reported as an error.
  Not implemented: `subscriptions/listen`, the tasks extension, `x-mcp-header`, `cacheScope`.
- Cooperative cancellation: `RunOptions::cancellation_token`, `ToolContext::cancellation()` and
  `is_cancelled()`, `RunResult::is_cancelled`. Tools already running finish; calls that had not
  started get a cancelled output. `teams` cancels a teammate through the same token.
- `FunctionTool::try_typed` and `ToolSpec::try_typed`, which return an error instead of panicking.
- `ScriptedModel` steps that panic or never return, for testing failure paths.
- `redis-tls` feature for `rediss://`; `scripts/publish.py` and `docs/RELEASING.md` for a paced,
  resumable crates.io release; `scripts/check_docs_drift.py`.

### Fixed

- A panic in a streamed run's background task is reported as an error instead of a normal end.
- A panic in a tool, hook or guardrail becomes a tool error instead of unwinding through the run.

## [0.1.0] - 2026-10-21

First release on crates.io: 14 crates under the `opai-agents-*` names (facade: `opai-agents`).

### Changed

- Minimum Rust version lowered from 1.99 to **1.89**. The floor is the dependencies' (`uuid` needs
  1.89), not the SDK's; a new `msrv` CI job builds the published crates, tests included, with 1.89.
  The exception is the `postgres` feature, which needs 1.94 because `sqlx` 0.9 declares it.
  The MCP tests no longer use `AtomicUsize::try_update` (unstable before 1.95).
- The scaffold binary is now `cargo-opai`, so the command is `cargo opai new` (it was
  `cargo agents new`). `cargo-agents` is a different tool on crates.io; sharing its binary name would
  make `cargo install` refuse or overwrite it. `opai-agents-scaffold` is now publishable.

### Added

- New crate `opai-agents-teams` (X-025), behind the facade feature `teams`: a lead creates
  long-lived teammates from registered roles; they run in parallel, message each other, share a task
  list with atomic claims and dependencies, and stay inside a budget. Includes cooperative shutdown,
  panic isolation, approval and plan-approval bubbling to the lead, `TeamHooks` quality gates and
  team-rooted tracing. In-process only. See `docs/AGENT-TEAMS.md` and example `35_teams`.
- Live console: `ConsoleLink` sends spans and (opt-in) the conversation of each run to the new
  `opai-console` service in `tools/console`, which shows a waterfall next to prompts, replies and
  tool calls in the browser. Loopback only, token-protected. See example `31_console`.
- `ConversationHub` and `ConversationRecorder` in `opai-agents-observability` record a run's
  conversation on a channel separate from spans, with a `metadata_only()` mode.

### Changed

- Examples `cancel`, `settings` and `reasoning` are now numbered 32, 33 and 34.

### Fixed

- `OtelSink` keeps a span open from its start to its end, so child spans are real children in the
  OpenTelemetry trace instead of separate roots.
- Chat Completions and MCP streams no longer corrupt multi-byte characters (CJK text, emoji) that
  a network chunk boundary splits in two.
- Chat Completions streams that announce a message before a tool call no longer give both the same
  `output_index`.
- A Chat Completions request whose output JSON schema cannot be generated now fails instead of
  silently dropping `response_format`.
- A stdio MCP server that stops reading its input can no longer block `close()` forever.
- `CompactingSession` gives up on a summarizer that does not answer (`summarize_timeout`, default
  60 s) and pauses automatic compaction after a failure (`failure_cooldown`, default 60 s) instead
  of retrying at the end of every run.
- `InMemoryProcessor`, which is installed by default, is bounded (10 000 entries per collection,
  oldest dropped first) so a long-running process no longer grows without limit. Configure it with
  `InMemoryProcessor::with_capacity`.
- A poisoned lock no longer cascades into panics across the SDK.

### Changed

- **Security:** the MCP streamable-HTTP transport follows redirects only within the original origin
  (at most 5), so configured headers and request bodies never reach another host. It also sets a
  10 s connect timeout and a timeout on notifications and session close.
- `#[function_tool]` rejects unknown options at compile time. Previously a misspelt option compiled
  and was ignored.
- Streams that end without `response.completed` (Responses) or a `finish_reason` (Chat Completions)
  now log a warning, since the reply may be truncated.
- The agent loop is split into `run/turn_loop.rs`; behaviour is unchanged.
- **Breaking:** `RunConfig`'s flat fields are grouped into `model` / `tracing` / `session` /
  `handoff` / `tools` (`ModelOptions`, `TracingOptions`, ...): `config.model_settings` becomes
  `config.model.settings`, `config.workflow_name` becomes `config.tracing.workflow_name`, etc.
  `RunOptions` is `#[non_exhaustive]`; build it with `RunOptions::new()` and the chain methods
  (`.max_turns(..)`, `.temperature(..)`, `.reasoning(..)`, `.session(..)`, `.tune(..)`).
- **Breaking:** `ToolResult` is renamed `ToolOutput`; its `interruptions` / `nested_state` fields
  are no longer public. Use `ToolOutput::output` / `::interrupted`, `is_interrupted()`, `value()`.
- **Breaking:** `ModelSettings::reasoning` is a typed `Reasoning` instead of a JSON value:
  `Reasoning::effort(ReasoningEffort::High).with_summary(ReasoningSummary::Auto)`. Convert an
  existing value with `Reasoning::try_from(json!({...}))`.
- Chat Completions now sends `reasoning.effort` as `reasoning_effort`, as Python does. A request
  that also put `reasoning_effort` in `extra_args` now fails with the duplicate-key error.
- `default_provider()` returns an owned `Arc<dyn ModelProvider>`.
- **Security:** a resolution given through `RunState` (`approve`, `reject`, `resolve`,
  `provide_output`) applies only to a call with the same agent, tool and arguments as the paused
  call it answered; a call whose arguments changed pauses again instead of running.
- A paused `RunState` records a fingerprint of the agent graph; resuming it with agents whose
  names, tools or handoffs changed fails with a `UserError`. Call
  `RunState::allow_agent_graph_change()` to resume anyway.
- New crate `opai-agents-providers`: presets for DeepSeek, Gemini, Groq, OpenRouter, Anthropic and
  Ollama, `ProviderRegistry::from_env()` discovery, `prefix/model` routing with inference,
  a capability matrix and `UnsupportedCapability` (X-013).
- Requests a provider cannot honour are refused before they are sent: reasoning, a JSON schema,
  tool calls, parallel calls, `store`, images and PDFs, reported as `ModelError::Unsupported`
  naming the provider and the capability.
- A small model catalogue warns when a run names a retired model (and what to use instead), a
  superseded one, or an id that looks like a typo of a known one (X-014).
- `opai_agents::run(instructions, input)`: one line, provider and model taken from the
  environment. `AGENTS_OFFLINE=1` answers locally, so a project can be tried before any key
  exists. New `providers` feature on the facade (off by default) (X-015).
- `RunOptions::provider(..)` sets the provider that resolves a run's model names.
- `RunStateStore` (save / load / delete with a TTL) plus `RunStateMigration` and shared schema
  checks: the contract the file, Redis and Postgres backends implement (X-016).
- `opai_agents_core::session_contract`: the behaviour every `Session` backend must have, as a
  suite any backend can run in its own tests. `InMemorySession` and `SqliteSession` pass it
  (X-017).
- New crate `opai-agents-sessions` with `FileSession`: a conversation in a JSONL file, locked
  across processes, rewritten atomically. Session ids are validated (no path traversal, no
  symlinks), and it passes the contract above.
- `FileRunStateStore`: a paused run per file, signed with HMAC-SHA256 when `AGENTS_STATE_KEY` is
  set, with a TTL and a migration hook. A state edited outside the process fails to load instead
  of resuming.
- New crate `opai-agents-observability` with `BatchedProcessor`, a `TracingProcessor` that turns
  spans into JSON in the callback and hands them to a background thread through a bounded queue,
  so a slow or broken exporter costs the run nothing. A full queue drops the event and counts it
  (`dropped()`); `force_flush` waits at most 2 s (X-018).
- `ConsoleSink` and `JsonlSink`: where those events go, one JSON object per line.
- `Metrics` / `MetricsSink` over the same stream: turns, tool calls, tool failure rate, input and
  output tokens, and cost from a `PriceTable`. A model with no price counts as zero, never as an
  estimate.
- `TracingSink`: each span end becomes a `tracing` event under the target `opai_agents` (`DEBUG`,
  or `WARN` when the span failed; starts are `TRACE`), so `RUST_LOG=opai_agents=debug` shows a
  run next to the application's own logs. Events, not `tracing` spans: the export runs on a writer
  thread with no span entered, so `span_id` / `parent_id` are fields.
- `OtelSink` (`otel` feature): spans as OpenTelemetry spans, in whatever pipeline the process has
  installed. Only the OpenTelemetry API is depended on, so the exporter — OTLP or anything else —
  stays the application's choice. The SDK's span and parent ids travel as attributes, because the
  API assigns ids itself.
- A hand-written `impl Tool` is now buildable outside the SDK: `ToolSpec::new` / `ToolSpec::typed`
  and the `ToolPolicy` builders (`needs_resolution`, `enabled`, `timeout`, `timeout_behavior`,
  `on_failure`, `timeout_message`), plus `NeedsResolution::dynamic`. See
  `examples/src/11_structured_tool.rs` (X-023).
- `#[derive(ToolMeta)]` + `TypedTool` + `FunctionTool::from_typed::<Args>()` (X-024): a tool declared
  by annotations on its argument struct. Field doc comments are the parameter descriptions, the
  struct's doc comment is the tool description, and `#[tool(name, timeout, requires(..),
  needs_approval, needs_approval_if = path, on_failure = "..{error}..")]` carries the rest; the body
  is `impl TypedTool` taking the parsed arguments and returning a typed result. The `structured_tool`
  example now uses it; `impl Tool` stays for a tool with state of its own.
- `CapabilityParts::deferred(description)` (X-021): a capability the model loads on demand. Until it
  calls `load_capability`, the prompt carries one catalog line and none of the capability's tools or
  instructions; loading is per agent and run, idempotent, and saved in a paused run's `RunState`
  (`loaded_capabilities`, absent in older states). Guardrails and model settings apply from the start.
- New crate `opai-agents-guardrails` (X-020): `MaxLength`, `PiiRedactor` (redacts by default;
  emails, Luhn-valid cards, SSNs, phone numbers), `PromptInjection` (a heuristic, documented as
  one), `BlockedKeywords` (whole-word, literal) and `SchemaValidator` (a subset of JSON Schema that
  refuses the rest). Each becomes an input or output guardrail with `.input()` / `.output()`; a bad
  configuration is a `BuildError` at construction; a failure never quotes the text. Facade feature
  `guardrails`.
- New crate `opai-agents-tools` (X-019): `calculator` (a parser, not `eval`), `clock` (one
  RFC 3339 shape), `file_read` / `file_list` (one directory, nothing outside it: `..`, symlinks
  that leave and absolute paths refused) and `web_fetch` (`webfetch` feature: private, loopback,
  link-local and reserved addresses refused at connect time, including the `9.`/`11.`/`21.`/`30.`
  blocks, also when written into the URL as a decimal, hex or octal number; cross-origin and
  `https`-to-`http` redirects refused, five hops at most; no system proxy; the 1 MiB response cap
  is applied while reading). Facade features `tools` and `webfetch`.
- Facade features `sessions` (`FileSession`, `FileRunStateStore`) and `observability`
  (`BatchedProcessor`, sinks, `Metrics`), both off by default, and two examples: `file_session` and
  `trace_export`.
- `RedisSession` and `PostgresSession` (`redis` / `postgres` features): the same conversation
  contract in a list or in rows, for a history more than one process shares, with `replace_items`
  atomic in both. Both backends run the shared contract suite, and both ship a `RunStateStore`.
  Values are bound in every statement; a table name is validated before it reaches SQL.
- Every preset has a contract test against a mock server: the protocol it speaks, the reasoning
  field that protocol uses, and a capability it lacks refusing the request instead of sending it.
- **Breaking:** guardrails return a `Verdict`: `Pass`, `Fail { reason, severity }` or
  `Transform(content)` (continue with rewritten content, e.g. redacted data).
  `GuardrailFunctionOutput::tripwire_triggered` is now a method; `pass` / `trip` keep working and
  `fail(reason, severity)` / `transform(content)` are new. A `High` or `Critical` failure cancels
  the guardrails still running in its group. Only a guardrail built with `run_in_parallel(false)`
  may transform the input.
- The run loop shares one `Arc<Agent>` per agent instead of deep-copying the agent on every turn,
  handoff and tool call (about 20% faster on `twenty_tools_ten_turns`).
- **Breaking:** `Agent::resolve_instructions`, `enabled_tools`, `all_function_tools` and
  `enabled_handoffs` take `self: &Arc<Agent>`, and `ToolEnabled::resolve` takes `&Arc<Agent>`.
  Wrap a bare agent with `Arc::new(agent)` to call them.
- **Breaking:** `ModelProvider::get_model` is `async` so a provider can await credentials or a
  lookup. Implement it with `#[async_trait]` (re-exported as `opai_agents::async_trait`) and
  `.await` direct calls.

### Added

- `#[function_tool(timeout = <seconds>)]`.
- `opai_agents_core::utf8_stream::Utf8Stream` for decoding chunked byte streams.
- A `twenty_tools_ten_turns` benchmark.
- `Extensions`: typed per-run values (`RunOptions::with(value)`, `ctx.get::<T>()`,
  `ctx.require::<T>()`).
- `Runtime`: owns the default model provider, the runtime behind blocking calls, the tracing
  switch and processors, and the handoff history markers; `Runtime::global()` keeps the
  process-wide default. `RunOptions::runtime(rt)` runs on a runtime of your own, so two runs on
  two runtimes share no state (X-022). The `set_trace_processors`, `set_tracing_disabled` and
  `set_conversation_history_wrappers` free functions act on `Runtime::global()`. Code that spawns
  its own tasks inside a run should wrap them in `tracing::with_run_tracing_disabled`, which now
  carries the runtime along. Trace processor callbacks no longer run with the processor lock held.
- The `Tool` trait (`FunctionTool::from_tool`), `FunctionTool::typed` for schema-from-struct tools,
  `Tool::requires` / `FunctionTool::require::<T>()` checked before the first turn, and
  `ToolMiddleware` / `ModelMiddleware`.
- A `:<effort>` model-name suffix (`"o4-mini:high"`) for names resolved through a provider.
- `Capability` / `CapabilityParts`: named bundles of tools, instructions, guardrails, default model
  settings and required extensions, mounted with `Agent::capability(..)`.
- `prelude` modules (`opai_agents::prelude`, `opai_agents_core::prelude`) with the types most
  programs need; the facade one also brings `#[function_tool]`.
- `async_trait` is re-exported at the crate root and in the prelude.
- `RunOptions` gains `.tracing(..)`, `.tools(..)` and `.session_opts(..)` so a run can be
  configured without touching nested structs, and `ToolExecutionConfig::new().concurrency(..)`.
- Examples typed their tool arguments with `FunctionTool::typed` instead of hand-written schemas,
  and two new ones join them: `capability` (a named bundle) and `verdict_transform` (redact and
  continue).
- `RunOptions::resolve_with(|call| async { .. })` resolves paused tool calls in-process and
  resumes until the run finishes; `ToolResolution::reject(..)` / `::output(..)` constructors.

### Removed

- Python-name aliases that duplicated the resolution API: `NeedsApproval`, `NeedsApprovalFn`,
  `with_needs_approval`, `with_needs_approval_fn`, `ToolApprovalItem`, `ApprovalDecision`,
  `ApprovalStore`. Use `NeedsResolution`, `with_needs_resolution`, `with_needs_resolution_fn`,
  `ToolInterruptionItem`, `ToolResolution` and `ResolutionStore`. The `needs_approval` macro option
  is still accepted.
- `RunState::approval_status`; use `RunState::resolution_status`.
- **Breaking:** `McpServer::needs_approval_for` is renamed `needs_resolution_for`. Human-approval
  names (`approve` / `reject`, MCP `RequireApproval`, `ToolErrorKind::ApprovalRejected`) stay.

[Unreleased]: https://github.com/charleslyh/openai-agents-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/charleslyh/openai-agents-rs/releases/tag/v0.1.0
