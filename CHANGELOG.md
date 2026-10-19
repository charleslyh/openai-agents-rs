# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Nothing has been released yet; the first
entry will be `0.1.0`.

## [Unreleased]

### Fixed

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
- New crate `openai-agents-providers`: presets for DeepSeek, Gemini, Groq, OpenRouter, Anthropic and
  Ollama, `ProviderRegistry::from_env()` discovery, `prefix/model` routing with inference,
  a capability matrix and `UnsupportedCapability` (X-013).
- Requests a provider cannot honour are refused before they are sent: reasoning, a JSON schema,
  tool calls, parallel calls, `store`, images and PDFs, reported as `ModelError::Unsupported`
  naming the provider and the capability.
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
  lookup. Implement it with `#[async_trait]` (re-exported as `openai_agents::async_trait`) and
  `.await` direct calls.

### Added

- `#[function_tool(timeout = <seconds>)]`.
- `openai_agents_core::utf8_stream::Utf8Stream` for decoding chunked byte streams.
- A `twenty_tools_ten_turns` benchmark.
- `Extensions`: typed per-run values (`RunOptions::with(value)`, `ctx.get::<T>()`,
  `ctx.require::<T>()`).
- `Runtime`: owns the default model provider and the runtime behind blocking calls;
  `Runtime::global()` keeps the process-wide default.
- The `Tool` trait (`FunctionTool::from_tool`), `FunctionTool::typed` for schema-from-struct tools,
  `Tool::requires` / `FunctionTool::require::<T>()` checked before the first turn, and
  `ToolMiddleware` / `ModelMiddleware`.
- A `:<effort>` model-name suffix (`"o4-mini:high"`) for names resolved through a provider.
- `Capability` / `CapabilityParts`: named bundles of tools, instructions, guardrails, default model
  settings and required extensions, mounted with `Agent::capability(..)`.
- `prelude` modules (`openai_agents::prelude`, `openai_agents_core::prelude`) with the types most
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
