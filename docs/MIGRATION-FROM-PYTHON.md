# Migrating from openai-agents-python

For someone who knows the Python SDK (@ `v0.23.1`) and wants the same thing in Rust. It lists
what maps directly, what changes shape, and — the part that actually costs time — **what you have
to write yourself**.

The full support matrix is [COMPAT.md](./COMPAT.md); the reasoning behind each difference is
[DEVIATIONS.md](./DEVIATIONS.md).

## 1. What maps almost one-to-one

| Python | Rust |
|--------|------|
| `Agent(name=..., instructions=..., tools=..., handoffs=..., model=..., model_settings=..., output_type=..., hooks=...)` | `Agent::new(name).instructions(..).tools(..).handoffs(..).model(..).model_settings(..).output_type(..).hooks(..)` |
| `Runner.run(agent, input)` | `Runner::run(&agent, input, RunOptions::default()).await` |
| `Runner.run_sync(...)` | `Runner::run_blocking(&agent, input, RunOptions::default())` (D-001) |
| `Runner.run_streamed(...)` + `async for` | `Runner::run_streamed(agent, input, opts)` + `next_event().await` |
| `@function_tool` | `#[function_tool]` (schema from Rust types, doc comment = description; D-002) |
| `handoff(agent)` / `handoff(agent, tool_name_override=...)` | `handoff(agent)` / `handoff_with(agent, name, description)` |
| `input_guardrails` / `output_guardrails` | `input_guardrail(name, fn)` / `output_guardrail(name, fn)`, returning a `GuardrailFunctionOutput` (`Verdict`: pass, fail, transform) |
| `RunHooks` / `AgentHooks` subclasses | `impl RunHooks` / `impl AgentHooks` (every method has a no-op default) |
| `SQLiteSession` | `SqliteSession` (feature `sqlite`; **the same database file works in both SDKs**, D-038) |
| `RunConfig(...)` / `RunOptions(...)` | `RunConfig::default()` plus grouped fields or chain methods, see [RunConfig groups](#runconfig-groups). `RunConfig` is `#[non_exhaustive]`, so a struct literal does not compile outside the crate |
| `ModelSettings(...)` | `ModelSettings { .. }` (22 fields; 3 OpenAI-only ones are not ported, D-017) |
| `Runner.run(agent, state)` after HITL | `Runner::run_state(&agent, state, opts)` |
| `ScriptedModel` / `ModelStep` | `ScriptedModel` / `ModelStep` in `openai_agents::testing` |

## 2. Differences you will hit on day one

| Python | Rust | Why |
|--------|------|-----|
| `RunContextWrapper[TContext]`, generic over the context | `RunContextWrapper` carrying `Arc<dyn Any>`; `context::<T>()` / `try_context::<T>()` | No generic parameter threaded through `Agent` / `Runner` (D-014). Type errors surface at runtime with an explicit message |
| `Agent` is mutable, `agent.tools.append(t)` | `Agent` is a plain cloneable value; `Agent::new(..).tools(vec![..])` | Owned graph (D-030) |
| Cyclic handoffs: build, then mutate `handoffs` | `handoff_to_name("A")` — a late-bound handoff resolved by name at run time | Rust cannot build a cyclic owned graph (D-030) |
| `max_turns=10` default; `None` disables | same: `RunOptions::default()` is `Some(10)`, `None` disables the limit | D-042 |
| Exceptions (`MaxTurnsExceeded`, `ModelBehaviorError`, …) | `AgentsError` enum (`AgentsError::MaxTurns(..)`, `AgentsError::Model(ModelError::Behavior(..))`) | Rust has no exception hierarchy |
| `failure_error_function=None` raises | `FunctionTool::raise_on_error()` | B7 |
| Default API is Responses | Default is **Chat Completions**; opt into Responses with `set_default_openai_api(Responses)` or the `openai_responses/<model>` prefix | D-I: nearly every compatible server speaks Chat Completions |
| `AsyncOpenAI` client passed around | `OpenAIProvider::from_env()` / `CompatibleProvider::new(base_url)`; `MultiProvider` for `prefix/model` routing | D-013 |
| Tool returns `str` or a structured output part | Tool returns one `serde_json::Value`, rendered as text | D-044: images / files cannot be handed back to the model |
| `Agent.as_tool(...)` with ~18 keywords | `AsToolConfig { name, description, needs_resolution, max_turns }` | D-043 |
| `enable_verbose_stdout_logging()` | the `tracing` crate | — |

## 3. What you must write yourself

This is the real cost of migrating, and it is why the README's positioning section exists.

| You want | Python gives you | In Rust you write |
|----------|------------------|-------------------|
| Web search, file search, code interpreter, image generation | `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`, `ImageGenerationTool` | A `FunctionTool` (or an MCP server) that calls the API you want |
| Remote MCP through OpenAI | `HostedMCPTool` | `McpClient::streamable_http(..)` — your SDK talks to the server instead of OpenAI |
| Shell commands | `ShellTool` / `LocalShellTool` | A `FunctionTool` that runs `tokio::process::Command`. **Do your own sandboxing** |
| File edits | `ApplyPatchTool` + `apply_diff` | A `FunctionTool` over your own patch application |
| Computer use | `ComputerTool` + `Computer` | A `FunctionTool` driving your own automation |
| Trace export to a backend | `add_trace_processor(BatchTraceProcessor(...))`, `set_tracing_export_api_key` | `impl TracingProcessor` for OTLP / Langfuse / whatever you use. Cloud export is not planned (D-004) |
| Long-context handling via the API | `OpenAIResponsesCompactionSession` (`responses.compact`) | `CompactingSession` + `ModelSummarizer` (any model), or `ContextWindowTrimmer` / `ToolOutputTrimmer` as a `call_model_input_filter`. Provider-neutral, and not OpenAI-only (D-039) |
| Redis / MongoDB / SQLAlchemy session storage | `agents.extensions.memory.*` | `impl Session` |
| Realtime / voice | `agents.realtime`, `agents.voice` | Not available |
| A sandboxed execution runtime | `agents.sandbox` | Not available; isolate at the process/container level |

## 4. Things that are better here

Not a migration aid, but worth knowing before you decide it is a downgrade:

- `run_blocking` / `run_blocking_on` work from any thread, with or without a runtime (D-001).
- `CancelMode::{Immediate, AfterTurn}` and `cancel_on_drop(true)` — Python's `cancel()` has no
  "let the turn finish" mode (D-031).
- `CompatibleProvider`: a base URL with an optional key, for vLLM / Ollama / LiteLLM / gateways.
- The Chat Completions adapter tolerates non-conforming servers (missing tool-call ids, list
  `content`, error chunks) and replays DeepSeek reasoning / Gemini thought signatures /
  Claude thinking blocks (D-040).
- `ensure_strict_json_schema` has depth and node budgets, so a hostile or pathological schema
  cannot blow up the process.
- The stdio MCP client clears the child's environment down to a safe allow-list instead of
  inheriting your API keys (D-041).
- `RunOptions::resolve_with(|call| async { .. })`: Python returns every pause to you; here a
  callback resolves the paused calls and the run resumes by itself, and the explicit `RunState`
  flow is still there when the run has to outlive the process (X-009).
- A resolution given through `RunState` is bound to the agent, tool and arguments of the call it
  answered: an approval cannot be carried to a call whose arguments changed, even in a stored
  state (X-010). Resuming also checks a fingerprint of the agent graph (X-011).
- A guardrail returns a `Verdict`: pass, fail (with a reason and a severity) or
  `Transform(content)`, so it can redact and let the run continue instead of only stopping it
  (X-012).
- `FunctionTool::typed`: one `#[derive(Deserialize, JsonSchema)]` struct gives you the tool's
  schema and its parsed arguments, so a tool needs no hand-written JSON.
- `Capability`: a named bundle of tools, instructions, guardrails and model settings you mount on
  an agent, equal to handing over its parts by hand (X-007).
- `ModelProvider::get_model` is `async`, so a provider can fetch credentials or look a deployment
  up before it builds the model (X-008).

## RunConfig groups

Python's `RunConfig` is one flat dataclass; here its fields are grouped by topic. Each group is a
`#[non_exhaustive]` struct with chain methods, and `RunConfig` has `with_model` / `with_tracing` /
`with_session` / `with_handoff` / `with_tools`. In practice you rarely touch `RunConfig` at all:
`RunOptions` chains the common ones (`model`, `temperature`, `max_turns`, `reasoning`, `session`,
`hooks`, `workflow_name`, `context`, `resolve_with`, `with`) and hands you a closure for each group
(`.tracing(..)`, `.tools(..)`, `.session_opts(..)`, `.config(..)`).

| Python `RunConfig` field | Rust |
| --- | --- |
| `model` | `config.model.model_ref` (chain: `ModelOptions::model(..)`) |
| `model_provider` | `config.model.provider` |
| `model_settings` | `config.model.settings` |
| `reasoning_item_id_policy` | `config.model.reasoning_item_id_policy` |
| `call_model_input_filter` | `config.model.input_filter` (chain: `ModelOptions::input_filter(..)`) |
| `tracing_disabled` | `config.tracing.disabled` |
| `workflow_name`, `trace_id`, `group_id` | `config.tracing.workflow_name`, `.trace_id`, `.group_id` |
| `trace_metadata` | `config.tracing.metadata` |
| `trace_include_sensitive_data` | `config.tracing.include_sensitive_data` |
| `tracing` | `config.tracing.config` |
| `session_input_callback`, `session_settings` | `config.session.input_callback`, `.settings` |
| `nest_handoff_history`, `handoff_history_mapper`, `handoff_input_filter` | `config.handoff.nest_history`, `.history_mapper`, `.input_filter` |
| `tool_not_found_behavior`, `tool_execution`, `tool_name_collision_policy` | `config.tools.not_found_behavior`, `.execution`, `.name_collision_policy` |
| `tool_error_formatter`, `output_guardrail_blocked_message` | `config.tools.error_formatter`, `.guardrail_blocked_message` |
| `input_guardrails`, `output_guardrails` | `config.input_guardrails`, `config.output_guardrails` |

```rust
let options = RunOptions::new()
    .with(tenant)
    .temperature(0.2)
    .workflow_name("support")
    .tools(|t| *t = t.clone().not_found_behavior(ToolNotFoundBehavior::ReturnErrorToModel));
```
