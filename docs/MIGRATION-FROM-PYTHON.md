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
| `input_guardrails` / `output_guardrails` | `input_guardrail(name, fn)` / `output_guardrail(name, fn)` |
| `RunHooks` / `AgentHooks` subclasses | `impl RunHooks` / `impl AgentHooks` (every method has a no-op default) |
| `SQLiteSession` | `SqliteSession` (feature `sqlite`; **the same database file works in both SDKs**, D-038) |
| `RunConfig(...)` / `RunOptions(...)` | `RunConfig { field, ..Default::default() }` — always keep the `..Default::default()` tail |
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
| `Agent.as_tool(...)` with ~18 keywords | `AsToolConfig { name, description, needs_approval, max_turns }` | D-043 |
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
