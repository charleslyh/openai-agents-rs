# 偏离分析素材 · 初稿（SCRATCH）

> **状态：临时素材文档，不是结论。**
> 本文件是**第一步**的产物：仅基于
> 1) `vendor/openai-agents-python`（pinned `v0.23.1`）源码精读；
> 2) `src/` Rust 实现精读；
> 交叉得出的原始差异清单。
>
> 撰写时刻意**没有**阅读 `docs/DEVIATIONS.md` 与 `docs/COMPAT.md`，以免被既有结论前置偏置。
> 与既有文档的交叉确认、去重、编号对齐与最终定稿在**第二步**完成。
>
> **第二步已完成**：结论见 [`../READINESS.md`](../READINESS.md)。交叉确认结果——
> 本清单中的 A12/A14/A15/A18/B1–B11 等已被 `DEVIATIONS.md` Scope 与 D-003…D-041 覆盖；
> A1（→ D-042）、A16（→ D-043）、A13（→ D-044）、A19（→ D-045）为**既有文档未覆盖的新项**，已入账；
> 另修正 `COMPAT.md` 两处漂移（ModelSettings 字段数、`ModelStep.retry_advice`）。
>
> 行号证据：`vendor/.../src/agents/**`（简写 `py:`）与 `src/**`（简写 `rs:`）。

---

## 0. 规模与结构对照（先建立量级感）

| 维度 | Python v0.23.1 | Rust 本仓 | 说明 |
|---|---|---|---|
| `agents/` 顶层 + 子包 Python 代码 | **85,476 行**（含 `realtime/` 2.6k、`sandbox/` 21k+、`models/` 8k、`run_internal/` 16k、`extensions/` 7k、`voice/` 3k、`mcp/` 3.6k、`tracing/` 4k） | — | 含大量 OpenAI 专有与实验能力 |
| `run_internal/` 核心循环 | 16,000+ 行 | — | `run_loop.py` 2.7k、`turn_resolution.py` 3.7k、`tool_execution.py` 2.8k、`tool_planning.py` 1.1k、`session_persistence.py` 1.4k |
| Rust `src/` | — | **15,252 行**（`run.rs` 3,019 / `mcp` 1,952 / `retry` 970 / `run_state` 817 / `tracing` 602 / `strict_schema` 602 / `model` 1,361 …） | 单仓单 crate + `macros` |
| 测试 | 404 个 `tests/*.py` | 22 个 `tests/*.rs`（8,282 行）+ 27 个 parity JSON | Rust 侧以行为测试 + wiremock + Python oracle 三层为主 |
| 公共 API 符号（`__all__`） | ~300 个 | ~180 个（`lib.rs` 重新导出） | 覆盖率约 60%，但**按"能力"计覆盖率更高**（见 §6） |

**结论量级**：Rust 版不是"完整移植"，而是**核心 agent 运行时 + 协议中立适配层**的移植，
有意裁掉了 OpenAI 托管能力、Realtime/Voice、Sandbox、扩展存储后端。

---

## 1. Python 侧：关键 API 与实现语义（从 tarball 读到的事实）

### 1.1 公共面（`py:__init__.py`）

- 配置入口：`set_default_openai_key / set_default_openai_client / set_default_openai_api /
  set_default_openai_responses_transport / set_default_openai_harness /
  set_default_openai_agent_registration / set_tracing_export_api_key / enable_verbose_stdout_logging`
- Agent：`Agent` / `AgentBase` / `StopAtTools` / `ToolsToFinalOutputFunction` / `ToolsToFinalOutputResult` / `AgentToolStreamEvent`
- Runner：`Runner` / `RunConfig` / `RunOptions` / `RunState` / `RunErrorHandlers` / `ToolExecutionConfig` /
  `ToolNameCollisionPolicy` / `ToolNotFoundBehavior` / `ReasoningItemIdPolicy` /
  `OutputGuardrailBlockedMessage(Args|Formatter)` / `ToolErrorFormatter(Args)`
- 模型：`Model` / `ModelProvider` / `ModelTracing` / `MultiProvider` / `OpenAIProvider` /
  `OpenAIChatCompletionsModel` / `OpenAIResponsesModel` / `OpenAIResponsesWSModel` /
  `OpenAIResponsesWebSocketOptions` / `OpenAIAgentRegistrationConfig`
- 工具：`Tool` 联合 13 型（`FunctionTool` / `FileSearchTool` / `WebSearchTool` / `ComputerTool` /
  `HostedMCPTool` / `CustomTool` / `ShellTool` / `ApplyPatchTool` / `LocalShellTool` /
  `ImageGenerationTool` / `CodeInterpreterTool` / `ToolSearchTool` / `ProgrammaticToolCallingTool`）
- 防护：`Input/OutputGuardrail` + `ToolInput/ToolOutputGuardrail` + 对应 `*_result` 与装饰器
- 记忆：`Session` / `SessionABC` / `SQLiteSession` / `OpenAIConversationsSession` /
  `OpenAIResponsesCompaction(Aware)Session` / `SessionSettings`
- 追踪：`Trace` / `Span` / `SpanData` 全族 + 15 个 `*_span()` 构造 + `TracingProcessor` / `set_trace_processors` / `add_trace_processor` / `flush_traces` / `set_tracing_disabled`
- 其他：`repl.run_demo_loop`、`apply_diff`、`sandbox` 子包、`computer`/`editor`、`prompts.Prompt`

### 1.2 运行循环（`py:run.py` + `run_internal/`）

执行顺序（证据 `py:run.py:596-2247`）：

1. 参数解析 → **输入为 `RunState` 时走恢复路径**，`max_turns` 被 `run_state._max_turns` 覆盖（`run.py:658`）
2. `prepare_input_with_session(...)` 合并会话历史（`681-704`）
3. 构造 `OpenAIServerConversationTracker`；`session_persistence_enabled = session and tracker is None`（`721`）→ **session 与服务端托管会话互斥**
4. 建 trace / `RunState` / `SandboxRuntime` / `PromptCacheKeyResolver`
5. 首轮输入落库（`968-976`）
6. `while True`：
   - 输入守卫只在 `current_turn == 0 and not resuming_turn` 构建（`1002-1006`），按 `run_in_parallel` 拆两组
   - `resolve_interrupted_turn`（恢复态，`1086-1252`）
   - `agent_span` → `current_turn += 1` → **`> max_turns` 判定**（`1506-1513`）
   - `run_single_turn`：`get_all_tools` → `initialize_computer_tools` → `get_system_prompt`/`get_prompt` →
     `get_handoffs` → `resolve_tool_name_collisions` → `get_output_schema` → `get_new_response` →
     `get_single_step_result_from_response`
   - `get_new_response`：`maybe_filter_model_input` → 去重 → **`maybe_reset_tool_choice`** →
     `on_llm_start` → `previous_response_id`/`conversation_id`/`prompt_cache_key` →
     `prepare_compaction_model_input` → `get_response_with_retry` → usage 累加 → `on_llm_end`
   - 响应解析：`process_model_response` → `execute_tools_and_side_effects`
   - 中途落库（非 FinalOutput，`1846-1908`）
   - 分支：`NextStepFinalOutput` / `NextStepInterruption` / `NextStepHandoff` / `NextStepRunAgain`

### 1.3 `max_turns` 精确语义

- `DEFAULT_MAX_TURNS = 10`（`py:run_config.py:45`）
- **`RunOptions.max_turns = None` 表示不限制**（`run_config.py:591-592`）
- 判定：`current_turn += 1; if max_turns is not None and current_turn > max_turns`（`run.py:1506-1507`）→ 先自增再比较
- 抛 `MaxTurnsExceeded(f"Max turns ({max_turns}) exceeded")`，**先问 `error_handlers["max_turns"]`**（`1523-1531`）

### 1.4 工具执行

- **并行**：默认 `gather_with_cancel` 并发 6 类执行器（function / computer / custom / shell / apply_patch / local_shell）；`parallel=False` 时按类别顺序串行（`tool_planning.py:951-1097`）
- 并发上限 `RunConfig.tool_execution.max_function_tool_concurrency`（默认 `None` = 全部同时起）
- **input guardrails**：invoke 前跑；`raise_exception` → `ToolInputGuardrailTripwireTriggered`；`reject_content` → 用 message 代替输出、不执行工具（`tool_execution.py:2716-2745`）
- **output guardrails**：拿到结果后跑；`reject_content` → 替换输出（`2750-2785`）
- `pre_approval_tool_input_guardrails=True` 时，**在产出 approval 中断之前也先跑一遍**
- **HITL**：`function_needs_approval` → `ToolApprovalItem` → `NextStepInterruption`；恢复走 `resolve_interrupted_turn`（批准执行 / 拒绝生成 rejection item）
- **超时**：`timeout_seconds`（必须 >0、有限、仅 async 装饰器支持）；`timeout_behavior` = `error_as_result` | `raise_exception`；取消后 0.25s 排空宽限
- **错误**：`failure_error_function`（默认返回 "An error occurred while running the tool…"），`None` 则抛出
- **`tool_use_behavior` 4 态**（`turn_resolution.py:760-792`）：`run_llm_again` / `stop_on_first_tool` /
  `{"stop_at_tool_names": [...]}`（匹配 `name` 或 `qualified_name`）/ `callable(context, results) -> ToolsToFinalOutputResult`
- **`reset_tool_choice`**：仅当 agent 用过工具时把 `tool_choice` 置 `None`（`tool_execution.py:561-569`，调用点 `run_loop.py:2618` / `2184`）
- **工具输出多形态**：`ToolOutputText/Image/FileContent` → `input_text` / `input_image` / `input_file`（`items.py:946-1031`）
- **未找到工具**：默认抛 `ModelBehaviorError`；`tool_not_found_behavior="return_error_to_model"` 时返回 `"Tool '<name>' not found."`

### 1.5 handoff

- 仅 `ResponseFunctionToolCall` 且 **qualified_name == name**（带 namespace 的调用永不路由到 handoff）
- 多重 handoff：只执行第一个，其余生成 `"Multiple handoffs detected, ignoring this one."`
- `input_filter`：handoff 自带 > `RunConfig.handoff_input_filter`；必须返回 `HandoffInputData`
- `nest_handoff_history` / `handoff_history_mapper`
- **双轨**：`session_step_items` 存全量给 session，`new_step_items` 用过滤后的给模型（`turn_resolution.py:702-736`）
- 服务端托管会话下：`input_filter` 非 None → `UserError`；`nest_handoff_history=True` → 降级 + warning

### 1.6 结构化输出

- 判定：本轮无工具/审批待跑 且（有 message item 或 无工具活动）
- `refusal` → `ModelRefusalError`（先问 handler）
- 有 schema 且非 plain text → `validate_json`
- **结构化但无文本 → `ModelBehaviorError`；无 handler 时不抛，返回 `NextStepRunAgain` 继续下一轮**（`turn_resolution.py:1052-1070`，唯一的软失败路径）
- FinalOutput 后跑输出守卫；tripwire 在"终态工具输出"场景下会**保留占位 item 后仍然 raise**（`blocked_output.py`）

### 1.7 会话 / 持久化

| 时机 | 位置 |
|---|---|
| 首轮用户输入 | `run.py:968-975` |
| 每轮结束（非 FinalOutput） | `run.py:1846-1908` |
| FinalOutput | `save_final_turn_items_after_guardrails` |
| **Interruption** | `run.py:2091-2099`（`_should_defer_interrupted_session_items` 为真时跳过） |
| 输入守卫 tripwire | `persist_session_items_for_guardrail_trip` |
| max_turns handler 输出 | `run.py:1537-1561` |

- 读取侧：`limit` → `strip_internal_input_item_metadata` → `apply_reasoning_item_id_policy` →
  `session_input_callback` → `drop_orphan_function_calls` → 去重
- `previous_response_id` / `conversation_id` / `auto_previous_response_id` 由 `OpenAIServerConversationTracker` 管理
- compaction：`prepare_compaction_model_input` / `record_compaction_model_response` / `_apply_post_write_compaction`，仅对 `is_openai_responses_compaction_aware_session` 生效

### 1.8 错误与重试

- `RunErrorHandlers` 三个接入点：`max_turns` / `model_refusal` / `invalid_final_output`
- 模型重试：`get_response_with_retry`（`model_retry.py:574-681`）、`ModelRetrySettings{max_retries, policy, backoff}`、
  provider `get_retry_advice`、`conversation_locked` 兼容路径、`ReplaySafety`

### 1.9 `RunConfig` 全字段（`py:run_config.py:353-582`）

`model` / `model_provider` / `model_settings` / `handoff_input_filter` / `nest_handoff_history` /
`handoff_history_mapper` / `input_guardrails` / `output_guardrails` / `tracing_disabled` / `tracing` /
`trace_include_sensitive_data` / `workflow_name` / `trace_id` / `group_id` / `trace_metadata` /
`session_input_callback` / `call_model_input_filter` / `tool_error_formatter` / `session_settings` /
`reasoning_item_id_policy` / **`sandbox`** / **`tool_execution{max_function_tool_concurrency, pre_approval_tool_input_guardrails}`** /
`tool_not_found_behavior` / `tool_name_collision_policy` / `output_guardrail_blocked_message`

---

## 2. Rust 侧：能力边界（`src/` 精读）

### 2.1 已实现（与 Python 语义对齐）

| 能力 | Rust 位置 | 对齐度 |
|---|---|---|
| `Agent` 全字段（含 dynamic instructions、output_type、hooks、mcp） | `rs:agent.rs:145-181` | 高 |
| `Runner::run` / `run_state` / `run_streamed` / `run_blocking` | `rs:run.rs:671-805` | 高（`run_blocking` 为 Rust 特有） |
| 主循环：turn / max_turns / 输入守卫分流 / 工具执行 / handoff / tool_use_behavior 4 态 | `rs:run.rs:1347-2295` | 高 |
| `reset_tool_choice` + `AgentToolUseTracker` 等价 | `rs:run.rs:1246,1453-1455` | 高 |
| 工具名冲突策略 | `rs:run.rs:914-989` | 高 |
| `needs_approval`（Fixed / Dynamic）+ `RunState` approve/reject + sticky + 嵌套下钻 | `rs:tool.rs:152-185`、`rs:run_state.rs:63-355` | 高 |
| 工具 input/output guardrails + 3 种 behavior | `rs:tool_guardrails.rs` | 高 |
| 工具超时 / `ToolTimeoutBehavior` / `timeout_error_function` | `rs:run.rs:2685-2706` | 高 |
| `failure_error_function` 三态（Default/Custom/Raise） | `rs:tool.rs:29-52` | 高 |
| 输出守卫 + blocked-output 占位与 session 净化 | `rs:run.rs:2348-2365`、`rs:run.rs:1143-1163` | 高 |
| 结构化输出 + `validate_json` + 空输出软失败续跑 | `rs:run.rs:1775-1828` | 高 |
| `RunErrorHandlers`（3 类） | `rs:run.rs:309-376` | 高 |
| handoff（含多重 handoff、input_filter、nesting、late_bound） | `rs:run.rs:2030-2193`、`rs:handoffs/` | 高 |
| Hooks（`RunHooks` / `AgentHooks`，各 7 个回调） | `rs:lifecycle.rs` | 高 |
| Session trait + `InMemorySession` + `SqliteSession`（**schema 与 Python 兼容**） | `rs:memory.rs`、`rs:memory/sqlite.rs:132-148` | 高 |
| `session_input_callback` + 来源归因 | `rs:memory.rs:173-295` | 高（Rust 增强） |
| MCP 客户端：stdio + streamable HTTP + `ToolFilter` + `RequireApproval` + `McpServerManager` | `rs:mcp/` | 中（见 §3） |
| 重试：`RetryPolicy` / `retry_policies` / `ReplaySafety` / provider advice / `conversation_locked` 兼容 | `rs:retry.rs` | 高（Rust 更结构化） |
| `strict_schema` 严格化（含深度/节点预算防 DoS） | `rs:strict_schema.rs` | 高（Rust 增强） |
| 本地 tracing（9 种 span、processor、开关） | `rs:tracing/mod.rs` | 中（无导出） |
| `ModelProvider` / `MultiProvider` 前缀路由 / `CompatibleProvider` | `rs:model/` | 高（Rust 增强） |
| `#[function_tool]` 宏 + schemars schema | `macros/` | 高 |
| 流式：`StreamEvent` 三型 + 标准 Responses wire 事件重放 | `rs:stream_events.rs`、`rs:model/wire_events.rs` | 中高 |

### 2.2 代码中显式声明"未移植 / 不支持"（grep 结果：`TODO/FIXME/unimplemented!` 命中数为 **0**）

| 位置 | 内容 |
|---|---|
| `rs:mcp/mod.rs:20-22` | legacy SSE transport、prompts & resources、`include_server_in_tool_names`、`tool_meta_resolver`、per-server retries、tool guardrails、**`HostedMCPTool` out of scope** |
| `rs:memory.rs:92` | Python 持久化 session 未移植（`InMemorySession` 仅进程内） |
| `rs:memory/history.rs:6-7` | 其他 item 类型属 OpenAI-hosted 工具，本 SDK 无 |
| `rs:context.rs:127-128` | `ToolOutputTrimmer` 未移植结构化（多段）输出与 `tool_search_output` 收缩 |
| `rs:tracing/mod.rs:3` | **Cloud OpenAI 导出有意不支持（D-004）** |
| `rs:model/wire_events.rs:13` | 未移植 `response.output_text.annotation.added` 与 logprob 载荷 |
| `rs:model/openai/chat_completions.rs:100-101,135-137` | Chat Completions 不支持 `conversation_id` |
| `rs:model/openai/chat_convert.rs:13` | 未移植 Claude thinking blocks / Gemini thought signatures |
| `rs:model/openai/chat_convert.rs:337-345,429-434` | `item_reference` / compaction item / `output_audio` 在 CC 路径不支持 |
| `rs:model_settings.rs:3` | 仅 Python 常用子集 |
| `rs:run_state.rs:4` | RunState schema **不是** Python 1.18 线格式 |
| `rs:memory/compaction.rs:8-10` | Python `OpenAIResponsesCompactionSession`（`responses.compact`）不可移植，改为 provider 中立实现 |

---

## 3. 原始差异清单（未编号、未与既有文档对齐）

### A 类：语义级偏离（会影响用户可观测行为）

| # | 主题 | Python | Rust | 影响 |
|---|---|---|---|---|
| A1 | `max_turns = None` | **不限制** | `unwrap_or(DEFAULT_MAX_TURNS)` → 限制为 10（`rs:run.rs:1193`） | 中：想跑无限轮的用户被静默限流 |
| A2 | 输入守卫运行范围 | 仅**首轮** agent 的 `input_guardrails` + `RunConfig.input_guardrails`；恢复态跳过 | 同（`rs:run.rs:1306-1314`） | 一致 ✓ |
| A3 | 顺序（blocking）守卫位置 | sandbox 场景下**在 session 创建之前**跑 | 无 sandbox；在首轮模型调用前跑 | 无影响 |
| A4 | 会话落库时机 | 首轮输入 / 每轮结束 / FinalOutput / **Interruption** / **守卫 tripwire** / **max_turns handler** | 每轮开始 `flush`；**仅在成功结束时**最终 flush（`rs:run.rs:1099-1109`）；中断不落库 | 中：HITL 暂停期间的已完成轮次在 Rust 侧不入 session |
| A5 | `previous_response_id` 管理 | `OpenAIServerConversationTracker`（`auto_previous_response_id`、server item id 追踪、hydrate from state 取最后一个非空 id） | 仅"上一轮有 response_id 就覆盖"（`rs:run.rs:1622-1624`） | 中：`auto_previous_response_id`、断链修复缺失 |
| A6 | `conversation_id` | Responses 支持；Chat Completions 不支持 | 透传；CC 路径报 `Unsupported` | 一致 |
| A7 | session ↔ 服务端托管会话互斥 | 显式校验（`validate_session_conversation_settings`） | 部分：`input_filter` + server-managed → `UserError`（`rs:run.rs:2137-2144`） | 基本一致 |
| A8 | 结构化输出"无文本" | `NextStepRunAgain` 软失败续跑 | 同（`rs:run.rs:1818-1826`） | 一致 ✓ |
| A9 | 输出守卫 tripwire + 终态工具输出 | 保留占位 item、净化 session、仍 raise | 同（`save_blocked_turn` + raise） | 一致 ✓ |
| A10 | 工具失败隔离 | `isolate_function_tool_failures`（>1 个 function run 时） | 首错即返回并取消兄弟（`rs:run.rs:2765`） | 低：Rust 靠 `failure_error_function` 先行转换，行为接近 |
| A11 | 工具并发类别 | 6 类执行器并行 | 仅 function tool，可选 `max_function_tool_concurrency` | 低（Rust 无其他类别） |
| A12 | `pre_approval_tool_input_guardrails` | 支持（`RunConfig.tool_execution`） | `ToolExecutionConfig` 只有 `max_function_tool_concurrency` | 中：审批前拦截能力缺失 |
| A13 | 工具输出多形态 | `ToolOutputText/Image/FileContent` → `input_text/image/file` | `value_to_tool_string`（`rs:run.rs:2977`）→ 一律字符串 | 中：无法回传图片/文件给模型 |
| A14 | `RunConfig` 默认 API | Responses | **Chat Completions**（`rs:run.rs:76-81`） | 有意偏离（协议中立定位） |
| A15 | `RunState` 线格式 | 1.18 wire | `openai-agents-rs/2`，process-local | 有意偏离 |
| A16 | `Agent.as_tool` 参数 | ~18 个（含 `custom_output_extractor` / `on_stream` / `run_config` / `hooks` / `session` / `parameters` 结构化入参 / `input_builder` / `include_input_schema` / `on_stream_max_pending_events` / `is_enabled` / `failure_error_function`） | `AsToolConfig` 仅 4 个（`name` / `description` / `needs_approval` / `max_turns`，`rs:agent.rs:132-141`） | 高：嵌套 agent 的可观测性（on_stream）与自定义抽取缺失 |
| A17 | `RunItem` 种类 | 含 `MCPApprovalRequest/ResponseItem`、`MCPListToolsItem`、`ToolSearchCall/OutputItem`、`CompactionItem` | 7 种（Message/ToolCall/ToolCallOutput/ToolApproval/HandoffCall/HandoffOutput/Reasoning） | 中：MCP 审批与 compaction 不可观测 |
| A18 | `ModelSettings` 字段 | 24 个 | 21–22 个；缺 `prompt_cache_retention`、`context_management`、`prompt_cache_options` | 低-中 |
| A19 | `RunOptions` | 含 `auto_previous_response_id` | 无 | 低 |
| A20 | `Usage` | `requests` / details | 有（`usage.rs` + `RequestUsage`） | 一致 ✓ |
| A21 | `ItemHelpers` | 十余个方法 | 6 个（`rs:items.rs:179-248`） | 低 |

### B 类：能力缺失（Python 有，Rust 无）

| # | 能力 | Python 规模 | 用户影响（agent 场景） |
|---|---|---|---|
| B1 | **Hosted 工具**：WebSearch / FileSearch / CodeInterpreter / ImageGeneration / ToolSearch / ProgrammaticToolCalling / HostedMCP | `tool.py` 约 1,000 行 | **高**：用户想要"联网搜索/代码执行/画图"必须自己写 FunctionTool |
| B2 | **本地复杂工具**：ComputerTool / ShellTool / LocalShellTool / ApplyPatchTool / CustomTool | `tool.py` + `computer.py` + `editor.py` + `apply_diff.py` 约 800 行 | **高**：Coding Agent 场景（shell、apply_patch）需要自研 |
| B3 | **Realtime / Voice**（WebSocket 语音） | `realtime/` 2.6k + `voice/` 3k | 高（若做语音助手） |
| B4 | **Sandbox 运行时**（Docker / Unix local / 快照 / 挂载安全） | `sandbox/` 21k+ | 高（若做可执行代码的 agent） |
| B5 | **Tracing 导出**（`BackendSpanExporter` / `BatchTraceProcessor` / OpenAI dashboard） | `tracing/processors.py` 922 行 | **高**：生产可观测性受限，需自建 processor |
| B6 | **Conversations API**（`OpenAIConversationsSession`） | `memory/openai_responses_compaction_session.py` | 中：服务端托管会话不可用 |
| B7 | **`responses.compact`**（服务端 compaction） | 同上 | 中：Rust 改为 provider 中立 `CompactingSession`（**等价替代**） |
| B8 | **扩展存储后端**（Redis / MongoDB / SQLAlchemy / Dapr / 加密 / AdvancedSQLite） | `extensions/memory/` 6k+ | 中：Rust 仅 `InMemory` + `Sqlite` |
| B9 | **扩展模型**（LiteLLM / any-llm） | `extensions/models/` 2.6k | 低：Rust 用 `CompatibleProvider` 覆盖 |
| B10 | **prompt 对象**（`Agent.prompt` / `DynamicPromptFunction`） | `prompts.py` | 低（服务端 prompt，本就 out of scope） |
| B11 | `run_demo_loop` / `repl` | `repl.py` | 低（CLI 便利） |
| B12 | 可视化扩展（Graphviz） | `extensions/visualization.py` | 低 |
| B13 | MCP：prompts / resources / legacy SSE / 服务端 MCP | `mcp/server.py` 2.7k | 中 |
| B14 | `tool_namespace` / `defer_loading` / `allowed_callers` / `output_json_schema` | `tool.py:1662-1699` | 低-中（Responses-only 特性） |
| B15 | `Agent.clone(**kwargs)` | `agent.py:571-604` | 无（Rust 用 `Clone` + builder） | 低 |
| B16 | Span 类型：Speech / SpeechGroup / Transcription / MCPListTools | `tracing/span_data.py` | 低（语音/MCP 相关） |
| B17 | `SpanData` 详细载荷（input/output/tools/mcp_data/error 时间戳） | 同上 | **中**：Rust span 只存名字 + usage |
| B18 | `set_default_openai_harness` / `OpenAIAgentRegistrationConfig` | `_config.py` | 低 |
| B19 | `set_default_openai_responses_transport`（WebSocket） | — | 无 WS |
| B20 | `apply_diff` / `ApplyPatchEditor` | `apply_diff.py` 411 行 | 中（Coding Agent） |

### C 类：Rust 特有增强（Python 无）

| # | 能力 | 位置 | 价值 |
|---|---|---|---|
| C1 | `run_blocking` / `run_blocking_on`（任意线程、免 runtime 陷阱） | `rs:run.rs:784-868` | 高（Rust 生态刚需） |
| C2 | `CompactingSession` + `ModelSummarizer`（**任意 provider**） | `rs:memory/compaction.rs` | 高（Python 仅 OpenAI） |
| C3 | `ContextWindowTrimmer` / `ToolOutputTrimmer` / `chain_input_filters` | `rs:context.rs` | 高 |
| C4 | `CancelMode::{Immediate, AfterTurn}` + `cancel_on_drop` | `rs:result.rs:188-265` | 高（防孤儿任务 / 浪费 token） |
| C5 | stdio MCP **环境变量白名单**（默认 `env_clear()`） | `rs:mcp/stdio.rs:26-29` | 高（安全：不泄漏 API key） |
| C6 | `strict_schema` 深度/节点预算（DoS 防护） | `rs:strict_schema.rs:13-18` | 中 |
| C7 | `RunState` 嵌套审批下钻（`Agent.as_tool` 多层 HITL） | `rs:run_state.rs:337-355` | 中 |
| C8 | `CompatibleProvider`（免 key / 第三方网关） | `rs:model/openai/mod.rs:122-159` | 高 |
| C9 | 重试 `RetryPolicy` 组合子 `all/any` + hard veto + delegable replay veto | `rs:retry.rs:444-600` | 中 |
| C10 | `SqliteSession` 表名白名单校验（SQL 注入防护） | `rs:memory/sqlite.rs:48-60` | 中 |
| C11 | session 回调**来源归因**（`_agents_session_origin`） | `rs:memory.rs:173` | 中 |
| C12 | `ModelSettings.preserve_raw_usage` / `extra_args` 等网关适配 | `rs:model_settings.rs` | 中 |

---

## 4. 关键语义点核对表（逐条对照，供第二步确认）

| 语义点 | Python | Rust | 是否一致 |
|---|---|---|---|
| `current_turn += 1` 后再比 `> max_turns` | ✓ | ✓（`rs:run.rs:1348,1379`） | ✅ |
| 输入守卫仅首轮 + 非恢复态 | ✓ | ✓ | ✅ |
| 结构化输出空文本软失败续跑 | ✓ | ✓ | ✅ |
| 输出守卫 tripwire 终态工具输出处理 | ✓ | ✓ | ✅ |
| handoff 后 `should_run_agent_start_hooks` | ✓ | ✓（`rs:run.rs:2185-2190`） | ✅ |
| handoff 多重只执行第一个 | ✓ | ✓ | ✅ |
| `maybe_reset_tool_choice` 仅"用过工具"后清 | ✓ | ✓ | ✅ |
| `tool_use_behavior` 第 4 态必须返回 `ToolsToFinalOutputResult` | ✓ | ✓（类型强制） | ✅ |
| session 与 server-managed 互斥 | ✓ | 部分 | ⚠️ |
| `max_turns=None` 无限制 | ✓ | ✗ | ❌ |
| 中断时落库 session | ✓ | ✗ | ❌ |
| 工具输出多形态（image/file） | ✓ | ✗ | ❌ |
| `pre_approval_tool_input_guardrails` | ✓ | ✗ | ❌ |
| `auto_previous_response_id` | ✓ | ✗ | ❌ |
| `Agent.as_tool` 流式回调 `on_stream` | ✓ | ✗ | ❌ |

---

## 5. 待第二步确认的问题（带着问题去读 DEVIATIONS / COMPAT）

1. A1（`max_turns=None`）是否已被记录为有意偏离？若无，是 bug 还是设计？
2. A4（中断不落库）是否已被记录？
3. A13（工具输出多形态）是否已被记录？
4. A16（`as_tool` 参数裁剪）是否已被记录？
5. A12（`pre_approval_tool_input_guardrails`）是否已被记录？
6. `docs/` 中已有的 D-XXX 编号体系是什么？本清单需要与之一一映射（避免重复编号、遗漏）。
7. COMPAT 的"支持矩阵"口径与本清单是否一致？有无本清单未发现的已支持项。
8. 哪些 B 类缺失其实已被 README 的"Positioning / Out of scope"覆盖，哪些是**文档未声明的隐性缺口**。

---

## 6. 初步判断（素材阶段的主观结论，待第二步校准）

- **核心 agent 循环（run loop / tools / handoffs / guardrails / HITL / structured output / sessions / retry）覆盖度 ≈ 90%+**，且语义对齐质量高（大量细节如"多重 handoff"、"空结构化输出软失败"、"blocked output 占位"都已实现）。
- **缺口集中在三类**：
  1. OpenAI 托管能力（hosted tools / Conversations / compaction / trace 导出）——**定位性裁剪，合理**；
  2. 本地复杂工具（Shell / Computer / ApplyPatch / Custom / 多模态输出）——**对 Coding Agent 场景影响最大**；
  3. Realtime/Voice/Sandbox——**独立产品线，短期不应纳入**。
- **Rust 侧增强（C 类）多于多数移植项目**，说明这不是"抄一遍"，而是有明确的 Rust 生态定位。
