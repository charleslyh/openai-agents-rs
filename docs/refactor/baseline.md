# 基线与已核实事实

> 规划依据。所有行号取自重构前的单 crate 代码,拆分后会漂移,以符号名为准。

## 1. 规模

单 crate `openai-agents`,约 20.8k 行,45 个 `.rs`;`macros/` 为独立 proc-macro crate。

| 文件 | 行数 | 备注 |
| --- | --- | --- |
| `run/agent_loop.rs` | 1621 | `run_loop_inner` 一个函数约 1140 行 |
| `mcp/mod.rs` | 1140 | 类型、过滤、Trait、Client、工具转换、约 400 行内联测试混在一起 |
| `retry.rs` | 981 | 设置、策略、延迟、执行器、openai 专属 advice 混在一起 |
| `model/openai/chat_completions.rs` | 863 | |
| `run_state.rs` | 834 | |
| `model/openai/chat_convert.rs` | 750 | |
| `tracing/mod.rs` | 661 | |
| `run/tools.rs` | 641 | |
| `model/openai/mod.rs` | 626 | |
| `strict_schema.rs` | 617 | |
| `model/wire_events.rs` | 575 | |

布局不一致:`memory.rs` 与 `memory/` 并存,其余目录模块均用 `mod.rs`。

## 2. 外部重依赖的落点(已收敛,没有泄漏到核心)

| 依赖 | 文件 |
| --- | --- |
| `reqwest` | `model/openai/mod.rs`、`mcp/http.rs` |
| `async_openai` | `model/openai/mod.rs`、`model/openai/chat_completions.rs` |
| `rusqlite` | `memory/sqlite.rs` |
| `tokio::process` | `mcp/stdio.rs` |

## 3. 反向(向上)依赖清单

| 编号 | 位置 | 依赖 |
| --- | --- | --- |
| C1 | `error.rs:103,113,118,211,246,254` | `ToolGuardrailFunctionOutput`、`default_tool_timeout_error_message`、`retry_after_from_headers`、`Input/OutputGuardrailResult` |
| C2 | `model/mod.rs:79`、`model/openai/mod.rs`(多处) | `run::DefaultOpenAiApi` 及全局 getter/setter(定义于 `run/config.rs:31-54`) |
| C3 | `agent.rs:386` | `run::build_agent_as_tool` |
| C4 | `tool.rs:12,113,127` | `run_state::RunState`(`ToolResult.nested_state`) |
| C5 | `context.rs:24` | `run::{CallModelData, CallModelInputFilter, ModelInputData}` |
| C6 | `agent.rs:12,183,185,230,236,363` | `mcp::{McpConfig, McpServer, mcp_function_tools}` |
| C8 | `guardrail.rs` | `tracing` |
| C9 | `handoffs/history.rs`、`run/agent_loop.rs:1202` | `pyjson`(`pub(crate)`) |
| C10 | `model/mod.rs:13-14,64-87` | `openai::OpenAIProvider::from_env()`,注册 `openai` 等三个前缀 |

## 4. 核实结论(用于风险评级)

### C1:载荷类型是叶子
- `ToolGuardrailFunctionOutput`:`output_info: Value`、`behavior: ToolGuardrailBehavior`(`Allow` / `RejectContent{message}` / `RaiseException`)。
- `GuardrailFunctionOutput`:`output_info: Value`、`tripwire_triggered: bool`。
- `InputGuardrailResult`:`guardrail_name: String` + `GuardrailFunctionOutput`。
- `OutputGuardrailResult`:`guardrail_name`、`agent_name: String`、`agent_output: Value` + `GuardrailFunctionOutput`。只存名字,不持有 `Agent`。
- `default_tool_timeout_error_message`、`retry_after_from_headers`(`retry.rs:604`,`Option<&str> -> Option<f64>`)是纯函数。
- 结论:可无泛型化地下沉,风险低。

### C4:`RunState` 只出现在 `ToolResult.nested_state`
- `tool.rs` 仅有三处:导入、字段 `nested_state: Option<Box<RunState>>`、构造函数 `interrupted(.., nested_state: RunState)`。
- `ApprovalStore`(`run_state.rs:68`)被 `RunState`、`result.rs`、`run/tools.rs`、`run/agent_loop.rs` 使用。
- 注意:这不是"审批视图"问题,而是 `ToolResult` 携带嵌套 `RunState`(agent-as-tool 被中断时)。可选手段:
  1. `RunState` 数据结构整体下沉到 core(执行逻辑留在 runtime)。
  2. `nested_state` 改为 core 内定义的不透明载体(如 `Box<dyn Any + Send + Sync>` 或已序列化 JSON)。
- 倾向 1(类型安全),待依赖核查 `RunState` 的字段是否都能在 core 内表达;列入开放问题。

### C10:model 与 openai 的耦合
- `model/mod.rs:64-86` 构造默认 `MultiProvider` 时直接调用 `OpenAIProvider::from_env()`,并读取 `run::DefaultOpenAiApi`。
- `model/provider.rs` 无直接 `use`,仅有文案级耦合(`default_prefix = "openai"`、错误提示含 `OPENAI_API_KEY`)。
- `model/wire_events.rs` 使用者:`run/errors.rs`、`memory/history.rs`(只用 `FAKE_RESPONSES_ID`)、`model/scripted.rs`、`model/mod.rs`、openai 三个文件。因此它**不是** openai 专属,不能直接归入 openai crate;应拆分:通用部分留 core/types,openai 专属部分归 openai。

### C3 / C5 / C6
- `agent.rs` 只在 `as_tool` 处依赖 `run::build_agent_as_tool`(`:386`),和 `mcp_function_tools`(`:363`)。
- `context.rs` 对 `run` 的唯一依赖是 `CallModelData`、`CallModelInputFilter`、`ModelInputData`,均为纯数据或过滤器类型,可下沉到 core。

## 5. 大文件拆分切点(P1 用)

### `mcp/mod.rs`(1140)
| 目标文件 | 内容 | 现行范围 |
| --- | --- | --- |
| `types.rs` | `McpTool`、`McpCallToolResult`、`render_tool_result` | 61-127 |
| `filter.rs` | `ToolFilterContext`、`DynamicFilter`、`ToolFilter`、`RequireApproval` | 159-262 |
| `server.rs` | `McpConfig`、`trait McpServer` | 263-315 |
| `client.rs` | `McpClient` | 316-614 |
| `tools.rs` | `check_required_arguments`、`to_function_tool`、`mcp_function_tools` | 615-740 |
| `tests.rs` | 内联测试 | 741- |

### `run/agent_loop.rs`(1621)
| 目标 | 内容 | 现行范围 |
| --- | --- | --- |
| `session_writer.rs` | `SessionWriter` | 212-280 |
| `finalize.rs` | `finalize_run`、`finished_result`、`interrupted_result` | 1424-1559 |
| `resolve.rs` | `collect_agents`、`resolve_agent_by_name`、`default_provider`、`resolve_model` | 1560-1621 |
| 拆 `run_loop_inner` | 按阶段拆成 turn / model call / tool 执行 / handoff 子函数(需逐段评估借用与状态) | 281-1423 |

### `retry.rs`(981)
| 目标 | 内容 | 现行范围 |
| --- | --- | --- |
| `settings.rs` | 常量、`ModelRetryBackoffSettings`、`ModelRetrySettings` | 24-121 |
| `policy.rs` | 类型与 `RetryPolicy` | 122-443 |
| `policies.rs` | `mod retry_policies` | 444-603 |
| `delay.rs` | `retry_after_from_headers`、`default_retry_delay` 等 | 604-692 |
| `executor.rs` | `evaluate_retry`、`call_with_retry` 等 | 693-894 |
| 移到 openai | `openai_retry_advice` | 895- |

## 6. P0 基线采集清单(执行 P0 时填写)

以下数据需在真实环境运行后记录,本文档暂留空,不得臆造。

采集于重构前,Rust toolchain 与依赖均为当时状态。

| 项 | 命令 | 结果 |
| --- | --- | --- |
| 格式 | `cargo fmt --all -- --check` | 通过 |
| clippy | `cargo clippy --all-features --all-targets -- -D warnings` | 通过,零 error / zero warning |
| 测试(无默认特性) | `cargo test --no-default-features` | 25 个 target,197 passed / 0 failed |
| 测试(全特性) | `cargo test --all-features` | 262 passed / 0 failed |
| 文档 | `RUSTDOCFLAGS=-D warnings cargo doc --no-deps --all-features` | 通过 |
| vendor | `bash scripts/sync_vendor.sh --check` | `vendor ok: openai-agents-python 0.23.1 (pin v0.23.1)` |
| oracle | `.venv/bin/python scripts/run_parity.py --check` | 未采集(需 `.venv`;运行前先 `bash scripts/setup_venv.sh`) |
| 示例体积(默认) | `cargo build --release --example 01_hello` | 6 924 880 B |
| 示例体积(仅 openai) | `cargo build --release --no-default-features --features openai --example 01_hello` | 6 882 304 B |

说明:示例 `01_hello` 的 `required-features` 含 `openai`,不能与 `--no-default-features` 直接组合;上表"仅 openai"是关闭 `mcp` 后的体积。关闭 `mcp` 仅省约 42 KB,即当前 feature 对体积的边际收益很小,重依赖主要来自 reqwest / rustls。

每完成一个 P1 步骤后重测:197 / 262 两个数字不得下降,体积不应显著上升。
