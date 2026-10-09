# 能力满足度评估与改进建议

> 本文回答的是"**这个 SDK 能不能满足用户在 agent 方面的需求**"，偏离分析只是手段。
>
> 结论基于：
> 1. `vendor/openai-agents-python` @ `v0.23.1` 全量源码精读（`run_internal/`、`tool.py`、`models/`、`run.py`、`run_config.py` 等）；
> 2. 本仓 `src/` 全量精读 + `tests/`、`scripts/` 结构审查；
> 3. 与 `docs/DEVIATIONS.md` / `docs/COMPAT.md` 交叉确认（补齐了 5 个未记录项、修正了 2 处文档漂移）。
>
> **后续（已实施）**：本文 §4 的高优先级项已落地，见 [§6 实施记录](#六实施记录)。
>
> 原始素材（未受既有文档影响的独立分析）保留在
> [`docs/analysis/SCRATCH-01-raw-deviation.md`](./analysis/SCRATCH-01-raw-deviation.md)。

---

## 一、总体结论

**能。** 在"用 Rust 构建一个跑在任意 OpenAI 协议兼容端点上的 agent"这件事上，本 SDK 的能力是**完整且可用**的，
且工程质量高于同类移植项目（三层验证 + Python oracle + 显式偏离台账）。

三个关键判断：

| 判断 | 依据 |
|---|---|
| **核心运行时语义可信** | 主循环、工具、handoff、guardrail、HITL、结构化输出、session、retry 八大块与 Python 逐条对齐；连"多重 handoff 取第一个"、"结构化空输出软失败续跑"、"输出守卫拦截终态工具输出后仍 raise"这类细节都实现了。这不是抄 API 形状，是抄了语义。 |
| **缺口是有意的、且边界清晰** | `docs/DEVIATIONS.md` 的 Scope 表把"OpenAI 托管能力"划出去，并且**全仓 `TODO` / `FIXME` / `unimplemented!` 命中数为 0**——未实现项全部以显式声明而非沉默占位存在。这比"实现了 80% 但不知道缺什么"健康得多。 |
| **Rust 侧不是纯追随者** | `run_blocking`、provider 中立的 `CompactingSession` / `ContextWindowTrimmer`、`CancelMode` + `cancel_on_drop`、stdio MCP 环境变量白名单、`strict_schema` DoS 预算、`CompatibleProvider`、`RetryPolicy` 组合子——共 12 项 Python 没有的能力。定位是"协议中立的 Rust 原生 agent 运行时"。 |

**但要清醒的两点：**

1. 缺口里**对 Coding Agent 场景影响最大**的是本地复杂工具（Shell / Computer / ApplyPatch / Custom）与工具输出多形态（图片/文件）。Python 用户迁移过来会发现"我的 shell 工具没了"。
2. 偏离台账本身开始出现**漂移**（本次发现 COMPAT.md 与代码/D-017、D-020 互相矛盾）。台账一旦失真，"我知道我缺什么"这个最大优势就会失效。

---

## 二、按 agent 需求域的满足度

| 需求域 | 满足度 | 说明 |
|---|---|---|
| 单 agent + 工具循环（function calling） | ✅ 完整 | `#[function_tool]` + schemars schema、并行执行、超时、错误降级、4 种 `tool_use_behavior` |
| 多 agent 编排（handoff） | ✅ 完整 | 含 `input_filter` / `nest_handoff_history` / `on_handoff` / 循环图（`handoff_to_name`） |
| Agent 作为工具（`as_tool`） | ⚠️ 部分 | 能用，但嵌套 run 只能用 `RunOptions::default()`：无 context / session / hooks / run_config / 流式回调（D-043） |
| 人工审批 HITL | ✅ 完整 | fixed/dynamic `needs_approval`、`RunState` approve/reject、sticky 决策、嵌套下钻、JSON 序列化恢复 |
| 输入/输出防护栏 | ✅ 完整 | 含 `run_in_parallel`、并行 tripwire 取消在飞模型调用、工具级 input/output guardrails |
| 结构化输出 | ✅ 完整 | `output_type` + strict schema + 校验失败处理 + `final_output_as::<T>()` |
| 长对话 / 上下文管理 | ✅ 完整（且优于 Python） | Session + SqliteSession（与 Python 库文件互通）+ `ToolOutputTrimmer` / `ContextWindowTrimmer` / `CompactingSession`（任意 provider） |
| 流式输出 | ✅ 完整 | 三种后端统一的 Responses wire 事件词汇表、token 级 delta、`cancel` 两种模式 |
| 可观测性 | ⚠️ 部分 | **本地 span 树完整，但没有导出后端**（D-004）。生产使用必须自己实现 `TracingProcessor`（OTLP / Langfuse）；且 Rust span 载荷比 Python 薄（无 input/output/tools） |
| 任意模型 / 网关 | ✅ 完整（强项） | `CompatibleProvider`、免 key、`MultiProvider` 前缀路由、Chat Completions 默认、第三方非标兼容（DeepSeek reasoning、Gemini thought signature、Claude thinking blocks） |
| 重试与容错 | ✅ 完整（强项） | `RetryPolicy` 组合子 + `ReplaySafety` + provider advice + per-attempt timeout + `conversation_locked` 兼容 |
| MCP 工具生态 | ✅ 基本完整 | stdio + streamable HTTP、ToolFilter、审批；缺 hosted MCP、prompts/resources、legacy SSE（D-041） |
| **联网搜索 / 代码解释器 / 画图** | ❌ 不可用 | OpenAI 托管工具（D-006）。必须自己写 FunctionTool 或挂 MCP server |
| **Shell / 文件编辑 / Computer use** | ❌ 不可用 | Python 有 `ShellTool` / `ApplyPatchTool` / `ComputerTool`，本仓无。Coding Agent 需完全自研 |
| **语音 / Realtime** | ❌ 不可用 | 明确 out of scope |
| **沙箱执行** | ❌ 不可用 | 明确 out of scope（建议用 MCP server 或自写 function tool 承载） |
| **生产级 trace 看板** | ❌ 不可用 | 自建 processor |

**一句话**：对话式 / 工具调用式 / 多 agent 编排式 / 长记忆式 agent —— 满足。
**服务端托管工具依赖型**、**本地执行环境依赖型**、**语音型** —— 不满足，且短期不打算满足。

---

## 三、最终偏离结论（已与既有台账合并）

### 3.1 三类偏离

| 类别 | 数量级 | 性质 | 处置 |
|---|---|---|---|
| **定位性裁剪**（hosted tools、Realtime/Voice、Sandbox、Conversations API、cloud trace export、`Agent.prompt`） | 6 大块 | 有意，Scope 表已声明 | 不改，保持声明 |
| **语义级偏离**（`max_turns=None`、as_tool 参数面、工具输出多形态、`auto_previous_response_id`、中断期不落库 `RunState` 线格式、默认 API=Chat Completions …） | ~20 项，已全部入账 | 部分有意、部分是遗漏 | 见 3.2 |
| **能力缺口**（Shell/Computer/ApplyPatch/Custom 工具、扩展存储后端、span 详细载荷、`ItemHelpers` 子集 …） | ~20 项，多数低优先 | 已知 | 见 3.2 |

### 3.2 本次新增入账（既有文档未覆盖）

| ID | 问题 | 严重度 |
|---|---|---|
| **D-042** | `RunOptions.max_turns = None` 在 Python 是**关闭限制**，Rust 静默回退为 10 —— 且 `RunOptions::default()` 恒为 `Some(10)`，无测试覆盖 | **中（建议按 bug 修）** |
| **D-043** | `Agent.as_tool` 只支持 4 个参数；嵌套 run 拿不到 context/session/hooks/run_config，也没有 `on_stream` | 中 |
| **D-044** | 工具返回只能是单个 JSON 值 → 字符串；无法回传图片/文件给模型 | 中（Coding Agent 场景） |
| **D-045** | 无 `auto_previous_response_id`、无服务端会话 tracker，`previous_response_id` 需手工串联 | 低-中 |
| **D-046** | `RunResult` 不暴露 `context_wrapper`；异常不携带 `run_data` | 低 |

### 3.3 本次修正的文档漂移

| 位置 | 原状 | 实际 | 处置 |
|---|---|---|---|
| `COMPAT.md` Model settings | "19 fields"，且把 `extra_query` / `preserve_raw_usage` 列为 **Not ported** | 实际 **22 fields**，两者均已移植（D-017 也这么说） | 已修正 |
| `COMPAT.md` Testing | "`retry_advice` not ported" | `ModelStep::with_retry_advice` 已实现（`src/model/scripted.rs:174`），D-020 也这么说 | 已修正 |

**这类漂移是本仓目前最需要防范的退化。** 建议见 §4.6。

---

## 四、改进建议

### 4.1 代码组织结构（对齐 Rust 社区规范）

| # | 建议 | 理由 |
|---|---|---|
| 1 | **`src/run.rs`（3,019 行）拆为 `src/run/` 模块目录**：`mod.rs`（`Runner` 门面 + `RunConfig` / `RunOptions`）、`loop.rs`、`turn.rs`（单次模型调用与响应解析）、`tools.rs`（`plan_tool_calls` / `execute_planned_tools`）、`handoff.rs`、`finalize.rs`、`session.rs`（`SessionWriter`）、`errors.rs` | 单文件 3k 行是 Rust 生态公认的可维护性红线；`mod.rs` + 子模块是社区标准形态。`src/model/`、`src/mcp/`、`src/memory/` 已经是这个形态，`run.rs` 是唯一例外 |
| 2 | **所有配置结构加 `#[non_exhaustive]`**：`RunConfig`、`RunOptions`、`AsToolConfig`、`ToolExecutionConfig`、`McpConfig` | 现在给 `AsToolConfig` 加字段是破坏性变更（D-043 要加 8 个字段就会 break 用户）。`#[non_exhaustive]` + `Default` 是 Rust 处理"配置对象演进"的标准答案 |
| 3 | **`Arc<dyn Fn(..)>` 收敛为具名 trait**：`ToolInvoker`、`GuardrailFn`、`InputFilterFn`、`OnHandoff` 等 | 现在 `lib.rs` 有 8 个裸 `pub type X = Arc<dyn Fn..>`。具名 trait 能让用户 `impl` 自己的类型、让 rustdoc 生成可读页面、让错误信息可读 |
| 4 | **消灭 `.expect("snapshot")` 类 panic 路径**：`result.rs` / `run.rs` 里 `snapshot.lock().expect("snapshot")`、`tool_guardrail_log.lock().expect(..)` | Mutex 中毒或未来重构会让库 panic 而不是返回错误。用 `.unwrap_or_else(\|e\| e.into_inner())` |
| 5 | **`resolve_blocked_message` 里的 `std::panic::catch_unwind` 需重新考虑**（`run.rs:221`） | 用户 crate 设 `panic = "abort"` 时 `catch_unwind` 不会捕获而是直接 abort。建议改为让 formatter 返回 `Result`，或在文档明确"`output_guardrail_blocked_message` 的 formatter 不得 panic，且需要 `panic = "unwind"`" |
| 6 | **crate 级文档示例**：`lib.rs` 加 `#![doc = include_str!("../examples/01_hello.rs")]` 或手写可运行示例 | docs.rs 首页无示例是 Rust 库的常见差评点；README 有但 `cargo doc` 看不到。已有 `#![deny(missing_docs)]` 是好习惯，应补上 `#![doc(html_root_url)]` / `#![warn(clippy::pedantic)]` |
| 7 | **补 `CHANGELOG.md`**（Python 有 release-please 生成，Rust 没有） | 0.1.x 阶段的 API 演进需要给用户迁移依据；也是 `cargo semver-checks` 的对照物 |
| 8 | **考虑 `ModelError::Behavior` 独立成 `ModelBehaviorError` 类型** | Python 是独立异常类，用户 `match` 时更容易；现在挤在 `ModelError` 里 |
| 9 | **`pyjson.rs` 标为 `pub(crate)` 已对**；但 `items.rs` 的 `ItemHelpers::text_message` / `function_tool_call` 是"test helper shape"却 `pub` | 建议在文档注明或在 `testing` 模块重导出，避免用户误当生产 API |

### 4.2 验证机制：从 3 层扩到 7 层

现有 3 层（ScriptedModel / wiremock / Python oracle）已经比大多数 Rust SDK 强，但**层级类型单一**——全是"我写过的断言"，缺少"随机化/属性/端到端/门禁"这几类。

| 层 | 现状 | 建议新增 | 优先级 |
|---|---|---|---|
| **L0 静态门禁** | 有 `--no-default-features` 测试 | `cargo fmt --check`、`cargo clippy --all-features -- -D warnings`、`cargo doc --no-deps`（`RUSTDOCFLAGS=-D warnings`）、`cargo deny check`（许可证/供应链）、`cargo audit`、`cargo semver-checks`（API 破坏）、MSRV 声明与检查 | **高** |
| **L1 行为（ScriptedModel）** | ✅ 已有 22 个文件 8,282 行 | 引入 `tokio::time::pause()` 做**确定性并发**测试（工具并发顺序、`CancelMode::AfterTurn`、session 回调归因）；目前这些依赖真实调度 | 高 |
| **L1.5 属性 / 随机化** | ❌ **完全没有** | `proptest`：`ensure_strict_json_schema` 幂等性与不崩溃（已有 DoS 预算，但没有随机 schema 验证）、`chat_convert` 双向 round-trip、`RunState` JSON `serialize→deserialize` 等价、`pyjson` 与 Python `json.dumps` 一致性、handoff history nesting 的 flatten/restore 往返 | **高（最大空缺）** |
| **L2 HTTP 契约（wiremock）** | ✅ 1,267 行 + 591 行 third_party | 引入 **`insta` 快照**：把请求体整体快照化。现在断言是逐字段的，契约漂移容易漏 | 中 |
| **L2.5 跨 SDK 文件互通** | 只有 SqliteSession 测过 | 扩大到 `RunState` 之外的所有持久化格式（session items JSON 形状） | 低 |
| **L3 Python oracle parity** | ✅ 27 个场景 | **① 加 `--check` 模式**：现在 `--write-golden` 直接覆盖，漂移会被"重新生成"掩盖；**② 覆盖率驱动的补齐**：把 Python `tests/`（404 个文件）中与 core loop 相关的用例拉清单，逐条标注「已 parity / 仅 Rust 测试 / 未覆盖」，把差距显式化 | **高** |
| **L4 真实端点冒烟（nightly）** | ❌ | 真实网关矩阵（OpenAI / vLLM / Ollama / LiteLLM / DeepSeek / Gemini / Claude）。secret 驱动、**不阻塞 PR**、失败出报告。wiremock 无法替代真实服务器的怪癖 | 中 |
| **L5 性能 / 并发基准** | ❌ | `criterion`：run loop 单轮开销、工具并发扇出、session 写入、流式事件分发（`mpsc(64)` 是有界队列，慢消费者会反压——需要有数字） | 中 |

配套：
- **覆盖率门禁**：`cargo llvm-cov`，`src/run.rs` / `src/run_state.rs` / `src/strict_schema.rs` 设下限（如 75%）。
- **CI 矩阵**：`--no-default-features` / default / `--all-features` × stable / beta × Linux / macOS / **Windows**（MCP stdio 在 Windows 上语义不同，目前完全没测）。
- **`sync_vendor.sh --check` 进 CI 并 fail**（现在只在 CONTRIBUTING 里写着让人手跑）。

### 4.3 语义正确性（按优先级修）

1. **D-042 `max_turns = None`** —— 建议按 bug 修（Python 明确 `None` = 关闭限制），并加 parity 场景。
2. **D-043 `as_tool` 的 `run_config` / `hooks` / `session` / `context`** —— 机械性补全，性价比最高；让嵌套 run 继承外层配置是用户最直觉的期待。
3. **D-044 工具输出多形态** —— 若要做 Coding Agent / 多模态 agent，这是刚需（至少让 Responses 路径支持 `input_image`）。
4. **D-045 `auto_previous_response_id`** —— 与 "server conversation tracker" 一起做，D-034 的 `rewind` 也依赖它。

### 4.4 可观测性（生产落地的最大短板）

Cloud export 明确不做（D-004）是对的，但建议：

- **提供官方 OTLP `TracingProcessor` 示例**（放在 `examples/` 或独立 `contrib/`），把"自己实现 processor"从文档一句话变成可运行代码。
- **补齐 span 载荷**：`SpanData` 至少要有 `input` / `output`（受 `trace_include_sensitive_data` 控制）、`tools`、`error`。现在 `trace_include_sensitive_data=False` 的实际脱敏效果有限，因为本来就没存多少数据。
- `Span` 补 `started_at` / `ended_at` / `error`。没有时间戳的 trace 在生产上基本没法用。

### 4.5 定位与文档

- **把 README 的 "Positioning" 提升为 `docs/POSITIONING.md`** 并在 README 顶部引用。这是本项目最重要的一个决策，值得独立文档（含"什么不该用本 SDK"）。
- README 加 badge（CI / docs.rs / crates.io）—— Rust 用户的第一印象。
- 补 `docs/MIGRATION-FROM-PYTHON.md`：Python 用户逐 API 对照，明确"哪些要自己写"（这是 README 一句话带过的最大痛点）。

### 4.6 防止台账失真（流程建议）

本次发现的 2 处 COMPAT 漂移说明：**偏离台账靠人工维护会退化**。建议：

1. **CI 校验**：加一个 `scripts/check_docs_consistency.py` 或 Rust 测试，校验
   - `COMPAT.md` 声明的 `ModelSettings` 字段数 == 实际字段数；
   - `COMPAT.md` / `DEVIATIONS.md` 中引用的 `D-NNN` ID 全部存在且无重复；
   - `DEVIATIONS.md` 中"Not ported"提到的符号在 `src/` 中确实不存在（防止"已实现却仍标未实现"）。
2. **PR 模板 checklist**：改 `src/` 行为 → 必须同步 `DEVIATIONS.md`；改公开 API → 必须同步 `COMPAT.md`。写进 `docs/CONTRIBUTING.md`（现在只写了 commit message 格式和三层验证）。
3. **定期 re-audit**：`vendor/PINNED_VERSION` 升级时强制重跑一次完整的偏离审计（现在是靠人记得）。

---

## 五、结论

**这是一个定位清晰、工程质量扎实、语义对齐度高的 Rust agent SDK，能满足绝大多数非语音、非托管工具依赖的 agent 需求。**

最强的三件事：① 核心 loop 语义对齐质量；② 三层验证 + Python oracle；③ 显式且诚实的偏离台账。

最需要投入的三件事：① 补 `max_turns=None` 等语义漏洞 + `as_tool` 配置面；② 引入属性测试与静态门禁，把验证从 3 层扩到 7 层；③ 防止台账漂移（CI 校验 + PR checklist）。

不建议做的：hosted tools、Realtime/Voice、Sandbox —— Scope 判断是对的，保持。

---

## 六、实施记录

§4 中标记为"高优先级 / 建议修"的项已落地：

| 建议 | 处置 | 说明 |
|---|---|---|
| D-042 `max_turns = None` | ✅ 修复 | `max_turns` 全程保持 `Option<usize>`，无限制时跳过比较；`RunState.max_turns` 同步改 `Option`，schema 升到 `openai-agents-rs/3`（`/2` 的数字载入为 `Some(n)`）。两个测试：`max_turns_none_disables_the_limit`、`unlimited_run_state_round_trip_keeps_max_turns_null` |
| L0 静态门禁 | ✅ 新增 | `cargo fmt --check`、`clippy --all-features --all-targets -- -D warnings`、`RUSTDOCFLAGS=-D warnings cargo doc`。为此清理了 192 条存量告警：`result_large_err`(126) / `field_reassign_with_default`(35) / `needless_question_mark`(8) 移入 `Cargo.toml [workspace.lints]` 作为**只减不增的台账**，其余逐条修掉（含 `LoopStart` 大变体装箱、`await_holding_lock` 在测试中加白名单注释、3 个失效的 intra-doc 链接） |
| CI 修复 | ✅ | 原 `cargo test --features testing` 指向一个**不存在的 feature**，CI 一直是红的。现为 lint / test(ubuntu+macos) / parity 三个 job |
| `--check` 模式 | ✅ 新增 | `run_parity.py --check` 只比较、不覆盖，漂移逐字段打印；`--write-golden` 仍是唯一写入路径 |
| L1.5 属性测试 | ✅ 新增 | `tests/property_core.rs`（proptest）：strict schema 不 panic + 幂等 + 每个 object 节点封闭；context trimmer 只删不造 / 保留 pinned / 幂等；tool output trimmer 不增长；token 估算随文本单调 |
| Span 时间戳与错误 | ✅ 新增 | `Span::{started_at, ended_at, error}` + `SpanError` + `SpanGuard::set_error`；已在 `MaxTurnsExceeded` 与工具失败处埋点（只放 tool_name，不放错误文本以免泄敏） |
| 迁移文档 | ✅ 新增 | `docs/MIGRATION-FROM-PYTHON.md` |
| `#[non_exhaustive]` | ❌ 回退 | 试过，但 `#[non_exhaustive]` 会**完全禁止**外部 crate 用结构体字面量构造（连 `..Default::default()` 也不行），本仓测试与示例就有 13 处、下游也会被打破；而它防的"新增字段破坏调用方"其实 `..Default::default()` 已经防住了。改为在文档里要求保留 `..Default::default()` 尾 |
| `run.rs` 拆分为 `run/` | ⏸ 未做 | 3k 行单文件，改动面大且易冲突；建议作为独立 PR |
| insta 快照 / criterion 基准 / 真实端点 nightly / 覆盖率门禁 | ⏸ 未做 | 需要新依赖与 secret 配置，建议后续独立推进 |

### 两个新发现（非偏离，但影响工程健康度）

1. **CI 曾长期失效**：`--features testing` 不存在，等于 CUDA 从未跑过第 2、3 层。已修。
2. **`tests/mcp_interop.rs::an_agent_uses_tools_from_an_http_server` 偶发失败**：单独跑稳定，全量并发跑时偶发（Python HTTP server 起端口竞争）。建议后续给它固定端口或加等待重试，否则 CI 会间歇性变红。
