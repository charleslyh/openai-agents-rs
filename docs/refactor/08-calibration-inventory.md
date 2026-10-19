# 08 实现校准清单(M0)

目的:在 M1 重构之前,把已有的明显坏味道列出来并给出处置,避免带进新结构。
范围:行为零变化。处置分三类:**立即修**(M0 内修)、**重构中顺带**(M1 触及该处时处理)、**接受**(记录在案,不动)。

## 基线(2026-10-16,master)

| 项 | 数值 |
| --- | --- |
| `cargo fmt --check` | 通过(M0 起点有 2 个文件格式漂移,已格式化) |
| `cargo clippy --all-features --all-targets -D warnings` | 通过 |
| `cargo test --workspace --all-features` | 288 通过 / 0 失败 |
| `check_module_deps.py` / `check_runner_api.py` / `sync_vendor.sh --check` | 通过(6 个 Runner 方法,pin v0.23.1) |
| 非测试 `lock().expect/unwrap` | 65 处(testing 12、mcp 20、core 31、openai 2) |
| `allow(clippy::too_many_arguments)` | 4 处 |
| `allow(unused_assignments)` / `allow(dead_code)`(非测试) | 1 / 2 处 |
| workspace lints 允许清单 | `result_large_err`、`field_reassign_with_default`、`needless_question_mark` |
| `run_loop_inner` | 约 640 行、7 参数 |

bench(`cargo bench -p openai-agents --bench run_loop`,本机):

| 场景 | 时间 |
| --- | --- |
| single_turn_no_tools | 约 14.0 µs |
| one_tool_then_answer | 约 27.6 µs |
| eight_parallel_tools | 约 77.5 µs |
| **twenty_tools_ten_turns(新增)** | **约 347 µs** |
| stop_on_first_tool | 约 19.3 µs |
| session_add_then_get | 约 1.56 ms |

## A. 结构类

| # | 位置 | 证据 | 处置 |
| --- | --- | --- | --- |
| A1 | `core/src/run/agent_loop.rs:199` | `run_loop_inner` 约 640 行、7 参数、约 20 个 mut 局部变量,含 `allow(unused_assignments)` | 立即修:拆分 |
| A2 | `core/src/run/finalize.rs:24,129`、`run/turn/handoff.rs:31`、`run/turn/tools.rs:168` | 4 处 `too_many_arguments`,最大 13 参数 | 立即修:参数收进结构体 |
| A3 | `core/src/tool.rs:106-142`、`run_state/` | 同义别名并存:`NeedsApproval/NeedsResolution`、`ToolApprovalItem/ToolInterruptionItem`、`ApprovalStore/ResolutionStore`、`ApprovalDecision/ToolResolution` | 立即修:保留 resolution 一套 |
| A4 | `run_state/mod.rs:187`、`openai/src/chat_completions/body.rs:136` | `allow(dead_code)` | 立即修 |
| A5 | `core/src/run/config.rs:104` | `catch_unwind` 兜底 | 接受(先核实原因,M1 随 Arc 改造处理) |
| A6 | `Arc::new(agent.clone())` 共 11 处 | 热路径深拷贝 | 重构中顺带(M1 路径 A) |
| A7 | 全局状态 9 处 | `resolve.rs` 三个 OnceLock 等 | 重构中顺带(M1 Runtime) |

## B. 可靠性与资源

| # | 位置 | 证据 | 处置 |
| --- | --- | --- | --- |
| B1 | 65 处锁 `expect/unwrap` | 锁中毒连锁 panic | 立即修 |
| B2 | `core/src/tracing/processor.rs:10,89-97` | 默认安装的 `InMemoryProcessor` 有 4 个无上限 Vec | 立即修(先写复现测试) |
| B3 | `macros/src/lib.rs:27-49` | 未知属性被 `_ => {}` 吞掉,`timeout = 5` 编译通过却不生效 | 立即修 |
| B4 | `macros/src/lib.rs:200,203` | schema 序列化用 expect | 立即修 |
| B5 | `openai/src/transport.rs:88`、`mcp/src/http.rs:58` | 按 chunk `from_utf8_lossy`,多字节字符被切开后变 U+FFFD | 立即修 |
| B6 | `openai/src/chat_completions/mod.rs:210`、`responses.rs:196` | 流提前断开被当成成功(Responses 合成 completed 且 usage 为 0) | 立即修 |
| B7 | `mcp/src/stdio.rs:89-102,224` | 写 stdin 无 timeout 且持锁,`close()` 抢同一把锁,走不到 kill | 立即修 |
| B8 | `mcp/src/http.rs:94` | 默认跟随 10 次重定向;自定义头跨 host 转发,307/308 重放 POST | 立即修 |
| B9 | `mcp/src/http.rs:236,258` | `notify` 与 close 的 DELETE 无超时,Client 无默认超时 | 立即修 |
| B10 | `openai/src/chat_completions/body.rs:31` | `if let Ok(schema)` 吞掉 schema 错误,Responses 路径却会报错,行为不一致 | 立即修 |
| B11 | `core/src/memory/compaction.rs:384-390,200` | 摘要失败后每次 `add_items` 都重试,摘要调用无 timeout、无退避 | 立即修 |
| B12 | `openai/src/stream_layout.rs:102,190-191` | message 与 call 的 output_index 算法不一致,疑似冲突 | 先写测试验证,确认后修 |
| B13 | `mcp/src/stdio.rs:150`、`openai/src/transport.rs:75`、`mcp/src/http.rs:53` | 单行/SSE 缓冲无上限 | 重构中顺带 |
| B14 | `openai/src/transport.rs:111` | 非 JSON 的 `data:` 行(保活)使整个流失败 | 重构中顺带 |
| B15 | `mcp/src/stdio.rs:114-122` | 只 kill 直接子进程,`npx`/`uvx` 的孙进程成孤儿 | 重构中顺带 |
| B16 | `openai/src/chat_completions/mod.rs:309` | `tool_calls[].index` 缺失回落 0,并行调用被合并 | 重构中顺带 |
| B17 | `openai/src/responses.rs:173-179` | `response.failed` 丢 `code`,`retry_advice` 拿不到可重试信号 | 重构中顺带 |
| B18 | `openai/src/lib.rs:61` | `reqwest::Client::new()` 无 connect 超时 | 重构中顺带 |
| B19 | `core/src/memory/compaction.rs:381-384` | 摘要期间持有写锁(网络调用),同会话写入全部排队 | 重构中顺带 |
| B20 | `core/src/context.rs:42-47`、`compaction.rs:330` | token 估算每次序列化全部历史,base64 图片被当文本计入 | 重构中顺带 |
| B21 | `core/src/error/mod.rs:163,166` | `Status/Connection/Transport` 三个变体 Display 相同 | 重构中顺带 |
| B22 | `core/src/error/mod.rs:259`、`payload.rs:100-109` | `OutputGuardrailTripwire` 约 120-136B,`ModelStatusError` 约 100B,`result_large_err` 根因 | 重构中顺带(装箱这两个载荷) |

## C. 重复

| # | 位置 | 证据 | 处置 |
| --- | --- | --- | --- |
| C1 | `chat_completions/mod.rs:114-124,163-177` 与 `responses.rs:60-72,95-107` | 4 个方法重复请求样板 | 重构中顺带 |
| C2 | `chat_completions/mod.rs:180-198` 与 `responses.rs:110-127` | 服务器忽略 stream 时的整包回退几乎逐行相同 | 重构中顺带 |
| C3 | `conversation_id` 的 Unsupported 校验与文案 | 两份 | 重构中顺带 |

## D. 接受(记录在案)

- `mcp/src/stdio.rs`:`kill_on_drop(true)` 与 `close()` 的 2 秒后 kill 基本合规;`stderr` 继承与 Python 默认一致。
- `InMemorySession` 为 `std::sync::Mutex`,临界区同步、无持锁 await。
- 同一 session 上两个并发 run 会交错,与 Python 语义一致,文档注明。
- `Session::replace_items` 默认实现先 clear 再 add,注释已说明,内置实现已覆盖。
- Chat 路径只读 `choices[0]`,`n>1` 被忽略;`response.incomplete` 不暴露 `incomplete_details`。
- 流式解析未发现 unwrap/越界。

## M0 修复顺序

1. 特征化测试:`InMemoryProcessor` 增长、MCP/stream 的 B5/B6/B7/B8 复现测试、`stream_layout` 的 B12。
2. 机械修复:B1(锁)、A3(别名)、A4(死代码)、B3/B4(宏)。
3. 结构拆分:A1、A2,以行为测试和 parity 为护栏。
4. 逐项修 B2、B5-B11。
5. 全量验收命令 + bench 对比基线。

## M0 修复结果(2026-10-19)

| 项 | 起点 | 现在 |
| --- | --- | --- |
| 非测试 `lock().expect/unwrap` | 65 | 0(改为 `unwrap_or_else(PoisonError::into_inner)`,临界区都是简单集合操作,中毒后状态仍一致) |
| `allow(clippy::too_many_arguments)` | 4 | 0(`RunRecord`/`FinalizeEnv`/`ToolTurn`/`HandoffTurn` 收参) |
| `allow(unused_assignments)`、非测试 `allow(dead_code)` | 1 / 2 | 0 / 0 |
| `run_loop_inner` | 约 640 行、7 参数 | 约 35 行;循环体拆入 `run/turn_loop.rs` 的 `TurnLoop`,每个阶段一个方法 |
| `cargo test --workspace --all-features` | 288 | 305 通过 / 0 失败 |
| `cargo test --no-default-features` | - | 229 通过 / 0 失败 |
| bench(`twenty_tools_ten_turns`) | 347 µs | 345 µs,其余场景在噪声内 |
| clippy / fmt / rustdoc `-D warnings`、`check_module_deps`、`check_runner_api`、`sync_vendor --check` | 通过 | 通过 |
| Rust 侧 `parity_scenarios`(对 golden) | 通过 | 通过 |

已修条目:A1、A2、A3、A4、B1、B2、B3、B4、B5、B6(降级为告警,见下)、B7、B8、B9、B10、B11、B12。
所有修复都带回归测试(`InMemoryProcessor` 有界、宏属性、UTF-8 跨 chunk、MCP 跨源重定向、压缩冷却与超时、`output_index` 冲突)。

### 与计划的偏差(如实记录)

- **"无单函数 > 80 行"不再作为验收指标**(用户决定):硬拆会破坏内聚性,不再处理。
- **B6 没有改成报错**:Responses 路径在"网关不发 `response.completed`"时合成终结事件并 `warn!`。这是 Rust 对 Python 标准行为的**有意放宽**(Python 的 Runner 会抛 `ModelBehaviorError`),已登记为 `DEVIATIONS.md` D-049。
- **Python oracle 已在 `.venv`(Python 3.12)中跑通**:`.venv/bin/python conformance/run_parity.py --check` 全部 `golden unchanged`,`cargo test --test parity_scenarios` 通过。
- **`A5`(`catch_unwind`)未动**:按计划放到 M1。
- **B13-B22 未动**:均为"重构中顺带"。

### 新增发现(扫描时顺带)

- B12 的修复使消息先于工具调用时的 `output_index` 与 Python 一致;原先两者会都落在 0。
- MCP HTTP 默认只同源跟随重定向,这是行为变化(安全收紧),需写入 CHANGELOG。

## 上游评估:0.23.1 → 0.24.0(2026-10-15 发布)

来源:GitHub Releases 页面。该页对 0.24.0 的 Bug Fixes 列表被截断("Read more"),下表只覆盖可见部分;
正式跟进前需对 `vendor` 升级后的 diff 再核一遍。

| 上游变更 | 对本项目的影响 | 处置 |
| --- | --- | --- |
| feat: 在转换后的 MCP 工具上暴露 annotations (#5356) | `mcp` crate 的工具转换未带 annotations | 低优先级,M3 前评估 |
| feat: 支持运行级工具异常格式化 (#5326) | 本项目已有 `RunConfig.tool_error_formatter`,但只覆盖 `approval_rejected`/`tool_not_found`(D-009)。上游把范围扩到运行级异常 | 需对照上游实现后决定是否扩展 |
| fix: 兄弟工具失败时保留已完成工具的输出 (#5332);回调失败后保留工具输出 (#5318) | **可能影响行为**:并行工具批中一个失败时,其余已完成输出是否保留 | M1 前用特征化测试确认本项目现状 |
| fix: 跨 RunState 恢复保留 function-call 别名 (#5358);保留 agent 归属 (#5362);已完成的 stop 工具结束恢复时拒绝待处理输入 (#5366) | 触及 RunState 恢复语义,与 M1 的"agent 图指纹校验"相关 | 并入 M1 HITL 工作,先写对照测试 |
| fix: 尊重调用方 trace 内的 Runner tracing 关闭 (#5296) | 对应本项目的 `tracing_disabled` 语义 | 核对 |
| fix: 自动压缩因待审批而推迟 (#5363);OpenAI 会话写入分批 (#5334);Redis 压缩历史原子替换 (#5354) | 对应上游 `OpenAIResponsesCompactionSession`/Redis,本项目不移植(见 `compaction.rs` 模块说明) | 不跟进 |
| sandbox / voice / realtime / 依赖升级 | 本项目非目标 | 不跟进 |

结论:0.24.0 没有已知的破坏性变更。**不立即升级 pin**;第 3、4 行涉及并行工具失败与 RunState 恢复,
与 M1 重构直接相关,应作为 M1 前的特征化测试输入,在 M1 中一并对齐后再升级到 0.24.x。
此前版本(0.22 到 0.23)的 "bind tool approvals to concrete invocations"(#4257)、"scope function
tool approvals to their owning agent"(#5144)已经说明上游正持续收紧审批绑定,与计划中的
"审批摘要绑定"方向一致。
