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
| A6 | `Arc::new(agent.clone())` 共 11 处 | 热路径深拷贝 | 已修(M1 路径 A):循环内全部共享 `Arc<Agent>`,只剩公共入口每次运行 1 次、`as_tool` 构造时 1 次;`twenty_tools_ten_turns` 345µs → 265µs(-20%) |
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
| `cargo test --workspace --all-features` | 288 | 305 通过 / 0 失败(进入 M1 前);322 通过 / 0 失败(`Runtime` 落地后) |
| `cargo test --no-default-features` | - | 229 通过 / 0 失败(进入 M1 前);246 通过 / 0 失败(`Runtime` 落地后) |
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

## M1 进展(持续更新)

| 项 | 状态 |
| --- | --- |
| D1(context 是否泛型化) | 已决议:不泛型化。`docs/refactor/09-d1-context-generics.md` |
| `Extensions`:按类型取值的业务数据 | 完成(`643c8bf`)。`RunOptions::with(v)` / `ctx.get::<T>()` / `ctx.require::<T>()`,`RunOptions::context` 与 `context::<T>()` 保留并共用同一存储 |
| `RunConfig` 分组 options | 完成(`cb225d6`)。24 个平铺字段拆为 `ModelOptions` / `TracingOptions` / `SessionOptions` / `HandoffOptions` / `ToolOptions` 加两个护栏列表,均可链式构造;`Debug` 改为派生 |
| `RunOptions` 链式方法 + `non_exhaustive` | 完成(`c9a08e3`)。`RunOptions::new().max_turns(5).temperature(0.2).session(s)`;`tune()` 原地改单个 model setting |
| `Runtime`(provider 注册表 + 阻塞 runtime) | 完成。9 处全局简化:provider 三处、阻塞 runtime 一处、facade 的 `Once` 一处合并进 `Runtime`;`set_default_model_provider_factory` 降为 `Runtime::global()` 的薄封装 |
| 追踪全局开关与 processor | **未完成**。仍在 `tracing` 模块的 `static`;`Runtime::tracing_disabled` / `set_tracing_disabled` 只是转发。迁移需要把 `Runtime` 穿到每个 `agent_span()` / `turn_span()` 调用点,单独一步做 |
| handoff history wrappers | **未完成**。仍在 `handoffs::history` 的 `static`。需要 `HandoffOptions` 或 `Runtime` 传递,与 `nest_handoff_history` 的签名一起改 |
| `Tool` trait + `Tool::typed` + `ToolOutput` 隐藏内部字段 | 完成。`tool.rs`:`Tool::{spec,policy,requires,call}`;`FunctionTool` 是默认实现(`impl Tool for FunctionTool`);`FunctionTool::from_tool` 把任意 `impl Tool` 适配成 `FunctionTool`,跑同一条执行路径;`FunctionTool::typed`/`typed_with_result` 用 `schemars` 生成严格 schema,免宏构造;`ToolResult` 改名 `ToolOutput`,`interruptions`/`nested_state` 两个字段降为 `pub(crate)`(`#[non_exhaustive]`),外部只能用 `::output`/`::interrupted` 构造、`is_interrupted()`/`value()` 读取 |
| 工具/模型中间件 | 完成。新增 `tool_middleware.rs`(`ToolMiddleware` trait + `FunctionTool::with_middleware`,onion 组合,可短路)与 `model_middleware.rs`(`ModelMiddleware` trait,挂在 `ModelOptions::middleware`,在 `call_model_input_filter` 之后按添加顺序执行)。仅覆盖请求改写;包装响应/流是后续工作,登记在 `DEVIATIONS.md` |
| `Tool::requires` + 启动期校验 | 完成。`extensions.rs` 新增 `TypeRequirement`;`FunctionTool::require::<T>()` 声明依赖,`run/agent_loop.rs` 的 `validate_tool_requirements` 在第一个 turn 前遍历可达 agent 图(含 handoff 目标)逐一检查,缺失类型报错带 agent 名、工具名、类型名、修复提示;不声明的工具不受影响(仍是运行时失败) |
| 类型化 `Reasoning` + 模型名后缀 | 完成。`ModelSettings::reasoning` 由 `Option<Value>` 改为 `Option<Reasoning>`(`effort`/`summary` 为开放枚举,未知值原样往返;`extra` 承载 `mode` 等未建模字段;`TryFrom<Value>` 兼容旧写法);`RunOptions::reasoning(..)` 链式设置。适配层:Responses 发 `reasoning` 对象;Chat Completions 补上 Python 已有的 `reasoning_effort` 映射(此前缺失,示例 25 只能走 `extra_args`)。`"o4-mini:high"` 这类后缀在按名解析时剥离并填入 effort(显式设置优先,X-006) |
| `Capability` 具名组合 | 完成。`agent/capability.rs`(放在 `agent` 模块内,不扩大已登记的 8 模块环):`Capability` trait + `CapabilityParts`(tools / instructions / 输入输出护栏 / 默认 model settings / requires)+ `CapabilityParts::named(..)`;`Agent::capability(..)` 挂载时计算一次 parts,与 agent 自身字段分开存放,在读取点合并(`enabled_tools`、`resolve_instructions`、输入/输出护栏、`prepare_turn` 的 settings、启动期 requires 校验),因此 builder 调用顺序无关、"整体挂载 = 手工给零件"有等价测试(`behavior_capability.rs`)。**未含**:hooks(agent 只有一个 hooks 槽,需组合器);5 个内置 bundle 随 M3/M4 的 crate 落地 |
| 去克隆(路径 A) | 完成(`e48c5ac`)。见 A6 |
| 术语收敛为 resolution | 完成(`d1a0ca8`)。删除重复的 `RunState::approval_status`;`McpServer::needs_approval_for` 改名 `needs_resolution_for`;人工审批专属名(`approve`/`reject`、MCP `RequireApproval`、`ToolErrorKind::ApprovalRejected`、`StickyDecision`)按 D-048 保留 |
| `prelude` | 完成。`core::prelude` 35 个名字(后加 `async_trait`)(按 examples/demo 的导入频次选取:Agent/Runner/RunOptions/RunResult/RunState、工具与 resolution、护栏、handoff、session、model/provider、Reasoning、Runtime、Extensions);facade `prelude` 另加 `#[function_tool]` 并换成 facade 的 `Runner`。README 快速开始与示例 01 已改用。**未含**:把 `ResolutionStore`、`StickyDecision`、序列化辅助等移出顶层(随 examples 迁移一起做) |
| `get_model` 改 async | 完成。`ModelProvider::get_model` 为 `async fn`(`async_trait`,root 与 prelude 重导出);`resolve_model` 随之 async,在 turn 内与 Python 同一位置 await;`CachedProvider` 不跨 await 持锁,并发未命中时以先写入者为准共享同一实例(有测试)。X-008 |
| HITL `resolve_with` | 完成。`run/hitl.rs`:`RunOptions::resolve_with(闭包)` 在 `Runner::run`/`run_state`/`run_blocking` 内循环"暂停 → 回调 → `RunState::resolve` → 恢复",复用原有 resolution 路径(嵌套 `as_tool`、`max_turns`、session 行为与手写循环一致);流式运行设置了 resolver 时返回 `UserError`(fail-closed)。示例 12 的审批样板从约 34 行降到 7 行,显式持久化流程保留在 `HITL_MODE=persist`。X-009 |
| 审批摘要绑定 | 完成。`ResolutionStore` 新增私有 `bindings`:经 `RunState` 给出的决议记录 `(agent, tool, arguments)`,`plan_tool_calls` 改用 `status_for_call` 校验,不匹配则视为未决(重新暂停或走 sticky)并告警。state JSON 中 `by_call.<id>.bound` 为新增可选字段,旧状态照常可读,不升 schema 版本。存储完整参数而非摘要:避免 core 引入哈希依赖,且比摘要更严格;防篡改由 M3 的 HMAC 负责。X-010 |
| RunState agent 图指纹 | 完成。`run/resolve.rs::agent_graph_fingerprint`(FNV-1a 64,长度前缀,按名排序:agent 名、本地与 capability 工具名、handoff 工具名;不含 MCP 与 `is_enabled`),只在暂停时计算一次(热路径无开销);`run_state`/`run_state_streamed` 统一走 `check_resumable`(名字 + 指纹),不一致返回 `UserError`;`RunState::allow_agent_graph_change()` 放行。state JSON 新增可选 `agent_graph`,旧状态不校验。X-011 |
| 护栏 Verdict 三态 | 完成。`GuardrailFunctionOutput` 改为 `output_info` + `verdict: Verdict`(`Pass`/`Fail{reason, severity}`/`Transform`),`pass`/`trip` 保留,`tripwire_triggered` 由字段改为方法;`run_group` 用 `FuturesUnordered` 串行取结果,遇 High/Critical 失败即丢弃其余 future(取消),结果仍按声明顺序返回;`single_transform` 对"同一内容两个 Transform"fail-closed;仅 `run_in_parallel(false)` 的输入护栏可 Transform(并行时报错而非静默失效),多个 Transform 报错;输入变换同时替换 session 待存输入(第一轮 flush 让位于阻塞护栏);输出变换只改 `final_output`(模型消息原文保留)。`RunState` JSON 增加 `verdict`,保留 `tripwire_triggered`。内置护栏随 M4 的 guardrails crate。X-012 |
| 迁移面(示例/demo/tests/benches) | 完成。示例不再给嵌套字段赋值(`RunOptions` 新增 `.tracing/.tools/.session_opts` 三个闭包方法,`ToolExecutionConfig::new().concurrency(..)`);07/08/12/16/17/19 的工具改用 `FunctionTool::typed`(`#[derive(Deserialize, JsonSchema)]` 生成 schema 并解析参数),19/07 用 `()` 表示无参数;新增示例 `26_capability`(整体挂载 = 手工给零件)与 `27_verdict_transform`(脱敏后继续);`benches/run_loop.rs` 改用 `RunOptions::new()`;demo/web 的 `Tick::Event` 因 `AgentsError` 变大而装箱。`MIGRATION-FROM-PYTHON.md` 补上 Rust 独有项(resolve_with、绑定决议与图指纹、Verdict、typed、Capability、async get_model)并改为纯链式示例。示例 22/06 的显式导入保留(非阻塞) |
| Provider 预设表(M2) | 进行中。新建 `crates/providers`:`spec.rs`(预设表 + `Capabilities`/`Capability` 位集 + 品牌前缀推断)、`provider.rs`(`Provider` builder + `validate_endpoint` https/loopback 校验)、`registry.rs`(`ProviderRegistry::from_env` 发现 + `ProviderRouter` 前缀/推断/默认路由)。依赖 core + openai(唯一允许依赖另一个扩展的扩展,已在 `check_module_deps.py` 登记)。**已完成**:`guard.rs` 的 `GuardedModel` 包装 provider 返回的 model,在请求发出前按矩阵校验(reasoning / JSON schema / tools / parallel / store / 图片 / PDF;仅当 provider 缺该能力时才扫描输入,常规路径零开销),不支持则 `ModelError::Unsupported`(`UnsupportedCapability` 文本)。**已完成**:`catalog.rs` 模型目录(仅收录历史确定的模型:Retired/Legacy/Current)+ `advise()`(下线给替代、legacy 给当前、近邻给 did-you-mean),router 在 `get_model` 时 `tracing::warn!` 一次;未知 id 一律放行。**已完成**:`offline.rs`(`OfflineModel` + `OfflineProvider`,`AGENTS_OFFLINE=1` 优先于任何 key,caps=full 因为不发请求);本地预设需 `enable_env`(Ollama 用 `OLLAMA_HOST`)才注册,避免无配置时误连 localhost;facade 新增 `providers` feature 与 `openai_agents::run()`(`zero_config_options()` 取 provider + 模型名 `AGENTS_MODEL`/`OPENAI_MODEL`/preset 默认/`gpt-4o-mini`),core 新增 `RunOptions::provider()`;新增示例 28 与 facade 测试。**已完成**:守卫改为 `GuardedProvider`,在 `Provider::build()` 处包装(单 provider 与 registry 路由行为一致);facade 新增 `behavior_provider_presets.rs` 契约测试(5 个):每个 Chat 预设的 reasoning 走 `reasoning_effort`、openai 走 `reasoning` 对象、ollama 无 reasoning 时请求被拒且不发出、路径正确、矩阵自检。**未做**:Anthropic 原生适配。facade feature 档位:现有 `openai`(默认)/`mcp`/`sqlite`/`providers` 已各自独立可选,不再另加 minimal/standard/full 三档(避免改默认特性面)。X-013 / X-014 / X-015 |
| 会话契约套件 + RunStateStore(M3) | 完成。core 新增 `session_contract`(tier 3,已登记 TIERS):空会话/顺序/limit 取尾部/pop_item/replace_items(含空)/并发读不见半写历史/clear/1MiB 往返 + `assert_concurrent_writes(Arc<dyn Session>, n)`;`memory::state_store` 新增 `RunStateStore` trait、`RunStateMigration`、`is_supported_schema`/`finish_load`。`InMemorySession`、`SqliteSession`(给它补了 tests/)、`FileSession` 均已跑通。新建 `crates/sessions`(依赖 core,已登记 CRATE_DEPS):`FileSession` 用进程内 Mutex + `fs2` 跨进程 flock,重写走 tmp+rename+fsync,append 走 append+fsync;`file_name()` 拒绝空/分隔符/点开头/控制字符/超长(其余替换为 `_`),读取前 `symlink_metadata` 拒绝符号链接。**已完成**:`state_file.rs` 的 `FileRunStateStore`:每 key 一个 `.state.json`(envelope:`$schemaVersion`/`state`/`expires_at`/`mac`),HMAC-SHA256 对 state 的规范 JSON 签名(`AGENTS_STATE_KEY`,比较用常数时间),无 key 时 warn 并继续工作;签名缺失或不符 → `UserError`(不是 `None`);TTL 过期读为不存在并删除;原子写(tmp+rename);目录级 `.state.lock` 跨进程锁;`load` 走 `finish_load` 应用迁移钩子。**已完成**:`crates/sessions` 的 `redis.rs` 与 `postgres.rs`,两者均跑通同一契约(对真实 redis-server / postgres 验证过,CI 无服务时 skip)。共用 `state.rs`(envelope:签名/HMAC、TTL、迁移钩子、常数时间比较)与 `validate_id`;Redis 用 list + MULTI 保证 replace 原子、可选 TTL;Postgres 用行 + 事务,全部值绑定、表名经 `validate_table` 校验(字母数字下划线、≤63),`create_table` 用 `pg_advisory_xact_lock` 避免 `IF NOT EXISTS` 自身竞态(实测并发创建会撞 `pg_class_relname_nsp_index`)。sqlx 0.9 要求动态 SQL 标记安全,统一经 `statement()`(唯一做该标记处)。facade 新增 `redis`/`postgres` feature。**未做**:OTLP 导出、tracing 桥接、示例。X-016 / X-017 |
| 可观测(M3) | 进行中。新建 `crates/observability`(依赖 core,已登记 CRATE_DEPS):`export.rs` 把 `Trace`/`Span` 转成扁平 JSON(`kind`/`span_kind`/`name`/`span_id`/`parent_id`/时间戳/`error`/`usage`),不含请求体、工具参数与模型输出(span 本身不持有);`batched.rs` 的 `BatchedProcessor` 实现 `TracingProcessor`,回调内只做序列化 + `try_send` 到有界 `sync_channel`,写盘在独立 std 线程完成(不用 tokio 任务:`force_flush` 是同步等待,单线程 runtime 下后台任务无法推进会死锁);队列满丢弃并计数(`dropped()`,第 1 次与每 100 次 `warn!`),`force_flush` 最多等 `FLUSH_TIMEOUT`=2s;`sinks.rs` 提供 `ConsoleSink`/`JsonlSink`(每批 append 打开,写失败计入 dropped);`metrics.rs` 提供 `Metrics`/`MetricsSink`/`PriceTable`,未知模型计 0 不估算。测试 11 个(7 集成 + 2 单测 + 2 doctest)覆盖:事件到达 sink、满队列不阻塞(带耗时断言)、force_flush、失败 sink 计数、JSONL 一行一事件、坏路径构造期报错、8 线程 × 25 span 全计数。**已完成**:`otel.rs` 的 `OtelSink`,把结束的 span 转成 OpenTelemetry span 写入进程已安装的 tracer(OTLP 亦可),只依赖 opentelemetry API(不引入 exporter/gRPC/运行时);SDK 自己的 span_id/parent_id 作为 `agents.*` 属性携带(OTel API 自行分配 id);start 事件与 trace 事件不产生 span(OTel 没有 trace span,start 无结束时间)。facade 新增 `otel` feature。**决定推迟**:自带 OTLP exporter(会把 gRPC 栈与 tokio 运行时带进构建,且批导出器与"写盘在 std 线程"冲突;应用侧装管线即可,代价仅一行)。**已完成**:`bridge.rs` 的 `TracingSink`,span 事件转成 `tracing` event(target `openai_agents`;结束 DEBUG、失败 WARN、开始/trace TRACE),只发白名单字段(无正文);因导出在写线程、无进入中的 span,只能发 event,父子关系靠 `span_id`/`parent_id` 字段。M3 可观测项全部完成。X-018 |

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
