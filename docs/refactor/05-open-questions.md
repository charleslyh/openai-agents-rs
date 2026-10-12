# 决策记录与待决策问题

## 已决策

| 编号 | 决策 | 说明 |
| --- | --- | --- |
| D1 | a:facade 装配默认 provider,core 不知道 openai | C10 的落地方式 |
| D2 | a:`testing` 独立 crate(见下方说明,条件是核心单元测试不依赖它) | 已核实当前 `src/` 无内联测试使用 `ScriptedModel` |
| D3 | T1:tracing 保持 core 内的独立模块,依赖只允许 `usage`,由依赖检查脚本强制;以后需要再抽 crate | 取代原"独立 crate"的选择 |
| D4 | `mod.rs` 布局 | 沿用倾向 |
| D5 | 目录不带前缀(`crates/core`、`crates/openai` ...),仅 package name 带 `openai-agents-` 前缀 | |
| D6 | facade 默认 feature 含 `openai`,不含 `mcp` | 需同步改 examples、tests、CI 的 `required-features` |
| D8 | `conformance/` 存放 Python oracle、parity 场景、golden | 沿用倾向 |
| D9 | 方案 A:粗粒度,常用功能合并在同一 crate,避免过细拆分 | 取代原先 B 的推荐 |
| D10 | 不发布到 crates.io | P4 的发布配置(`exclude`、发布顺序)取消;package name 前缀保留 |
| D12 | `as_tool` 保持在 core,不引入新 trait | 见下方分析 |

## 受 D9 影响而失效或降级的项

| 项 | 变化 |
| --- | --- |
| D7(retry 归属) | 失效:没有独立 types crate,retry 整体留在 core;仅 `openai_retry_advice` 移出到 openai crate |
| D11 / C4(`RunState` 与 `tool.rs`) | 失效:`RunState` 与 `tool` 同在 core,无需处理 |
| C3(`as_tool` 与 run) | 失效:agent 与 run 同在 core,无需拆 |
| C1(`error` 反向依赖) | 降级为可选整理:不再是拆 crate 的前置条件 |
| C8(`guardrail` -> `tracing`) | 方向合法(core -> tracing),保留 |

仍必须解决的环:C2、C5(可顺手)、C6、C7、C10(目标是扩展 crate 单向依赖 core)、C11。

## D2 说明:什么是"双份类型"

`cargo test` 编译 core 的内联单元测试(`#[cfg(test)]`)时,会把 core 本身编译成一个"测试版本"。若这些测试 `use` 了依赖 core 的 `testing` crate,`testing` 链接的是另一份"库版本"的 core。于是测试里的 `Agent`、`Model` 与 `testing::ScriptedModel` 实现的 `Model` 是两个不同的类型,编译会报 trait 不匹配。

- 只发生在 core 自己的内联单元测试里。
- `tests/` 下的集成测试不受影响,因为它们把 core 当外部库使用,只有一份。
- 现状核实:`src/` 中除 `model/scripted.rs` 自身外,没有任何内联测试使用 `ScriptedModel`(其余只在注释里出现),`#[cfg(test)]` 仅在 `memory/history.rs`、`strict_schema.rs`、`mcp/*`、`model_settings.rs`、`model/openai/mod.rs` 中,且不依赖它。
- 约束:core 内联测试不得使用 `testing`;需要脚本模型的测试一律写成集成测试(放 facade 的 `tests/`)。该约束写入 `core/AGENTS.md`。
- 结论:D2 选 a 可行。

## D12 分析:`as_tool`

`build_agent_as_tool` 只是一个闭包:调用 `Runner::run` / `Runner::run_state`,中断时调用 `result.to_state()`。它本身属于运行时能力。

- 方案 A 下 `agent` 与 `run` 都在 core,不存在跨 crate 环,保持现状 `Agent::as_tool -> run::build_agent_as_tool` 即可,零 API 变化。
- 只有将来把 runtime 从 core 拆出时才需要引入 `AgentRunner` trait 或扩展 trait;现在引入属于过度设计。
- 结论:保持现状;在 core 内把它放进 `run/as_tool.rs`,让 `agent.rs` 不再因它反向依赖 `run/tools.rs` 的内部文件。

## D3 与 D9 的冲突(已决议:选 T1)

原 D3 选了"tracing 独立成 crate",但这与 D9 的"常用功能放同一 crate"原则有张力:

1. `tracing` 依赖 `usage::Usage`(`add_usage`、`turn_usage_json` 等)。要让 tracing 成为 core 的下层 crate,`Usage` 必须在 tracing 里或在更下层 crate,否则出现 tracing -> core -> tracing 的环。
2. run loop、guardrail 都无条件使用 tracing,不存在不带 tracing 的 core,可剥离性测试不通过。

可选做法:

| 选项 | 做法 | 评价 |
| --- | --- | --- |
| T1 | tracing 保持 core 内的独立模块,依赖只允许 `usage`,用依赖检查脚本强制 | 与 D9 一致,推荐;以后需要可再抽 |
| T2 | tracing 独立 crate,`Usage` 一并迁入该 crate(core re-export) | 满足 D3,但 `Usage` 放在 tracing 里语义别扭 |
| T3 | 再加一个 `types` crate | 与 D9 冲突,不推荐 |

决议:T1。P2 的依赖检查脚本中加入规则:`tracing` 模块只允许 `use crate::usage`(及外部 crate),反向依赖不受限。
