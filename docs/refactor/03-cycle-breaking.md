# 循环依赖清单与打破手段

核实依据见 [baseline.md](./baseline.md) 第 3、4 节。

| 编号 | 问题 | 手段 | API 变化 | 风险 | 可在单 crate 内先做? |
| --- | --- | --- | --- | --- | --- |
| C1 | `error` 依赖 guardrail、tool_guardrails、tool、retry(6 处) | 载荷类型 `ToolGuardrailFunctionOutput`(含 `ToolGuardrailBehavior`)、`GuardrailFunctionOutput`、`Input/OutputGuardrailResult` 下沉到 types;`default_tool_timeout_error_message`、`retry_after_from_headers` 下沉为纯函数。已核实这些类型不持有 `Agent`,无需泛型化 | 仅模块路径 | 低 | 是 |
| C2 | `model`、`model/openai` 依赖 `run::DefaultOpenAiApi` 及全局 getter/setter | 枚举与全局状态移入 openai 扩展;`RunConfig` 不再持有。过渡期可先移到 `model/` 下 | 路径变化;`RunConfig` 字段若有则移除 | 低 | 是(先移到 `model/`) |
| C3 | `agent` 依赖 `run::build_agent_as_tool` | `as_tool` 改由 runtime 提供扩展 trait,或 core 定义 `AgentRunner` trait,runtime 实现并注入 | `agent.as_tool(..)` 调用路径变化 | 中 | 部分(先抽 trait) |
| C4 | `tool` 依赖 `RunState`(仅 `ToolResult.nested_state` 与 `interrupted(..)`) | 方案 1:`RunState` 数据结构下沉到 core,执行逻辑留 runtime;方案 2:`nested_state` 改为不透明载体。倾向 1,需先核查 `RunState` 字段是否都可在 core 表达 | `ToolResult` 或 `RunState` 路径 | 中 | 否(需核查后定) |
| C5 | `context` 依赖 `run::{CallModelData, CallModelInputFilter, ModelInputData}` | 三个纯数据/过滤器类型下沉到 core 的 `model_input` 模块 | 路径变化 | 低 | 是 |
| C6 | `agent` 依赖 `mcp` | core 保留 `McpServer` trait、`McpConfig`、`mcp_function_tools`;client、transport、manager 移入 mcp 扩展。需先按契约与实现拆 `mcp/mod.rs` | 路径变化 | 中 | 是(先拆文件,再拆 crate) |
| C7 | `memory/compaction` 依赖 context、model、tracing | 与 context 一起成为 context 扩展 | 路径变化 | 低 | 是 |
| C8 | `guardrail` 依赖 `tracing` | tracing 独立为 L1,只依赖 types | 无 | 低 | 是 |
| C9 | `pyjson` 为 `pub(crate)`,被 handoffs/history 与 run 使用 | 放入 types,`#[doc(hidden)]` 公开,作为内部支撑 | 新增 hidden 公开项 | 低 | 是 |
| C10 | `model/mod.rs` 默认 provider 直接构造 `OpenAIProvider::from_env()` 并注册 `openai*` 前缀 | core 的 `MultiProvider` 提供前缀注册;默认 provider 装配上移到 facade(或 runtime 以注入方式),取决于 D1 | `default_model_provider()` 位置变化;无 openai 时的行为需保持 | 中 | 部分 |
| C11 | 宏生成代码引用 `openai_agents::schemars/serde` | 宏支持 `crate = "..."` 覆盖,默认指向 facade;各 crate 内测试显式指定路径 | 宏新增可选参数 | 低 | 否(拆 crate 后才能验证) |

## 补充问题

- **`wire_events.rs`** 不是 openai 专属:`run/errors.rs`、`memory/history.rs`(只用 `FAKE_RESPONSES_ID`)、`scripted.rs` 也使用。拆分时通用常量/构造留 types 或 core,openai 专属部分归 openai。
- **`openai_retry_advice`**(`retry.rs:895`)属于 openai 专属,应移出 retry。
- **`run_loop_inner`** 约 1140 行单函数,拆分需逐段评估借用与状态,不是机械切文件;安排在单 crate 阶段,并以行为测试保护。

## 建议处理顺序

1. 低风险且纯类型移动:C1、C2(先移到 `model/`)、C5、C8、C9。
2. 先拆文件再拆 crate:C6、C7。
3. 需要设计决策:C3、C4、C10(见 05-open-questions)。
4. 拆 crate 后验证:C11。
