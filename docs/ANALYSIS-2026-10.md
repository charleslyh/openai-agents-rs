# 深度分析与改进建议（2026-10）

对象：`openai-agents-rs`（Python `openai-agents` v0.23.1 的 Rust 移植，7 个 crate，约 20.3k 行源码，262 测试）。
方法：静态阅读源码 + 架构文档 + CI 配置 + 与同类开源项目的横向对比。行号/路径均为 2026-10-16 工作区快照。

---

## 0. 一句话结论

**工程质量（A-）明显高于架构设计（C+）和生态定位（D）。** 项目把"与 Python 行为对齐"做到了工业级（vendor pin、Python oracle parity、wiremock 快照、模块依赖检查器、61 条编号偏差、coverage ratchet），但公共 API 是**从 Python dataclass 直译**而来的：类型擦除、God Object、8 模块循环依赖、全局可变状态、非惯用的手动 JSON。它目前是"给已经熟悉 Python SDK 的人用的 Rust 版 SDK"，而不是"给 Rustacean 用的 SDK"——这是后续所有问题的根因，也是所有建议的出发点。

---

## 1. 做得好的部分（不要动）

这些是真金白银的资产，重构时优先保护：

| 项 | 证据 | 评价 |
|---|---|---|
| 分层拆 crate 且 core 无重型依赖 | `crates/core/Cargo.toml` 只有 9 个依赖，无 `reqwest`/`rusqlite`；`scripts/check_module_deps.py` 在 CI 强制 tier 方向 | 教科书级的"依赖倒置 + 可剥离性"落地 |
| 行为对齐有 oracle | `conformance/run_parity.py --check`、41+16 个 `chat_convert` 共享样例、`vendor/PINNED_VERSION` = v0.23.1 | 同类项目几乎都没有，是本仓库最独特的竞争力 |
| 偏差有编号、有状态、有理由 | `docs/DEVIATIONS.md` 61 条 D-/B- 编号 | 维护者心智负担大，但对用户价值极高 |
| feature 粒度正确 | `mcp` 默认关闭（D6），`default-features=false` 构建无 HTTP 客户端 | 认真做过成本分析（`docs/refactor/05-open-questions.md` D13） |
| 文档密度高 | 每个 crate 有 `AGENTS.md`，每个公共项有 doc（`#![deny(missing_docs)]`） | 高于绝大多数同类项目 |
| 决策留痕 | `docs/refactor/01..07` 记录方案对比与被否决的理由 | 半年后回看不会失忆 |

---

## 2. 结构与设计问题

### P0-1 领域内核存在 8 模块有向环 —— 违反 ADP（无环依赖原则）· 影响最大

```40:42:docs/refactor/06-progress.md
- 除下面这个环之外,模块图是 DAG。`run`、`result`、`context`、`memory`、`error`、`retry`、`tracing` 等都在环外。
- 唯一允许的环(领域内核,8 个模块):`agent`、`guardrail`、`handoffs`、`lifecycle`、`mcp`、`model`、`tool`、`tool_guardrails`。成因:容器持有回调,回调接收 `&Agent`。
- 检查器强制:任何其他模块进入环、或出现新的环,CI 失败。
```

成因精确（这一点诊断得很好）：回调签名里传了 `Arc<Agent>`。

```50:54:crates/core/src/agent.rs
pub type DynamicInstructions = Arc<
    dyn Fn(RunContextWrapper, Arc<Agent>) -> Pin<Box<dyn Future<Output = String> + Send>>
        + Send
        + Sync,
>;
```

```159:166:crates/core/src/tool.rs
pub type IsEnabledFn = Arc<
    dyn Fn(
            RunContextWrapper,
            Arc<crate::agent::Agent>,
        ) -> Pin<Box<dyn Future<Output = bool> + Send>>
        + Send
        + Sync,
>;
```

问题在于：**"允许一个环"是把 ADP 从原则降级成了配额**。代价是这 8 个模块无法单独编译/单独测试/单独抽取成 crate，任何一次 `Agent` 字段变更都会在这 8 个模块里同时产生级联编辑，而 CI 的环检查器只能阻止环变大，无法阻止环内部继续劣化。

真正的解法是砍掉环的成因，而不是给环发许可证：回调需要的通常只是 agent 的**只读投影**（名字、指令、handoff 描述），不是 `Arc<Agent>` 本身。引入一个放在叶子模块的 `AgentView`（`Arc<str>` 名字、`Option<Arc<str>>` 描述、`&'static` 无关的工具列表），让 `tool` / `guardrail` / `handoffs` 依赖它而不是 `agent`，环即消失（`agent` → `tool` → `view` 单向）。`Agent` 已经是无 `Send` 约束的纯数据容器，这个改造面不大。

### P0-2 `RunConfig` / `RunOptions` 是 God Object —— 违反 SRP、信息专家模式

```185:244:crates/core/src/run/config.rs
pub struct RunConfig {
    pub model: Option<ModelRef>,
    pub model_provider: Option<Arc<dyn ModelProvider>>,
    pub model_settings: Option<ModelSettings>,
    pub tracing_disabled: bool,
    pub workflow_name: Option<String>,
    pub trace_id: Option<String>,
    pub group_id: Option<String>,
    pub trace_metadata: Option<Value>,
    pub trace_include_sensitive_data: bool,
    pub input_guardrails: Vec<InputGuardrail>,
    pub output_guardrails: Vec<OutputGuardrail>,
    pub tool_not_found_behavior: ToolNotFoundBehavior,
    pub tool_execution: Option<ToolExecutionConfig>,
    pub session_input_callback: Option<SessionInputCallback>,
    pub session_settings: Option<SessionSettings>,
    pub tracing: Option<TracingConfig>,
    pub output_guardrail_blocked_message: Option<OutputGuardrailBlockedMessage>,
    pub reasoning_item_id_policy: Option<ReasoningItemIdPolicy>,
    pub tool_name_collision_policy: ToolNameCollisionPolicy,
    pub tool_error_formatter: Option<ToolErrorFormatter>,
    pub nest_handoff_history: bool,
    pub handoff_history_mapper: Option<HandoffHistoryMapper>,
    pub handoff_input_filter: Option<HandoffInputFilter>,
    pub call_model_input_filter: Option<CallModelInputFilter>,
```

24 个公共字段（`RunOptions` 另有 8 个），横跨 5 个关注点：模型/追踪/会话/护栏/工具错误/handoff。后果：

- `Default` 手写 24 行（第 268-297 行），`Debug` 手写且只打印 6 个字段（第 299-314 行），二者必须永远同步；加字段时三处改动、四处易错。
- 用户调试时 `Debug` 输出信息量严重不足（这也是为什么 `Debug` 被手写成只打 6 个字段——字段太多，打全是噪音）。
- 文档页面是一个 24 字段的平面列表，读者无法判断"我要改会话行为应该看哪几个字段"。

`Agent` 同样 17 个公共字段，`FunctionTool` 12 个（其中两个还是 `Vec<ToolInputGuardrail>` / `Vec<ToolOutputGuardrail>` 这类嵌入式子集合）。

### P0-3 Context 类型擦除 —— 用运行时错误换掉了 Rust 最核心的卖点

```15:16:crates/core/src/run_context.rs
/// A user-supplied context value.
pub type ContextValue = Arc<dyn Any + Send + Sync>;
```

```3:7:crates/core/src/run_context.rs
//! The SDK is not generic over the user context: it is carried as
//! `Arc<dyn Any + Send + Sync>` and downcast on demand. This keeps `Agent` / `Runner`
//! free of type parameters at the cost of discovering context type errors at runtime, so
//! [`RunContextWrapper::try_context`] reports a precise error instead of silently returning
//! `None`.
```

注释很诚实，但结论值得商榷：**官方 JS/TS SDK 是 `Agent<TContext, TOutput>` 泛型**（`run(agent, input, { context })` 由编译器校验类型），Python 也有 `Agent[TContext]` 泛型。一个 Rust 移植反而放弃了两个官方版本都有的编译期保证，这是相对上游的**降级**，不是"Rust 语言差异"。

`try_context` 的错误信息做得再好，也只能把 panic 变成"运行到第 3 轮工具调用时返回一个 `UserError`"。而 `Arc<dyn Any>` 的收益（无类型参数）其实可以用默认类型参数同时拿到：`Agent<TContext = ()>` + 保留 `dyn Any` 路径做逃生舱。

### P1-4 工具是"类型擦除闭包"而非 trait —— 违反 OCP，且抬高了学习成本

```97:104:crates/core/src/tool.rs
pub type ToolInvoker = Arc<
    dyn Fn(
            ToolContext,
            String,
        ) -> Pin<Box<dyn Future<Output = Result<ToolResult, AgentsError>> + Send>>
        + Send
        + Sync,
>;
```

`Agent.tools: Vec<FunctionTool>` 是**唯一**的工具形态，没有 `Tool` trait。Python 有 `FunctionTool / HostedTool / McpTool / ComputerTool` 的多态层，这里被压平成"具体类型 + 闭包"。后果：

- 用户无法自定义工具类型（OCP 违反）：想给工具加统一鉴权/限流/审计，只能改 SDK 或在每个闭包里手写。
- `FunctionTool::new` 的备选路径要求**手写 JSON Schema**，且闭包签名是 `(ToolContext, String) -> Result<ToolResult>`——用户自己 `serde_json::from_str` 解析参数（`examples/src/17_hitl.rs` 第 26-39 行就是这个样子）。这与 `#[function_tool]` 的"零样板"体验形成巨大落差，同一个概念两套心智模型。
- `ToolResult` 携带 `interruptions` / `nested_state`（`crates/core/src/tool.rs` 第 67-74 行）——**内部调度概念泄漏进了公共返回值类型**，普通工具作者必须面对与他无关的字段。

### P1-5 热路径上的 `Arc::new(agent.clone())`

```200:205:crates/core/src/tool.rs
    pub async fn resolve(&self, context: &RunContextWrapper, agent: &crate::agent::Agent) -> bool {
        match self {
            Self::Fixed(v) => *v,
            Self::Dynamic(f) => f(context.clone(), Arc::new(agent.clone())).await,
        }
    }
```

`Agent` 含 `Vec<FunctionTool>`（每个含 2 个 `Arc<dyn Fn>` + 2 个 guardrail Vec）、`Vec<Handoff>`、`Vec<Arc<dyn McpServer>>`。每轮、每个工具、每个 handoff、每次动态指令都深拷贝一次（`enabled_tools` / `enabled_handoffs` / `resolve_instructions` 共 7 处）。`FunctionTool` 是 `Clone`（Vec<Arc> 克隆），`Agent` 克隆成本随工具数线性增长，且发生在每个 turn 的热路径上。benchmark 只测了 `run_loop`（`crates/facade/benches/run_loop.rs`），可能掩盖了这一项。

### P1-6 全局可变状态偏多 —— 隐式耦合，测试与多租户的双重陷阱

```42:44:crates/core/src/run/resolve.rs
static DEFAULT_PROVIDER_FACTORY: OnceLock<ProviderFactory> = OnceLock::new();
static DEFAULT_MODEL_PROVIDER: OnceLock<Arc<dyn ModelProvider>> = OnceLock::new();
static MISSING_PROVIDER: OnceLock<Arc<dyn ModelProvider>> = OnceLock::new();
```

另有 `tracing/mod.rs:24` 的 `TRACING_DISABLED: AtomicBool`、`tracing/processor.rs:7` 的全局 processor slot、`handoffs/history.rs:31` 的 `WRAPPERS: Mutex<...>`、`openai/src/default_api.rs:20` 的 `DEFAULT_OPENAI_API: Mutex<...>`、`run/mod.rs:234` 的全局懒启动 tokio runtime。

其中 facade 用 `Once` 隐式安装 provider：

```84:91:crates/facade/src/runner.rs
#[cfg(feature = "openai")]
fn install_default_provider() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        openai_agents_core::set_default_model_provider_factory(default_model_provider);
    });
}
```

"设个环境变量就能跑"的体验确实好，但它意味着：**库在用户不知情时修改了进程级全局状态**；单进程内跑两套不同 provider 的配置会互相干扰；测试之间会相互污染（`mcp_interop` 已经因为共享 OS 资源被迫全局串行，见 `06-progress.md` 第 44-62 行）。对比 `rust-genai`：一切通过 `Client` / `ClientBuilder` 显式注入（`AuthResolver`、`ServiceTargetResolver`、`with_reqwest`）。

### P1-7 过程式 API 与"半成品"的手工构造

`RunConfig` 只提供了 2 个 `with_*` 构造器（`with_tool_error_formatter`、`with_call_model_input_filter`，第 246-265 行），其余 22 个字段和 `RunOptions` 全部只能靠 `RunOptions { run_config: RunConfig { .., ..Default::default() }, ..Default::default() }` 这种**俄罗斯套娃字面量**（`examples/src/33_reasoning.rs` 第 57-61 行）。这是当前 API 学习成本最高的单点：用户要同时理解 `RunOptions`、`RunConfig`、`ModelSettings` 三层容器才能改一个 `temperature`。

### P1-8 命名语义漂移（Python 术语 vs Rust 抽象）

`needs_resolution` / `NeedsResolution` / `NeedsApproval` / `with_needs_approval_fn` / `NeedsApprovalFn` 五套同义符号共存（`crates/core/src/tool.rs` 第 106-142 行），`ToolApprovalItem`/`ToolInterruptionItem`、`approvals`/`resolutions` 亦如此（D-012）。抽象本身（"resolution" = approve / reject / provide_output）比 Python 的 "approval" 更准确，是领域建模的进步；但同义别名不删，等于让每个读者同时维护两套术语表。

### P2-9 未标注 `#[non_exhaustive]`

全仓库 `grep non_exhaustive` = 0 命中。所有公共 struct 都是开放字段 + 开放构造，意味着**未来任何字段增删都是破坏性变更**，0.x 阶段也无法平滑演进。对比：`ModelSettings` 已经是 Python 25 字段的子集（20 个），后续补齐必然破坏构造式；`RunState` schema 已经迭代到 `/4`（`run_state/mod.rs:22`），说明结构仍在变动。

### P2-10 `ModelSettings` 的高频字段仍是裸 JSON

```121:131:crates/core/src/model_settings.rs
    pub reasoning: Option<Value>,
    pub verbosity: Option<Verbosity>,
    pub metadata: Option<Map<String, Value>>,
    pub store: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub include_usage: Option<bool>,
```

`tool_choice` / `truncation` / `verbosity` 做了类型化（很好），但 `reasoning`——**当前最常用的推理模型设置**——是 `Option<Value>`，用户必须写 `json!({"effort": effort, "summary": "auto"})`（`examples/src/33_reasoning.rs` 第 39 行）。而且 Responses 与 Chat Completions 两侧的写法还完全不同（`reasoning` vs `extra_args` 里的 `reasoning_effort`），这个 API 差异本应由 SDK 吸收，现在完全转嫁给了用户。`preserve_raw_usage`、`response_include`、`extra_args` 等也仍是 `Option<Map<String, Value>>`（Python 遗留）。

### P2-11 公共命名空间过宽

`crates/core/src/lib.rs` 顶层 re-export 约 27 组、200+ 个符号，facade 又 `pub use openai_agents_core::*;`。没有 `prelude`。后果：内部概念（`ResolutionStore`、`StickyDecision`、`ItemHelpers`、`Deserialize*`/`Serialize*` 辅助函数、`RUN_STATE_SCHEMA_VERSION`）与用户每日接触的概念（`Agent`、`Runner`、`RunOptions`）平级陈列在文档首页；且 glob 导出后新增符号极易与用户代码冲突。

### P2-12 `RunResult` 里藏着内部快照

```50:51:crates/core/src/result.rs
    /// Internal snapshot used by [`Self::to_state`] when interrupted.
    pub(crate) interrupt_state: Option<InterruptSnapshot>,
```

`pub(crate)` 字段出现在公共 struct 中是必要的妥协，但它说明 `RunResult` 同时承担"结果值对象"和"可续跑状态容器"两个职责。用户可见的后果就是 HITL 流程必须写 while 循环 + `to_state()` + `from_json` + `run_state`（`examples/src/17_hitl.rs` 第 65-98 行，34 行样板），且 `run_state` 还要求 `state.starting_agent_name == agent.name`（字符串匹配，不是类型匹配）。

---

## 3. 可用性与学习成本

按"新人第一次打开这个项目"的路径评估：

| 环节 | 体验 | 问题 |
|---|---|---|
| README 快速上手 | 好 | 11 行可跑，与官方 SDK 一致 |
| 第一个自定义工具 | **差** | `#[function_tool]` 极好；但一旦需要动态参数、`Option`、上下文或运行时构造，就掉进 `FunctionTool::new` + 手写 JSON Schema + 手工 `serde_json::from_str` |
| 改一个 model 参数 | **差** | 需要理解 `RunOptions` / `RunConfig` / `ModelSettings` 三层嵌套 + 两个 `..Default::default()` |
| 理解 HITL | 中 | 概念清晰（interruptions → resolution → resume），但样板 34 行，且术语与 Python 不一致 |
| 找一个类型在哪 | 差 | 200+ 符号平铺，无 prelude，无能力分组 |
| 判断"我该用哪个 guardrail" | 中 | 4 类 guardrail（input/output/tool input/tool output）+ 4 类 Result 类型，命名对称但语义边界需读 DEVIATIONS |
| 排查错误 | 中 | `AgentsError` 12 个变体、文档齐全；但 `result_large_err` 被 allow（63 处），说明错误类型本身就是"大对象" |
| 知道"哪些功能没有" | 好 | `docs/COMPAT.md` + DEVIATIONS 的"out of scope"表非常清晰 |

**学习曲线的形状是"前 10 分钟极陡下降，第 2 小时出现断崖"**：hello/tools/streamed 三分钟跑通；想做任何非示例化的事情（自定义工具形态、类型化 context、非默认 provider、typed reasoning）就立刻撞上类型擦除和手工 JSON。

对比参照（见第 4 节）：Rig 的心智模型是 `AgentBuilder → .prompt()`；`genai` 是 `Client → exec_chat()`；本项目是 `Agent → RunOptions → RunConfig → (ModelSettings|TracingConfig|Session|...)`。前两者是"一个入口 + 可选修饰"，本项目是"一个入口 + 一个配置树"。

---

## 4. 同类开源项目对比

数据采集于 2026-10-16（GitHub 页面）。star 数为当时快照。

| 项目 | 语言 | Star | 许可 | 定位 | API 形态 | 关键设计选择 |
|---|---|---|---|---|---|---|
| **openai-agents-python**（上游） | Python | 官方 | MIT | 官方 SDK | `Agent[TContext]` 泛型 + `Runner.run` | 四原语：Agents / Handoffs / Guardrails / Sessions |
| **openai-agents-js**（官方） | TS | 3.9k | MIT | 官方 SDK | `Agent<TContext, TOutput>` + zod `tool()` | zod 同时给运行时校验和编译期推导 |
| **Rig** `0xPlaygrounds/rig` | Rust | **8.9k** | MIT | 通用 LLM 应用框架 | `AgentBuilder::new(model).preamble().build()` + `.prompt()` | rig-core（契约）/ rig-agent（编排）分离；20+ provider、10+ 向量库、WASM、cassette 录制回放、可序列化 `AgentRun` 状态机 |
| **ADK-Rust** `zavora-ai/adk-rust` | Rust | 708 | Apache-2.0 | 生产级 Agent 框架（Google ADK 的 Rust 实现） | `LlmAgentBuilder` + `Launcher` | **43 个 crate**，9 层分类（模型/工具/会话/编排/协议/安全/实时/运维）；分级 feature（minimal→standard→enterprise→full）；图工作流 + 检查点 + 时间旅行；A2A v1.0；OTel；`cargo-adk` 脚手架 |
| **AutoAgents** `liquidos-ai/AutoAgents` | Rust | 763 | MIT/Apache-2.0 | 多 Agent 框架 | `#[agent]` / `#[tool]` / `#[derive(ToolInput, AgentOutput)]` | **宏驱动 + 编译期强类型**：`output=MathAgentOutput`、`input=AdditionArgs`；WASM 工具沙箱；类型化 pub/sub；Python binding |
| **rust-genai** `jeremychone/rust-genai` | Rust | 908 | Apache-2.0/MIT | 多 provider 客户端（非 agent 框架） | `Client::new()` + `exec_chat` | 27+ provider、原生协议适配器、按模型名前缀自动路由（`gpt-*`/`claude-*`）、显式注入 `AuthResolver`/`ServiceTargetResolver` |

### 关键洞察

1. **Rust 生态里活下来的项目，卖点都不是"对齐某个 Python SDK"，而是 Rust 独有的能力**：Rig 卖"20+ provider + 10+ 向量库 + WASM + 统一契约"；ADK-Rust 卖"43 crate 的广度 + 性能数据 + 类型安全 + 图工作流"；AutoAgents 卖"派生宏带来的编译期类型"；genai 卖"多 provider 的统一客户端 + 零魔法注入"。
2. **"移植"是负债不是资产**：上游 v0.23.1 会继续演进（hosted tools、workflow、sandbox），本项目要么持续付出 parity 成本，要么停止对齐。而社区用户不会因为"它和 Python 一样"而选择 Rust 版本——他们会直接用 Python 版本。
3. **分发为零**：`docs/refactor/05-open-questions.md` D10 明确"不发布到 crates.io"，但 README 又写着 `openai-agents = "0.1"`（第 35-37 行）。不发布 = 无用户 = 无反馈 = 无法验证"API 是否好用"这个最大的未知数。ADK-Rust 已发布 2.3.0、累计下载 50 万次，这就是反馈回路的差距。
4. **本项目的真正差异化**：`vendor` + `conformance` 构成的行为对齐验证体系，在同类项目里独一无二。它应该被包装成对外的**可验证承诺**（"与 v0.23.1 在 N 个场景上逐字节一致"），而不是只当作 CI 的一个 job。

---

## 5. 工程与仓库卫生（低成本、立刻可做）

| # | 问题 | 证据 | 严重度 |
|---|---|---|---|
| E1 | **README 引用了不存在的文件** | `README.md:263` 链接 `docs/READINESS.md`，该文件不存在 | 高（第一印象） |
| E2 | **README 版本说明与决策冲突** | `README.md:35-37` 写 `openai-agents = "0.1"`；`docs/refactor/05-open-questions.md` D10 写"不发布到 crates.io" | 高 |
| E3 | **DEVIATIONS D-007 已过时** | `docs/DEVIATIONS.md` D-007 仍写"Single crate `openai-agents` with modules + Cargo features"，P3 已拆成 7 crate | 中 |
| E4 | 无 `CHANGELOG.md` | 113 次提交，破坏性路径变更只记录在 `06-progress.md` | 中 |
| E5 | 无 MSRV 声明 | `rust-toolchain.toml` pin `1.99.0`，但 `Cargo.toml` 无 `rust-version`；edition 停留在 2021 | 中 |
| E6 | clippy allow 清单只减不增，但 `result_large_err` 63 处未推进 | `Cargo.toml:30-38` 自述"known, tracked backlog" | 中 |
| E7 | `target/` 20 GB | 本地卫生 | 低 |
| E8 | `vendor/` 33 MB（1772 文件）随仓库流转 | 已 gitignore，但 `sync_vendor.sh` 每次 CI 重新拉取 | 低 |
| E9 | 顶层 docs 为英文、refactor docs 为中文 | 混用 | 低（建议明确规则） |
| E10 | CI 依赖 Python 工具链（parity / MCP interop / 两个 `check_*.py`） | `.github/workflows/ci.yml` | 中（纯 Rust 贡献者门槛） |

---

## 6. 改进建议（按优先级）

### 阶段一：可信度（1–2 周，零 API 破坏）

1. **修文档事实**：补 `docs/READINESS.md` 或删除 README 中的链接；README 安装段改为"git 依赖 / 待发布"并说明发布计划；更新 D-007 描述为 7-crate 布局。
2. **补 `CHANGELOG.md` 与 `rust-version`**：`rust-version = "1.99"`（或下探到 N-2 再声明），并在 README 写清 MSRV 策略。
3. **新增"能力导航"页**：按用户任务（我想加工具 / 我想接别的 provider / 我想做人工审批 / 我想持久化会话）组织入口，指向对应 example + 类型。这是降低学习成本性价比最高的一件事——比改 API 便宜一个数量级。
4. **给 DEVIATIONS 加索引**：61 条按主题（tools / tracing / session / mcp / state）分组 + 顶部"最常遇到的 10 条"。
5. **推进 `result_large_err`**：把 `ModelStatusError` / `AgentsError` 的大变体装箱（`Box<...>` 字段），这是非破坏性的（字段类型从值变 `Box` 会破坏构造，需评估；更稳妥是先缩小 `ModelStatusError` 里的 headers/body 存储）。
6. **决定发布策略**：至少发布 `core` / `macros` / `openai` / `facade` 四个包到 crates.io。D10 是在"没有真实用户反馈"的前提下做出的，这个前提本身就该被打破。

### 阶段二：API 兼容演进（1–2 月，无破坏或可平滑迁移）

7. **加 `prelude`**：导出 30 个高频类型，文档与 examples 全面改用它，既解决命名空间宽度（P2-11），又不破坏既有路径。
8. **配置对象分组**（治 P0-2）：把 `RunConfig` 的 24 个字段按主题收进 `TracingOptions` / `SessionOptions` / `ToolErrorOptions` / `HandoffOptions` 四个子结构，`RunConfig` 保留 4 个分组字段 + 保留 `Deref` 或 `From` 让旧字段读写路径可用；同步提供 `RunOptions::builder()`（过程式构造，缓解套娃字面量 P1-7）。
9. **类型化工具构造器**：新增 `FunctionTool::typed::<Args, F, Fut>()`——从 `schemars` 生成 schema、自动反序列化为 `Args`、返回 `Value`。让"不用宏"的路径也零样板，与 `#[function_tool]` 心智模型统一（治 P1-4），并把 `ToolResult` 的 `interruptions` / `nested_state` 从公共构造器中隐藏。
10. **类型化 reasoning**：新增 `Reasoning { effort, summary }`（`#[non_exhaustive]`），`ModelSettings::reasoning` 改为该类型并提供 `From<Value>` 兼容旧写法；在 adapter 层把 Responses 的 `reasoning` 与 Chat Completions 的 `reasoning_effort` 统一映射（治 P2-10，顺带消灭 example 25 里的分支）。
11. **HITL 便捷 API**：`RunState::approve_all()` / `reject_all()`，以及一个 `run_until_done(&agent, input, options, |interruptions| async { ... })` 的续跑闭包，把 34 行样板压到 5 行。
12. **消灭热路径克隆**（P1-5）：`enabled_tools` / `enabled_handoffs` / `resolve_instructions` 等 7 处去掉 `Arc::new(agent.clone())`，改为传 `&Agent`（配合阶段三的 `AgentView` 更彻底）。同步给 `benches/run_loop.rs` 加"20 个工具 × 10 轮"的场景，防止回归。

### 阶段三：结构性重构（3–6 月，含破坏性变更）

13. **打破 8 模块环**（P0-1）：引入叶子模块 `agent::view`（或独立的 `view` 模块）承载 `AgentView` 只读投影，`tool` / `guardrail` / `handoffs` / `lifecycle` 改为依赖 `AgentView`；把 `check_module_deps.py` 的环白名单清零。**这是本项目最值得做的一次架构投资**——它把"模块图是 DAG"从配额恢复为原则，并解锁后续把 tracing / memory 真正抽成 crate 的可能。
14. **Context 泛型化**（P0-3）：`Agent<TContext = ()>`（或 `RunContext<T>`）+ 保留 `Arc<dyn Any>` 逃生舱；目标是让 `run(agent, input, opts)` 在 context 类型不匹配时**编译失败**——这将是相对官方 Python/JS SDK 的明确优势，也是最好的对外宣传点。
15. **工具 trait 化**（P1-4）：引入 `trait Tool`（`name` / `json_schema` / `invoke` / `needs_resolution`），`FunctionTool` 作为默认实现；让"统一鉴权/限流/审计的工具包装器"在不改 SDK 的前提下可写。
16. **收敛全局状态**（P1-6）：把 `set_default_*` 系列标注为"进程级便利 API，库作者请勿调用"，并在文档给出显式注入示例；`TracingProcessor` / `ModelProvider` 优先从 `RunConfig` 注入。
17. **`#[non_exhaustive]` + 构造器**（P2-9）：对 `RunConfig` / `RunOptions` / `ModelSettings` / `RunState` / `FunctionTool` 加标注并配齐 builder——这是让 0.x 能持续演进的前提。
18. **术语收敛**（P1-8）：二选一（`resolution` 语义更准，建议保留它、删除 `needs_approval*` 别名并在 0.2 前给足 deprecation 窗口）。

### 阶段四：定位与生态（并行）

19. **明确路线**：在 README 顶部用一句话回答"这是 Python SDK 的 Rust 移植，还是一个 Rust 原生的 agents SDK"。建议答案：**"用 Python SDK 的语义与行为契约，配 Rust 的类型与零成本抽象"**——即对齐**行为**（已有 oracle 保证），但不再对齐 **API 形态**（阶段三的泛型/trait 化正是这一路线的落地）。这能同时保住现有资产与差异化。
20. **把 parity 变成产品**：发布"与 v0.23.1 在 N 个场景上逐字节对齐"的公开报告，并把 `conformance/scenarios` 改造成可对外运行的 eval suite。这是同类项目都不具备、而用户真正关心的能力。
21. **可观测性补齐**：提供一个开箱即用的 OTel/OTLP `TracingProcessor`（D-004 已把云导出排除，本地 OTel 正是用户自己实现的痛点），外加 metrics（turn 数、工具失败率、token 成本）。
22. **生态补板**：按用户真实需求排序——更多 provider 预设（DeepSeek/Gemini/Anthropic 的 `CompatibleProvider` 预设）、`prelude`、以及一个 `cargo-generate` 模板。不要追 ADK-Rust 的 43 crate 广度。

---

## 7. 建议的落地顺序（依赖感知）

```
阶段一（文档/流程）  ──────────────────────────────► 立即可做，无风险
        │
        ├─► 6. 发布到 crates.io ─────────► 拿到真实反馈（越早越好）
        │
阶段二（兼容演进）───► 7 prelude / 8 配置分组 / 9 typed tool / 10 typed reasoning / 11 HITL / 12 去克隆
        │
        └─► 13 打破模块环 ──► 14 context 泛型 ──► 15 Tool trait ──► 17 non_exhaustive
                                    │
                                    └─► 19 定位声明 ──► 20 parity 对外 ──► 21 OTel
```

**如果只能做三件事**：① 打破 8 模块环（13）；② context 泛型化（14）；③ 修文档并发布（1 + 6）。前两件让这个项目从"移植"变成"Rust 原生 SDK"，第三件让它开始有用户。
