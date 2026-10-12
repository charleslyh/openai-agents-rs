# 评估:模块依赖图是否应当完全无环

状态:已确认(不追求完全无环,收缩内核并以白名单守住),收缩已执行,内核 9 -> 8 个模块。依据全部来自对当前代码的统计(`scripts/check_module_deps.py` 的图)。

## 现状

去掉 `agent -> run` 之后,图中只剩一个环(领域内核,9 个模块)。每条边实际引用的符号:

| 边 | 引用 | 性质 |
| --- | --- | --- |
| `guardrail`、`handoffs`、`lifecycle`、`mcp`、`model`、`tool`、`tool_guardrails` -> `agent` | `Agent`(回调参数) | 回调需要看到 `Agent` |
| `agent` -> `guardrail` / `handoffs` / `lifecycle` / `mcp` / `model` / `tool` | `InputGuardrail`、`Handoff`、`AgentHooks`、`McpServer`、`Model`、`FunctionTool` | `Agent` 持有它们 |
| `tool` -> `run_state` | `RunState`(`ToolResult.nested_state`) | 嵌套 HITL |
| `run_state` -> `guardrail` / `tool` / `tool_guardrails` | 结果类型、一个常量 | 纯数据引用 |
| `tool` <-> `tool_guardrails` | `ToolInputGuardrail` 与 `ToolContext` | 互相引用 |

## 模拟:各种拆法的效果

| 假设 | 剩余的环 |
| --- | --- |
| 现状 | 9 个模块 |
| 只去掉 `tool -> agent` | 9 个模块(其余回调仍经 `agent` 成环) |
| `run_state` 的 3 条边去掉 | 8 个模块 |
| 所有回调不再接收 `Agent` | `run_state`、`tool`、`tool_guardrails` |
| 上一条再加 `run_state` | `tool`、`tool_guardrails` |

结论:完全无环需要两类改动,第一类破坏 API,第二类不破坏。

- **破坏 API**:所有回调改成不接收 `Agent`(这些模块的源码中共约 56 处 `Agent` 提及:`lifecycle` 16、`mcp` 14、`guardrail` 9、`handoffs` 5、`tool_guardrails` 5、`model` 4、`tool` 3。这是 grep 计数,含非回调用法,只用于量级判断;另有 tests 与 examples 受影响)。
- **不破坏 API**:`run_state` 离开环、`ToolContext` 与 `ToolInputGuardrail` 解耦。

## 完全无环的收益

| 收益 | 在方案 A(单 crate)下的实际价值 |
| --- | --- |
| 编译时间、二进制体积 | 无。Rust 只禁止 crate 之间成环,模块环对 rustc 没有任何代价 |
| 能把内核拆成多个 crate | 无。D9 已决定不拆内核(types / runtime 合并) |
| 叶子模块可独立测试 | 很小。`tool_guardrails` 等不需要 `Agent` 也能单测 |
| 分层更好讲 | 有,但已由"环白名单"覆盖大部分:环外模块不可被拉入,AI 无法新造跨层引用 |

## 完全无环的代价

| 代价 | 说明 |
| --- | --- |
| API 破坏面大 | 每个用户回调的签名变化。守卫、hooks、`is_enabled`、MCP `list_tools`、`CallModelData` 全部受影响 |
| 偏离 Python | Python 回调收到完整的 `Agent`(`on_start(ctx, agent)`、`guardrail(ctx, agent, input)`、`is_enabled(ctx, agent)`)。偏离需要新增 `DEVIATIONS` 条目,并削弱 parity 对照 |
| 必须引入替代抽象 | 两条路:`AgentView` 小结构(与 `Agent` 双份字段,长期同步有漂移风险,与"降低 AI 偏移"目标相反);或 `&dyn` 加 downcast(类型不安全)。泛型化 `Agent<..>` 会污染所有签名 |
| 行为风险 | tracing、hooks、守卫都消费 `Agent` 字段(名称、instructions、tools),要保证视图字段齐全 |
| 测试与示例改动量大 | `tests/` 下 23 个 `.rs` 与 `examples/` 中涉及回调的用例都要改 |

## 不破坏 API 的"收缩内核"

这部分建议做,收益是真实的,代价小:

| 动作 | 效果 |
| --- | --- |
| `run_state` 离开环:`ToolInputGuardrailResult` / `ToolOutputGuardrailResult` 下沉到 `error/payload.rs`,`DEFAULT_APPROVAL_REJECTION_MESSAGE` 下沉,`run_state` 直接引用 `error::payload` | 内核 9 -> 8。`RunState` 全文不出现 `Agent`,它只是因为引用了这几个纯数据才被拉进环。`tool -> run_state` 变为普通的 DAG 边 |
| `ToolContext` 从 `tool.rs` 抽到独立小模块 | 消除 `tool <-> tool_guardrails` 互引,二者的关系变成单向 |

做完之后,环里只剩一个纯粹的结构性原因:**容器持有回调,回调接收 `&Agent`**。这是最小、最诚实的描述,也是 Python 的同一形状。

## 建议

1. 不追求完全无环。收益在单 crate 方案下接近零,代价是大范围 API 破坏与偏离 Python。
2. 保留"环白名单",并把内核环收缩到最小(做上表两项,不改 API)。
3. 在检查器里写清环的成因,让后来者知道为什么这里允许成环、以及怎样才算违规(新模块进入环)。
4. 若将来决定把内核拆成多个 crate,那时完全无环才成为必要条件,届时再评估 `AgentView` 方案。现在做是过早设计。

## 待确认

- 是否同意"不追求完全无环,收缩内核并以白名单守住"?
- 是否现在执行"收缩内核"这两项(`run_state` 出环、`ToolContext` 解耦)?
