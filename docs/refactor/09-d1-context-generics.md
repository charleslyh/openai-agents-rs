# 09 决策检查点 D1:run context 是否泛型化

状态:**待用户确认**。M1 在此之前不开工。
范围说明:计划写的是"只改 tool + guardrail + handoffs 的 spike"。我没有在仓库里改这些(会动 90+ 个文件,
且不可逆地污染 M0 基线),而是做了两件更便宜的事:在 `/tmp` 里写两个最小原型,用真实编译器输出比较;
再用 grep 量化泛型会扩散到的范围。下面的结论建立在这两项上,没有在真实代码库里编译验证过。

## 现状

`RunContextWrapper` 内部是 `Option<Arc<dyn Any + Send + Sync>>`,取值用 `context::<T>()` / `try_context::<T>()`
(`crates/core/src/run_context.rs`)。类型不匹配在运行时才报错,错误信息已经可读
(`run context is not of type X`)。DEVIATIONS D-014 已记录此取舍。

## 方案

- **G 泛型**:`Agent<C>`、`FunctionTool<C>`、`Handoff<C>`、各类 Guardrail、`RunHooks<C>`、`RunConfig<C>`、
  `RunOptions<C>`、`RunResult<C>`、`RunState<C>` 都带 `C`。
- **E 擦除 + 启动期校验**:保持现状,新增在 `Agent` 上声明期望的 context 类型,运行开始时校验一次,
  类型不符立刻返回带 agent 名、工具名的错误,而不是等某个工具首次调用时才失败。

## 原型证据

编译器对 G 的错误输出(原型 `generic.rs`):

```
error[E0308]: mismatched types
  expected `Agent<Db>`, found `Agent<Session>`           // handoff 到不同 context 的 agent
error[E0277]: the trait bound `FnTool<Session, ..>: Tool<Db>` is not satisfied   // 工具按错了 context 写
```

- 两类错误都在编译期被拦住,位置准确。这是 G 相对 Python/JS SDK 的真实优势。
- 第二条错误提到了 `FnTool<Session, {closure@...}>` 和 `Tool<Db>`,对不熟悉 trait 的人偏难读。
- 闭包不写类型标注时(`tool(|c| ...)`),类型能从 `Agent::<Db>::tool` 推断出来,没有额外负担。
- E 的错误是运行时字符串 `run context is not of type Session`,可读,但要等到被执行到。

关于零上下文场景(hello 示例):

- 只写 `struct Agent<C = ()>` **不够**。`Agent::new()` 会报 `E0282 type annotations needed for Agent<_>`,
  默认类型参数不参与表达式推断。
- 可行的办法已验证:把 `new()` 放在 `impl Agent<()>`,把 `with_context::<C>()` 放在 `impl<C> Agent<C>`。
  这样 `Agent::new()` 仍是一行,带 context 的才需要写类型。

## 影响面(grep 实测,含 examples/demo)

| 项 | 数量 |
| --- | --- |
| 命名 `Agent` 的文件 | 93 |
| `FunctionTool` 出现次数 | 198 |
| `Handoff` 出现次数 | 72 |
| `RunContextWrapper` 出现次数 / 文件数 | 97 / 30 |
| `RunHooks` / `AgentHooks` | 25 |
| `as_tool` 相关 | 37 |

泛型会扩散到上面所有类型,以及 `#[function_tool]` 宏(它按名字识别 `ToolContext`,需要改成识别 `ToolContext<C>`
并让生成的构造函数对 `C` 泛型)。

## 需要先回答的两个设计问题

1. **跨 context 类型的 handoff 与 `as_tool`**:G 下它们编译不过。当前 `Agent.as_tool` 的嵌套运行共享父 context,
   所以不是退化,但"子 agent 用另一种 context"这种用法会被禁止。Python 也是单 `TContext`,行为一致。
2. **`RunState` 序列化**:context 本身不序列化。G 下恢复一个暂停的运行要求调用方提供同一个 `C`,
   这在类型上是更诚实的,但会让 `RunState::from_json` 多一个类型参数,HITL 示例的代码变长。

## 成本对比

| | G 泛型 | E 擦除 + 启动校验 |
| --- | --- | --- |
| 类型错误发现时机 | 编译期 | 运行开始时(第一个 turn 之前) |
| 公共类型签名 | 全部多一个 `C` | 不变 |
| 零上下文示例 | 需要 `impl Agent<()>` 技巧才不退化 | 不变 |
| 迁移面 | 约 90+ 个文件 | 小 |
| 与 M1 其他改动的叠加风险 | 高(Tool trait、Capability、Runtime 都要带 `C`) | 低 |
| 与 `docs/refactor/07` 结论的关系 | 与"泛型污染全部签名"的判断一致,需要推翻 | 沿用 |

## 我的建议

选 **E**,理由:

1. M1 已经同时重做 `Tool` trait、`Capability`、`Runtime`、options。再叠加一个贯穿所有类型的 `C`,
   首发前的返工风险最大,而 E 能拿到"出错尽早且信息明确"这一大半收益。
2. G 的独有收益(编译期)是真的,但目标用户首先要解决的是学习成本;`Tool<C>`、`Handoff<C>`、
   `Verdict` 加 `C` 之后,错误信息里会出现更多泛型。
3. E 不关门:启动期校验用到的"agent 声明期望类型"字段,以后若转 G,恰好就是 `C` 的位置。

反过来,若你更看重"相对官方 SDK 的编译期类型优势"作为卖点,选 G 是合理的,
代价是 M1 工期上调约 1.5-2 人周,并推翻 `07` 号文档的结论。

## 需要你确认

- 选 G 还是 E。
- 若选 E:`Agent` 上声明 context 类型的 API 形式(`.context_type::<T>()` 还是在 `RunOptions::context(value)` 时自动记录)
  我倾向后者,让运行时校验只在"agent 声明了类型且与传入值不符"时触发。
