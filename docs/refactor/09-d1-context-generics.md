# 09 决策检查点 D1:run context 是否泛型化

状态:**已决议(2026-10-19)**:不泛型化。采用"固定字段 + `Extensions` 类型映射 + 需求声明 + 宏"。

## 结论

`Agent`、`Tool`、`Handoff`、Guardrail、Hooks、`RunConfig`、`RunResult`、`RunState` 都**不带** `C` 参数。
通用数据(agent 名、turn 序号、usage 等)是 `RunContextWrapper` / `ToolContext` 的固定、强类型字段。
业务数据放进按类型取值的 `Extensions` 容器。工具与能力可以声明自己需要哪些类型,运行开始时一次性校验。

## 为什么不选泛型 `Agent<C>`

评估对象是两类用户:写 agent 的使用者,以及写 Tool、Capability、护栏的扩展者。

- 不碰 context 的扩展点(`Model`、`ModelProvider`、`Session`、`RunStateStore`、`TracingProcessor`)不受泛型影响。
- 碰 context 的扩展点(`Tool`、Guardrail、Hooks、中间件、`Capability`)在泛型下都要写成 `impl<C> ... for X`。
  我们计划内置的扩展(Calculator、WebFetch、FileSystem、PII 等)基本不读 context,`C` 对它们是纯噪音。
- 应用内的自定义工具,泛型确实更顺(`ctx.context` 直接是 `&AppCtx`)。
- 可复用的扩展 crate 在泛型下需要 `C: HasDb` 一类的 trait,或要求所有应用共用一个 `AppCtx`,生态越大越别扭。
- 零上下文场景需要 `impl Agent<()> { fn new }` 的技巧,否则 `Agent::new()` 报 `E0282`。
- 影响面 93 个文件,并推翻 `07-acyclic-evaluation.md` 的"泛型污染全部签名"结论。

泛型唯一不可替代的收益是编译期拦截。下面的"启动期校验"把它前移到"第一次运行的第一个 turn 之前",
对需求已声明的类型,这是稳定可复现的失败,不依赖数据路径。

## 设计

### 1. 固定字段(强类型,无 `C`)

`RunContextWrapper` 与 `ToolContext` 持有通用信息,不放进业务容器:

- 已有:`usage`、`tool_name`、`tool_call_id`、`tool_arguments`。
- 新增(增量,可分步):当前 agent 名、turn 序号、trace id。

### 2. `Extensions`:按类型取值的业务数据容器

```rust
let opts = RunOptions::new()
    .with(Db::connect()?)        // 每个类型一个槽
    .with(UserId(7));

let db = ctx.get::<Db>();        // Option<&Db>
let db = ctx.require::<Db>()?;   // 错误带 agent 名、工具名、类型名
```

- 同一类型放两次:后者覆盖前者,debug 构建下 `tracing::warn!`。
- 值以 `Arc<dyn Any + Send + Sync>` 存放,`Extensions` 克隆是廉价的。
- 与 `http::Extensions`、axum `Extension` 同一模式,Rust 使用者熟悉。

### 3. 兼容 Python 风格的单值

`RunOptions::context(value)` 保留,等价于 `with(value)`,也就是放进 `Extensions` 的一个槽。
`ctx.context::<T>()` / `try_context::<T>()` 保留,等价于 `get` / `require`。
现有示例与测试不必改。

### 4. 需求声明与启动期校验(M1 的 `Tool` / `Capability` 一并落地)

- `Tool::requires() -> &[TypeRequirement]`、`Capability::requires()` 声明所需类型。
- 运行开始时遍历可达的 agent 图(tools、handoffs、guardrails、capabilities),
  缺任何类型就返回 `UserError`,信息里带 agent 名、工具名、期望类型,在第一个 turn 之前发生。
- `#[function_tool]` 中写 `db: Ctx<Db>` 参数,宏生成取值与需求声明,使用者不写样板。

### 5. 已知管不住的情况

- 直接调用 `ctx.get::<T>()` 而没有声明需求:仍是运行时。
- 运行时动态生成的 agent(动态 `instructions`、`is_enabled` 回调里构造):不在启动期遍历范围内。
- 这两点接受,并在文档里写明。

## 偏差登记

`Extensions` 是 Rust 独有扩展(Python 只有单个 `TContext`),登记到 `DEVIATIONS.md` 的"扩展"分类,
不进入 parity 对照。

## 对 M1 的影响

- `RunContextWrapper` 先做固定字段与 `Extensions`,是 M1 的第一步,不依赖其他改动。
- `Tool` trait 与 `Capability` 的设计带 `requires()`,不带 `C`。
- 不再需要 `Agent<C = ()>`、`map_context` 之类的适配器。

## 否决方案记录

- **G 泛型 `Agent<C>`**:见上,扩展者负担与影响面过大。
- **渐进类型 `Agent<C = AnyContext>`**:同一 SDK 两种用法,文档与测试面翻倍,未做原型,不采用。
- **单槽擦除(现状)**:够用,但可复用扩展无法各自携带依赖,所以升级为 `Extensions`。

## 原型证据(`/tmp`,未入库)

- 泛型:`Agent<Db>` 与 `Agent<Session>` 互相 handoff、工具按错 context 编写,都在编译期报错;
  错误里出现 `FnTool<Session, {closure}>: Tool<Db>`,对不熟悉 trait 的人偏难读。
- `struct Agent<C = ()>` 单独不够,`Agent::new()` 报 `E0282`;`impl Agent<()> { fn new }` 加
  `impl<C> Agent<C> { fn with_context }` 可行。
- 影响面(grep 实测):命名 `Agent` 的文件 93;`FunctionTool` 198 次;`Handoff` 72 次;
  `RunContextWrapper` 97 次/30 个文件;`RunHooks`/`AgentHooks` 25;`as_tool` 37。
