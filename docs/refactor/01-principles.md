# 重构原则与统一聚类准则

状态:草案,待确认。

## 1. 唯一主轴:依赖方向(分层)

- 一个单元属于"能容纳其全部依赖的最低层"。
- 同层内再按"变更原因相同"聚合。
- 下层绝不依赖上层。
- 同层扩展之间绝不互相依赖。

所有目录、crate 的划分都只用这一条准则。出现平局时用两条裁决规则:

1. **外部重依赖隔离**:引入 `reqwest`、`rusqlite`、`async-openai`、`tokio/process` 的代码必须在扩展 crate,不得进入核心层。
2. **可剥离性测试**:删掉该单元后,下层仍能编译且语义完整,则它是扩展。

## 2. 分层

| 层 | 内容 | 对应 crate(方案 B) |
| --- | --- | --- |
| L0 基础 | 纯数据与纯函数,无 I/O、无 Agent 概念:usage、error 及数据载荷、items、stream_events、model_settings、retry 的设置与纯函数、strict_schema、pyjson | `types` |
| L1 横切 | 仅依赖 L0:tracing | `tracing` |
| L2 领域内核 | 契约与领域模型:Agent、Tool、Handoff、Guardrail、Hooks、RunContext、Model/ModelProvider trait、Session trait 与 InMemorySession、McpServer trait | `core` |
| L3 运行时 | 执行循环与状态:run loop、RunState 执行、RunResult、流式 | `runtime` |
| L3 扩展 | 只依赖 L2(必要时 L3),彼此不依赖:openai、mcp、sqlite、context、testing | 各扩展 crate |
| L4 聚合 | facade:re-export、feature 开关、examples、跨 crate 端到端测试 | `openai-agents` |

## 3. AI 落位决策树

新增或移动代码时按序判断,命中即停:

1. 是否引入 I/O 或重依赖(HTTP、SQLite、子进程)? -> 扩展 crate。
2. 是否只是数据或纯函数,不涉及 Agent/Tool? -> `types`。
3. 是否是被 Agent 或 Runner 引用的契约(trait 或类型)? -> `core`。
4. 是否驱动执行流程或持有运行状态? -> `runtime`。
5. 是否只在测试或示例中使用? -> `testing` 或 facade 的 `tests/`、`examples/`。

## 4. 依赖方向禁止清单

- `types` 不得依赖 `core`、`runtime` 及任何扩展。
- `core` 不得依赖 `runtime` 及扩展。
- 扩展之间禁止互相依赖,只能依赖 `core` 或 `types`。
- 扩展若必须依赖 `runtime`,须在 `docs/refactor/` 中单独登记并说明原因。
- 只有 facade 做扁平 `pub use` 聚合;其余 crate 内部用路径引用。
- 不允许为了绕开依赖方向而使用全局可变状态或 `dyn Any` 传递领域对象(确需例外要登记)。

## 5. 布局与体量约定

| 项 | 约定 | 机械约束 |
| --- | --- | --- |
| 模块布局 | 统一用 `mod.rs`(现有仅 `memory.rs` 例外,迁移成本最小) | clippy `mod_module_files` |
| 文件体量 | 软上限 500 行,硬上限 800 行,超限按职责拆分 | CI 脚本统计行数(豁免清单需登记) |
| 内联测试 | 过大时拆为同级 `tests.rs` | 评审 |
| `lib.rs` / `mod.rs` | 只放 `mod` 声明与 re-export,不放逻辑 | 评审 |
| feature 与 crate | crate 负责"可剥离编译";feature 只用于 facade 开关与同 crate 内轻量可选项;扩展 crate 内不再按 feature 分裂 | 评审 + CI feature 矩阵 |
| 每个 crate | 含 `AGENTS.md`:职责、允许依赖、落位示例、反例 | 评审 |

硬上限的两个例外需要在迁移时处理:`run_loop_inner`(约 1140 行的单函数)按阶段拆分,不是单纯切文件。

## 6. 重构执行原则

1. **行为不变**:parity golden 与 insta 快照不得变化;变化即缺陷,除非被明确列为 API 破坏项。
2. **纯移动与行为修改分开提交**,commit 格式 `type(module): description`。
3. **每步可独立回滚**,每步后必须通过:`cargo fmt`、`cargo clippy --all-features --all-targets -- -D warnings`、`cargo test --no-default-features`、`cargo test --all-features`。
4. **文档同步**:公共 API 或行为变化同步 `docs/COMPAT.md`、`docs/DEVIATIONS.md`。
5. **不扩大范围**:不顺手优化逻辑;不改 `[workspace.lints]` backlog 语义。
6. **vendor/ 与 `sync_vendor.sh` 校验语义不变**,仅在路径变化时同步更新。
