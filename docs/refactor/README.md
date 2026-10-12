# 重构规划索引

目标:分离核心与扩展、统一聚类准则、拆大文件,降低维护与 AI 扩展时的偏移风险,并支持扩展按 crate 剥离编译。

已确认约束:公共 API 可破坏(0.1.0 未发布,且不发布到 crates.io);**采用方案 A**(粗粒度,6 个 crate 左右);目录名不带前缀,仅 package name 带 `openai-agents-` 前缀;不镜像 Python 目录,映射表放 `COMPAT.md`;行为、parity golden、insta 快照不得变化;`vendor/` 不变。

目标目录(方案 A):

```
crates/
  facade/      package openai-agents        默认 feature: openai(不含 mcp);examples、benches、集成测试
  core/        package openai-agents-core   类型、契约、运行时、memory、context、tracing(内部模块,仅允许依赖 usage)
  openai/      package openai-agents-openai
  mcp/         package openai-agents-mcp
  sqlite/      package openai-agents-sqlite
  testing/     package openai-agents-testing(ScriptedModel)
  macros/      package openai-agents-macros
conformance/   Python oracle、parity 场景与 golden
vendor/        不变
```

crate 边界以 [02-options.md](./02-options.md) 方案 A 为准;context(trimmer、compaction)在方案 A 中留在 core。

| 文档 | 内容 | 状态 |
| --- | --- | --- |
| [baseline.md](./baseline.md) | 现状、核实事实、拆分切点、P0 基线清单 | 事实已核实;P0 数据待采集 |
| [01-principles.md](./01-principles.md) | 聚类准则、分层、落位决策树、禁止清单 | 原则有效;分层中的 types/runtime 在方案 A 下是 core 内部模块层,不是 crate |
| [02-options.md](./02-options.md) | 方案 A / B / C 与对比 | 已选 A |
| [03-cycle-breaking.md](./03-cycle-breaking.md) | 循环依赖与手段 | 见 05 中失效/降级说明 |
| [04-roadmap.md](./04-roadmap.md) | P0 至 P4 | P3b 取消,P4 发布部分取消 |
| [05-open-questions.md](./05-open-questions.md) | 决策记录、D2 说明、D12 分析、D3 冲突 | 全部决策已完成(D3 选 T1) |
| [06-progress.md](./06-progress.md) | P0 / P1 落地进度、API 路径变化、验证状态 | 进行中 |

下一步:执行 P0(基线采集)与 P1 中确定性最高的项(`memory.rs` 布局统一、大文件拆分、C2/C5/C9 纯类型移动)。
