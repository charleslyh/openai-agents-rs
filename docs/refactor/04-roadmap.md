# 分阶段落地路线(确定性由高到低)

通用验证(每个 commit 后都要过):

```bash
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
cargo test --no-default-features
cargo test --all-features
RUSTDOCFLAGS=-D warnings cargo doc --no-deps --all-features
bash scripts/sync_vendor.sh --check
```

parity 阶段额外:`bash scripts/setup_venv.sh`、`.venv/bin/python scripts/run_parity.py --check`、`cargo test --test parity_scenarios --test mcp_interop --test chat_convert_parity`。
任何 insta 快照或 parity golden 的变化都视为缺陷。

## P0 建立安全网(确定性接近 100%,不改代码)

- 步骤:跑通上述验证;按 [baseline.md](./baseline.md) 第 6 节记录基线(测试数、`cargo tree`、`--timings`、示例二进制体积);用脚本导出 `use crate::` 依赖图。
- 回滚点:无(只新增文档/脚本)。
- 风险:无。

## P1 零语义整理(高确定性,仍是单 crate)

1. `memory.rs` -> `memory/mod.rs`,统一布局;开启 `mod_module_files`。
2. 拆大文件,不改逻辑(切点见 baseline 第 5 节):`mcp/mod.rs`、`retry.rs`、`run/agent_loop.rs` 外围部分(session_writer / finalize / resolve)、`chat_completions.rs`、`run_state.rs`、`chat_convert.rs`、`tracing/mod.rs`、`run/tools.rs`、`strict_schema.rs`、`wire_events.rs`。拆前再读一次文件确认切点。
3. `mcp/mod.rs` 分为契约(`server.rs`、`types.rs`)与实现(`client.rs`、`manager.rs`、transport)。
4. 单 crate 内修复纯类型移动的环:C1、C2(先移到 `model/`)、C5、C9;`openai_retry_advice` 移到 `model/openai`。
5. tests 按领域分组(behavior / contract / parity / property),examples、docs、scripts 整理;同步 CI 与 CONTRIBUTING 中的 `--test` 名称与路径。
6. 每个 commit 一个动作;纯移动与改动分开。

- 回滚点:每个 commit 独立可回退。
- 风险:低。

## P2 层次检查与剩余环(中高确定性)

- 增加依赖方向检查脚本(基于 `use crate::` 的允许边表),CI 强制。
- 处理 C3、C4、C6、C7、C8、C10 中不需要物理拆分即可完成的部分;C3、C4、C10 先落地 D 系列决策。
- 拆 `run_loop_inner` 为阶段函数,以行为测试保护。
- 回滚点:按环逐个提交。风险:中(C3、C4、C10)。

## P3 物理拆 crate(中确定性,增量)

**P3a:达成方案 A**
1. 迁入 `crates/`,根改为 virtual workspace,`[workspace.dependencies]` 统一版本;保留 `[workspace.lints]`。
2. 单体改名 core,新建 facade。
3. 依次抽出:macros(含 `crate = "..."` 覆盖,C11)-> sqlite -> mcp -> openai -> testing。
   (`context` 不抽,见 D13。)

**P3b:达成方案 B**
- 抽出 types、tracing;core 与 runtime 分离。

每一步要求:`cargo test --no-default-features` 与 `--all-features` 通过;facade 的 feature 矩阵(各扩展单独开/关)能编译。

- 回滚点:每抽一个 crate 一个 commit/PR。
- 风险:中(多 crate 路径、宏、`testing` 的 dev-dependency 环)。

## P4 收尾

- `docs/COMPAT.md` 增加"Python 模块 -> Rust crate/模块"映射表;更新 CONTRIBUTING、`.cursor/rules`、`.github/workflows`(分 crate 矩阵);各 crate 补 `AGENTS.md`。
- 剥离体积对比:`cargo tree`、各 feature 组合的示例二进制体积、`--timings`。
- 发布配置(取决于 D10):`exclude` 掉 `vendor/`、测试资产,确定发布顺序。

## 实施注意

- commit 格式 `type(module): description`。
- 公共 API 变化同步写入 `docs/DEVIATIONS.md` / `docs/COMPAT.md`。
- 若发现与 baseline 的核实事实不符,先更新文档再进入下一阶段。
