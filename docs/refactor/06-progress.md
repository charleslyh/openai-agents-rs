# 落地进度

## P0 基线(已完成)

见 [baseline.md](./baseline.md) 第 6 节。格式、clippy、doc、vendor 全部通过;测试 197(无默认特性)/ 262(全特性)。

## P1 已完成项

| 步骤 | 内容 | 结果 |
| --- | --- | --- |
| P1.1 | `src/memory.rs` -> `src/memory/mod.rs`,统一 `mod.rs` 布局 | 完成 |
| P1.2 | `mcp/mod.rs`(1140 行)拆为 `types` / `filter` / `server` / `client` / `tools` / `tests` | `mod.rs` 降至 54 行,内联测试移入 `mcp/tests.rs` |
| P1.3 | `retry.rs`(981 行)拆为 `settings` / `policy` / `policies` / `delay` / `executor`;`openai_retry_advice` 迁入 `model/openai/retry_advice.rs` | `retry.rs` 降至 32 行 |
| P1.4 | `run/agent_loop.rs`(1621 行)拆出 `session_writer` / `finalize` / `resolve`;`build_agent_as_tool` 拆出为 `run/as_tool.rs` | 主体降至 1345 行,`run_loop_inner` 仍集中在此 |
| P1.5 | C2:`DefaultOpenAiApi` 及 getter/setter/全局移入 `model/default_api.rs`;C5:`ModelInputData`、`CallModelData`、`CallModelInputFilter` 移入 `model/input.rs`;C9:`pyjson` 移入 `support/pyjson.rs` | `model` 不再反向依赖 `run`;`context` 不再依赖 `run` |

### 公共 API 路径变化(0.1.0 未发布,允许)

- `crate::run::DefaultOpenAiApi` -> `crate::model::DefaultOpenAiApi`(`openai` feature 下)。
- `crate::run::{CallModelData, CallModelInputFilter, ModelInputData}` -> `crate::model::*`。
- 顶层 `openai_agents::{DefaultOpenAiApi, CallModelData, ...}` re-export 保持不变,仅随 `openai` feature 门控。
- `crate::mcp::*` 顶层导出不变(内部拆为子模块)。
- `crate::retry::*` 顶层导出不变(`retry_policies` 仍为 `crate::retry::retry_policies`)。
- 新增内部模块:`support`、`model::input`、`model::default_api`(feature 门控)、`model::openai::retry_advice`。

## P2 已完成项

| 步骤 | 内容 | commit |
| --- | --- | --- |
| P2.0 | `wire_events` 拆为 `mod.rs`(id/常量/响应对象)与 `emit.rs`(事件发射);`emit_completed` 加 `openai` feature 门控,修掉 `--no-default-features` 下的 dead-code(重构前就存在) | `4564e29` |
| P2.1 | `scripts/check_module_deps.py`:tier 方向检查,接入 CI 与 CONTRIBUTING | `43d3b72` |
| C1 | `error` 成为叶子:守卫载荷类型与 `retry_after_from_headers` 下沉到 `error/payload.rs`、`error/retry_after.rs`,原模块 re-export | `0c21e58` |
| C3 | `AsToolConfig` 与 `Agent::as_tool` 移到 `run/as_tool.rs`(同 crate 内固有 impl,调用方零改动);`agent` 不再依赖 `run`。例外表清空 | `1f396d3` |
| 内核收缩 | `ToolInput/OutputGuardrailResult` 与 `DEFAULT_APPROVAL_REJECTION_MESSAGE` 下沉到 `error/payload.rs`,`run_state` 改引 `error`,出环;`ToolContext` 移到 `run_context.rs`,`tool_guardrails` 不再依赖 `tool`。环白名单收紧到 8 个模块 | 见 git log |
| 检查器增强 | 新增"环白名单":只允许一个 9 模块的领域内核环。同时忽略文档注释里的路径,消除 `context <-> memory` 的误报 | `1f396d3`、`2d95b78` |
| clippy | `[workspace.lints]` 的 `result_large_err` 计数从 126 修正为 63(重新统计) | `0c21e58` |

### 当前依赖结构

- 除下面这个环之外,模块图是 DAG。`run`、`result`、`context`、`memory`、`error`、`retry`、`tracing` 等都在环外。
- 唯一允许的环(领域内核,8 个模块):`agent`、`guardrail`、`handoffs`、`lifecycle`、`mcp`、`model`、`tool`、`tool_guardrails`。成因:容器持有回调,回调接收 `&Agent`。
- 检查器强制:任何其他模块进入环、或出现新的环,CI 失败。

### mcp_interop 偶发失败(已修复,commit 见 git log)

**现象**:`tests/mcp_interop.rs` 里两个 HTTP 测试约 8-12% 的概率失败(`Connection reset by peer` / `Connection refused`),与重构无关,但持续干扰验证。

**根因**(靠捕获失败时 Python 服务器的 stderr 确认):失败那次,服务器日志是
`error while attempting to bind on address (127.0.0.1, N): [errno 48] address already in use`。
测试先 `bind(0)` 取一个空闲端口再释放,然后启动 Python 服务器去绑同一端口。macOS 上 socket 是"先创建、再设置 close-on-exec"两步,如果此时另一个测试线程恰好 `spawn` 了子进程,这个 socket 会泄漏进子进程,**端口一直被占到那个子进程退出**。于是服务器绑定失败,而客户端连上的是一个半死的监听者。

**排除过的假设**(每个都用 60-120 次重复运行验证,不是凭感觉):
- 就绪判断太弱:改为读取 HTTP 状态行,仍失败 4/80。
- 客户端复用了失效的空闲连接:关闭连接池,反而 12/80。
- 端口落在临时端口段、被别的连接占用:改用固定段并加计数器,仍失败 12/100。
- 两个 HTTP 测试互相干扰:只让它们互斥,仍失败 3/100。

**判别实验**:单独并行跑 HTTP 测试 0/60;`--test-threads=1` 0/60;去掉 `connection_failures_are_clear_errors` 仍失败 6/60,去掉 `stdio_servers_do_not_inherit_the_environment` 仍失败 10/60。说明问题不在某一对测试,而在"任何创建 socket 的线程"与"任何 spawn 子进程的线程"同时进行。

**修复**:所有创建 socket 或启动服务器的测试共用一把锁(`one_at_a_time`),整个测试期间持有。150 次重复 0 失败(此前 100 次中 12 次)。代价是这些测试变为串行,总耗时仍在几秒内。

**留意**:这是测试基础设施的问题,不是 SDK 的 bug。`mcp/http.rs` 没有改动。

## 大文件拆分(P1 收尾,每项一个 commit,零语义改动)

| 文件 | 拆前 | 拆后 | commit |
| --- | --- | --- | --- |
| `run_state.rs` | 834 | `run_state/{mod 392, approvals 200, codec 263}` | `377ca2b` |
| `model/openai/chat_completions.rs` | 863 | `chat_completions/{mod 530, stream_layout 213, body 146}` | `9c34b23` |
| `model/openai/chat_convert.rs` | 750 | `chat_convert/{mod 75, to_chat 500, from_chat 194}`,按源码里已有的章节分隔线切 | `2db1461` |
| `tracing/mod.rs` | 661 | `tracing/{mod 420, processor 125, types 136}`,`tracing` 仍只依赖 `usage` | `edccf84` |
| `strict_schema.rs` | 617 | `strict_schema/{mod 386, refs 153, tests 91}`,8 个单测数量不变 | `546fa31` |
| `model/openai/mod.rs` | 630 | `{mod 86, provider 232, request 213, transport 118}`,provider 的测试随 `resolve_api` 一起移入 provider | `ff10867` |
| `run/tools.rs` | 567 | `tools/{plan 269, execute 238, output 80, mod 19}`,对应 Python 的 tool_planning / tool_execution | `77fc48d` |

每项拆分后:fmt、两种 feature 的 clippy、doc 零告警,测试 262 / 197 通过,依赖检查通过;涉及 openai 与 parity 的拆分另外验证了 insta 请求快照与 `chat_convert_parity`、`run_parity.py --check`(golden 无变化)。

### 仍超过 500 行的文件

| 文件 | 行数 | 原因 | 处理 |
| --- | --- | --- | --- |
| `run/agent_loop.rs` | 1345 | `run_loop_inner` 一个函数约 1140 行 | 需要按阶段拆函数,不是切文件;风险最高,单独安排 |
| `model/openai/chat_completions/mod.rs` | 530 | `stream_response` 一个方法约 395 行 | 同上,是单方法问题 |
| `model/openai/chat_convert/to_chat.rs` | 500 | 恰好在软上限,且内部按 `items_to_chat_messages` 一个主流程组织 | 暂不处理 |

## P3 物理拆 crate(主体完成)

仓库改为 virtual workspace(`members = ["crates/*"]`,`resolver = "2"`),版本与 lint 由
`[workspace.dependencies]` / `[workspace.lints]` 统一。

| 步骤 | 内容 | commit |
| --- | --- | --- |
| P3.0 | 单体迁入 `crates/core`;新建 `crates/facade` 保留 `openai-agents` 这个包名与每一条公共路径 | `735f6a5` |
| P3.1 | 抽出 `macros`(含 C11 的 `crate = "..."` 覆盖) | 同上 |
| P3.2 | 抽出 `testing`(`ScriptedModel` 与 item 构造器,D2);core 只留一个小的公开 `replay_response`,测试替身不再需要私有的 wire 辅助函数 | `1b80a89` |
| P3.3 | 抽出 `sqlite`,`rusqlite` 离开 core,只在 facade 的 `sqlite` feature 下编译 | `4b53a59` |
| P3.4 | 抽出 `mcp`:`McpServer` 契约留在 core,客户端与两种传输进 crate,`mcp` 变为 facade 的**可选** feature(D6) | `f8cb825` |
| P3.5 | 抽出 `openai`:适配器离开 core,由 facade 装配默认 provider(C10 / D1) | `67487eb` |
| P3.6 | `check_module_deps.py` 增加 crate 级规则,从 `cargo metadata` 读真实依赖,禁止 core 引入 HTTP、数据库与子进程相关的库 | `369502e` |

### core 现在的依赖

只有 `async-trait`、`futures`、`schemars`、`serde`、`serde_json`、`thiserror`、`tokio`、
`tracing`、`uuid`。没有 `reqwest`、`async-openai`、`rusqlite`,也没有 `tokio` 的 `process`
与 `io-util`。`core/Cargo.toml` 不再定义任何 feature。

### facade 的 feature

| feature | 默认 | 作用 |
| --- | --- | --- |
| `openai` | 是 | `OpenAIProvider` 与 Responses / Chat Completions 适配器 |
| `mcp` | **否** | MCP 客户端,stdio 与 streamable HTTP |
| `sqlite` | 否 | `SqliteSession` |

`mcp` 默认关闭是 D6 的落地:不接 MCP 服务器的构建不会编译它的进程与 HTTP 代码。CI 与文档里
的 `cargo test --test mcp_interop` 因此改为 `cargo test --features mcp --test mcp_interop`——
不加这个 flag 时该目标会编译成零个测试并且**仍然通过**,这是必须显式给出的原因。

### 每一步的验收

`cargo test --workspace --no-default-features` 与 `--all-features` 都要通过;facade 的
feature 矩阵(每个扩展单独开/关)要能编译;fmt、两种 feature 的 clippy 与 doc、
`check_module_deps.py`、`run_parity.py --check`、`sync_vendor.sh --check` 都要干净。

- `--all-features` 仍是 262,P0 基线不变。
- `--no-default-features` 是 200,比基线的 197 多 3,三项都能点名:
  - `sse_events_handle_chunking_crlf_and_multiline_data`:此前藏在 `mcp` feature 后面,现在随
    mcp crate 一起无条件编译。
  - `api_defaults_to_chat_completions`:此前在 core 的 `openai` feature 后面,现在随 openai
    crate 一起无条件编译。
  - `provider::CompatibleProvider` 的 doctest:此前在 `src/model/openai/provider.rs`,现在在
    openai crate 里,不再受 feature 门控。

  以 P3 前的 `d086e1f`(单独 worktree 采集)为基准做多重集比对:具名测试 194 -> 196,doctest
  6 -> 7,**两个方向都没有任何一个测试丢失**。
  (口径:现在 196 具名 + 7 doctest = 203 listed,其中 3 个 doctest 标 `ignore`;基线 194 + 6
  = 200 listed,2 个标 `ignore` 加 macros 的 1 个共 3 个。故实际执行 197 -> 200,
  `--all-features` 262 不变。)

### 体积与编译时间

- 冷启动(清空 `target/` 后构建全量测试)约 67 秒 -> 约 40 秒。
- 增量编译**变差**:默认调试信息下,改 `core` 后重编一个测试目标约 1.5 秒 -> 约 5.1 秒。原因是
  core 变动后下游每个 crate 都要重新链接。
- 把调试信息降到 `line-tables-only` 后回到约 1.8 秒。**这只是一次实验,没有写进仓库配置**;
  若要采纳需单独决定。

### 可观察的行为变化

登记在 `docs/DEVIATIONS.md` 的 D-047:crate 布局、feature 语义、以及 `mcp` 默认关闭。
只设 `OPENAI_API_KEY` 的零配置运行、嵌套 `as_tool` 运行、无 key 与未开 `openai` feature 时的
报错文案,都在临时工程里实测确认未变。

### 防回归的两个脚本

- `scripts/check_module_deps.py`:除模块级 tier 方向外,新增 crate 级规则。用 5 种注入的违规
  验证过,每种都能拦下。
- `scripts/check_runner_api.py`:新增。facade 的 `Runner` 必须与 core 的方法一致,防止 facade
  在转发时漏掉或改掉方法签名。注入两种偏差验证过。

## 剩余 P1 / P2 项

- 拆 `run_loop_inner`(约 1140 行的单函数)。**未做。** P1.4 只把它周围的代码拆了出去,函数
  本身仍是约 1140 行,含在 `run/agent_loop.rs`(1345 行)里。
- tests 按领域分组:决定不做,理由见下文。
- 集成测试按 crate 分散:**未做。** 23 个 `tests/*.rs` 目标全部仍在 `crates/facade/tests/`,
  其他 crate 只有单元内联测试。原因是它们大多跨 crate(一个 `behavior_*` 会同时用到 core、
  openai、testing),留在 facade 才能一次装配齐全。这条与 P3 前"分布问题留给 P3 随 crate 边界
  一并解决"的打算不符,登记为偏差(见下)。
- C10:`default_model_provider` 对 openai 的装配上移(D1 已决)。**已在 P3.5 完成。**
- examples / docs / scripts 整理:docs 里与 crate 相关的部分已同步(README、COMPAT、
  CONTRIBUTING、DEVIATIONS、CI、七个 `AGENTS.md`)。

## 计划与实际的偏差

对照 `04-roadmap.md`。这张表是"这次重构离原计划有多远"的直接答案。

### 做了,且与原计划一致

| 计划 | 实际 |
| --- | --- |
| P0 安全网 | 完成。基线 197 / 262 |
| P1.1-P1.5 零语义整理 | 完成 |
| P1 大文件拆分(7 个文件) | 完成,每项一个 commit |
| P2.0 `wire_events` 拆分 | 完成 |
| P2 依赖方向检查脚本 + CI | 完成,`43d3b72` |
| P3.0 workspace + facade | 完成,`735f6a5` |
| P3.1 macros(C11) | 完成,同上 |
| P3.2 testing(D2) | 完成,`1b80a89` |
| P3.3 sqlite | 完成,`4b53a59` |
| P3.4 mcp(D6) | 完成,`f8cb825` |
| P3.5 openai(C10 / D1) | 完成,`67487eb` |
| D8 `conformance/` | 完成,`d086e1f` |

### 计划里有、但按决议不做的

| 计划 | 决议 | 理由 |
| --- | --- | --- |
| P3b:抽出 `types`、`tracing`,core 与 runtime 分离 | 取消 | D9 选粗粒度方案 A;D3 选 T1,tracing 留 core 内 |
| P3a 里的 `context` crate | 取消 | D13:无可剥离的外部依赖,且 run loop 无条件使用它 |
| D7(retry 独立归属) | 降级 | D9:retry 留 core,只移出 `openai_retry_advice` |
| D11 / C4(`RunState` 与 `tool`) | 失效 | 同在 core,无需处理 |
| C3(`as_tool` 与 run) | 失效 | D12 + 同在 core |
| P4 的发布配置(`exclude`、发布顺序) | 取消 | D10 不发布到 crates.io |

### 计划里有、仍未做的

| 计划 | 状态 |
| --- | --- |
| 拆 `run_loop_inner`(约 1137 行单函数) | **未做**,仍是最大的单项 |
| `stream_response`(392 行单方法,`crates/openai/src/chat_completions/mod.rs`) | 未做 |
| 集成测试按 crate 分散 | **未做**:23 个测试目标全在 `crates/facade/tests/`,因为它们大多跨 crate |
| CI 分 crate 矩阵 | 未做:仍是两个 OS × 三个 feature 档,没有 per-crate 矩阵 |
| tests 按领域分组 | 决定不做(`tests/` 子目录不会被 Cargo 自动发现,见下) |
| examples 整理 | 未整理 |

### 计划里没有、实际新增的

| 项 | 原因 |
| --- | --- |
| `scripts/check_runner_api.py` | facade 重新实现了 `Runner`,需要防止它的方法与 core 漂移 |
| `check_module_deps.py` 的 crate 级规则 | 拆 crate 后,模块级 tier 检查管不到跨 crate 的依赖 |
| `mcp` 默认关闭带来的 CI / 文档改动 | D6 的连带影响:不加 `--features mcp` 时测试目标会静默编译成零个测试 |
| 增量编译变慢(1.5s -> 5.1s) | 拆 crate 后 core 变动会触发下游每个 crate 重新链接。降到 `line-tables-only` 可回到 1.8s,**未采纳** |

### 数字上的偏差

`--no-default-features` 从 197 变成 200,三项都能点名,见上文 P3 的验收小节。
`--all-features` 262 不变。

## conformance/(D8,已完成)

`d086e1f`:跨语言共享的资产集中到 `conformance/`,Rust 测试和脚本目录不再混放。

| 迁移前 | 迁移后 |
| --- | --- |
| `tests/parity/scenarios/*.json`(26 个) | `conformance/scenarios/` |
| `tests/parity/chat_convert_cases.json` | `conformance/chat_convert_cases.json` |
| `tests/mcp/server.py` | `conformance/mcp_server.py` |
| `scripts/run_parity.py`、`chat_convert_oracle.py`、`mcp_oracle.py`、`setup_venv.sh` | `conformance/` 下同名 |

`scripts/` 现在只剩与 oracle 无关的两个仓库级脚本:`sync_vendor.sh`、`check_module_deps.py`。

文本替换之外需要手工核对的两处:`run_parity.py` 用路径分量拼接目录(`ROOT / "tests" / "parity" / "scenarios"`,字符串替换抓不到),已改;`ROOT = parents[1]` 与 `setup_venv.sh` 的 `cd ..` 因 `conformance/` 与 `scripts/` 同为根目录下一层,无需改。

迁移前后对照:场景文件 26 个不变;`run_parity.py --check` 13 项 golden 全部 unchanged;依赖 Python 的 5 个测试目标(`parity_scenarios`、`chat_convert_parity`、`mcp_interop`、`context_trimming`、`sqlite_session`)通过数不变且无 skip;全量 262 / 197 通过。

## tests/ 不按子目录分组(决定:不做)

实验证实:`tests/` 子目录里的 `.rs` 不会被 Cargo 自动发现,必须逐个写 `[[test]]`,或改成 `tests/<group>/main.rs` 合并成一个二进制。

- 写 `[[test]]`:新增测试忘记登记就**静默不运行**,恰好是"降低 AI 扩展时的偏移与错误"要避免的那类陷阱。
- 合并二进制:`behavior_tracing` 等依赖进程级全局状态(全局 trace processor、`set_var`),合并后互相污染。
- 现状已经有 `behavior_*` 前缀约定,目标名也被 CI 与文档引用。
- P3 拆 crate 时各测试会按归属散到各 crate 的 `tests/`(sqlite、mcp、openai 等),此时再分组会返工。

结论:保持扁平加前缀约定;分布问题留给 P3 随 crate 边界一并解决。
