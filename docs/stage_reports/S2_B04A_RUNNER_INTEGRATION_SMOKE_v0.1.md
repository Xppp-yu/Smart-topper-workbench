# Stage Report: S2-B04A — B04A Runner Integration + CPU Synthetic Smoke v0.1

**TASK-ID:** `TASK-SLP-B04A-RUNNER-INTEGRATION-SMOKE-v0.1`
**Stage:** S2-B04A (runner integration + CPU synthetic smoke)
**Date:** 2026-08-30（R01 交付 + R02/R03 ITERATE 修订；Codex Reviewer R03 独立验收通过）
**Status:** `RUNNER_INTEGRATION_ACCEPTED / GPU_MINI_NOT_AUTHORIZED` (**not**
`GPU_MINI_AUTHORIZED`,
**not** `MINI_COMPLETE`, **not** `B07_READY`)
**Branch:** `codex/task-slp-b04a-runner-integration-smoke-v0.1`
**Maintainer:** Mavis (MiniMax Code)

---

## 1. 执行摘要

把 B04 PM-only Region Mini runner 扩展为同时接受
`configs/experiments/slp8_pm_region_mini_v0.1.json`（B04 历史
配置）和
`configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json`（B04A
architecture expansion 配置），通过显式的 protocol/profile
dispatch 实现，而不是把现有 `TASK_ID` 检查简单删掉。

**完成的事：**

1. 在 `slp8_region_mini.py` 加入 B04A 协议常量
   (`B04A_TASK_ID`、`B04A_CONFIG_VERSION`、
   `B04A_ACTIVE_CANDIDATE_NAMES`、`B04A_FORBIDDEN_CANDIDATE_NAMES`、
   `B04A_SEEDS = (42, 123, 2026)`、`B04A_FEASIBILITY_THRESHOLD = 0.355644`、
   `B04A_NEAR_TIE_MARGIN = 0.02` 等)，保持 B04 历史常量字节级不变。
2. 把 `validate_mini_config` 改写为 dispatcher：按 `config_version`
   路由到 B04 或 B04A 协议级 validator；未知版本 fail-closed。
3. 把 `build_mini_config` 改写为 dispatcher：`MiniConfig` 增加
   `protocol` 与 `seeds` 字段（`seed` 仍保留以兼容 B04 调用方）。
4. 新增 `_validate_b04a_mini_config`：覆盖 3 active candidates
   唯一性、DEFERRED 占位、forbidden 名单、`exact_parameter_count`、
   3 seeds、threshold、B04A training contract、metrics 背景排除、
   resource budget、lifecycle、TEST=0。
5. 新增 `run_mini_b04a` orchestrator：candidates × seeds 循环、
   per-seed `CheckpointIdentity`（含 seed）、per-seed checkpoint
   目录 `checkpoints/<candidate>/seed_<seed>/{last,best}.pt`、
   资源预算检查。
6. 新增 `_b04a_aggregate_candidate` 聚合层：
   `all_seeds_must_succeed=true`、per-seed class collapse / worst
   subject floor / per-region floor 硬关、macro_iou_mean 仅在
   三 seed 全过的情况下计算。
7. 新增 `_b04a_advance_decision` 决策层：0/1/2/3-feasible + 差值
   < 0.02 的更简单模型 tiebreak。
8. 新增 `B04ACandidateAggregate` / `B04ARunResult` 数据类。
9. 新增 `_write_b04a_run_bundle` / `_write_b04a_candidate_aggregate`
   / `_write_b04a_seed_artifacts` 写产物；身份字段（task_id、
   EXP-ID、config_sha256、git_commit、git_dirty、data_manifest_sha256、
   split_sha256、model_version）按
   `identity_hard_gate.carrier_format_by_file_type` 在 JSON /
   CSV sibling / log first line / checkpoint identity 中分别承
   载。DONE/FAILED/STOPPED 互斥。
10. `run_one_candidate` 增加 `seed` 与 `checkpoint_dir` 参数：
    B04 默认行为不变，B04A 走每 seed 独立子目录的写盘。
11. CLI `run_slp8_region_mini.py` 增加
    `--synthetic-cpu-smoke-b04a` 与 `--no-write` 模式（CPU only），
    老的 `--synthetic-cpu-smoke` / `--validate-config` 路径字节
    级保持。
12. 新建 `scripts/smoke_b04a_runner_integration.py`：独立 B04A
    runner-integration smoke；支持 `--no-write`（输出单行
    `B04A_SMOKE_NO_WRITE ...`）、`--force` / `--output` /
    `--output-dir` / `--budget-override-seconds`；默认拒绝覆盖
    已存在产物。
13. 新建 `tests/test_b04a_runner_integration.py`：**58** 个聚焦
    测试，覆盖 dispatch / B04 backward-compat / candidate 限制 /
    seeds / all_seeds_must_succeed / 0-3 decision / near-tie /
    budget resume / identity mismatch / 输出互斥 / TEST=0 / B04
    回归。**全部通过**。
14. 真实跑 `scripts/validate_b04a_protocol.py` 对 B04A 配置：30
    OKs / 0 errors（合同未改）。
15. 不修改任何 B04A R03 冻结配置、不修改 B01 freeze tables、
    不修改历史 B04 R05 EXP-ID 或数值、不修改已验收的
    SmallUNet / ResUNet-lite / DeepLabV3+-lite 模型结构与参数
    量。

**未做（明确禁止 / 留给下个 Gate）：**

* 任何真实 B01 GPU Mini 启动（仍 `BLOCKED`，待 Owner 单独授权）。
* B07 Full 协议（仍 `BLOCKED_BY_B04A`）。
* 任何 B04A R03 协议项变更（threshold / seeds / augmentation /
  budget / 候选名单 / 架构）。
* 任何 B01 TEST 行读取（`enable_test_access` 永不被调用；
  `test_access.kind = "declarative_policy"` 是合同声明，不是
  运行时计数）。

---

## 2. 修改与新增文件

| 状态 | 路径 | 说明 |
|---|---|---|
| Modified | `src/topper_perception/neural/slp8_region_mini.py` | 新增 B04A 协议常量 + protocol dispatch + B04A 验证器 + B04A orchestrator + B04A 写产物 + per-seed `run_one_candidate` 支持；**不**修改 B04 历史行为（`B04_CANDIDATE_NAMES`、B04 validator、b04 `run_mini` 路径字节级保持；只多一层 dispatcher）。`MiniConfig` 增加 `protocol` 和 `seeds` 字段。 |
| Modified | `scripts/run_slp8_region_mini.py` | 新增 `--synthetic-cpu-smoke-b04a` 与 `--no-write` 模式；B04 历史路径（`--validate-config` / `--synthetic-cpu-smoke` / 真实 B01）字节级保持。 |
| New | `scripts/smoke_b04a_runner_integration.py` | 独立 B04A runner-integration smoke。`--no-write` 输出单行 `B04A_SMOKE_NO_WRITE ...`；写作模式输出 run-level bundle + 互斥的 `DONE/FAILED/STOPPED.json`；默认拒绝覆盖已存在产物。 |
| New | `tests/test_b04a_runner_integration.py` | **58** 个聚焦测试（B04 backward-compat + B04A dispatch + 候选限制 + seeds + all_seeds_must_succeed + 0-3 decision + near-tie + budget resume + identity mismatch + 输出互斥 + TEST=0 + B04 回归）。**全部通过。** |
| New | `docs/tasks/TASK_SLP_B04A_RUNNER_INTEGRATION_SMOKE_v0.1.md` | 任务合同（本文件配套）。 |
| New | `docs/stage_reports/S2_B04A_RUNNER_INTEGRATION_SMOKE_v0.1.md` | 本文件。 |
| Modified | `docs/PROJECT_STATUS.md` | S2_B04A 行状态从 `IMPLEMENTATION_SMOKE_ACCEPTED / RUNNER_INTEGRATION_NOT_STARTED` 更新为 `RUNNER_INTEGRATION_READY_FOR_REVIEW`；下一 Gate 明确写为 `B04A-MINI-RUN` BLOCKED + 待 Owner 单独授权。 |
| Modified | `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md` | B04A 段状态同步；下一 Gate 明确。 |
| **Unchanged (frozen)** | `configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json` | 协议冻结配置；runner-integration 任务**不**修改此文件。 |
| **Unchanged (frozen)** | `src/topper_perception/neural/slp8_region_models.py` | ResUNet-lite / DeepLabV3+-lite / SmallUNet 结构与参数；B04A implementation smoke 已锁定。 |
| **Unchanged** | `tests/test_b04a_implementation.py`, `tests/test_b04a_protocol_validator.py` | 已验收的 B04A 实现 / 协议测试；本任务不重写。 |

未触碰：

* B01 freeze tables（A06 SHA `024f5abe...` 不动）。
* B04 R05 历史 EXP-ID `EXP-SLP-B04-PM-REGION-MINI-20260828-AUTODL-R05`
  与数值（`0.439625` / `0.051631`）。
* `tests/test_slp8_region_mini.py` 15+ 个 B04 mini 测试（仅
  通过 `run_mini` 接口调用验证向后兼容）。
* `docs/SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md` /
  `docs/EXPERIMENT_GOVERNANCE_AND_GPU_EXECUTION_PLAN_v0.1.md` 等治理
  文档（路线/治理未变）。
* `docs/PROJECT_STATUS.md` 中非 B04A 行的其它状态（PoPu P5.2-C 等
  已完成阶段不动）。

附加产物（已声明、可审计）：

| 路径 | 性质 | 说明 |
|---|---|---|
| `outputs/reports/b04a_runner_integration_smoke_v0.1.json` | 声明产物（默认 git-ignored outputs 区内） | `scripts/smoke_b04a_runner_integration.py` 写出；`--force` 才能覆盖。 |
| `outputs/experiments/_b04a_runner_integration_smoke/` | 声明产物 | `--force` 时被 nuke 后重写；包含 manifest、status、candidate_decision、budget_report、logs/run.log、per-seed `checkpoints/<candidate>/seed_<seed>/{last,best}.pt` 与互斥的 `DONE.json` / `FAILED.json` / `STOPPED.json`。 |
| `outputs/_b04a_smoke_test/` | 调试用临时输出（来自 CLI 的手工 smoke） | 已用 `shutil.rmtree` 清理；不纳入交付。 |

---

## 3. Protocol / profile dispatch 设计

`validate_mini_config(cfg)` 不再是单协议 validator；它按
`config_version` 派发：

```text
config_version = "slp8_region_mini_v0.1"            -> B04   (frozen)
config_version = "slp8_pm_architecture_expansion_mini_v0.1" -> B04A  (frozen)
any other config_version                       -> ConfigValidationError (fail-closed)
```

- `_CONFIG_VERSION_TO_PROTOCOL` 是协议 → 名字的查找表（新增），
  集中维护，不在函数体里散 `if/elif`。
- `_validate_b04_mini_config` 是 B04 历史 validator 主体（**完全**
  从原 `validate_mini_config` 搬出，字节级保持；不在函数体内拆
  任何 task_id if/else）。
- `_validate_b04a_mini_config` 是 B04A 新增 validator（新增），
  走完候选 + 训练 + 数据集 + metrics + 资源 + lifecycle + TEST
  合同。
- `_build_b04_mini_config` / `_build_b04a_mini_config` 是协议
  专属 `MiniConfig` 构造器（新增 dispatcher + 共享 `MiniConfig`
  dataclass；B04 构造器字节级保持）。
- `run_mini` 仍是 B04 orchestrator（不变）。`run_mini_b04a`
  是 B04A orchestrator（新增），独立的 `B04ARunResult`。
- `write_mini_artifacts` 仍是 B04 artifact writer（不变）。
  `_write_b04a_run_bundle` / `_write_b04a_seed_artifacts` /
  `_write_b04a_candidate_aggregate` 是 B04A 新增 writer。
- 共享 `run_one_candidate`：B04 默认行为完全不变（`seed` 默
  认 `None` 时回退到 `config.seed`）；B04A 给每 seed 独立
  `checkpoint_dir` 子目录。

这避免了"在每个函数里 task_id if/else" 的散乱模式，同时保留
了 B04 的字节级历史行为（B04 走原代码路径）。

---

## 4. B04A 关键 gate 摘要

| Gate | 来源 | B04A 行为 |
|---|---|---|
| Protocol dispatch | `config_version` | 未知版本 fail-closed |
| Active candidate set | `B04A_ACTIVE_CANDIDATE_NAMES` | 仅三个注册候选；多 / 少 / 重复 / forbidden（`TinyFCN`, `SegFormer-B0`）都 fail-closed |
| `exact_parameter_count` | `B04A_EXACT_PARAMETER_COUNTS` | 118,121 / 120,809 / 53,449 必须精确匹配 |
| Seeds | `B04A_SEEDS = (42, 123, 2026)` | 精确三元；漏 / 多 / 错值都 fail-closed |
| Augmentation | `augmentation_policy_per_candidate` | 三候选都必须是 `"none"` |
| Resource budget | `B04A_RESOURCE_BUDGET` | 45 / 135 / 8192；resume 恢复 candidate-level 与 total-level 累计 |
| `all_seeds_must_succeed` | `_b04a_aggregate_candidate` | 任一 seed FAILED/STOPPED/非有限/class collapse/worst floor/per-region floor → 候选 INFEASIBLE，禁 partial mean |
| Feasibility threshold | `B04A_FEASIBILITY_THRESHOLD = 0.355644` | 与 B02 `0.205644 + 0.15` margin 冻结 |
| Decision | `_b04a_advance_decision` | 0/1/2/3-feasible；3 → top 2 by `macro_iou_mean`；`\|diff\|<0.02` 时 prefer lower parameter count |
| Identity | `CheckpointIdentity` + `_b04a_identity_block` | 每 seed / 每候选 / 每 checkpoint 都有 task_id / candidate / model_version / seed / config_sha256 / git_commit / git_dirty / data_manifest_sha256 / split_sha256 |
| Output | `_write_b04a_run_bundle` | DONE/FAILED/STOPPED 互斥；JSON top-level / CSV sibling / log first line 各按协议 `identity_hard_gate.carrier_format_by_file_type` |
| TEST=0 | top-level + dataset `test_access_policy` | `this_run_loads_test=False`、`test_access_in_this_run='denied'`、`load_test_in_mini=False`、`test_access_in_mini='denied'`；smoke summary 用 `kind="declarative_policy"` 合同声明 |

---

## 5. 验证结果

### 5.1 B04A runner-integration 测试（`tests/test_b04a_runner_integration.py`）

```
58 passed in 95.86s (0:01:35)
```

按类拆分：

* `TestProtocolDispatch` (4)：B04 dispatch、B04A dispatch、未知
  版本、缺 `config_version`。
* `TestB04BackwardCompatibility` (4)：B04 validator 接受、Miniconfig
  形状不变、B04 配置文件字节级保持、`run_mini` 仍可调通。
* `TestB04ACandidateRestrictions` (6)：TinyFCN / SegFormer-promoted
  / unknown / mixed-into / duplicate / missing / exact_parameter
  count off-by-one。
* `TestB04ASeedsContract` (5)：常量精确三元、Miniconfig 匹配、
  漏 / 多 / 单 seed 退化。
* `TestB04AAllSeedsMustSucceed` (7)：全过 → feasible；任一
  failed/stopped/class collapse / worst-subject floor / per-region
  floor → INFEASIBLE；partial mean 禁止。
* `TestB04ACandidateDecision` (4)：0/1/2/3-feasible 全部覆盖。
* `TestB04ANearTieTiebreak` (3)：0.02 margin 内更简单模型赢；margin
  外不触发；margin 常量核对。
* `TestB04AResourceBudget` (6)：常量、配置验证、错值拒绝、
  accumulator resume。
* `TestB04AIdentityCheckpointOutput` (5)：identity 必含字段、
  每 seed model_version、identity mismatch 拒绝、相同 identity
  接受、输出互斥。
* `TestB04ATestZero` (5)：validator 拒绝 `load_test=True`、
  validator 拒绝 dataset 层 `load_test_in_mini=True`、smoke 脚本
  源码扫描无 B01 TEST 调用、runner 源码扫描无 B01 TEST 调用。
* `TestB04AEndToEndSmoke` (6)：`--no-write` 单行 / `--force` 写
  完整 bundle + 互斥 terminal file / `--no-write` 不动磁盘 /
  CLI `--synthetic-cpu-smoke-b04a` --no-write / 拒绝非 B04A 配置
  / 小预算容忍 `--no-write` 路径。
* `TestB04Regression` (1)：B04 SmallUNet / registry /
  `B04A_EXACT_PARAMETER_COUNTS` 一致性 + 118,121 参数。

### 5.2 B04A 协议验证器（仍冻结）

```
$ python scripts/validate_b04a_protocol.py configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json
OKs: 30, Errors: 0 — VALIDATION PASSED
```

合同未改；30 OKs / 0 errors 保持。

### 5.3 B04 mini 回归（向后兼容）

`tests/test_slp8_region_mini.py` 的 15+ 个聚焦 B04 测试（包含
`TestSmallUnetArchitecture` / `TestCandidateRegistry` /
`build_synthetic_dataset` / `TestPredict`）仍可通过
`test_b04_mini_runner_still_callable` 等 B04 路由调用 — 详见
`tests/test_b04a_runner_integration.py::TestB04BackwardCompatibility`
与 `TestB04Regression`。

### 5.4 B04A 端到端 synthetic CPU smoke

```
$ python scripts/smoke_b04a_runner_integration.py --no-write
B04A_SMOKE_NO_WRITE protocol=B04A config_version=slp8_pm_architecture_expansion_mini_v0.1 \
  candidates=3 seeds=3 terminal_state=DONE feasible=0 failed=0 stopped=0 \
  advanced=0 near_tie_applied=False all_seeds_attempted=True any_seed_feasible=False
```

关键确认：

* `protocol=B04A` — dispatcher 把 B04A 配置送到了 B04A 协议路径。
* `candidates=3 seeds=3` — 三候选 × 三 seed 的编排被完整执行。
* `terminal_state=DONE` — 无 FAILED / STOPPED；所有 9 个 (candidate,
  seed) 编排都跑完了。
* `feasible=0` — synthetic 数据（4 train / 2 val / 1 epoch）显然
  过不了 0.355644 阈值；smoke **不**试图给真实排名，只验证编排路
  径。
* `advanced=0` — 0 feasible → `MINI_NOT_FEASIBLE`（合契约）。
* `near_tie_applied=False` — 0 feasible 不触发 tiebreak。
* `all_seeds_attempted=True` — 9 个组合全跑完；`any_seed_feasible
  =False` — 无 seed 命中 0.355644（这正是 synthetic 数据的预期结
  果，不是 bug）。

写作模式（`--force`）下，smoke 写出：

* `outputs/reports/b04a_runner_integration_smoke_v0.1.json`
* `outputs/experiments/_b04a_runner_integration_smoke/manifest.json`
* `outputs/experiments/_b04a_runner_integration_smoke/resolved_config.json`
* `outputs/experiments/_b04a_runner_integration_smoke/input_manifest_hashes.json`
* `outputs/experiments/_b04a_runner_integration_smoke/environment.json`
* `outputs/experiments/_b04a_runner_integration_smoke/status.json`
* `outputs/experiments/_b04a_runner_integration_smoke/candidate_decision.json`
* `outputs/experiments/_b04a_runner_integration_smoke/budget_report.json`
* `outputs/experiments/_b04a_runner_integration_smoke/logs/run.log`
* 互斥的 `DONE.json` / `FAILED.json` / `STOPPED.json` 之一
* 9 个 `checkpoints/<candidate>/seed_<seed>/{last,best}.pt`
* 每个 seed 子目录的 `epoch_metrics.csv`、
  `metrics_summary.json`、`reload_consistency.json`、
  `worst_subject.json`、`predictions_manifest.csv`（CSV 都带同名
  `.identity.json` sibling）
* 每个 candidate 根目录的 `aggregate_decision.json`

### 5.5 `python -m py_compile` 与 `git diff --check`

```
$ python -m py_compile src/topper_perception/neural/slp8_region_mini.py
$ python -m py_compile scripts/run_slp8_region_mini.py
$ python -m py_compile scripts/smoke_b04a_runner_integration.py
$ python -m py_compile tests/test_b04a_runner_integration.py
(no output — all compile cleanly)
```

`git diff --check` 干净。

---

## 6. 已验证事实

1. `validate_mini_config` 按 `config_version` 派发，B04 走 B04
   路径、B04A 走 B04A 路径、未知版本 fail-closed；不存在函数体内
   `if/elif` on `task_id`。
2. B04 `MiniConfig.protocol == "B04"`、`seeds == (42,)`、
   `candidates == (slp8_tiny_fcn_v0.1, slp8_small_unet_v0.1)`。
3. B04A `MiniConfig.protocol == "B04A"`、`seeds == (42, 123, 2026)`、
   `candidates == (slp8_small_unet_v0.1, slp8_resunet_lite_v0.1,
   slp8_deeplabv3plus_lite_v0.1)`，DEFERRED 的 `slp8_segformer_b0_v0.1`
   被自动从 active 集合剔除。
4. `B04A_EXACT_PARAMETER_COUNTS` 严格匹配 118,121 / 120,809 / 53,449；
   off-by-one 立即 fail-closed。
5. 三 seed 严格 42 / 123 / 2026；漏 seed / 多 seed / 单 seed 退化
   全部 fail-closed。
6. `all_seeds_must_succeed=true` 强制：任一 per-seed 失败
   （FAILED / STOPPED / 非有限 / class collapse / worst-subject
   floor / per-region floor）→ 整个候选 INFEASIBLE；partial-seed
   mean 已被测试显式拒绝。
7. 0/1/2/3-feasible 决策规则全部测试通过：0 → `MINI_NOT_FEASIBLE`、
   1 → 单 candidate 晋级、2 → 双晋级无 champion、3 → top 2 + near-tie
   tiebreak。
8. Near-tie tiebreak：`|diff| < 0.02` 时 prefer lower parameter
   count（118,121 < 120,809）；`|diff| ≥ 0.02` 时不触发。
9. Resource budget 45 / 135 min；resume 时
   `BudgetAccumulatorState.restore` 把 candidate-level 与
   total-level 累计秒数都恢复到新 `ResourceBudgetState`，不重置
   也不重复计算。
10. Identity mismatch 在 resume 时抛 `ResumeIdentityError`；相同
    identity 接受。
11. 输出目录冲突（`DONE.json` / `FAILED.json` / `STOPPED.json`
    sentinel 或任何非 `.gitkeep` 文件）默认拒绝；`--force` 可覆
    盖；terminal file 三选一互斥。
12. TEST=0：runner 与 smoke 源码扫描确认不导入也不调用
    `enable_test_access` / `TestLeakageError` / `load_b01_freeze_tables` /
    `compute_class_stats(ml_split="test")`；smoke summary 的
    `test_access.kind == "declarative_policy"`，不是运行时计数。
13. B04 mini 回归：historical `validate_mini_config` 主体 / 协议
    检查 / 资源预算 / checkpoint 互斥 / `run_mini` 端到端仍可调
    通；B04 配置 / 输出 / EXP-ID 字节级保持。
14. B04A 协议验证器 `scripts/validate_b04a_protocol.py` 对冻结
    配置返回 30 OKs / 0 errors（合同未改）。
15. `git diff --check` 干净；本工作树未 commit / 未 push / 未创建
    PR。

---

## 7. 推断

1. 把 B04 与 B04A 两个协议映射到同一 `MiniConfig` 形状但不同
   `protocol` / `seeds` 字段，比把两个 runner 拆成两个模块更小
   地改动现有行为；同时 `run_one_candidate` 的 `seed` /
   `checkpoint_dir` 注入足以让 B04A 走独立 per-seed 子目录而
   B04 走原 `checkpoints/<candidate>/` 路径，不污染 B04 的
   输出布局。
2. `_b04a_aggregate_candidate` 把 `all_seeds_must_succeed` 写
   在一个聚合函数里，比把它分散到 `run_mini_b04a` 主循环里更
   易测、易审计；本任务用 7 个单元测试直接打聚合层，覆盖了
   6 个失败注入路径 + 1 个 happy path。
3. B04A synthetic CPU smoke 在 `n_train=4 / n_val=2 / max_epochs=1`
   的极小预算下完成 3 候选 × 3 seed = 9 个 (candidate, seed) 组合
   的端到端编排；wall time 秒级，与协议预算 45/135 min 拉开 4
   个数量级。Runner 的 budget 检查未触发（`budget_status=ok`），
   编排路径与 identity 写出路径均被验证。
4. Identity carrier format 在 B04A 中通过 `identity_hard_gate` 的
   7 个必含字段（`experiment_id` / `task_id` / `git_commit` /
   `git_dirty` / `config_sha256` / `data_manifest_sha256` /
   `split_sha256` / `model_version`）覆盖；checkpoint 内部
   `identity` dict + CSV sibling `.identity.json` + log 首行 JSON
   三种承载体保持一致。

---

## 8. 未验证 / NOT RUN

1. 真实 B01 GPU Mini（仍 `BLOCKED`；需要 Owner 单独授权 + 单独
   `TASK-ID`；本任务仅交付 runner-integration 合规）。
2. CUDA Smoke：host `torch==2.13.0+cpu`，`torch.cuda.is_available()
   == False` → 显式 `NOT_RUN`；CPU smoke 已 mandatory 执行。
3. B01 freeze tables：`scripts/smoke_b04a_runner_integration.py`
   不加载；`run_mini_b04a` 在 synthetic CPU 路径下不调用
   `load_b01_freeze_tables`。
4. 真实 TEST 读取：未发生（`TestB04ATestZero` 5 个测试双重验证）。
5. B04A R03 协议 / 配置文件 / 已验收实现测试（`test_b04a_implementation.py` /
   `test_b04a_protocol_validator.py`）：本任务不重写。
6. B07 Full 协议：仍 `BLOCKED_BY_B04A`；未被本任务开启。
7. SegFormer-B0 实装：协议 DEFERRED；本任务不实现。
8. `ruff` / `pre-commit`：当前环境未安装，记为 `NOT RUN`。
9. 任何 Owner / 网页 GPT 阶段的二审意见；本任务交付的是
   "runner integration ready for review"，**不**包括
   "runner integration accepted" / "GPU Mini authorized" /
   "Mini complete" / "B07 ready"。

---

## 9. 限制与禁止结论

**限制**

1. 当前阶段名 = `RUNNER_INTEGRATION_ACCEPTED / GPU_MINI_NOT_AUTHORIZED`；这只
   意味着 runner 与合成 CPU 编排通过 Reviewer 验收，**不**等同于
   "GPU Mini 完成"或"候选可以排名"。
2. 协议冻结 ≠ Mini / Full 完成。
3. 标签为 `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED` /
   `source_review_status=NOT_REVIEWED`；danaLab / uncover only。
4. 真实 GPU Mini 仍需 Owner 单独授权 + `TASK-SLP-B04A-MINI-RUN`
   任务。
5. CUDA Smoke 在本机 CPU-only build 上未跑；GPU 可移植性需后续
   真实 GPU 环境。

**禁止结论**

- ❌ B04A Mini 完成；新候选优于 SmallUNet；架构比较形成最终排名。
- ❌ 适用于产品、硬件、舒适性、医学、整夜稳定性、气囊控制。
- ❌ SegFormer 临时纳入；TEST 结果可见或可推断。
- ❌ 压力值是 kPa；标签是人工像素级标注。
- ❌ 因 runner integration 已接受而自动推导 `GPU_MINI_AUTHORIZED` 或直接进入 `B04A-MINI-RUN`。
- ❌ 性能排名或 champion 选择（仅协议规定的 0/1/2/3 决策）。

---

## 10. Reviewer checklist

- [x] B04 / B04A 协议 dispatch 已实现（`_CONFIG_VERSION_TO_PROTOCOL`
  + `_protocol_of_config`，**不**在函数体内散 `task_id if/else`）。
- [x] B04 配置 / `B04_CANDIDATE_NAMES` / B04 historical validator 字
  节级保持；B04 mini smoke 仍可调通。
- [x] B04A 配置通过 dispatcher 接受并产出
  `protocol="B04A"`、`seeds=(42,123,2026)`、3 候选 active 集合
  的 `MiniConfig`。
- [x] 未知 `config_version` 被 `ConfigValidationError` 拒绝。
- [x] B04/B04A 候选混用、`TinyFCN` 进 B04A active、`SegFormer-B0`
  promote 出 `DEFERRED`、重复 active 候选、缺 active 候选、
  `exact_parameter_count` 偏差均被 fail-closed。
- [x] `all_seeds_must_succeed=true`：任一 per-seed 失败 → 候选
  INFEASIBLE；partial-seed mean 显式禁止。
- [x] 0/1/2/3-feasible 决策 + near-tie tiebreak（`|diff|<0.02`
  prefer lower parameter count）均被显式测试。
- [x] Resource budget 45 / 135 / 8192；`BudgetAccumulatorState.restore`
  把 candidate-level 与 total-level 累计秒数带过去。
- [x] Identity mismatch → `ResumeIdentityError`；同 identity 接受。
- [x] Output 目录冲突被默认拒绝；DONE/FAILED/STOPPED 互斥；
  `check_output_dir_safety` 与 B04 runner 行为一致。
- [x] TEST = 0（runner 与 smoke 源码扫描 + validator
  `test_access_policy` 双重验证）。
- [x] B04 mini 回归仍可通过。
- [x] `python -m py_compile` 干净；`git diff --check` 干净。
- [x] Stage name 在 `docs/PROJECT_STATUS.md` S2_B04A 行 =
  `RUNNER_INTEGRATION_ACCEPTED / GPU_MINI_NOT_AUTHORIZED`。
- [x] 没有 `GPU_MINI_AUTHORIZED` / `MINI_COMPLETE` / `B07_READY` 声明。
- [x] 下一 Gate 写明：`B04A-MINI-RUN` 仍 `BLOCKED`，需要 Owner
  单独授权 + 真实 GPU 环境。

---

## 11. R02 ITERATE 修订（2026-08-29）

Codex Reviewer 在 R01 提交后指出三处合同级问题。本节记录针对
R02 ITERATE 的修订与自检。

### 11.1 修订项 1 — 真实 B04A 路径错误调用单-seed B04 runner

**问题**：R01 的 `scripts/run_slp8_region_mini.py::_run_real_b01`
无条件调用 `run_mini`，导致 B04A 真实授权路径只跑 `config.seed=42`、
不执行 `[42,123,2026]`、不应用 `all_seeds_must_succeed`、使用 B04
90 分钟总预算、不产生 B04A 候选级聚合与 Top-2 决策。

**修订**：
* `_run_real_b01` 重写为 `config.protocol` 派发入口；通过
  `validate_mini_config` + `build_mini_config` 之后路由到
  `_run_real_b01_b04` 或 `_run_real_b01_b04a`；未知 protocol
  fail-closed BEFORE 任何 training 产物被写入。
* 抽出共享 B01 输入合同加载函数 `_load_b01_freeze_and_contract`：
  freeze manifest 一致性 + `B01FreezeSnapshot` + `verify_b01_contract`
  走同一条路径；B01 合同违反仍 fail-closed（两协议皆然）。
* 新增 `_run_real_b01_b04`（B04 真实路径，调用 `run_mini`，B04
  90 分钟总预算）。
* 新增 `_run_real_b01_b04a`（B04A 真实路径）：
  - 自动构建 per-(candidate, seed) resume 映射
    `checkpoints/<candidate>/seed_<seed>/last.pt`（新函数
    `_auto_detect_resume_candidates_b04a`，与 B04 的
    `_auto_detect_resume_candidates` 行为一致）。
  - 设置 B04A 资源预算 `45/135/8192`（每候选 45 分钟 / 候选，
    135 分钟总上限，CUDA 8 GiB）。
  - 调用 `run_mini_b04a`（已带 fail-closed 跨协议守卫拒绝非 B04A
    protocol；新增 `run_mini` 顶部 fail-closed 跨协议守卫拒绝非
    B04 protocol）。
  - 通过 `_write_b04a_run_bundle` 写 B04A schema 产物
    （manifest / candidate_decision / budget_report / per-seed
    identity siblings / 9 个 `checkpoints/<candidate>/seed_<seed>/{last,best}.pt`）。
  - 互斥的 terminal file（DONE/FAILED/STOPPED）。

**自检覆盖**：
* `TestRunRealB01Dispatch::test_b04_real_path_calls_b04_helper` —
  B04 config 派发到 `_run_real_b01_b04`（不调用 b04a）。
* `TestRunRealB01Dispatch::test_b04a_real_path_calls_b04a_helper` —
  B04A config 派发到 `_run_real_b01_b04a`（不调用 b04）。
* `TestRunRealB01Dispatch::test_b04a_helper_passes_b04a_budget_and_resume_map` —
  B04A helper 构造的预算严格为 45/135/8192；`run_mini_b04a` 收到的
  `resume_from_per_candidate_seed` 是 per-(candidate, seed) 映射
  （本测试构造的 2 个 seed 命中且只有命中的 seed 出现在 map 中）。
* `TestRunRealB01Dispatch::test_unknown_protocol_fail_closed_in_real_path` —
  协议被 monkeypatch 为 "B99" 时 `_run_real_b01` 抛
  `MiniProtocolError("not recognised")`，且没有 `checkpoints/`
  子目录被写入。
* `TestRunMiniCrossProtocolGuards::test_run_mini_rejects_b04a_config` —
  直接 `run_mini(config=b04a_cfg, ...)` 抛 `MiniProtocolError` 匹配
  "B04A"。
* `TestRunMiniCrossProtocolGuards::test_run_mini_b04a_rejects_b04_config` —
  直接 `run_mini_b04a(config=b04_cfg, ...)` 抛 `MiniProtocolError`
  匹配 "B04A"。

### 11.2 修订项 2 — `validate-only` 对 B04A 仍写 B04 identity

**问题**：R01 的 `_run_validate_config` 接受 B04A config，但仍写
`TASK_ID` / `MINI_VERSION` / `B04_CANDIDATE_NAMES` / `B04_MAX_PARAMETERS`。

**修订**：`_run_validate_config` 拆分为派发入口 + 两条独立分支：
* `_run_validate_config_b04` — 唯一可以引用 `TASK_ID` /
  `MINI_VERSION` / `B04_CANDIDATE_NAMES` / `B04_MAX_PARAMETERS` 的
  路径；`status.json` / `DONE.json` / `input_manifest_hashes.json`
  写 B04 身份。
* `_run_validate_config_b04a` — 唯一可以引用 `B04A_TASK_ID` /
  `B04A_CONFIG_VERSION` / `B04A_ACTIVE_CANDIDATE_NAMES` /
  `B04A_MAX_PARAMETERS` / `B04A_SEEDS` /
  `B04A_FORBIDDEN_CANDIDATE_NAMES` / `B04A_FEASIBILITY_THRESHOLD`
  的路径；`status.json` / `DONE.json` / `input_manifest_hashes.json`
  写 B04A 身份；额外记录 `deferred_candidates`（从 config 中
  `role=DEFERRED` 解析，SegFormer-B0 保持 DEFERRED）。
* 静态检查：B04A 分支不引用 `TASK_ID` / `MINI_VERSION` /
  `B04_CANDIDATE_NAMES` / `B04_MAX_PARAMETERS`（用 AST 文本扫描
  验证）。

**自检覆盖**：
* `TestRunValidateConfigIdentity::test_b04_validate_only_writes_b04_identity` —
  通过 `subprocess.run` 调起 `scripts/run_slp8_region_mini.py
  --validate-config`（B04 config），读 `status.json` /
  `DONE.json` / `input_manifest_hashes.json`，断言所有身份字段
  都是 B04，并断言 B04A 任务/配置 ID、SegFormer-B0、协议名都不在
  产物中。
* `TestRunValidateConfigIdentity::test_b04a_validate_only_writes_b04a_identity` —
  同样通过 subprocess 跑 B04A `--validate-config`，断言
  `status.json` 中 `task_id=B04A_TASK_ID`、`config_version=B04A_CONFIG_VERSION`、
  `protocol=B04A_PROTOCOL_NAME`、`registered_candidates=B04A_ACTIVE_CANDIDATE_NAMES`、
  `model_parameter_cap=300_000`、`feasibility_threshold=0.355644`、
  `seeds=[42,123,2026]`、`forbidden_candidates=B04A_FORBIDDEN_CANDIDATE_NAMES`、
  `deferred_candidates` 含 `slp8_segformer_b0_v0.1`；并断言 B04
  任务 ID / 协议名不在 B04A 产物中。
* CLI 手动 smoke 已确认（脚本运行后 `status.json` 与 `DONE.json`
  写入 B04A 身份；B04 输入得到 B04 身份）。

### 11.3 修订项 3 — `_b04a_advance_decision` 拓扑边界与 tiebreak 修正（R02 实现，**已被 R03 取代**）

**问题**：R01 的 `_b04a_advance_decision` 只比较 1st-vs-2nd 并交换
顺序，不改变 Top-2 晋级集合；不处理 2nd-vs-3rd near-tie；不区分
参数量差异 10% 内外的 tiebreak 基础；worst-subject 缺失不
fail-closed。

**R02 修订（已被 R03 取代）**：R02 完整替换了
`_b04a_advance_decision`、新增 `B04AAdvanceTiebreak` /
`B04AAdvanceDecision` dataclass、把 `B04ARunResult.advance_decision`
plumb 到 `candidate_decision.json`。**但 R02 的 2nd boundary
比较的是 surviving top1/top2，而非 1st-round REJECTED 候选** —
这一缺陷在 R03 ITERATE 中被 Codex Reviewer 指出并修正（见 §12）。
R02 段余下文本仅作为历史记录保留。
* `B04AAdvanceTiebreak`（每条边界审计记录）：`pair` /
  `macro_iou_difference` / `parameter_difference_ratio` /
  `tiebreak_basis` / `selected` / `rejected`。
* `B04AAdvanceDecision`（3-feasible 整体决策）：`advanced` /
  `near_tie_applied` / `near_tie_margin` / `tiebreaks`。
* `_b04a_pair_tiebreak(higher, lower)` 实现四类基础：
  - `none`：`|Δiou| >= 0.02` 保持原序。
  - `parameter_count`：`ratio = |p_h - p_l| / min(p_h, p_l) > 0.10`，
    选参数更少者。
  - `worst_subject_iou`：`ratio ≤ 0.10` 且 worst_subject_iou 不同，
    选更高者。
  - `worst_subject_iou_then_param`：worst_subject_iou 在 1e-9
    内并列，选参数更少者。
  - `failed_no_worst_subject`：worst_subject_iou 缺失或非有限 →
    `selected=None, rejected=None`，`advanced=()`，
    `near_tie_applied=True`，上游 terminal_state 提升为 `FAILED`。
* 排序键 `(-macro_iou_mean, name)` (DESC, ASC) — 稳定且对输入
  顺序无关。
* 2nd boundary 用 surviving top1/top2（依 1st-loser 是 top1 还是
  top2）；2nd boundary near-tie 会让更简单的 top3 替换 1st-loser
  槽位。
* `B04ARunResult` 新增 `advance_decision: B04AAdvanceDecision` 字段；
  `as_dict` 与 `candidate_decision.json` 都写入该决策与每条边界
  审计。

**自检覆盖**（`TestB04ANearTieTiebreakExt` + `TestB04ANearTieTiebreak`）：
* `test_first_vs_second_near_tie_both_still_advance` — Top1 vs Top2
  near-tie，DeepLabV3+-lite 不进入 advanced。
* `test_second_vs_third_near_tie_with_simpler_third` — 1st boundary
  交换后 surviving top2 与 top3 仍 near-tie，simpler top3 替换
  1st-loser；advanced = (B, C)。
* `test_param_diff_within_10pct_worst_subject_tiebreak` — ResUNet
  (120,809) vs SmallUNet (118,121)，ratio 0.0228 < 0.10，
  basis=`worst_subject_iou`，SmallUNet 胜（worst_subject_iou 更高）。
* `test_param_diff_over_10pct_fewer_params_wins` — 200k vs 100k
  (ratio 1.0 > 0.10)，basis=`parameter_count`；advanced = (small, big)。
* `test_missing_worst_subject_fail_closed` — Top2 中一个
  worst_subject_iou=None，basis=`failed_no_worst_subject`，
  `advanced=()`，`near_tie_applied=True`。
* `test_non_near_tie_strict_order` — 0.4 vs 0.1 巨大 gap，
  basis=`none` 严格按 macro_iou 排序。
* `test_input_order_independence` — 输入字典顺序（forward /
  reverse / 乱序）不影响 `advanced` / `tiebreaks` / 各条
  `pair` / `tiebreak_basis` / `selected` / `rejected`。
* 既有 `TestB04ACandidateDecision` 与 `TestB04ANearTieTiebreak` 已
  迁移到新 `B04AAdvanceDecision` dataclass API（`decision.advanced`
  / `decision.near_tie_applied` / `decision.tiebreaks`），并补
  充了 `pair` / `macro_iou_difference` /
  `parameter_difference_ratio` 的精确断言。

### 11.4 R02 自检结果

* `tests/test_b04a_runner_integration.py`：**73 / 73** 通过（含
  全部 R02 新增用例与既有 R01 用例）。
* `tests/test_b04a_implementation.py`：79 / 79 通过。
* `tests/test_b04a_protocol_validator.py`：全过。
* `tests/test_slp8_region_models.py`：38 / 38 通过。
* `tests/test_slp8_region_mini.py`（SmallUNet / registry /
  synthetic）：15 / 15 通过。
* B04 mini regression（`TestSmallUnetArchitecture` /
  `TestCandidateRegistry` / `build_synthetic_dataset` /
  `TestPredict`）：15 / 15 通过。
* `tests/test_check_markdown_links.py`：6 / 6 通过。
* `git diff --check` 干净。
* `python -m py_compile src/topper_perception/neural/slp8_region_mini.py
  scripts/run_slp8_region_mini.py
  scripts/smoke_b04a_runner_integration.py
  tests/test_b04a_runner_integration.py` 全过。
* CLI 手动 smoke：B04 `--validate-config` 写 B04 身份；B04A
  `--validate-config` 写 B04A 身份（含 `deferred_candidates` /
  `forbidden_candidates` / `seeds` / `feasibility_threshold` /
  `model_parameter_cap=300_000`）。
* `scripts/smoke_b04a_runner_integration.py --no-write` 成功输出
  单行 `B04A_SMOKE_NO_WRITE` 摘要。
* `scripts/validate_b04a_protocol.py
  configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json`
  30 OKs / 0 errors（合同未改）。
* TEST = 0（runner / smoke 源码扫描 + validator
  `test_access_policy` 双重验证）。

### 11.5 R02 ITERATE 后的状态

* 阶段名 = `RUNNER_INTEGRATION_READY_FOR_REVIEW`（保持 R01）。
* 下一 Gate = **Codex Reviewer R02 独立验收**。本任务 R02 修订
  自检通过；Codex R02 独立复跑 / 独立验收**尚未进行**。
* **不得**声明 `RUNNER_INTEGRATION_ACCEPTED` /
  `GPU_MINI_AUTHORIZED` / `MINI_COMPLETE` / `B07_READY`。
* GPU Mini 继续 `BLOCKED`；`B07` 仍 `BLOCKED_BY_B04A`。
* 全部交付物均**未** commit / **未** push / **未** 创建 PR；本
  任务的 handoff 完成后停止，等待 Codex R02 独立验收。
* **注**：R02 的 2nd boundary 算法被 Codex R03 进一步指出
  缺陷；R03 ITERATE 已在 §12 修复并替换 R02 实现。R02 段仅作
  历史记录保留。

---

## 12. R03 ITERATE 修订（2026-08-29）

Codex Reviewer 在 R02 复跑后指出 2nd boundary 拓扑边界
的合同缺陷：R02 的实现让 1st-round **胜者**（surviving
top1/top2）再次进入 2nd boundary，而非 1st-round **败者**
（rejected 候选）。R03 修复此 bug 并删除 dedup 回填逻辑。

### 12.1 修订项 — 2nd boundary 拓扑边界（Reviewer 复现案例）

**问题**：R02 的 `_b04a_advance_decision` 在 1st 轮 tiebreak
之后，让 surviving top1/top2（即 1st 轮的胜者所在的槽位）
再与 top3 比较。这与合同"first_loser 必须进入第二轮"的
语义不一致。

**Reviewer 复现案例**：

```
A: macro_iou=0.500, params=100,000
B: macro_iou=0.495, params=90,000
C: macro_iou=0.478, params=80,000
```

R02 错误路径：A vs B → B 获胜 → 错误比较 B vs C → advanced
= ("B", "C")，2nd audit pair = ("B", "C")。

R03 正确路径：

1. 1st boundary：top1=A vs top2=B，diff=0.005，ratio=10k/90k
   ≈ 0.111 > 0.10，basis=`parameter_count`，B (90k) < A
   (100k)，B 获胜（B 拿 slot 1）。
2. **first_loser = A**（1st 轮 REJECTED 候选）。
3. 2nd boundary：first_loser=A vs top3=C，diff=0.022 ≥
   0.02，basis=`none`，A 获胜（A 拿 slot 2）。
4. advanced = ("B", "A")，2nd audit pair = ("A", "C")。

**绝不能得到** ("B", "C")，**绝不能** 把 B 写为 2nd
boundary 参与者。

**修订**（`src/topper_perception/neural/slp8_region_mini.py`）：
* 完整重写 `_b04a_advance_decision` 的 2nd boundary 段：
  取消 `second_pair_higher = top1 if first_loser == top2[0]
  else top2` 的错误赋值；改用 `first_loser_pair = next((kv
  for kv in feasible if kv[0] == first_loser), None)` 取出
  1st 轮 REJECTED 候选的 (name, aggregate)，再喂给
  `_b04a_pair_tiebreak(higher=first_loser_pair, lower=top3)`。
* advanced = `(first_winner, second_winner)`；不再有
  duplicate / dedup 回填逻辑。
* 防御性 fail-closed：若 `first_winner == second_winner`
  （合同不可能；若发生则视为 contract violation），返回
  `advanced=()`、`near_tie_applied=True`、完整 tiebreak 审计。
* 更新 `B04AAdvanceDecision` docstring 与
  `_b04a_advance_decision` docstring 反映正确的两轮边界
  拓扑（1st 胜者拿 slot 1，first_loser 与 top3 争 slot 2）。

### 12.2 新增 R03 回归测试（`TestB04ANearTieTiebreakR03`）

1. `test_reviewer_exact_scenario_advanced_is_b_then_a` —
   Reviewer 复现案例：A(0.500, 100k) / B(0.495, 90k) /
   C(0.478, 80k)；断言 `advanced == ("B", "A")`、2nd pair
   == ("A", "C")、`tb12.basis == "parameter_count"`、
   `tb23.basis == "none"`、`tb23.macro_iou_difference ≈ 0.022`。
2. `test_third_replaces_first_loser_when_second_boundary_near_tie` —
   A(0.500, 100k) / B(0.495, 90k) / C(0.490, 30k)；1st B 胜，
   2nd A vs C 仍是 near-tie 且 ratio=70k/30k > 0.10，C 胜；
   advanced = ("B", "C")，2nd pair = ("A", "C")，
   `tb23.basis == "parameter_count"`。
3. `test_first_boundary_no_near_tie_second_boundary_top2_vs_top3` —
   A(0.500, 100k) / B(0.40, 90k) / C(0.39, 80k)；1st boundary
   basis=none，A 胜；2nd boundary 比较 B（first_loser）vs C，
   **不是** A（surviving top1）vs C；advanced = ("A", "C")。
4. `test_first_boundary_no_near_tie_second_boundary_no_near_tie` —
   A(0.500, 100k) / B(0.40, 90k) / C(0.20, 80k)；两轮都 basis=
   none；advanced = ("A", "B")，C 被淘汰；2nd pair = ("B", "C")。
5. `test_second_boundary_fail_closed_missing_worst_subject` —
   故意让 1st 轮成功（basis=worst_subject_iou_then_param，
   A 与 B ws 相同 → p_h=p_l 保持 A），2nd 轮 ws 缺失 →
   `advanced=()`、`tb23.basis == "failed_no_worst_subject"`、
   `tb23.pair == ("B", "C")`、`tb23.selected/rejected` 都
   为 None。
6. `test_input_order_independence_under_r03_contract` —
   forward / reverse / scrambled 三种插入顺序，断言
   `advanced` 和 `tiebreaks` 完全一致；与 R02 区别在于
   tiebreak pair / selected / rejected 的具体值。
7. `test_advanced_always_two_distinct_candidates` — 永远
   `len(advanced) == 2` 且 `advanced[0] != advanced[1]`。

同时更新以下 R02 既有测试以反映 R03 正确语义：
* `test_near_tie_prefers_lower_parameter_count` — 2nd pair
  从 ("SmallUNet", "DeepLabV3+-lite") 改为
  ("ResUNet", "DeepLabV3+-lite")；`tb23.macro_iou_difference`
  从 0.195 改为 0.20（因 surviving top2 不再参与 2nd boundary）。
* `test_second_vs_third_near_tie_with_simpler_third` — 2nd
  pair 从 ("cand_B", "cand_C") 改为 ("cand_A", "cand_C")；
  `tb23.macro_iou_difference` 从 0.005 改为 0.010。
* `test_param_diff_over_10pct_fewer_params_wins` — 2nd pair
  从 ("small", "middle") 改为 ("big", "middle")。

### 12.3 R03 自检结果

* `tests/test_b04a_runner_integration.py`：**80 / 80** 通过
  （R03 新增 7 个用例 + R02 修订后的既有 73 个用例）。
* `tests/test_b04a_implementation.py`：79 / 79 通过。
* `tests/test_b04a_protocol_validator.py`：全过。
* `tests/test_slp8_region_models.py`：38 / 38 通过。
* `tests/test_slp8_region_mini.py`（SmallUNet / registry /
  synthetic + 全部 162 个 integration / contract / resume
  / determinism / CLI 用例）：162 / 162 通过。
* B04 mini regression：15 / 15 通过。
* `tests/test_check_markdown_links.py`：6 / 6 通过。
* `git diff --check` 干净。
* `python -m py_compile src/topper_perception/neural/slp8_region_mini.py
  scripts/run_slp8_region_mini.py
  scripts/smoke_b04a_runner_integration.py
  tests/test_b04a_runner_integration.py` 全过。
* CLI 手动 smoke：B04 `--validate-config` 写 B04 身份；B04A
  `--validate-config` 写 B04A 身份。
* `scripts/smoke_b04a_runner_integration.py --no-write` 输出
  `B04A_SMOKE_NO_WRITE protocol=B04A candidates=3 seeds=3
  terminal_state=DONE`。
* `scripts/validate_b04a_protocol.py` 30 OKs / 0 errors。
* TEST = 0（runner / smoke 源码扫描 + validator
  `test_access_policy` 双重验证）。

### 12.4 Codex Reviewer R03 验收状态

* 阶段名 = `RUNNER_INTEGRATION_ACCEPTED / GPU_MINI_NOT_AUTHORIZED`。
* Codex Reviewer R03 独立复跑通过：Reviewer `A/B/C` 反例、80 项集成测试、完整 162 项 B04 回归、167 项合同/模型测试、协议验证、no-write smoke、链接、编译与 `git diff --check` 全部通过。
* 下一 Gate = 发布本次验收提交；之后只有 Owner 单独授权并分配独立 EXP-ID，才可进入 `TASK-SLP-B04A-MINI-RUN`。
* **不得**声明 `GPU_MINI_AUTHORIZED` / `MINI_COMPLETE` / `B07_READY`。
* GPU Mini 继续 `BLOCKED`；`B07` 仍 `BLOCKED_BY_B04A`。
* 全部交付物仍未 commit / push / 创建 PR；当前等待发布，不自动启动 GPU Mini。
