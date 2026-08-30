# TASK-SLP-B04A-BUDGET-REPORT-FIX-v0.1

状态：`CODE_FIX_ACCEPTED / R03_OWNER_GATE_PENDING / GPU_RERUN_NOT_AUTHORIZED`

## 目标

修复 B04A run-level `budget_report.json` 将总耗时写成预算上限、将
CUDA 峰值写成固定 `0.0` 的证据载体缺陷。run-level 报告必须使用本次
运行的实际 wall-clock 秒数，并从所有 candidate/seed 的预算记录中汇总
实际最大 CUDA MiB。

## 非目标

- 不修改候选、训练参数、数据、split、晋级规则或已有 R03 产物。
- 不运行 AutoDL/GPU Mini、Full 或 TEST。
- 不覆盖既有 EXP-ID；R03 原始证据保持只读。
- 不因修复通过而自动授权 B07 或新的实验。

## 允许修改

- `src/topper_perception/neural/slp8_region_mini.py`
- `tests/test_b04a_runner_integration.py`
- 本任务单及对应 stage report
- `docs/PROJECT_STATUS.md`、`docs/SLP_AGENT_TASK_BACKLOG_v0.1.md` 的窄范围状态更新

## 必须行为

1. `budget_report.elapsed_total_seconds == B04ARunResult.wall_clock_seconds`，
   且不得等于配置预算上限，除非实测值恰好相等。
2. `budget_report.peak_cuda_mb` 必须是全部已产生 per-seed
   `budget_report.peak_cuda_mb` 的有限非负最大值。
3. 缺失、非数值、负数或非有限 CUDA 峰值必须 fail closed，不能回退到
   `0.0`。
4. 保留 run-level identity、terminal state、candidate 汇总和 B04 历史路径。
5. 增加真实 bundle writer 回归测试，明确拒绝占位值再次通过。

## 验证

- 针对新增回归测试。
- 完整 `tests/test_b04a_runner_integration.py`。
- 相关 B04/B04A regression、protocol validator、`py_compile`、
  `git diff --check`。
- TEST=0；GPU Mini/Full=`NOT RUN`。

## R03 证据处置

R03 原始归档及解压副本不可修改。Reviewer 可从其 `DONE.json`、candidate
aggregate 与 9 个 per-seed budget record 独立重算实际资源摘要，并以
单独 stage report 记录；不得把补充记录伪装成原始 runner 产物。

## Reviewer checklist 与下一 Gate

- [x] run-level elapsed 使用实测 wall clock。
- [x] run-level CUDA peak 来自全部 candidate/seed 的实际最大值。
- [x] malformed/missing peak fail closed。
- [x] 既有 identity、terminal、TEST=0 回归通过。
- [x] R03 原始证据未被覆盖或提交。

下一 Gate：代码修复已经 Codex 独立复核接受；由 Owner 决定
`ACCEPT_WITH_LIMITATIONS / ITERATE / NEW_EXP_REQUIRED`。本任务不授权重跑。
