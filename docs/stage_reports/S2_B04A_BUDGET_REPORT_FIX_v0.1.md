# S2 B04A Budget Report Fix v0.1

TASK-ID：`TASK-SLP-B04A-BUDGET-REPORT-FIX-v0.1`

状态：`CODE_FIX_ACCEPTED / R03_ACCEPT_WITH_LIMITATIONS_RECOMMENDED / OWNER_GATE_PENDING`

## 结论

B04A run-level bundle writer 的资源汇总缺陷已定位并修复。旧实现明确把
`elapsed_total_seconds` 写成 `max_wall_seconds_total` 占位值，并把
`peak_cuda_mb` 固定写成 `0.0`。修复后：

- 总耗时来自 `B04ARunResult.wall_clock_seconds`；
- CUDA 峰值来自所有 candidate/seed 的 per-seed budget report 最大值；
- wall clock 或 peak 缺失、非数值、负数、NaN/Inf 时 fail closed；
- identity、候选决策、terminal state 与 B04 历史路径不变。

## 文件

- `src/topper_perception/neural/slp8_region_mini.py`
- `tests/test_b04a_runner_integration.py`
- `docs/tasks/TASK_SLP_B04A_BUDGET_REPORT_FIX_v0.1.md`
- 本报告

## R03 原始证据独立重算

只读来源：
`outputs/reviews/b04a_r03_f0fac82/EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-AUTODL-R03/`

- terminal：`DONE`；launcher exit code（外部下载记录）=`0`
- 训练单元：9/9；checkpoint：18/18
- predictions split：每 seed TRAIN=3,645 / VAL=450 / TEST=0
- reload：9/9 `reload_consistent=true`、`hash_match=true`、
  `max_abs_diff=0.0`
- `DONE.json.wall_clock_seconds=2128.7466315738857`
- candidate elapsed 合计=`2126.87068743631` 秒
- 9 个 per-seed budget record 的最大 CUDA peak=`363.412109375` MiB
- 冻结阈值：总 wall 8,100 秒、每 candidate 2,700 秒、peak 8,192 MiB
- 三候选均 FEASIBLE；晋级记录为
  `slp8_deeplabv3plus_lite_v0.1` 与 `slp8_resunet_lite_v0.1`

原始 `budget_report.json` 的 `8100.0 / 0.0` 仍保留，不覆盖、不伪装成
runner 原产物。本报告只记录 Reviewer 对不可变证据的独立重算。

## 验证

- 新增资源汇总与 malformed peak 定向回归：7 passed。
- 修改前既有 B04A integration：129 passed。
- 修改后完整 B04A integration：136 passed。
- B04 Mini 完整回归：167 passed。
- protocol/model/Markdown 联合回归：94 passed。
- B04A implementation：78 passed / 1 host-assumption failure；失败测试把
  当前主机硬编码为 CUDA unavailable，但实测 smoke 正常返回
  `cuda_run=True`，不是本任务代码断言失败。
- validator：30 OKs / 0 errors；`py_compile`、`git diff --check` 通过。
- GPU Mini/Full：`NOT RUN`。
- TEST：0。

## 已验证、推断、未验证与限制

已验证：缺陷来自 writer 的明确 placeholder；R03 训练、身份链、split、
reload、候选聚合和 per-seed 资源记录可从本地证据读取。

推断：原始顶层资源汇总错误不改变模型权重或指标，但使原始 run-level
资源载体不满足严格审计合同，因此不能无说明地把原始包标记为完全合格。

未验证：尚未在修复 SHA 上重新运行真实 GPU Mini；CUDA 独立重复运行的
byte-identical 确定性仍未验证。

限制：SLP8 GT 仍为 `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`、
`source_review_status=NOT_REVIEWED`、uncover-only；不是人工像素级、医学、
产品、舒适性或气囊控制 GT。

## 下一 Gate

本地回归和 Codex 独立复核已完成，建议 R03 结论为
`ACCEPT_WITH_LIMITATIONS`：接受训练与候选晋级证据，同时永久保留原始
run-level budget carrier 缺陷和本 Reviewer 补充记录。是否要求新 EXP
重跑由 Owner 决定；本任务不授权 GPU、B07 或 TEST。
