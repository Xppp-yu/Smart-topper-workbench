# TASK-SLP-B08-FULL-RUNNER-AND-ONE-FOLD-PREFLIGHT-v0.1

状态：`READY_FOR_CODE_REVIEW / GPU_PREFLIGHT_NOT_AUTHORIZED / FULL_NOT_RUN / TEST_NOT_ACCESSED / NOT_COMMITTED / NOT_PUSHED`

## 目标

实现 B07 的独立 SLP8 Full Runner：冻结配置加载、30-unit execution plan、
subject-fold dataset routing、fold-transactional checkpoint/resume、严格 OOF
合并、identity/resource/terminal carriers，以及 synthetic CPU smoke。代码
验收后另行冻结 SHA，再由 Owner 决定是否运行 RTX 4090 单 fold 预检。

## 允许修改

- `src/topper_perception/neural/slp8_region_full.py`
- `scripts/run_slp8_region_full.py`
- `scripts/smoke_b08_full_runner.py`
- `tests/test_slp8_region_full.py`
- 本任务单、对应 stage report、项目状态和 Backlog 的窄范围更新

## 必须行为

- 精确加载 B07 config/fold manifest 并验证 byte SHA。
- 计划必须是 2 candidates × 5 folds × 3 seeds = 30 unique units。
- 每 unit 只接收 fold-TRAIN / fold-VAL；subject overlap=0。
- TEST rows/labels/onehot/statistics/predictions/metrics 始终为 0。
- 每个 unit 独立目录和 `complete.json`；完成单元不可覆盖。
- resume 必须验证完整 identity 和 unit/candidate/total budget accumulators。
- 每 seed OOF 必须覆盖 91 subjects / 4,095 samples，0 duplicate/missing。
- exactly one terminal JSON；失败 unit 不得静默丢弃。
- no-write/validate-only 在训练前完成，不创建实验目录。

## 非目标

- 不修改 B07 协议、fold、候选或训练超参数。
- 不访问 B01 TEST 数据；不得调用 `enable_test_access`。
- 不运行真实 GPU preflight、30-unit Full 或最终 TEST。
- 不根据 synthetic 指标排名候选。

## 验证与 Gate

需要 unit/failure-path/synthetic smoke、B04/B04A 回归、validator、compile、
links 和 diff check。代码接受后状态最多为
`RUNNER_ACCEPTED / READY_FOR_OWNER_PREFLIGHT_AUTHORIZATION`；真实 one-fold 仍
需新 clean Git SHA、独立 EXP-ID、机器/预算记录和 Owner 明确授权。
