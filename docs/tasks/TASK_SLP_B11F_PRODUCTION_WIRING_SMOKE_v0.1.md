# TASK-SLP-B11F-PRODUCTION-WIRING-SMOKE-v0.1

## 1. 目的

在不连接 AutoDL、不使用 GPU、不读取 TEST、不创建正式 EXP-ID 或 checkpoint 的前提下，
以本地 CPU 和真实 B01 development 数据验证 B11F 修复后的生产训练接线：完整 development
metadata/TEST 门禁、确定性 128 样本子集上的 normalization/class weights、NumPy 到 Torch
权重转换、单样本真实 microbatch 的 forward/loss/backward，以及恰好一次 AdamW step。
冻结的正式 batch size=16 仍须由 config 核验并记录，但不在本地 CPU 上模拟其显存/吞吐表现。

本任务是运行前 smoke，不是 final fit，不产生研究指标，也不授权恢复失败的 R01。

## 2. 边界

- TASK-ID：`TASK-SLP-B11F-PRODUCTION-WIRING-SMOKE-v0.1`
- 基线：`86415948afed4aa30e9cced2bfabbacba03aed5e`
- 分支：`codex/task-slp-b11f-production-wiring-smoke-v0.1`
- 允许修改：
  - `.gitattributes`（仅保护本任务 evidence 的 byte identity）
  - `scripts/smoke_slp8_b11f_production_wiring.py`
  - `tests/test_slp8_b11f_production_wiring.py`
  - 本任务文档及对应 stage report
  - `docs/evidence/b11f_production_wiring_smoke/**`
  - `docs/PROJECT_STATUS.md`
  - `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`
- 原始 SLP8 数据只读。
- `load_test=False`；`_test_rows is None`；TEST carriers 固定为 0。
- 强制 `CUDA_VISIBLE_DEVICES="-1"` 和 `device="cpu"`；GPU/AutoDL 禁止。
- 不允许正式 `EXP-SLP-B11F-*` 标识、正式 output root、checkpoint、resume 或 epoch loop。
- 只允许一个单样本真实 microbatch 和一次 optimizer step；正式 batch size 仍固定为 16。
- Owner 已于 2026-09-04 在当前 Codex 任务中明确授权 R07 closure commit/push；该授权不包含
  AutoDL 连接、no-training preflight、正式 EXP-ID、GPU 或 TEST。

## 3. 必须复用的生产路径

1. `load_protocol()` 核验冻结 B11F config/candidate 合同。
2. `load_development_samples()` 只加载 B01 TRAIN+VAL，共 4,095 samples / 91 subjects。
3. `compute_fold_normalization_from_samples()` 和
   `compute_fold_class_weights_from_samples()` 使用确定性的前 128 个真实 development 样本，
   只验证函数与数据接线，不把结果当作正式 4,095 样本统计量。
4. `class_weights_to_tensor()` 必须返回真实 `numpy.ndarray`，再严格执行
   `torch.from_numpy(...).to("cpu").to(torch.float32)`。
5. 使用真实 `Slp8RegionDataset`、`build_dataloader()`、`build_model()` 和
   `deterministic_cross_entropy_2d()`。
6. optimizer 固定 `AdamW(lr=0.001, weight_decay=0.0001)`；seed=42；batch size=16。

## 4. 通过条件

- 输入 microbatch 为 `[1, 1, 192, 84]` / `float32`，label 为
  `[1, 192, 84]` / `int64`，summary 同时记录冻结的正式 batch size=16。
- 九类权重按 class ID 0..8，有限；NumPy dtype 为 `float64`，Torch dtype 为
  `float32`，device 为 CPU。
- loss 有限；梯度存在且有限；一次 step 后至少一个可训练参数发生变化。
- summary 只记录 smoke 诊断和 hash/carriers，不包含绝对数据路径或 validation 指标。
- 完整 4,095 样本 statistics 的数值、耗时和资源占用仍标为 NOT RUN。
- 输出不得覆盖既有文件。
- TEST=0、GPU NOT RUN、AutoDL NOT CONNECTED。

## 5. 预定验证

```bash
uv run python -m pytest tests/test_slp8_b11f_production_wiring.py -q
uv run python -m pytest tests/test_slp8_b11f_production_wiring.py tests/test_slp8_region_final_fit.py tests/test_slp8_b11_candidate_freeze.py tests/test_slp8_region_full.py -q
uv run python scripts/smoke_slp8_b11f_production_wiring.py --b01-freeze-dir data/processed/slp8_training_tables_v0.1 --dataset-root E:/TeamProjects/datasets/smart-topper/SLP2022/SLP/SLP_8Region_Pressure_VAL_v1.1 --output-json outputs/analysis/b11f_production_wiring_smoke_20260904_r07.json
uv run python -m pytest tests/test_check_markdown_links.py -q
uv run python -m py_compile scripts/smoke_slp8_b11f_production_wiring.py
git diff --check
git status --short --branch
```

## 6. 当前 Gate

R06 首轮复审发现 output publish 的 TOCTOU 覆盖竞争，结论 `ITERATE`。R07 已改用
原子 hard-link no-clobber 发布并通过竞争反例及同路径重跑拒绝。R07 脱敏 summary 已纳入
`docs/evidence/b11f_production_wiring_smoke/`。R01--R06 均不是正式 EXP，未产生 checkpoint。

同一 Codex 执行者按独立审查清单完成 R07 技术复审：`ACCEPT`，P0/P1/P2 均为 0。
人员独立性限制保留；Owner 已单独授权 closure commit/push。

当前 Gate：

`B11F_AUTODL_NO_TRAINING_PREFLIGHT_PREPARATION / AUTODL_NOT_AUTHORIZED / GPU_NOT_AUTHORIZED / TEST_DENIED`
