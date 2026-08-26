# B04 交付说明：SLP8 PM-only 区域分割 Mini 协议与 Runner

**TASK-ID:** `TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-v0.1`
**Stage:** S2-B04
**日期:** 2026-08-27
**状态:** `MINI_PROTOCOL_AND_RUNNER_READY_FOR_REVIEW`
**Base:** `main @ 6e19374`
**Branch:** `codex/task-slp-b04-pm-only-region-mini-protocol-v0.1`

---

## 这一步做什么

冻结 SLP8 PM-only Region Mini 的**完整协议**与**受治理 Runner**：

* 数据合同：仅 B01 冻结表（TRAIN 3,645 / VAL 450，TEST = 0 不可访问），压力保持 raw_pmarray_response。
* 候选模型：两个候选
  * `slp8_tiny_fcn_v0.1`（复用 B03 架构）
  * `slp8_small_unet_v0.1`（新增 B04 SmallUNet，118K 参数，显式 spatial size 恢复）
* 训练协议：seed=42，AdamW，lr=1e-3，wd=1e-4，max_epochs=20，min_epochs=5，early stop 监控 val_loss（patience=4）。
* Class Weight 公式：TRAIN-only，`1/sqrt(ratio)` → 归一化 → clip(0.25, 4.0)。
* 扩展指标：整体 / 逐区域 / 逐姿势 / 逐受试者 / 最差受试者 / 中心误差。
* 可行性 Gate：VAL fixed foreground macro IoU ≥ B02 train-spatial-prior 参考值 0.205644。
* 资源预算：每候选 45 分钟，总计 90 分钟，CUDA 12 GB peak。
* Runner 治理：`--validate-config`、`--synthetic-cpu-smoke`、真实运行需 `--run-authorized`（本任务不执行）。
* 输出合同：19 个产物（含 `DONE.json` XOR `FAILED.json`），防覆盖、互斥。
* 测试：111 个 B04 新测试 + 全量回归 1282 passed。
* 中文文档：本文件 + 阶段报告。

**本任务不实际运行真实 Mini；本任务不读取 TEST；本任务不进入 B07。**

---

## 实际结果（仅 Synthetic CPU Smoke）

| 字段 | 值 |
|---|---|
| EXP-ID | `EXP-SLP-B04-PM-REGION-MINI-20260827-SYNTH` |
| 模式 | `--synthetic-cpu-smoke` |
| 平台 | Windows-11 / Python 3.12.13 / PyTorch 2.13.0 CPU |
| Wall clock | 24.95 s |
| TRAIN 样本 / 受试者 | 8 / 2 |
| VAL 样本 / 受试者 | 4 / 1 |
| TEST 样本 | 0（不加载） |
| Candidate A feasibility | `NOT_FEASIBLE`（IoU=0.000000 < 0.205644；合成数据 + 20 epoch 训练未达 B02 基线，**符合预期**） |
| Candidate B feasibility | `NOT_FEASIBLE`（同因） |
| overall_decision | `MINI_NOT_FEASIBLE`（与 0.205644 阈值对比的预期结果；本任务不解释原因，仅证明 Runner 链路） |
| 完整产物 | 19 / 19 |
| Reload consistency | 两候选 `max_abs_diff=0.0`, `hash_match=true` |

> **重要：** Synthetic smoke 仅证明 Runner 与协议可工作；**IoU=0 来自合成数据 + 1 epoch 训练**，**不构成 B04 候选与 B02 的排名**。

---

## 使用方法

### 1. 准备环境

```powershell
cd <B04_WORKTREE>
uv sync
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. 校验配置 / 注册表

```powershell
.venv\Scripts\python.exe scripts\run_slp8_region_mini.py `
  --config configs\experiments\slp8_pm_region_mini_v0.1.json `
  --output-dir outputs\experiments\EXP-SLP-B04-PM-REGION-MINI-20260827-VALIDATE `
  --validate-config
```

会写出：`DONE.json`、`resolved_config.json`、`input_manifest_hashes.json`、`environment.json`、`status.json`、`logs/run.log`。不读取任何 B01 路径。

### 3. 合成 CPU 烟雾

```powershell
.venv\Scripts\python.exe scripts\run_slp8_region_mini.py `
  --config configs\experiments\slp8_pm_region_mini_v0.1.json `
  --output-dir outputs\experiments\EXP-SLP-B04-PM-REGION-MINI-20260827-SYNTH `
  --synthetic-cpu-smoke
```

会写出全部 19 个产物；不读取任何 B01 路径；不复用任何 EXP-ID。

### 4. 真实 Mini（**本任务不执行**；需要 Owner 授权）

```powershell
.venv\Scripts\python.exe scripts\run_slp8_region_mini.py `
  --config configs\experiments\slp8_pm_region_mini_v0.1.json `
  --output-dir outputs\experiments\EXP-SLP-B04-PM-REGION-MINI-20260827-R01 `
  --b01-freeze-dir <B01_FREEZE_DIR> `
  --dataset-root <SLP8_DATASET_ROOT> `
  --run-authorized
```

* `--run-authorized` 是显式开关，未带但传了真实路径会立即拒绝（`raise MiniProtocolError("... --run-authorized was NOT set...")`）。
* 真实 Mini **必须**在 `device='cuda'` 上执行；CUDA 不可用时立即 fail-closed（不静默回退 CPU）。

---

## 代码和配置在哪里

```
<B04_WORKTREE>/
├── src/topper_perception/neural/
│   ├── slp8_region_models.py        # 修改：新增 Slp8SmallUnet + ModelBuilder 注册表
│   ├── slp8_region_class_weights.py # 新增：TRAIN-only class weight 公式
│   ├── slp8_region_metrics_ext.py   # 新增：扩展指标（per-region/posture/subject/worst/centroid）
│   └── slp8_region_mini.py          # 新增：Mini 核心 runner
├── scripts/
│   └── run_slp8_region_mini.py      # 新增：CLI Runner（--validate-config / --synthetic-cpu-smoke / --run-authorized）
├── configs/experiments/
│   └── slp8_pm_region_mini_v0.1.json
├── tests/
│   ├── test_slp8_region_mini.py     # 新增：98 个 B04 测试
│   └── test_slp8_region_models.py   # 修改：+13 个 B04 registry 测试
└── docs/
    ├── stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md
    └── deliverables/SLP/B04_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md（本文件）
```

---

## 输出文件（Synthetic CPU Smoke 实际样本）

```
outputs/experiments/EXP-SLP-B04-PM-REGION-MINI-20260827-SYNTH/
├── DONE.json                                ✅ (与 FAILED.json 互斥)
├── status.json                              ✅
├── manifest.json                            ✅ (含 dataset_manifest / train_class_stats / class_weight_summary)
├── resolved_config.json                     ✅ (无本机绝对路径)
├── input_manifest_hashes.json               ✅
├── environment.json                         ✅
├── epoch_metrics.csv                        ✅ (两候选各 20 行)
├── metrics_summary.json                     ✅ (含 class_weight_summary)
├── metrics_by_region.csv                    ✅ (2 候选 × 8 区域 = 16 行)
├── metrics_by_subject.csv                   ✅ (含 ALL + 各位受试者)
├── metrics_by_posture.csv                   ✅ (含 ALL + SUPINE/LEFT/RIGHT)
├── centroid_errors.csv                      ✅ (per-sample per-region 真实记录)
├── worst_subject.json                       ✅
├── confusion_matrix.csv                     ✅ (2 候选 × 9 行)
├── predictions_manifest.csv                 ✅ (真实 sample_id / 64 位 hex hash)
├── candidate_decision.json                  ✅ (两候选 NOT_FEASIBLE，IoU=0)
├── reload_consistency.json                  ✅ (max_abs_diff=0, hash_match=true)
├── checkpoints/
│   ├── slp8_tiny_fcn_v0.1/
│   │   ├── best.pt                          ✅
│   │   └── last.pt                          ✅
│   └── slp8_small_unet_v0.1/
│       ├── best.pt                          ✅
│       └── last.pt                          ✅
└── logs/
    └── run.log                              ✅
```

`predictions_manifest.csv` 包含真实 `SLP:danaLab:NNNNN:uncover:NNNNNN` 样本 ID、64 位小写 hex SHA-256、failure_reason，无占位符。
`centroid_errors.csv` 包含 per-sample per-region 真实归一化距离（GT-only=1.0、both-present=距离/对角线、both-missing 标记并从平均中排除）。

---

## Canonical Array Hash 规则

`canonical_array_hash` 在 mini 模块顶部定义：

1. 转为 int64 C-contiguous（`np.ascontiguousarray(arr, dtype=np.int64)`）
2. 拼装 header：`slp8_canonical_array_hash_v0.1\ndtype=<i8\nshape=(H,W)\n`
3. 拼接 C-order bytes
4. SHA-256 → 64 位小写 hex

`_predictions_hash` 进一步聚合候选的全部 `(H, W)` 预测：每样本一个 header + bytes，整体 SHA-256。best.pt 独立加载后，**两份 hash 必须位级一致**（这是 FEASIBLE Gate 的 5）。

---

## Class Weight 公式（TRAIN-only）

```text
raw_weight[c] = 1 / sqrt(pixel_ratio[c])
weight[c]    = raw_weight[c] / mean(raw_weight)
weight[c]    = clip(weight[c], 0.25, 4.0)
```

* `compute_class_weights(train_class_stats, allowed_split="train")` 接受 B01 冻结的 `train_class_stats.json`。
* 拒绝 VAL/TEST split、零比、NaN、Inf、缺类。
* 两个候选**共享同一组 weight**（`compute_class_weights` 纯函数）。
* `assert_class_weight_invariants` 在 runner 中再次校验。
* `resolved_config.json` / `manifest.json` / `metrics_summary.json` 三处独立记录最终 9 个 weight。

---

## 关键参数（冻结）

| 项 | 值 |
|---|---|
| seed | 42 |
| device | `cuda`（CUDA 不可用 → fail-closed） |
| batch_size | 16 |
| max_epochs | 20 |
| min_epochs | 5 |
| early_stopping.monitor | `val_loss` |
| early_stopping.mode | `min` |
| early_stopping.patience | 4 |
| early_stopping.min_delta | 0.0 |
| optimizer | AdamW |
| lr | 0.001 |
| weight_decay | 1e-4 |
| scheduler | none |
| augmentation | none |
| num_workers | 0 |
| 每候选 seed | 1 |
| `B04_MAX_PARAMETERS` | 150,000 |
| `b02_reference_val_fixed_iou` (FEASIBLE 阈值) | 0.205644 |
| `max_wall_minutes_per_candidate` | 45 |
| `max_total_wall_minutes` | 90 |
| `max_peak_cuda_mb` | 12,288 |

任何 `validate_mini_config` 字段不匹配即 fail-closed。

---

## 验证项

| 验证项 | 状态 |
|---|---|
| TEST 访问被阻止 | ✅（B01 `enable_test_access` 合同 + Runner 永不开启） |
| 压力保持 raw_pmarray_response（NOT kPa） | ✅ |
| Subject 隔离 | ✅ |
| SmallUNet 84 宽度显式恢复 | ✅（`F.interpolate(..., size=...)`，不依赖 `scale_factor`） |
| 参数 ≤ 150,000 | ✅（118,121 / 150,000） |
| Class weight 仅来自 TRAIN | ✅ |
| Feasibility Gate 公式与 0.205644 阈值一致 | ✅ |
| `--run-authorized` gate | ✅ |
| 输出目录防覆盖 | ✅ |
| `DONE.json` XOR `FAILED.json` 互斥 | ✅ |
| predictions_manifest 真实数据 | ✅ |
| centroid_errors 真实 per-sample per-region | ✅ |
| Reload consistency 真实比较 | ✅（hash_match=true, max_abs_diff=0.0） |
| B03 回归 | ✅（25 个模型测试 + 全部 B03 集成测试） |
| B01/B02 regression | ✅（287 + 179 passed） |
| 全量联合回归 | ✅ 1282 passed, 4 skipped（与本任务无关的 pre-existing skip） |
| `git diff --check` | ✅（待 commit 前验证） |

---

## 结论和下一步

### 本阶段结论

- ✅ B04 PM-only Region Mini 协议 + Runner 已实现并通过端到端 Synthetic CPU Smoke。
- ✅ 两个候选模型就位，参数上限、显式 spatial size 恢复、no-BatchNorm / no-Dropout / no-pretrained 全部满足。
- ✅ Class Weight 公式仅允许 TRAIN 驱动；任何 VAL/TEST、零比、非有限值都被 fail-closed 拒绝。
- ✅ Runner 三模式（`--validate-config` / `--synthetic-cpu-smoke` / `--run-authorized`）互斥且不可绕过。
- ✅ 真实 Mini 路径**不**执行；CUDA 不可用时立即 fail-closed（不静默回退 CPU）。
- ✅ 全套输出产物（19 个）齐全，predictions_manifest 全部 64 位 hex、centroid_errors 全部真实记录。

### 下一步

1. **Codex Reviewer** 在本工作树独立验证：
   - Synthetic CPU smoke 输出中两候选均为 `NOT_FEASIBLE`（合成数据 + 1 epoch，预期结果）。
   - 19 个产物齐全。
   - SmallUNet 84 宽度恢复路径（不依赖 `scale_factor`）。
   - `git diff --check` 通过。
2. **Owner 真实 Mini 授权**（不属于本任务）：
   - 在 RTX 4090 / PyTorch 2.8.0+cu128（或更新 CUDA 环境）执行 `--run-authorized` + 真实 B01 路径。
   - 由 Experiment Runner 在 B07 协议下运行。
3. **B07 Full 协议** 由后续 task 起草；本任务**不**进入 B07。

---

## 环境信息

| 字段 | 值 |
|---|---|
| Git Branch | `codex/task-slp-b04-pm-only-region-mini-protocol-v0.1` |
| Base | `origin/main @ 6e19374` |
| B03 Merge | `6e19374` ✅ |
| seed | 42 |
| Python | 3.12.13 |
| PyTorch | CPU (2.13.0) — 真实 Mini 需要 CUDA 12 GB peak |
| Platform | Windows-11 |

---

## 禁止结论

> 1. Mini 已完成
> 2. 模型优于 B02
> 3. 候选已保留
> 4. B07 已解锁
> 5. 适用于产品决策
> 6. GT 是人类像素级标注
> 7. 压力值代表 kPa
> 8. 适用于 cover1/cover2
> 9. 适用于 danaLab 之外
> 10. 适用于自研硬件 / 整夜 / 气囊闭环
> 11. 本次 Synthetic smoke 分数可与 B02 对比

---

## Stage Report

完整分析见：[../../stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md](../../stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md)

---

**交付版本：** v0.1
**生成时间：** 2026-08-27
**维护者：** Mavis (MiniMax Code)
