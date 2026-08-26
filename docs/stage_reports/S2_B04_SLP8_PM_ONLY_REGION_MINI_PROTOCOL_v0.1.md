# Stage Report: S2-B04 — SLP8 PM-only Region Mini 协议与 Runner

**TASK-ID:** `TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-v0.1`
**Stage:** S2-B04
**日期:** 2026-08-27
**状态:** `MINI_PROTOCOL_AND_RUNNER_READY_FOR_REVIEW` — 协议、Runner、模型、指标、测试、中文文档已就绪；**未执行真实 Mini**，**未读取 TEST**，**未进入 B07**
**Base:** `main @ 6e19374`
**Branch:** `codex/task-slp-b04-pm-only-region-mini-protocol-v0.1`

---

## 1. 摘要

B04 阶段冻结 **SLP8 PM-only Region Mini 协议**、**Runner 治理**、**两个候选模型**、**TRAIN-only Class Weight 公式**、**扩展指标**、**可行性 Gate** 和 **资源预算**。本任务**不**执行真实 Mini（无真实 B01 数据、无 GPU 运行授权），但交付了完整可运行的代码与配置，并通过：

1. `--validate-config` 模式：纯配置/注册表校验，无任何 B01 路径接触。
2. `--synthetic-cpu-smoke` 模式：合成数据下完成两个候选的全量 Mini 链路（训练 / checkpoint / resume / reload / 指标 / 预测 / 候选决策），全套输出产物落盘。

**关键禁止结论（**已逐项落实于代码与文档**）：**

* ❌ 不宣称 Mini 已完成
* ❌ 不宣称 B04 任何候选"优于 B02"
* ❌ 不宣称候选已保留 / 排除
* ❌ 不宣告 B07 解锁

**真实 Mini 必须先取得 Owner 运行授权 + `--run-authorized` + 真实 B01 路径，并由 Experiment Runner 在 B07 协议下执行。**

---

## 2. 数据合同（与 B01 冻结表严格对齐）

* **TRAIN:** 3,645 samples / 81 subjects
* **VAL:** 450 samples / 10 subjects
* **TEST:** 必须保持不可访问（Runner 默认 `load_test=False`；任何 attempt 读 TEST rows 立即 `TestLeakageError`）。
* **Pressure:** `raw_passthrough_with_minmax_reference` 方法，float64 → float32，添加 channel，**不做 Min-Max**，`raw_semantics = raw_pmarray_response`（**NOT kPa**）。
* **Provenance:** `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`；`source_review_status = NOT_REVIEWED`。
* **输入校验：** Runner 启动时读取并校验：
  * `freeze_manifest.json`（顶层冻结 SHA、A06 split SHA `024f5abe...`）
  * `train_manifest` / `val_manifest`（3645/450/0）
  * `normalization_stats.json`（TRAIN-only fit）
  * `train_class_stats.json`（TRAIN-only per-class pixel ratio）
* **输入 SHA-256 落盘：** 全部输入文件 SHA-256 写入 `input_manifest_hashes.json`。
* **TEST 防泄漏合同：** 复用了 B01 `enable_test_access(purpose="final_evaluation")` 合同；B04 永不开启。

---

## 3. 两个候选模型

### 3.1 Candidate A — `slp8_tiny_fcn_v0.1`

* 复用 B03 `Slp8TinyFcn` 架构（最小全卷积网络，无池化、无下采样）。
* Input `[N, 1, 192, 84]` → logits `[N, 9, 192, 84]`。
* 训练参数：**1,401**。
* 行为控制：作为架构对照组。

### 3.2 Candidate B — `slp8_small_unet_v0.1`（新增）

* **结构（已冻结）：**
  * Input `[N, 1, 192, 84]`
  * Encoder block 1: `Conv 1→16` + ReLU, `Conv 16→16` + ReLU
  * `MaxPool2d(2)` → `[N, 16, 96, 42]`
  * Encoder block 2: `Conv 16→32` + ReLU, `Conv 32→32` + ReLU
  * `MaxPool2d(2)` → `[N, 32, 48, 21]`
  * Bottleneck: `Conv 32→64` + ReLU, `Conv 64→64` + ReLU
  * Decoder: **`F.interpolate(..., size=(96, 42), mode="bilinear", align_corners=False)`** —— **显式 spatial size 恢复**
  * concat skip2 → `Conv 96→32→32`
  * **`F.interpolate(..., size=(192, 84), ...)`** → concat skip1 → `Conv 48→16→16`
  * Final `Conv 16→9, kernel=1`
  * logits `[N, 9, 192, 84]`
* **不使用** pretrained weights、BatchNorm、Dropout、外部下载。
* **训练参数：118,121**（远低于 150,000 上限）。
* **宽度 84 经过两次下采样（84 → 42 → 21）的恢复** 严格依赖 `F.interpolate(..., size=...)` 的显式 size 参数，**禁止使用** `scale_factor` 模糊恢复。

### 3.3 模型注册表

`src/topper_perception/neural/slp8_region_models.py` 提供：

* `ModelBuilder` dataclass
* `register_model_builder(builder)`
* `get_model_builder(name)` / `list_model_builders()`
* 启动时自动注册 `slp8_tiny_fcn_v0.1` 与 `slp8_small_unet_v0.1`
* `B04_MAX_PARAMETERS = 150_000` 在 `slp8_region_models.py` 与 `slp8_region_mini.py` 双重固定；任何超过 150K 参数的候选自动被 `validate_mini_config` 与 `run_one_candidate` 双重 fail-closed 拒绝。

---

## 4. 训练协议（全部冻结）

| 项 | 值 |
|---|---|
| seed | 42 |
| device | `cuda`（CUDA 不可用时 fail-closed；仅 `--synthetic-cpu-smoke` 允许 CPU） |
| batch_size | 16 |
| max_epochs | 20 |
| min_epochs | 5 |
| optimizer | AdamW |
| lr | 0.001 |
| weight_decay | 1e-4 |
| scheduler | none |
| augmentation | none |
| num_workers | 0 |
| early stopping | `monitor=val_loss, mode=min, patience=4, min_delta=0.0` |
| 每候选 seed 数 | 1 |
| 每轮保存 | `checkpoints/<candidate>/last.pt` |
| best 选择 | **最低 val_loss**（mode='min'，earliest-epoch tie-break） |
| 重新调参 | ❌ 禁止按 VAL IoU 反复调参 |

`early_stopping.patience` 从 4 起算；`min_epochs=5` 之前不会触发 early stop。

---

## 5. Class Weight 公式（TRAIN-only）

`src/topper_perception/neural/slp8_region_class_weights.py` 实现了严格的不可绕开公式：

```text
raw_weight[c] = 1 / sqrt(pixel_ratio[c])
weight[c]    = raw_weight[c] / mean(raw_weight)
weight[c]    = clip(weight[c], 0.25, 4.0)
```

**禁止条款**：

* `compute_class_weights` 默认参数 `allowed_split="train"`；传入 `"val"` 或 `"test"` 立即 `ClassWeightError`。
* `pixel_ratio[c] <= 0` 立即拒绝（`1/sqrt(0)` 未定义）。
* `pixel_ratio[c]` 非有限（NaN/Inf）立即拒绝。
* normalized pre-clip 或 final `weight` 非有限立即拒绝。
* 两个候选**共享同一组 weight**（`compute_class_weights` 是纯函数，无随机性）。
* `assert_class_weight_invariants` 在写入 manifest 前再次校验。
* `resolved_config.json` 与 `manifest.json` 都会记录最终 9 个 weight。

---

## 6. 指标

| 类型 | 字段 |
|---|---|
| **整体** | `fixed_foreground_macro_iou`（固定 classes 1–8，不跳过空类）、`fixed_foreground_macro_dice`、`pixel_accuracy`、`background_iou`、`confusion_matrix`（9×9）、`val_loss` |
| **逐区域** | IoU / Dice / precision / recall / TP / FP / FN（共 8 个前景类） |
| **逐姿势** | `SUPINE` / `LEFT` / `RIGHT` / `ALL`，每个的 macro IoU / Dice / pixel accuracy |
| **逐受试者** | subject_id → macro IoU / Dice / pixel accuracy |
| **worst subject** | argmin(fixed_foreground_macro_iou) over subjects with n_samples > 0 |
| **中心误差** | 每样本、每区域 GT/pred centroid；Euclidean 距离 ÷ 图像对角线 `√(192²+84²)`；GT 存在但 pred 缺失 → 1.0；两者都缺失 → 跳过该区域平均；GT/pred 都存在 → 距离/对角线 |

**禁止条款**：

* 中心误差 **不** 替代 IoU；两者都报告。
* 任何"空类跳过"行为都被显式禁止（B02 固定类宏指标 + B04 v0.1 fixed foreground macro 都不跳过）。
* `metrics_version = slp8_region_metrics_ext_v0.1`。

---

## 7. 可行性 Gate

**B02 同一 VAL 上的 best fixed foreground macro IoU 参考值 = `0.205644`**（train spatial prior 基线）。

候选标记 `FEASIBLE` **必须同时满足** 8 条：

1. 运行状态成功（`success=True`）
2. `n_test_samples == 0`
3. 所有 loss / metrics `finite`
4. checkpoint save / resume / reload 通过（独立 `fresh_model.load_state_dict(best)` 后 logits `allclose`）
5. best checkpoint 独立加载后的预测 hash 与 in-process 预测 hash 一致
6. 所有逐区域 / 姿势 / 受试者 / 最差受试者指标完整
7. **VAL fixed foreground macro IoU >= 0.205644**
8. 未超过冻结资源预算

**决策矩阵：**

| 通过 FEASIBLE 数 | 处理 |
|---|---|
| 0 | `MINI_NOT_FEASIBLE` —— 停止，不进入 B07 |
| 1 | 保留该候选 |
| 2 | 两个候选都保留；B04 **不** 宣布最终冠军 |

`candidate_decision.json` 记录每位候选的 `feasibility`、`reason`（含 iou 数值与阈值）、`best_epoch` 等。

---

## 8. 资源预算（冻结但本任务不实际运行）

| 项 | 值 |
|---|---|
| `max_wall_minutes_per_candidate` | 45 |
| `max_total_wall_minutes` | 90 |
| `max_peak_cuda_mb` | 12288 |
| 并发 | 同一时刻仅 1 个候选 |
| 超预算处理 | 立即停止并写 `FAILED.json` / `STOPPED`；**禁止静默继续** |

当前实验环境**无 CUDA**（`torch.cuda.is_available() == False`），真实 Mini 启动会立即 fail-closed（`device='cuda' requested but torch.cuda.is_available() is False`）。

---

## 9. Runner 治理

### 9.1 CLI（`scripts/run_slp8_region_mini.py`）

三种运行模式：

```powershell
# 默认 / --validate-config
.venv\Scripts\python.exe scripts\run_slp8_region_mini.py `
  --config configs\experiments\slp8_pm_region_mini_v0.1.json `
  --output-dir outputs\experiments\EXP-SLP-B04-PM-REGION-MINI-20260827-VALIDATE `
  --validate-config

# 合成 CPU 烟雾（synthetic CPU smoke）
.venv\Scripts\python.exe scripts\run_slp8_region_mini.py `
  --config configs\experiments\slp8_pm_region_mini_v0.1.json `
  --output-dir outputs\experiments\EXP-SLP-B04-PM-REGION-MINI-20260827-SYNTH `
  --synthetic-cpu-smoke

# 真实 B01（必须显式 --run-authorized；本任务不执行）
.venv\Scripts\python.exe scripts\run_slp8_region_mini.py `
  --config configs\experiments\slp8_pm_region_mini_v0.1.json `
  --output-dir outputs\experiments\EXP-SLP-B04-PM-REGION-MINI-20260827-R01 `
  --b01-freeze-dir <B01_FREEZE_DIR> `
  --dataset-root <SLP8_DATASET_ROOT> `
  --run-authorized
```

**Gate 逻辑：**

* **未传 `--run-authorized` 但传了 `--b01-freeze-dir` / `--dataset-root`：** 立即 `MiniProtocolError` 拒绝进入训练（`raise MiniProtocolError("B01 freeze or dataset-root paths were supplied but --run-authorized was NOT set...")`）。
* **未传任何 mode：** 默认进入 `--validate-config`。
* **CUDA 不可用 + 真实路径 + `--run-authorized`：** 立即 fail-closed（`device='cuda' requested but torch.cuda.is_available() is False`）。

### 9.2 输出合同（每个 `--output-dir` 至少包含）

* `status.json` / `manifest.json` / `resolved_config.json` / `input_manifest_hashes.json` / `environment.json`
* `epoch_metrics.csv` / `metrics_summary.json`
* `metrics_by_region.csv` / `metrics_by_subject.csv` / `metrics_by_posture.csv`
* `centroid_errors.csv` / `worst_subject.json` / `confusion_matrix.csv`
* `predictions_manifest.csv`（逐样本真实 `sample_id` / `subject_id` / `label_sha256` / `prediction_sha256` / `failure_reason`）
* `candidate_decision.json` / `reload_consistency.json`
* `checkpoints/<candidate>/best.pt` / `checkpoints/<candidate>/last.pt`
* `logs/run.log`
* `DONE.json` **或** `FAILED.json`（互斥，写 `DONE` 时会清掉 `FAILED`，反之亦然）

### 9.3 输出目录安全

* 输出目录存在 `DONE.json` / `FAILED.json` / 任何非 `.gitkeep` 文件 → 拒绝覆盖。
* 同一 `EXP-ID` 不得复用。

---

## 10. 测试覆盖

`tests/test_slp8_region_mini.py`（98 测试）+ `tests/test_slp8_region_models.py` 新增 13 测试 = **111 个 B04 新测试**，全部通过。覆盖：

* SmallUNet 输入 / 输出 shape、参数上限（118,121 ≤ 150,000）
* 84 宽度经过两次下采样的显式恢复（`F.interpolate(..., size=...)`）
* 两个候选 registry（`MODEL_REGISTRY`）
* Class Weight 只能来自 TRAIN（拒绝 VAL/TEST、零比、NaN、Inf、缺类）
* 权重公式 / 归一化 / clip / 非有限拒绝
* 配置缺字段 / 非法值 fail-closed（21+ 个具体字段校验）
* CUDA 不可用 fail-closed；synthetic CPU 烟雾唯一 CPU 路径
* early stop 只能监控 `val_loss`
* non-finite loss / metric 拒绝
* checkpoint / resume / reload 独立验证 + prediction hash 一致
* 中心误差（both missing / GT only / both present）规则
* fixed foreground classes 1–8 不跳过空类
* per-subject / per-posture / worst subject
* output collision 检测
* `DONE.json` XOR `FAILED.json` 互斥
* predictions_manifest 真实 sample_id + 64 位 hex hash
* `--run-authorized` gate
* synthetic CPU runner smoke（端到端跑 `--synthetic-cpu-smoke`，校验全部 19 个产物存在 + 内容正确）

**联合回归（不带真实数据测试）：1282 passed, 4 skipped**（跳过的均为缺真实 B01 / 真实 SLP / 真实压力数据的预存跳过）。`test_slp_8region_pressure_dataset.py` 仍然因缺真实数据无法 collect（与本任务无关）。

---

## 11. Synthetic CPU Smoke 实际结果（本任务唯一可执行子集）

| 字段 | 值 |
|---|---|
| EXP-ID | `EXP-SLP-B04-PM-REGION-MINI-20260827-SYNTH`（示例） |
| 平台 | Windows-11 |
| Python | 3.12.13 |
| PyTorch | 2.13.0 (CPU) |
| Wall clock | 24.95 s |
| TRAIN 样本 / 受试者 | 8 / 2 |
| VAL 样本 / 受试者 | 4 / 1 |
| TEST 样本 | 0（不加载） |
| Subject overlap | 0 |
| Candidate A (TinyFCN) | `NOT_FEASIBLE`（IoU=0.000000 < 0.205644；这是 1 个 epoch 训练结果的预期，不是 bug） |
| Candidate B (SmallUNet) | `NOT_FEASIBLE`（同因） |
| overall_decision | `MINI_NOT_FEASIBLE`（与 B02 阈值对比的预期结果） |
| 完整产物 | 19 / 19 已落盘（`DONE.json` + 18 个数据/日志/检查点文件） |
| Reload consistency | 两候选 `max_abs_diff=0.0`, `hash_match=true` |
| Centroid error | 真实 per-sample per-region 记录（GT-only=1.0，both-present=距离/对角线） |

> **本任务不执行真实 Mini；Synthetic CPU smoke 仅用于证明 Runner 链路。Synthetic IoU = 0 是预期的，不构成模型/B02 排名结论。**

---

## 12. 禁止结论

* ❌ 不写"Mini 完成"
* ❌ 不写"模型优于 B02"
* ❌ 不写"候选已保留"
* ❌ 不写"B07 已解锁"
* ❌ 不写"可以进入 Full 协议"
* ❌ 不读 TEST 数据
* ❌ 不运行真实 Mini
* ❌ 不宣称产品 / 硬件 / 舒适 / 医疗 / 气囊控制 / 整夜稳定性结论
* ❌ 不外推到 danaLab / uncover 之外

> 公开数据结果不外推为自研硬件、舒适性、医疗效果、整夜稳定性或气囊闭环验证（见 [AGENTS.md §9](../../AGENTS.md) 与 [COLLABORATION_WORKFLOW.md §9](../COLLABORATION_WORKFLOW.md)）。

---

## 13. 下一步

1. **Reviewer 验收**：
   * 验证 Runner 在 `--synthetic-cpu-smoke` 下不读取任何 B01 路径。
   * 验证 19 个产物齐全。
   * 验证 class weights 仅来自 TRAIN。
   * 验证 small_unet 84-width 恢复走 `F.interpolate(..., size=...)` 而非 `scale_factor`。
   * 验证 Feasibility Gate 公式（IoU >= 0.205644）与真实配置 JSON 字段一致。
2. **Owner 授权真实 Mini 后**（不属于本任务）：
   * 在 RTX 4090 / PyTorch 2.8.0+cu128（或更新）环境执行 `--run-authorized --b01-freeze-dir ... --dataset-root ...`。
   * 走 Experiment Runner + Codex Reviewer 流程。
3. **B07 Full 协议**：B04 `MINI_PROTOCOL_AND_RUNNER_READY_FOR_REVIEW` 后，由后续 task 起草 B07 Full 协议；当前任务**不**进入 B07。

---

## 14. Git 信息

* **Base:** `main @ 6e19374`
* **Branch:** `codex/task-slp-b04-pm-only-region-mini-protocol-v0.1`
* **Working tree status:** clean
* **修改文件**（待 commit，见 `git status`）：
  * 新增 `src/topper_perception/neural/slp8_region_mini.py`
  * 新增 `src/topper_perception/neural/slp8_region_class_weights.py`
  * 新增 `src/topper_perception/neural/slp8_region_metrics_ext.py`
  * 修改 `src/topper_perception/neural/slp8_region_models.py`（新增 Slp8SmallUnet + Registry）
  * 新增 `configs/experiments/slp8_pm_region_mini_v0.1.json`
  * 新增 `scripts/run_slp8_region_mini.py`
  * 新增 `tests/test_slp8_region_mini.py`
  * 修改 `tests/test_slp8_region_models.py`（新增 B04 registry 测试）
  * 新增 `docs/stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md`
  * 新增 `docs/deliverables/SLP/B04_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md`
  * 修改 `docs/deliverables/README.md`（新增 B04 索引行）

> `outputs/experiments/_b04_synth*` 与 `outputs/experiments/_b04_validate*` 属于 outputs（已 gitignore），不会被 commit。

---

**阶段交付版本：** v0.1
**生成时间：** 2026-08-27
**维护者：** Mavis (MiniMax Code)
