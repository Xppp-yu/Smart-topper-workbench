# Stage Report: S2-B04 — SLP8 PM-only Region Mini 协议与 Runner (R02)

**TASK-ID:** `TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-AND-RUNNER-v0.1`
**Stage:** S2-B04
**日期:** 2026-08-27 (R02)
**状态:** `MINI_PROTOCOL_AND_RUNNER_READY_FOR_REVIEW` — 协议、Runner、模型、指标、测试、中文文档已就绪；**未执行真实 Mini**，**未读取 TEST**，**未进入 B07**
**Base:** `main @ 6e19374`
**Branch:** `codex/task-slp-b04-pm-only-region-mini-protocol-v0.1`
**前序 commit:** `aa0fd08` (R01)
**本 commit:** `R02`

---

## 1. R02 摘要

R01 提交后，Codex Reviewer 提出多项硬化要求。R02 在 R01 基础上完成：

1. **TASK-ID 统一**：所有代码、配置、测试、文档中的 TASK-ID 改为 `TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-AND-RUNNER-v0.1`。
2. **真实资源预算执行**：`time.monotonic` 在每个验证 epoch 后检查，`reset_peak_memory_stats` / `max_memory_allocated` 在 CUDA 设备上执行。超预算 → `STOPPED`。
3. **三状态机 DONE/FAILED/STOPPED 互斥**：任一候选 FAILED → 整体 FAILED；任一候选 STOPPED → 整体 STOPPED；只有所有候选正常完成才写 `DONE.json`。三个终端文件互斥。
4. **输出目录碰撞零修改**：`OutputCollisionError` 触发后，CLI **不**写入任何新文件（包括 `FAILED.json`/`status.json`），返回 exit code 2。
5. **真实 B01 输入合同 fail-closed**：`verify_b01_contract` 校验 `train/val/test=3645/450/0` 样本数、`81/10/0` 受试者数、A06 split SHA、provenance、source_review_status、setting、cover。任意不匹配 → fail-closed。
6. **Checkpoint/Resume 契约**：每个 checkpoint 嵌入完整 identity 块（task_id, candidate, model_version, seed, config SHA, A06 SHA, freeze SHA, class-weight SHA, input-hashes SHA）。Resume 必须验证 identity 匹配；DONE 状态拒绝 resume。
7. **确定性配置 + 跨进程验证**：`apply_settings` 锁定 `PYTHONHASHSEED=42`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `cudnn.deterministic=True`, `use_deterministic_algorithms=True`。两个独立进程运行 synthetic smoke 产生 byte-identical 输出（除 wall-clock timestamps）。
8. **centroid_errors.csv 补齐字段**：包含 `candidate`, `split`, `sample_index`, `sample_id`, `subject_id`, `posture`, `region`, `error`, `valid`, `invalid_reason`, `both_missing`。
9. **预算报告**：`budget_report.json` 包含 thresholds、candidates（每个 elapsed/budget_status/budget_report）、terminal_state、determinism。

**关键禁止结论（仍生效）：**

* ❌ 不宣称 Mini 已完成
* ❌ 不宣称 B04 任何候选"优于 B02"
* ❌ 不宣称候选已保留 / 排除
* ❌ 不宣告 B07 解锁

**真实 Mini 必须先取得 Owner 运行授权 + `--run-authorized` + 真实 B01 路径，由 Experiment Runner 执行（属于 B07 协议，**不在本任务范围**）。**

---

## 2. 完整文件清单

### 新增文件
- `src/topper_perception/neural/slp8_region_mini.py` (M 1453-2660) — B04 Mini 核心 runner、配置校验、状态机、artifact 写入
- `src/topper_perception/neural/slp8_region_class_weights.py` (M 1453-2660) — Class weight 公式
- `src/topper_perception/neural/slp8_region_metrics_ext.py` (M 1453-2660) — 扩展指标（per-region/posture/subject/worst/centroid）
- `src/topper_perception/neural/slp8_region_budget.py` (R02 新增) — 资源预算 monitor（monotonic + CUDA peak）
- `src/topper_perception/neural/slp8_region_determinism.py` (R02 新增) — 确定性配置（PYTHONHASHSEED / OMP / MKL / cudnn / CPU threads）
- `src/topper_perception/neural/slp8_region_resume.py` (R02 新增) — Checkpoint identity 与 resume 校验
- `src/topper_perception/neural/slp8_region_b01_contract.py` (R02 新增) — 真实 B01 输入合同 fail-closed 校验
- `configs/experiments/slp8_pm_region_mini_v0.1.json` — 冻结配置（R02 增补 `expected_split_counts`/`expected_subjects`/`lifecycle` 段）
- `scripts/run_slp8_region_mini.py` — CLI Runner（R02 增补 `--run-authorized` gate 提前、OutputCollisionError 改返 2）
- `tests/test_slp8_region_mini.py` (R02 重写) — 130 个 B04 测试
- `docs/stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md` — 阶段报告
- `docs/deliverables/SLP/B04_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md` — 交付说明

### 修改文件
- `src/topper_perception/neural/slp8_region_models.py` — 新增 `Slp8SmallUnet`、`ModelBuilder` 注册表、`B04_MAX_PARAMETERS=150_000`
- `tests/test_slp8_region_models.py` — 新增 13 个 B04 registry 测试
- `docs/deliverables/README.md` — 新增 B04 索引行

---

## 3. 数据合同

| 字段 | 值 |
|---|---|
| TRAIN samples / subjects | 3,645 / 81 |
| VAL samples / subjects | 450 / 10 |
| TEST samples / subjects | **0 / 0**（不可访问，验证合同 `n_test_samples == 0`） |
| Pressure | float64 → float32，raw passthrough（不做 Min-Max） |
| raw_semantics | `raw_pmarray_response`（NOT kPa） |
| Provenance | `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED` |
| source_review_status | `NOT_REVIEWED` |
| A06 split SHA | `024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706` |
| Input hash SHA-256 落盘 | `input_manifest_hashes.json` |

---

## 4. 两个候选模型

| 候选 | 版本 | 架构 | 参数量 |
|---|---|---|---|
| Candidate A | `slp8_tiny_fcn_v0.1` | 复用 B03 Slp8TinyFcn（最小全卷积网络，无池化） | 1,401 |
| Candidate B | `slp8_small_unet_v0.1` | R02 新增 SmallUNet（显式 `F.interpolate(..., size=...)` 恢复 84 宽度） | 118,121 |

**B04 参数上限 150,000** 在 `slp8_region_models.py: B04_MAX_PARAMETERS = 150_000` 与 `slp8_region_mini.py` 双重固定。

---

## 5. 训练协议（全部冻结）

| 项 | 值 |
|---|---|
| seed | 42 |
| device | `cuda`（CUDA 不可用时 fail-closed） |
| batch_size | 16 |
| max_epochs | 20 |
| min_epochs | 5 |
| optimizer | AdamW（lr=1e-3, weight_decay=1e-4） |
| scheduler | none |
| augmentation | none |
| num_workers | 0 |
| early stopping | `monitor=val_loss`, `mode=min`, `patience=4`, `min_delta=0.0` |
| 每候选 seed 数 | 1 |
| 每轮保存 | `checkpoints/<candidate>/last.pt` |
| best 选择 | **最低 val_loss**（earliest-epoch tie-break） |

---

## 6. 资源预算（R02 新增执行）

| 阈值 | 值 |
|---|---|
| `max_wall_minutes_per_candidate` | 45 |
| `max_total_wall_minutes` | 90 |
| `max_peak_cuda_mb` | 12,288 |

**执行机制**：
- `ResourceBudgetState.begin_candidate()` 在每个候选开始时启动 `time.monotonic()`。
- `ResourceBudgetState.check()` 在每个验证 epoch 完成后调用。
- 候选超预算 → `feasibility="STOPPED"`, `budget_status="per_candidate_wall_exceeded"` 或 `"total_wall_exceeded"` 或 `"cuda_peak_exceeded"`。
- 整体 `terminal_state="STOPPED"`，**不**写 `DONE.json`。
- 真实 Mini 启动时（CUDA 不可用）立即 fail-closed。

---

## 7. 状态机（R02 强化）

| 条件 | 终端状态 | 写入文件 |
|---|---|---|
| 任一候选 FAILED | FAILED | `FAILED.json` |
| 任一候选 STOPPED（无 FAILED） | STOPPED | `STOPPED.json` |
| 全部候选 FEASIBLE / NOT_FEASIBLE | DONE | `DONE.json` |

三个文件互斥：写 DONE 删除 FAILED/STOPPED，写 FAILED 删除 DONE/STOPPED，写 STOPPED 删除 DONE/FAILED。

---

## 8. Class Weight 公式（TRAIN-only）

```text
raw_weight[c] = 1 / sqrt(pixel_ratio[c])
weight[c]    = raw_weight[c] / mean(raw_weight)
weight[c]    = clip(weight[c], 0.25, 4.0)
```

- 仅 TRAIN：`compute_class_weights(..., allowed_split="train")`，VAL/TEST 立即 `ClassWeightError`。
- 拒绝零比 / NaN / Inf / 缺类。
- 拒绝非有限权重。
- `assert_class_weight_invariants` 在 runner 中再次校验。
- `resolved_config.json` / `manifest.json` / `metrics_summary.json` 三处独立记录 9 个 weight。

---

## 9. 指标

| 类型 | 字段 |
|---|---|
| 整体 | `fixed_foreground_macro_iou`（1-8，不跳过空类）、`fixed_foreground_macro_dice`、`pixel_accuracy`、`background_iou`、`confusion_matrix` (9×9)、`val_loss` |
| 逐区域 | IoU / Dice / precision / recall / TP / FP / FN（8 个前景类） |
| 逐姿势 | `SUPINE` / `LEFT` / `RIGHT` / `ALL` |
| 逐受试者 | subject_id → macro IoU / Dice / pixel accuracy |
| worst subject | argmin(fixed_foreground_macro_iou) over subjects with n_samples>0 |
| 中心误差 | per-sample per-region GT/pred centroid 距离 / `√(192²+84²)`；GT-only → 1.0；both-missing → 跳过 |

`centroid_errors.csv` 列：`candidate, split, sample_index, sample_id, subject_id, posture, region, error, valid, invalid_reason, both_missing`。

---

## 10. 可行性 Gate

B02 同一 VAL 上的 best fixed foreground macro IoU 参考值 = `0.205644`（train-spatial-prior）。

候选 FEASIBLE 需同时满足 8 条（运行成功、test=0、metrics finite、checkpoint/reload、best hash 一致、metrics 完整、IoU≥0.205644、未超预算）。否则 FEASIBLE 数 0 → `MINI_NOT_FEASIBLE`，整体 STOPPED；1 → 保留；2 → 都保留（**不**宣布冠军）。

---

## 11. Checkpoint / Resume 契约（R02 新增）

每个 checkpoint 嵌入：
- `identity`：`task_id, candidate, model_version, seed, n_classes, image_shape, config_sha256, a06_split_sha256, freeze_manifest_sha256, train_class_stats_sha256, class_weight_sha256, input_manifest_hashes_sha256`
- `early_stopper`：完整可恢复状态
- `train_loss_history` / `val_loss_history` / `epoch_metrics`
- `input_manifest_hashes`
- `rng_state`（Python + NumPy + torch）

Resume 失败条件：
- `OutputCollisionError` 触发；或
- identity 任意字段不匹配；或
- 目标 `output_dir` 已含 `DONE.json`。

真实 Mini（CUDA 不可用）立即 fail-closed。

---

## 12. 确定性契约（R02 新增）

`apply_settings(seed=42, cpu_threads=1)` 配置：
- `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
- `PYTHONHASHSEED=42`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
- `torch.set_num_threads(1)`, `torch.use_deterministic_algorithms(True, warn_only=True)`
- `torch.backends.cudnn.deterministic=True`, `cudnn.benchmark=False`

**跨进程验证**：`TestDeterminismSubprocess::test_two_independent_subprocess_smoke_runs_are_byte_identical` 跑两次独立 subprocess smoke，过滤 wall-clock timing 与 PID-dependent checkpoint SHA，对剩余 artefact 内容做 SHA-256 比较。

---

## 13. Synthetic CPU Smoke 实际结果

| 字段 | 值 |
|---|---|
| EXP-ID | `EXP-SLP-B04-PM-REGION-MINI-20260827-SYNTH` |
| 模式 | `--synthetic-cpu-smoke` |
| 平台 | Windows-11 / Python 3.12.13 / PyTorch 2.13.0 (CPU) |
| Wall clock | ~30s |
| TRAIN 样本 / 受试者 | 8 / 2 |
| VAL 样本 / 受试者 | 4 / 1 |
| TEST 样本 | 0（不加载） |
| Subject overlap | 0 |
| Candidate A (TinyFCN) | `NOT_FEASIBLE`（IoU=0.0 < 0.205644；1-epoch 训练预期结果） |
| Candidate B (SmallUNet) | `NOT_FEASIBLE`（同因） |
| overall_decision | `MINI_NOT_FEASIBLE`（与 0.205644 阈值对比的预期结果） |
| 完整产物 | 20 / 20（19 数据/日志 + 1 status） |
| Reload consistency | 两候选 `max_abs_diff=0.0`, `hash_match=true` |
| Cross-process determinism | two subprocess runs byte-identical（after scrubbing wall-clock + PID） |

> **本任务不执行真实 Mini；Synthetic CPU smoke 仅用于证明 Runner 链路。Synthetic IoU = 0 是预期的，不构成 B04 候选与 B02 的排名。**

---

## 14. 测试覆盖（R02）

`tests/test_slp8_region_mini.py` — **130 个 B04 测试** + `tests/test_slp8_region_models.py` 新增 13 个 B04 registry 测试 = **143 个 B04 新增测试**，全部通过。

覆盖：
- TASK-ID 统一
- SmallUNet 输入/输出 shape、84 宽度恢复、参数 ≤ 150,000
- 两个候选 registry
- Class weight 仅来自 TRAIN（拒绝 VAL/TEST/零比/NaN/Inf/缺类）
- 权重公式 / 归一化 / clip / 非有限拒绝
- 配置缺字段 / 非法值 fail-closed（含 `expected_split_counts`, `expected_subjects`, `lifecycle`）
- CUDA 不可用 fail-closed；synthetic CPU 烟雾唯一 CPU 路径
- early stop 只能监控 `val_loss`
- non-finite loss / metric 拒绝
- checkpoint / resume / reload 独立验证 + prediction hash 一致
- DONE / FAILED / STOPPED 互斥
- `centroid_errors.csv` 包含 `sample_id`, `subject_id`, `posture`, `valid`, `invalid_reason`
- 固定前景类 1–8 不跳过空类
- per-subject / per-posture / worst subject
- output collision 拒绝
- `--run-authorized` gate + 碰撞零修改
- 资源预算（tiny budget → STOPPED）
- Determinism（PYTHONHASHSEED=42、跨进程 byte-identical）
- B01 contract 真实输入 fail-closed
- resume identity 校验

联合回归（不带真实数据）：**1313 passed, 4 skipped**（与本任务无关的 pre-existing skip）。

---

## 15. Git 信息

* **Base:** `main @ 6e19374`
* **Branch:** `codex/task-slp-b04-pm-only-region-mini-protocol-v0.1`
* **前序 commit:** `aa0fd08`（R01）
* **Working tree status:** clean（R02 待 commit）

---

**阶段交付版本：** v0.1-R02
**生成时间：** 2026-08-27
**维护者：** Mavis (MiniMax Code)
