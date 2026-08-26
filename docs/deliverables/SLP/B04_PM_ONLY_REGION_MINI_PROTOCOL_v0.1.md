# B04 交付说明：SLP8 PM-only 区域分割 Mini 协议与 Runner (R02)

**TASK-ID:** `TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-AND-RUNNER-v0.1`
**Stage:** S2-B04
**日期:** 2026-08-27 (R02)
**状态:** `MINI_PROTOCOL_AND_RUNNER_READY_FOR_REVIEW`
**Base:** `main @ 6e19374`
**Branch:** `codex/task-slp-b04-pm-only-region-mini-protocol-v0.1`

---

## 这一步做什么（R02 强化）

R01 提交后，Codex Reviewer 提出多项硬化要求。R02 完成：

* **TASK-ID 统一**为 `TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-AND-RUNNER-v0.1`（覆盖代码、配置、测试、文档）
* **真实资源预算执行**：`time.monotonic` 每 epoch 检查；CUDA peak memory 跟踪（`reset_peak_memory_stats` / `max_memory_allocated`）；超预算 → `STOPPED`
* **三状态机** DONE/FAILED/STOPPED **互斥**：任一 FAILED → 整体 FAILED；任一 STOPPED → 整体 STOPPED；否则 DONE
* **输出目录碰撞零修改**：`OutputCollisionError` 不写 `FAILED.json`/`status.json`（CLI exit code 2）
* **真实 B01 输入合同 fail-closed**：counts / subject counts / A06 SHA / provenance / setting / cover 全部强制
* **Checkpoint/Resume 契约**：完整 identity 块嵌入每个 checkpoint；resume 必须验证
* **确定性配置 + 跨进程验证**：`PYTHONHASHSEED=42`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, cudnn deterministic, `use_deterministic_algorithms`；两独立 subprocess 跑 byte-identical 输出
* **centroid_errors.csv 补齐字段**：`candidate, split, sample_index, sample_id, subject_id, posture, region, error, valid, invalid_reason, both_missing`
* **预算报告** `budget_report.json` 包含 thresholds / candidates / terminal_state / determinism

仍生效的禁止结论：
* ❌ 不宣称 Mini 已完成
* ❌ 不宣称 B04 任何候选"优于 B02"
* ❌ 不宣称候选已保留 / 排除
* ❌ 不宣告 B07 解锁

**真实 Mini 必须先取得 Owner 运行授权 + `--run-authorized` + 真实 B01 路径，由 Experiment Runner 执行（B07 协议范围，**不在本任务**）。**

---

## 实际结果（Synthetic CPU Smoke）

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
| Candidate A (TinyFCN) | `NOT_FEASIBLE`（IoU=0.0 < 0.205644；预期） |
| Candidate B (SmallUNet) | `NOT_FEASIBLE`（同因） |
| overall_decision | `MINI_NOT_FEASIBLE` |
| 完整产物 | 20 / 20（19 数据/日志 + 1 status） |
| Reload consistency | 两候选 `max_abs_diff=0.0`, `hash_match=true` |
| Cross-process determinism | byte-identical after scrubbing |

> **重要：** Synthetic smoke 仅证明 Runner 链路；**IoU=0 来自合成数据 + 1 epoch 训练**，**不构成 B04 候选与 B02 的排名**。

---

## 使用方法

### 1. 准备环境

```powershell
cd <B04_WORKTREE>
uv sync
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. 校验配置 / Registry（无需 B01 路径）

```powershell
.venv\Scripts\python.exe scripts\run_slp8_region_mini.py `
  --config configs\experiments\slp8_pm_region_mini_v0.1.json `
  --output-dir outputs\experiments\EXP-SLP-B04-PM-REGION-MINI-20260827-VALIDATE `
  --validate-config
```

### 3. 合成 CPU 烟雾

```powershell
.venv\Scripts\python.exe scripts\run_slp8_region_mini.py `
  --config configs\experiments\slp8_pm_region_mini_v0.1.json `
  --output-dir outputs\experiments\EXP-SLP-B04-PM-REGION-MINI-20260827-SYNTH `
  --synthetic-cpu-smoke
```

### 4. 真实 Mini（**本任务不执行**；需 Owner 授权）

```powershell
.venv\Scripts\python.exe scripts\run_slp8_region_mini.py `
  --config configs\experiments\slp8_pm_region_mini_v0.1.json `
  --output-dir outputs\experiments\EXP-SLP-B04-PM-REGION-MINI-20260827-R01 `
  --b01-freeze-dir <B01_FREEZE_DIR> `
  --dataset-root <SLP8_DATASET_ROOT> `
  --run-authorized
```

* `--run-authorized` 是显式开关；未带但传了真实路径会立即拒绝（`MiniProtocolError`，CLI 不创建任何文件，退出码 2）
* 真实 Mini **必须**在 `device='cuda'` 上执行；CUDA 不可用时立即 fail-closed（不静默回退 CPU）

---

## 代码和配置在哪里

```
<B04_WORKTREE>/
├── src/topper_perception/neural/
│   ├── slp8_region_models.py            # 修改：Slp8SmallUnet + ModelBuilder 注册表
│   ├── slp8_region_class_weights.py     # 新增：TRAIN-only class weight 公式
│   ├── slp8_region_metrics_ext.py       # 新增：扩展指标
│   ├── slp8_region_mini.py              # 新增：Mini 核心 runner
│   ├── slp8_region_budget.py            # R02 新增：资源预算 monitor
│   ├── slp8_region_determinism.py       # R02 新增：确定性配置
│   ├── slp8_region_resume.py            # R02 新增：Resume identity 校验
│   └── slp8_region_b01_contract.py      # R02 新增：真实 B01 输入合同
├── scripts/
│   └── run_slp8_region_mini.py          # CLI Runner
├── configs/experiments/
│   └── slp8_pm_region_mini_v0.1.json    # 冻结配置（含 lifecycle / expected_* 段）
├── tests/
│   ├── test_slp8_region_mini.py         # R02 重写：130 个 B04 测试
│   └── test_slp8_region_models.py        # 修改：+13 个 B04 registry 测试
└── docs/
    ├── stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md
    └── deliverables/SLP/B04_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md（本文件）
```

---

## 输出文件（R02 含 20 个）

```
<OUTPUT_DIR>/
├── DONE.json OR FAILED.json OR STOPPED.json     # 互斥
├── status.json
├── manifest.json
├── resolved_config.json
├── input_manifest_hashes.json
├── environment.json
├── epoch_metrics.csv
├── metrics_summary.json
├── metrics_by_region.csv
├── metrics_by_subject.csv
├── metrics_by_posture.csv
├── centroid_errors.csv
├── worst_subject.json
├── confusion_matrix.csv
├── predictions_manifest.csv
├── candidate_decision.json
├── reload_consistency.json
├── budget_report.json                         # R02 新增
├── checkpoints/
│   ├── slp8_tiny_fcn_v0.1/
│   │   ├── last.pt
│   │   └── best.pt
│   └── slp8_small_unet_v0.1/
│       ├── last.pt
│       └── best.pt
└── logs/
    └── run.log
```

---

## 状态机与三种终态

| 条件 | 终端状态 | 写入文件 | CLI 退出码 |
|---|---|---|---|
| 全部候选正常完成 | `DONE` | `DONE.json` | 0 |
| 任一候选 FAILED | `FAILED` | `FAILED.json` | 1 |
| 任一候选 STOPPED（无 FAILED） | `STOPPED` | `STOPPED.json` | 1 |
| 输出目录碰撞 | n/a | 不写任何文件 | 2 |
| 无 `--run-authorized` 但传了 B01 路径 | n/a | 不写任何文件 | 2 |

三个终端文件 **互斥**（写一个会自动删另外两个）。

---

## 验证项

| 验证项 | 状态 |
|---|---|
| TEST 访问被阻止 | ✅（B01 `enable_test_access` 合同 + Runner `load_test=False`） |
| 压力保持 raw_pmarray_response | ✅ |
| Subject 隔离 | ✅ |
| SmallUNet 84 宽度显式恢复 | ✅（`F.interpolate(..., size=...)`） |
| 参数 ≤ 150,000 | ✅（118,121 / 150,000） |
| Class weight 仅来自 TRAIN | ✅ |
| Feasibility Gate 公式（IoU ≥ 0.205644） | ✅ |
| `--run-authorized` gate | ✅ |
| 输出目录碰撞 → 零修改 + 退出码 2 | ✅ |
| DONE / FAILED / STOPPED 互斥 | ✅ |
| predictions_manifest 真实数据 | ✅ |
| centroid_errors 真实 per-sample per-region | ✅ |
| 资源预算（time.monotonic + CUDA peak） | ✅ |
| Checkpoint identity + resume 校验 | ✅ |
| 真实 B01 输入合同 fail-closed | ✅ |
| 确定性（PYTHONHASHSEED + 两独立进程 byte-identical） | ✅ |
| B04 定向测试 130 个 | ✅ |
| B03 回归 | ✅（25 个模型 + 全部 B03 集成） |
| B01 / B02 / pressure infrastructure / experiment infra 回归 | ✅ |
| 联合回归 | ✅ 1313 passed, 4 skipped（pre-existing） |
| `git diff --check` | ✅（待 commit 前验证） |

---

## 环境信息

| 字段 | 值 |
|---|---|
| Git Branch | `codex/task-slp-b04-pm-only-region-mini-protocol-v0.1` |
| Base | `main @ 6e19374` |
| B03 Merge | `6e19374` ✅ |
| R01 commit | `aa0fd08` |
| R02 commit | （待 commit） |
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

完整分析：[../../stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md](../../stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md)

---

**交付版本：** v0.1-R02
**生成时间：** 2026-08-27
**维护者：** Mavis (MiniMax Code)
