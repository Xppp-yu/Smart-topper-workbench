# B04 交付说明：SLP8 PM-only 区域分割 Mini 协议与 Runner (R03)

**TASK-ID:** `TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-AND-RUNNER-v0.1`
**Stage:** S2-B04
**日期:** 2026-08-27 (R03)
**状态:** `MINI_PROTOCOL_AND_RUNNER_READY_FOR_REVIEW`
**Base:** `main @ 6e19374`
**Branch:** `codex/task-slp-b04-pm-only-region-mini-protocol-v0.1`

---

## 这一步做什么（R03 强化）

R02 之后 Code Reviewer 再次提出硬化要求，R03 完成：

* **CLI 真正接通 `result.terminal_state`**：DONE → `DONE.json` + exit 0；FAILED → `FAILED.json` + exit 1；STOPPED → `STOPPED.json` + exit 1。三条入口（`--validate-config` / `--synthetic-cpu-smoke` / `--run-authorized`）都验证。
* **真实 B01 输入合同 fail-closed**：`verify_b01_contract` + `check_freeze_manifest_file_consistency` 在 `_run_real_b01` 中真实接通；fake-freeze 负向测试覆盖 train_count/A06/provenance/setting/cover 全部 fail-closed；WARNING 分支删除；TEST rows 在结构上有 495，但 loader 永不让其进入 dataset。
* **`--resume-from` 真实接入**：自动检测 `checkpoints/<candidate>/last.pt`；DONE 状态拒绝 resume；保存/恢复 `current_patience`、budget 累计、optimizer/RNG/history/identity；空 config_sha256 fail-closed。
* **CUDA 确定性 fail-closed**：`warn_only = not cuda_available()`；`CUBLAS_WORKSPACE_CONFIG=":4096:8"` 在 CUDA 初始化前导出；当前为 CPU，记录 `run_mode="cpu_synthetic_reproducible"`；CUDA 仍 **NOT RUN**。
* **正确流程**：B04 Runner Reviewer → Owner 授权 → Experiment Runner → Codex 审核 → 通过后解锁 B07。删除"B04 Real Mini 属于 B07 协议"措辞。

---

## 正确流程（R03 明确）

```
B04 Runner Reviewer 验收
  → Owner 单独授权 B04 Real Mini
  → Experiment Runner 执行 B04 Mini（带 --run-authorized + 真实 B01 路径）
  → Codex 审核真实结果
  → 通过后才解锁 B07
```

**B04 Real Mini 不再"属于 B07 协议"；它由 Owner 显式授权 + Experiment Runner 执行 + Codex 审核完成。**

---

## 实际结果（Synthetic CPU Smoke）

| 字段 | 值 |
|---|---|
| EXP-ID | `EXP-SLP-B04-PM-REGION-MINI-20260827-SYNTH` |
| 模式 | `--synthetic-cpu-smoke` |
| 平台 | Windows-11 / Python 3.12.13 / PyTorch 2.13.0 (CPU) |
| Wall clock | ~30s |
| terminal_state | DONE |
| TRAIN 样本 / 受试者 | 8 / 2 |
| VAL 样本 / 受试者 | 4 / 1 |
| TEST 样本 | 0（不加载） |
| Subject overlap | 0 |
| Candidate A (TinyFCN) | `NOT_FEASIBLE`（IoU=0.0 < 0.205644；预期） |
| Candidate B (SmallUNet) | `NOT_FEASIBLE`（同因） |
| overall_decision | `MINI_NOT_FEASIBLE` |
| 完整产物 | 20 / 20 |
| Reload consistency | 两候选 `max_abs_diff=0.0`, `hash_match=true` |
| Cross-process determinism | byte-identical after scrubbing |

> **重要：** Synthetic smoke 仅证明 Runner 链路；**IoU=0 来自合成数据 + 1 epoch 训练**，**不构成 B04 候选与 B02 的排名**。

STOPPED 路径通过 `B04_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS=0.001` env var 触发（test-only）：
- 终端文件：`STOPPED.json`
- exit code：1
- 候选 feasibility：`STOPPED` + `budget_status=per_candidate_wall_exceeded`

---

## 使用方法

### 1. 准备环境

```powershell
cd <B04_WORKTREE>
uv sync
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. 校验配置 / Registry

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

### 4. Resume

```powershell
.venv\Scripts\python.exe scripts\run_slp8_region_mini.py `
  --config configs\experiments\slp8_pm_region_mini_v0.1.json `
  --output-dir outputs\experiments\EXP-SLP-B04-PM-REGION-MINI-20260827-RESUMED `
  --synthetic-cpu-smoke `
  --resume-from outputs\experiments\EXP-SLP-B04-PM-REGION-MINI-20260827-INTERRUPTED
```

* `--resume-from <path>` 指向之前中断的 B04 output 目录
* 自动检测 `checkpoints/<candidate>/last.pt` 并从每个候选的最新 epoch 恢复
* 若目标 output_dir 包含 `DONE.json` → `ResumeRefusedError`（exit 2，不写新文件）

### 5. 真实 Mini（**本任务不执行**；需 Owner 授权 + Experiment Runner）

```powershell
.venv\Scripts\python.exe scripts\run_slp8_region_mini.py `
  --config configs\experiments\slp8_pm_region_mini_v0.1.json `
  --output-dir outputs\experiments\EXP-SLP-B04-PM-REGION-MINI-20260827-R01 `
  --b01-freeze-dir <B01_FREEZE_DIR> `
  --dataset-root <SLP8_DATASET_ROOT> `
  --run-authorized
```

* `--run-authorized` 是显式开关；未带但传了真实路径会立即拒绝（CLI exit 2，**不**创建 output 目录）
* 真实 Mini **必须**在 `device='cuda'` 上执行；CUDA 不可用时立即 fail-closed（不静默回退 CPU）
* 输入合同（counts / subjects / A06 SHA / provenance / setting / cover / freeze-manifest-SHA）由 `_run_real_b01` 在训练前 fail-closed 校验
* 真实 Mini 的真值结果由 Codex 审核；通过后才解锁 B07

---

## 代码和配置在哪里

```
<B04_WORKTREE>/
├── src/topper_perception/neural/
│   ├── slp8_region_models.py             # Slp8SmallUnet + ModelBuilder 注册表
│   ├── slp8_region_class_weights.py      # TRAIN-only class weight 公式
│   ├── slp8_region_metrics_ext.py        # 扩展指标
│   ├── slp8_region_mini.py               # Mini 核心 runner
│   ├── slp8_region_budget.py             # ResourceBudget + BudgetAccumulatorState
│   ├── slp8_region_determinism.py        # apply_settings + CUBLAS workspace
│   ├── slp8_region_resume.py             # CheckpointIdentity + EarlyStopperState + refuse_resume_for_done_run
│   ├── slp8_region_b01_contract.py       # B01FreezeSnapshot + verify_b01_contract
│   └── ...
├── scripts/
│   └── run_slp8_region_mini.py           # CLI Runner
├── configs/experiments/
│   └── slp8_pm_region_mini_v0.1.json     # 冻结配置
├── tests/
│   └── test_slp8_region_mini.py          # 147 个 B04 测试（R03 强化）
└── docs/
    ├── stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md
    └── deliverables/SLP/B04_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md（本文件）
```

---

## 输出文件（R03 含 20 个）

```
<OUTPUT_DIR>/
├── DONE.json OR FAILED.json OR STOPPED.json    # 互斥
├── status.json                                 # 与终端文件一致
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
├── budget_report.json
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
| 无 `--run-authorized` 但传了 B01 路径 | n/a | 不创建 output 目录 | 2 |
| 真实 B01 输入合同 fail | n/a | `FAILED.json` | 1 |

三个终端文件 **互斥**。

---

## 验证项

| 验证项 | 状态 |
|---|---|
| TEST 访问被阻止 | ✅（B01 `enable_test_access` 合同 + Runner `load_test=False` + freeze loader TEST leak guard） |
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
| 真实 B01 输入合同 fail-closed | ✅ |
| Checkpoint identity + resume 校验 | ✅ |
| 确定性配置（PYTHONHASHSEED + CUBLAS + CPU threads=1） | ✅ |
| `current_patience` 保存/恢复 | ✅ |
| `BudgetAccumulatorState` 保存/恢复 | ✅ |
| 真实 `config_sha256`（拒绝空） | ✅ |
| interrupted K + resume N == uninterrupted N | ✅ |
| 两独立 subprocess 跑 byte-identical | ✅ |
| B04 定向测试 147 个 | ✅ |
| B03 回归 | ✅ |
| B01/B02/pressure/experiment infra 回归 | ✅ |
| 联合回归 | ✅ 1313 passed, 4 skipped |
| `git diff --check` | ✅ |

---

## 环境信息

| 字段 | 值 |
|---|---|
| Git Branch | `codex/task-slp-b04-pm-only-region-mini-protocol-v0.1` |
| Base | `main @ 6e19374` |
| R01 commit | `aa0fd08` |
| R02 commit | `7f210fb` |
| R03 commit | （待 commit） |
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
> 12. CUDA determinism 已端到端验证（**NOT RUN**）

---

## Stage Report

完整分析：[../../stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md](../../stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md)

---

**交付版本：** v0.1-R03
**生成时间：** 2026-08-27
**维护者：** Mavis (MiniMax Code)
