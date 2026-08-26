# B03 交付说明：SLP8 PM-only 区域分割 Smoke

**TASK-ID:** `TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1`
**Stage:** S2-B03
**日期:** 2026-08-27 (R03)
**状态:** DONE_WITH_LIMITATIONS — 已通过 Codex Reviewer 独立验收
**EXP-ID:** `EXP-SLP-B03-PM-REGION-SMOKE-20260827-R03`
**R02 Commit:** `885b04a`
**R03 Commit:** `8979c6f`

---

## 这一步做什么

实现并验证 SLP8 压力图区域分割的最小化 Smoke 测试。验证从 B01 冻结表到 PyTorch 像素级分割、训练、checkpoint、resume、reload 一致性、指标计算和审计产物的完整链路。

**关键约束：**
- 仅验证 pipeline 可运行，不与 B02 排名
- 不形成 TEST 精度结论
- 不得读取 TEST 数据
- 不使用增强、不使用 class weights

---

## 实际运行结果（R03）

| 字段 | 值 |
|------|-----|
| EXP-ID | `EXP-SLP-B03-PM-REGION-SMOKE-20260827-R03` |
| 状态 | DONE |
| 平台 | Windows-11 |
| Python | 3.12.13 |
| Wall clock | 4.33 s |
| TRAIN 受试者 | `00022`, `00072` |
| TRAIN 样本数 | 90 |
| VAL 受试者 | `00005` |
| VAL 样本数 | 45 |
| TEST 样本数 | 0（不加载） |
| TRAIN/VAL subject overlap | 0 |
| 归一化 stats SHA-256 | `0b1ef18b4769f8b1b47d077cfc4c06c8310c8fff5877a6e44afcd0df2f466c59` |

**Subject 选择规则：** 先按 subject_id 字符串排序，再用 `random.Random(seed=42).shuffle()` 确定性 shuffle 后取前 N 个。

### 训练损失

| Phase | TRAIN Loss | VAL Loss |
|-------|-----------|----------|
| initial | 2.7843 | 2.4951 |
| resumed | 2.2958 | 2.2199 |

### 指标（直接来自 metrics_summary.json）

#### initial phase

| 指标 | TRAIN | VAL |
|------|-------|-----|
| fixed foreground macro IoU | 0.030338 | 0.029954 |
| fixed foreground macro Dice | 0.057433 | 0.056802 |
| pixel accuracy | 0.659218 | **0.677049** |

#### resumed phase

| 指标 | TRAIN | VAL |
|------|-------|-----|
| fixed foreground macro IoU | **0.035363** | **0.034138** |
| fixed foreground macro Dice | **0.066868** | **0.064863** |
| pixel accuracy | **0.684938** | **0.699113** |

**重要：** 这些指标只用于验证 pipeline，**不与 B02 排名，不形成 TEST 精度结论**。

---

## 使用方法

### 1. 准备环境

```powershell
cd <B03_WORKTREE>
uv sync
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. 运行 CPU Smoke

**PowerShell（反引号 ` 续行）**：

```powershell
.venv\Scripts\python.exe scripts/run_slp8_region_smoke.py `
  --config configs/experiments/slp8_pm_region_smoke_v0.1.json `
  --output-dir outputs/experiments/EXP-SLP-B03-PM-REGION-SMOKE-20260827-R03 `
  --b01-freeze-dir <B01_FREEZE_DIR> `
  --dataset-root <SLP8_DATASET_ROOT> `
  --device cpu
```

**参数说明：**
- `--config`: 实验配置（嵌套结构）
- `--output-dir`: 输出目录（必须为空或不存在）
- `--b01-freeze-dir`: B01 冻结表目录
- `--dataset-root`: SLP8 数据集根目录
- `--device`: 设备（cpu 或 cuda）

---

## 代码和配置在哪里

```
<B03_WORKTREE>/
├── src/topper_perception/neural/
│   ├── slp8_region_dataset.py      # Dataset 类（raw passthrough normalization）
│   ├── slp8_region_models.py       # Slp8TinyFcn 模型
│   ├── slp8_region_checkpoint.py   # Checkpoint 管理（weights_only=True）
│   └── slp8_region_smoke.py        # Smoke 核心逻辑（real predictions + subset from config）
├── scripts/
│   └── run_slp8_region_smoke.py   # CLI Runner（嵌套 config + fail-closed 验证）
├── configs/experiments/
│   └── slp8_pm_region_smoke_v0.1.json
├── tests/
│   ├── test_slp8_region_dataset.py
│   ├── test_slp8_region_models.py
│   └── test_slp8_region_smoke.py
└── docs/stage_reports/
    └── S2_B03_SLP8_PM_ONLY_REGION_SMOKE_v0.1.md
```

---

## 输出文件（R03）

```
<OUTPUT_DIR>/EXP-SLP-B03-PM-REGION-SMOKE-20260827-R03/
├── DONE.json                       ✅
├── status.json                     ✅
├── manifest.json                   ✅
├── resolved_config.json            ✅ (无本机绝对路径)
├── input_manifest_hashes.json      ✅
├── runtime.json                    ✅
├── metrics_summary.json            ✅
├── metrics_by_region.csv           ✅
├── predictions_manifest.csv        ✅ (270 行真实数据)
├── failure_cases.csv               ✅ (仅表头)
├── reload_consistency.json         ✅
├── checkpoints/
│   ├── initial_epoch.pt            ✅
│   └── resumed_epoch.pt            ✅
└── logs/
    └── run.log                     ✅
```

`predictions_manifest.csv` 包含 270 行真实记录（90 train initial + 45 val initial + 90 train resumed + 45 val resumed），无占位符；每行带真实 `sample_id`、`subject_id`、`label_sha256`、`prediction_sha256`（64 位小写 hex）。

---

## Canonical Array Hash 规则

`canonical_array_hash` 在 smoke 模块顶部定义：

1. 转为 int64 C-contiguous（`np.ascontiguousarray(arr, dtype=np.int64)`）
2. 拼装 header：`slp8_canonical_array_hash_v0.1\ndtype=<i8\nshape=(H,W)\n`
3. 拼接 C-order bytes
4. SHA-256 → 64 位小写 hex

Header 用于防止不同 shape 但相同 byte 内容的碰撞。

---

## 验证项

| 验证项 | 状态 |
|--------|------|
| TEST 访问被阻止 | ✅ |
| TRAIN/VAL subject 隔离 | ✅ |
| 模型输出尺寸正确 [N,9,192,84] | ✅ |
| Checkpoint weights_only 安全加载 | ✅ |
| Reload 一致性实际比较 | ✅ |
| predictions_manifest 真实数据 | ✅ |
| canonical hash 规则统一 | ✅ |
| subset config 真正传入 run_smoke_test | ✅ |
| 单元测试 108 通过 | ✅ |
| 回归测试 371 通过 | ✅ |
| 真实 CPU Smoke 端到端 | ✅ |
| R02 输出未被覆盖 | ✅ |

---

## 结论和下一步

### 本阶段结论

- B03 Smoke 代码实现完成
- 真实 CPU Smoke 端到端通过（R03 EXP-ID）
- 完整链路可运行：dataset → model → train → checkpoint → resume → reload → metrics → artifacts
- TEST 访问控制符合要求
- 不记录本机绝对路径
- R02 输出未被覆盖

### 下一步

1. **S2-B04 已解锁为 READY**
2. 先冻结 Mini 协议、候选、指标、资源和停止条件
3. 未经 Owner 授权，不直接运行 GPU Mini/Full 或读取 TEST

---

## 环境信息

| 字段 | 值 |
|------|-----|
| Git Branch | `codex/task-slp-b03-pm-only-region-smoke-v0.1` |
| Base | `origin/main` |
| B02 Merge | `ccbd539` ✅ |
| R01 commit | `6219411` |
| R02 commit | `885b04a` |
| R03 commit | `8979c6f` |
| seed | 42 |
| Python | 3.12.13 |
| PyTorch | CPU |
| Platform | Windows-11 |

---

## 禁止结论

> 1. Smoke 指标代表 TEST 性能
> 2. 超过 B02 基线
> 3. 适用于产品决策
> 4. GT 是人类像素级标注
> 5. 压力值代表 kPa
> 6. 适用于 cover1/cover2
> 7. 适用于 danaLab 之外

---

## Stage Report

完整分析见：[../../stage_reports/S2_B03_SLP8_PM_ONLY_REGION_SMOKE_v0.1.md](../../stage_reports/S2_B03_SLP8_PM_ONLY_REGION_SMOKE_v0.1.md)

---

**交付版本：** v0.1-R03
**生成时间：** 2026-08-27
**维护者：** Mavis (MiniMax Code)
