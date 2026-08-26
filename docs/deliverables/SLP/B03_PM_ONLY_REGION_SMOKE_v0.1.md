# B03 交付说明：SLP8 PM-only 区域分割 Smoke

**TASK-ID:** `TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1`
**Stage:** S2-B03
**日期:** 2026-08-27 (R02)
**状态:** DONE — 真实 CPU Smoke 已完成
**EXP-ID:** `EXP-SLP-B03-PM-REGION-SMOKE-20260827-R02`

---

## 这一步做什么

实现并验证 SLP8 压力图区域分割的最小化 Smoke 测试。验证从 B01 冻结表到 PyTorch 像素级分割、训练、checkpoint、resume、reload 一致性、指标计算和审计产物的完整链路。

**关键约束：**
- 仅验证 pipeline 可运行，不与 B02 排名
- 不形成 TEST 精度结论
- 不得读取 TEST 数据
- 不使用增强、不使用 class weights

---

## 实际运行结果

| 指标 | 值 |
|------|-----|
| EXP-ID | `EXP-SLP-B03-PM-REGION-SMOKE-20260827-R02` |
| 状态 | DONE |
| 运行时间 | 9.01 秒 |
| TRAIN 受试者 | `00022`, `00072` |
| TRAIN 样本数 | 90 |
| VAL 受试者 | `00005` |
| VAL 样本数 | 45 |
| TEST 样本数 | 0（不加载） |
| TRAIN/VAL subject overlap | 0 |
| 归一化 stats SHA-256 | `0b1ef18b4769f8b1b47d077cfc4c06c8310c8fff5877a6e44afcd0df2f466c59` |

**注意：** 实际 subject IDs 由 seed=42 + 排序确定，每次运行一致。

### 训练损失

| Phase | TRAIN Loss | VAL Loss |
|-------|-----------|----------|
| initial | 2.7843 | 2.4951 |
| resumed | 2.2958 | 2.2199 |

### 指标（initial phase）

| 指标 | TRAIN | VAL |
|------|-------|-----|
| fixed foreground macro IoU | 0.0303 | 0.0300 |
| fixed foreground macro Dice | 0.0574 | 0.0568 |
| pixel accuracy | 0.6592 | 0.6686 |

**重要：** 这些指标只用于验证 pipeline，**不与 B02 排名，不形成 TEST 精度结论**。

---

## 输入数据

| 数据 | 来源 | 说明 |
|------|------|------|
| SLP_8Region_Pressure_VAL_v1.1 | `<SLP8_DATASET_ROOT>` | 原始压力图和标签 |
| slp8_training_tables_v0.1 | `<B01_FREEZE_DIR>` | B01 冻结的训练表 |
| normalization_stats.json | `<B01_FREEZE_DIR>` | TRAIN-only 参考统计 |

**TEST 数据：** 明确不加载（`load_test=False`）

---

## 使用方法

### 1. 准备环境

```bash
cd <B03_WORKTREE>
uv sync
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. 运行 CPU Smoke

```bash
.venv\Scripts\python.exe scripts/run_slp8_region_smoke.py \
  --config configs/experiments/slp8_pm_region_smoke_v0.1.json \
  --output-dir outputs/experiments/EXP-SLP-B03-PM-REGION-SMOKE-20260827-R02 \
  --b01-freeze-dir <B01_FREEZE_DIR> \
  --dataset-root <SLP8_DATASET_ROOT> \
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
│   └── slp8_region_smoke.py        # Smoke 核心逻辑（实际 reload comparison）
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

## 输出文件

```
<OUTPUT_DIR>/EXP-SLP-B03-PM-REGION-SMOKE-20260827-R02/
├── status.json                    # 总体状态（DONE）
├── manifest.json                  # 运行清单
├── resolved_config.json           # 解析后的配置（路径 redact）
├── input_manifest_hashes.json     # 输入文件 SHA-256
├── runtime.json                   # 运行时信息
├── metrics_summary.json           # 完整指标
├── metrics_by_region.csv          # 每个 (split, phase, region) 的指标
├── predictions_manifest.csv       # 预测元信息（不含像素数据）
├── failure_cases.csv              # 失败案例（无失败时仅表头）
├── reload_consistency.json        # 实际比较结果
├── checkpoints/
│   ├── initial_epoch.pt          # 初始训练检查点
│   └── resumed_epoch.pt          # 恢复后检查点
├── logs/
│   └── run.log                   # 运行日志
└── DONE.json                      # 最终状态
```

---

## 模型架构

**Slp8TinyFcn** — 最小全卷积网络

```
Input [N, 1, 192, 84]
→ Conv2d(1, 8, 3, padding=1) + ReLU
→ Conv2d(8, 16, 3, padding=1) + ReLU
→ Conv2d(16, 9, 1)
→ logits [N, 9, 192, 84]
```

**参数量：** ~1,401

---

## 验证项

| 验证项 | 状态 |
|--------|------|
| TEST 访问被阻止 | ✅ |
| TRAIN/VAL subject 隔离 | ✅ |
| 模型输出尺寸正确 [N,9,192,84] | ✅ |
| Checkpoint weights_only 安全加载 | ✅ |
| Reload 一致性实际比较 | ✅ |
| 单元测试 94 通过 | ✅ |
| 回归测试 371 通过 | ✅ |
| 真实 CPU Smoke 端到端 | ✅ |

---

## 结论和下一步

### 本阶段结论

- B03 Smoke 代码实现完成
- 真实 CPU Smoke 端到端通过
- R02 Reviewer 提出的所有 ITERATE 修复项已完成
- 完整链路可运行：dataset → model → train → checkpoint → resume → reload → metrics → artifacts
- TEST 访问控制符合要求
- 不记录本机绝对路径

### 下一步

1. **Codex Reviewer 验收 R02**
2. **可选：SLP Mini/Full Run**
3. **可选：进入 S2-B04**

---

## 环境信息

| 字段 | 值 |
|------|-----|
| Git Branch | `codex/task-slp-b03-pm-only-region-smoke-v0.1` |
| Base | `origin/main` |
| B02 Merge | `ccbd539` ✅ |
| R02 Commit | 待提交 |
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
> 7. danaLab 之外的受试者

---

## Stage Report

完整分析见：[../../stage_reports/S2_B03_SLP8_PM_ONLY_REGION_SMOKE_v0.1.md](../../stage_reports/S2_B03_SLP8_PM_ONLY_REGION_SMOKE_v0.1.md)

---

**交付版本：** v0.1-R02
**生成时间：** 2026-08-27
**维护者：** Mavis (MiniMax Code)
