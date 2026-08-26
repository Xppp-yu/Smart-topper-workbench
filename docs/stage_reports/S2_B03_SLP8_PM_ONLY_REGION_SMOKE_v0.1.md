# Stage Report: S2-B03 — SLP8 PM-only Region Segmentation Smoke

**TASK-ID:** `TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1`
**Stage:** S2-B03
**日期:** 2026-08-27 (R03)
**状态:** DONE — R03 收口完成，全部测试通过且真实 CPU Smoke 已运行
**EXP-ID:** `EXP-SLP-B03-PM-REGION-SMOKE-20260827-R03`
**R02 Commit:** `885b04a`
**R03 Commit:** 见 Git log（HEAD）

---

## 摘要

B03 阶段实现并验证了 SLP8 压力图区域分割的最小化 Smoke 测试，覆盖从 B01 冻结表到 PyTorch 像素级分割、训练、checkpoint、resume、reload 一致性、指标计算和审计产物的完整链路。

R03 在 R02 基础上完成以下收口：
1. **predictions_manifest.csv** 不再使用占位符（`train_sample_000000` 等），改为逐条来自 `_collect_predictions` 收集的真实预测，附带样本 ID、subject ID、label/prediction SHA-256、shape 和失败原因。
2. **配置 subset 消费** 不再硬编码 `n_train_subjects=2 / n_val_subjects=1`，改从 `SmokeConfig` 读取并真实传入 `run_smoke_test`。
3. **canonical array hash** 规则统一：先转 int64 contiguous，再拼装 `version + dtype + shape` header + C-order bytes 后取 SHA-256。
4. **程序化检查脚本** 在 R03 真实运行后通过全部断言（270 行真实记录、64 位小写 hex、无占位符、reload consistent=true、R02 未被覆盖）。

**重要：** Smoke 只验证 pipeline 可运行，不与 B02 排名，不形成 TEST 精度结论。

---

## 1. Canonical Array Hash 规则

B03 使用统一的 `canonical_array_hash` 函数计算 label / prediction 的 SHA-256：

1. **统一 dtype**：将输入数组转为 `int64` 的 C-contiguous 视图（`np.ascontiguousarray(arr, dtype=np.int64)`）。
2. **附加 header**：
   ```
   slp8_canonical_array_hash_v0.1
   dtype=<i8
   shape=(H, W)
   ```
3. **拼接字节**：`header_bytes + arr_int.tobytes()`。
4. **SHA-256**：对拼接结果计算 SHA-256，返回 64 位小写 hex。

Header 防止不同 shape 但相同 byte 内容的数组发生碰撞（已在测试中验证）。

---

## 2. Subject 子集选择规则

`select_smoke_subjects` 实现确定性选择：

1. 按 subject_id **字符串排序**得到有序列表。
2. 用 `random.Random(seed).shuffle()` 对每个 split 的有序 subject 列表做确定性 shuffle。
3. 取前 `n_train_subjects` / `n_val_subjects` 个。

> 即"先按 subject ID 排序，再使用固定 seed=42 确定性 shuffle 后选取"，不是单纯"排序后前 N 名"。

TRAIN/VAL subject overlap 由 `verify_subject_isolation` 强制校验为 0。

---

## 3. R03 真实 CPU Smoke 结果

### 3.1 运行命令

**Windows PowerShell（使用反引号 ` 续行）**：

```powershell
.venv\Scripts\python.exe scripts/run_slp8_region_smoke.py `
  --config configs/experiments/slp8_pm_region_smoke_v0.1.json `
  --output-dir outputs/experiments/EXP-SLP-B03-PM-REGION-SMOKE-20260827-R03 `
  --b01-freeze-dir <B01_FREEZE_DIR> `
  --dataset-root <SLP8_DATASET_ROOT> `
  --device cpu
```

**或单行形式**：

```powershell
.venv\Scripts\python.exe scripts/run_slp8_region_smoke.py --config configs/experiments/slp8_pm_region_smoke_v0.1.json --output-dir outputs/experiments/EXP-SLP-B03-PM-REGION-SMOKE-20260827-R03 --b01-freeze-dir <B01_FREEZE_DIR> --dataset-root <SLP8_DATASET_ROOT> --device cpu
```

### 3.2 关键值

| 字段 | 值 |
|------|-----|
| EXP-ID | `EXP-SLP-B03-PM-REGION-SMOKE-20260827-R03` |
| 状态 | DONE |
| 平台 | Windows-11 |
| Python | 3.12.13 |
| PyTorch | CPU |
| Wall clock | 4.33 s |
| seed | 42 |
| TRAIN 受试者 | `00022`, `00072` |
| TRAIN 样本数 | 90 |
| VAL 受试者 | `00005` |
| VAL 样本数 | 45 |
| TEST 样本数 | 0（不加载） |
| TRAIN/VAL subject overlap | 0 |
| 归一化 stats SHA-256 | `0b1ef18b4769f8b1b47d077cfc4c06c8310c8fff5877a6e44afcd0df2f466c59` |

### 3.3 训练损失

| Phase | TRAIN Loss | VAL Loss |
|-------|-----------|----------|
| initial | 2.7843 | 2.4951 |
| resumed | 2.2958 | 2.2199 |

### 3.4 指标（直接读自 metrics_summary.json）

#### initial phase

| 指标 | TRAIN | VAL |
|------|-------|-----|
| fixed foreground macro IoU | 0.030338 | 0.029954 |
| fixed foreground macro Dice | 0.057433 | 0.056802 |
| pixel accuracy | 0.659218 | **0.677049** |
| background IoU | 0.0 | 0.0 |
| n_classes_present_in_pred | 8 | 8 |
| n_classes_present_in_gt | 8 | 8 |

#### resumed phase

| 指标 | TRAIN | VAL |
|------|-------|-----|
| fixed foreground macro IoU | **0.035363** | **0.034138** |
| fixed foreground macro Dice | **0.066868** | **0.064863** |
| pixel accuracy | **0.684938** | **0.699113** |
| background IoU | 0.0 | 0.0 |
| n_classes_present_in_pred | 8 | 8 |
| n_classes_present_in_gt | 8 | 8 |

**重要：** 这些指标只用于验证 pipeline，**不与 B02 排名，不形成 TEST 精度结论**。1 epoch 训练精度低符合预期。

### 3.5 Checkpoint SHA-256

| Checkpoint | SHA-256（前 16 位） |
|-----------|---------------------|
| `initial_epoch.pt` | `55b91f22df548502...` |
| `resumed_epoch.pt` | `f9020f4d73a3dd42...` |

完整 SHA-256 见 `metrics_summary.json`。

### 3.6 参数变化证据

| 阶段 | total_diff | 参数是否改变 |
|------|-----------|-----------|
| initial 训练后 | > 1e-6 | ✅ |
| resume 训练后 | > 1e-6 | ✅ |

### 3.7 Reload 一致性（实际比较结果）

| 指标 | 值 |
|------|-----|
| `consistent` | `true` |
| `max_abs_diff` | `0.0` |
| 比较方法 | `torch.allclose(rtol=1e-5, atol=1e-6)` |

---

## 4. 产物清单（R03）

```
<OUTPUT_DIR>/EXP-SLP-B03-PM-REGION-SMOKE-20260827-R03/
├── DONE.json                       ✅
├── status.json                     ✅ (status=DONE)
├── manifest.json                   ✅
├── resolved_config.json            ✅ (无本机绝对路径)
├── input_manifest_hashes.json      ✅
├── runtime.json                    ✅
├── metrics_summary.json            ✅ (含 train/val × initial/resumed)
├── metrics_by_region.csv           ✅ (32 行)
├── predictions_manifest.csv        ✅ (270 行真实数据)
├── failure_cases.csv               ✅ (仅表头，无失败)
├── reload_consistency.json         ✅ (consistent=true, max_abs_diff=0.0)
├── checkpoints/
│   ├── initial_epoch.pt            ✅
│   └── resumed_epoch.pt            ✅
└── logs/
    └── run.log                     ✅
```

### 4.1 predictions_manifest.csv 字段

每行 270 条真实记录（无占位符）：

| 字段 | 说明 |
|------|------|
| split | `train` / `val` |
| phase | `initial` / `resumed` |
| sample_id | 真实样本 ID（如 `SLP:danaLab:00022:uncover:000001`） |
| subject_id | 真实 subject_id |
| label_sha256 | label 数组的 canonical SHA-256（64 位小写 hex） |
| prediction_sha256 | prediction 数组的 canonical SHA-256 |
| label_shape | GT 形状 |
| prediction_shape | 预测形状 |
| failure_reason | `ok` / `non_finite_pressure` / `label_out_of_range` |

行数分布：

| Split | Phase | 行数 |
|-------|-------|-----|
| train | initial | 90 |
| val | initial | 45 |
| train | resumed | 90 |
| val | resumed | 45 |
| **Total** | — | **270** |

不保存大体积逐像素预测到 Git。predictions_manifest 仅记录元信息或精简的 SHA-256。

---

## 5. R02 → R03 修复对照

| 项 | R02 | R03 |
|----|-----|-----|
| predictions_manifest | 占位 `train_sample_000000` | 270 条真实记录 + 真实 SHA-256 |
| subset 配置消费 | `n_train_subjects=2, n_val_subjects=1` 硬编码 | 从 `SmokeConfig` 读取 |
| canonical hash | 仅 `arr.tobytes()` | 版本 + dtype + shape header + bytes |
| 真实行数匹配 | 运行时未做 assert | `test_manifest_row_count_matches_real_predictions` 断言 |
| Subject 选择描述 | 排序后前 N 名 | 排序 + seed=42 确定性 shuffle |
| 测试 | 94 passed | 108 passed |

---

## 6. 测试结果

### 6.1 单元测试

```
tests/test_slp8_region_dataset.py  ✅
tests/test_slp8_region_models.py   ✅
tests/test_slp8_region_smoke.py   ✅
─────────────────────────────────
108 passed in 9.52s
```

新增 / 强化测试：

- `TestCanonicalArrayHash`（6 个）：hash 长度 / hex / 稳定性 / 修改改变 hash / 头部包含版本 / shape 防止碰撞
- `TestPredictionRecord`（4 个）：必填字段 / 真实写入无占位 / 行数匹配 / 预测变化改 hash
- `TestSubsetConfigFlowsToManifest`（2 个）：解析字段与 config 一致 / 运行 `run_smoke_test` 验证 record 数随 config 变化

### 6.2 回归测试

```
tests/test_slp8_training_table_freeze.py    ✅ 259 passed, 2 skipped
tests/test_slp8_non_learning_region_baseline.py  ✅
tests/test_slp_pressure_infrastructure.py   ✅
tests/test_neural_checkpoint.py             ✅
tests/test_experiment_contracts.py          ✅
tests/test_experiment_runner.py             ✅
tests/test_experiment_artifacts.py          ✅
─────────────────────────────────────────────────────────
371 passed, 2 skipped
```

### 6.3 git diff --check

无 whitespace 错误。

---

## 7. 输入合同验证

| 合同 | 状态 |
|------|------|
| Dataset: SLP_8Region_Pressure_VAL_v1.1 | ✅ |
| Training-table: slp8_training_tables_v0.1 | ✅ |
| Pressure shape: [192, 84], float64 | ✅ |
| Region label: [192, 84], int64 | ✅ |
| Classes: 0-8 (9 classes) | ✅ |
| TRAIN: 3645 samples / 81 subjects | ✅ (B01) |
| VAL: 450 samples / 10 subjects | ✅ (B01) |
| TEST: 495 samples / 11 subjects (NOT LOADED) | ✅ (load_test=False) |
| Provenance: V221_CORRECTED_SUPPORT_AUTO_ACCEPTED | ✅ |
| raw_semantics: raw_pmarray_response | ✅ (NOT kPa) |
| source_review_status: NOT_REVIEWED | ✅ |
| danaLab only, uncover only | ✅ |
| raw_passthrough_with_minmax_reference | ✅ (raw passthrough) |
| TRAIN-only normalization | ✅ (fit_split=train) |
| Smoke subset: 排序 + seed=42 确定性 shuffle | ✅ |
| Subject overlap = 0 | ✅ |

---

## 8. 模型架构

**Slp8TinyFcn** — 最小全卷积网络

```
Input [N, 1, 192, 84]
→ Conv2d(1, 8, 3, padding=1) + ReLU
→ Conv2d(8, 16, 3, padding=1) + ReLU
→ Conv2d(16, 9, 1)
→ logits [N, 9, 192, 84]
```

---

## 9. 训练合同

| 合同 | 值 |
|------|-----|
| device | cpu |
| seed | 42 |
| batch_size | 4 |
| initial_epochs | 1 |
| resume_epochs | 1 |
| optimizer | AdamW |
| lr | 0.001 |
| weight_decay | 0.0001 |
| loss | CrossEntropyLoss（unweighted） |
| class weights | 不使用 |

---

## 10. 已验证

1. 配置合同嵌套结构和 fail-closed 验证
2. Real config file 集成测试
3. Raw passthrough normalization 语义
4. Slp8TinyFcn 模型 forward/backward
5. Checkpoint save/load (weights_only=True)
6. Resume 后参数改变
7. **Reload 一致性实际比较**（非硬编码）
8. **predictions_manifest 真实数据**（无占位符）
9. **canonical array hash 规则**（version + dtype + shape）
10. **subset config 真正传入** `run_smoke_test`
11. **270 行真实记录 + 64 位小写 hex** 程序化检查通过
12. 14 个产物文件全部存在
13. 真实 CPU Smoke 端到端运行通过
14. R02 输出目录未被覆盖

## 11. 合理推断

- 模型在更多 epoch 上可继续降低 loss
- 不同受试者子集产生不同指标

## 12. 尚未验证

- TEST 评估（明确不加载）
- GPU 性能
- Mini/Full 训练

## 13. 限制

- Smoke 不做模型排名或超过 B02
- 1 epoch 训练无收敛保证
- 仅 2 个 TRAIN subject + 1 个 VAL subject
- 精度符合"未充分训练"预期

## 14. 禁止结论

> 1. Smoke 指标代表 TEST 性能
> 2. 超过 B02 基线
> 3. 适用于产品决策
> 4. GT 是人类像素级标注
> 5. 压力值代表 kPa
> 6. 适用于 cover1/cover2
> 7. 适用于 danaLab 之外

---

## 15. 下一 Gate

| Gate | 前置 | 状态 |
|------|------|------|
| S2-G03 | B03 Smoke DONE | ✅ 通过（R03） |
| S2-G04 | B03 Reviewer ACCEPT | ⏳ 待 Codex Reviewer |
| S2-G05 | SLP Mini Run | 可选 |
| S2-G06 | SLP Full Run | 可选 |

---

## 16. Git 信息

| 字段 | 值 |
|------|-----|
| Branch | `codex/task-slp-b03-pm-only-region-smoke-v0.1` |
| Base | `origin/main` |
| B02 Merge | `ccbd539` ✅ |
| R01 commit | `6219411` |
| R02 commit | `885b04a` |
| R03 commit | 见 HEAD |

---

**Report 版本:** v0.1-R03
**生成时间:** 2026-08-27
**维护者:** Mavis (MiniMax Code)
