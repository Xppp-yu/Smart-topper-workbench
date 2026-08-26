# Stage Report: S2-B03 — SLP8 PM-only Region Segmentation Smoke

**TASK-ID:** `TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1`
**Stage:** S2-B03
**日期:** 2026-08-27 (R02)
**状态:** DONE — 通过 R02 Reviewer ITERATE 修复并完成真实 CPU Smoke
**EXP-ID:** `EXP-SLP-B03-PM-REGION-SMOKE-20260827-R02`

---

## 摘要

B03 阶段实现并验证了 SLP8 压力图区域分割的最小化 Smoke 测试，覆盖从 B01 冻结表到 PyTorch 像素级分割、训练、checkpoint、resume、reload 一致性、指标计算和审计产物的完整链路。

R02 修正了以下 Reviewer 提出的问题：
1. 配置合同改为嵌套结构（`cfg["training"]["seed"]` 等）
2. Normalization 修正为 raw passthrough（不再 Min-Max 缩放）
3. Checkpoint reload 一致性实际比较 logits（不再硬编码 True）
4. 补全 metrics_summary / metrics_by_region / predictions_manifest / failure_cases / logs 等产物

**重要：** Smoke 只验证 pipeline 可运行，不与 B02 排名，不形成 TEST 精度结论。

---

## 1. R02 修复内容

### 1.1 配置合同（嵌套结构）

| 路径 | 类型 | 用途 |
|------|------|------|
| `cfg["task_id"]` | str | TASK-ID 校验 |
| `cfg["provenance"]` | str | `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED` |
| `cfg["raw_semantics"]` | str | `raw_pmarray_response` |
| `cfg["model"]["n_classes"]` | int | 必须为 9 |
| `cfg["model"]["input_shape"]` | list | 必须为 `[192, 84]` |
| `cfg["training"]["seed"]` | int | 种子 |
| `cfg["training"]["device"]` | str | `cpu` 或 `cuda` |
| `cfg["training"]["batch_size"]` | int | 正整数 |
| `cfg["training"]["lr"]` | float | 学习率 |
| `cfg["training"]["weight_decay"]` | float | 权重衰减 |
| `cfg["training"]["epochs"]["initial"]` | int | ≥ 1 |
| `cfg["training"]["epochs"]["resume"]` | int | ≥ 1 |
| `cfg["dataset"]["smoke_subset"]["n_train_subjects"]` | int | 必须为 2 |
| `cfg["dataset"]["smoke_subset"]["n_val_subjects"]` | int | 必须为 1 |
| `cfg["dataset"]["smoke_subset"]["seed"]` | int | 种子 |
| `cfg["dataset"]["normalization"]["method"]` | str | `raw_passthrough_with_minmax_reference` |
| `cfg["dataset"]["normalization"]["fit_split"]` | str | `train` |

`SmokeConfig` 现在所有字段都是必填，无默认参数掩盖配置缺失。

### 1.2 Normalization 语义（Raw Passthrough）

B01 的方法是 `raw_passthrough_with_minmax_reference`，其真实含义是：

* **原始 pressure 数值保持不变**
* **只做 float64 → float32 dtype 转换**
* **增加 channel 维度：(192, 84) → (1, 192, 84)**
* **global_min / global_max / global_mean / global_std 仅作为 TRAIN-only reference 记录**
* **不执行 Min-Max 缩放**
* **raw semantics 仍为 `raw_pmarray_response`，NOT kPa**

修复后 `NormalizationStats.apply` 仅做 dtype 和维度转换，输出值与原 float32 输入等值。

### 1.3 Reload Consistency（实际比较）

修复前：`reload_consistent = True`（硬编码）

修复后：
1. `resumed_model.eval()` 设置 eval 模式
2. 固定取得 `reference_batch`（来自 train_dataloader）
3. 保存 `resumed_model` 对该 batch 的 logits
4. 从 `resumed_epoch.pt` 创建全新模型 `fresh_model` 并加载
5. `fresh_model.eval()` 设置 eval 模式
6. 对相同 batch 推理
7. 使用 `torch.allclose(resumed_logits, fresh_logits, rtol=1e-5, atol=1e-6)` 比较
8. 不一致时加入 `verification_failures`，写 `FAILED.json`，禁止 `DONE.json`

测试 `test_runner_does_not_hardcode_reload_true` 静态扫描源码，禁止硬编码 `reload_consistent = True`。

### 1.4 产物补全

| 产物 | 路径 | 说明 |
|------|------|------|
| `status.json` | 顶层 | 总体状态（DONE/FAILED） |
| `manifest.json` | 顶层 | 运行清单（含 dataset_manifest + config） |
| `resolved_config.json` | 顶层 | 解析后配置（路径字段 redact） |
| `input_manifest_hashes.json` | 顶层 | 输入文件 SHA-256 |
| `runtime.json` | 顶层 | 运行时信息（平台、Python、wall_clock） |
| `metrics_summary.json` | 顶层 | 含 train/val × initial/resumed 完整指标 |
| `metrics_by_region.csv` | 顶层 | 每个 (split, phase, region) 的指标 |
| `predictions_manifest.csv` | 顶层 | 每条预测的元信息（不含大体积像素数据） |
| `failure_cases.csv` | 顶层 | 失败案例（无失败时仅表头） |
| `reload_consistency.json` | 顶层 | 实际比较结果 + checkpoint SHA |
| `logs/run.log` | logs/ | 运行时日志 |
| `checkpoints/initial_epoch.pt` | checkpoints/ | 初始训练检查点 |
| `checkpoints/resumed_epoch.pt` | checkpoints/ | 恢复后检查点 |
| `DONE.json` 或 `FAILED.json` | 顶层 | 最终状态（且仅一份） |

**predictions_manifest.csv** 字段：

| 字段 | 说明 |
|------|------|
| split | `train` / `val` |
| phase | `initial` / `resumed` |
| sample_id | 样本 ID |
| subject_id | 受试者 ID |
| label_sha256 | GT 像素 SHA-256 |
| prediction_sha256 | 预测像素 SHA-256 |
| label_shape | GT 形状 |
| prediction_shape | 预测形状 |
| failure_reason | `ok` / `non_finite_pressure` / `label_out_of_range` |

**重要：** 不保存大体积逐像素预测到 Git。predictions_manifest 仅记录元信息或精简的 SHA-256。

---

## 2. 真实 CPU Smoke 结果（R02）

### 2.1 运行命令

```bash
.venv\Scripts\python.exe scripts/run_slp8_region_smoke.py \
  --config configs/experiments/slp8_pm_region_smoke_v0.1.json \
  --output-dir outputs/experiments/EXP-SLP-B03-PM-REGION-SMOKE-20260827-R02 \
  --b01-freeze-dir <B01_FREEZE_DIR> \
  --dataset-root <SLP8_DATASET_ROOT> \
  --device cpu
```

### 2.2 运行结果

| 指标 | 值 |
|------|-----|
| **EXP-ID** | `EXP-SLP-B03-PM-REGION-SMOKE-20260827-R02` |
| **状态** | DONE |
| **运行时间** | 9.01 秒（wall clock） |
| **TRAIN 受试者** | `00022`, `00072`（前 2 名按 ID 排序，seed=42） |
| **TRAIN 样本数** | 90（2 subjects × 45 frames） |
| **VAL 受试者** | `00005`（前 1 名按 ID 排序，seed=42） |
| **VAL 样本数** | 45（1 subject × 45 frames） |
| **TEST 样本数** | 0（明确不加载） |
| **TRAIN/VAL subject overlap** | 0（已验证） |
| **归一化 stats SHA-256** | `0b1ef18b4769f8b1b47d077cfc4c06c8310c8fff5877a6e44afcd0df2f466c59` |
| **freeze_manifest SHA-256** | 实际生成（见 `input_manifest_hashes.json`） |

### 2.3 训练损失

| Phase | TRAIN Loss | VAL Loss |
|-------|-----------|----------|
| initial | 2.7843 | 2.4951 |
| resumed | 2.2958 | 2.2199 |

### 2.4 指标（initial phase）

| 指标 | TRAIN | VAL |
|------|-------|-----|
| fixed foreground macro IoU | 0.0303 | 0.0300 |
| fixed foreground macro Dice | 0.0574 | 0.0568 |
| pixel accuracy | 0.6592 | 0.6686 |
| background IoU | 0.0 | 0.0 |
| n_classes_present_in_pred | 8 | 8 |
| n_classes_present_in_gt | 8 | 8 |

**重要说明：**

- TRAIN/VAL 精度相近说明模型未过拟合 smoke 子集
- 精度较低符合预期（1 epoch 训练、最小模型、像素级分割从随机初始化开始）
- 仅为 pipeline 验证指标，不形成任何排名或结论

### 2.5 Checkpoint SHA-256

| Checkpoint | SHA-256 |
|-----------|---------|
| `initial_epoch.pt` | `15de19acc48655370da4b86eb17448700aa138277967ce5ffec0881462b08ec8` |
| `resumed_epoch.pt` | `0d9d2ae1979573981779c2970a5ea9ef813319530d9ec16b2e6f56335bd44fbd` |

### 2.6 参数变化证据

| 阶段 | total_diff | 参数是否改变 |
|------|-----------|-----------|
| initial 训练后 | 0.6257 | ✅ |
| resume 训练后 | 0.4482 | ✅ |

### 2.7 Reload 一致性

| 指标 | 值 |
|------|-----|
| `consistent` | `true` |
| `max_abs_diff` | `0.0` |
| 比较方法 | `torch.allclose(rtol=1e-5, atol=1e-6)` |
| 实际比较对象 | resumed_model vs fresh_model on same reference_batch |

---

## 3. 测试结果

### 3.1 单元测试

```
tests/test_slp8_region_dataset.py  ✅ 36 tests
tests/test_slp8_region_models.py   ✅ 27 tests
tests/test_slp8_region_smoke.py   ✅ 31 tests
─────────────────────────────────
94 passed in 12.58s
```

### 3.2 回归测试

```
tests/test_slp8_training_table_freeze.py    ✅ 259 passed, 2 skipped
tests/test_slp8_non_learning_region_baseline.py  ✅
tests/test_slp_pressure_infrastructure.py   ✅
tests/test_neural_checkpoint.py             ✅
tests/test_experiment_contracts.py          ✅
tests/test_experiment_runner.py             ✅
tests/test_experiment_artifacts.py          ✅
─────────────────────────────────────────────────────────
371 passed, 2 skipped (B01/B02 + 基础设施)
```

### 3.3 git diff --check

无 whitespace 错误。

---

## 4. 输入合同验证

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
| Smoke subset: TRAIN 前 2 subjects, VAL 前 1 subject | ✅ |
| seed=42 | ✅ |
| Subject overlap = 0 | ✅ |

---

## 5. 模型架构

**Slp8TinyFcn** — 最小全卷积网络

```
Input [N, 1, 192, 84]
→ Conv2d(1, 8, 3, padding=1) + ReLU
→ Conv2d(8, 16, 3, padding=1) + ReLU
→ Conv2d(16, 9, 1)
→ logits [N, 9, 192, 84]
```

| 要求 | 状态 |
|------|------|
| 输出 [N, 9, 192, 84] | ✅ |
| 无 pooling | ✅ |
| 无 pretrained weights | ✅ |
| 无外部下载 | ✅ |
| fail-closed shape/dtype/finite 检查 | ✅ |

---

## 6. 训练合同

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

## 7. 已验证

1. 配置合同嵌套结构和 fail-closed 验证（9 个测试）
2. Real config file 集成测试
3. Raw passthrough normalization 语义（8 个测试）
4. Slp8TinyFcn 模型 forward/backward
5. Checkpoint save/load (weights_only=True)
6. Resume 后参数改变
7. **Reload 一致性实际比较**（非硬编码）
8. Train/Val metrics 计算
9. 全部 14 个产物文件存在
10. 真实 CPU Smoke 端到端运行

## 8. 合理推断

- 模型在更多 epoch 上可继续降低 loss（趋势已可见）
- 不同受试者子集会得到不同指标

## 9. 尚未验证

- TEST 评估（明确不加载）
- GPU 性能
- Mini/Full 训练

## 10. 限制

- Smoke 不做模型排名或超过 B02
- 1 epoch 训练无收敛保证
- 仅 2 个 TRAIN subject + 1 个 VAL subject
- 背景 IoU 接近 0（pixel accuracy 偏高但前景区分不足）符合未训练预期

## 11. 禁止结论

> 1. Smoke 指标代表 TEST 性能
> 2. 超过 B02 基线
> 3. 适用于产品决策
> 4. GT 是人类像素级标注
> 5. 压力值代表 kPa
> 6. 适用于 cover1/cover2

---

## 12. 下一 Gate

| Gate | 前置 | 状态 |
|------|------|------|
| S2-G03 | B03 Smoke DONE | ✅ 通过 |
| S2-G04 | B03 Reviewer ACCEPT | 待 Codex Reviewer |
| S2-G05 | SLP Mini Run | 可选 |
| S2-G06 | SLP Full Run | 可选 |

---

## 13. Git 信息

| 字段 | 值 |
|------|-----|
| Branch | `codex/task-slp-b03-pm-only-region-smoke-v0.1` |
| Base | `origin/main` |
| B02 Merge | `ccbd539` ✅ |
| R02 Commit | 待提交 |

---

**Report 版本:** v0.1-R02
**生成时间:** 2026-08-27
**维护者:** Mavis (MiniMax Code)
