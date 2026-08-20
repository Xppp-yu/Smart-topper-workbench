# P5.2-B/R5.2-B — PoPu 神经网络 Mini 筛选真实运行结果报告 v0.1

## 1. 结论

**状态：COMPLETE — MINI_ACCEPTED。**

P5.2-B Mini 筛选已按冻结协议在 AutoDL（RTX 4090）上真实运行并 `SUCCEEDED`，Reviewer 已独立复核并接受。三个候选模型 `matrix_mlp`、`tiny_cnn`、`small_resnet` 的可行性 Gate 均为 `proceed`，按预注册规则**三者都进入 P5.2-C Full 公平比较**。Mini 只作可行性筛选，**不形成最终排名**；**不得**据此宣布 `matrix_mlp` 或 `small_resnet` 为总体冠军——尚未与 P5.1 SVM 在相同 Full 协议下完成最终比较。

## 2. 阶段目标与完成判定

- **目标**：用真实 PoPu 数据在冻结的 Mini 协议下，对三个神经网络候选做可行性筛选，排除明显不可行的候选，供后续 Full 公平比较继续。
- **完成判定**：`state=SUCCEEDED`、`git dirty=false`、`reproducible_seed=true`、`overall_verdict=proceed`，三模型各自 `viability.verdict=proceed`；Reviewer 从 `*_best.json` 预测文件独立重算 record-level 指标并给出 `MINI_ACCEPTED`。

## 3. 运行证据

- **EXP-ID**：`EXP-P5.2-B-MINI-SCREEN-20260819-R01`；TASK-ID：`TASK-P5.2-B-MINI-SCREEN-v0.1`。
- **Git**：SHA `02611137945bb4ae99402ac03fa0fabc95438e47`，`dirty=false`。
- **scope / runner_type**：`mini` / `popu_neural_mini`。
- **设备**：`device=cuda`；GPU `NVIDIA GeForce RTX 4090`（24,564 MB，driver `595.58.03`）；CUDA 可用、`device_count=1`、PyTorch `2.8.0+cu128`、cuDNN `91002`。
- **远程环境**：AutoDL Linux，Python `3.12.3`；执行命令 `scripts/run_experiment.py --config configs/experiments/popu_neural_mini_v0.1.json`。
- **数据 Manifest**：P2 质量 manifest SHA-256 `9d3398a587b183f7e27ea68ada2eda1e5e82ebadb2ac9caf7a74b5763d3e954c`，`verified=true`，`size_bytes=1,187,628`。
- **Cohort**：516 条记录考虑，11 条排除，**505 条 ACCEPT**，**5,050 个选中 snapshot**（505 × 10）。
- **切分**：`train_subject_ids=["1","2","3","4"]`、`val_subject_ids=["5","6"]`（无 test）；`train_samples=6,760`（含左右翻转增强 3,380）、`val_samples=1,670`、`val_records=167`。
- **验证集类别分布**：`empty` 20 snapshots（2 条记录）、`supine` 400、`prone` 410、`left` 420、`right` 420。
- **种子**：`seed=42`，`reproducible_seed=true`。

## 4. 生成产物

```text
outputs/experiments/EXP-P5.2-B-MINI-SCREEN-20260819-R01/
├── DONE.json / status.json / manifest.json
├── resolved_config.json / metrics.json / train_log.json
├── checkpoints/<model>_{latest,best}.pt
├── predictions/<model>[_best].json
└── logs/run.log

outputs/evidence_archives/EXP-P5.2-B-MINI-SCREEN-20260819-R01.tar.gz
```

证据包 SHA-256：`c36bd9295b1f201b3c758180851d6f4c4dfeb9b82be91895cde7679f244c5a59`。

## 5. 核心结果

### 5.1 最佳 checkpoint 的 snapshot-level 结果（来自 `metrics.json` 的 `best_*`）

| 模型 | best_epoch | macro-F1 | balanced accuracy | 错误数 | verdict |
|---|---|---|---|---|---|
| matrix_mlp | 3 | 0.994261 | 0.994238 | 12 / 1670 | proceed |
| tiny_cnn | 4 | 0.975201 | 0.975134 | 51 / 1670 | proceed |
| small_resnet | 5 | 0.992342 | 0.992287 | 16 / 1670 | proceed |

### 5.2 Reviewer 独立重算的 record-level 结果

Reviewer 从各 `*_best.json` 预测文件独立按 `record_id` 对 10 个 snapshot 概率取平均后重算（每记录 1 票）：

| 模型 | record macro-F1 | record balanced accuracy | 错误记录数 |
|---|---|---|---|
| matrix_mlp | 0.995237 | 0.995238 | 1 / 167 |
| tiny_cnn | 0.975647 | 0.975598 | 5 / 167 |
| small_resnet | 0.995237 | 0.995238 | 1 / 167 |

`matrix_mlp` 与 `small_resnet` 的 record-level 结果并列，均只有 1 条 `left→right` 错误记录。

### 5.3 逐受试者 record-level（Reviewer 重算）

| 受试者 | matrix_mlp record macro-F1 | small_resnet record macro-F1 |
|---|---|---|
| subject 5 | 1.0 | 1.0 |
| subject 6 | 0.990471 | 0.990471 |

### 5.4 可行性 Gate

三模型 `finite / no_leakage / same_split / checkpoint_ok / resume_ok / reload_ok / resource_ok / learning_signal_ok` 八项全部通过，各自 `viability.verdict=proceed`；汇总 `overall_verdict=proceed`。未触发 `exclude` 或 `needs_fix`。

## 6. 已验证 / 合理推断 / 尚未验证

- **已验证**：三候选在冻结 Mini 协议下真实训练/评估完成，`SUCCEEDED`，checkpoint/resume/reload、固定 seed 复现、数据 Manifest SHA-256 校验、record-level Reviewer 独立重算均成立。
- **合理推断**：三候选在固定 6 个开发受试者、单 seed 下具备学习信号与可行性，未发现基础设施或协议问题，可进入 Full。
- **尚未验证**：全量跨受试者表现、多 seed/多 fold 稳定性、与 P5.1 SVM 在相同 Full 协议下的最终比较。

## 7. 决策

- 三候选 Gate 均为 `proceed`，**按预注册规则三者都进入 P5.2-C Full 公平比较**。
- Mini 只作**可行性筛选**，不形成最终排名。
- **不得**宣布 `matrix_mlp` 或 `small_resnet` 为总体冠军；二者与 `tiny_cnn` 在 Mini 上的数值差异不构成候选排序依据。
- 尚未与 P5.1 `calibrated_linear_svm` 在相同 Full 协议下完成最终比较；P5.2-C 完成并经 Reviewer 接受后才冻结 PoPu 总体候选。

## 8. 已知限制

- 仅固定 6 个开发受试者（`["1".."6"]`），不是全量受试者分布。
- 单 seed（`seed=42`），未覆盖多 seed/多 fold 波动。
- 验证集只有 2 个受试者（`["5","6"]`）。
- `empty` 类只有 2 条验证记录 / 20 snapshots，类别估计极不稳定。
- 同一记录的 10 帧高度相关，snapshot-level 指标被高相关帧放大。
- Mini 高分**不能外推**为全量跨受试者表现或产品验证；仍属公开 PoPu 数据的候选证据。

## 9. 下一步

先由 Controller/Reviewer 冻结 P5.2-C Full 公平比较协议和配置（与 P5.1 相同的受试者隔离原则、record 聚合与主指标），然后另行进行代码实现与 GPU 授权。**本任务不运行 Full。**

## 10. 不能得出的结论

- 不得得出 CNN/MLP/SVM 谁是 PoPu 总体最优候选；
- 不得由 Mini 数值形成任何候选排名或宣布冠军；
- 不得把固定 6 受试者、单 seed 的高分外推为跨受试者泛化、自研硬件、舒适性或产品验证。
