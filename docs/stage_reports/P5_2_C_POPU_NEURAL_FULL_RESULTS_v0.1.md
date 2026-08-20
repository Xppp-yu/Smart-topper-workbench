# P5.2-C/R5.2-C — PoPu 神经网络 Full 公平比较结果与 Reviewer 验收 v0.1

## 1. 结论

**状态：COMPLETE — SMALL_RESNET_ACCEPTED。**

正式 Full 实验 `EXP-P5.2-C-FULL-COMPARISON-20260820-R01` 已在 AutoDL RTX 4090 上完成 3 模型 × 3 repeats × 5 folds，共 45 个训练单元。最终状态为 `SUCCEEDED`。Reviewer 对完整证据包做了独立完整性、切分、覆盖、指标与选择规则复核，接受 `small_resnet` 作为 **PoPu 固定睡姿五分类的总体研究候选模型族**，并放行 P6 `UNKNOWN/REJECT` 与错误分析。

该决定冻结的是候选**架构族与研究路线**，不是一个可直接部署到任意设备的通用 checkpoint；也不构成外部数据集或产品验证。

## 2. 实际执行与恢复记录

- **EXP-ID**：`EXP-P5.2-C-FULL-COMPARISON-20260820-R01`。
- **训练代码 Git**：`ca9abb08a3c0bacacec07de7ef50b53c00952af6`，`dirty=false`。
- **协议**：60 个受试者、5,006 records、50,060 snapshots；3 repeats × 5 个 subject-grouped folds；三个神经网络使用同一 split manifest。
- **完成单元**：45 / 45。
- **首次聚合异常**：历史 P5.1 SVM 概率 CSV 以 6 位小数保存，少量行的概率和漂移至 `1.000002`，超过冻结的 `1e-6` 校验阈值；训练单元未丢失。
- **受限修复**：只对 SVM 的 NLL/Brier/ECE 校准诊断允许不超过 `5e-6` 的有限序列化漂移并重新归一化；不修改类别预测、历史主指标或候选排序。
- **恢复代码 Git**：`4bf589917435f9edadd82567101fae5ff324a1fd`；恢复前 45 单元已经完成，恢复只重新聚合，重训单元为 0。
- **最终状态**：`SUCCEEDED`，`DONE.json` 存在，当前无 `FAILED.json`。

## 3. 证据包

- 文件名：`EXP-P5.2-C-FULL-COMPARISON-20260820-R01-FINAL.tar.gz`
- SHA-256：`131e6fd6f66254d410114d3f25c13a2e04d6c9c5a86e294da8612ca1dafe7b5b`
- 本地复核副本：`C:\Users\23939\AppData\Local\Temp\smarttopper-autodl\EXP-P5.2-C-FULL-COMPARISON-20260820-R01-FINAL.tar.gz`
- 归档约束：证据包约 198 MB，保留在受治理的外部证据位置，不提交到常规 Git；Git 只保存验收摘要、哈希和结论边界。
- 结构：284 个文件，包括 99 JSON、92 CSV、90 checkpoint 和 3 个日志文件。

## 4. 核心 record-level 结果

| 模型 | macro-F1 mean | macro-F1 std | balanced accuracy mean | 最差受试者 macro-F1 |
|---|---:|---:|---:|---:|
| calibrated_linear_svm | 0.945168 | 0.002169 | 0.942864 | 0.740556 |
| matrix_mlp | 0.974183 | 0.001077 | 0.974814 | 0.755706 |
| tiny_cnn | 0.978906 | 0.000351 | 0.978885 | 0.820635 |
| **small_resnet** | **0.986649** | 0.002832 | **0.986636** | **0.882483** |

`small_resnet` 相对 `tiny_cnn` 的 record macro-F1 提升为 `0.007743`，高于冻结的近似平局 margin `0.005`；相对 P5.1 `calibrated_linear_svm` 提升为 `0.041481`。按冻结选择规则，独立重算的推荐仍为 `small_resnet`。

## 5. Reviewer 独立复核

- 45 个完成标记严格覆盖 3 模型 × 3 repeats × 5 folds，无缺失或重复。
- 225 个由完成标记引用的产物大小与 SHA-256 全部匹配。
- 15 个 fold 使用同一受试者分组 split manifest；outer-test 受试者集合与清单一致。
- 神经网络 OOF snapshot 共 450,540 行；record 预测共 45,054 行；每个模型、repeat 的样本与记录覆盖均恰好一次。
- 60 个受试者在每个 repeat 均完整覆盖；预测概率有限且归一化，无空值。
- Reviewer 从 record 预测独立重算 accuracy、balanced accuracy、macro-F1，与结果文件在数值精度内一致。
- 45 / 45 checkpoint 重载检查通过；峰值 CUDA 分配 39.32 MB。
- 冻结选择器独立重算仍返回 `small_resnet`。

机器可读 Reviewer 决策见：`outputs/reports/popu_neural_full_p5_2_c_reviewer_acceptance_v0.1.json`。

## 6. 已验证 / 合理推断 / 尚未验证

- **已验证**：在公开 PoPu Tactilus 固定睡姿五分类任务、受试者分组重复交叉验证口径下，Small ResNet 的 record macro-F1、balanced accuracy、最差受试者表现均为四候选最佳，并通过完整证据复核。
- **合理推断**：Small ResNet 是进入 P6 阈值与错误分析、P7 软件鲁棒性研究的最佳 PoPu 总体候选模型族。
- **尚未验证**：SLP、PressurePose 或自采数据上的迁移效果；跨传感器域泛化；身体部位识别；整夜连续过程；自研硬件、舒适性、安全性和产品效果。

## 7. 决策与下一步

1. 接受并冻结 `small_resnet` 为 PoPu 固定睡姿五分类总体研究候选模型族。
2. 保留 P5.1 `calibrated_linear_svm` 作为传统模型对照，不删除其历史产物。
3. P6 从 **HOLD** 转为 **READY**：使用 Full OOF record 概率设计 `UNKNOWN/REJECT`，阈值只从开发 OOF 证据选择，并单独检查空床、侧卧混淆、低置信与高置信错误。
4. P6 完成前不直接进入外部数据迁移，也不把当前分数解释为产品准确率。

## 8. 不能得出的结论

- 不能称为 SLP、PressurePose 或自采数据上的最优模型；
- 不能称为身体部位识别模型；
- 不能称为可直接部署的最终 checkpoint；
- 不能据公开 PoPu 结果宣称自研传感器、整夜、舒适性、安全性或闭环产品已经验证。
