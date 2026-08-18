# P5/R5 — PoPu 固定姿态受试者隔离 Baseline 报告 v0.1

## 1. 阶段目标与完成判定

**状态：COMPLETE — FIRST_ROUND_BASELINE（在 P4a 特征表上完成受试者隔离的五分类**首轮** Baseline；候选尚未正式冻结；未训练任何区域模型，未接入其它数据集）。**

本阶段回答「在 PoPu 固定姿态数据上，只靠压力矩阵衍生的 71 个无标签特征，能否稳定区分 `empty / supine / prone / left / right`，且对新受试者成立」。评价协议（受试者隔离切分 + GroupKFold 选型 + held-out 一次性测试）先冻结到版本化配置，再执行。

结论措辞仅为「**PoPu 上的受试者隔离公开数据候选结果**」，不构成产品能力、自研硬件或闭环效果的任何声明。

## 2. 实际执行

```bash
uv run pytest -q
uv run python scripts/baseline_popu.py
```

自动测试结果：`52 passed`（既有 31 + P4a 9 + 本阶段 12）。

输入：P4a 特征表 [popu_features_p4a_v0.1.csv](../../data/processed/popu/popu_features_p4a_v0.1.csv)（51,000 行 × 88 列，71 个特征列）；冻结协议 [popu_baseline_p5_v0.1.json](../../configs/experiments/popu_baseline_p5_v0.1.json)。

## 3. 冻结的评价协议

| 项 | 冻结值 |
|---|---|
| group | `subject_id`（唯一分组键） |
| held-out test | `{5,10,15,20,25,30,35,40,45,50,55,60}` 共 12 人（20%，每 5 个取 1） |
| 开发集 | 其余 48 人；`GroupKFold(n_splits=5, 不 shuffle)`，受试者整体只落一个 fold |
| 预处理 | `SimpleImputer(median) + StandardScaler` 全封装进 Pipeline，仅各训练折拟合 |
| 特征 | 仅 P4a summary 的 71 个 `feature_columns`；标签/元数据列全部排除 |
| 选型 | 开发集 OOF `macro_f1` 最高；`dummy` 永不入选；平局取 `balanced_accuracy` |
| 测试 | P5 v0.1 实际口径：held-out test 在开发集选型后对**每个候选模型各评估一次**；模型选择未读取任何 test 分数。P5.1 起改用 repeated subject-grouped CV 复核候选，不再声称存在一个未查看的 PoPu test；真正外部确认留给 SLP/PressurePose/PMD 适用任务与未来自研同步数据 |

候选：`dummy(stratified)` 下限、`logreg(multinomial)`、`rf(n=200)`、`knn(k=5)`。

## 4. 输入与产物

产物（`baseline_version=v0.1`）：

- [逐样本预测表](../../data/processed/popu/popu_baseline_p5_predictions_v0.1.csv)（404,240 行 = 2 cohort × 4 模型 × 全部 snapshot；含 sample_id/subject_id/y_true/y_pred/confidence/split/model_version）
- [模型比较](../../outputs/metrics/popu_baseline_p5_model_comparison_v0.1.csv)
- [逐类别指标](../../outputs/metrics/popu_baseline_p5_per_class_v0.1.csv)
- [逐受试者指标](../../outputs/metrics/popu_baseline_p5_per_subject_v0.1.csv)
- [混淆矩阵长表](../../outputs/metrics/popu_baseline_p5_confusion_v0.1.csv)
- [混淆矩阵图](../../outputs/figures/popu_baseline_p5_confusion_matrices_v0.1.png)（primary × 4 模型 × dev/test）
- [汇总 JSON](../../outputs/reports/popu_baseline_p5_summary_v0.1.json)

## 5. 实际结果

### 5.1 主结果：primary cohort（固定姿态 + ACCEPT，50,060 snapshot）

| 模型 | dev macro-F1 (std) | dev bal-acc | dev acc | **test macro-F1** | **test bal-acc** | **test acc** |
|---|---:|---:|---:|---:|---:|---:|
| dummy (下限) | 0.2008 (0.004) | 0.2008 | 0.2472 | 0.2019 | 0.2018 | 0.2498 |
| **logreg** | **0.9260** (0.023) | 0.9238 | 0.9113 | **0.9466** | **0.9466** | **0.9339** |
| rf | 0.9180 (0.023) | 0.9184 | 0.8992 | 0.9438 | 0.9442 | 0.9308 |
| knn | 0.8674 (0.026) | 0.8676 | 0.8364 | 0.8828 | 0.8838 | 0.8561 |

**当前首轮领先候选 = `logreg@multinomial`**（开发集 OOF macro-F1 最高 0.9260；held-out 上也最高 0.9466，选型与测试一致，无过拟合迹象）。注意：P5 v0.1 对每个候选模型都各评估了一次 held-out test；P5.1 起不再沿用“仅对最终候选使用一次 test”的严格口径，改用 repeated subject-grouped CV 复核候选（且不声称存在未查看的 PoPu test）。因此 logreg 只称**首轮领先候选，未正式冻结**。

逐类别（logreg，primary test）：`empty` P/R/F1 = 1.0/1.0/1.0，`supine` 0.9458，`prone` 0.9190，`left` 0.9294，`right` 0.9389。

### 5.2 敏感性分析：combined cohort（ACCEPT + WARN，51,000 snapshot）

与主结果独立输出、不混写。combined 多出 22 个单格点空床帧（NaN 主轴，`principal_axis_degrees/anisotropy` 两列共 44 个 NaN），由折内 median imputer 处理。

| 模型 | dev macro-F1 | test macro-F1 | test acc |
|---|---:|---:|---:|
| dummy | 0.2039 | 0.2021 | 0.2498 |
| **logreg** | **0.9341** | **0.9460** | 0.9342 |
| rf | 0.9159 | 0.9362 | 0.9217 |
| knn | 0.8610 | 0.8852 | 0.8603 |

结论与主结果一致（logreg 最优，test macro-F1 0.9460）；加入 WARN 帧未改变候选排序，也未被污染。

### 5.3 关键口径：NaN 分布

- primary 主口径 **0 个 NaN 行**、0 个 NaN 格点——ACCEPT 帧主轴均有限。
- 22 个 NaN 行全部为 `empty` 且 `quality_status=WARN`（P4a 所述单格点空床），只进入 combined 敏感性分析；由 Pipeline 内 imputer 在各训练折拟合后填补，未在全表上预填。

## 6. 失败模式

1. **空床 vs 卧姿近乎可分**：`empty` 在 primary test 上 F1=1.0（100/100），说明「在床/离床」类信号极强。
2. **四类卧姿间对称混淆是主要错误源**：主导非对角为 `supine↔prone`、`left↔right`、`prone↔left`、`supine↔left`（logreg primary test：supine→prone 73、left→right 47、right→left 104、prone→left 79）。符合左右侧卧、仰俯卧在压力分布上的天然对称性。
3. **强受试者差异**：开发集 17/48 人 accuracy<0.9（最差 subject 17 = 67.5%）；held-out test 中 2/12 人 <0.9（subject 15 = 68.2%，subject 30 = 89.8%）。subject 15 是显著难例，其压力图偏离群体典型，说明候选的泛化受个体体型/接触模式影响，需在更大受试者范围复核。

## 7. 已验证、合理推断、尚未验证

### 已验证

- 受试者隔离协议按冻结配置执行：12 个 held-out 与 48 个开发受试者完全隔离；`GroupKFold` 保证同一受试者只落一个 fold（单元测试 `test_split_subjects_*` 与代码路径）。
- 只使用 71 个特征列，且与 P4a summary `feature_columns` 逐一核对相等；`feature_columns()` 从构造上排除 `subject_id/posture/variation/路径/snapshot/cohort` 等（单元测试）。
- 填补/标准化封装在 Pipeline 内、仅各训练折拟合（单元测试 `test_pipeline_imputer_fits_only_on_training_fold` 直接断言 imputer 统计量==训练折中位数）。
- held-out test 对每个候选各评估一次（v0.1 口径，未重复抽样），模型选择完全基于开发集（`select_best_model` 只读 `split=dev` 行，单元测试）。
- 52 条测试全绿；产物齐备且行数核对一致（预测表 404,240 行 = 2×4×全量 snapshot）。

### 合理推断

- 在 PoPu 内，logreg 的受试者隔离 macro-F1 ≈ 0.95（test），是当前低成本候选的最优；RF 略低但相近。
- 空床/卧姿的强可分性与四卧姿的对称混淆符合物理直觉。

### 尚未验证

- 非产品能力、非自研硬件、非闭环效果；不证明整夜稳定性、舒适性或安全。
- 未在 TIP / SLP / PressurePose / PMD 上验证；未做降密度、坏点、噪声鲁棒性（P6/P7）。
- 未做 `UNKNOWN/REJECT` 阈值与置信度校准（P6）；未做身体区域监督（继续 HOLD）。
- subject 15 等难例的成因（体型/接触异常）未做个案解剖。

## 8. 对后续阶段的决策

- P5/R5 首轮基线完成；`logreg@multinomial`（Pipeline：imputer median → scaler → LR）为**当前首轮领先候选**，尚未正式冻结。
- 先进入 P5.1：横向比较框架修正（改用 repeated subject-grouped CV 排名，不再声称存在未查看的 PoPu test）、模块化增强、候选复核。P5.1 通过前不正式冻结模型，也不设置 `UNKNOWN/REJECT` 阈值。
- 不训练区域模型；PoPu 区域监督继续 HOLD。
- P5.1 复核通过后，再进入 P6：以验证受试者 OOF 置信度选择 `UNKNOWN/REJECT` 阈值，并分析高置信错误。

## 9. 下一阶段最小输入与放行条件

P5.1 最小输入：本阶段的逐样本预测表（含置信度）+ 当前首轮领先候选 `logreg`。放行条件：以 repeated subject-grouped CV 口径复核候选排序（逐 snapshot 与逐记录/逐受试者稳定性均报告）；P5.1 通过后才允许冻结模型并进入 P6。

P6 最小输入（在 P5.1 之后）：P5.1 冻结的候选 + 逐样本预测表（含置信度）。放行条件：阈值仅由验证（开发集 OOF）受试者选择；`UNKNOWN/REJECT` 对空床与卧姿、高低置信错误的取舍被明确记录。

## 10. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-18 | 首跑 P5/R5：冻结受试者隔离协议，评估 dummy/logreg/rf/knn；候选=logreg（primary test macro-F1 0.9466）；`52 passed`。 |
