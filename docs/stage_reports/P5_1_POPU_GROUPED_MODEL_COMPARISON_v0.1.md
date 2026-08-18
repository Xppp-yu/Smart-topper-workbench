# P5.1/R5.1 — PoPu 重复受试者分组横向比较与候选冻结报告 v0.1

## 1. 阶段目标与完成判定

**状态：COMPLETE — CANDIDATE_FROZEN（在 P4a 特征表上以 repeated subject-grouped CV 完成 7 候选横向比较、特征消融，并冻结 `calibrated_linear_svm` 为 PoPu research candidate）。**

本阶段回答「P5 v0.1 的首轮领先候选 `logreg` 在更严格的受试者隔离口径（重复分组交叉验证）下是否成立；7 个候选按既定选择阶梯谁胜出；哪些特征组贡献最大；能否把胜者冻结成可独立加载的研究候选」。评价协议、选择规则、特征消融、候选冻结全部先冻结到版本化配置 [popu_model_comparison_p5_1_v0.1.json](../../configs/experiments/popu_model_comparison_p5_1_v0.1.json)，再在真实全量数据上执行。

结论措辞仅为「**PoPu 研究候选（research candidate）**」：公开数据上的研究阶段结果，不是产品模型、不是外部验证模型、不构成自研硬件或闭环效果的任何声明。

## 2. 实际执行

```bash
uv run pytest -q
uv run python -u scripts/model_comparison_popu.py \
  --config configs/experiments/popu_model_comparison_p5_1_v0.1.json
```

自动测试结果：`115 passed`（P5.1-A 框架阶段 31 + 本次新增 6 个 `FrozenClassOrderClassifier` 测试 + 既有基线，共 115）。全量真实运行退出码 0，约 233 次 estimator fits（每候选 5 折 × 3 repeat = 15 次 fit；另含 CalibratedLinearSVM 内部校准 cv）。每候选单折 fit/inference 时间（冒烟计时）：`svm 2.98s / 0.022s`、`rf 2.86s / 0.066s`、`knn 0.20s / 2.56s`、`logreg 1.57s / 0.009s`、`centroid 0.19s`、`et 1.02s`、`dummy 0.13s`。

运行中修复的两个集成缺陷（均先写失败测试再改）：
1. 消融块局部变量 `groups` 遮蔽了外层 subject groups，导致 `evaluate_grouped_oof` 收到 3 元素 groups（Int指数越界）；修复后该函数对所有调用者强制校验 `groups`/`sample_ids` 长度 == `x` 长度（`test_oof_evaluation_rejects_groups_misaligned_with_samples`）。
2. sklearn 在 fit 时把 `classes_` 排序（`['empty','left','prone','right','supine']`），与冻结标签顺序 `['empty','supine','prone','left','right']` 不一致，冻结模型冒烟 `classes_order_matches=false`；新增 `FrozenClassOrderClassifier` 适配器使冻结工件 `classes_` 精确等于冻结顺序，`predict_proba` 列按冻结顺序输出（6 个新测试）。

## 3. 冻结的评价协议

| 项 | 冻结值 |
|---|---|
| 评价方式 | **repeated subject-grouped CV**：5 折 × 3 repeats，seeds = `[11, 22, 33]`（一个 seed 生成一套受试者分组折，正好 3 repeats），group = `subject_id`，所有候选共享同一套折 |
| 预处理 | `SimpleImputer(median)`/`StandardScaler` 全封装进各候选 Pipeline，仅各训练折拟合，绝不在全表上预填/预缩放 |
| 指标口径 | 每个 repeat 独立计算，再报 mean/std；绝不把 3 个 repeat 合并成 150,180 行算单一分数 |
| 排名主口径 | **record-level**：每个 JSON 的 10 个 snapshot 概率取平均 → 每记录一个预测（按 `repeat + record_id` 分组，三次 repeat 的同一记录 snapshot 绝不混合）；primary cohort（固定姿态 + ACCEPT）为主校准 |
| 选择阶梯 | 主准则 `record_macro_f1_mean`；`dummy` 排除；margin 0.005 内视为平局；tie-break 1 `record_balanced_acc_mean`；tie-break 2 最差受试者 macro-F1；复杂度仅为最后 tie-break |
| 敏感性 | combined（ACCEPT+WARN）仅作敏感性分析，不用于选择 |
| 特征消融 | Round 1 全部 71 特征 × 7 候选；Round 2 仅 top-2 × {intensity_only(14), mask_geometry_only(21), grid_zones_only(36), intensity_geometry(35), all(71)}，`all` 复用 Round 1，不做笛卡尔积 |
| 候选 | dummy(stratified) 下限、logistic_regression(multinomial)、centroid(template)、calibrated_linear_svm、random_forest(n=200)、extra_trees(n=200)、knn(k=5) |
| 范围边界 | 只读 P4a 特征表与分组 OOF；绝不查阅 P5 v0.1 历史 held-out test；不声称存在未查看的 PoPu test |

## 4. 输入与产物

输入：P4a 特征表 [popu_features_p4a_v0.1.csv](../../data/processed/popu/popu_features_p4a_v0.1.csv)（primary cohort 50,060 snapshot / 60 subjects / 5,006 records，0 NaN 行）；冻结协议 [popu_model_comparison_p5_1_v0.1.json](../../configs/experiments/popu_model_comparison_p5_1_v0.1.json)。

产物（统一前缀 `popu_model_comparison_p5_1_v0.1`）：

- [OOF 逐 snapshot 预测表](../../data/processed/popu/popu_model_comparison_p5_1_oof_predictions_v0.1.csv)（1,051,260 行 = 7 模型 × 3 repeat × 50,060；含 `repeat/fold_id/sample_id/group_id/y_true/y_pred/confidence` 与每类概率列 `proba__<label>`、`record_id`）
- [record-level 预测表](../../outputs/metrics/popu_model_comparison_p5_1_record_level_v0.1.csv)（105,126 行 = 7 × 3 × 5,006）
- [模型比较汇总](../../outputs/metrics/popu_model_comparison_p5_1_summary_v0.1.csv)
- [逐类别指标](../../outputs/metrics/popu_model_comparison_p5_1_per_class_v0.1.csv)（7 × 5）
- [逐受试者指标](../../outputs/metrics/popu_model_comparison_p5_1_per_subject_v0.1.csv)（7 × 60，跨 repeat 聚合）
- [fold/repeat 指标](../../outputs/metrics/popu_model_comparison_p5_1_fold_repeat_v0.1.csv)（7 × 3 × 5，含 seed）
- [特征消融](../../outputs/metrics/popu_model_comparison_p5_1_feature_ablation_v0.1.csv)（2 × 5）
- [混淆矩阵长表](../../outputs/metrics/popu_model_comparison_p5_1_confusion_v0.1.csv) + [混淆矩阵图](../../outputs/figures/popu_model_comparison_p5_1_confusion_matrices_v0.1.png)
- [稳定性图](../../outputs/figures/popu_model_comparison_p5_1_stability_v0.1.png)、[最差受试者图](../../outputs/figures/popu_model_comparison_p5_1_worst_subject_v0.1.png)
- [汇总 JSON](../../outputs/reports/popu_model_comparison_p5_1_summary_v0.1.json)（机器可读，含完整选择/消融/冻结记录）
- [冻结候选](../../outputs/models/popu_research_candidate_p5_1_v0.1.joblib)（16,097 B）+ [元数据](../../outputs/models/popu_research_candidate_p5_1_v0.1.metadata.json)（特征 schema、标签顺序、模型配置、训练数据版本、随机种子、已知限制、smoke 结果）

## 5. 实际结果

### 5.1 主结果：record-level（primary cohort，mean ± std across 3 repeats）

| 模型 | record macro-F1 | record bal-acc | record acc | snapshot macro-F1 | 最差受试者 acc |
|---|---:|---:|---:|---:|---:|
| dummy (下限) | 0.2058 ± 0.0005 | 0.2084 | 0.2578 | 0.2015 | 0.2000 (subj 60) |
| centroid | 0.7851 ± 0.0004 | 0.7856 | 0.7349 | 0.7806 | 0.4086 (subj 46) |
| knn | 0.8855 ± 0.0014 | 0.8858 | 0.8588 | 0.8742 | 0.6549 (subj 60) |
| random_forest | 0.9318 ± 0.0010 | 0.9321 | 0.9161 | 0.9263 | 0.5882 (subj 38) |
| extra_trees | 0.9369 ± 0.0003 | 0.9372 | 0.9224 | 0.9315 | 0.6549 (subj 38) |
| logistic_regression | 0.9424 ± 0.0010 | 0.9418 | 0.9296 | 0.9364 | 0.6706 (subj 17) |
| **calibrated_linear_svm** | **0.9452 ± 0.0022** | **0.9429** | **0.9353** | **0.9395** | **0.6784 (subj 17)** |

选择结果：`within_margin = [logistic_regression, calibrated_linear_svm]`（0.9452 vs 0.9424，差 0.0028 < margin 0.005，视为统计平局）；tie-break 1 `record_balanced_acc_mean`（0.9429 vs 0.9418）判定 **calibrated_linear_svm 胜出**（复杂度阶梯未被使用）。结论与 P5 v0.1 方向一致（线性模型阵营最优，svm 校准后略高于 logreg），排序可解释，无过拟合迹象。

### 5.2 逐类别（svm，record-level，mean across repeats）

| 标签 | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| empty | 1.0000 | 0.9748 | **0.9872** | 53.0 |
| supine | 0.9420 | 0.9498 | 0.9459 | 1242.0 |
| prone | 0.9356 | 0.9466 | 0.9410 | 1236.0 |
| left | 0.9225 | 0.8981 | 0.9101 | 1233.0 |
| right | 0.9382 | 0.9450 | 0.9416 | 1242.0 |

`empty`（空床）几乎完美可分（F1 0.987）；`left` 是相对最弱类（recall 0.898，其它四类中唯一 F1 < 0.92）。

### 5.3 特征消融（Round 2，top-2 × 5 组，record macro-F1）

| 特征组 | n | logreg | svm |
|---|---:|---:|---:|
| intensity_only | 14 | 0.5282 | 0.5201 |
| mask_geometry_only | 21 | 0.8926 | 0.8887 |
| grid_zones_only | 36 | 0.8233 | 0.8253 |
| intensity_geometry | 35 | 0.9039 | 0.9020 |
| all | 71 | 0.9424 | 0.9452 |

结论：**几何/形状特征承载绝大多数判别力**（单用 21 列几何即达 ~0.89）；仅强度单用 ~0.52；网格分区单用 ~0.82；强度+几何（35 列）0.90，仍明显低于全量 0.945——网格分区与强度叠加是达到最优所必需。两个 top-2 模型在各组的排序完全一致。

### 5.4 混淆矩阵（svm，record-level，pooled over repeats，合计 15,018 记录）

主导非对角均为卧姿间对称混淆：`left→right 163`、`supine→prone 62`、`left→prone 128`、`prone→supine 108`、`supine→left 79`、`right→left 131`。对角 3539/3510/3322/3521（supine/prone/left/right），`empty` 仅 4 条错分（3 条入四卧姿、1 条 supine→empty 0 无）。与 P5 v0.1 的混淆结构一致。

### 5.5 最差受试者（svm，record acc mean across repeats）

`subject 17 = 0.6784`（最差），其次 `subject 15 = 0.6824`、`subject 31 = 0.7529`、`subject 43 = 0.7882`。受试者 17 与 15 是显著难例（与 P5 v0.1 中 held-out 难例 subject 15 = 68.2% 相互印证），说明个体体型/接触模式显著影响泛化；60 名受试者中仅 4 人 record acc < 0.8。

## 6. 失败模式

1. **四类卧姿对称混淆是主要错误源**：左右侧卧、仰俯卧在压力分布上的天然对称性（left↔right 双向 ~300 条）构成绝大多数非对角错误。
2. **强受试者差异**：svm 最差受试者 subject 17 acc 0.678 / 15 acc 0.682；rf 的最差受试者（subj 38）只有 0.588。跨模型最差受试者 F1 从 0.74（svm）到 0.62（rf）不等。
3. **仅强度或仅网格分区明显不足**：强度单用 ~0.52、网格单用 ~0.82；特征消融显示几何（含 mask/bbox/centroid/cop/principal）是主判别源，但必须叠加分区与强度才到最优。
4. 置信度最高的错误未被本阶段分析（留给 P6 UNKNOWN/REJECT 与高置信错误分析）。

## 7. 已验证、合理推断、尚未验证

### 已验证

- repeated subject-grouped CV 按冻结协议执行：3 个 seed 各生成一套受试者分组 5 折，所有 7 候选共享同一套折；同一受试者只落一个 fold（`generate_group_folds` 单元测试）。
- OOF 预测携带 `proba__<label>` 每类概率列，行有限且和 ≈ 1；冻结标签顺序与 `estimator.classes_` 对齐（`_align_proba` 对意外类 raise、缺失类 zero-fill）。
- 记录聚合按 `(repeat, record_id)` 分组，10 个 snapshot 概率取平均，label/group 冲突即拒绝；record-level 表每行 `n_snapshots=10`。
- 每个 repeat 单独计算指标，mean/std 跨 3 个 repeat，绝不合并成 150,180 行算单分。
- 预处理只拟合各训练折（Pipeline 内建，绝不在全表预填）；primary 0 NaN 行。
- 选择只读分组 OOF，margin/平局阶梯按配置执行；`dummy` 被排除；未读取任何历史 held-out test。
- 特征消融 `all` 组明确复用 Round 1（`reused_from_round1=true`），无重复计算。
- 冻结候选完整保存 Pipeline（imputer→scaler→CalibratedLinearSVM，外包 `FrozenClassOrderClassifier`）+ 特征 schema + 标签顺序 + 配置 + 训练数据版本 + 随机种子 + 已知限制；独立 joblib 重载后 `predict`/`predict_proba` 冒烟 OK（`classes_order_matches=true`，200 样本，proba 有限且和=1）。
- 115 条测试全绿；产物行数核对一致（OOF 1,051,260 = 7×3×50,060；record 105,126 = 7×3×5,006；per_subject 420 = 7×60；per_class 35 = 7×5；fold_repeat 105 = 7×3×5）。

### 合理推断

- 在 PoPu primary 口径下，`calibrated_linear_svm` record macro-F1 ≈ 0.945 是当前最优研究候选，与 logreg 在 0.005 margin 内统计平局、靠固定 tie-break 胜出。
- 几何特征承载主判别力、需叠加分区与强度的结论符合压力矩阵物理直觉。
- `left` 相对最弱与左右侧卧压力对称性相关，属物理可分性问题而非工程缺陷。

### 尚未验证

- 非产品能力、非自研硬件、非闭环效果；不证明整夜稳定性、舒适性或安全。
- 未在 SLP2022 / PressurePose / TIP / PMD 上验证；未做降密度、坏点、噪声鲁棒性（P6/P7）。
- 未设置 `UNKNOWN/REJECT` 阈值与置信度校准（P6）。
- 未做个案解剖 subject 17/15 难例的成因（体型/接触模式）。

## 8. 对后续阶段的决策

- P5.1 COMPLETE — CANDIDATE_FROZEN：`calibrated_linear_svm`（svm@calibrated_linear，Pipeline：imputer median → scaler → CalibratedLinearSVM，外包 FrozenClassOrderClassifier）冻结为 **PoPu research candidate** `popu_research_candidate_p5_1_v0.1`（16,097 B，smoke OK）。
- **P6 放行**：候选已冻结 → P6 `UNKNOWN/REJECT` 与错误分析 = READY_TO_IMPLEMENT；阈值仅由验证受试者选择，高置信错误需个案分析。
- P5 v0.1 的历史 held-out 数字保持历史首轮证据，不被改写、不用于 P5.1 选择。
- 不训练区域模型；PoPu 区域监督继续 HOLD。

## 9. 下一阶段最小输入与放行条件

P6 最小输入：P5.1 冻结候选 `popu_research_candidate_p5_1_v0.1.joblib` + 本阶段 record/OOF 预测表（含置信度）。放行条件：`UNKNOWN/REJECT` 阈值仅由验证（开发集 OOF）受试者选择；空床与卧姿、高低置信错误的取舍被明确记录；冻结候选经独立 `predict_proba` 冒烟验证。

## 10. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-18 | 首跑 P5.1：7 候选 repeated subject-grouped CV + 特征消融 + 冻结 `calibrated_linear_svm`（record macro-F1 0.9452）；修复消融 groups 遮蔽与冻结类序两个集成缺陷（新增测试，全量 `115 passed`）。 |
