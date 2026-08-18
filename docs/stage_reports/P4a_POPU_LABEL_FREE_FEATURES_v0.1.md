# P4a/R4a — PoPu 无标签逐快照特征表报告 v0.1

## 1. 阶段目标与完成判定

**状态：COMPLETE（生成可追溯、无标签泄漏的逐 snapshot 特征表；未训练任何模型）。**

本阶段把 PoPu 固定姿态记录中的每一个压力 snapshot 转成一行特征，作为下一步受试者隔离姿态 Baseline 的输入。特征只来自原始压力矩阵、P3.1 冻结的 `largest_component` 接触 Mask / Geometry，以及一个文档化的 `4×3` 网格分区；`subject_id`、`posture`、`variation`、文件名与 snapshot 标识一律不进特征列。

完成不代表得到任何姿态或身体区域结论，也不代表接触 Mask 是解剖或物理真值。

## 2. 实际执行

```bash
uv run pytest -q
uv run python scripts/features_popu.py
```

自动测试结果：`40 passed`（既有 31 条 + 本阶段新增 9 条）。

输入范围：P1 盘点与 P2 质量结果中的全部 `5,160` 条记录。`5,100` 条固定姿态记录（`ACCEPT=5,006`、`WARN=94`）逐 snapshot 展开；`60` 条 `others.json` 因缺少固定姿态标签标记为 `EXCLUDED`，其 `35,247` 个 snapshot **不展开**进特征表。

## 3. 输入与产物

输入：

- [P1 Inventory](../../data/processed/popu/popu_tactilus_inventory_v0.1.csv)
- [P2 质量结果](../../outputs/metrics/popu_tactilus_quality_results_v0.1.csv)
- [P4a 特征配置](../../configs/experiments/popu_features_p4a_v0.1.json)
- [冻结的 P3.1 Mask 规则](../../configs/experiments/popu_geometry_frozen_v0.2.json)（只读，作为 mask_rule 依据）

产物（`feature_schema_version=v0.1`）：

- [逐 snapshot 特征表](../../data/processed/popu/popu_features_p4a_v0.1.csv)（51,000 行 × 88 列）
- [primary cohort 键清单](../../data/processed/popu/popu_features_p4a_primary_cohort_v0.1.csv)（50,060 行）
- [EXCLUDED manifest](../../data/processed/popu/popu_features_p4a_excluded_v0.1.csv)（60 行 `others.json`）
- [schema/summary JSON](../../outputs/reports/popu_features_p4a_summary_v0.1.json)
- [姿态/受试者分布 CSV](../../outputs/metrics/popu_features_p4a_distribution_v0.1.csv)（300 行 = 60 受试者 × 5 姿态）
- [代表图](../../outputs/figures/popu_features_p4a_overview_v0.1.png)（五姿态热力图 + Mask 轮廓 + 逐姿态行数柱状图）

## 4. 特征 schema

特征列共 `71` 列，与标签/追溯列严格分离：

| 组 | 列数 | 说明 |
|---|---:|---|
| 原始强度 | 14 | `sum/mean/std/min/max`、`p25/p50/p75/p90/p95/p99` 分位数、`nonzero_cell_count/nonzero_fraction`、`positive_mean` |
| Mask / Geometry | 18 | `mask_threshold_raw`、`mask_cell_count/fraction`、`component_count`、`bbox_*`（含 `bbox_area`）、`centroid_*_fraction`、`cop_*_fraction`、`principal_axis_degrees/anisotropy`、`contact_signal_sum` |
| 形状 | 3 | `bbox_aspect_ratio`（高/宽）、`mask_extent`（mask 面积/bbox 面积，紧致填充率）、`mask_compactness`（等周比 P²/(4πA)） |
| 网格分区 | 36 | `4×3=12` 个等宽带区域 × `{sum, fraction, peak}` |

标签与追溯列（不进模型）：`sample_id`、`dataset_id`、`source_relative_path`、`subject_id`、`posture`、`variation`、`snapshot_index`、`snapshot_key`、`quality_status`、`cohort`、`rows`、`columns`、`matrix_orientation`、`mask_rule_version`、`feature_schema_version`、`feature_status`、`feature_reason`。

关键口径：

- `mask_rule_version = largest_component@frozen_v0.2`（参数与 `popu_geometry_frozen_v0.2.json` 一致）。
- `matrix_orientation` 固定为 `row-major origin=upper-left`，与 `load_tactilus_record` 还原一致。
- 网格分区为 `np.linspace(0, N, bands+1).astype(int)` 等宽带：64 行 → 4×16 行带，27 列 → 3×9 列带。
- `cohort = primary` 当且仅当固定姿态且 `quality_status=ACCEPT`；`warn` 对应 `quality_status=WARN`。

## 5. 实际结果

| 结果 | 数量 | 含义 |
|---|---:|---|
| 处理记录 | 5,160 | P1 全量 |
| 可展开记录 | 5,100 | 固定姿态 + ACCEPT/WARN |
| EXCLUDED 记录 | 60 | `others.json`，`missing_fixed_posture_label`，不展开 |
| 特征行（snapshot） | 51,000 | 5,100 × 10 |
| primary（ACCEPT） | 50,060 | 固定姿态 + ACCEPT |
| warn | 940 | 固定姿态 + WARN |
| feature_status=OK | 50,978 | 主轴等均可定义 |
| feature_status=WARN | 22 | `insufficient_cells_for_principal_axis` |

逐姿态行数：`empty=600`、`left=12,600`、`prone=12,600`、`right=12,600`、`supine=12,600`。

逐受试者：60 名受试者，每人 `empty=10` + 四卧姿各 `210` = `850` 个 snapshot。

`feature_status=WARN` 的 22 行全部为 `empty` 姿态，`principal_axis_degrees/anisotropy` 为 `NaN`（单有效格点无法定义主轴），分布在受试者 `30 / 53 / 54 / 58`。其余 69 个特征列全部有限值。

## 6. 已验证、合理推断、尚未验证

### 已验证

- 51,000 行逐 snapshot 特征表可复现生成；`sample_id` 全表唯一。
- 特征列 71 列全部为数值，且不包含 `subject_id/posture/variation/source_relative_path/snapshot` 等任何标签字段；用相同矩阵、不同标签生成的特征向量完全一致（单元测试 `test_feature_values_do_not_depend_on_subject_or_posture_labels`）。
- 网格分区列求和等于 `intensity_sum`、占比列求和 ≈ 1（单元测试）。
- `others.json`（60 条）只进 EXCLUDED manifest，未混入固定姿态样本。
- primary cohort（50,060）全部为 ACCEPT；warn（940）全部为 WARN。
- 冻结 `largest_component` 规则被 `describe_geometry` 复用，未重新实现阈值/连通域逻辑。

### 合理推断

- 该表可作为 A4 受试者隔离姿态 Baseline 的输入；逐 snapshot 粒度保留了记录内时间信息。
- 22 个单格点 `empty` snapshot 的 NaN 主轴可被下游显式丢弃，而不影响其余特征。

### 尚未验证

- 特征对受试者隔离姿态分类的实际区分力（属于 P5/A4，本阶段不训练）。
- Mask/几何与人体解剖轮廓、真实接触面积、肩/腰/骨盆的误差。
- 被 `largest_component` 丢弃的分离连通域中真实肢体接触的信息损失。
- 在 TIP、SLP、PressurePose、自研床垫或整夜动态数据上的泛化。

## 7. 对后续阶段的决策

- P4a 完成；不生成任何肩/腰/骨盆监督标签，区域监督继续 HOLD。
- P5/A4 读取特征表时，应以 `cohort` 与 `feature_status` 显式过滤：先只取 `primary`（固定姿态 + ACCEPT），并按 `feature_status=OK` 决定是否剔除 22 个 NaN 主轴样本；同时报告“全量 ACCEPT+WARN”与“仅 ACCEPT”两套口径。

## 8. 下一步最小输入与放行条件

P5/A4 的最小输入：本特征表 + 冻结的 `cohort/feature_status` 口径 + 受试者隔离切分（GroupKFold / LOSO）协议。放行条件：验证集选模型、测试集只使用一次；不把随机逐帧切分当作跨受试者泛化。

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-18 | 首跑 P4a：生成 51,000 行逐 snapshot 特征表、primary cohort、EXCLUDED manifest 与本报告；`40 passed`。 |
