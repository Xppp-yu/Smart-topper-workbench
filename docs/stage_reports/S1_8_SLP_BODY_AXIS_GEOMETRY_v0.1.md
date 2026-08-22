# S1_8: SLP Body Axis and Bounding-Box Geometry — v0.1

**Task ID**: TASK-SLP-A08-BODY-AXIS-GEOMETRY-v0.1
**Date**: 2026-08-22
**Stage**: S1 (Data Pipeline — Geometry Infrastructure)
**Adapter Version**: slp_body_geometry_v0.1
**Data Dependencies**: A05 Canonical Samples, A06 Subject Split, A07 Joint EDA

---

## 1. 目标与范围

基于 A05 Canonical Adapter、A06 frozen split 和 A07 Joint EDA，实现人体轴（body axis）、人体 bbox 和方向置信度的确定性几何模块。

**本任务是几何基础设施，不是 Ground Truth 构建，不生成最终区域标签，不训练区域分割模型。**

### 严格分离原则

| 维度 | 处理方式 |
|------|---------|
| J0 vs J1 | 仅处理 J0（RGB 空间）；J1（PM 空间）由下游任务处理 |
| A06 split | 仅读取 split manifest，不修改 |
| Ground Truth | 输出为几何特征，不称为 segmentation truth |
| 确定性 | 相同输入 → 相同输出，无静默补齐 |

---

## 2. 实现的几何原语

### 2.1 人体轴（Body Axes）

| 轴 | 定义 | 来源节点 | 有效条件 |
|----|------|---------|---------|
| **Shoulder Axis** | 左肩 → 右肩（有向线段） | J_R_SHOULDER, J_L_SHOULDER | 两侧均 visible（非 occluded） |
| **Hip Axis** | 左髋 → 右髋（有向线段） | J_R_HIP, J_L_HIP | 两侧均 visible |
| **Longitudinal Axis** | 髋中点 → 肩中点（有向线段） | shoulder + hip midpoints | 两中点均 valid |
| **Body Center** | 肩中点与髋中点的中点 | shoulder + hip midpoints | 任一端点 valid |

### 2.2 方向（Orientation）

- **angle_degrees**: 纵向轴与水平轴夹角（0–360°），用 `atan2(dy, dx)` 计算
- **orientation_confidence**: 轴与垂直方向的对齐程度（|sin(angle)|），1.0 = 垂直，0.0 = 水平
- **orientation_status**: `normal` | `face_up` | `ambiguous` | `reject`
  - `face_up` 阈值：|dy| > 200px（肩-髋 y 差超过 200px）
- **face_up_detected**: boolean 标志（从 A07 分析已知旋转的样本）

### 2.3 边界框（Bounding Box）

- 基于所有 **visible 且 in-bounds** 的节点计算
- 轴对齐（AABB）
- **valid 条件**: span > 0 且 ≤ 1200px（J0 最大人体跨度）
- 如果只有 1 个有效节点 → span=0 → valid=False（最小 bbox 需要 ≥ 2 个点）

### 2.4 节点有效性分类

每个节点被分类为以下状态之一：

| 状态 | 条件 |
|------|------|
| `visible` | is_valid=True AND confidence=1 AND in-bounds |
| `occluded` | is_valid=True AND confidence=0 |
| `invalid` | is_valid=False（NaN coords）|
| `out_of_bounds` | is_valid=True AND confidence=1 AND x/y 越界 |

---

## 3. 质量字段

### 3.1 状态枚举

**AxisStatus**: `accept_full` | `accept_partial` | `uncertain_missing_shoulders` | `uncertain_missing_hips` | `uncertain_missing_both` | `reject_insufficient_visible` | `reject_no_valid_joints` | `reject_all_occluded`

**BboxStatus**: `accept` | `uncertain` | `reject`

**OrientationStatus**: `normal` | `face_up` | `ambiguous` | `reject`

### 3.2 计数字段

| 字段 | 描述 |
|------|------|
| `total_joints` | 总节点数 = 14 |
| `valid_joints` | 坐标非 NaN 的节点数 |
| `visible_joints` | valid AND confidence=1 AND in-bounds |
| `occluded_joints` | confidence=0（坐标可能 valid）|
| `out_of_bounds_joints` | 坐标越界节点数 |
| `missing_joints` | 坐标 NaN 的节点数 |

### 3.3 错误码

| 错误码 | 触发条件 |
|--------|---------|
| `extreme_frame_jump` | 来自 A07 异常检测 |
| `anomalous_bone_length` | 来自 A07 异常检测 |
| `left_right_flip_suspected_shoulder` | shoulder axis 方向异常 |
| `left_right_flip_suspected_hip` | hip axis 方向异常 |
| `face_up_180_rotation_suspected` | 纵向轴 |dy| > 200px |
| `out_of_bounds_joints:N` | N 个节点越界 |
| `missing_joints:N` | N 个节点缺失（NaN）|
| `insufficient_visible_joints` | visible < 2 |
| `bbox_invalid_span` | bbox span ≤ 0 或 > 1200px |
| `bbox_partial_visibility` | visible < 50% |

---

## 4. 确定性规则

1. **相同输入 → 相同输出**: 纯函数，无随机性
2. **不静默补齐**: 关键节点缺失 → explicit reject/uncertain，不插值
3. **置信度不足 → reject/uncertain**: overall_confidence < 0.5 → reject；[0.5, 0.8) → uncertain
4. **规则不依赖模型输出**: 几何规则是固定的，不根据模型结果调整
5. **轴计算只看 visible joints**: confidence=0 的节点被排除

---

## 5. 真实数据结果

### 5.1 总体统计

| 指标 | 数值 |
|------|------|
| 处理帧数 | 14,625 |
| 跳过帧数 | 0 |
| **ACCEPT** | 4,079 (27.9%) |
| **UNCERTAIN** | 5,261 (36.0%) |
| **REJECT** | 5,285 (36.1%) |
| 平均 overall_confidence | 0.387 |
| 平均 orientation_confidence | 0.291 |

### 5.2 Axis Status 分布

| Status | Count | % |
|--------|-------|---|
| accept_full | 8 | 0.05% |
| accept_partial | 4,559 | 31.2% |
| uncertain_missing_both | 5,564 | 38.1% |
| reject_insufficient_visible | 1,042 | 7.1% |
| reject_no_valid_joints | 3,452 | 23.6% |

### 5.3 Bbox Status 分布

| Status | Count | % |
|--------|-------|---|
| accept | 307 | 2.1% |
| uncertain | 9,337 | 63.8% |
| reject | 4,981 | 34.1% |

### 5.4 异常检测

| 异常类型 | 数量 |
|---------|------|
| Left/right flip 可疑 | 81 |
| Face-up rotation 可疑 | 3,835 |
| Extreme frame jump（A07）| 0 |
| Anomalous bone length（A07）| 132 |

### 5.5 Out-of-bounds 分析

| 越界节点数 | 帧数 |
|-----------|------|
| 1 | 90 |
| 2 | 194 |
| 3 | 178 |
| 4 | 170 |
| 5 | 111 |
| 6+ | 85 |

**关键发现**: head_cervical（J0）和 neck_c7（J1）是最常见的越界节点，与 A07 EDA 报告一致。

---

## 6. 输出产物

| 产物 | 路径 |
|------|------|
| 几何 CSV | `outputs/analysis/slp_body_geometry_v0.1/slp_body_geometry_v0.1.csv` |
| Error Cases CSV | `outputs/analysis/slp_body_geometry_v0.1/slp_body_geometry_error_cases_v0.1.csv` (10,546 rows) |
| JSON Schema | `outputs/analysis/slp_body_geometry_v0.1/slp_body_geometry_v0.1.schema.json` |
| QA Summary | `outputs/reports/slp_body_geometry_v0.1/slp_body_geometry_summary_v0.1.json` |
| Overlay Manifest | `outputs/reports/slp_body_geometry_v0.1/overlay_manifest_v0.1.json` (12 overlays) |
| 几何模块 | `src/topper_perception/io/slp_body_geometry.py` |
| 运行脚本 | `scripts/run_slp_body_geometry.py` |
| 单元测试 | `tests/test_slp_body_geometry.py` |

---

## 7. 已知局限

1. **高 REJECT 率（36.1%）**: 来自 SLP 数据特性——大多数帧中只有肩膀和髋部可见，手臂和腿部被遮挡。这导致大量 `bbox_invalid_span` 错误。
2. **Face-up 检测阈值**: 200px 阈值是基于 SLP 数据的经验值，可能不适用于所有场景。
3. **方向置信度偏低**: 0.291 平均值反映了大多数 SLP 主体方向并非严格垂直。
4. **A07 anomaly CSV**: 必须在 A08 之前运行 A07 以获取 extreme_jump 和 anomalous_bone_length 标志。
5. **Bbox valid=0**: 当只有 1 个 visible in-bounds 节点时，bbox span=0 → valid=False → REJECT。这是保守设计，确保 bbox 至少是 2D 而非 1D 点。

---

## 8. 回归测试

所有现有测试通过（A03–A07）：
- `test_slp_frame_index.py`: 通过
- `test_slp_homography.py`: 通过
- `test_slp_canonical_adapter.py`: 通过
- `test_slp_joint_eda.py`: 通过
- `test_slp_subject_split.py`: 通过
- `test_slp_region_annotation_schema.py`: 通过
- `test_slp_inventory.py`: 通过（1 skipped）

A08 新增测试（48 tests）：全部通过。

---

## 9. 下一步建议

- **A09**: 基于 body axis 和 bbox，实现 region proposal 生成（不使用模型）
- **A10**: 区域 Ground Truth 构建（基于 A08 几何 + A05 标注）
- **A11**: 使用 A08 几何特征训练区域分割模型
- **优化方向**: 降低 bbox REJECT 率（考虑放宽 span=0 条件，或使用单一节点作为退化 bbox）
- **方向置信度**: 考虑引入 prior knowledge（如头部位置先验）来改进 face_up 检测
