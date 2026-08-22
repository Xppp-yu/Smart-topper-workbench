# TASK-SLP-B01: Pressure-only Model Experiment Infrastructure — v0.1

**Task ID**: `TASK-SLP-B01-PRESSURE-ONLY-INFRA-v0.1`

**Date**: 2026-08-22

**Stage**: S1 (Data Pipeline — Pressure-only Infrastructure)

**Status**: `IMPLEMENTED — READY_FOR_REVIEW`

---

## 1. 阶段目标与完成判定

本阶段目标是在 Ground Truth 尚未冻结前，提前完成 Pressure-only 模型实验基础设施。

完成判定（Gate）：

- [x] Pressure-only Input Adapter 实现并测试完成
- [x] Region Label Provider 接口实现并测试完成
- [x] Metrics 模块实现并测试完成
- [x] Pressure 扰动模块实现并测试完成
- [x] Density Transform 模块实现并测试完成
- [x] Experiment Config & Manifest 接口实现并测试完成
- [x] 84 个定向测试全部通过
- [x] A03-A08 回归测试通过（157 passed, 1 skipped）
- [x] `git diff --check` 通过

---

## 2. 核心约束遵守情况

### 2.1 Pressure-only 约束

| 约束 | 状态 | 验证 |
|------|------|------|
| 最终模型输入只能是 Pressure Map | ✅ | `input_contract.modality = "PM"` |
| 不读取 RGB | ✅ | `visual_modalities_loaded = False` |
| 不读取 IR | ✅ | 仅有 provenance URI，无加载 |
| 不读取 Depth | ✅ | 仅记录 URI，不进入 tensor |
| 不读取 Skeleton | ✅ | 不在模态列表中 |
| 不读取 Point Cloud | ✅ | 不在模态列表中 |
| 不读取 Homography 作为特征 | ✅ | 仅记录 provenance |

### 2.2 数据边界

| 约束 | 状态 | 验证 |
|------|------|------|
| 不训练正式模型 | ✅ | 仅实现基础设施，无训练代码 |
| 不生成正式 mIoU | ✅ | 使用 synthetic arrays 测试 |
| 不生成区域 Ground Truth | ✅ | Region Label Provider 仅接口，无实现 |
| 不修改原始 SLP 数据 | ✅ | 所有操作基于 A05/A06 |
| 不修改 A06 split | ✅ | 仅读取 manifest |

---

## 3. 模块清单

### 3.1 新增文件

| 路径 | 角色 | 行数 |
|------|------|------|
| `src/topper_perception/io/slp_pressure_only_adapter.py` | Pressure-only Input Adapter | ~600 |
| `src/topper_perception/io/slp_region_label_provider.py` | Region Label Provider 接口 | ~800 |
| `src/topper_perception/evaluation/slp_pressure_metrics.py` | Metrics 模块 | ~550 |
| `src/topper_perception/evaluation/slp_pressure_perturbation.py` | Pressure 扰动模块 | ~500 |
| `src/topper_perception/evaluation/slp_density_transform.py` | Density Transform 模块 | ~450 |
| `src/topper_perception/experiments/slp_pressure_experiment.py` | Experiment Config 接口 | ~500 |
| `tests/test_slp_pressure_infrastructure.py` | 综合测试 | ~800 |

### 3.2 模块详情

#### 3.2.1 Pressure-only Input Adapter (`slp_pressure_only_adapter.py`)

**功能**：
- 从 A05 Canonical Sample 读取数据
- 仅加载 PM (Pressure Map) 作为模型输入
- 使用 A06 frozen subject split
- 排除 quarantine 样本
- 保留 provenance 和 quality flags

**关键类**：
- `SlpPressureOnlyAdapter`: 主适配器
- `PressureOnlySample`: 单样本数据结构
- `PressureInputContract`: 输入合同文档

**输入合同**：
```
{
  "modality": "PM",
  "image_size": [84, 192],  # (width, height)
  "dtype": "float32",
  "value_range": [0.0, 1.0],
  "preprocessing": ["normalize_to_0_1", "to_tensor_format"]
}
```

#### 3.2.2 Region Label Provider 接口 (`slp_region_label_provider.py`)

**功能**：
- 未来真值接入接口（当前不生成真值）
- 支持 region label URI
- 支持 label schema version
- 支持 label quality tier
- 支持 review status
- 支持 ignore/uncertain mask
- 支持 subject isolation 验证

**关键类**：
- `RegionLabelProvider`: 主接口
- `RegionLabel`: 单区域标签
- `SampleLabels`: 样本所有标签
- `RegionSchema`: 区域 schema 加载器

**关键接口字段**：
- `annotation_id`: 标签唯一标识
- `label_tier`: R0/R1/R2/R3
- `review_status`: pending/accepted/rejected/uncertain
- `quality_tier`: high/medium/low/rejected
- `is_ignore`, `is_uncertain`: 掩码标志

#### 3.2.3 Metrics 模块 (`slp_pressure_metrics.py`)

**功能**：
- mIoU (mean Intersection over Union)
- Macro-F1
- Per-region IoU
- Accuracy, Precision, Recall
- Confusion matrix
- Ignore/uncertain label 处理
- Empty/missing class 处理

**关键类**：
- `SegmentationMetrics`: 综合指标
- `RegionMetrics`: 单区域指标

**测试覆盖**：
- Synthetic data 指标计算
- Ignore label 排除
- Empty class 处理
- 确定性验证

#### 3.2.4 Pressure 扰动模块 (`slp_pressure_perturbation.py`)

**功能**：
- `random_noise`: 加性高斯噪声
- `sensor_noise`: 真实传感器噪声（热噪声/量化/坏点）
- `pressure_drift`: 基线压力漂移
- `dead_sensor`: 永久失效（卡在零值）
- `missing_sensor`: 整行/列缺失
- `local_outlier`: 局部异常读数
- `left_shift`, `right_shift`, `up_shift`, `down_shift`: 移位

**特性**：
- 固定 seed 可复现
- 原始输入不被覆盖
- 输出 perturbation config
- 每个函数有单元测试

**预设配置**：
- `create_light_perturbation_preset()`: 轻度扰动
- `create_heavy_perturbation_preset()`: 重度扰动
- `create_degradation_preset()`: 传感器退化

#### 3.2.5 Density Transform 模块 (`slp_density_transform.py`)

**功能**：
- 100%, 50%, 25%, 12.5% 密度
- Uniform grid 布局
- Sparse grid 布局
- Local high-density 布局

**特性**：
- 输出尺寸合同保持
- 记录保留的 sensor positions
- 不把 resize 当成真实硬件验证
- 只验证变换正确性

**关键函数**：
- `downsample_to_density()`: 主变换函数
- `select_uniform_positions()`: 均匀采样
- `select_sparse_positions()`: 稀疏采样
- `select_local_high_density_positions()`: 局部高密度采样

#### 3.2.6 Experiment Config 模块 (`slp_pressure_experiment.py`)

**功能**：
- experiment_id
- input_contract_version
- split_manifest
- region_label_manifest (可空，A17 前)
- preprocessing
- model_name
- random_seed
- perturbation_config
- density_config
- metrics
- runtime_device

**关键类**：
- `PressureExperimentConfig`: 完整实验配置
- `PreprocessingConfig`: 预处理配置
- `PerturbationConfig`: 扰动配置
- `DensityConfig`: 密度配置
- `MetricsConfig`: 指标配置

**验证规则**：
- `validate_experiment_config()`: 验证配置合法性
- `validate_exp_id()`: 验证 EXP-ID 格式
- `strict_label_check`: A17 前可跳过真值检查

---

## 4. 测试结果

### 4.1 新增测试（84 passed）

```
tests/test_slp_pressure_infrastructure.py
├── TestPressureOnlyAdapter (7 tests)
│   ├── test_adapter_creates_with_valid_inputs ✅
│   ├── test_adapter_excludes_quarantine ✅
│   ├── test_adapter_respects_split_filter ✅
│   ├── test_adapter_extracts_visual_uris_for_provenance_only ✅
│   ├── test_adapter_does_not_load_visual_modalities ✅
│   ├── test_adapter_verifies_subject_isolation ✅
│   └── test_input_contract_documentation ✅
├── TestRegionLabelProvider (6 tests)
│   ├── test_provider_rejects_empty_manifest_path ✅
│   ├── test_mock_provider_works ✅
│   ├── test_mock_provider_has_labels ✅
│   ├── test_label_tiers_are_recognized ✅
│   ├── test_review_statuses_are_recognized ✅
│   └── test_region_schema_loads ✅
├── TestSegmentationMetrics (8 tests) ✅
├── TestPressurePerturbations (26 tests) ✅
├── TestDensityTransform (13 tests) ✅
├── TestExperimentConfig (18 tests) ✅
├── TestIntegration (4 tests) ✅
└── TestExistingModules (3 tests) ✅
```

### 4.2 A03-A08 回归测试（157 passed, 1 skipped）

```
tests/test_slp_canonical_adapter.py: 20 passed
tests/test_slp_subject_split.py: 19 passed (1 skipped)
tests/test_slp_frame_index.py: passed
tests/test_slp_homography.py: passed
tests/test_slp_joint_eda.py: passed
tests/test_slp_body_geometry.py: 48 passed
tests/test_slp_region_annotation_schema.py: passed
tests/test_slp_inventory.py: passed
```

---

## 5. Pressure-only 输入合同

```json
{
  "contract_version": "slp_pressure_only_input_contract_v0.1",
  "modality": "PM",
  "image_size": [84, 192],
  "dtype": "float32",
  "value_range": [0.0, 1.0],
  "preprocessing": ["normalize_to_0_1", "to_tensor_format"],
  "notes": [
    "Pressure-only: visual modalities (RGB/IR/Depth) are NEVER model input.",
    "PM PNG values are 0-1 float32, loaded with cv2.IMREAD_UNCHANGED."
  ]
}
```

---

## 6. 已知限制

1. **无真实数据测试**: 本任务仅使用 synthetic arrays 测试，未在真实 SLP 数据上运行
2. **无正式模型**: 未实现模型架构，仅提供输入接口
3. **无正式 mIoU**: 指标测试使用 synthetic labels
4. **Region Label 为接口**: `RegionLabelProvider` 是未来真值接入的接口，当前不生成真值
5. **Density Transform 为理论变换**: 不模拟真实硬件限制

---

## 7. 禁止结论

本阶段**不能**据此声称：

- 模型在 SLP 上已验证
- Pressure-only 方案优于多模态方案
- 任何 mIoU 性能
- 传感器鲁棒性已验证
- 产品效果已验证

---

## 8. 下一步

| TASK-ID | 内容 | 依赖 |
|---------|------|------|
| TASK-SLP-A17 | Region Reference v1.0 Freeze | A16 完成 |
| TASK-SLP-B02 | 非学习区域基线 | B01 完成 |
| TASK-SLP-B03 | 单模态 Region Smoke | B01 完成 |
| TASK-SLP-B07 | Full 协议冻结 | B04-B06 完成 |

---

## 9. 验收检查清单

Reviewer 应至少检查：

1. [x] `slp_pressure_only_adapter.py` 仅加载 PM，不加载视觉模态
2. [x] `visual_modalities_loaded = False` 在 provenance 中明确
3. [x] `model_input_tensor_modalities = ("PM",)` 明确记录
4. [x] quarantine 样本在默认情况下被排除
5. [x] A06 split manifest 仅读取，不修改
6. [x] `RegionLabelProvider` 是接口，不生成真值
7. [x] 标签质量 tier 和 review status 正确映射
8. [x] Metrics 模块正确处理 ignore/uncertain 标签
9. [x] 扰动模块使用固定 seed 可复现
10. [x] 原始 pressure map 不被修改
11. [x] Density transform 记录 retained positions
12. [x] Experiment config 验证规则正确
13. [x] 所有新测试通过
14. [x] A03-A08 回归测试通过
15. [x] `git diff --check` 通过

---

*Generated by TASK-SLP-B01-PRESSURE-ONLY-INFRA-v0.1 — Mavis Agent*
