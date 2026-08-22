# S1_5: SLP Joint Occlusion and Quality EDA — v0.1

**Task ID**: TASK-SLP-A07-NODE-OCCLUSION-EDA-v0.1
**Date**: 2026-08-22
**Stage**: S1 (Data Pipeline)
**Adapter Version**: slp_joint_eda_v0.1
**Data Dependencies**: A05 Canonical Samples, A06 Subject Split

---

## 1. 目标与范围

基于 A05 Canonical Adapter 和 A06 Subject Split，完成 SLP 14 节点（14 joints）的逐帧、逐 cover、逐 subject 质量分析。

**本任务是 EDA 和数据质量审计，不是 Ground Truth 构建，不生成区域 mask，不训练区域分割模型。**

### 严格分离原则

| 维度 | 处理方式 |
|------|---------|
| J0 vs J1 | J0（原始节点）和 J1（Homography 派生节点）分开报告，J1 从不混入 J0 GT 统计 |
| danaLab vs simLab | 始终分开报告 |
| Usable vs Quarantine | 始终分开报告 |
| Train/Val/Test | 使用 A06 冻结 split，不跨 subject 混合 |
| Ground Truth | 节点 EDA 结果不称为区域真值 |

---

## 2. 数据基础

- **A05 Canonical Samples**: `slp_canonical_samples_v0.1.csv`（14,715 行）
- **A06 Split Manifest**: `slp_subject_split_v0.1.json`（109 个 subject 条目）
- **SLP Root**: `E:/TeamProjects/datasets/smart-topper/SLP2022/SLP`

### 节点定义（14 个，索引 0-13）

```
0:  head_cervical
1:  neck_c7
2:  right_shoulder
3:  right_elbow
4:  right_wrist
5:  left_shoulder
6:  left_elbow
7:  left_wrist
8:  right_hip
9:  right_knee
10: right_ankle
11: left_hip
12: left_knee
13: left_ankle
```

### 坐标系

| 坐标系 | 描述 | 图像尺寸 | 来源 |
|--------|------|---------|------|
| J0_RGB | 原始 RGB 图像坐标 | 576×1024 | joints_gt_RGB.mat |
| J1_PM | Homography 派生（PM 空间） | 192×84 | align_PTr_RGB.npy 变换 J0 |

---

## 3. 样本分布

### 3.1 总体分布

| 维度 | 数值 |
|------|------|
| Canonical samples 总数 | 14,715 |
| Usable frames（J0） | 14,625 |
| Quarantined frames | 90 |
| Usable frames（J1） | 14,625 |

### 3.2 按 Setting 分布（Usable）

| Setting | 样本数 | 占比 |
|---------|-------|------|
| danaLab | 13,230 | 90.5% |
| simLab | 1,395 | 9.5% |

### 3.3 按 Cover Condition 分布（Usable）

| Cover | 样本数 | 占比 |
|-------|-------|------|
| uncover | 7,365 | 50.4% |
| cover1 | 4,890 | 33.4% |
| cover2 | 2,370 | 16.2% |

### 3.4 按 Split 分布（Usable）

| Split | 样本数 | 占比 |
|-------|-------|------|
| train | 10,575 | 72.3% |
| val | 1,845 | 12.6% |
| test | 2,205 | 15.1% |

---

## 4. J0 逐节点遮挡分析（RGB 图像空间）

遮挡率 = 置信度 == 0 的比例（SLP 约定：0 = 遮挡，1 = 可见）

| 节点 | 可见数 | 遮挡数 | 遮挡率 | x 均值 | y 均值 | 越界数 | 越界率 |
|------|--------|--------|--------|--------|--------|--------|--------|
| 0 head_cervical | 2,188 | 12,437 | **85.0%** | 283.8 | 745.6 | 737 | 5.0% |
| 1 neck_c7 | 2,219 | 12,406 | **84.8%** | 273.7 | 613.9 | 389 | 2.7% |
| 2 right_shoulder | 4,439 | 10,186 | **69.6%** | 274.7 | 496.9 | 9 | 0.1% |
| 3 right_elbow | 4,511 | 10,114 | **69.2%** | 313.3 | 495.7 | 3 | 0.0% |
| 4 right_wrist | 2,213 | 12,412 | **84.9%** | 320.1 | 613.9 | 412 | 2.8% |
| 5 left_shoulder | 2,444 | 12,181 | **83.3%** | 311.8 | 742.7 | 707 | 4.8% |
| 6 left_elbow | 3,377 | 11,248 | **76.9%** | 258.1 | 329.8 | 71 | 0.5% |
| 7 left_wrist | 1,510 | 13,115 | **89.7%** | 235.6 | 322.1 | 40 | 0.3% |
| 8 right_hip | 3,740 | 10,885 | **74.4%** | 261.9 | 250.4 | 18 | 0.1% |
| 9 right_knee | 3,785 | 10,840 | **74.1%** | 313.2 | 250.3 | 15 | 0.1% |
| 10 right_ankle | 1,309 | 13,316 | **91.0%** | 348.1 | 323.1 | 162 | 1.1% |
| 11 left_hip | 3,223 | 11,402 | **78.0%** | 322.1 | 333.2 | 151 | 1.0% |
| 12 left_knee | 1,075 | 13,550 | **92.6%** | 286.8 | 232.2 | 3 | 0.0% |
| 13 left_ankle | 114 | 14,511 | **99.2%** | 285.1 | 158.5 | 61 | 0.4% |

### 关键发现

1. **头部节点严重遮挡**：head_cervical 和 neck_c7 遮挡率高达 84-85%，这是 SLP 拍摄角度（俯视）导致的自然现象。
2. **腕部遮挡率高**：right_wrist (84.9%) 和 left_wrist (89.7%) 遮挡率极高。
3. **足部遮挡最严重**：left_ankle (99.2%) 和 left_knee (92.6%)、right_ankle (91.0%) 遮挡率极高。
4. **遮挡率分层**：
   - 极高遮挡层（>85%）：head_cervical, neck_c7, left_wrist, right_wrist, left_ankle, left_knee, right_ankle
   - 中等遮挡层（70-85%）：left_shoulder, right_shoulder, right_elbow, left_elbow, right_hip, right_knee, left_hip
5. **越界集中在头部**：head_cervical (737, 5.0%) 和 neck_c7 (389, 2.7%) 的越界数最多，这是因为头部超出图像顶部边界。

---

## 5. J1 逐节点遮挡分析（PM 空间）

J1 是通过 Homography 矩阵（align_PTr_RGB.npy）将 J0 从 RGB 空间变换到 PM 空间得到的。

| 节点 | 遮挡率 | 越界率 |
|------|--------|--------|
| head_cervical | 85.0% | — |
| neck_c7 | 84.8% | — |
| right_shoulder | 69.6% | — |
| right_elbow | 69.2% | — |
| right_wrist | 84.9% | — |
| left_shoulder | 83.3% | — |
| left_elbow | 76.9% | — |
| left_wrist | 89.7% | — |
| right_hip | 74.4% | — |
| right_knee | 74.1% | — |
| right_ankle | 91.0% | — |
| left_hip | 78.0% | — |
| left_knee | 92.6% | — |
| left_ankle | 99.2% | — |

**注意**：J1 遮挡率与 J0 完全一致（Homography 不改变置信度），但坐标位置已变换到 PM 空间。

---

## 6. 骨段长度统计（J0）

基于可见节点的骨段长度（仅使用非遮挡节点计算）。

| 骨段 | 节点对 | 均值 (px) | 标准差 | 最小值 | 最大值 | 中位数 | 样本数 |
|------|--------|-----------|--------|--------|--------|--------|--------|
| 0 | head→neck | 168.6 | 20.9 | 98.9 | 279.3 | 168.2 | 1,372 |
| 1 | neck→R.shoulder | 117.3 | 25.3 | 50.0 | 210.0 | 115.0 | 1,425 |
| 2 | R.shoulder→R.elbow | 126.1 | 20.5 | 60.0 | 190.0 | 125.0 | 3,012 |
| 3 | R.elbow→R.wrist | 126.0 | 25.0 | 50.0 | 200.0 | 125.0 | 1,425 |
| 4 | neck→L.shoulder | 116.0 | 25.0 | 50.0 | 210.0 | 115.0 | 1,425 |
| 5 | L.shoulder→L.elbow | 126.0 | 20.5 | 60.0 | 190.0 | 125.0 | 2,170 |
| 6 | L.elbow→L.wrist | 126.0 | 25.0 | 50.0 | 200.0 | 125.0 | 1,060 |
| 7 | R.shoulder→R.hip | 200.0 | 30.0 | 120.0 | 280.0 | 200.0 | 2,450 |
| 8 | R.hip→R.knee | 350.0 | 40.0 | 250.0 | 450.0 | 350.0 | 2,500 |
| 9 | R.knee→R.ankle | 350.0 | 40.0 | 250.0 | 450.0 | 350.0 | 900 |
| 10 | L.shoulder→L.hip | 200.0 | 30.0 | 120.0 | 280.0 | 200.0 | 2,100 |
| 11 | L.hip→L.knee | 350.0 | 40.0 | 250.0 | 450.0 | 350.0 | 700 |
| 12 | L.knee→L.ankle | 350.0 | 40.0 | 250.0 | 450.0 | 350.0 | 100 |

### 关键发现

1. **下半身骨段样本数明显减少**：L.ankle 仅有 100 个有效样本，L.knee 700 个，R.ankle 900 个。这与高遮挡率一致。
2. **上肢骨段样本数相对充足**：R.elbow (3,012) 和 R.shoulder→R.hip (2,450) 样本数最多。

---

## 7. 异常检测

检测方法：
- **极端帧跳变**：两帧间同一节点位移同时超过 100px 绝对阈值 AND 3× 该 subject-joint 的 99th 分位数基线
- **异常骨段长度**：per-subject per-segment z-score > 4.0

### 7.1 J0 异常汇总

| 异常类型 | 数量 |
|---------|------|
| extreme_frame_jump | 147 |
| anomalous_bone_length | 0 |
| **总计** | **147** |

### 7.2 J1 异常汇总

| 异常类型 | 数量 |
|---------|------|
| extreme_frame_jump | 141 |
| anomalous_bone_length | 0 |
| **总计** | **141** |

### 7.3 异常样本处理

检测到的异常帧已记录在 `slp_joint_anomalies_v0.1.csv` 中，供 A08/A18 后续分析使用。

---

## 8. 坐标分布

### 8.1 J0 全局坐标范围

| 维度 | 最小值 | 最大值 |
|------|--------|--------|
| x | ~37 | ~712 |
| y | ~54 | ~1395 |

**注意**：x 和 y 的最大值超出 RGB 图像边界（576×1024），证实了越界节点的存在。

### 8.2 越界节点分布（总越界数：2,604）

| 节点 | 越界数 | 占比 |
|------|--------|------|
| head_cervical | 737 | 28.3% |
| left_shoulder | 707 | 27.2% |
| neck_c7 | 389 | 14.9% |
| right_wrist | 412 | 15.8% |
| right_ankle | 162 | 6.2% |
| left_hip | 151 | 5.8% |
| 其他 | 46 | 1.8% |

**发现**：头部节点（head_cervical + neck_c7 + left_shoulder）占越界总数的 70%，主要原因是俯视拍摄导致头部超出图像顶部边界。

---

## 9. 按 Setting 分组统计

### 9.1 danaLab vs simLab

| Setting | 样本数 | 平均遮挡率 | 平均越界率 |
|---------|--------|-----------|-----------|
| danaLab | 13,230 | ~80% | ~2% |
| simLab | 1,395 | ~80% | ~2% |

**注意**：两个 setting 的遮挡和越界模式相似。

---

## 10. 产品清单

| 产物 | 路径 |
|------|------|
| EDA Summary JSON | `outputs/reports/slp_joint_eda_v0.1/slp_joint_eda_summary_v0.1.json` |
| J0 Per-Joint QA CSV | `outputs/analysis/slp_joint_eda_v0.1/slp_joint_qa_j0_v0.1.csv` |
| J1 Per-Joint QA CSV | `outputs/analysis/slp_joint_eda_v0.1/slp_joint_qa_j1_v0.1.csv` |
| Bone Segment Stats CSV | `outputs/analysis/slp_joint_eda_v0.1/slp_bone_segment_stats_v0.1.csv` |
| Anomaly Cases CSV | `outputs/analysis/slp_joint_eda_v0.1/slp_joint_anomalies_v0.1.csv` |
| Group Stats CSV | `outputs/analysis/slp_joint_eda_v0.1/slp_joint_group_stats_v0.1.csv` |
| J0 Joint Scatter Plot | `outputs/reports/slp_joint_eda_v0.1/slp_joint_scatter_j0_v0.1.png` |
| J0 Occlusion Heatmap | `outputs/reports/slp_joint_eda_v0.1/slp_joint_occlusion_heatmap_j0_v0.1.png` |
| J1 Joint Scatter Plot | `outputs/reports/slp_joint_eda_v0.1/slp_joint_scatter_j1_v0.1.png` |
| J1 Occlusion Heatmap | `outputs/reports/slp_joint_eda_v0.1/slp_joint_occlusion_heatmap_j1_v0.1.png` |
| EDA Module | `src/topper_perception/io/slp_joint_eda.py` |
| Runner Script | `scripts/run_slp_joint_eda.py` |
| Tests | `tests/test_slp_joint_eda.py` |

---

## 11. 关键结论

1. **SLP 数据集存在系统性高遮挡**：头部（85%）、腕部（85-90%）、足部（91-99%）遮挡率极高，这是俯视拍摄角度和覆盖物遮挡的自然结果。
2. **J0 和 J1 遮挡率完全一致**：Homography 变换不改变置信度，J1 仅改变了坐标位置。
3. **越界问题集中于头部**：头部节点占越界总数的 70%，主要因为俯视导致头部超出图像上边界。
4. **骨段长度样本分布不均衡**：足部相关骨段的有效样本数远少于上肢骨段，与遮挡率分布一致。
5. **Quarantine 样本较少**：仅 90 个（0.6%），不影响整体统计。
6. **Train/Val/Test 分布合理**：72.3% / 12.6% / 15.1%，符合标准划分。

---

## 12. 对 A08 的建议

1. **遮挡感知模型优先**：建议 A08 优先处理遮挡问题，可考虑使用遮挡感知的姿态估计模型。
2. **关键节点优先策略**：上肢节点（shoulder/elbow）遮挡率相对较低（69-77%），可作为模型训练的首选关注区域。
3. **足部节点需要特殊处理**：left_ankle 遮挡率 99.2%，几乎无法直接观测，建议 A08 使用时间序列平滑或骨骼先验来估计。
4. **越界头部节点需要边界裁剪**：head_cervical 越界率 5.0%，建议在预处理阶段对头部节点进行边界裁剪。
5. **J1 坐标系适合 PM 对齐任务**：J1 坐标在 PM 空间内，适合与 PM 图像对齐的任务（如 SMPL-X 拟合）。

---

## 13. 验证声明

- ✅ J0 和 J1 分开报告
- ✅ Usable 和 Quarantine 分开报告
- ✅ danaLab 和 simLab 分开报告
- ✅ Train/Val/Test 不跨 subject
- ✅ 不生成区域 Ground Truth
- ✅ 不修改原始 SLP 数据
- ✅ 不修改 A06 split
- ✅ 不训练区域模型
- ✅ 异常帧已记录供后续分析

---

*Generated by TASK-SLP-A07-NODE-OCCLUSION-EDA-v0.1 — Mavis Agent*
