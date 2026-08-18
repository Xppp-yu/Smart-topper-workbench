# PoPu 参考管线路线图

> 这是 PoPu（Tactilus）参考链路的**推进导航页**：现在在哪、下一步做什么、每个阶段的输入/输出/通过/停止条件，以及后续数据各接力什么。逐阶段的详细方法、指标定义和数据口径见 [验证 Workflow 总蓝图](VALIDATION_WORKFLOW_MASTER.md)，本页不重复展开。

## 1. 一句话定位

PoPu 是**公开压力矩阵数据集**，用于在真实压力信号上建立可复现的算法候选链路（读取 → 质量 → Mask → 特征 → 姿态 Baseline → 鲁棒性）。**PoPu 的任何结果都是候选证据，不是产品验证。**

## 2. PoPu 能验证什么 / 缺少什么真值

| 能验证 | 不能验证 / 缺少的真值 |
|---|---|
| 矩阵读取、质量门、接触 Mask、几何/特征提取链路可执行 | 自研硬件、气囊闭环、整夜稳定性、舒适性、安全性 |
| 受试者隔离下固定姿态分类的**候选**区分力 | 没有“真姿态”以外的独立真值来源（无视频/动捕/人工逐帧标注） |
| 软件层面降密度、坏点、噪声的候选敏感性（P7 才做） | 真实低密度硬件布局、点距、量程、噪声与坏点行为 |
| 算法方法学（切分、选型、指标）的可行性 | 与 SLP/PressurePose/PMD 逐行配对或共享受试者 |
| — | 身体区域（Head/Torso/Arm/Leg、肩/腰/骨盆）的逐记录监督真值（区域监督 HOLD） |

## 3. 已完成：P0–P5 v0.1

| 阶段 | 内容 | 状态 | 阶段报告 |
|---|---|---|---|
| P0 | 环境、数据路径与最小读取链路 | COMPLETE | — |
| P1 | 全量 PoPu Tactilus Inventory | COMPLETE | [P1](stage_reports/P1_POPU_TACTILUS_INVENTORY_v0.1.md) |
| P2 | 样本画廊与 ACCEPT/WARN/REJECT 质量门 | COMPLETE | [P2](stage_reports/P2_POPU_TACTILUS_QUALITY_GATE_v0.1.md) |
| P3 | 接触 Mask 与 Geometry | COMPLETE | [P3](stage_reports/P3_POPU_CONTACT_MASK_AND_GEOMETRY_v0.1.md) |
| P3.1 | Mask 候选策略比较并冻结 `largest_component` | COMPLETE | [P3.1](stage_reports/P3_1_POPU_MASK_STRATEGY_FREEZE_v0.1.md) |
| P3.2 | COCO 区域标注—压力记录对齐审计（区域监督 HOLD） | COMPLETE | [P3.2](stage_reports/P3_2_POPU_SEGMENTATION_ALIGNMENT_AUDIT_v0.1.md) |
| P4a | 无标签逐 snapshot 特征表（51,000 行 × 71 特征） | COMPLETE | [P4a](stage_reports/P4a_POPU_LABEL_FREE_FEATURES_v0.1.md) |
| P5 | 受试者隔离姿态 Baseline（首轮，候选未冻结） | COMPLETE — FIRST_ROUND_BASELINE | [P5](stage_reports/P5_POPU_SUBJECT_ISOLATED_POSTURE_BASELINE_v0.1.md) |

## 4. 推进顺序与每阶段的输入 / 输出 / 通过 / 停止

```text
P5 首轮 Baseline（已完成）
    → P5.1 横向比较与候选复核（框架已实现，比较待运行）
    → P6 UNKNOWN/REJECT 与错误分析
    → P7 软件鲁棒性（降密度、坏点、噪声）
    → PoPu 参考验证包
```

### P5.1 横向比较与候选复核

- 输入：P5 v0.1 逐样本预测表（含置信度）+ 当前首轮领先候选 `logreg`。
- 输出：配置驱动模型注册 + 通用分组 evaluator + 记录聚合的横向比较框架（repeated subject-grouped CV；不再沿用“仅对最终候选使用一次 held-out test”口径，也不声称存在未查看的 PoPu test）、模块化增强、候选复核结果。
- 通过条件：repeated subject-grouped CV 口径下候选排序与 v0.1 一致或可解释地变化；逐 snapshot 与逐记录/逐受试者稳定性都报告；候选具备完整追溯（config / split / 指标 / 预测明细）。
- 停止条件：**候选冻结**；冻结前不设置 UNKNOWN/REJECT 阈值、不进入 WSL 工程化。

### P6 UNKNOWN/REJECT 与错误分析

- 输入：P5.1 冻结候选 + 逐样本预测（含置信度）。
- 输出：confidence 阈值表、UNKNOWN/REJECT 口径、高置信错误个案。
- 通过条件：阈值仅由验证（开发集 OOF）受试者选择；空床与卧姿、高低置信错误的取舍被明确记录。
- 停止条件：进入 P7 前完成阈值与错误分析文档。

### P7 软件鲁棒性

- 输入：P6 后的候选 + 特征表 + 原始矩阵。
- 输出：降密度、坏点注入、噪声的候选敏感性表和错误图。
- 通过条件：明确区分软件消融与硬件验证；结论只表述为软件候选敏感性。
- 停止条件：生成 PoPu 参考验证包所需的全部证据齐备。

### PoPu 参考验证包

- 输入：P5.1/P6/P7 全部候选与证据。
- 输出：`validation_release/<candidate_id>/` 验证包（输入—真值—方法—结果—限制闭合）。
- 通过条件：包中每一级都有输入、真值、指标、输出与限制；候选满足晋级门槛。
- 停止条件：候选冻结并移交 WSL；PoPu 包仍不自动等于产品已验证。

## 5. 复用原则

**复用代码和方法，不复用未经验证的结论。** 代码、Pipeline、切分逻辑、指标函数可以在阶段间迁移；任何数字（macro-F1、阈值、候选排名）在被同口径复核前都只是候选值。

## 6. 后续数据各接力什么任务

| 数据 | 当前状态 | 计划接力的任务 | 约束 |
|---|---|---|---|
| SLP2022 | 已下载，`PRESENT_NOT_INTEGRATED` | 静态身体关节、跨模态对齐验证 | 需先建立 Adapter/Manifest 与坐标映射校验 |
| PressurePose | 已下载，`PRESENT_NOT_INTEGRATED` | Real：压力与 RGB-D/人体几何配准验证；Synthetic：预训练、几何可行性、扩充实验 | 需先建立 Adapter/Manifest；Real/Synthetic 分开评价 |
| TIP | 未纳入本工作台副本 | 肩、髋、人体轴、姿态与时间信息的主要候选真值 | 需下载记录 + Adapter + Manifest |
| 自研硬件同步数据 | 未进入本工作台 | 最终硬件、区域、连续过程与闭环验证（B0–B5 线） | 是“研究候选 → 产品证据”的必需缺口 |

以上数据均须分别评价，不能与 PoPu、PMD 逐行拼接成同一个受试者样本；适配完成前不宣称已验证。

## 7. 导航

- 阶段总看板与状态： [PROJECT_STATUS.md](PROJECT_STATUS.md)
- 验证方法学与证据边界总蓝图： [VALIDATION_WORKFLOW_MASTER.md](VALIDATION_WORKFLOW_MASTER.md)
- 阶段报告目录： [stage_reports/](stage_reports/)
- 冻结协议： P4a [popu_features_p4a_v0.1.json](../configs/experiments/popu_features_p4a_v0.1.json)、P5 [popu_baseline_p5_v0.1.json](../configs/experiments/popu_baseline_p5_v0.1.json)
