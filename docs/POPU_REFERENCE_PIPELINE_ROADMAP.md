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

## 3. 已完成：P0–P5.1

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
| P5.1 | 横向比较与候选复核（repeated subject-grouped CV + 特征消融 + 冻结传统模型候选） | COMPLETE — TRADITIONAL_CANDIDATE_FROZEN | [P5.1](stage_reports/P5_1_POPU_GROUPED_MODEL_COMPARISON_v0.1.md) |

## 4. 推进顺序与每阶段的输入 / 输出 / 通过 / 停止

```text
P5 首轮 Baseline（已完成）
    → P5.1 横向比较与候选复核（已完成，传统模型候选已冻结）
    → P5.2 PoPu 神经网络公平比较（P5.2-A CNN 底座与 Smoke → P5.2-B Mini 筛选 → P5.2-C Full 公平比较）
    → 冻结 PoPu 总体候选
    → P6 UNKNOWN/REJECT 与错误分析（总体候选冻结后放行）
    → P7 软件鲁棒性（降密度、坏点、噪声）
    → PoPu 参考验证包
    → SLP/PressurePose Adapter 与各自任务线
```

### P5.1 横向比较与候选复核

- 输入：P4a v0.1 特征表（50,060 snapshots / 60 subjects / 5,006 records）+ 冻结的比较协议。
- 输出：配置驱动模型注册 + 通用分组 evaluator（逐 snapshot 概率列）+ 记录聚合（每 JSON 10 snapshot 概率平均）；repeated subject-grouped CV（5 折 × 3 repeats，group=subject_id）7 候选横向比较、top-2 × 5 特征消融、候选冻结。
- 通过条件：候选排序可解释（`calibrated_linear_svm` record macro-F1 0.9452 为最优，logreg 0.9424 在 margin 0.005 内统计平局、由 tie-break 1 胜出）；逐 snapshot/记录/受试者稳定性都报告；候选具备完整追溯（config / split / 指标 / 预测明细）。
- 停止条件：**传统模型候选冻结**为 `popu_research_candidate_p5_1_v0.1`（16,097 B，独立重载 smoke OK）；该候选是传统模型、不覆盖 CNN，PoPu 总体候选需经 P5.2 神经网络公平比较后确定；总体候选冻结前不设置 UNKNOWN/REJECT 阈值、不进入 WSL 工程化。

### P5.2 PoPu 神经网络公平比较

在 P5.1 传统候选之上新增神经网络候选的公平比较，不改写 P5.1 数值和历史报告。

#### P5.2-A：CNN 训练底座与 Smoke

- 输入：P4a v0.1 特征表 + 原始压力矩阵；候选包括 P5.1 `calibrated_linear_svm`（冻结对照）、原始压力矩阵 MLP、TinyCNN、Small ResNet，必要时 CNN + 71 工程特征。
- 输出：MLP/TinyCNN/Small ResNet 训练与评估接口、受试者隔离与 fold 内归一化、标签映射与左右翻转标签交换、checkpoint/resume、1 epoch 小样本 Smoke。
- 通过条件：受试者切分隔离、train-fold normalization、标签映射、左右翻转标签交换、checkpoint/resume、CPU 与 CUDA Smoke、模型重载预测全部通过。
- 停止条件：此阶段不跑 Full CV；通过后由 Controller 另行签发 P5.2-B 的 EXP 配置。

#### P5.2-B：Mini 筛选（MINI_READY_TO_RUN — REVIEWER_ACCEPTED，未运行）

- 输入：P5.2-A 通过的候选 + 开发受试者固定子集（冻结 `["1","2","3","4","5","6"]`，看结果前固定）+ 冻结 P2 质量 manifest（`primary` = ACCEPT-only cohort，WARN/EXCLUDED 不入 Mini；其 SHA-256 冻结并在读取前校验）。
- 输出：3-5 epochs、固定种子（seed=42）、早停（仅 `val_loss`/`min`/`patience=2`/`min_epochs=3`）与 best-checkpoint（`argmin val_loss`）规则的 Mini Run 产物；`device=cuda`（无 CUDA 直接失败）；显式 `train_subject_ids`/`val_subject_ids`（≥2 验证）；逐 epoch/逐类别指标、checkpoint/resume/reload、固定 seed 复现。
- 通过条件：排除明显不可行候选（指标有限 + 学习信号 `best_val_balanced_accuracy > 1/5 + 0.05`）；协议问题记 `needs_fix`；不形成最终排名。
- 停止条件：只有 `proceed` 的候选进入 Full。
- 就绪产物：[协议报告](stage_reports/P5_2_B_POPU_NEURAL_MINI_PROTOCOL_v0.1.md)、[冻结配置](../configs/experiments/popu_neural_mini_v0.1.json)、`runner_type=popu_neural_mini`；Reviewer 首轮复核返回 `REVIEW_NEEDS_FIX`，已按 6 项要求完成首轮修复提交并追加数据 Manifest SHA-256 哈希校验；Reviewer 最终复核已接受（commit `4b8b73e`、`346 passed`、ACCEPT-only cohort 505 ACCEPT / 5,050 snapshots），标记 `MINI_READY_TO_RUN`；尚未运行真实 Mini/Full。

#### P5.2-C：Full 公平比较

- 输入：P5.2-B 通过的候选 + 与 P5.1 相同的受试者隔离原则、记录聚合与主指标。
- 输出：record macro-F1 + repeated splits 波动、最差受试者、逐类别、校准、参数量、推理时间和训练成本。
- 通过条件：若神经网络相对 SVM 提升 < 0.005 且稳定性/难例无实质改善，优先保留更简单的 SVM；只有 Reviewer 接受后才冻结 PoPu 总体候选。
- 停止条件：冻结 PoPu 总体候选并进入 P6。

### P6 UNKNOWN/REJECT 与错误分析

- 前置：P5.2 完成总体候选选择并冻结后放行；P5.1 传统候选不足以单独作为最终阈值基准。
- 输入：冻结的总体候选 + 逐样本预测（含置信度）。
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
- 冻结协议： P4a [popu_features_p4a_v0.1.json](../configs/experiments/popu_features_p4a_v0.1.json)、P5 [popu_baseline_p5_v0.1.json](../configs/experiments/popu_baseline_p5_v0.1.json)、P5.1 [popu_model_comparison_p5_1_v0.1.json](../configs/experiments/popu_model_comparison_p5_1_v0.1.json)、P5.2-B Mini [popu_neural_mini_v0.1.json](../configs/experiments/popu_neural_mini_v0.1.json)
