# SLP 两阶段连续开发总计划 v0.2

## 1. 决策摘要

本计划吸收队长《智能睡眠顶垫支撑控制——感知层算法开发清单 V1.0》，但不把整份感知层清单错误地压到 SLP 数据集上。SLP 的核心职责是验证：

- 多模态人体位置、方向、姿态与 14 节点定位；
- 三种遮盖条件下的人体几何鲁棒性；
- 从节点和图像构造肩、胸背/躯干、腰腹、骨盆/髋、大腿等粗区域参考标签；
- 压力图能否预测这些粗区域及其不确定性。

SLP 不能独立验证：

- 自研传感器的空床基线、坏点、饱和、零漂和真实物理压力；
- 翻身、稳定 10–30 秒和动作前后连续时序；
- 气囊动作、Target Benefit、Spillover、Rollback；
- 舒适性、肌肉代偿、压疮风险或医学结论。

路线分为两阶段：

```text
阶段 I：Region Reference 形成前
S0 Inventory
→ S1 配对/坐标审计
→ S2 Canonical Sample/拆分
→ S3 节点与几何基线
→ S4 OpenCV 预标注
→ S5 人工复核/QC
→ Gate R1：Region Reference v1.0 冻结

阶段 II：Region Reference 形成后
冻结区域训练集
→ 区域启发式基线
→ 单模态模型
→ 遮盖压力测试
→ 有限融合
→ Full 公平比较
→ UNKNOWN/REJECT
→ SLP 研究候选与自研数据合同
```

关键更正：**SLP 8-region pressure-only GT（`SLP_8Region_Pressure_VAL_v1.1`）是当前项目的 PROJECT_ACCEPTED_REFERENCE_GT。** 其 provenance = `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`，`source_review_status = NOT_REVIEWED`，**不是人工像素级标注，不是医学/皮肤界面应力/产品真值**。OpenCV 自动输出（R0/R1）仍不是 Ground Truth。

历史说明：原 OpenCV+人工复核路线（`slp_region_annotation_v0.1`，R0–R3 10 区词表）已标记为 HOLD/SUPERSEDED，不作为当前训练合同。A09R（2026-08-24）已将 SLP8 GT 设为默认训练数据并完成 Reviewer 验收，B01 现为 READY。

## 2. 队长感知层清单与 SLP 的关系

| 队长清单能力 | SLP 可做程度 | SLP 任务 | 最终验证来源 |
|---|---|---|---|
| Q1 在床/离床 | 弱，不是 SLP 主任务 | 仅检查是否存在可用 empty/transition 证据 | 自研连续压力数据 |
| Q2 姿态 | 可做静态三类/节点几何 | 遮盖条件下姿态与方向辅助任务 | SLP + 自研数据 |
| Q3 位置、方向、轮廓 | 强项 | 节点、人体轴、视觉/压力轮廓 | SLP |
| Q4 肩、躯干、腰、骨盆、腿 | 可建立粗区域参考 | **SLP_8Region_Pressure_VAL_v1.1**（默认）+ 原 OpenCV+人工复核路线（HOLD） | SLP8 Pressure GT + 自采独立真值 |
| Q5 区域载荷、面积、集中 | 只能做部分空间 proxy | 不用单帧归一化 PM 图片宣称绝对载荷 | 自研已标定压力阵列 |
| Q6 异常支撑状态 | 不足 | 只形成候选特征，不冻结诊断状态 | 自研人体/床垫实验 |
| Q7 动作后变化 | 不支持 | 不在 SLP 上开发 | 自研同步气囊实验 |
| Q8 保持/继续/回退 | 不支持 | 不在 SLP 上开发 | 自研闭环实验 |

### 2.1 可从 SLP 继承到产品路线的内容

- Canonical Sample 和多模态坐标合同；
- 人体轴、位置、粗区域和不确定性定义；
- 受试者隔离评价；
- 遮盖条件压力测试；
- 模型置信度、拒识与最差受试者分析；
- 区域输出接口，为队长清单中的 `RegionFeature` 和 `SupportPerception` 提供上游区域。

### 2.2 不得从 SLP 直接继承的内容

- 绝对压力阈值、区域牛顿载荷或统一“正常压力”；
- 空床漂移、坏点、饱和、20 Hz 时序稳定状态；
- 气囊区域、动作指令、收益/惩罚评分；
- 产品传感器密度和硬件有效性。

## 3. 数据与真值层级

| 等级 | 名称 | 来源 | 是否可训练 | 报告名称 |
|---|---|---|---|---|
| J0 | Original Joint GT | RGB/IR 人工 14 节点 | 是 | 原始节点真值 |
| J1 | Mapped Joint Reference | J0 经 homography 映射到 Depth/PM | 可用但需权重/QC | 映射节点参考，不称无偏 GT |
| R0 | Geometry Seed | 节点、人体轴、体型先验生成 polygon | 仅作预标注 | 几何代理 |
| R1 | OpenCV Proposal | R0 + RGB/IR/Depth/PM 前景/边界细化 | 不直接训练正式模型 | 自动伪标签 |
| R2 | Human Reviewed Reference | 人工 accepted/edited，QC 通过 | 是 | 操作性训练真值 |
| R3 | Double-reviewed Consensus | 双审/仲裁后的高可信子集 | 是，优先评价 | 共识参考集 |
| P0 | Product Ground Truth | 自采同步压力+独立人体/区域真值 | 是 | 产品外部验证真值 |

任何文件必须携带 `label_tier`。训练代码有两套并行路线：

- **路线 A（SLP8 pressure-only GT）**：SLP_8Region_Pressure_VAL_v1.1 通过 adapter 接入，provenance=`V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`，`NOT_REVIEWED`；由 A09R→B01 管道处理，不得混入 R2/R3 tier。
- **路线 B（R2/R3 polygon，历史 HOLD）**：原 `slp_region_annotation_v0.1` 10 区词表，现改为 HOLD，不再作为当前训练入口；若研究 R0/R1 弱监督，必须使用不同 EXP-ID 和明确标记。

## 4. 阶段 I：Region Reference 形成前

### A0：范围、许可与数据冻结

状态：S0 Inventory 已完成；许可仍需核实。

任务：

- 保存数据根目录和版本信息，原始数据只读；
- 核实 SLP 许可条款和团队可接受用途；
- 保存 S0 结构摘要，不上传原始 SLP 到 GitHub；
- 固定两组 quarantine：simLab 00003/00004 `cover2/depthRaw`。

Gate：许可状态可追踪；数据路径、数据版本和 quarantine manifest 明确。

### A1：Frame Index 与跨模态完整配对

建立唯一主键：

```text
slp::{setting}::{subject_id}::{cover_condition}::{frame_index}
```

每行记录：

- RGB/IR/IRraw/Depth/DepthRaw/PM URI；
- 原始文件字节数与可选 SHA-256；
- joints frame index、遮挡标志；
- homography 文件和 hash；
- PM calibration 文件和适用索引；
- 缺失、越界、解码和 quarantine 标志。

不得按目录排序后 `zip()` 静默配对。必须按显式 frame index join，并报告 join 前后行数和覆盖率。

Gate：主键唯一；无意外多对多；缺失全部可定位。

### A2：Homography 与坐标审计

- 明确 `align_PTr_<modality>` 的源坐标、目标坐标和乘法方向；
- 实现齐次坐标变换、除法和 singular/near-singular 检查；
- 生成固定 subject/cover/frame 的 overlay；
- 统计映射后节点越界率、无效率和 RGB↔IR 往返误差；
- J1 映射失败时不补点，进入 quarantine。

Gate：人工 overlay 审查通过；坐标合同冻结；误差报告可复算。

### A3：Canonical SLP Sample 与拆分

Canonical Sample 与标签层分离：

```text
SlpFrame {
  sample_id, setting, subject_id, cover, frame_index,
  modality_uris, homography_refs, calibration_refs,
  quality_flags, provenance
}

JointAnnotation {
  sample_id, coordinate_frame, joints[14], occluded[14],
  label_tier, source_artifact, transform_hash
}

RegionAnnotation {
  sample_id, coordinate_frame, region_id, polygons,
  label_tier, label_source, review_status,
  algorithm_version, transform_hash, reviewer, confidence
}
```

- Split 只存独立 manifest，不写入样本本体；
- 同一受试者的全部模态、遮盖条件、帧进入同一 split；
- `simLab` 作为独立场景层，不与 danaLab 随机逐帧混合。

Gate：schema、adapter、split tests 全部通过，模型尚未读取 TEST。

### A4：节点/人体轴基线

在区域标签冻结前允许做：

- 14 节点可视化和完整性统计；
- 肩中心、髋中心、躯干轴、人体方向和 bbox；
- J0 上的 RGB/IR 轻量节点 baseline；
- J1 上的 PM/Depth 仅作派生标签可行性分析。

不允许宣称身体区域分割已完成。

### A5：Region Ontology v1.0

首版只保留能影响支撑区域决策的粗类别：

```text
head_neck
shoulder_left / shoulder_right
thorax_back
abdomen_waist
pelvis_hip
thigh_left / thigh_right
lower_leg_foot_left / lower_leg_foot_right
```

约束：

- `pelvis_hip` 不等于精确臀部；
- `abdomen_waist` 是床面投影粗区，不是腰椎定位；
- 侧卧时左右肩可重叠，标签需允许 occluded/uncertain；
- 细肘、腕、膝、踝先保留节点，不作为第一版区域分割目标。

### A6：OpenCV 区域预标注器

#### A6.1 输入优先级

1. J0 原始 RGB/IR 节点；
2. DepthRaw 或 Depth 的几何前景；
3. IR/IRraw 的温度结构；
4. RGB，主要用于 uncover 和人工复核；
5. PM 仅作接触区域辅助，不能决定非接触身体部位不存在。

#### A6.2 算法模块

```text
FramePairLoader
→ HomographyProjector
→ JointQualityGate
→ BodyAxisEstimator
→ GeometryRegionSeeder
→ ForegroundProposal
→ RegionBoundaryRefiner
→ TopologyValidator
→ OverlayRenderer
→ AnnotationExporter
```

建议 OpenCV 方法：

- Depth：有效深度范围、床面/背景差、连通域、形态学 closing/opening；
- RGB uncover：由骨骼 polygon 生成 sure foreground/background seeds，再用 GrabCut；
- IR：CLAHE/平滑只用于辅助轮廓，不用于改变原始训练输入；
- covered：以节点几何为主，视觉轮廓为低权重辅助，因为轮廓包含被褥；
- 区域分割：骨段 capsule + 肩/髋宽度先验 + body mask 约束；
- 后处理：polygon clip、自交修复、面积/邻接/左右一致性检查。

严禁：

- 仅凭肤色或固定像素高度划腰、臀；
- 用 PM 零值删除身体区域；
- 强 Gaussian blur 制造平滑人体形状；
- 让算法覆盖原始 proposal 或人工版本；
- 用 OpenCV proposal 直接作为最终模型测试真值。

#### A6.3 输出

每次生成不可变版本：

```text
annotation_id
sample_id
region_id
coordinate_frame
proposal_polygon
refined_polygon
label_tier=R1
algorithm_version
parameter_hash
joint_source_tier
homography_hash
alignment_confidence
anatomical_confidence
quality_flags
```

### A7：人工复核工具与流程

每个任务页至少显示：

- RGB、IR、Depth、PM 四联图；
- 节点、body mask、R0 几何区、R1 OpenCV 区叠加；
- 原始/目标坐标系和映射质量；
- `accept / edit / reject / uncertain`；
- reason codes：alignment、occlusion、blanket contour、joint error、region ambiguity、other。

人工修改必须新增版本，不覆盖 R1。导出 R2 时保存 reviewer、时间、工具版本和修改前后 polygon。

### A8：Pilot、双审和 Region Reference Gate

Pilot 在 A1/A2 后选择 12–20 名受试者，覆盖：

- danaLab/simLab（仅适用模态）；
- uncover/cover1/cover2；
- 身高、体重、腰围、臀围分层；
- 不同人体方向和节点遮挡程度；
- 所有目标区域。

流程：

1. 小批预标注；
2. 两名复核者在校准子集上独立审阅；
3. 统计区域 IoU、边界距离、accept/edit/reject 率和耗时；
4. 冻结 ontology 和操作手册；
5. 扩量后保持分层双审抽检；
6. 争议样本仲裁，不多数票静默覆盖。

Gate R1 必须同时满足：

- R2/R3 主键唯一、版本可追踪；
- 无跨 split 标签泄漏；
- 每区域/遮盖/姿态/体型覆盖报告齐全；
- 审阅者一致性达到 Pilot 后冻结的阈值；
- reject/uncertain 样本不进入默认训练集；
- 质量报告和 manifest SHA-256 完成；
- Reviewer 接受 `SLP Region Reference v1.0`。

## 5. 阶段 II：Region Reference 形成后

### B0：冻结区域数据集

**路线 A（SLP8 GT，A09R 管道）**：由 B01 任务执行；输入为 SLP_8Region_Pressure_VAL_v1.1（4,590 samples，102 danaLab），adapter 已就绪。 B01 已构建并冻结 pressure-only 8-region 训练/验证/测试数据入口（`slp8_training_tables_v0.1`），含：

* TRAIN/VAL/TEST manifests（CSV + JSONL，共 4,590 samples，subject overlap = 0）
* 顶层 freeze manifest（content-addressed；A06 SHA `024f5abe...`、SLP8 source manifest SHA、每 split manifest SHA、TRAIN-only normalization SHA）
* TRAIN-only normalization（`raw_pmaray_response`，**NOT kPa**；fit_split = train；method = `raw_passthrough_with_minmax_reference`；epsilon `1e-12`；全部压力值 finite）
* 数据卡（8-region、`V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`、`NOT_REVIEWED`、danaLab only、uncover only、禁止结论）
* TEST 防泄漏合同：`enable_test_access(purpose="final_evaluation")` 显式开启才允许读取 TEST label/onehot 或计算 TEST 类别统计；结构性检查（行数 / 主体数 / sample_id 唯一 / 路径 / 文件存在 / hash）默认允许。

详见 [S2_B01 阶段报告](stage_reports/S2_B01_SLP8_TRAINING_TABLE_FREEZE_v0.1.md)。B01 已以 `DONE_WITH_LIMITATIONS` 通过 Codex 验收；B02 非学习区域基线也已以 `DONE_WITH_LIMITATIONS` 验收，B03 PM-only Smoke 保持 `READY`，B04 仅由 B03 阻塞。

**路线 B（R2/R3 polygon，原 HOLD 路线）**：A09R 后已改为 HOLD；如未来重新打开：输入仅 R2/R3；R0/R1 单独弱监督实验；固定 Train/VAL/Test subjects；生成 dataset card、类别/区域覆盖、遮盖分层和版本 hash；测试集标签在模型和规则冻结前保持不可见。

### B1：非学习基线

当前路线 A 使用 SLP8 GT + A06 split 建立 pressure-only 非学习基线；以下 polygon/节点基线仅属于未来可选路线 B，不得混入当前训练合同。

- 纯节点几何区域；
- 人体轴分段区域；
- PM contact mask 与几何区交集；
- 报告 region IoU/Dice/中心误差，为神经网络提供最低比较线。

### B2：单模态 Mini

当前路线 A 先运行 PM-only；Depth/IR/RGB 属于需要独立标签与对齐合同的可选路线，不阻塞 pressure-only 主线：

- PM-only：产品相关主线；
- Depth-only：几何性能上限；
- IR-only：遮盖和弱光参考；
- RGB-only：uncover 参考。

Mini 只检查可学习性、吞吐、显存、checkpoint、resume、reload 和标签链路，不排名。

### B3：单模态 Full 候选

- 每模态保留 1–2 个候选；
- 相同 subject folds、训练预算和 region reference；
- 指标：macro region IoU/Dice、中心误差、逐区域、逐遮盖、逐受试者、worst-subject；
- 同时报告 inference p50/p95、模型大小和失败案例。

### B4：遮盖压力测试（当前 HOLD）

SLP8 GT 仅含 uncover；cover1/cover2 与 cross-cover 在获得对应参考 GT 前保持 HOLD。若未来重新打开：

- uncover、cover1、cover2 分开报告；
- uncover train → covered test；
- mixed-cover train → 各条件 test；
- 关注 shoulder、abdomen_waist、pelvis_hip 的下降和拒识率。

### B5：有限融合

仅在单模态证据充分后比较：

- PM + Depth；
- PM + IR；
- 必要时 PM + Depth + IR。

先 late fusion；增益若小于预注册 margin，不升级复杂架构。禁止默认全排列搜索。

### B6：不确定性与拒识

- alignment 低置信、区域歧义、严重遮盖、OOD 进入 REJECT；
- 选择阈值只用 TRAIN/VAL；
- 报告 coverage-risk、逐受试者覆盖、最差条件和高置信错误；
- 区域未知时输出 `UNKNOWN_REGION`，不强制映射到腰/骨盆。

### B7：SLP Full 公平比较与冻结

- 受试者隔离 OOF 或固定外层测试；
- record/frame 粒度明确；
- 配对差值和置信区间；
- 保存预测、模型、配置、代码 SHA、数据 manifest、环境和 Reviewer 结论；
- 最终只冻结 SLP 研究候选，不宣称产品验证。

### B8：队长清单的可交付接口

SLP 最终交付：

```text
HumanGeometry {
  occupancy_evidence_optional,
  posture_static,
  body_axis,
  body_bbox,
  joint_locations,
  coarse_regions,
  region_confidence,
  cover_condition,
  sensor_quality,
  reject_reason
}
```

它可以接到未来 `SupportPerception`，但不在 SLP 内输出气囊动作。

## 6. 自研数据上的后续阶段

队长清单中的 Mechanical Features、Support State、Intervention Verification 另建产品数据路线：

```text
自研 Raw/Calibration/QC
→ SLP 迁移的人体区域
→ 区域 Load/Area/Q95/Gradient/Neighbor
→ Stable Gate
→ Before/After ΔFeature
→ Target Benefit / Spillover
→ HOLD / NEXT / ROLLBACK / NO_ACTION
```

必须同步记录压力矩阵、气囊状态、动作、稳定窗口、动作后压力和用户/独立真值。没有这些数据，不提前冻结 `FOCAL_OVERLOAD`、`UNDER_SUPPORT` 或动作评分公式。

## 7. 连续开发与审计原则

- 一个任务只解决一个明确问题；代码、真实运行、Reviewer 分开。
- 每个任务使用 TASK-ID；真实实验另用 EXP-ID。
- 任务必须声明输入、输出、测试、禁止事项、停止条件和不能得出的结论。
- Coding Agent 完成实现和 Smoke 后停止；不能自行启动昂贵 Full。
- Experiment Runner 只运行冻结配置；失败时保留 FAILED 产物。
- Reviewer 独立重算关键指标和抽查原始 overlay。
- 每完成 3–5 个任务或一个 Gate，由 Codex 做一次横向审计；Region Reference 冻结和 Full 结束时做总审计。

详细任务见 `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`。

## 8. 私密 GitHub 与网页 GPT/Claude Code

仓库上传范围：

- 上传：代码、测试、配置模板、路线、任务、阶段报告、脱敏小型摘要；
- 不上传：SLP 原始数据、解压包、绝对本地路径、`paths.local.json`、密钥、令牌、大 checkpoint、逐帧隐私图像；
- 小型 overlay 如需上传，必须确认数据许可和隐私，并放入明确的 sanitized samples 目录。

当前 GitHub App 在 ChatGPT 中适合读取和分析仓库；写入、提交和推送应由本地 Codex/Claude Code 在明确授权和分支规则下完成。网页 GPT 每次讨论前应先读取：

1. `README.md`
2. `AGENTS.md`
3. `docs/PROJECT_STATUS.md`
4. 本计划
5. Agent 任务清单
6. 当前阶段报告和目标 TASK-ID

## 9. 总体验收

本路线完成不等于完整感知层完成。SLP 路线验收是：

- 关节和区域证据链可信；
- 遮盖条件下性能已量化；
- 区域参考标签可复核；
- 模型具备受试者级评价和拒识；
- 输出可以迁移到自研顶垫的数据合同。

完整感知层验收仍需队长清单 Stage A–F 在自研压力与气囊数据上逐层完成。
