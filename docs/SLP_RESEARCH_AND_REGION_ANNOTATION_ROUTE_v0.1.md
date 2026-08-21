# SLP 研究与身体区域标注路线 v0.1

> 状态说明（2026-08-21）：本文件保留首轮路线。后续连续开发、队长《感知层算法开发清单 V1.0》映射、OpenCV+人工复核标签 Gate 及两阶段任务，以 [SLP 两阶段连续开发总计划 v0.2](SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md) 和 [SLP Agent 任务清单 v0.1](SLP_AGENT_TASK_BACKLOG_v0.1.md) 为准。

## 1. 当前结论

SLP 正式进入准备阶段，但研究证据必须拆成两条独立主线：

1. **关节定位主线**：RGB/IR 的 14 个骨骼节点是数据集直接提供的人工标注；映射到 Depth/PM 的节点属于单应性派生标签，可能含映射偏差。
2. **身体区域主线**：SLP 没有腰、臀、胸、腹等像素级区域真值。区域标签只能按“节点几何生成 → OpenCV/图像辅助细化 → 人工复核”建立，并始终保留派生来源和不确定性。

PoPu 的受试者隔离、Manifest、冻结协议、Runner、checkpoint 和证据归档框架可以复用；PoPu 的标签、模型输出头和数值结论不能直接迁移到 SLP。

## 2. 已核实的数据边界

| 项目 | 当前本地证据 |
|---|---|
| 场景 | `danaLab`、`simLab` |
| 受试者 | README 与目录分别显示 102、7 名；最终以 S0 Inventory 为准 |
| 遮盖条件 | `uncover`、`cover1`、`cover2` |
| 每名受试者每种条件 | 45 个对应姿态帧；三种遮盖条件分别保存相同编号序列 |
| danaLab 模态 | PM、RGB、IR/IRraw、Depth/DepthRaw |
| simLab 模态 | RGB、IR/IRraw、Depth/DepthRaw；目录中没有逐帧 PM |
| 原始人工关节真值 | `joints_gt_RGB.mat`、`joints_gt_IR.mat`，shape=`3×14×45` |
| 跨模态映射 | `align_PTr_<modality>.npy` 的 `3×3` 单应矩阵 |
| 压力校准 | danaLab `PMcali.npy`，shape=`3×45` |
| 区域真值 | 不存在 |

重要限制：README 明确说明 RGB/IR 为人工节点标注，其他模态由对齐映射生成，可能产生偏差。SLP 的图片版 PM 又按单帧范围归一化，不能默认等同于未经归一化的绝对压力；压力物理量使用方式必须在 S1 单独核实。

## 3. 任务拆分

| 任务 | 输入 | 监督 | 主要用途 | 证据地位 |
|---|---|---|---|---|
| J1 RGB/IR 关节定位 | RGB 或 IR | 原始人工 14 节点 | 建立视觉模态基线 | 原始数据集真值 |
| J2 Depth/PM 关节定位 | Depth 或 PM | 单应性映射节点 | 压力/深度空间定位 | 派生真值，需量化映射误差 |
| R1 粗身体区域 | 节点 + 体型数据 | 几何模板区域 | 区域可行性与人工预标注 | 代理标签 |
| R2 图像辅助区域 | R1 + RGB/IR/Depth/PM | OpenCV 轮廓/前景细化 | 减少人工描绘工作 | 自动伪标签 |
| R3 人工复核区域 | R2 overlay | 接受/修改/拒绝 | 建立可训练区域集 | 人工复核派生标签，不等于独立解剖真值 |
| Z1 顶垫 Zone 映射 | R3 + 自研传感器坐标 | 产品坐标合同 | 后续气囊区域输入 | 必须在自采数据上重新验证 |

姿态分类、关节定位、身体区域分割和产品 Zone 映射分别评价，禁止压成一个 Accuracy。

## 4. 分阶段推进路线

### S0：全量 Inventory 与许可门

- 扫描 `setting × subject × cover × modality` 文件覆盖率、帧号连续性和异常文件。
- 审计 RGB/IR 节点、单应矩阵、压力校准文件和体型数据完整性。
- 明确 SLP 的许可、引用和允许用途；在许可未核实前仅作内部非商业研究。
- 产物：模态 Inventory、标注 Inventory、数据质量摘要。
- Gate：样本主键与缺失率可解释，原始数据保持只读。

### S1：跨模态配对和坐标审计

- 冻结主键：`setting/subject/cover/frame`。
- 检查 45 帧在各模态是否一一覆盖；严禁仅按排序静默配对。
- 验证单应矩阵方向、齐次坐标除法、边界点和逆变换数值稳定性。
- 抽取固定受试者和固定帧，生成 RGB/IR/Depth/PM 的关节 overlay。
- 统计映射后越界率、RGB↔IR 往返误差、不同遮盖条件的覆盖率。
- Gate：映射失败样本进入 quarantine；不插值伪造缺帧。

### S2：冻结 Canonical Sample 与受试者拆分

- 建立 SLP Adapter，不让模型直接依赖原始目录命名。
- Canonical Sample 至少包含 source、setting、subject、cover、frame、modality URI、原始/派生标注来源、homography 版本和质量标记。
- 在任何模型结果出现前冻结 subject-level Train/VAL/Test；`danaLab` 和 `simLab` 分层管理。
- 训练统计只在 Train 内拟合；同一受试者的全部模态、遮盖条件和帧必须进入同一 split。

### S3：关节定位单模态基线

- 先跑 RGB-only、IR-only、Depth-only、PM-only；每种模态只保留 1–2 个轻量候选。
- RGB/IR 用原始人工节点评价；Depth/PM 结果必须同时报告“映射标签误差边界”。
- 指标：PCK/PCKh、归一化关节距离、逐关节误差、逐受试者误差、最差受试者和遮盖条件分层结果。
- 只有物理坐标与标定可信时才报告毫米误差；否则不得把像素误差换算成毫米。

### S4：身体区域伪标签 Pilot

- 先选固定的小规模 Pilot，覆盖不同体型、三种遮盖条件和三种卧姿。
- 从节点和体型描述生成粗区域，再用 OpenCV/图像信号细化。
- 人工复核所有 Pilot；根据拒绝原因修正规则，不直接扩到全量。
- Gate：区域定义一致、复核界面可追踪、关键区域通过率达到预先冻结标准后才扩量。

### S5：遮盖压力测试

- 分别报告 `uncover`、`cover1`、`cover2`。
- 增加 `uncover train → covered test` 和混合遮盖训练的泛化比较。
- 重点报告肩、胸、腰/腹、骨盆/臀、膝等区域/节点的差异，不只报告全局平均。

### S6：有限多模态融合

- 先比较 Pressure+Depth、Pressure+IR；只有明确增益才进入三模态。
- 首选可解释的 late fusion；中间特征融合需由单模态结果证明必要性。
- 不做“全部模态 × 全部模型 × 全部超参数”的笛卡尔积搜索。

### S7：Full 公平比较

- 固定受试者 folds、样本、质量门、训练预算、评价粒度和随机种子。
- Full 仅进入每条任务线最好的 1–2 个候选及简单基线。
- 使用受试者级配对差值和置信区间；保留 OOF 预测、逐关节/逐区域、逐遮盖、逐受试者结果。
- Reviewer 接受前不冻结 SLP 候选。

### S8：自研顶垫外部验证

- SLP 只提供公开数据方法学与候选证据，不证明自研传感器、硬件密度、气囊控制或舒适性。
- 最终需要自采压力、RGB-D/其他独立真值、传感器物理坐标和时间同步数据。

## 5. 区域标签生成方案

### 5.1 区域词表 v0.1

建议首版只保留可由14节点稳定约束的粗区域：

- `head`
- `chest_upper_torso`
- `abdomen_waist`
- `pelvis_hip`
- `left/right_upper_arm`
- `left/right_lower_arm`
- `left/right_upper_leg`
- `left/right_lower_leg`

“臀”在二维俯视和盖被条件下不能从髋节点直接获得独立边界。首版使用 `pelvis_hip`，不要宣称精确臀部解剖分割；如业务必须拆出 `buttock`，必须增加人工标注规则和不确定性字段。

### 5.2 几何预标注

- 肩中心、髋中心和躯干轴由左右肩/髋节点计算。
- 胸、腰腹、骨盆沿肩中心到髋中心的轴向比例划分。
- 上下肢使用节点连线的 capsule/polygon；宽度由肩宽、髋宽、身高和体型数据提供先验。
- 关节遮挡标记、越界、左右交叉或骨段长度异常时降低置信度或拒绝生成。

这些区域是可重复的几何代理，不是像素级人体解剖真值。

### 5.3 OpenCV/图像辅助细化

OpenCV 的职责是边界和质量辅助，不能自行推断“这里就是腰或臀”：

1. 将 RGB/IR/Depth 对齐到目标参考坐标；保存使用的矩阵与方向。
2. 以骨骼 capsule 作为前景种子，结合 Depth 前景、IR 温度结构、GrabCut/形态学和连通域获得身体/被褥轮廓候选。
3. 用轮廓约束几何区域，保留原始 polygon 与细化 polygon。
4. 对 PM 标签额外记录 pressure-contact mask；不能把无压力区域自动判为“没有该身体部位”。
5. 生成统一 overlay，供人工接受、修改或拒绝。

盖被后的视觉轮廓包含被子而非真实体表，因此 cover1/cover2 的 OpenCV 细化默认置信度低于 uncover。

### 5.4 人工复核状态

每个区域标签必须保存：

- `label_source`: `joint_geometry` / `opencv_refined` / `human_edited`
- `joint_gt_source`: `manual_rgb` / `manual_ir` / `homography_derived`
- `review_status`: `pending` / `accepted` / `edited` / `rejected`
- `reviewer_id`、`reviewed_at`、`reason_code`
- 原始和修订 polygon、算法版本、参数版本、homography hash
- `anatomical_confidence` 与 `alignment_confidence`

质量抽检建议：Pilot 全量双人复核；扩量后至少对每个遮盖条件、卧姿和体型分层抽取固定比例双审，并报告一致性。争议样本进入 adjudication，不以多数票静默覆盖。

## 6. 训练与速度策略

- S0/S1 在 Windows 本地完成，不占 GPU。
- SLP 小文件多，先测 DataLoader 吞吐；训练前生成版本化索引和只读缓存/分片，禁止反复上传全部小文件。
- Mini 仅验证链路和学习信号；Full 前冻结候选，避免无边界搜索。
- 图像缓存、节点标签、区域伪标签分别版本化；任何标签修改不得覆盖历史版本。

## 7. 当前执行顺序

1. 运行 S0 全量 Inventory。
2. 根据缺失和异常冻结 S1 配对审计范围。
3. 建立 12–20 名受试者的区域标注 Pilot 清单，但暂不自动生成全量区域。
4. 完成 overlay 与人工复核合同后，再实现 OpenCV 区域预标注器。
5. 关节单模态基线和区域 Pilot 分两条任务线推进。

## 8. 当前不能得出的结论

- SLP 没有现成腰、臀等身体区域真值。
- 单应映射后的 PM/Depth 节点不等同于无偏人工标注。
- OpenCV 自动轮廓不等同于人体解剖区域。
- 人工复核伪标签仍不是独立医学级真值。
- SLP 结果不能直接证明自研顶垫分区、气囊动作或舒适性有效。
