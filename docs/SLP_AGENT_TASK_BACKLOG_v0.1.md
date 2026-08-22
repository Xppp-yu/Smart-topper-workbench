# SLP Agent 连续开发任务清单 v0.1

## 1. 使用规则

本清单供网页 GPT 做方案讨论、Claude Code/Codex 做定点实现、Experiment Runner 执行冻结实验、Reviewer 独立验收。任何 Agent 每次只领取一个 `TASK-ID`。

任务状态：

- `DONE`：真实产物和验收均存在；
- `READY`：依赖满足，可以开始；
- `BLOCKED`：依赖未满足；
- `HOLD`：方向保留但当前不做。

每个任务的完成条件：

1. 代码/文档变更明确；
2. 定向测试通过；
3. 必要时真实运行并保存产物；
4. 阶段报告区分“已验证/合理推断/尚未验证”；
5. `git diff --check` 通过；
6. 不覆盖历史产物，不修改原始数据；
7. Reviewer 接受后才更新为 DONE。

## 2. 标准任务包模板

```text
TASK-ID:
Objective:
Why now:
Prerequisites:
Inputs:
In scope:
Out of scope:
Files allowed to change:
Data boundary:
Implementation requirements:
Tests/commands:
Expected artifacts:
Acceptance criteria:
Fail-closed conditions:
Prohibited conclusions:
Reviewer checklist:
```

## 3. 阶段 I：Region Reference 形成前

### TASK-SLP-A00：S0 Inventory 与标注边界

- 状态：`DONE_WITH_QUARANTINE`。
- 已有：109 人、1,941 groups、1,939 OK、2 missing depthRaw groups。
- 入口：`docs/stage_reports/S0_SLP_FULL_INVENTORY_AND_ANNOTATION_BOUNDARY_v0.1.md`。

### TASK-SLP-A01：许可与数据版本登记

- 状态：`READY`。
- 目标：找到并记录 SLP 许可、引用、允许用途和数据版本。
- 输入：本地 README、下载来源和官方页面。
- 输出：`docs/stage_reports/S0_1_SLP_LICENSE_AND_DATA_VERSION_v0.1.md`。
- 验收：明确 commercial/non-commercial、再分发、衍生标签和示例图片上传边界。
- Fail-closed：许可不清时，不向 GitHub 上传任何 SLP 图像/原始文件；仅内部研究。

### TASK-SLP-A02：内容可解码与数值 QA

- 状态：`READY`。
- 目标：在不加载全量进内存的情况下逐文件检查解码、shape、dtype、finite、范围。
- 输入：S0 Inventory。
- 输出：content QA CSV、summary、quarantine manifest、测试。
- 检查：RGB/IR/depth PNG，IRraw/depthRaw NPY，PM PNG，MAT/NPY 小型标注。
- 验收：每模态成功率、异常率、shape 分布、数值范围分布可复算。
- 禁止：用内容异常静默删除整名受试者。

### TASK-SLP-A03：Frame Master Index

- 状态：`READY`。
- 目标：按显式 frame index 建立跨模态主表。
- 主键：`setting, subject_id, cover_condition, frame_index`。
- 输出：`slp_frame_index_v0.1.parquet/csv`、join coverage 报告、测试。
- 验收：主键唯一；join 前后行数一致；两组已知 depthRaw 缺失精确保留为空值和 quality flag。
- 禁止：按排序 `zip()`、用图片补造 raw depth。

### TASK-SLP-A04：Homography 数学与方向审计

- 状态：`DIRECTION_CONFIRMED_BY_README_AND_AUDIT_AND_OVERLAY — READY_FOR_REVIEW`。
- 目标：实现和验证 modality↔PM 坐标转换。
- 输出：transform 模块、往返误差表、越界表、固定 overlay、测试。
- 入口：`docs/stage_reports/S1_2_SLP_HOMOGRAPHY_AUDIT_v0.1.md`。
- 已交付：327 / 327 矩阵可逆、max probe round-trip 4.55 × 10⁻¹³、danaLab RGB/IR direct in-bounds 99.28 % / 99.27 %、6 名抽样 overlay、27 / 27 单元测试。
- 验收：矩阵 singular 检查、齐次除法、方向合同和单位说明齐全。
- 停止条件：方向无法由 README/overlay 确认时，标记 BLOCKED，不凭模型分数倒推方向。

### TASK-SLP-A05：Canonical Sample 与 Adapter

- 状态：`IMPLEMENTED_AND_REAL_RUN_COMPLETE — READY_FOR_REVIEW`。
- 目标：实现 SLP Adapter、Frame/Joint/Region 分层对象。
- 输出：schema、adapter、provenance、quarantine、测试。
- 已有：`src/topper_perception/io/slp_canonical.py`、`scripts/build_slp_canonical_samples.py`、`tests/test_slp_canonical_adapter.py`（20 测）、`configs/annotations/slp_canonical_sample_v0.1.schema.json`、`docs/stage_reports/S1_3_SLP_CANONICAL_ADAPTER_v0.1.md`、`docs/stage_reports/SLP_CANONICAL_SAMPLE_EXAMPLES_v0.1.md`；真实 SLP 全量 14,715 canonical sample、quarantine 90（与 A03 / S0 已知 90 个 depthRaw 缺失一致）、traceable_rate 0.9883、A04 几何字段全部保留。
- 入口：`docs/stage_reports/S1_3_SLP_CANONICAL_ADAPTER_v0.1.md`。
- 验收：读取单帧时可回到全部原始 URI 和 transform；标签层不污染基础样本；保留 A04 报告中的 H 方向与 round-trip / in-bounds 字段。
- 禁止：把 split、review status 或模型预测写回原始样本。

### TASK-SLP-A06：受试者拆分冻结

- 状态：`DONE`。
- 目标：在任何模型分数出现前冻结 subject-level split/folds。
- 已有：split manifest JSON、SHA-256 `024f5abe`、19 单元测试、6 isolation tests PASS、62 回归测试 PASS；109 subjects split 81/10/18（danaLab 80/10/10，simLab 0/0/100 TEST held-out）；90 quarantined frames 单独统计，不混入 train/val；确定性复现验证通过。
- 入口：`docs/stage_reports/S1_4_SLP_SUBJECT_SPLIT_FREEZE_v0.1.md`。
- 要求：danaLab/simLab 分层；同一受试者所有模态、遮盖、帧不可跨 split。
- 禁止：根据后续 TEST 分数调整受试者。

### TASK-SLP-A07：节点与遮挡 EDA

- 状态：`BLOCKED_BY_A05_A06`（A05 COMPLETE，A06 COMPLETE，ready to start）。
- 目标：统计14节点坐标、遮挡、越界、骨段长度和异常。
- 输出：逐节点/遮盖/场景/受试者 QA 表与图。
- 验收：J0 和 J1 分开报告；不把映射节点混入原始 GT 汇总。

### TASK-SLP-A08：人体轴与粗几何基线

- 状态：`BLOCKED_BY_A07`。
- 目标：由肩、髋、thorax/head 建立 body axis、bbox 和方向置信度。
- 输出：几何模块、overlay、错误案例、测试。
- 验收：左右翻转、坐标旋转、节点缺失/遮挡均有测试。

### TASK-SLP-A09：Region Ontology 与机器合同

- 状态：`IMPLEMENTED_PENDING_REVIEW`。
- 目标：冻结 coarse region 词表、label tier 和 review 字段。
- 输出：`configs/annotations/slp_region_annotation_v0.1.schema.json`、说明和测试。
- 验收：训练默认只接受 R2/R3；region、coordinate frame、来源和版本必填。

### TASK-SLP-A10：几何 Region Seeder

- 状态：`BLOCKED_BY_A08_A09`。
- 目标：用节点、人体轴和体型先验生成 R0 polygon。
- 输出：head/shoulder/thorax/waist/pelvis/thigh/lower-leg 几何 proposal。
- 验收：确定性、无自交、边界合法、左右一致；缺关键节点时 reject/uncertain。
- 禁止：称为分割真值。

### TASK-SLP-A11：OpenCV Foreground Proposal

- 状态：`BLOCKED_BY_A04_A10`。
- 目标：分别实现 Depth、IR、RGB uncover 的身体/被褥前景候选。
- 输出：可插拔 foreground backend、参数配置、overlay、测试。
- 技术：depth threshold/background、connected components、morphology、GrabCut seed、IR 辅助轮廓。
- 依赖：当前项目环境尚未安装 `cv2`；本任务新增单一可选 `vision` 依赖，优先 `opencv-python-headless`。若后续确需 ArUco/contrib 功能，再改为 `opencv-contrib-python-headless`，两者不得同时安装。
- 验收：原图不覆盖；每 backend 输出 mask+confidence+reason；cover 条件分开评价。
- 禁止：将 blanket contour 当体表真值。

### TASK-SLP-A12：Region Boundary Refiner

- 状态：`BLOCKED_BY_A11`。
- 目标：将 R0 与前景候选融合生成 R1。
- 输出：refined polygon、topology validator、质量标志。
- 验收：proposal/refined 均保留；面积突变、离开人体、区域重叠异常触发 fail/review。

### TASK-SLP-A13：人工复核工具

- 状态：`BLOCKED_BY_A09_A12`。
- 目标：四联图查看、polygon 编辑、accept/edit/reject/uncertain、reason code。
- 输出：本地复核界面或可导入 CVAT/Label Studio 的包；导出符合 schema。
- 验收：不可覆盖历史；reviewer/time/tool version 完整；断点续审。
- 禁止：把受试者可识别图片上传公共服务。

### TASK-SLP-A14：Pilot 采样 Manifest

- 状态：`BLOCKED_BY_A03_A07`。
- 目标：选择12–20人 Pilot，覆盖 setting、cover、体型、方向、遮挡和区域。
- 输出：冻结 manifest 和抽样理由。
- 验收：抽样发生在看区域模型结果前；不以“容易标”替代代表性。

### TASK-SLP-A15：Pilot 自动预标注与参数冻结

- 状态：`BLOCKED_BY_A12_A14`。
- 目标：运行 R0→R1，不做全量。
- 输出：R1 annotations、overlay、失败率、每帧耗时、参数 hash。
- 验收：所有失败保留；不同 cover 分开统计。

### TASK-SLP-A16：Pilot 人工复核与一致性

- 状态：`BLOCKED_BY_A13_A15`。
- 目标：校准复核者并形成 R2/R3。
- 输出：accept/edit/reject、IoU/边界差、耗时和争议报告。
- 验收：双审阈值在 Pilot 后冻结；低一致区域可降级或合并词表。

### TASK-SLP-A17：Region Reference v1.0 Freeze

- 状态：`BLOCKED_BY_A16`。
- 目标：冻结操作性训练真值。
- 输出：manifest、schema version、SHA-256、dataset card、QC 报告。
- Gate：Reviewer 接受后，阶段 II 才可启动区域模型。
- 禁止：把 R1 未审标签混入默认 TEST。

### TASK-SLP-A18：节点定位轻量基线

- 状态：`BLOCKED_BY_A05_A06`（A05 COMPLETE，A06 COMPLETE，can start）。
- 目标：验证 RGB/IR J0 与 PM/Depth J1 的学习链路。
- 输出：Smoke/Mini，不做区域分割结论。
- 验收：PCK/PCKh/normalized error，逐节点/遮盖/受试者报告。

## 4. 阶段 II：Region Reference 形成后

### TASK-SLP-B01：冻结区域训练表

- 状态：`BLOCKED_BY_A17`。
- 输入：R2/R3 + A06 split。
- 输出：训练/验证/测试 manifest、类别覆盖、数据卡。
- 验收：TEST reference 不被开发代码读取；R0/R1 单独标记。

### TASK-SLP-B02：非学习区域基线

- 状态：`BLOCKED_BY_B01`。
- 目标：节点几何、人体轴分段、PM contact-intersection 三种基线。
- 验收：使用同一 R2/R3 和同一 split；保留失败案例。

### TASK-SLP-B03：单模态 Region Smoke

- 状态：`BLOCKED_BY_B01`。
- 模态：PM、Depth、IR、RGB。
- 目标：验证数据吞吐、loss、输出、checkpoint/resume/reload。
- 禁止：Smoke 分数排名。

### TASK-SLP-B04：单模态 Region Mini

- 状态：`BLOCKED_BY_B02_B03`。
- 目标：每模态保留最多1–2个可行候选。
- 指标：region IoU/Dice、中心误差、逐区域、逐 cover、worst subject。
- Gate：只淘汰不可行候选，不用小样本宣布冠军。

### TASK-SLP-B05：遮盖条件压力测试

- 状态：`BLOCKED_BY_B04`。
- 比较：uncover、cover1、cover2、cross-cover。
- 输出：区域性能下降、拒识率和错误案例。

### TASK-SLP-B06：有限融合 Mini

- 状态：`BLOCKED_BY_B04_B05`。
- 候选：PM+Depth、PM+IR；三模态需额外理由。
- Gate：增益小于冻结 margin 时保留简单模型。

### TASK-SLP-B07：Full 协议冻结

- 状态：`BLOCKED_BY_B04_B06`。
- 目标：冻结 folds、候选、预算、指标、选择规则、资源和停止条件。
- 禁止：协议与 Full 运行同一任务完成。

### TASK-SLP-B08：Full Runner 与一折预检

- 状态：`BLOCKED_BY_B07`。
- 目标：实现受治理 runner；先做单折时间/显存预算。
- Gate：Reviewer 接受预检后才运行 Full。

### TASK-SLP-B09：Full 公平比较

- 状态：`BLOCKED_BY_B08`。
- 目标：真实 subject-isolated Full。
- 输出：OOF/TEST predictions、逐区域/遮盖/受试者、模型和日志。
- 停止：OOM、NaN、split leak、manifest mismatch、reload mismatch。

### TASK-SLP-B10：UNKNOWN/REJECT

- 状态：`BLOCKED_BY_B09`。
- 目标：区域不确定、alignment failure、OOD 和低置信拒识。
- 输出：coverage-risk、最差受试者、高置信错误。

### TASK-SLP-B11：SLP 研究候选冻结

- 状态：`BLOCKED_BY_B09_B10`。
- 目标：冻结模型族、接口、限制和自研数据迁移合同。
- 禁止：宣称气囊控制、舒适性或产品效果。

## 5. 产品数据后续任务，不属于 SLP

| TASK-ID | 内容 | 必需数据 |
|---|---|---|
| TASK-PROD-C01 | Canonical Pressure Frame、baseline、坏点、饱和 | 自研原始矩阵与校准 |
| TASK-PROD-C02 | Stable/MOVING/SETTLING | 连续 20 Hz 或实际采样流 |
| TASK-PROD-C03 | 区域 Load/Ratio/Area/Q95/Gradient | 已标定压力 + SLP迁移区域 |
| TASK-PROD-C04 | Overload/Under-support 等候选状态 | 独立实验标签和个体 baseline |
| TASK-PROD-C05 | Before/After、Spillover、COP shift | 同步气囊动作日志 |
| TASK-PROD-C06 | Benefit/Penalty/HOLD/ROLLBACK | 多轮干预与安全审查 |

## 6. 建议的并行方式

在不造成依赖冲突时：

```text
Lane 1 数据：A02 → A03 → A04 → A05 → A06
Lane 2 标签：A09 → A10 → A11 → A12 → A13
Lane 3 审核：A14 → A15 → A16 → A17
Lane 4 节点模型：A07 → A08 → A18
```

并行规则：Lane 2 可先写合同和单元测试，但真实区域生成必须等待 A04/A08；Lane 4 可以在区域 Gate 前推进，但不得提前做区域模型。

## 7. 每轮交接给网页 GPT/Claude Code 的最小上下文

```text
请先读取 AGENTS.md、docs/PROJECT_STATUS.md、
docs/SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md、
docs/SLP_AGENT_TASK_BACKLOG_v0.1.md 和当前 TASK-ID 的前置报告。

本轮只处理 TASK-SLP-XXX；不要运行未授权 Full；
不要修改原始数据；不要把 R0/R1 称为真值；
结束时交付代码、测试结果、产物路径、已验证/未验证和下一 Gate。
```
