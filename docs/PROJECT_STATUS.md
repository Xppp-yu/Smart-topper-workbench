# Smart Topper Windows Research Workbench — 项目状态

最后更新：2026-08-18  
状态口径：只把已运行且存在可追溯产物的步骤标记为 `COMPLETE`；代码、计划或空目录不等于完成。

## 当前一句话状态

P5/R5 首轮 Baseline 已完成：在 P4a 特征表上完成受试者隔离的五分类 Baseline（empty/supine/prone/left/right）。协议为 12 个 held-out 受试者 + 开发集 GroupKFold 选型；`logreg` 为当前首轮领先候选（primary test macro-F1 0.9466），dummy 下限 0.20。候选尚未正式冻结——P5 v0.1 对每个候选模型各评估了一次 held-out test；P5.1 的配置驱动模型注册、分组 evaluator、记录聚合框架已实现并通过测试（`READY_TO_RUN — FRAMEWORK_IMPLEMENTED`），全量横向复核运行前不冻结模型或 UNKNOWN 阈值。结果仅为 PoPu 首轮公开数据候选，区域监督继续 HOLD。

## 阶段看板

| 阶段 | 目标 | 状态 | 已有证据 | 下一步门槛 |
|---|---|---|---|---|
| P0 | Windows 环境、数据路径与最小读取链路 | COMPLETE | 健康检查、真实 PoPu 图、PMD 可读性检查 | 数据版本和研究任务可被定位 |
| P1/R1 | 全量 PoPu Tactilus Inventory | COMPLETE | [P1 阶段报告](stage_reports/P1_POPU_TACTILUS_INVENTORY_v0.1.md)、CSV、JSON、标签分布图 | 明确有标签集和未标注 `others.json` 的用途 |
| P2/R2 | 样本画廊与 `ACCEPT/WARN/REJECT` 质量门 | COMPLETE | [P2 阶段报告](stage_reports/P2_POPU_TACTILUS_QUALITY_GATE_v0.1.md)、质量 CSV、两张画廊图；`5,006 ACCEPT / 94 WARN / 60 EXCLUDED / 0 REJECT` | WARN 暂保留，后续比较全量有标签与仅 ACCEPT 两个口径 |
| P3/R3 | 接触 Mask 与 Geometry | COMPLETE | [首轮 P3 报告](stage_reports/P3_POPU_CONTACT_MASK_AND_GEOMETRY_v0.1.md)、[P3.1 冻结报告](stage_reports/P3_1_POPU_MASK_STRATEGY_FREEZE_v0.1.md)、冻结配置与 Geometry v0.2 | P4a 只使用版本化冻结规则；新真值到位时才重新打开 Mask 决策 |
| P3.1/R3.1 | Mask 候选策略比较 | COMPLETE | 三策略 `15,480` 行、`3+3+3` 叠加图；冻结 `largest_component`，v0.2 为 `5,098 OK / 2 WARN / 0 REJECT` | 进入 P4a，并保留 `mask_strategy` 与完整追溯字段 |
| P3.2/R3.2 | COCO 区域标注—压力记录对齐审计 | COMPLETE / SUPERVISION_HOLD | [P3.2 审计报告](stage_reports/P3_2_POPU_SEGMENTATION_ALIGNMENT_AUDIT_v0.1.md)；`1,670` 人体标注一对三歧义，`60` 一对一候选均为空床 | 获得官方映射或独立同步真值前，不生成区域监督训练集 |
| P4a/R4a | 无标签特征表 | COMPLETE | [P4a 阶段报告](stage_reports/P4a_POPU_LABEL_FREE_FEATURES_v0.1.md)、`51,000` 行特征表、primary cohort、EXCLUDED manifest；`40 passed` | 每行样本可追溯，原始/空间/质量特征与标签列分离 |
| P4b/R4b | 区域标签关联与监督特征集 | BLOCKED_BY_MISSING_PAIRING_AND_P4A | P3.2 已证明当前人体标注无法唯一配对 | 有文档可追溯的逐记录、逐帧配对规则；否则不生成区域监督训练集 |
| P5/R5 | 姿态 Baseline 与受试者隔离评价 | COMPLETE — FIRST_ROUND_BASELINE | [P5 阶段报告](stage_reports/P5_POPU_SUBJECT_ISOLATED_POSTURE_BASELINE_v0.1.md)、逐样本预测、混淆矩阵、逐受试者表；当前首轮领先候选=`logreg`，primary test macro-F1 0.9466；`52 passed` | P5.1 横向复核候选；复核通过前不正式冻结模型 |
| P5.1/R5.1 | 横向比较框架修正与候选复核 | READY_TO_RUN — FRAMEWORK_IMPLEMENTED | 配置驱动模型注册、通用分组 evaluator、记录聚合已实现并通过 `31` 条新增测试（全量 `83 passed`）；尚未运行 P5.1 全量比较，候选未冻结 | repeated subject-grouped CV 口径下候选排序与 v0.1 一致或可解释地变化；逐 snapshot/记录/受试者稳定性都报告；复核通过后才冻结候选 |
| P6/R6 | `UNKNOWN/REJECT` 与错误分析 | BLOCKED_BY_P5_1 | P5 逐样本预测（含置信度）+ 当前首轮领先候选 logreg 已就绪，但候选未冻结 | P5.1 复核冻结候选后放行；阈值仅由验证受试者选择 |
| P7/R7 | 降密度、坏点、区域研究 | BLOCKED_BY_P6 | 尚未开始 | 不将软件消融误称为硬件验证 |
| P8/R8 | 冻结候选并移交 WSL 工程化 | BLOCKED_BY_P7 | 尚未开始 | 算法规格、配置、切分和限制完整冻结 |

## 当前数据使用边界

- 固定姿态有标签集：`5,100` 条 JSON 记录；其中四类卧姿各 `1,260` 条，空床 `60` 条。
- `others.json`：每位受试者一条、共 `60` 条，缺少姿态与 variation 标签；保留给后续转身、过渡或 `UNKNOWN` 研究，不伪造为固定姿态训练样本。
- PoPu 与 PMD 是独立公开数据集，不能逐行配对。
- 所有公开数据结果仅证明研究数据链路或算法候选，不证明自研传感器、气囊闭环、整夜稳定性、舒适性或产品效果。

## 如何阅读与更新

1. 每一阶段先看 `docs/stage_reports/` 中对应报告，获得“本阶段做了什么、得到什么、没得到什么、下一步是什么”。
2. 再按报告中的精确路径打开 CSV、JSON、PNG 或模型，查看原始证据。
3. 每次完成一个可运行阶段，新增一份阶段报告并更新本页；失败或暂停也记录原因，不改写历史结论。

具体格式见 [阶段记录与报告约定](STAGE_REPORTING_CONVENTION.md)。
