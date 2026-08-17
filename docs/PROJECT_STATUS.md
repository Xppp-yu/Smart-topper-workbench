# Smart Topper Windows Research Workbench — 项目状态

最后更新：2026-08-17  
状态口径：只把已运行且存在可追溯产物的步骤标记为 `COMPLETE`；代码、计划或空目录不等于完成。

## 当前一句话状态

PoPu Tactilus 的数据接入、全量结构盘点、质量策略和首次 Geometry 提取已完成；P3.1 的 Mask 策略比较与 P3.2 的区域标注—压力记录对齐审计代码已就绪、尚未运行。当前不能直接训练部位或姿态模型。

## 阶段看板

| 阶段 | 目标 | 状态 | 已有证据 | 下一步门槛 |
|---|---|---|---|---|
| P0 | Windows 环境、数据路径与最小读取链路 | COMPLETE | 健康检查、真实 PoPu 图、PMD 可读性检查 | 数据版本和研究任务可被定位 |
| P1/R1 | 全量 PoPu Tactilus Inventory | COMPLETE | [P1 阶段报告](stage_reports/P1_POPU_TACTILUS_INVENTORY_v0.1.md)、CSV、JSON、标签分布图 | 明确有标签集和未标注 `others.json` 的用途 |
| P2/R2 | 样本画廊与 `ACCEPT/WARN/REJECT` 质量门 | COMPLETE | [P2 阶段报告](stage_reports/P2_POPU_TACTILUS_QUALITY_GATE_v0.1.md)、质量 CSV、两张画廊图；`5,006 ACCEPT / 94 WARN / 60 EXCLUDED / 0 REJECT` | WARN 暂保留，后续比较全量有标签与仅 ACCEPT 两个口径 |
| P3/R3 | 接触 Mask 与 Geometry | PARTIAL | [P3 阶段报告](stage_reports/P3_POPU_CONTACT_MASK_AND_GEOMETRY_v0.1.md)、Geometry CSV、Mask/Geometry 叠加图；无读取拒绝 | P3.1 比较 Mask 策略并冻结一个 Geometry 输入规则 |
| P3.1/R3.1 | Mask 候选策略比较 | READY_TO_RUN | `compare_popu_mask_strategies.py`、单元测试通过；尚无真实数据输出 | 复核稳定性指标和代表性叠加图，不自动按分数冻结 |
| P3.2/R3.2 | COCO 区域标注—压力记录对齐审计 | READY_TO_RUN | `audit_popu_segmentation.py`、单元测试通过；尚无真实数据输出 | 获得可追溯的逐记录、逐帧配对规则；否则区域监督训练保持 HOLD |
| P4a/R4a | 无标签特征表 | BLOCKED_BY_P3.1 | 尚未开始 | Geometry 输入规则冻结；每行样本可追溯，原始/空间/质量特征与标签列分离 |
| P4b/R4b | 区域标签关联与监督特征集 | BLOCKED_BY_P3.2_AND_P4A | 尚未开始 | 有文档可追溯的逐记录、逐帧配对规则；否则不生成区域监督训练集 |
| P5/R5 | 姿态 Baseline 与受试者隔离评价 | BLOCKED_BY_P4A | 尚未开始 | GroupKFold/LOSO、逐样本预测和错误图 |
| P6/R6 | `UNKNOWN/REJECT` 与错误分析 | BLOCKED_BY_P5 | 尚未开始 | 阈值仅由验证受试者选择 |
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
