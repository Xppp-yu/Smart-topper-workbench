# Smart Topper Windows Research Workbench — 项目状态

最后更新：2026-08-20
状态口径：只把已运行且存在可追溯产物的步骤标记为 `COMPLETE`；代码、计划或空目录不等于完成。

## 当前一句话状态

P5.2-C Full 公平比较已完成并经 Reviewer 接受：`EXP-P5.2-C-FULL-COMPARISON-20260820-R01` 在 AutoDL RTX 4090 上完成 3 模型 × 3 repeats × 5 folds 共 45 个训练单元；独立复核确认结果与选择规则，`small_resnet` 以 record macro-F1 `0.986649 ± 0.002832`、balanced accuracy `0.986636` 成为 PoPu 固定睡姿五分类总体研究候选模型族。P5.1 `calibrated_linear_svm` 保留为传统模型对照。P6 `UNKNOWN/REJECT` 与错误分析现已放行但尚未开始；结果仍仅为 PoPu 公开数据研究证据，区域监督继续 HOLD。

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
| P5/R5 | 姿态 Baseline 与受试者隔离评价 | COMPLETE — FIRST_ROUND_BASELINE | [P5 阶段报告](stage_reports/P5_POPU_SUBJECT_ISOLATED_POSTURE_BASELINE_v0.1.md)、逐样本预测、混淆矩阵、逐受试者表；首轮领先候选=`logreg`（历史证据），primary test macro-F1 0.9466；`52 passed` | 已被 P5.1 满足（候选在 repeated grouped CV 口径下复核并冻结） |
| P5.1/R5.1 | 横向比较框架修正与候选复核 | COMPLETE — TRADITIONAL_CANDIDATE_FROZEN | [P5.1 阶段报告](stage_reports/P5_1_POPU_GROUPED_MODEL_COMPARISON_v0.1.md)、OOF 1,051,260 行、record 105,126 行、逐受试者/逐类别/消融表、混淆矩阵与图；winner=`calibrated_linear_svm`（record macro-F1 0.9452），冻结 `popu_research_candidate_p5_1_v0.1`（16,097 B，独立重载 smoke OK）；`115 passed` | 传统模型候选已冻结，不覆盖 CNN；进入 P5.2 神经网络公平比较，总体候选冻结后才放行 P6 |
| P5.2-A/R5.2-A | CNN 训练底座与 Smoke | COMPLETE — CPU_CUDA_SMOKE_PASS | [P5.2-A 阶段报告](stage_reports/P5_2_A_POPU_NEURAL_CPU_CUDA_SMOKE_v0.1.md)；CPU `EXP-P5.2-A2-CPU-SMOKE-20260819-R02` 与 CUDA `EXP-P5.2-A2-CUDA-SMOKE-20260819-R02` 均通过。CUDA R02：干净 Git SHA `5803f5c`、RTX 4090、PyTorch `2.8.0+cu128`、1,000 样本、1 epoch；三模型训练、checkpoint/resume、参数变化、独立重载一致、固定 seed 复现均通过；`271 passed` | P5.2-B 必须另行冻结 Mini 配置并授权；当前 Smoke 指标不排名；不跑 Full CV |
| P5.2-B/R5.2-B | Mini 筛选 | COMPLETE — MINI_ACCEPTED | [P5.2-B 结果报告](stage_reports/P5_2_B_POPU_NEURAL_MINI_RESULTS_v0.1.md)、[P5.2-B 协议报告](stage_reports/P5_2_B_POPU_NEURAL_MINI_PROTOCOL_v0.1.md)；`EXP-P5.2-B-MINI-SCREEN-20260819-R01`（git `0261113`、RTX 4090、`device=cuda`）`SUCCEEDED`；三候选 `matrix_mlp`/`tiny_cnn`/`small_resnet` Gate 均 `proceed`；Reviewer record-level 独立重算通过；`reproducible_seed=true`、`overall_verdict=proceed` | 三候选都进入 P5.2-C Full；先冻结 Full 公平比较协议与配置，再实现与授权；Mini 不排名 |
| P5.2-C/R5.2-C | Full 公平比较 | COMPLETE — SMALL_RESNET_ACCEPTED | [P5.2-C 结果与验收](stage_reports/P5_2_C_POPU_NEURAL_FULL_RESULTS_v0.1.md)、[协议](stage_reports/P5_2_C_POPU_NEURAL_FULL_PROTOCOL_v0.1.md)；45/45 单元完成，最终 `SUCCEEDED`；Reviewer 独立复核完整性、split/OOF 覆盖、指标和选择规则；winner=`small_resnet`，record macro-F1 `0.986649 ± 0.002832`，相对 SVM `+0.041481` | 总体研究候选模型族已冻结；P6 仅从开发 OOF 证据选择 UNKNOWN/REJECT 阈值并完成错误分析 |
| P6/R6 | `UNKNOWN/REJECT` 与错误分析 | READY — NOT_STARTED | P5.2-C 已接受 `small_resnet`；完整 Full OOF record 概率和 Reviewer 决策已就绪 | 先冻结 P6 协议；阈值仅由开发 OOF 受试者选择；报告覆盖空床、低置信与高置信错误，禁止把阈值当作外部/产品验证 |
| P7/R7 | 降密度、坏点、区域研究 | BLOCKED_BY_P6 | 尚未开始 | 不将软件消融误称为硬件验证 |
| P8/R8 | 冻结候选并移交 WSL 工程化 | BLOCKED_BY_P7 | 尚未开始 | 算法规格、配置、切分和限制完整冻结 |

## 当前数据使用边界

- 固定姿态有标签集：`5,100` 条 JSON 记录；其中四类卧姿各 `1,260` 条，空床 `60` 条。
- `others.json`：每位受试者一条、共 `60` 条，缺少姿态与 variation 标签；保留给后续转身、过渡或 `UNKNOWN` 研究，不伪造为固定姿态训练样本。
- PoPu 与 PMD 是独立公开数据集，不能逐行配对。
- 所有公开数据结果仅证明研究数据链路或算法候选，不证明自研传感器、气囊闭环、整夜稳定性、舒适性或产品效果。

## 治理口径

四层角色（Controller / Coding Agent / Experiment Runner / Reviewer）、`TASK-ID` / `EXP-ID`、`QUEUED` 后参数不可变，以及“Coding Agent 通过 Smoke 后即停止、不陪跑 Mini/Full”的边界，见 [实验治理与远程 GPU 执行方案](EXPERIMENT_GOVERNANCE_AND_GPU_EXECUTION_PLAN_v0.1.md) 与 [Codex × Claude Code 协作约定](../COLLABORATION_WORKFLOW.md)。

## 如何阅读与更新

1. 每一阶段先看 `docs/stage_reports/` 中对应报告，获得“本阶段做了什么、得到什么、没得到什么、下一步是什么”。
2. 再按报告中的精确路径打开 CSV、JSON、PNG 或模型，查看原始证据。
3. 每次完成一个可运行阶段，新增一份阶段报告并更新本页；失败或暂停也记录原因，不改写历史结论。

具体格式见 [阶段记录与报告约定](STAGE_REPORTING_CONVENTION.md)。
