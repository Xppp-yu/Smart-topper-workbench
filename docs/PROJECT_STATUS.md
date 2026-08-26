# Smart Topper Windows Research Workbench — 项目状态

最后更新：2026-08-25（B02 SLP8 非学习区域基线 `DONE_WITH_LIMITATIONS`）
状态口径：只把已运行且存在可追溯产物的步骤标记为 `COMPLETE`；代码、计划或空目录不等于完成；验收前保持 `READY_FOR_REVIEW`。

## 当前一句话状态

P5.2-C 已接受 `small_resnet` 为 PoPu 固定睡姿五分类总体研究候选模型族。P6/P6.1 已完成 UNKNOWN/REJECT、原始重点案例复核、温度校准与三模型一致性模拟；P7 真实 checkpoint 扰动推理仍待运行。SLP A09R 已以 `COMPLETE_WITH_LIMITATIONS` 验收：`SLP_8Region_Pressure_VAL_v1.1`（4,590 samples，102 danaLab，8 区）确立为当前项目 SLP8 pressure-only 区域分割参考 GT，8 区 schema + adapter（66 tests）+ 全量 validator（4590/4590，0 failures）和 A06 split 兼容性（3645/450/495，0 overlap）均经 Reviewer 复核；A10–A17 OpenCV/人工复核路线为 HOLD/SUPERSEDED。SLP B01、B02 均已以 `DONE_WITH_LIMITATIONS` 验收。SLP B03 PM-only Region Smoke 也已以 `DONE_WITH_LIMITATIONS` 验收：真实 CPU Smoke 使用 TRAIN 90 / VAL 45 / TEST 0，checkpoint/resume/reload、真实逐样本预测哈希和审计产物均通过，Reviewer 联合复核为 479 passed、2 skipped。B04 已解锁为 `READY`，但须先冻结 Mini 协议并取得运行授权。

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
| P6/R6 | `UNKNOWN/REJECT` 与错误分析 | COMPLETE — BOUNDED_LIMITATION_RECORDED | [P6 结果](stage_reports/P6_POPU_REJECT_RESULTS_v0.1.md)、[P6.1 校准与一致性结果](stage_reports/P6_1_POPU_CALIBRATION_ENSEMBLE_RESULTS_v0.1.md)；重点原始案例 36/36 标签与结构一致；24 条 record 跨 3 repeats 持续错判；一致性候选降低 WAR 但牺牲最差受试者覆盖率 | 停止继续调 PoPu 阈值；一致性规则仅作研究候选；进入 P7 时保留高置信错误与覆盖率边界 |
| P7/R7 | 降密度、噪声、坏点、坏行坏列 | PROTOCOL_AND_PERTURBATIONS_READY | [P7 协议](stage_reports/P7_POPU_SOFTWARE_ROBUSTNESS_PROTOCOL_v0.1.md)、冻结配置、确定性扰动模块与测试 | 提取 15 个 Small ResNet fold checkpoint，在对应 outer-test 原始 record 上运行 clean/扰动推理；不将软件消融误称为硬件验证 |
| P8/R8 | 冻结候选并移交 WSL 工程化 | BLOCKED_BY_P7 | 尚未开始 | 算法规格、配置、切分和限制完整冻结 |

## SLP 阶段看板

| 阶段 | 目标 | 状态 | 已有证据 | 下一步门槛 |
|---|---|---|---|---|
| S0 | 全量目录、模态与标注边界审计 | COMPLETE_WITH_QUARANTINE | [S0 结果](stage_reports/S0_SLP_FULL_INVENTORY_AND_ANNOTATION_BOUNDARY_v0.1.md)、1,941 组 Inventory、109 人标注表和摘要；2 组 depthRaw 缺失 | 精确 quarantine；进入 S1 配对与坐标审计 |
| S1 | 跨模态配对、homography 与 overlay 审计 | IN_PROGRESS_WITH_A05_A06_A07_A08_DELIVERED | [A03 Frame Master Index](stage_reports/S1_1_SLP_FRAME_MASTER_INDEX_v0.1.md)、[A04 Homography 审计](stage_reports/S1_2_SLP_HOMOGRAPHY_AUDIT_v0.1.md)、[A05 Canonical Adapter](stage_reports/S1_3_SLP_CANONICAL_ADAPTER_v0.1.md)、[A06 Subject Split Freeze](stage_reports/S1_4_SLP_SUBJECT_SPLIT_FREEZE_v0.1.md)、[A07 Joint Occlusion EDA](stage_reports/S1_5_SLP_JOINT_EDA_v0.1.md)、[A08 Body Axis Geometry](stage_reports/S1_8_SLP_BODY_AXIS_GEOMETRY_v0.1.md)、[A09R GT Contract Realign](stage_reports/S1_A09R_SLP8_GT_CONTRACT_REALIGN_v0.1.md)、[Examples](stage_reports/SLP_CANONICAL_SAMPLE_EXAMPLES_v0.1.md)、[SLP 两阶段总计划](SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md)、[Agent 任务清单](SLP_AGENT_TASK_BACKLOG_v0.1.md) | A09R 完成：SLP_8Region_Pressure_VAL_v1.1 已接受为项目参考 GT（4,590 samples，102 danaLab，8 区）；A10–A17 路线改为 HOLD/SUPERSEDED；B01–B03 已完成，B04 READY |
| S1_A09R | SLP8 GT 合同校准 | COMPLETE_WITH_LIMITATIONS | [A09R 报告](stage_reports/S1_A09R_SLP8_GT_CONTRACT_REALIGN_v0.1.md)；8区 schema；adapter；66 dataset tests + 221 regressions（1 skipped）；全量 validator 4590/4590、0 failures；A06 split 兼容性（102 主体，3645/450/495，0 overlap） | B01 训练表冻结 READY；严格保留 NOT_REVIEWED、uncover-only 和非产品 GT 边界 |
| S2_B01 | SLP8 训练表冻结 | DONE_WITH_LIMITATIONS | [S2_B01 报告](stage_reports/S2_B01_SLP8_TRAINING_TABLE_FREEZE_v0.1.md)；`slp8_training_table_freeze_v0.1` 模块；`build_slp8_training_tables.py` + `validate_slp8_training_tables.py`；`tests/test_slp8_training_table_freeze.py`（82 passed）；真实数据 build 输出 3,645/450/495；full validator（含 deterministic rebuild）`ALL CHECKS PASSED in 56.0s`；287 regression tests passed，联合套件共 369 passed（1 skipped：A05 CSV 不存在）；TRAIN-only normalization（`raw_pmarray_response`、NOT kPa）；TEST 防泄漏合同（默认不读取 TEST；`enable_test_access(purpose="final_evaluation")` 后必须显式 `load_test=True` 重新加载才返回 TEST rows）；data card；gitignored 输出目录 | B02、B03 已完成；B04 `READY`；继续保留 NOT_REVIEWED、uncover-only 和非产品 GT 边界 |
| S2_B02 | SLP8 非学习区域基线 | DONE_WITH_LIMITATIONS | [S2_B02 报告](stage_reports/S2_B02_SLP8_NON_LEARNING_REGION_BASELINE_v0.2.md)；四种 pressure-only 基线；R02 TRAIN 3,645 / VAL 450、TEST 0；VAL fixed IoU：all-background `0.000000`、train spatial prior `0.205644`、body-axis partition `0.109308`、axis-contact intersection `0.109713`；输出保留逐区域/受试者/posture 指标、预测哈希、诊断与失败审计；R03 runner fail-closed；Codex 独立复核 259 passed、2 skipped | B03 已完成；B04 `READY`；继续保留 NOT_REVIEWED、uncover-only、非产品 GT 与 TEST 默认拒绝边界 |
| S2_B03 | SLP8 PM-only Region Smoke | DONE_WITH_LIMITATIONS | [S2_B03 报告](stage_reports/S2_B03_SLP8_PM_ONLY_REGION_SMOKE_v0.1.md)；`Slp8TinyFcn`；CPU TRAIN 90 / VAL 45 / TEST 0；initial+resume 各 1 epoch；真实 prediction manifest 270 行；checkpoint/resume/reload 一致；Reviewer 独立运行与 R03 loss、IoU、预测哈希逐行一致；联合复核 479 passed、2 skipped | B04 `READY`；先冻结 Mini 协议，不把 Smoke 指标用于排名，不读取 TEST，不形成产品结论 |
| S2 | Canonical Sample 与受试者拆分冻结 | COMPLETE_WITH_A06_SPLIT_FROZEN | [A06 Split Freeze](stage_reports/S1_4_SLP_SUBJECT_SPLIT_FREEZE_v0.1.md)、manifest JSON、SHA-256 `024f5abe`；109 subjects split 81/10/18 (danaLab 81/10/11，simLab 0/0/7 TEST held-out)；quarantine 90 frames 单独统计；6 isolation tests PASS；A06 split 与 SLP8 GT 完全兼容（102 主体，0 overlap） | A18 节点基线使用此 split；B01 使用 A06 split 作为训练基础；fold 设计由 B07 Full 协议另行冻结 |
| S3 | RGB/IR/Depth/PM 单模态关节基线 | BLOCKED_BY_S2 | 尚未开始 | 每模态保留 1–2 个候选，分开报告原始与映射标签 |
| S4 | 身体区域伪标签 Pilot 与人工复核 | HOLD / OPTIONAL_FUTURE / SUPERSEDED_FOR_CURRENT_SLP8_GT | A09R 已将 SLP8 GT 设为默认训练合同；原 A10–A17 OpenCV/人工复核路线已改 HOLD | 不再以 A17 R2/R3 为前置；SLP8 区域训练使用已冻结 GT |
| S5–S7 | 遮盖压力测试、有限融合、Full 比较 | HOLD / DIFFERENT_TRACK | 尚未开始；RGB/IR/Depth/PM 模态路线独立于 SLP8 GT | 固定候选、拆分、预算和受试者级评价；SLP8 GT 训练（B01）不依赖此路线 |

## 当前数据使用边界

- 固定姿态有标签集：`5,100` 条 JSON 记录；其中四类卧姿各 `1,260` 条，空床 `60` 条。
- `others.json`：每位受试者一条、共 `60` 条，缺少姿态与 variation 标签；保留给后续转身、过渡或 `UNKNOWN` 研究，不伪造为固定姿态训练样本。
- **SLP 8-region pressure-only GT**（`SLP_8Region_Pressure_VAL_v1.1`，4,590 samples，102 danaLab，uncover only）：`V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`，`source_review_status=NOT_REVIEWED`；是当前 SLP8 区域分割的 PROJECT_ACCEPTED_REFERENCE_GT；**不是人工像素级标注；不是医学/皮肤界面应力/产品真值**；压力保持 raw PMarray response semantics，不得称为 kPa；仅 danaLab/uncover，不得外推到产品、硬件、舒适性、医疗或气囊控制。B01 已基于此 GT + A06 split 冻结 pressure-only 8-region 训练/验证/测试数据入口（`slp8_training_tables_v0.1`），详见 [S2_B01 阶段报告](stage_reports/S2_B01_SLP8_TRAINING_TABLE_FREEZE_v0.1.md)；TEST 访问受 `allow_test + purpose=final_evaluation` 合同约束。
- **SLP 10-region polygon 路线**（`slp_region_annotation_v0.1`，R0–R3 tier）：历史内部治理合同，**不再是当前训练合同**，不得与 8-region 数据混用。
- PoPu 与 PMD 是独立公开数据集，不能逐行配对。
- 所有公开数据结果仅证明研究数据链路或算法候选，不证明自研传感器、气囊闭环、整夜稳定性、舒适性或产品效果。

## 治理口径

当前采用多 Agent 协作制度 v0.2：Owner 决策，网页 GPT 负责战略讨论、任务起草和 GitHub 二审，Claude Code 负责单任务本地实现，Codex 负责本地控制与阶段验收，Runner 只执行冻结实验，GitHub 保存已提交/已推送的共享基线。`TASK-ID` / `EXP-ID`、本地状态快照、并行 worktree、`QUEUED` 后参数不可变及 Smoke/Mini/Full 边界，见 [多 Agent 协作制度 v0.2](../COLLABORATION_WORKFLOW.md) 与 [实验治理与远程 GPU 执行方案](EXPERIMENT_GOVERNANCE_AND_GPU_EXECUTION_PLAN_v0.1.md)。

## 如何阅读与更新

1. 每一阶段先看 `docs/stage_reports/` 中对应报告，获得“本阶段做了什么、得到什么、没得到什么、下一步是什么”。
2. 再按报告中的精确路径打开 CSV、JSON、PNG 或模型，查看原始证据。
3. 每次完成一个可运行阶段，新增一份阶段报告并更新本页；失败或暂停也记录原因，不改写历史结论。

具体格式见 [阶段记录与报告约定](STAGE_REPORTING_CONVENTION.md)。
