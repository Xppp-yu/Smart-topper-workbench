# Smart Topper Windows Research Workbench — 项目状态

最后更新：2026-09-01（B09 30-unit Full 运行准备 R05 窄范围收口已落盘；B09 准备
合同 / validator / 测试 / AutoDL 命令合同 / 资源估算均已重构并通过 35 项
B09 定向测试；当前 Gate 因 CLI bridge blocker 保持 `ITERATE / CLI_BRIDGE_REQUIRED`；
30-unit Full 与任何 TEST 均未授权；B08 仍为最近一次已运行/已验收阶段）
状态口径：只把已运行且存在可追溯产物的步骤标记为 `COMPLETE`；代码、计划或空目录不等于完成；验收前保持 `READY_FOR_REVIEW`。

## 当前一句话状态

PoPu P5.2-C 已接受 `small_resnet` 为固定睡姿五分类总体研究候选模型族；P6/P6.1 已完成 UNKNOWN/REJECT、错误复核、温度校准与三模型一致性模拟；P7 Full 软件扰动证据已完成并复核（14 conditions × 5 seeds × 15 folds），但不构成硬件、产品、整夜睡眠或控制安全验证。SLP A09R、B01–B04 已验收；B04A R03 已在冻结 SHA `f0fac823`、TEST=0 下完成并由 Owner `ACCEPT_WITH_LIMITATIONS`，晋级 DeepLabV3+-lite 与 ResUNet-lite；原始 budget carrier 缺陷及 Reviewer 重算永久披露，writer 修复已合入 `main@ae24d96`。B07 已接受开发期 91-subject、5-fold、3-seed、2-candidate、30-unit Full 协议；B08 runner 与真实 RTX 4090 one-fold preflight 已验收（`02fb3649`，fold_1/seed42/ResUNet-lite，wall 155.33s，peak 368.76 MiB，best epoch 22，TEST=0）。B09 30-unit Full 运行准备 R05 已修正 audit 对真实两位小数 budget serialization 的兼容性（容差 `0.005001`，同时保留精确 key 集合、重算和上限审计），并对齐真实 B08 writer schema：已存在的 terminal `DONE.json`、unit OOF、budget limits 与 unit peak 不再误列为 bridge 缺口。仍需独立 bridge 解决的仅是 30-unit CLI 入口、`status.json` 的 frozen hashes、六个 TEST=0 carrier，以及（若保留 91-subject 审计）candidate seed `total_subjects` carrier。A06 三方 binding 和 TEST=0 fail-closed 合同保持有效。当前 Gate 因 CLI 仍显式拒绝 30-unit real B01 入口而保持 `ITERATE / CLI_BRIDGE_REQUIRED / GPU_FULL_NOT_AUTHORIZED / TEST_DENIED`；30-unit Full 与任何 TEST 仍未授权。

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
| P7/R7 | 降密度、噪声、坏点、坏行坏列 | COMPLETE — SOFTWARE_PERTURBATION_ONLY | [P7 Full 结果与复核](stage_reports/P7_POPU_SOFTWARE_ROBUSTNESS_FULL_RESULTS_v0.1.md)、[P7 恢复归档登记](stage_reports/P7_RECOVERED_ARCHIVE_REGISTER_v0.1.md)；14 conditions × 5 seeds × 15 folds；原 tar 的历史验证保持不变，当前恢复包以独立 SHA 登记 | 只允许软件扰动结论；恢复包不得冒充原 tar；硬件故障、产品、整夜和控制安全需独立 EXP-ID |
| P8/R8 | PoPu 后续工程化或硬件验证 | NOT_SCHEDULED / SEPARATE_GATE | P7 不阻塞 SLP；尚无硬件故障验证 | 另行定义自研硬件、同步真值和产品边界，不从 P7 自动外推 |

## SLP 阶段看板

| 阶段 | 目标 | 状态 | 已有证据 | 下一步门槛 |
|---|---|---|---|---|
| S0 | 全量目录、模态与标注边界审计 | COMPLETE_WITH_QUARANTINE | [S0 结果](stage_reports/S0_SLP_FULL_INVENTORY_AND_ANNOTATION_BOUNDARY_v0.1.md)、1,941 组 Inventory、109 人标注表和摘要；2 组 depthRaw 缺失 | 精确 quarantine；进入 S1 配对与坐标审计 |
| S1 | 跨模态配对、homography 与 overlay 审计 | IN_PROGRESS_WITH_A05_A06_A07_A08_DELIVERED | [A03 Frame Master Index](stage_reports/S1_1_SLP_FRAME_MASTER_INDEX_v0.1.md)、[A04 Homography 审计](stage_reports/S1_2_SLP_HOMOGRAPHY_AUDIT_v0.1.md)、[A05 Canonical Adapter](stage_reports/S1_3_SLP_CANONICAL_ADAPTER_v0.1.md)、[A06 Subject Split Freeze](stage_reports/S1_4_SLP_SUBJECT_SPLIT_FREEZE_v0.1.md)、[A07 Joint Occlusion EDA](stage_reports/S1_5_SLP_JOINT_EDA_v0.1.md)、[A08 Body Axis Geometry](stage_reports/S1_8_SLP_BODY_AXIS_GEOMETRY_v0.1.md)、[A09R GT Contract Realign](stage_reports/S1_A09R_SLP8_GT_CONTRACT_REALIGN_v0.1.md)、[Examples](stage_reports/SLP_CANONICAL_SAMPLE_EXAMPLES_v0.1.md)、[SLP 两阶段总计划](SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md)、[Agent 任务清单](SLP_AGENT_TASK_BACKLOG_v0.1.md) | A09R 完成；A10–A17 HOLD/SUPERSEDED；B01–B04 完成；B04A R02 FAILED，reload/identity 修复已验收；R03 运行准备已完成，待任务发布、AutoDL Preflight 和 Owner 授权 |
| S1_A09R | SLP8 GT 合同校准 | COMPLETE_WITH_LIMITATIONS | [A09R 报告](stage_reports/S1_A09R_SLP8_GT_CONTRACT_REALIGN_v0.1.md)；8区 schema；adapter；66 dataset tests + 221 regressions（1 skipped）；全量 validator 4590/4590、0 failures；A06 split 兼容性（102 主体，3645/450/495，0 overlap） | B01 训练表冻结 READY；严格保留 NOT_REVIEWED、uncover-only 和非产品 GT 边界 |
| S2_B01 | SLP8 训练表冻结 | DONE_WITH_LIMITATIONS | [S2_B01 报告](stage_reports/S2_B01_SLP8_TRAINING_TABLE_FREEZE_v0.1.md)；真实 build 3,645/450/495；TEST 默认拒绝并要求 `purpose="final_evaluation"` + 显式 reload | 作为 B04A 的不变数据合同；继续保留 NOT_REVIEWED、uncover-only 和非产品 GT 边界 |
| S2_B02 | SLP8 非学习区域基线 | DONE_WITH_LIMITATIONS | [S2_B02 报告](stage_reports/S2_B02_SLP8_NON_LEARNING_REGION_BASELINE_v0.2.md)；VAL fixed IoU：all-background `0.000000`、train spatial prior `0.205644`、body-axis partition `0.109308`、axis-contact intersection `0.109713` | `0.205644` 作为 B04A 前瞻性 margin 的固定参考，不根据新结果重估 |
| S2_B03 | SLP8 PM-only Region Smoke | DONE_WITH_LIMITATIONS | [S2_B03 报告](stage_reports/S2_B03_SLP8_PM_ONLY_REGION_SMOKE_v0.1.md)；TinyFCN CPU TRAIN 90 / VAL 45 / TEST 0；checkpoint/resume/reload 一致 | 历史 Smoke 只证明链路；TinyFCN 的 Smoke 分数不得用于模型排名 |
| S2_B04 | SLP8 PM-only Region Mini | DONE_WITH_LIMITATIONS | [B04 协议](stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md)、[B04 R05 结果](stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_RESULTS_v0.1.md)；TRAIN 3,645 / VAL 450 / TEST 0；SmallUNet `0.439625`（FEASIBLE），TinyFCN `0.051631`（NOT_FEASIBLE） | 历史结果不改写；SmallUNet 作为 B04A incumbent，不直接视为最终架构候选 |
| S2_B04A | SLP8 PM-only 受控架构扩展 Mini | GPU_MINI_R02_FAILED / RELOAD_FIX_ACCEPTED / IDENTITY_FIX_ACCEPTED / GPU_R03_NOT_AUTHORIZED | [Protocol](stage_reports/S2_B04A_SLP8_PM_ARCHITECTURE_EXPANSION_MINI_PROTOCOL_v0.1.md)、[Implementation+Smoke](stage_reports/S2_B04A_IMPLEMENTATION_SMOKE_v0.1.md)、[Runner Integration+Smoke](stage_reports/S2_B04A_RUNNER_INTEGRATION_SMOKE_v0.1.md)、[Mini Run 准备报告](stage_reports/S2_B04A_MINI_RUN_PREPARATION_v0.1.md)、[Reload 修复报告](stage_reports/S2_B04A_RELOAD_CONSISTENCY_FIX_v0.1.md)、[Identity 修复报告](stage_reports/S2_B04A_EXPERIMENT_IDENTITY_CARRIER_FIX_v0.1.md)、[Identity 修复合同](tasks/TASK_SLP_B04A_EXPERIMENT_IDENTITY_CARRIER_FIX_v0.1.md)、[冻结配置](../configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json)；R02 9/9 checkpoint + FAILED；reload 修复回归 167 + 247 passed；identity 修复 R01 暴露 5 项 checkpoint/terminal identity 缺陷已修复；R02 暴露 5 项覆盖/异常/单源/合同/文档缺陷已修补；R04 ITERATE 收口：合同 `Files allowed to change` 补 `slp8_region_resume.py` 与 `smoke_b04a_runner_integration.py`；`_b04a_identity_block` git_commit fail-closed 拒绝空 / 空白 / `unresolvable_git_commit` / 非 hex / 错长度；`_resolve_git_identity` unresolvable 抛 `MiniProtocolError`；CLI 在 dispatch 前冻结 run identity context；post-validation FAILED 不重解析 git identity；新增真实 B01 post-validation failure 测试（freeze fixture + `freeze_manifest.json` + Owner EXP-ID + 训练前注入异常，TEST=0）；identity 修复经 Codex 独立验收：B04A integration 129/129、B04 Mini 167/167、validator 30 OKs/0 errors、54 个 synthetic identity carriers 0 mismatch；TEST=0；R05 ITERATE 收口：`run_mini_b04a()` 加必填 `git_commit`/`git_dirty` keyword-only 参数，删内部 resolver 调用；`_run_synthetic_cpu_smoke_b04a` / `_run_real_b01_b04a` / smoke 脚本 均传入 frozen 值；新增正常成功路径回归测试（call_count == 1，bundle carrier 审计）；R04 断言 `>= 1` → `== 1`；R04 real-B01 fixture TEST=0 不变量通过 `freeze._test_rows is None` 断言；12 个 R04+R05 定向测试与完整回归全部通过 | 合并修复并冻结新的 main SHA；之后由 Owner 单独授权全新 R03 EXP-ID；B07 继续 `BLOCKED_BY_B04A` |
| S2_B07 | SLP8 PM-only Full 协议冻结 | PROTOCOL_ACCEPTED / COMPUTE_NOT_RUN | [B07 任务](tasks/TASK_SLP_B07_FULL_PROTOCOL_FREEZE_v0.1.md)、[阶段报告](stage_reports/S2_B07_SLP8_PM_ONLY_FULL_PROTOCOL_v0.1.md)、冻结 folds/config；91 subjects × 5-fold，2 candidates × 3 seeds = 30 units，TEST=0；validator/failure-injection/links 18 passed | B08 Runner 实现完成待 Review；GPU preflight 未授权 |
| S2_B09_runprep | SLP8 PM-only 30-unit Full 运行准备 | ITERATE / CLI_BRIDGE_REQUIRED / GPU_FULL_NOT_AUTHORIZED / TEST_DENIED | [B09 准备任务书 R05](tasks/TASK_SLP_B09_FULL_RUN_PREPARATION_v0.1.md)、validator `scripts/validate_b09_full_run_preparation.py`（R05：budget map 完整 30 unit key + 2 candidate key 集合审计、从 30 个 complete.json 重算与 budget_report cross-check，并按真实两位小数 writer precision 比较；TEST=0 evidence 6 carrier 严格类型校验；A06 split 三方 binding；仅保留真实 CLI bridge/schema 缺口）、35 项定向测试、AutoDL 命令合同（仅作 `DO NOT RUN — CLI BRIDGE NOT IMPLEMENTED` 拟议模板）、B08 R03 锚点资源估算、§16 CLI bridge 最小 blocker | Codex 独立 Review；Owner 签发独立 `TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1`；30-unit Full 与 TEST 仍未授权 |
| S2_B08 | SLP8 PM-only Full Runner 与 one-fold preflight runner | ACCEPT / R03_PREFLIGHT_PASSED / TEST_DENIED | runner=`02fb364902736a64ee8708440f0dd0bdddf860bc`；B08 80 passed、links 6 passed；RTX 4090/Torch 2.13 strict probe 与真实 fold_1/seed42/ResUNet-lite R03 DONE；best epoch 22，wall 155.329s，peak 368.764 MiB，reload consistent，OOF 855 samples/19 subjects，TEST=0；本地 archive SHA 匹配并审计通过；R01/R02 失败永久披露 | B09 Full 运行准备可开始，但 30-unit Full 与 TEST 仍需独立 Owner 授权 |
| S2 | Canonical Sample 与受试者拆分冻结 | COMPLETE_WITH_A06_SPLIT_FROZEN | [A06 Split Freeze](stage_reports/S1_4_SLP_SUBJECT_SPLIT_FREEZE_v0.1.md)、manifest JSON、SHA-256 `024f5abe`；109 subjects split 81/10/18 (danaLab 81/10/11，simLab 0/0/7 TEST held-out)；quarantine 90 frames 单独统计；6 isolation tests PASS；A06 split 与 SLP8 GT 完全兼容（102 主体，0 overlap） | A18 节点基线使用此 split；B01 使用 A06 split 作为训练基础；fold 设计由 B07 Full 协议另行冻结 |
| S3 | RGB/IR/Depth/PM 单模态关节基线 | BLOCKED_BY_S2 | 尚未开始 | 每模态保留 1–2 个候选，分开报告原始与映射标签 |
| S4 | 身体区域伪标签 Pilot 与人工复核 | HOLD / OPTIONAL_FUTURE / SUPERSEDED_FOR_CURRENT_SLP8_GT | A09R 已将 SLP8 GT 设为默认训练合同；原 A10–A17 OpenCV/人工复核路线已改 HOLD | 不再以 A17 R2/R3 为前置；SLP8 区域训练使用已冻结 GT |
| S5–S6 | 遮盖压力测试、有限融合 | HOLD / DIFFERENT_TRACK | 尚未开始；cover GT 和独立融合合同未满足 | 不阻塞 PM-only B04A；不得把 uncover 结果外推到 cover |

## 自研传感器初步证据

| 阶段 | 目标 | 状态 | 已有证据 | 下一步门槛 |
|---|---|---|---|---|
| H0 | 32×32 传感器导出链路与基础响应阶段性测试 | EVIDENCE_SNAPSHOT_WITH_LIMITATIONS | [H0 阶段报告](stage_reports/H0_SELF_COLLECTED_SENSOR_STAGE_TEST_v0.1.md)；空载、重复放置、空间位置和 540 秒恒定载荷的小规模摘要已脱敏入库；原始 CSV 保持本地只读 | 供应商解释 20 Hz / 26 fps / 300 ms / 空时间戳关系，并提供 mapped ADC 到物理单位的标定合同；随后另立正式硬件测量协议与独立复算任务 |

## 当前数据使用边界

- 固定姿态有标签集：`5,100` 条 JSON 记录；其中四类卧姿各 `1,260` 条，空床 `60` 条。
- `others.json`：每位受试者一条、共 `60` 条，缺少姿态与 variation 标签；保留给后续转身、过渡或 `UNKNOWN` 研究，不伪造为固定姿态训练样本。
- **SLP 8-region pressure-only GT**（`SLP_8Region_Pressure_VAL_v1.1`，4,590 samples，102 danaLab，uncover only）：`V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`，`source_review_status=NOT_REVIEWED`；是当前 SLP8 区域分割的 PROJECT_ACCEPTED_REFERENCE_GT；**不是人工像素级标注；不是医学/皮肤界面应力/产品真值**；压力保持 raw PMarray response semantics，不得称为 kPa；仅 danaLab/uncover，不得外推到产品、硬件、舒适性、医疗或气囊控制。B01 已基于此 GT + A06 split 冻结 pressure-only 8-region 训练/验证/测试数据入口（`slp8_training_tables_v0.1`），详见 [S2_B01 阶段报告](stage_reports/S2_B01_SLP8_TRAINING_TABLE_FREEZE_v0.1.md)；TEST 访问受 `allow_test + purpose=final_evaluation` 合同约束。
- **SLP 10-region polygon 路线**（`slp_region_annotation_v0.1`，R0–R3 tier）：历史内部治理合同，**不再是当前训练合同**，不得与 8-region 数据混用。
- PoPu 与 PMD 是独立公开数据集，不能逐行配对。
- 所有公开数据结果仅证明研究数据链路或算法候选，不证明自研传感器、气囊闭环、整夜稳定性、舒适性或产品效果。
- H0 自采传感器摘要只证明当前采集软件下的数据导出与小规模基础响应；不是物理压力标定、精度、寿命、环境、人体、舒适性、控制安全或产品可靠性验证。

## 治理口径

当前采用多 Agent 协作制度 v0.2：Owner 决策，网页 GPT 负责战略讨论、任务起草和 GitHub 二审，Claude Code 负责单任务本地实现，Codex 负责本地控制与阶段验收，Runner 只执行冻结实验，GitHub 保存已提交/已推送的共享基线。`TASK-ID` / `EXP-ID`、本地状态快照、并行 worktree、`QUEUED` 后参数不可变及 Smoke/Mini/Full 边界，见 [多 Agent 协作制度 v0.2](../COLLABORATION_WORKFLOW.md) 与 [实验治理与远程 GPU 执行方案](EXPERIMENT_GOVERNANCE_AND_GPU_EXECUTION_PLAN_v0.1.md)。

## 如何阅读与更新

1. 每一阶段先看 `docs/stage_reports/` 中对应报告，获得“本阶段做了什么、得到什么、没得到什么、下一步是什么”。
2. 再按报告中的精确路径打开 CSV、JSON、PNG 或模型，查看原始证据。
3. 每次完成一个可运行阶段，新增一份阶段报告并更新本页；失败或暂停也记录原因，不改写历史结论。

具体格式见 [阶段记录与报告约定](STAGE_REPORTING_CONVENTION.md)。
