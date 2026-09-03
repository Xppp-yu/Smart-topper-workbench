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
- 已有：split manifest JSON、SHA-256 `024f5abe`、19 单元测试、6 isolation tests PASS、62 回归测试 PASS；109 subjects split 81/10/18（danaLab 81/10/11，simLab 0/0/7 TEST held-out）；90 quarantined frames 单独统计，不混入 train/val；确定性复现验证通过。
- 入口：`docs/stage_reports/S1_4_SLP_SUBJECT_SPLIT_FREEZE_v0.1.md`。
- 要求：danaLab/simLab 分层；同一受试者所有模态、遮盖、帧不可跨 split。
- 禁止：根据后续 TEST 分数调整受试者。

### TASK-SLP-A07：节点与遮挡 EDA

- 状态：`DONE`。
- 目标：统计14节点坐标、遮挡、越界、骨段长度和异常。
- 输出：逐节点/遮盖/场景/受试者 QA 表与图。
- 已有：J0/J1 分开报告（14,625 usable frames，90 quarantined）；逐节点遮挡率（85%/84.8% 头部，69-92% 躯干/四肢）；越界分析（头部占 70%）；147 J0 异常 + 141 J1 异常；骨段长度统计；Group-level 分组统计；5 个可视化图；28 单元测试 PASS；A08 无阻塞。
- 入口：`docs/stage_reports/S1_5_SLP_JOINT_EDA_v0.1.md`。
- 验收：J0 和 J1 分开报告；不把映射节点混入原始 GT 汇总。

### TASK-SLP-A08：人体轴与粗几何基线

- 状态：`DONE`。
- 目标：由肩、髋、thorax/head 建立 body axis、bbox 和方向置信度。
- 输出：几何模块、overlay、错误案例、测试。
- 已有：几何模块 `src/topper_perception/io/slp_body_geometry.py`；运行脚本 `scripts/run_slp_body_geometry.py`；48 单元测试（全部通过）；几何输出 schema JSON；几何 CSV（14,625 帧）；error cases CSV（10,546 行）；QA summary JSON；overlay manifest（12 张）；A08 Stage Report；14,625 帧真实 SLP 数据运行结果：ACCEPT 4,079 (27.9%)、UNCERTAIN 5,261 (36.0%)、REJECT 5,285 (36.1%)；A03–A07 回归测试全部通过（109 passed, 1 skipped）。
- 验收：左右翻转、坐标旋转、节点缺失/遮挡均有测试；A06 split 未被修改。

### TASK-SLP-A09：Region Ontology 与机器合同

- 状态：`SUPERSEDED_BY_A09R`（2026-08-24）。
- **Owner 决策**：`SLP_8Region_Pressure_VAL_v1.1`（4,590 samples，102 danaLab，8 区）已接受为项目参考 GT。SLP8 区域训练使用该数据集，不依赖 A10–A17 OpenCV/人工复核路线。
- 旧 `slp_region_annotation_v0.1`（10 区，R0–R3）标记为**历史内部治理合同**，不得作为当前训练入口。
- 入口：`docs/stage_reports/S1_A09R_SLP8_GT_CONTRACT_REALIGN_v0.1.md`。
- 验收：A09R 阶段报告已记录 Owner 决策、数据合同、A06 split 兼容性和禁止结论。

### TASK-SLP-A09R：SLP8 GT 合同校准（当前任务）

- 状态：`DONE_WITH_LIMITATIONS`（2026-08-24；Codex Reviewer 验收）。
- 目标：使仓库治理文档、类别合同、adapter、测试与实际 SLP_8Region_Pressure_VAL_v1.1 一致。
- 已有：8区 schema（`configs/annotations/slp_8region_pressure_gt_v1.1.schema.json`）、adapter（`src/topper_perception/io/slp_8region_pressure_dataset.py`）、validator（`scripts/validate_slp_8region_pressure_dataset.py`）、66 dataset tests + 221 regressions（1 skipped）、全量 validator（4590/4590，0 failures）、A06 split 兼容性（102 主体，3645/450/495，0 overlap）、统一 containment helper（`relative_to`）、全量 `allow_pickle=False`。
- 入口：`docs/stage_reports/S1_A09R_SLP8_GT_CONTRACT_REALIGN_v0.1.md`。

### TASK-SLP-A10：几何 Region Seeder

- 状态：`HOLD / OPTIONAL_FUTURE`。
- **不再是 B01 前置**。A09R 已通过 SLP8 GT 提供训练数据。若未来需要节点驱动的区域播种，再重新打开。

### TASK-SLP-A11：OpenCV Foreground Proposal

- 状态：`HOLD / OPTIONAL_FUTURE`。
- 目标：分别实现 Depth、IR、RGB uncover 的身体/被褥前景候选。
- 输出：可插拔 foreground backend、参数配置、overlay、测试。
- 技术：depth threshold/background、connected components、morphology、GrabCut seed、IR 辅助轮廓。
- 依赖现状：`pyproject.toml` 当前已将 `opencv-python` 列为基础依赖，与本 HOLD 任务原计划的 optional/headless 口径不一致。重新打开 A11 前必须先单独决定保留基础依赖，或迁移为唯一的 optional `opencv-python-headless`；不得同时安装 GUI/headless/contrib 多个 OpenCV wheel。
- 验收：原图不覆盖；每 backend 输出 mask+confidence+reason；cover 条件分开评价。
- 禁止：将 blanket contour 当体表真值。

### TASK-SLP-A12：Region Boundary Refiner

- 状态：`HOLD / OPTIONAL_FUTURE`。

### TASK-SLP-A13：人工复核工具

- 状态：`HOLD / OPTIONAL_FUTURE`。

### TASK-SLP-A14：Pilot 采样 Manifest

- 状态：`HOLD / OPTIONAL_FUTURE`。

### TASK-SLP-A15：Pilot 自动预标注与参数冻结

- 状态：`HOLD / OPTIONAL_FUTURE`。

### TASK-SLP-A16：Pilot 人工复核与一致性

- 状态：`HOLD / OPTIONAL_FUTURE`。

### TASK-SLP-A17：Region Reference v1.0 Freeze

- 状态：`SUPERSEDED_FOR_CURRENT_SLP8_GT`。
- **已由 A09R 的 SLP_8Region_Pressure_VAL_v1.1 替代**。原路线（A10–A16）的区域真值不再是当前训练前置。

### TASK-SLP-A18：节点定位轻量基线

- 状态：`BLOCKED_BY_A05_A06`（A05 COMPLETE，A06 COMPLETE，can start）。
- 目标：验证 RGB/IR J0 与 PM/Depth J1 的学习链路。
- 输出：Smoke/Mini，不做区域分割结论。
- 验收：PCK/PCKh/normalized error，逐节点/遮盖/受试者报告。

## 4. 阶段 II：Region Reference 形成后

### TASK-SLP-B01：冻结区域训练表

- 状态：`DONE_WITH_LIMITATIONS`（2026-08-25；Codex Reviewer 已验收；82 项 B01 单元/集成测试 + 287 项 A09R/schema/adapter/infra/split 回归测试全绿，联合套件共 369 passed（1 skipped：A05 CSV 不存在）；真实数据 build 3645/450/495；full validator 含 deterministic rebuild `ALL CHECKS PASSED in 56.0s`；TEST 合同：默认不读取 TEST rows，`enable_test_access` 后必须显式 `load_test=True` 重新加载才返回）。
- 输入：SLP_8Region_Pressure_VAL_v1.1 + A06 split (`slp_subject_split_v0.1.json`, SHA `024f5abe...`)。
- 输出：`data/processed/slp8_training_tables_v0.1/{train,val,test}_manifest.{csv,jsonl}`、`freeze_manifest.json`（顶层，content-addressed）、`normalization_stats.json`（TRAIN-only fit）、`{train,val}_class_stats.json`（TEST 仅结构性计数）、`dataset_card.md`。
- 模块：`src/topper_perception/io/slp8_training_table_freeze.py`（A06/source loaders、freeze row builder、TEST 访问守卫、TRAIN-only normalization、freeze manifest、dataset card、`Slp8TrainingTableFreezer`）；CLI `scripts/build_slp8_training_tables.py` 与 `scripts/validate_slp8_training_tables.py`；测试 `tests/test_slp8_training_table_freeze.py`。
- 验收：TEST reference 不被开发代码读取（`TestLeakageError` 除非 `enable_test_access(purpose="final_evaluation")`）；数据只从相对路径加载（无绝对路径 / `..` / 同前缀 sibling 逃逸）；normalization 仅在 TRAIN subjects 上 fit；subject overlap = 0；3645/450/495 与 A06 完全一致；A06 SHA 与 source manifest SHA 校验通过；deterministic rebuild 产生相同的 freeze manifest core SHA。
- 入口：[S2_B01 阶段报告](stage_reports/S2_B01_SLP8_TRAINING_TABLE_FREEZE_v0.1.md)。

### TASK-SLP-B02：非学习区域基线

- 状态：`DONE_WITH_LIMITATIONS`（2026-08-25；Codex Reviewer 已验收；实现提交 `e9bf4b5` / `af9ed87`，fail-closed 收口 `cc63ab5`）。
- 目标：建立 pressure-only 非学习最低比较线；实际实现 all-background、TRAIN spatial prior、pressure body-axis partition、pressure axis-contact intersection 四种基线。节点几何基线因 B01 不提供已批准的 joint predictor 输入而明确保持 `NOT_IMPLEMENTED`，未静默混入派生标签。
- 真实运行：R02 使用 B01 冻结表在 CPU 上运行 TRAIN 3,645 / VAL 450，TEST 0；VAL fixed IoU 分别为 `0.000000`、`0.205644`、`0.109308`、`0.109713`。R01 保留且未被覆盖。
- 验收：R03 后定向测试 179 passed，B01 回归 80 passed、2 skipped，合计 259 passed、2 skipped；输出目录防覆盖、失败路径 fail-closed、TRAIN/VAL 分离、TP/FP/FN precision/recall、预测 SHA-256、诊断与失败计数均经 Reviewer 复核。
- 入口：[S2_B02 阶段报告](stage_reports/S2_B02_SLP8_NON_LEARNING_REGION_BASELINE_v0.2.md)。
- 限制：使用 `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED` / `NOT_REVIEWED` 的 SLP8 pressure-only 参考 GT；不是人工像素级、医学、皮肤界面应力、硬件或产品 GT；未读取 TEST，不能形成最终测试结论。

### TASK-SLP-B03：单模态 Region Smoke

- 状态：`DONE_WITH_LIMITATIONS`（2026-08-27；R03 `8979c6f`；Codex Reviewer 已验收）。
- 当前模态：PM-only；Depth/IR/RGB 属于独立对齐合同后的可选路线。
- 目标：验证数据吞吐、loss、输出、checkpoint/resume/reload。
- 结果：CPU TRAIN 90 / VAL 45 / TEST 0；initial+resume 各 1 epoch；270 行真实预测证据；独立复核运行与 R03 的 loss、IoU、预测哈希一致；联合测试 479 passed、2 skipped。
- 禁止：Smoke 分数排名；不得读取 TEST label/onehot 除非显式 `enable_test_access(purpose="final_evaluation")`。

### TASK-SLP-B04：单模态 Region Mini

- 状态：`DONE_WITH_LIMITATIONS`（2026-08-29；协议、Runner、真实 RTX 4090 Mini R05 与本地归档验收完成）。
- 目标：每模态保留最多1–2个可行候选。
- 指标：region IoU/Dice、中心误差、逐区域、逐 posture、worst subject；当前仅 uncover。
- 真实结果：`EXP-SLP-B04-PM-REGION-MINI-20260828-AUTODL-R05` 状态 `DONE`；TRAIN 3,645 / VAL 450 / TEST 0；SmallUNet VAL fixed foreground macro IoU `0.439625`，通过冻结门槛 `0.205644`；TinyFCN `0.051631`，未通过。两候选均完成 checkpoint reload 一致性校验，0 candidate failed。
- 候选决策：`slp8_tiny_fcn_v0.1` 不再进入后续候选池；`slp8_small_unet_v0.1` 作为 B04A incumbent，而不是直接视为已经充分比较后的最终 Full 候选。该前瞻性调整不改写 B04 的历史 Gate 和结果。
- 限制：仅 danaLab/uncover、参考 GT 为 `NOT_REVIEWED` 自动接受标签、未读取 TEST；不能外推产品、硬件、舒适性、医疗、整夜或气囊控制。

### TASK-SLP-B04A：PM-only Architecture Expansion Mini

- 状态：`RUNNER_INTEGRATION_ACCEPTED / GPU_MINI_NOT_AUTHORIZED`（R03 协议冻结 + Implementation+Smoke + Runner Integration+Smoke；2026-08-30 Codex Reviewer R03 独立验收通过）。
- 目标：在不改写 B04 历史的前提下，以 SmallUNet 为 incumbent，受控比较 residual CNN、atrous/multi-scale CNN 和可选 transformer 三类不同架构假设，最多保留 1–2 个进入 B07。
- 已冻结候选：`slp8_small_unet_v0.1`（incumbent, 118,121 params）、`slp8_resunet_lite_v0.1`（new, 120,809 params, 1x1 Conv2d shortcut；shortcut 精确分解 `enc1=32 + enc2=544 + bottleneck=2112 = 2,688 = ResUNet - SmallUNet`）、`slp8_deeplabv3plus_lite_v0.1`（new, 53,449 params, Option A plain atrous）；SegFormer-B0 `DEFERRED`（7 项公平性前置项目未解决）。
- 已冻结协议：`configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json`；阈值=0.355644（B02 0.205644 + margin 0.15）；seeds=[42,123,2026]；3 候选 aug=none；ResUNet shortcut 显式 1x1 Conv2d（无 Identity / extra conv）；DeepLab 选 Option A（无 Xception / depthwise-separable）；参数上限 300,000；class collapse/worst-subject/per-region guardrails；`all_seeds_must_succeed=true`。
- exact_parameter_count 字段：3 候选均加入，验证器递归计算 Conv2d 参数并匹配（118,121 / 120,809 / 53,449）。
- 实现：`src/topper_perception/neural/slp8_region_models.py` 追加 `_ResidualBlock` / `Slp8ResUnetLite` / `_AsppModule` / `Slp8DeepLabV3PlusLite` 与对应工厂 / registry 条目；`Slp8SmallUnet` 行为未变（38 + 15 个 B04 回归测试全通过）。
- Runner 集成（`TASK-SLP-B04A-RUNNER-INTEGRATION-SMOKE-v0.1`，本任务交付）：`slp8_region_mini.py` 通过 `_CONFIG_VERSION_TO_PROTOCOL` + `_protocol_of_config` 集中派发 B04 vs B04A 协议；`validate_mini_config` / `build_mini_config` / `run_mini` 按协议路由；B04 历史路径字节级保持；B04A 新增 `_validate_b04a_mini_config` / `_build_b04a_mini_config` / `run_mini_b04a` 与 B04ACandidateAggregate / B04ARunResult / `_b04a_advance_decision`（0/1/2/3-feasible + near-tie tiebreak）；共享 `run_one_candidate`（新增 `seed` + `checkpoint_dir` 可选参数，B04 默认行为不变）。
- Runner 集成测试：`tests/test_b04a_runner_integration.py` **80 / 80**（R02 修订 73 + R03 修订新增 7 个：Reviewer 复现案例 advanced=("B","A")、3rd 替换 first_loser、无 1st-near-tie 时 2nd 边界比较 top2 vs top3、两轮都无 near-tie、2nd 边界 fail-closed missing worst-subject、输入顺序独立、advanced 永远两个不同候选；既有 dispatch / B04 backward-compat / candidate 限制 / seeds / all_seeds_must_succeed / 0-3 decision / budget resume / identity mismatch / 输出互斥 / TEST=0 / B04 回归 全通过）。
- 验证器：`scripts/validate_b04a_protocol.py`（30 OKs / 0 errors，合同未改）+ `tests/test_b04a_protocol_validator.py`（27 R02 + 23 R03）+ `tests/test_check_markdown_links.py`（6）+ `tests/test_b04a_implementation.py`（79 聚焦实现测试）= 135 项全部通过；B04A 涉及 markdown 相对链接检查 0 errors。
- Implementation Smoke：`scripts/smoke_b04a_implementation.py` 在 CPU 上对三候选完成 forward + backward + checkpoint roundtrip + same-seed determinism；支持 `--output` / `--force` / `--no-write`；默认拒绝覆盖已存在输出；`test_access.kind = "declarative_policy"`（非运行时计数）；CUDA Smoke 显式 `NOT_RUN`（本机 `torch==2.13.0+cpu` CPU-only build）。
- Runner Integration Smoke：`scripts/smoke_b04a_runner_integration.py`（CPU synthetic 3 candidates × 3 seeds 端到端）；`--no-write` 输出单行 `B04A_SMOKE_NO_WRITE ...`；写作模式输出 manifest / resolved_config / input_manifest_hashes / environment / status / candidate_decision / budget_report / logs/run.log + 9 个 `checkpoints/<candidate>/seed_<seed>/{last,best}.pt` + 互斥 terminal file；默认拒绝覆盖已存在产物；`--force` 显式允许。B04A synthetic CPU smoke：`candidates=3 seeds=3 terminal_state=DONE feasible=0 failed=0 stopped=0 advanced=0 near_tie_applied=False all_seeds_attempted=True any_seed_feasible=False`（synthetic 数据小到无法 beat 0.355644 阈值，证明编排路径，不证明性能排名）。
- CLI 新增 `--synthetic-cpu-smoke-b04a` + `--no-write` 模式（B04 历史 `--synthetic-cpu-smoke` / `--validate-config` 路径字节级保持）。
- R02 ITERATE 修订完成（2026-08-29）：本轮 R02 修订针对 Codex Reviewer R02 ITERATE 指出的三处合同级问题。已修复：
  1. **真实 B04A 路径错误调用单-seed B04 runner**——`scripts/run_slp8_region_mini.py::_run_real_b01` 已重写为 `config.protocol` 派发入口：B04 → `_run_real_b01_b04` → `run_mini`；B04A → `_run_real_b01_b04a` → `run_mini_b04a`。`run_mini` 顶部加 fail-closed 跨协议守卫（拒绝非 B04 protocol）；`run_mini_b04a` 已有的 fail-closed 守卫拒绝非 B04A。B04A 真实路径构造 per-(candidate, seed) resume 映射 `checkpoints/<candidate>/seed_<seed>/last.pt`；B04A 状态/identity/预算/决策产物使用 B04A schema；预算 45/135/8192；新增 `_load_b01_freeze_and_contract` 共享 B01 输入合同加载。
  2. **validate-only 对 B04A 仍写 B04 identity**——`_run_validate_config` 拆分为派发入口 + `_run_validate_config_b04` / `_run_validate_config_b04a` 两条独立分支。B04A 分支仅使用 `B04A_TASK_ID` / `B04A_CONFIG_VERSION` / `B04A_ACTIVE_CANDIDATE_NAMES` / `B04A_MAX_PARAMETERS` / `B04A_SEEDS` / `B04A_FORBIDDEN_CANDIDATE_NAMES` / `B04A_FEASIBILITY_THRESHOLD`；不引用 `TASK_ID` / `MINI_VERSION` / `B04_CANDIDATE_NAMES` / `B04_MAX_PARAMETERS`。`status.json` / `DONE.json` / `input_manifest_hashes.json` 都按协议身份写；`deferred_candidates` 来自 config 中 `role=DEFERRED` 条目（SegFormer-B0 保持 DEFERRED）。新增 2 个 subprocess 测试分别跑 B04 / B04A `--validate-config` 并对磁盘上真实 JSON 断言。
  3. **`_b04a_advance_decision` 拓扑边界与 tiebreak 修正**（R02 实现，**R03 已替换**）——R02 完整替换了 `_b04a_advance_decision` 与新增 `B04AAdvanceTiebreak` / `B04AAdvanceDecision` dataclass；tiebreak_basis 含 `none` / `parameter_count` / `worst_subject_iou` / `worst_subject_iou_then_param` / `failed_no_worst_subject`；`B04ARunResult` 新增 `advance_decision` 字段；`candidate_decision.json` 写入完整 tiebreak 审计。**但 R02 的 2nd boundary 错误地比较了 surviving top1/top2，而非 1st 轮 REJECTED 候选**（R03 修复）。
- **R03 ITERATE 修订完成（2026-08-29）**：本轮 R03 修订针对 Codex Reviewer R03 独立复跑指出的 2nd boundary 拓扑边界合同缺陷。已修复：
  1. **`_b04a_advance_decision` 2nd boundary 必须比较 first_loser vs top3**——取消 R02 错误的 `second_pair_higher = top1 if first_loser == top2[0] else top2`（该表达式永远等于 first_winner）；改用 `first_loser_pair = next((kv for kv in feasible if kv[0] == first_loser), None)` 取出 1st 轮 REJECTED 候选的 (name, aggregate) 喂给 `_b04a_pair_tiebreak`。`advanced = (first_winner, second_winner)`，删除 R02 的 dedup 回填逻辑。防御性 fail-closed：`first_winner == second_winner` 时返回 `advanced=()`。`_b04a_advance_decision` docstring 反映正确的两轮拓扑（1st 胜者拿 slot 1，first_loser 与 top3 争 slot 2）。
  2. **新增 7 个 R03 回归测试**（`TestB04ANearTieTiebreakR03`）：Reviewer 复现案例 advanced=("B","A") 且 2nd pair=("A","C")、3rd 替换 first_loser、无 1st-near-tie 时 2nd 边界比较 top2 vs top3、两轮都无 near-tie、2nd 边界 fail-closed missing worst-subject、输入顺序独立、advanced 永远两个不同候选。R02 既有 3 个测试的 2nd pair 与 macro_iou_difference 断言同步更新为 R03 正确语义。
- 当前自测结果：核心套件 `tests/test_b04a_runner_integration.py` **80 / 80**（R03 新增 7 + R02 修订后 73）；`tests/test_b04a_implementation.py` 79 / 79；`tests/test_b04a_protocol_validator.py` 全过；`tests/test_slp8_region_models.py` 38 / 38；`tests/test_slp8_region_mini.py` **162 / 162**（完整套件，含 SmallUNet/registry/synthetic + integration / contract / resume / determinism / CLI 全部分类）；B04 mini regression 15/15；`tests/test_check_markdown_links.py` 6/6；`git diff --check` 干净；TEST=0。
- Codex Reviewer R03 独立验收：Reviewer `A/B/C` 反例输出 `advanced=("B","A")` 且第二边界为 `(A,C)`；80 / 80 集成测试、162 / 162 完整 B04 回归、167 / 167 合同与模型测试、30 OKs / 0 errors、no-write smoke、链接、编译与 `git diff --check` 全部通过。
- 阶段名 = `RUNNER_INTEGRATION_ACCEPTED / GPU_MINI_NOT_AUTHORIZED`；这只接受 runner 与 synthetic smoke，**不**等同于 GPU Mini 完成或候选排名。**不得**声明 `GPU_MINI_AUTHORIZED` / `MINI_COMPLETE` / `B07_READY`。
- Mini Run 准备（2026-08-30）：`tasks/TASK_SLP_B04A_MINI_RUN_v0.1.md` 已起草为 `RUN_PREPARATION_READY_FOR_REVIEW / GPU_MINI_NOT_AUTHORIZED`；拟用独立 EXP-ID，最终 Git SHA/配置 hash/机器与 Owner 授权仍待填写。当前 raw SLP8 数据与 B01 transfer archive 存在，但 extracted B01 freeze 目录不存在；运行前必须按准备合同验证 archive SHA `23B323...269A`、无覆盖解包、CUDA smoke 实际执行以及 TEST=0。
- 入口：`tasks/TASK_SLP_B04A_PROTOCOL_FREEZE_v0.1.md`（协议）；`tasks/TASK_SLP_B04A_IMPLEMENTATION_SMOKE_v0.1.md`（实现+Smoke）；`tasks/TASK_SLP_B04A_RUNNER_INTEGRATION_SMOKE_v0.1.md`（runner 集成+Smoke）；`tasks/TASK_SLP_B04A_MINI_RUN_v0.1.md`（真实 Mini 准备合同）；[B04A Protocol 阶段报告](stage_reports/S2_B04A_SLP8_PM_ARCHITECTURE_EXPANSION_MINI_PROTOCOL_v0.1.md)；[B04A Implementation+Smoke 阶段报告](stage_reports/S2_B04A_IMPLEMENTATION_SMOKE_v0.1.md)；[B04A Runner Integration+Smoke 阶段报告](stage_reports/S2_B04A_RUNNER_INTEGRATION_SMOKE_v0.1.md)；[B04A Mini Run 准备报告](stage_reports/S2_B04A_MINI_RUN_PREPARATION_v0.1.md)；[B04A 交付说明](deliverables/SLP/B04A_PM_ARCHITECTURE_EXPANSION_MINI_PROTOCOL_v0.1.md)。
- GPU Mini R02（2026-08-30）：RTX 4090 上完成三候选 × 三 seeds 的 9/9 checkpoint，但 runner 将重载 `best.pt` 与后续 final model 比较，导致 `best_epoch < final_epoch` 的 seed 被错误标记为 reload FAILED；R02 终态保持 `FAILED`，R01/R02 证据包 SHA `75b9cd09...cb6494` 已回传本地，任何 `advanced` 字段不得作为正式晋级。
- Reload 修复（`TASK-SLP-B04A-RELOAD-CONSISTENCY-FIX-v0.1`）：best epoch 写盘前生成固定 CPU logits probe 并随 `best.pt` 保存；重载后对同一 probe 做数值比较，同时保留全量 prediction hash。定向 5 passed、B04 完整回归 167 passed、B04A/模型/协议联合 247 passed、协议 30 OKs/0 errors、no-write smoke DONE；Codex 已独立复核并 `ACCEPT`，但未授权 GPU。
- R02 复核新增阻塞：run-level `experiment_id` 不是 Owner 授权的 R02 EXP-ID，`data_manifest_sha256` 错写为 config SHA，且 run-level `model_version` 为空。必须另立 experiment-identity carrier 修复任务，不得混入 reload 修复。
- Identity carrier 修复（`TASK-SLP-B04A-EXPERIMENT-IDENTITY-CARRIER-FIX-v0.1`，2026-08-30，MiniMax Code 提交 Codex 独立复核）：CheckpointIdentity 扩展 `experiment_id` / `data_manifest_sha256` 字段；`_b04a_identity_block` 显式接收 EXP-ID 与 freeze_manifest_file_sha，run-level `model_version` 改用 `multi_candidate[... ]` 格式；CLI 新增 `--experiment-id`，真实 B01 路径在读取 B01 数据前 fail-closed，不创建 output_dir；resume 拒收 EXP-ID / data_manifest / model_version 漂移与 pre-fix 旧 checkpoint。同步 sentinel `SYNTHETIC_EXP_ID` 与 deterministic `_compute_synthetic_manifest_sha256()` 隔离合成与真实身份。回归：`test_b04a_runner_integration.py` 102/102（含新增 `TestB04AExperimentIdentityCarriers` 22 tests）、`test_b04a_implementation.py` 79/79、`test_b04a_protocol_validator.py` 50/50、`test_slp8_region_models.py` 38/38、`test_check_markdown_links.py` 6/6、`test_slp8_region_mini.py` 164/164（3 subprocess determinism 子集 deselected / `NOT RUN`），validator 30 OKs/0 errors。`py_compile` 干净。
- Identity carrier 修复 R02 ITERATE（2026-08-30，Codex Reviewer R01 独立审查 ITERATE）：R01 暴露 5 项真实 artifact 缺陷：
  1. `_run_synthetic_cpu_smoke_b04a` 把 EXP-ID / data_manifest_sha256 传给 `_write_b04a_run_bundle` 但没传给 `run_mini_b04a`，导致 `best.pt`/`last.pt` 中 `experiment_id=""`、`data_manifest_sha256=""`；
  2. CheckpointIdentity 顶层缺 `git_commit` / `git_dirty`，且协议要求的精确 `split_sha256` 字段名未独立存在（只有旧 `a06_split_sha256`）；
  3. `DONE.json` / `FAILED.json` / `STOPPED.json` 等 terminal JSON 缺 7 个 required identity fields；
  4. `B04ARunResult` 与 `_write_b04a_run_bundle` 是双 identity source，可被传入不一致值；
  5. 缺真实 artifact（best.pt / log / CSV sidecar / terminal JSON / DONE/FAILED/STOPPED）identity 审计测试。
  R02 已全部修复：`run_mini_b04a` / `run_mini` 与所有 `_write_b04a_*` 函数及 `write_status_files` 都从同一 frozen identity 构造；`run_mini_b04a` 拒绝 `B04ARunResult` 与 `_write_b04a_run_bundle` 参数不一致；CheckpointIdentity 增加 `git_commit` / `git_dirty` / `split_sha256` 三个字段并通过 `verify_resume_identity` 自动覆盖；terminal JSON 携带完整 7 字段；新增真实 artifact 审计测试覆盖 best.pt / last.pt / logs / CSV sidecar / DONE/FAILED/STOPPED / 跨载体一致性 / 不一致 fail-closed / resume 漂移。`TEST=0`，`B07` 继续 `BLOCKED_BY_B04A`。
- Identity carrier 修复 R03 ITERATE（2026-08-30，Codex Reviewer R02 独立审查 ITERATE）：R02 暴露 5 项覆盖/异常/单源/合同/文档缺陷：
  1. `write_status_files` 覆盖顺序错（`extra` 可篡改 identity）→ 改为 `extra → identity` 顺序；7 required fields 冲突时 fail-closed；辅助字段（`task_id` / `synthetic` / `data_manifest_source` / `a06_split_sha256`）正常 merge；
  2. 通用 except 终态 `FAILED.json` / `status.json` 缺 7 字段 → 新增 `_build_post_validation_identity` 辅助函数 + `main()` mutating 分支 + pre-validation 失败继续 no-write；
  3. 双 identity source 未根除 → `B04ARunResult` 增 `git_commit` / `git_dirty` 字段；`run_mini_b04a` 入口解析一次后写入 result；`_b04a_identity_block` 不再内部调 `_resolve_git_identity`；
  4. 合同 `Files allowed to change` 不全 → 显式加入 `slp8_region_resume.py`（checkpoint/resume identity schema）与 `smoke_b04a_runner_integration.py`（synthetic smoke identity propagation）；
  5. 文档状态 → `IDENTITY_FIX_ITERATE_R03 / READY_FOR_REVIEW`。
  R03 修补 5 个新测试覆盖 conflict fail-closed / consistent merge / post-validation FAILED / helper fail-closed / git identity frozen。
- Identity carrier 修复 R04 ITERATE 收口（2026-08-30，Codex Reviewer R03 收口要求）：R04 修补 5 项：
  1. 实际修改任务合同：`Files allowed to change` 加入 `slp8_region_resume.py`（checkpoint/resume identity schema）与 `smoke_b04a_runner_integration.py`（synthetic carrier propagation）；
  2. `git_commit` fail-closed：`_b04a_identity_block` 通过新增 `_validate_git_commit_strict` 拒绝空 / 空白 / `unresolvable_git_commit` 哨兵 / 非 hex / 错长度（不是 40 或 64 hex 字符）；
  3. `_resolve_git_identity` 改为严格模式：`git rev-parse HEAD` / `git status --porcelain` 失败或返回非 SHA 值时抛 `MiniProtocolError`（不再返回 `unresolvable_git_commit` 哨兵）；
  4. CLI 在 dispatch 前冻结 run identity context：`main()` 顶部调用 `_resolve_git_identity()` 一次；post-validation FAILED 走 mutating 分支用 helper 构造 identity（不再重解析 git identity）；resolver 失败时 return 2 不写文件；
  5. 文档状态 → `IDENTITY_FIX_ITERATE_R04 / READY_FOR_REVIEW`。
  R04 新增 10 个测试覆盖 git_commit fail-closed（空 / 空白 / sentinel / 非 hex / 错长度 / 接受 40+64 hex）、resolver 失败抛错、CLI 冻结 context 使用、resolver 不可解时 return 2、真实 B01 post-validation failure（freeze fixture + `freeze_manifest.json` + Owner EXP-ID + 训练前注入异常 + FAILED.json 7 字段 + `data_manifest_sha256 == freeze_manifest.json file SHA` + `git_commit == dispatch-time frozen` + `TEST=0`）。`TEST=0` 保持。R04 严禁写 `ACCEPTED` / `R03 GPU AUTHORIZED` / `B07 READY`。
- Identity carrier 修复 R05 ITERATE 收口（2026-08-30，Codex Reviewer R04 不 ACCEPT：正常训练路径仍二次解析 Git identity）：R05 修补 5 项：
  1. `run_mini_b04a()` 增加必填 `git_commit: str` / `git_dirty: bool` keyword-only 参数，入口加 `_validate_git_commit_strict` 验证，删内部 `_resolve_git_identity()` 调用；所有 carrier（CheckpointIdentity / B04ARunResult / best.pt / last.pt）只用传入值；
  2. `_run_synthetic_cpu_smoke_b04a()`、`_run_real_b01_b04a()` 均传入 `frozen_git_commit` / `frozen_git_dirty`；
  3. `smoke_b04a_runner_integration.py` 解析一次 git identity 并传入两个 `run_mini_b04a()` 调用点；
  4. 新增 `TestR05B04ANormalSuccessGitIdentity`（2 个测试）：`call_count == 1` + DONE bundle carrier 审计（run-level JSON + per-seed checkpoint pt identity）；
  5. R04 断言 `>= 1` → `== 1`；R04 real-B01 fixture `freeze._test_rows is None` 断言 TEST=0；
  6. 文档状态 → `IDENTITY_FIX_ITERATE_R05 / READY_FOR_REVIEW`。
  12 个 R04+R05 定向测试全部通过。
- Identity carrier 修复 R05 Codex 独立验收（2026-08-30）：`ACCEPT`。Reviewer 独立得到 R05 2/2、B04A integration 129/129、B04 Mini 167/167、Markdown 6/6、protocol validator 30 OKs/0 errors，`py_compile` 与 `git diff --check` 通过；写作模式 synthetic DONE bundle 共审计 54 个 identity carriers（8 run-level JSON + 18 CSV sidecar + 18 checkpoint + 10 log 首行），6 个 strict run identity 字段 0 mismatch。真实 B01 fixture 保留结构性 TEST manifest，但 `load_test=False` 且 `_test_rows is None`，未读取 TEST labels。
- 下一 Gate：合并 identity 修复并冻结新的 `main` Git SHA；随后另立 R03 run-preparation/Owner-authorization record 与全新 EXP-ID。当前仍为 `GPU_R03_NOT_AUTHORIZED`，`B07` 继续 `BLOCKED_BY_B04A`，任何 TEST 仍禁止。
- R03 运行准备（`TASK-SLP-B04A-R03-RUN-PREPARATION-v0.1`，2026-08-30）：已冻结 proposed EXP-ID `EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-AUTODL-R03`、candidate code SHA `bdd4c3d5...`、config/archive/freeze/split hashes、AutoDL no-training Preflight 和唯一正式启动/恢复命令。当前 `PREPARATION_COMPLETE / READY_FOR_AUTODL_PREFLIGHT / GPU_R03_NOT_AUTHORIZED`；任务发布后须重新冻结 clean released SHA，Preflight 经 Codex 复核后仍需 Owner 单独授权。
- R03 正式运行与 budget carrier 修复（2026-08-30）：Owner 授权后，`EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-AUTODL-R03` 在 RTX 4090、Git `f0fac823...`、TEST=0 下终态 DONE；9/9 单元、18 checkpoint、9/9 reload 一致，候选晋级记录为 DeepLabV3+-lite 与 ResUNet-lite。Reviewer 发现原始 run-level `budget_report.json` 使用 placeholder `elapsed_total_seconds=8100.0`、`peak_cuda_mb=0.0`；`TASK-SLP-B04A-BUDGET-REPORT-FIX-v0.1` 已改为实测 wall clock 与全部 per-seed 最大 CUDA peak，并对缺失/非法值 fail closed。原始 R03 证据不可变；下一 Gate 由 Owner 决定 `ACCEPT_WITH_LIMITATIONS / ITERATE / NEW_EXP_REQUIRED`，B07/TEST 仍未授权。

### TASK-SLP-B05：遮盖条件压力测试

- 状态：`HOLD_BY_MISSING_COVER_GT`。
- 比较：uncover、cover1、cover2、cross-cover。
- 输出：区域性能下降、拒识率和错误案例。

### TASK-SLP-B06：有限融合 Mini

- 状态：`HOLD / DIFFERENT_TRACK`。
- 候选：PM+Depth、PM+IR；三模态需额外理由。
- Gate：增益小于冻结 margin 时保留简单模型。

### TASK-SLP-B07：Full 协议冻结

- 状态：`PROTOCOL_ACCEPTED / COMPUTE_NOT_RUN / TEST_DENIED`；B04A R03 已由 Owner `ACCEPT_WITH_LIMITATIONS`，前置 Gate 解除；B05/B06 为可选独立路线。
- 目标：冻结 folds、候选、预算、指标、选择规则、资源和停止条件。
- 证据硬 Gate：运行 manifest 必须内嵌 `experiment_id`、`git_commit`、`git_dirty`、`config_sha256`、`data_manifest_sha256`、`split_sha256` 和 `model_version`；不得依赖外部终端记录补 Git SHA。
- 禁止：协议与 Full 运行同一任务完成。
- 当前草案：91 development subjects / 4,095 samples 固定 5-fold；DeepLabV3+-lite + ResUNet-lite；seeds `[42,123,2026]`；30 units；pooled OOF primary；15 min/unit、450 min total、8192 MiB；TEST=0。入口：`configs/experiments/slp8_pm_full_protocol_v0.1.json`、`configs/experiments/slp8_pm_full_folds_v0.1.json`、`docs/tasks/TASK_SLP_B07_FULL_PROTOCOL_FREEZE_v0.1.md`。

### TASK-SLP-B08：Full Runner 与一折预检

- 状态：`ACCEPT / R03_ONE_FOLD_PREFLIGHT_PASSED / TEST_DENIED`（2026-09-01）。
- 目标：独立 Reviewer 验收；随后冻结 clean SHA/新 EXP-ID，由 Owner 单独决定 RTX 4090 one-fold preflight 授权。
- 实现交付：
  - `src/topper_perception/neural/slp8_region_full.py`（~2400 行）：真实 B01 TRAIN+VAL runner（`load_b01_freeze_tables(..., load_test=False)` + `_test_rows is None` 断言 + subject fold 路由 + fold-TRAIN-only normalization/class weights）；`Slp8RegionDataset` 真实训练（B04A 原语复用待后续任务）；OOF 逐样本 predictions（`unit_oof.csv` + `merge_seed_oof` pooled 重算）；synthetic/real 路径分离守卫；Unit 事务化（atomic `complete.json`、无 `--force`）；真实 resume（identity 验证 + budget accumulator 不归零）；validate-only/no-write 零文件创建。
  - `scripts/run_slp8_region_full.py`：CLI `run_validate_only()` 零文件；`run_synthetic_cpu_smoke()`；`run_full()` 入口。
  - `scripts/smoke_b08_full_runner.py`：独立 standalone smoke。
  - `tests/test_slp8_region_full.py`：74 passed；含 bounded one-fold、best reload OOF、per-subject、FAILED/STOPPED terminal 与 interruption/resume。
  - `scripts/validate_b07_protocol.py`：已恢复为 HEAD（超出范围）；CRLF worktree 下预期报 SHA mismatch。
- Synthetic CPU smoke 结果：30/30 DONE；22.7s；winner 选中；budget OK；TEST=0 declarative policy；temp dir 清理完毕。
- 测试结果：B08 74 passed；B04/B04A core+models+links 347 passed；py_compile、working/staged diff check 干净。B07 独立 validator 在 Windows CRLF checkout 报 worktree-byte SHA mismatch；validator 文件保持 HEAD，B08 committed-blob SHA tests PASS。
- 2026-08-31 R01 真实 RTX 4090 one-fold 在首 epoch 前失败：real loader 返回
  `RegionSample`，但 guard 错误调用 `.get()`；旧 preflight 外层也未为该异常写根级
  terminal。Round 7 本地修复两项合同并新增回归，B08 **76 passed**；R01 保留，
  runner 已冻结为 `e0ba25a9aa0b33be971327ca398d822f7c7d1c8a`；R02 结果见
  下一条，Full 与 TEST 仍未授权。
- 2026-08-31 R02 在 Torch 2.13 strict deterministic CUDA 的首个 NLLLoss2d
  forward 失败（zero optimizer steps）；唯一 FAILED terminal 生效，但空 OOF
  stack 二次异常覆盖原始 error。Round 8 已本地实现数学等价 deterministic
  cross entropy、失败 OOF 保护与 DONE coverage guard；R03 runner 已冻结为
  `02fb364902736a64ee8708440f0dd0bdddf860bc`。
- 2026-09-01 R03 RTX 4090 one-fold 完成：唯一 DONE，best epoch 22，wall
  155.329s，peak CUDA 368.764 MiB，reload consistent；TRAIN 3240/72 subjects，
  VAL OOF 855 unique samples/19 subjects，TEST=0。archive SHA
  `55aa5a4c708a1ba9b2f9ee89a8d05d937e61f4edcd02bdf83b1f5600b478a298`
  本地匹配并通过 `LOCAL_ARCHIVE_AUDIT_PASSED`；checkpoint/OOF/archive 不入 Git。
- 关键设计决策：
  - Runner 使用 `committed_file_sha256(repo_root, relative_path)` via `git show :path`（committed tree，非工作树 CRLF）
  - `partition_records_for_fold(records, *, val_subject_ids)` 按 B07 fold subject 路由，TRAIN/VAL overlap=0
  - OOF：real mode 逐样本 CSV（sample_id / subject_id / fold_id / seed / candidate / predicted_class）
  - Unit 事务：成功写 `complete.json` 后才改 terminal state；已有 DONE.json 拒绝覆盖
- Gate：B08 已验收；下一步仅为 B09 Full 运行准备与独立 Owner 授权。
- 严禁：未经新授权运行 30-unit Full、TEST evaluation 或 `enable_test_access`；不得提交 archive/checkpoint/OOF/raw data。

### TASK-SLP-B09：Full 公平比较

- 状态：`COMPLETE / OWNER_ACCEPTED_WITH_LIMITATIONS / TEST_DENIED`
  （2026-09-02）。
- R01：`8b3ebda`、RTX 4090、30/30 DONE、TEST=0、budget OK；开发期 winner 为
  `slp8_deeplabv3plus_lite_v0.1`（mean pooled IoU `0.494134`），ResUNet-lite 为
  `0.456795`。
- 原始证据限制：30 个 `units/*/complete.json.identity` 均漏写
  `git_dirty=false`；冻结 SHA 上 audit-only 为 27 OK / 30 ERR。run-level identity
  正常，缺陷不进入训练/OOF/指标计算；原始证据不得修补。
- writer 已补入冻结 `config.git_dirty`，30-unit carrier 与 resume 回归覆盖；B08/B09
  相关测试 178 passed。详见
  `stage_reports/S2_B09_SLP8_PM_ONLY_FULL_R01_RESULTS_AND_IDENTITY_FIX_v0.1.md`。
- 目标：真实 subject-isolated Full。
- 输出：开发期 subject-isolated OOF/VAL predictions、逐区域/受试者、模型和日志；默认 TEST=0。
- 停止：OOM、NaN、split leak、manifest mismatch、reload mismatch。
- Owner 已接受 R01 的永久 provenance 限制，不要求 R02；下一 Gate 为 B10。
  TEST 继续禁止。

### TASK-SLP-B10：UNKNOWN/REJECT

- 状态：`ACCEPT_WITH_LIMITATIONS / HARD-CONSENSUS-ONLY / TEST_DENIED`。
- B09 OOF 只含三 seed hard masks，不含 logits/probabilities；因此本轮严格命名为
  seed-consensus，不声称概率 confidence、校准、alignment failure 或真实 OOD。
- 3/3 一致覆盖 87.3766%，错误率 8.7916%；拒绝 12.6234% 分歧像素捕获
  44.5922% 的多数票错误，但 55.4078% 的错误仍为三 seed 一致错判。
- 输出：`stage_reports/S2_B10_SLP8_OOF_CONSENSUS_REJECT_v0.1.md`；ignored
  `outputs/analysis/slp_b10_consensus_r01/` 下含 summary、91-subject 与 top-50
  高一致性错误表。

### TASK-SLP-B11：SLP 研究候选冻结

- 状态：`CANDIDATE_CONTRACT_ACCEPTED / FINAL_FIT_NOT_RUN / TEST_DENIED`。
- 已冻结 `slp8_pm_research_candidate_v0.1`：DeepLabV3+-lite、danaLab/uncover
  pressure-only、192×84 九类 mask。未来 B11F 在全部 91 development subjects 上
  训练 seeds 42/123/2026，固定 epochs 15/20/12；当前 GPU/final checkpoint 均未运行。
- primary 为三模型 hard majority；3/3 可选 `UNKNOWN_REGION` 拒识仅为有限研究信号，
  不是概率 confidence、OOD 或安全机制。
- 禁止：宣称气囊控制、舒适性或产品效果。

### TASK-SLP-B11F：最终开发集拟合

- 状态：`R01_FAILED_BEFORE_TRAINING_LOOP / RUNTIME_FIX_READY_FOR_REVIEW / GPU_NOT_AUTHORIZED / TEST_DENIED`。
- 已实现冻结 DeepLabV3+-lite 在全部 91 development subjects / 4,095 TRAIN+VAL
  samples 上的三 seed final-fit 入口，固定 epochs 15/20/12，不使用 early stopping，
  不产生伪 validation 分数。
- 真实入口要求 clean SHA、Owner `--run-authorized`、新 B11F EXP-ID、全新输出目录
  和 CUDA；B01 始终 `load_test=False` 且 `_test_rows is None`。
- 每 checkpoint 携带完整 identity 与 TEST=0 carrier，并经固定 audit batch 独立重载
  一致性检查。首轮独立审查的 AdamW、B01 binding、determinism、identity/resume、资源
  载体问题已修。第二轮审查的 resume 状态、显式授权、RNG、CUDA 初始化、环境持久化
  问题已完成 R02；R03 补 environment SHA 跨 STOPPED/resume/DONE 绑定、篡改拒绝与参数一致性
  回归；R04 再补 RUNNING 恢复环境 preflight、既有 `final.pt` 拒绝覆盖、单根状态原子迁移、
  真实 DataLoader shuffle RNG 与 CUDA 顺序回归。R05 持久化并恢复最后 epoch loss、累计
  wall time 与历史 peak，逐 batch 和 reload 后核验显存上限，并在 completion/DONE 前执行
  有限值与严格 JSON 门禁。R05 独立只读复审结论为 `ACCEPT`，无 P0/P1/P2；定向
  31 passed、联合 114 passed，有限 final-epoch resume 与 NaN fail-closed 探针通过；
  TEST=0，GPU NOT RUN。
- 运行准备 R01 独立审查结论 `ITERATE`：首次启动未绑定 Owner-authorized environment，
  手工剩余分钟不能证明跨恢复累计 45 分钟。当前不授权 AutoDL preflight/GPU/TEST。
- 两项 P1 已在 release `9af268fa168207a269abbef22e522ac04fd6b6c5` 修复：新增 no-training
  canonical environment fingerprint 并在首次 output 前 fail-closed 核验；新增原子
  `budget.json`，固定首次 UTC start/deadline/core SHA，停机时间计入且恢复不重置 2,700 秒。
  缺失/漂移授权、budget 篡改、deadline 刷新、过期与时钟回退反例覆盖；定向 40 passed、
  联合 123 passed。该时点 proposed EXP-ID 尚未 QUEUED，Owner authorization 尚为 PENDING；
  后续 preflight 与精确授权记录见下项。
- 运行准备 R02 与 AutoDL no-training preflight 均已 `ACCEPT`。AutoDL 通过完整历史
  Git bundle（SHA-256 `654bba6e...b0f39b0b50`）取得 runner；RTX 4090、三项输入 hash、
  正式输出缺失、validator/validate-only、TEST=0 均通过；授权 environment fingerprint 为
  `a5a9342b...ba045669`。
- Owner 已于 `2026-09-03T18:40:09+08:00` 授权精确三 seed final fit；下一 Gate：
  R01 在首个 training batch 前因 class-weight NumPy→Torch 接口错误失败；旧 EXP-ID 与授权
  均不可复用。修复已与 B09 转换路径对齐并以真实 NumPy mock 回归，定向 40、联合 123
  passed。R01 原始 `FAILED/environment/budget` JSON、root inventory、文件 SHA 清单与 operator
  traceback 已同步至 `docs/evidence/b11f_r01_failure`，等待独立复审。下一 Gate 为
  `B11F_RUNTIME_FIX_EVIDENCE_READY_FOR_REVIEW / GPU_NOT_AUTHORIZED / TEST_DENIED`；三个
  final checkpoint 审计完成前 B09T 继续阻塞。

### TASK-SLP-B09T：一次性最终 TEST 评价

- 状态：`BLOCKED_BY_B11F_FINAL_FIT_AND_SEPARATE_OWNER_AUTHORIZATION`。
- 目标：候选、代码、超参数、阈值、选择规则和报告模板全部冻结后，对 B01 TEST subjects 执行一次最终评价。
- 授权：必须由 Owner 单独授权 `purpose="final_evaluation"`，并由 Runner 显式 `load_test=True` 重新加载。
- 禁止：根据 TEST 结果回到 B04A/B07 调参或改候选；如发现协议/实现缺陷，原 TEST 结果标记 INVALID，修复和任何重测必须新 TASK/EXP 并单独说明污染风险。

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
