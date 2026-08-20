# P5.2-C/R5.2-C — PoPu 神经网络 Full 公平比较协议（冻结）v0.1

## 1. 结论

**状态：PROTOCOL_FROZEN — PENDING_REVIEW（协议/配置/校验测试已冻结，尚未运行）。**

本任务 `TASK-P5.2-C1-FULL-PROTOCOL-v0.1` **只冻结** P5.2-C Full 公平比较的协议、配置、校验测试与状态记录，**没有**实现 Full runner，**没有**运行任何真实 Mini/Full，**没有**连接或启动 AutoDL。**P5.2-C Full 尚未运行。**

冻结内容：`configs/experiments/popu_neural_full_v0.1.json`、本协议文档、纯 stdlib/NumPy 校验模块 `neural/full_protocol.py` 与 `neural/full_splits.py`、以及对应的单元测试。`runner_type` 新增 `popu_neural_full` 仅为让冻结配置可被治理 runner 校验，其执行入口是**失败即关的占位 stub**（`NotImplementedError`），确保任何误触 Full 执行都会立即报错而不是假装跑过。

P5.1 `calibrated_linear_svm`、P5.2-A CPU/CUDA Smoke、P5.2-B Mini 的历史证据均未修改或覆盖；P5.1 冻结的 SVM 候选也未重训/覆盖/重冻结。

## 2. 冻结的 Full 协议

| 项 | 冻结值 | 说明 |
|---|---|---|
| scope / runner_type | `full` / `popu_neural_full` | 治理 runner 在 QUEUED 前要求干净 Git worktree；runner_type 执行入口为失败即关 stub |
| 数据集 | PoPu Tactilus | 只读；不复制原始数据入仓 |
| cohort | `primary`（ACCEPT-only） | 冻结 P2 质量 manifest 为唯一 cohort 来源；WARN/EXCLUDED/REJECT 在建样本前丢弃 |
| 质量 manifest | `outputs/metrics/popu_tactilus_quality_results_v0.1.csv` | 顶层 `data_manifests` + `parameters.quality_manifest_sha256` 双重固定 |
| manifest SHA-256 | `9d3398a587b183f7e27ea68ada2eda1e5e82ebadb2ac9caf7a74b5763d3e954c` | 读取前校验，缺失/不匹配立即失败 |
| 全量数据边界 | **60 受试者 / 5,006 记录 / 50,060 snapshot / 每记录 10 snapshot** | 加载后校验，任一计数不符即失败（fail-closed）；无 WARN/EXCLUDED/REJECT |
| 候选模型 | `matrix_mlp` / `tiny_cnn` / `small_resnet` | 不因 Mini 指标淘汰任何 NN 候选 |
| 传统对照 | P5.1 `calibrated_linear_svm` | **不重训/覆盖/重冻结**；仅作冻结参考 |
| 禁用 | 无 CNN+工程特征、无额外架构 | 不新增候选或特征工程 |
| 全局 seed | `42` | 与 outer fold seeds 无关 |

### 2.1 冻结训练参数

三候选（`matrix_mlp` / `tiny_cnn` / `small_resnet`）使用**完全相同**的训练参数，逐项冻结并 fail-closed 校验：

| 参数 | 冻结值 |
|---|---|
| batch_size | `32` |
| num_workers | `0` |
| loss | `cross_entropy` |
| optimizer | `AdamW`（`lr=1e-3`、`weight_decay=1e-4`） |
| deterministic cudnn | `true`（`torch.backends.cudnn.deterministic=True`） |
| cudnn benchmark | `false`（`torch.backends.cudnn.benchmark=False`） |
| AMP | 当前 torch `autocast` + `GradScaler` 路径 |
| 冻结标签顺序 | `["empty", "supine", "prone", "left", "right"]` |

确定性训练 seed（每个 outer fold、纯算术派生，见第 5 节）：
- `stage_a_train_seed = 2_000_000 + outer_seed * 100 + local_fold`
- `stage_b_refit_seed = 3_000_000 + outer_seed * 100 + local_fold`
- **每个候选开始训练前必须重新 `set_seed`**；三候选在同一 fold 使用相同派生 seed；两个 seed 均写入 split manifest。

### 2.2 冻结 SVM 参考证据绑定

P5.1 `calibrated_linear_svm` 的六份历史证据文件**只读绑定**（路径 + SHA-256 + 大小），本任务只计算并记录哈希、**不修改这些文件**。未来 Full runner 在读取前必须校验每个哈希，且只使用其中 `calibrated_linear_svm` 行：

| 文件 | SHA-256（前 12 位） | 大小 (B) |
|---|---|---|
| `data/processed/popu/popu_model_comparison_p5_1_oof_predictions_v0.1.csv` | `807afca919b7` | 191,504,818 |
| `outputs/metrics/popu_model_comparison_p5_1_record_level_v0.1.csv` | `13aafaaf048b` | 10,798,258 |
| `outputs/metrics/popu_model_comparison_p5_1_summary_v0.1.csv` | `8a637809f1d2` | 2,281 |
| `outputs/metrics/popu_model_comparison_p5_1_fold_repeat_v0.1.csv` | `a7c34ca17ec2` | 7,375 |
| `outputs/metrics/popu_model_comparison_p5_1_per_class_v0.1.csv` | `7ec36b48ffad` | 5,349 |
| `outputs/metrics/popu_model_comparison_p5_1_per_subject_v0.1.csv` | `3b1f757ee17f` | 49,669 |

## 3. 外层公平评估协议（复用 P5.1）

- `kind=repeated_subject_grouped_cv`，`group=subject_id`，`n_splits=5`，`n_repeats=3`，`shuffle=true`，`seeds=[11,22,33]`。
- **seeds 即 repeats**：seeds 列表中的每个 seed 生成一套 subject-grouped fold set，恰好 3 个 repeat；所有候选共享同一套外层受试者折。
- 同一受试者绝不同时出现在 train 与 test；每个受试者每个 repeat 恰好被 OOF 验证一次。
- 指标先按 repeat 汇总，再取 mean/std；**绝不合并 150,180 个 OOF snapshot**。
- **PoPu 无可重命名的 never-seen 外部测试集**：只做 cross-validation 估计，不宣称外部泛化。
- 生成可审计切分 manifest（见第 5 节），记录 repeat / outer seed / local fold / train+test 受试者 / inner train+val 受试者 / 派生种子 / manifest SHA-256。

## 4. 神经网络内部选 epoch（两阶段，无泄漏）

外层测试折**绝不**用于早停、超参、checkpoint、normalization、calibration。

**Stage A（epoch 选择，每个 outer fold）**

- 在 outer-train 内做确定性 subject-grouped inner split，`inner_n_splits=4`，inner seed 由 `outer_seed + local_fold` 派生（见第 5 节）。
- 一个 inner fold 作 validation（`inner_validation_fold = local_fold % inner_n_splits`），其余作 inner training。
- normalization **仅在 inner training** 上 fit；左右翻转增强（左↔右标签交换）**仅 inner training**。
- `max_epochs=15`、`min_epochs=5`、`patience=3`、`monitor=val_loss`、`optimizer={lr:1e-3, weight_decay:1e-4}`。
- **不做超参搜索**。

**Stage B（outer refit）**

- 读取 Stage A 选出的 `best_epoch`，重新初始化模型。
- 用**全部 outer-train 受试者**重新 fit normalization（仅 outer-train）；左右翻转增强仅 outer training。
- 固定 `best_epoch` 训练，随后**对 outer-test 仅推理一次**。

## 5. 种子派生公式（确定性，无 hash()/进程随机）

```text
outer_seed(repeat)          = OUTER_SEEDS[repeat] = [11, 22, 33][repeat]
inner_seed(repeat, fold)    = 1_000_000 + outer_seed(repeat) * 100 + local_fold
inner_validation_fold(fold) = local_fold % 4
stage_a_train_seed(repeat, fold) = 2_000_000 + outer_seed(repeat) * 100 + local_fold
stage_b_refit_seed(repeat, fold) = 3_000_000 + outer_seed(repeat) * 100 + local_fold
```

纯整数算术，**不使用 Python `hash()`，不使用进程级随机状态**，保证跨机器可复现、可逐行审计。`stage_a_train_seed`（Stage A epoch 选择）与 `stage_b_refit_seed`（Stage B 外层 refit）均写入 split manifest；**每个候选开始训练前必须重新 `set_seed`**，且三候选在同一 fold 使用相同派生 seed。split manifest 记录自身 SHA-256（对内容做 canonical JSON 哈希），运行前后均可复核。

## 6. 主指标与产物

- **主指标**：`record_macro_f1_mean`（record = 10 个 snapshot 概率取平均后 argmax，与 P5.1 口径一致）。
- 同时输出：record/snapshot 的 macro-F1 + balanced-accuracy + accuracy（3 repeats mean±std）、per-repeat/per-fold 指标、逐类别 precision/recall/F1、逐受试者 record accuracy + macro-F1（含 worst subject）、混淆矩阵、record-level multiclass NLL、Brier score、ECE（15 equal-width bins）、param count、checkpoint size、per-fold 训练时间、推理时间（含样本数）、峰值 CUDA 显存、epoch 选择 + refit 全量日志、OOF snapshot 预测、record-level 预测、checkpoint reload 一致性。
- 校准诊断**仅作诊断**，不做 temperature scaling，不改排名。冻结公式（record-level）：
  - `NLL = -mean(log(clip(p_true, 1e-15, 1)))`
  - `multiclass Brier = mean(sum_k((p_k - y_k)^2))`，其中 `y_k` 为 one-hot 目标
  - ECE：`confidence = max probability`、`correct = (argmax == label)`，15 个等宽 bins，左闭右开、最后一个 bin 包含 1.0，按 bin 样本占比加权 `abs(accuracy - confidence)`
  - 概率行必须 finite、每项位于 `[0,1]`、行和在容差（`1e-6`）内等于 1，否则 fail-closed

## 7. 最终选择规则（固定）

0. **所有排名指标先检查 finite 且范围合理**（NaN/Inf 立即失败；F1/accuracy/worst-subject 落在 `[0,1]`）；
1. gate/evidence 失败者排除；
2. 主准则 = `record_macro_f1_mean`；
3. 差距 ≤ `margin=0.005` 视为近似并列（near-tie）；
4. 并列裁决：先 `record_balanced_acc_mean`，再 `worst_subject_macro_f1_mean`，再 `record_macro_f1_mean`，最后固定 `complexity_priority = calibrated_linear_svm → tiny_cnn → small_resnet → matrix_mlp`（**`complexity_priority` 仅在主指标、balanced accuracy、worst-subject 三者全部相同时才使用**）；
5. NN vs SVM：NN 高于 SVM **超过** 0.005 方可按主准则胜出；否则（near-tie 内）默认偏好 SVM，除非 NN 呈现预注册的实质性改进——(a) worst-subject macro-F1 绝对提高 ≥0.02，(b) 最弱类别 record F1 绝对提高 ≥0.01，(c) record macro-F1 标准差绝对降低 ≥0.001——且不落后 SVM 超过 0.005；
6. calibration / param count / inference / training 时长均报告但不改变排名；
7. **Reviewer 接受前不冻结候选**。

> 边界语义：`>` 0.005 才”凭主准则胜出”；恰好 0.005 属 near-tie，默认回落到 SVM。胜者选择**与候选输入顺序无关**（交换候选顺序仍得到同一 winner）。

## 8. 资源与停止条件

- `device=cuda`、`amp_enabled=true`、`max_cuda_mb=8000`、`max_total_train_seconds=21600`。
- 单 fold 计时预检**单独**进行（在 Full 授权前）。
- 超时即停止并请求 Reviewer；**不自动删模型、不缩小 fold**。
- fail-closed on：OOM / NaN / Inf / fold leak / manifest mismatch / git dirty / EXP-ID 已存在 / checkpoint reload 不一致。
- 支持**断点续训且不覆盖已完成 fold**；正式执行须在 screen/tmux 中（本任务禁止启动）。

## 9. 修改 / 新增文件

| 文件 | 改动 |
|---|---|
| `configs/experiments/popu_neural_full_v0.1.json` | 新增（冻结 Full 配置） |
| `docs/stage_reports/P5_2_C_POPU_NEURAL_FULL_PROTOCOL_v0.1.md` | 新增（本协议） |
| `src/topper_perception/neural/full_protocol.py` | 新增（常量/边界校验/配置校验/选择规则，纯 stdlib） |
| `src/topper_perception/neural/full_splits.py` | 新增（种子派生/切分 manifest/隔离校验，纯 NumPy+stdlib） |
| `src/topper_perception/experiments/contracts.py` | `RUNNER_TYPES` 增加 `popu_neural_full` |
| `src/topper_perception/experiments/runner.py` | 注册 `run_popu_neural_full`（`NotImplementedError` 占位 stub） |
| `configs/experiments/schema/experiment_v0.1.schema.json` | runner_type enum 增加 `popu_neural_full` |
| `tests/test_neural_full_protocol.py` | 新增（配置/边界/选择规则/runner stub） |
| `tests/test_neural_full_splits.py` | 新增（种子/切分 manifest/隔离） |
| `docs/PROJECT_STATUS.md` | P5.2-C 行更新为 `PROTOCOL_FROZEN_PENDING_REVIEW` |

**没有**实现 Full runner；schema/contracts 仅做最小枚举扩展。

## 10. 测试

| 文件 | 覆盖 |
|---|---|
| `tests/test_neural_full_protocol.py` | 冻结配置通过 schema + 冻结值校验；manifest SHA / 路径 / 边界 / 候选 / seed / dataset / quality_manifest / kind / shuffle / split_manifest / stage A/B / outer_refit / model_selection / resources / model_configs / optimizer / training_params / training_seeds / frozen_svm_reference_artifacts / calibration / complexity_priority 漂移逐一 fail-closed（含严格 bool 类型校验）；数据边界三态与跨一致性；选择规则（NN 超 margin、near-tie 默认偏好 SVM、worst-subject/weakest-class/std 三类实质改进、bal-acc 阶梯、gate 排除、SVM 唯一性、**NaN/Inf/越界立即失败、complexity_priority 仅在主指标/bal-acc/worst 全相同时使用、主指标打破并列先于 complexity、交换候选顺序 winner 不变**）；校准公式（NLL clip、Brier、ECE、概率行 finite/范围/行和 fail-closed、**空输入与 n_bins≠15 fail-closed**）；runner stub `NotImplementedError` |
| `tests/test_neural_full_splits.py` | 外层 seed 取值、inner seed 公式、**stage_a/stage_b 训练 seed 公式与确定性**、inner 校验折规则、确定性（无进程随机）；manifest 形状/SHA 确定性/逐 repeat 受试者分区/隔离不变量/记录种子（**含 stage_a·b 训练 seed**）；篡改破坏 SHA、outer 重叠、inner 逃逸、**篡改 seed/header/outer_seeds + 重算 SHA、重编号 fold + 重算 SHA 均 fail-closed** |

测试只使用构造的 60 个受试者 ID + 冻结配置，不读完整 PoPu 矩阵、不训练、不触发 GPU。

结果：`uv run pytest -q` → **444 passed**（含本轮新增 13 项），`git diff --check` 通过。

## 11. 已知限制

- 本文档与配置**冻结协议**，不代表任何 Full 结果；P5.2-C Full **尚未运行**。
- 全量数据边界（60/5,006/50,060）是 P2 质量 manifest 冻结口径的**预期**值，运行期仍按 manifest 实际读取并 fail-closed 校验。
- 未冻结具体 Full 执行时间与 AutoDL 节点；GPU 授权与 runner 实现属于后续独立任务。

## 12. 不能得出的结论

- 不得宣称任何候选已通过 Full 公平比较或已被冻结为总体冠军；
- 不得把本协议视为已产生真实 Full 结果；
- 不得据此宣布 CNN/MLP/SVM 谁是 PoPu 总体最优候选。
