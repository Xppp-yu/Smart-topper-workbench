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
```

纯整数算术，**不使用 Python `hash()`，不使用进程级随机状态**，保证跨机器可复现、可逐行审计。split manifest 记录自身 SHA-256（对内容做 canonical JSON 哈希），运行前后均可复核。

## 6. 主指标与产物

- **主指标**：`record_macro_f1_mean`（record = 10 个 snapshot 概率取平均后 argmax，与 P5.1 口径一致）。
- 同时输出：record/snapshot 的 macro-F1 + balanced-accuracy + accuracy（3 repeats mean±std）、per-repeat/per-fold 指标、逐类别 precision/recall/F1、逐受试者 record accuracy + macro-F1（含 worst subject）、混淆矩阵、record-level multiclass NLL、Brier score、ECE（15 equal-width bins）、param count、checkpoint size、per-fold 训练时间、推理时间（含样本数）、峰值 CUDA 显存、epoch 选择 + refit 全量日志、OOF snapshot 预测、record-level 预测、checkpoint reload 一致性。
- 校准诊断**仅作诊断**，不做 temperature scaling，不改排名。

## 7. 最终选择规则（固定）

1. gate/evidence 失败者排除；
2. 主准则 = `record_macro_f1_mean`；
3. 差距 ≤ `margin=0.005` 视为近似并列（near-tie）；
4. 并列裁决：先 `record_balanced_acc_mean`，再 `worst_subject_macro_f1_mean`；
5. NN vs SVM：NN 高于 SVM **超过** 0.005 方可按主准则胜出；否则（near-tie 内）默认偏好 SVM，除非 NN 呈现预注册的实质性改进——(a) worst-subject macro-F1 绝对提高 ≥0.02，(b) 最弱类别 record F1 绝对提高 ≥0.01，(c) record macro-F1 标准差绝对降低 ≥0.001——且不落后 SVM 超过 0.005；
6. calibration / param count / inference / training 时长均报告但不改变排名；
7. **Reviewer 接受前不冻结候选**。

> 边界语义：`>` 0.005 才“凭主准则胜出”；恰好 0.005 属 near-tie，默认回落到 SVM。

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
| `tests/test_neural_full_protocol.py` | 冻结配置通过 schema + 冻结值校验；manifest SHA / 边界 / 候选 / seeds / lr / monitor / margin / device / model_configs 漂移逐一 fail-closed；数据边界三态与跨一致性；选择规则（NN 超 margin、near-tie 默认偏好 SVM、worst-subject/weakest-class/std 三类实质改进、bal-acc 阶梯、gate 排除、SVM 唯一性）；runner stub `NotImplementedError` |
| `tests/test_neural_full_splits.py` | 外层 seed 取值、inner seed 公式、inner 校验折规则、确定性（无进程随机）；manifest 形状/SHA 确定性/逐 repeat 受试者分区/隔离不变量/记录种子；篡改破坏 SHA、outer 重叠、inner 逃逸三者 fail-closed |

测试只使用构造的 60 个受试者 ID + 冻结配置，不读完整 PoPu 矩阵、不训练、不触发 GPU。

结果：`uv run pytest -q` → **390 passed**（含本轮新增 44 项），`git diff --check` 通过。

## 11. 已知限制

- 本文档与配置**冻结协议**，不代表任何 Full 结果；P5.2-C Full **尚未运行**。
- 全量数据边界（60/5,006/50,060）是 P2 质量 manifest 冻结口径的**预期**值，运行期仍按 manifest 实际读取并 fail-closed 校验。
- 未冻结具体 Full 执行时间与 AutoDL 节点；GPU 授权与 runner 实现属于后续独立任务。

## 12. 不能得出的结论

- 不得宣称任何候选已通过 Full 公平比较或已被冻结为总体冠军；
- 不得把本协议视为已产生真实 Full 结果；
- 不得据此宣布 CNN/MLP/SVM 谁是 PoPu 总体最优候选。
