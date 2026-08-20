# P5.2-B/R5.2-B — PoPu 神经网络 Mini 筛选协议与代码就绪报告 v0.1

## 1. 结论

**状态：COMPLETE — MINI_ACCEPTED（协议已就绪并经真实 Mini 运行验收；运行结果见 [P5.2-B 结果报告](P5_2_B_POPU_NEURAL_MINI_RESULTS_v0.1.md)）。**

> 本文档原为协议与代码就绪报告（`MINI_READY_TO_RUN — REVIEWER_ACCEPTED`）。后续已按本协议在 AutoDL 上真实运行 Mini（`EXP-P5.2-B-MINI-SCREEN-20260819-R01`）并经 Reviewer 独立复核接受，P5.2-B 正式状态更新为 `COMPLETE — MINI_ACCEPTED`。协议内容保持不变。

首轮 Mini 协议与实现已提交，Reviewer 只读复核后返回 `REVIEW_NEEDS_FIX`；已按首轮 6 项要求与后续数据 Manifest SHA-256 哈希校验共 7 项修复，全部以新增提交完成（不重写历史，最终修复提交 `4b8b73e`）。修复项：① 主 cohort 明确为 ACCEPT-only（`primary`），并真正读取冻结的 P2 质量 manifest 过滤 WARN/EXCLUDED；② 冻结配置 `device` 由 `auto` 改为 `cuda`，CUDA 不可用时直接失败、不静默回退 CPU；③ 切分改为显式 `train_subject_ids`/`val_subject_ids`（≥2 验证受试者），不再由 ratio 派生；④ 早停只允许 `monitor=val_loss`，拒绝 NaN/Inf metric 与 `min_delta`；⑤ 可行性门：CUDA 缺 `peak_cuda_mb` 记为 `needs_fix`、`overall_verdict` 拒绝未知 verdict 字符串；⑥ 文档在 Reviewer 重新接受前标记 `REVIEW_NEEDS_FIX`；⑦ 冻结配置记录 P2 质量 manifest 的 SHA-256，Mini runner 在读取前校验、manifest.json 记录数据 Manifest 哈希（详见第 5.1 节）。Reviewer 最终复核已接受（`REVIEWER_ACCEPTED`），P5.2-B 正式标记为 `MINI_READY_TO_RUN`。

**本协议/实现任务（`4b8b73e`）没有运行任何真实 Mini 或 Full 实验**——当时 AutoDL 处于关闭状态，该任务禁止连接或启动服务器，也未读取完整 PoPu 数据训练。真实 Mini 由后续 Experiment Runner 在 AutoDL 上执行（`EXP-P5.2-B-MINI-SCREEN-20260819-R01`）并经 Reviewer 接受，结果见 [P5.2-B 结果报告](P5_2_B_POPU_NEURAL_MINI_RESULTS_v0.1.md)。P5.2-A CPU/CUDA Smoke 保持 COMPLETE；P5.1 `calibrated_linear_svm` 传统候选与 P5.2-A CPU Smoke、CUDA R01/R02 历史产物均未修改或覆盖。

### 1.1 Reviewer 最终复核证据

- 复核对象：HEAD commit `4b8b73e`（叠加在首轮修复提交之上，未重写历史）。
- 测试：`python -m pytest -q` → **346 passed**，`git diff --check` 通过。
- 数据 Manifest：`outputs/metrics/popu_tactilus_quality_results_v0.1.csv`，SHA-256 `9d3398a587b183f7e27ea68ada2eda1e5e82ebadb2ac9caf7a74b5763d3e954c`，文件大小 1,187,628 bytes，读取前校验通过。
- ACCEPT-only cohort 只读预检：516 条记录考虑，11 条排除，505 条 ACCEPT 保留，5,050 个选中 snapshot（505 × 10）。
- 无阻断 Mini 的问题，P5.2-B 获准标记为 `MINI_READY_TO_RUN`。

当时（协议/实现任务完成时）的下一交接点是 **部署 Git bundle + 单独上传 P2 质量 manifest 并校验 SHA-256**之后，由 Experiment Runner 在 AutoDL 上执行 Mini Run；该步骤后来已由 `EXP-P5.2-B-MINI-SCREEN-20260819-R01` 完成并经 Reviewer 接受，结果见 [P5.2-B 结果报告](P5_2_B_POPU_NEURAL_MINI_RESULTS_v0.1.md)。

## 2. 冻结的 Mini 协议

| 项 | 冻结值 | 说明 |
|---|---|---|
| scope | `mini` | 由治理 runner 强制要求干净 Git worktree 后才可 QUEUED |
| 开发受试者子集 | `["1","2","3","4","5","6"]` | 固定、数字序、**在看任何模型结果之前**冻结；不针对模型输出挑子集 |
| cohort | `primary`（ACCEPT-only） | 冻结 P2 质量 manifest（`popu_tactilus_quality_results_v0.1.csv`）为唯一 cohort 来源；WARN/EXCLUDED/REJECT 在建样本前被丢弃，绝不误入 primary |
| 数据 manifest 哈希 | `9d3398a5…3e954c`（SHA-256） | 冻结配置同时以顶层 `data_manifests` 与 `parameters.quality_manifest_sha256` 固定该哈希；Mini runner 读取 manifest 前先校验，缺失或不匹配立即失败 |
| 切分 | 受试者隔离、**显式** `train_subject_ids=["1","2","3","4"]`、`val_subject_ids=["5","6"]`（≥2 验证） | 不在运行期由 ratio 派生；同一受试者的记录不跨 split；无 test split |
| device | `cuda` | CUDA 不可用时**直接失败**（`resolve_device` 抛错），不静默回退 CPU；CPU 测试显式 `device=cpu` |
| seed | `42` | 固定；同 config+seed 复现 |
| epochs | `5`，`early_stopping.min_epochs=3` | 实际 3–5 epochs |
| early stopping | `monitor=val_loss`（**只允许 val_loss**）、`mode=min, patience=2, min_delta=0.0, min_epochs=3` | **只读验证指标**，永不读 test；`monitor≠val_loss`、NaN/Inf metric、非有限/负 `min_delta` 一律拒绝 |
| best checkpoint | `argmin(val_loss)`，改善判定 `< best - min_delta`，平局取最早 epoch | 固定规则写入 resolved config 与 metrics |
| 候选模型 | `matrix_mlp` / `tiny_cnn` / `small_resnet` | 三模型共享同一数据、split、preprocessing、augmentation、eval |
| 预处理 | 仅训练折 fit 的 `MatrixNormalizer`；左右翻转增强（仅训练折，左↔右标签交换） | 防泄漏 |
| 传统对照 | P5.1 `calibrated_linear_svm` 保留为冻结传统参考 | **不重训** |

## 3. 记录的能力（训练/评估）

每候选模型输出：

- 逐 epoch `train_loss` / `val_loss` 历史、`val_accuracy`、`val_macro_f1`、`val_balanced_accuracy`、每 epoch 耗时与 `amp_active`；
- `val_accuracy` / `val_macro_f1` / `val_balanced_accuracy` / `macro_precision` / `macro_recall`；
- 逐类别 precision / recall / F1 / support；混淆矩阵（sklearn 口径：行为真值、列为预测）；
- `actual_epochs` / `param_count` / `total_train_seconds` / `device` / `amp_active` / `peak_cuda_mb`；
- checkpoint `latest` / `best`、resume 续训参数变化、独立重载预测一致性、固定 seed 复现。

实现复用 P5.2-A 的 `training.py` / `data.py` / `checkpoint.py` / `dataset.py` / `models.py`，未新建第二条并行训练管线。

## 4. Mini 可行性 Gate

预定义、在结果之前冻结；只判定 **可继续 / 排除 / 需修复**，不给出最终排名，也不偏袒任何来自 P5.2-A 单 epoch 准确率的模型。

- **exclude（候选本身不可行）**：`finite`（损失/指标非有限）或 `learning_signal_ok`（`best_val_balanced_accuracy ≤ 1/5 + 0.05 = 0.25`）。
- **needs_fix（协议/基础设施问题，修复后重跑）**：`no_leakage`、`same_split`、`checkpoint_ok`、`resume_ok`、`reload_ok`、`resource_ok` 任一失败；其中 `resource_ok` 在 `device=cuda` 且缺 `peak_cuda_mb` 时判为失败。
- **proceed**：全部通过。

汇总：`needs_fix` 优先；全 `exclude` 才 `exclude`；否则 `proceed`。`overall_verdict` 只接受 `proceed/exclude/needs_fix` 三种字符串，遇到未知 verdict 直接抛 `ValueError`（不静默放行）。学习信号阈值仅由 chance（1/5）加 margin 推导，对三模型一致，与 P5.2-A 的 smoke 准确率无关。

## 5. 配置与治理

- 版本化配置：[`configs/experiments/popu_neural_mini_v0.1.json`](../../configs/experiments/popu_neural_mini_v0.1.json)
- EXP-ID：`EXP-P5.2-B-MINI-SCREEN-20260819-R01`；TASK-ID：`TASK-P5.2-B-MINI-SCREEN-v0.1`（符合治理格式）。
- 输出仅写 `outputs/experiments/<EXP-ID>/`；不修改 `configs/paths.local.json`、不复制原始数据入仓。
- 治理 runner 保留 Git SHA/dirty、Python/PyTorch/CUDA/GPU/seed/cmdline/time；**已存在 EXP-ID 时拒绝覆盖**（`ExpIdError`）。
- 新增 `runner_type`：`popu_neural_mini`（登记于 `contracts.RUNNER_TYPES`、`runner.RUNNER_REGISTRY`、schema enum），原 `popu_neural` smoke 路径不变。
- 冻结配置以顶层 `data_manifests` 固定 P2 质量 manifest 的路径与 SHA-256；通用治理 runner 在 QUEUED 前校验该文件存在且哈希一致，并把已验证的 `path`/`sha256`/`size_bytes` 写入 `manifest.json` 的 `data_manifests`（满足治理文档「数据 Manifest 哈希」要求）。

### 5.1 数据 Manifest 哈希与 AutoDL 部署

P2 质量 manifest（`outputs/metrics/popu_tactilus_quality_results_v0.1.csv`）是外部数据产物，**不在 Git bundle 中**，也不随代码提交进入仓库。

- **该文件不在 Git bundle 中**：它由 P2 质量门阶段在本地生成，属于 `outputs/` 生成物（`outputs/` 不入 Git）。
- **AutoDL 部署时必须单独上传**：远端执行 Mini 前，须把该文件单独拷贝到远端冻结配置 `quality_manifest` 所指向的路径。
- **上传后先校验 SHA-256 才能执行 Mini**：上传完成后，先计算文件 SHA-256 并与冻结值 `9d3398a587b183f7e27ea68ada2eda1e5e82ebadb2ac9caf7a74b5763d3e954c` 逐字符比对；完全一致才允许 QUEUED 并执行 Mini。Mini runner 本身也会在读取 manifest 前再次校验（缺失或不匹配立即失败），不依赖路径/文件名判断。

## 6. 测试

| 文件 | 覆盖 |
|---|---|
| `tests/test_neural_metrics.py` | accuracy / macro-F1 / balanced accuracy / 逐类别 P/R/F1/support / 混淆矩阵 / 零支撑类 / 非法输入 |
| `tests/test_viability.py` | gate `proceed` / `exclude` / `needs_fix`（含泄漏、checkpoint、resume、reload、同 split、资源超限）、CUDA 缺峰值内存 `needs_fix`、非法输入、`overall_verdict` 含未知 verdict 拒绝 |
| `tests/test_neural_mini.py` | 受试者隔离（显式 split）、三模型共享 split、cohort 排除 WARN/EXCLUDED、epoch history、早停边界、best checkpoint 规则、resume/reload、CPU 下 CUDA 峰值内存字段、gate 集成、经治理 runner 的 mini 运行、EXP-ID 覆盖拒绝、缺数据/缺 manifest 报错、质量 manifest SHA-256 正确/错误/缺失三态、metrics 与 train_log 记录哈希、经治理 runner 的 manifest.json 记录数据 Manifest 哈希 |
| `tests/test_neural_early_stopping.py` | `monitor=val_loss` 唯一性、NaN/Inf metric 拒绝、非有限/负 `min_delta` 拒绝、patience/min_epochs/改善判定 |
| `tests/test_neural_training.py` | `resolve_device("cuda")` 在 CUDA 不可用时抛错（无静默回退） |
| `tests/test_experiment_contracts.py` | `data_manifests` 可选字段、哈希小写归一化、缺 path/缺 sha256/非 64 位/非 hex 等非法输入拒绝 |
| `tests/test_experiment_runner.py` | 通用 runner 校验数据 manifest：正确 SHA 通过并写入 manifest.json、错误 SHA 拒绝、缺文件拒绝 |

测试只用 mock JSON 记录 + `tmp_path`，不读完整 PoPu、不跑真实 Mini、不触发远程 GPU。

结果：`python -m pytest -q` → **346 passed**（含本轮数据 Manifest 哈希校验的新增回归测试），`git diff --check` 通过。

## 7. 真实运行命令（已于后续 EXP 执行）

Mini Run 命令已由 Experiment Runner 在 AutoDL 上执行（`EXP-P5.2-B-MINI-SCREEN-20260819-R01`，`SUCCEEDED`），结果见 [P5.2-B 结果报告](P5_2_B_POPU_NEURAL_MINI_RESULTS_v0.1.md)：

```text
python scripts/run_experiment.py --config configs/experiments/popu_neural_mini_v0.1.json
```

## 8. 已知限制

- 本文档最初记录为 `MINI_READY_TO_RUN`（未运行真实 Mini）；真实 Mini 已运行并经 Reviewer 接受后，状态更新为 `COMPLETE — MINI_ACCEPTED`，见 [P5.2-B 结果报告](P5_2_B_POPU_NEURAL_MINI_RESULTS_v0.1.md)。
- 三模型 Mini 的实际 `proceed/exclude/needs_fix` 结果见结果报告：三者均为 `proceed`。
- 固定开发子集 `[1..6]` 是**筛选**集，不代表全量受试者分布；Full 公平比较仍按 P5.1 的受试者隔离原则在全部受试者上评估。
- 学习信号阈值 0.25 是最低方向性门槛，不构成任何模型优劣结论。

## 9. 不能得出的结论

- 尚未得出 CNN/MLP/SVM 谁是 PoPu 总体最优候选；
- Mini 已运行，但**不形成最终候选排名**；尚未运行 Full 公平比较；
- 公开 PoPu 结果仍是候选证据，不是自研硬件、舒适性或产品验证。
