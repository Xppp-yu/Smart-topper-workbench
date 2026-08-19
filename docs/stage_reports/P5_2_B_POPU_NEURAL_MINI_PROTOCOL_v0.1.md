# P5.2-B/R5.2-B — PoPu 神经网络 Mini 筛选协议与代码就绪报告 v0.1

## 1. 结论

**状态：MINI_READY_TO_RUN（不是 COMPLETE）。**

本任务完成了 P5.2-B Mini 筛选所需的代码、版本化配置、测试与说明，使项目达到 `MINI_READY_TO_RUN`：实现冻结、测试通过、配置就绪、治理接线完成。**本任务没有运行任何真实 Mini 或 Full 实验**——AutoDL 处于关闭状态，本任务禁止连接或启动服务器，也未读取完整 PoPu 数据训练。P5.2-A CPU/CUDA Smoke 保持 COMPLETE；P5.1 `calibrated_linear_svm` 传统候选与 P5.2-A CPU Smoke、CUDA R01/R02 历史产物均未修改或覆盖。

下一步是 **Reviewer 只读复核 + 配置冻结 + Controller 授权**之后，才由 Experiment Runner 在 AutoDL 上执行 Mini Run。

## 2. 冻结的 Mini 协议

| 项 | 冻结值 | 说明 |
|---|---|---|
| scope | `mini` | 由治理 runner 强制要求干净 Git worktree 后才可 QUEUED |
| 开发受试者子集 | `["1","2","3","4","5","6"]` | 固定、数字序、**在看任何模型结果之前**冻结；不针对模型输出挑子集 |
| 切分 | 受试者隔离 `val_ratio=0.2, test_ratio=0.0` | 6 受试者 → 5 训练 / 1 验证；同一受试者的记录不跨 split；无 test split |
| seed | `42` | 固定；同 config+seed 复现 |
| epochs | `5`，`early_stopping.min_epochs=3` | 实际 3–5 epochs |
| early stopping | `monitor=val_loss, mode=min, patience=2, min_delta=0.0, min_epochs=3` | **只读验证指标**，永不读 test |
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
- **needs_fix（协议/基础设施问题，修复后重跑）**：`no_leakage`、`same_split`、`checkpoint_ok`、`resume_ok`、`reload_ok`、`resource_ok` 任一失败。
- **proceed**：全部通过。

汇总：`needs_fix` 优先；全 `exclude` 才 `exclude`；否则 `proceed`。学习信号阈值仅由 chance（1/5）加 margin 推导，对三模型一致，与 P5.2-A 的 smoke 准确率无关。

## 5. 配置与治理

- 版本化配置：[`configs/experiments/popu_neural_mini_v0.1.json`](../../configs/experiments/popu_neural_mini_v0.1.json)
- EXP-ID：`EXP-P5.2-B-MINI-SCREEN-20260819-R01`；TASK-ID：`TASK-P5.2-B-MINI-SCREEN-v0.1`（符合治理格式）。
- 输出仅写 `outputs/experiments/<EXP-ID>/`；不修改 `configs/paths.local.json`、不复制原始数据入仓。
- 治理 runner 保留 Git SHA/dirty、Python/PyTorch/CUDA/GPU/seed/cmdline/time；**已存在 EXP-ID 时拒绝覆盖**（`ExpIdError`）。
- 新增 `runner_type`：`popu_neural_mini`（登记于 `contracts.RUNNER_TYPES`、`runner.RUNNER_REGISTRY`、schema enum），原 `popu_neural` smoke 路径不变。

## 6. 测试

| 文件 | 覆盖 |
|---|---|
| `tests/test_neural_metrics.py` | accuracy / macro-F1 / balanced accuracy / 逐类别 P/R/F1/support / 混淆矩阵 / 零支撑类 / 非法输入 |
| `tests/test_viability.py` | gate `proceed` / `exclude` / `needs_fix`（含泄漏、checkpoint、resume、reload、同 split、资源超限）、非法输入、`overall_verdict` |
| `tests/test_neural_mini.py` | 受试者隔离、三模型共享 split、epoch history、早停边界、best checkpoint 规则、resume/reload、CPU 下 CUDA 峰值内存字段、gate 集成、经治理 runner 的 mini 运行、EXP-ID 覆盖拒绝、缺数据报错 |

测试只用 mock JSON 记录 + `tmp_path`，不读完整 PoPu、不跑真实 Mini、不触发远程 GPU。

结果：`python -m pytest -q` → **310 passed**（271 既有 + 39 新增），`git diff --check` 通过。

## 7. 未运行的真实命令

下一阶段的 Mini Run 命令（**本任务未执行**）应由 Experiment Runner 在 AutoDL 打开、Reviewer 复核并授权后运行：

```text
python scripts/run_experiment.py --config configs/experiments/popu_neural_mini_v0.1.json
```

## 8. 已知限制

- 本阶段仅冻结协议与实现，未产出任何真实 Mini 指标；`MINI_READY_TO_RUN` 不等于 Mini 通过或候选排名。
- 三模型 Mini 的实际 `proceed/exclude/needs_fix` 结果要等真实运行后才有。
- 固定开发子集 `[1..6]` 是**筛选**集，不代表全量受试者分布；Full 公平比较仍按 P5.1 的受试者隔离原则在全部受试者上评估。
- 学习信号阈值 0.25 是最低方向性门槛，不构成任何模型优劣结论。

## 9. 不能得出的结论

- 尚未得出 CNN/MLP/SVM 谁是 PoPu 总体最优候选；
- 尚未运行 Mini/Full，无真实 Mini 指标，无最终候选排名；
- 公开 PoPu 结果仍是候选证据，不是自研硬件、舒适性或产品验证。
