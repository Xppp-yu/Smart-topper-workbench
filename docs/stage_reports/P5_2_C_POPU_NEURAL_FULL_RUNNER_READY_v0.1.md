# P5.2-C/R5.2-C — PoPu 神经网络 Full runner 就绪记录 v0.1

**状态：RUNNER_READY — PREFLIGHT_PENDING。**

本记录只证明 Full 执行代码已经实现并通过本地代码回归；没有运行一折 GPU 计时预检，也没有运行正式 Full，因而没有 Full 指标、模型排名或总体冠军结论。

## 已实现

- 冻结的 60 受试者、5,006 records、50,060 snapshots 数据边界，任一不符即失败。
- 3 repeats × 5 个外层受试者折；三个神经网络共享同一 split manifest。
- 每折 Stage A 内层受试者验证选 epoch，Stage B 重新初始化并使用全部 outer-train refit；outer-test 只推理一次。
- train-only normalization、train-only 左右翻转、固定标签顺序与派生 seed。
- snapshot/record OOF、逐折指标、混淆矩阵、逐类/逐受试者指标、NLL/Brier/ECE、训练/推理时间、checkpoint、显存峰值与 reload 一致性。
- P5.1 `calibrated_linear_svm` 六份证据先按路径、大小和 SHA-256 只读校验，再进入冻结选择规则。
- 每个 `repeat/fold/model` 以带内容哈希的 `complete.json` 作为事务完成标记；重新调用 runner 时只跳过完整且哈希一致的单元，不覆盖已完成单元。
- 独立的一折计时预检入口；预检不会创建或冒充正式 Full 结果。
- 历史 P5.1 SVM record CSV 为 6 位小数序列化，实测 15,018 行最大行和漂移为 `2.0e-6`；校准诊断读取时仅允许不超过 `5e-6` 的有限小数漂移并重新归一化，随后仍按冻结 `1e-6` 规则校验。该处理只影响不参与排名的 NLL/Brier/ECE 诊断，不修改 SVM 类别预测、历史主指标或候选排序。
- 正式 Full 若在最终聚合阶段失败，可使用治理恢复入口复核所有完成标记及内容哈希后仅重做汇总；已完成的 `repeat/fold/model` 单元不重训、不覆盖，并在 manifest 中记录原始与恢复 Git SHA。

## 本地验证

```text
446 passed, 14 warnings
```

警告来自既有 `joblib`/NumPy 反序列化弃用提示，与本次 Full runner 无新增失败。此结果只验证本地代码路径，不证明 CUDA Full 运行成功。

## AutoDL 执行 Gate

先运行：

```bash
python scripts/run_popu_neural_full_preflight.py
```

只有 `PREFLIGHT_SUCCEEDED`、`within_frozen_budget=True`、峰值 CUDA 显存不超过 8,000 MB，且 Reviewer 接受预检证据后，才运行：

```bash
python scripts/run_experiment.py --config configs/experiments/popu_neural_full_v0.1.json
```

两条命令都应在 `screen`/`tmux` 中执行。正式 Full 完成后仍需下载完整证据包并独立复核；runner 只输出 `recommended_winner_pending_reviewer`，不会自动冻结候选。
