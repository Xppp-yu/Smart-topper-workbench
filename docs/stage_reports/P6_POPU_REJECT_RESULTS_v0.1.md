# P6 — PoPu UNKNOWN/REJECT 与错误分析结果 v0.1

## 1. 当前状态

**总体 operating point 已找到；P6 暂不标记为最终验收通过。**

本次分析读取 P5.2-C Full 证据包中 Small ResNet 的 record-level OOF 概率，不重新训练。输入为 3 repeats × 5 folds、每 repeat 5,006 条 record 预测，共 15,018 条。阈值在 repeat 0/1 选择，repeat 2 只作独立评估。

代码、配置与测试：

- `src/topper_perception/neural/p6_reject.py`
- `scripts/analyze_popu_p6_reject.py`
- `configs/analysis/popu_p6_reject_v0.1.json`
- `tests/test_neural_p6_reject.py`

## 2. 冻结的输出规则

- 评价粒度：record；record 的五类概率已经由 P5.2 Full runner 聚合。
- 正常输出：概率向量有效且 `max_probability >= threshold` 时输出 argmax 睡姿。
- `UNKNOWN/REJECT`：输入无效、概率非有限/不归一，或 `max_probability < threshold`。
- 本阶段采用一个全局 `max_probability` 阈值；不使用按类别或按受试者阈值，以免在当前证据量上过拟合。
- 这只是 PoPu 固定五姿态研究评价规则，不是硬件、整夜、舒适性或控制策略规则。

## 3. 阈值选择与独立评估

开发约束：Wrong Action Rate ≤ 0.5%、接受样本准确率 ≥ 99.5%、覆盖率 ≥ 50%；选择满足约束的最低阈值以保留覆盖率。

| 数据 | 阈值 | 覆盖率 | 拒绝率 | 接受样本准确率 | Wrong Action Rate |
|---|---:|---:|---:|---:|---:|
| repeat 0/1 开发 | 0.94 | 94.20% | 5.80% | 99.53% | 0.439% |
| repeat 2 独立评估 | 0.94 | 96.90% | 3.10% | 99.61% | 0.380% |
| 全部 3 repeats 诊断 | 0.94 | 95.10% | 4.90% | 99.56% | 0.419% |

结论：`0.94` 可以作为当前总体 operating point，但不能仅凭总体指标验收，因为分层公平性和高置信错误仍不满足稳健闭环要求。

补充比较结果：在同样的开发/评估拆分和总体约束下，Top-2 margin 的候选规则为 `margin >= 0.90`，repeat 2 覆盖率 96.60%、接受准确率 99.63%、WAR 0.360%；normalized entropy 的候选规则为 `entropy <= 0.15`，覆盖率 96.84%、接受准确率 99.61%、WAR 0.380%。三种规则总体差异很小，不能据此解决受试者公平性和高置信错误问题。

## 4. 分层结果与错误

repeat 2 的类别结果显示：`empty` 样本只有 53 条，覆盖率 100%；`left` 覆盖率 95.38%、接受准确率 98.98%，是主要弱项；`prone` 接受准确率为 100%。

主要错误方向（全部 repeats、高置信错误）：

- `left → prone`：29 条；
- `prone → supine`：12 条；
- `right → prone`：9 条；
- `left → supine`：7 条。

全部 248 个错误中有 83 个 `max_probability >= 0.90`，说明简单的置信度拒识不能消除模型的过度自信错误。

受试者公平性方面，repeat 2 的受试者 15 接受覆盖率 87.06%、接受准确率 91.89%、Wrong Action Rate 7.06%；受试者 31 和 43 的接受准确率分别为 96.15% 和 96.25%。这构成当前 P6 的明确未闭环项。

## 5. 产物与下一步

正式公平性门槛为：最差受试者覆盖率 ≥ 80%、接受准确率 ≥ 95%、WAR ≤ 5%，以及各姿态覆盖率差异 ≤ 10 个百分点。本次 repeat 2 评估未通过：最差受试者接受准确率约 91.9%，最高 WAR 约 7.1%。因此机器可读结果中的 `p6_final_acceptance` 为 `false`。

实际分析产物位于 `outputs/analysis/EXP-P6-POPU-REJECT-20260820-R01/`，包括阈值表、逐类别/逐受试者结果、混淆矩阵、错误案例、高置信错误和带 uncertainty 列的 record 表。

已通过 WSL 只读路径复核重点案例：36/36 条记录均可找到；原始 `position` 与 OOF 标签一致；均为 64×27、10 snapshots，未发现这批案例的明显缺失或形状异常。因此当前证据更支持“模型真实高置信混淆”，而不是简单标签错位。原始案例审计产物为 `outputs/analysis/EXP-P6-POPU-REJECT-20260820-R01/raw_case_audit.csv`。P6 通过前不应把 `0.94` 写成部署阈值，也不应把 P6 结果当作产品准确率；P7 可在协议层准备，但正式结论应等待 P6 分层验收。
