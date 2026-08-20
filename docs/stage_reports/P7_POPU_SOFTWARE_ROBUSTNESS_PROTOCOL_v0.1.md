# P7 — PoPu 软件鲁棒性协议 v0.1

## 状态

**PROTOCOL_AND_PERTURBATIONS_READY — MODEL EVALUATION PENDING。**

P6/P6.1 已停止继续调阈值，Small ResNet 保留为 PoPu 总体研究候选；P6.1 的三模型一致性规则仅作为研究候选。P7 现在进入软件扰动评价，但尚未产生任何鲁棒性结果。

## 扰动条件

- 降密度：行列步长 2 和 4，采样后以最近邻恢复到 64×27。
- 噪声：以每条输入正值 p95 为尺度，加入 1%、5%、10% 高斯噪声并截断负值。
- 坏点：固定传感单元随机失效 1%、5%、10%，同一条件下跨 snapshot 保持同一 mask。
- 坏行/坏列：分别随机失效 1、2、4 行或列，同一条件下跨 snapshot 固定。
- 随机条件使用 5 个冻结 seed；所有模型和规则共享相同扰动实例。

## 评价口径

以 clean OOF 推理为基线，报告 record macro-F1、balanced accuracy、accuracy 及其下降；同时报告 P6 拒识覆盖率、接受准确率、WAR、逐类别/逐受试者退化，以及跨 seed 均值、标准差和最坏情况。

## 边界

本协议只刻画 PoPu 64×27 Tactilus 数据上的软件敏感性。降密度重建、数值噪声和置零故障都不等于真实传感器的点距、量程、漂移、串扰、饱和或硬件失效验证，不能据此声称真实低密度硬件通过。

当前已实现 `src/topper_perception/neural/p7_robustness.py`、冻结配置和单元测试。下一步需要从 Full 证据包读取 Small ResNet fold checkpoint，在对应 outer-test 原始 record 上重新推理 clean 与扰动条件；不得用现有 OOF 概率伪造扰动结果。
