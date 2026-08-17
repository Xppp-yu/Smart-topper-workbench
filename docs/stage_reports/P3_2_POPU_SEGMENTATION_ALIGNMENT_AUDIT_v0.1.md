# P3.2/R3.2 — PoPu 区域标注与压力记录对齐审计报告 v0.1

## 1. 阶段目标与完成判定

**状态：COMPLETE；区域监督结论为 HOLD。**

本阶段审计 `segmentation_data` 中每份 COCO 身体区域标注的结构、类别、图像引用、bbox、画布和多边形边界，并检查它能否唯一对应一条 Tactilus 压力记录。完成判定是给出可追溯的配对基数和监督边界，而不是必须得到可训练真值。

## 2. 实际执行

```bash
uv run pytest -q
uv run python scripts/audit_popu_segmentation.py --config configs/paths.local.json
```

最终自动测试纳入全项目 `31 passed`；P3.2 全量命令退出码为 `0`。

## 3. 输入与产物

输入：PoPu `segmentation_data` 的 `1,730` 份 `*_annotations.coco.json`，以及同一数据集的 `tactilus_data` 文件名与目录结构。

产物：

- [逐标注审计表](../../outputs/metrics/popu_segmentation_alignment_audit_v0.1.csv)
- [机器可读汇总](../../outputs/reports/popu_segmentation_alignment_summary_v0.1.json)
- [配对基数结果图](../../outputs/figures/popu_segmentation_alignment_v0.1.png)

## 4. 核心结果

| 项目 | 结果 |
|---|---:|
| 发现 COCO 文件 | 1,730 |
| 成功审计 | 1,730 |
| 读取拒绝 | 0 |
| 类别名称错误 | 0 |
| annotation 图像引用错误 | 0 |
| bbox 结构错误 | 0 |
| annotation 类别错误 | 0 |
| 多边形越界点 | 0 |
| `AMBIGUOUS_TACTILUS_CANDIDATES` | 1,670 |
| `ONE_TO_ONE_CANDIDATE` | 60 |

`1,670` 份人体标注各自对应同一受试者、姿态和 variation 下的 `3` 条 Tactilus 压力记录；文件结构没有提供规则证明应该选择 `_0`、`_1` 还是 `_2`。

唯一的 `60` 份一对一候选全部属于 `empty` 空床。按人体姿态分布：`left 416 / prone 418 / right 417 / supine 419` 全部是三候选歧义。因此当前可用于人体区域逐记录监督的一对一候选为 **0**。

## 5. 监督边界决策

### 可以使用

- 检查 COCO 类别与 27×64 画布结构；
- 研究 Head/Torso/Arm/Leg 标签形态和坐标约定；
- 作为候选标签来源，等待独立配对证据；
- 支持后续人工核查或数据提供方确认 `_0/_1/_2` 语义。

### 不可以使用

- 任意选择 `_0`、`_1` 或 `_2` 与 COCO 标注配对；
- 将 1,670 份标注复制给三条压力记录；
- 训练并报告肩、腰、骨盆或身体区域准确率；
- 把 `ONE_TO_ONE_CANDIDATE` 空床记录解释为人体区域真值。

## 6. 结论分层

### 已验证

- 1,730 份 COCO 文件结构可读，类别与画布合同一致；
- 人体标注到 Tactilus 记录的文件名级关系是 `1:3`，空床为 `1:1`；
- 当前文件结构无法建立人体标注的唯一逐记录配对。

### 合理推断

- PoPu 区域标注仍有结构研究价值，但没有额外元数据时不能承担压力图身体部位监督；
- P4a 无标签 Geometry/统计特征路线可继续，不需要等待该监督问题解决。

### 尚未验证

- 三条 Tactilus capture 是否分别对应不同 snapshot、重复采集或其他协议；
- COCO 图片与哪条压力记录在采集时间上同步；
- COCO 区域能否派生肩、腰、骨盆合同；
- 数据提供方是否有未公开的映射表。

## 7. 后续决策

P3.2 审计任务完成，但 P4b 区域监督特征集继续 `HOLD`。解除条件至少满足其一：获得官方映射表、找到同步采集元数据、完成人工可复核配对协议，或接入具有独立同步真值的数据集。

## 8. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-17 | 完成 1,730 份标注全量结构与配对审计；冻结区域监督 HOLD 结论。 |
