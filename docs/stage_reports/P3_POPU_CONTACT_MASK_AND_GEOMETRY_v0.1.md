# P3/R3 — PoPu 接触 Mask 与 Geometry 报告 v0.1

> 本文件保留 P3 首轮未冻结状态。后续收口结论见 [P3.1 Mask 冻结报告](P3_1_POPU_MASK_STRATEGY_FREEZE_v0.1.md) 与 [P3.2 区域标注对齐审计报告](P3_2_POPU_SEGMENTATION_ALIGNMENT_AUDIT_v0.1.md)。

## 1. 阶段结论

**状态：PARTIAL（全量计算、测试和可视化完成；当前 Mask 候选未冻结）。**

P3 已把 P2 的逐记录输入转换为可追溯的接触 Mask、外接框、非加权质心、压力中心（CoP）和主轴等几何量。它证明这些几何计算可以在真实数据上稳定执行；但视觉复核显示，简单的相对信号阈值仍可能保留分离弱信号并影响外接框。因此当前结果不能直接作为 P4 特征表的冻结输入，更不能解释为解剖身体部位或真实接触面积。

## 2. 已确认的 P2.1 输入策略

94 条 P2 `WARN` 暂不删除。P3 同时保留 P2 `ACCEPT` 与 `WARN`，把 P2 `EXCLUDED` 的 `others.json` 留在本阶段之外；后续模型阶段必须同时比较“全量有标签”与“仅 ACCEPT”两种口径。

## 3. 实际执行

```bash
uv run pytest -q
uv run python scripts/geometry_popu.py
```

自动测试：`13 passed`。

最终运行规则：每条记录使用 P2 的代表帧；正读数的 50 分位作为相对阈值，原始阈值最低为 1，删除少于 3 个单元或小于最大连通域 2% 的连通域。

## 4. 输入与产物

输入：

- [P2 质量结果](../../outputs/metrics/popu_tactilus_quality_results_v0.1.csv)
- [P3 规则配置](../../configs/experiments/popu_geometry_v0.1.json)

产物：

- [逐记录 Geometry CSV](../../outputs/metrics/popu_geometry_results_v0.1.csv)
- [机器可读汇总 JSON](../../outputs/reports/popu_geometry_summary_v0.1.json)
- [Mask 叠加图](../../outputs/figures/popu_mask_overlay_v0.1.png)
- [Geometry 叠加图](../../outputs/figures/popu_geometry_overlay_v0.1.png)

## 5. 实际结果

| 项目 | 结果 |
|---|---:|
| P2 可用固定姿态记录输入 | 5,100 |
| Geometry `OK` | 5,093 |
| Geometry `WARN` | 7 |
| Geometry `REJECT` | 0 |
| 继续 `EXCLUDED` 的 `others.json` | 60 |

7 条 `WARN` 均来自低信号空床记录，当前相对阈值后没有足够接触单元形成可用几何对象。这是符合数据状态的提示，不是读取错误。

非空床四类姿态的中位 Mask 占比约为 21%–22%；这仅描述当前候选规则下保留的相对强信号范围，不是人体真实接触面积。

## 6. 视觉复核与问题发现

已复核代表性 `ACCEPT` 记录的 Mask 与 Geometry 叠加图。50 分位阈值较首次 25 分位运行减少了大量边缘弱信号，但部分样本仍有分离的高信号小区域；若所有保留连通域共同参与 bbox，外接框可能被这些区域拉大。

这说明：

- `CoP`、整体强信号位置和主轴可作为**候选几何量**继续审阅；
- 当前 bbox、Mask 面积和组件数不应直接进入冻结特征表；
- 不能把此 Mask 解释为 Head/Torso/Arm/Leg 的分割结果。

## 7. 已验证、尚未验证与下一步

### 已验证

- 模块化 Mask/Geometry 代码可在真实 PoPu 数据上逐记录运行，结果可追溯到 P2 输入。
- 7 条低信号空床记录被显式保留为 `WARN`，没有被静默删除。
- 坐标、Mask、bbox、centroid、CoP、主轴和结果图均已经输出。

### 尚未验证

- 当前 Mask 是否最能代表人体接触区域；
- bbox 是否稳定到可支持位置或区域特征；
- 几何量对姿态的区分力、跨受试者稳定性和对 P2 WARN 的敏感性；
- 身体部位真值、硬件坐标映射、自研传感器适配和产品含义。

### P3.1 最小任务

在进入 P4 前，只比较三种 Mask 策略对同一批样本的影响：

1. 当前“相对阈值 + 小连通域过滤”；
2. 仅保留最大连通域；
3. 保留与主接触区邻近的主要连通域。

比较指标是 bbox 稳定性、Mask 面积、CoP 位移和代表图人工复核，不是姿态分类分数。选择后冻结一种 Geometry 输入规则，才可推进 P4。

## 8. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-17 | 初次运行发现 25 分位 Mask 吸收较多边缘弱信号。 |
| 2026-08-17 | 调整为 50 分位并加入相对连通域过滤后重跑；保留本报告记录的未冻结状态。 |
