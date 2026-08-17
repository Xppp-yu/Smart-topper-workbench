# P3.1/R3.1 — PoPu Mask 策略比较与研究用规则冻结报告 v0.1

## 1. 阶段目标与完成判定

**状态：COMPLETE（已冻结 P4a 使用的研究用 Mask；未验证解剖准确度）。**

本阶段在同一批 PoPu Tactilus 记录上比较三种可复用 Mask 策略，检查非空率、相邻帧稳定性、bbox、CoP 和人工叠加图，并冻结一项明确、版本化、可回退的 Geometry 输入规则。完成不代表识别出身体部位，也不代表得到真实物理接触边界。

## 2. 实际执行

```bash
uv run pytest -q
uv run python scripts/compare_popu_mask_strategies.py
uv run python scripts/geometry_popu.py \
  --strategy largest_component \
  --geometry-output outputs/metrics/popu_geometry_results_v0.2.csv \
  --summary-output outputs/reports/popu_geometry_summary_v0.2.json \
  --mask-figure-output outputs/figures/popu_mask_overlay_v0.2.png \
  --geometry-figure-output outputs/figures/popu_geometry_overlay_v0.2.png
```

最终自动测试：`31 passed`。

输入范围为 P2 的全部 `5,160` 条记录；三种策略均保留在比较表中。固定姿态 `ACCEPT/WARN` 共 `5,100` 条进入 Geometry，`60` 条 `others.json` 继续 `EXCLUDED`。

## 3. 输入与产物

输入与配置：

- [P2 质量结果](../../outputs/metrics/popu_tactilus_quality_results_v0.1.csv)
- [P3.1 比较配置](../../configs/experiments/popu_mask_strategy_comparison_v0.1.json)
- [冻结的 P4a Geometry 规则](../../configs/experiments/popu_geometry_frozen_v0.2.json)

比较产物：

- [逐记录逐策略比较表](../../outputs/metrics/popu_mask_strategy_comparison_v0.1.csv)
- [比较汇总](../../outputs/reports/popu_mask_strategy_comparison_summary_v0.1.json)
- [稳定性图](../../outputs/figures/popu_mask_strategy_stability_v0.1.png)
- [3 ACCEPT + 3 WARN + 3 DIVERGENCE 叠加图](../../outputs/figures/popu_mask_strategy_overlays_v0.1.png)

冻结版 Geometry 产物：

- [逐记录 Geometry v0.2](../../outputs/metrics/popu_geometry_results_v0.2.csv)
- [Geometry 汇总 v0.2](../../outputs/reports/popu_geometry_summary_v0.2.json)
- [冻结 Mask 代表图](../../outputs/figures/popu_mask_overlay_v0.2.png)
- [冻结 Geometry 代表图](../../outputs/figures/popu_geometry_overlay_v0.2.png)

## 4. 全量比较结果

三种策略共生成 `15,480 = 5,160 × 3` 行；结果为 `15,238 OK / 62 WARN / 180 EXCLUDED / 0 REJECT`。

| 策略 | OK | WARN | 中位 Mask IoU | 中位 bbox IoU | 中位 Mask 占比 | 中位 bbox 占比 | 中位组件数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `relative_filtered` | 5,095 | 5 | 0.968 | 0.986 | 0.217 | 0.635 | 3 |
| `largest_component` | 5,100 | 0 | 0.971 | 0.978 | 0.188 | 0.333 | 1 |
| `relative_closed` | 5,043 | 57 | 0.971 | 0.988 | 0.226 | 0.572 | 2 |

这些指标只说明当前公开数据上的可执行性与时间稳定性。较高 IoU 不等于更接近人体真实轮廓。

## 5. 冻结决策

P4a 研究特征工程冻结使用 `largest_component`：

```text
strategy = largest_component
positive_percentile = 50.0
minimum_raw_threshold = 1.0
minimum_component_cells = 3
minimum_component_fraction_of_largest = 0.02
```

最后两个连通域过滤参数为统一配置模式保留；`largest_component` 分支只选择阈值后面积最大的连通域，因此它们在本策略中不参与计算。

选择理由：

- `5,100/5,100` 条固定姿态输入均产生非空 Mask；
- 单一主连通域避免远端弱信号直接拉大 bbox；
- 中位 bbox 占比由 `relative_filtered` 的 `0.635` 降到 `0.333`；
- Mask IoU 与 CoP 位移不劣于其他候选到足以阻断后续研究；
- 代表图显示其更适合“主人体接触区/主轴/粗 Geometry”输入。

冻结限制：它可能丢弃真实但与主躯干分离的手臂或腿部接触。因此冻结仅服务 P4a 的研究用全局 Geometry，不得解释为身体部位分割；得到独立真值或自研硬件数据后允许重新打开此决策。

## 6. 冻结版 Geometry 验收

`largest_component` v0.2 全量输出：

| 状态 | 数量 | 说明 |
|---|---:|---|
| `OK` | 5,098 | Mask、bbox、CoP、质心和 PCA 主轴均可计算 |
| `WARN` | 2 | 空床只剩单个有效格点，PCA 主轴不可定义 |
| `EXCLUDED` | 60 | 无固定姿态标签的 `others.json` |
| `REJECT` | 0 | 无读取或计算拒绝 |

两条单格点记录已显式写为 `insufficient_cells_for_principal_axis`，主轴字段留空，不再输出 `NaN`。

## 7. 结论分层

### 已验证

- 三种策略可在 PoPu 全量记录和全部 snapshot 上运行并可追溯到源文件；
- 冻结规则已进入可复用 Geometry API、版本化配置和 v0.2 真实输出；
- `largest_component` 在当前输入上没有空 Mask，PCA 不可定义的单点记录被显式 WARN；
- 代表样本分类互斥，叠加图实际包含 `3 + 3 + 3` 组。

### 合理推断

- 对全局主轴、粗 bbox 和主人体接触区特征，`largest_component` 比保留多个远端小岛更稳妥；
- 该规则足以作为 P4a 无标签特征工程的统一研究输入。

### 尚未验证

- Mask 与人体解剖轮廓、肩/腰/骨盆或真实接触面积的误差；
- 被删除的分离连通域中有多少是真实肢体接触；
- 在 TIP、SLP、PressurePose、自研床垫或整夜动态数据上的泛化；
- 该规则能否作为生产算法。

## 8. 后续决策

P4a 可以开始，但必须读取 [冻结规则](../../configs/experiments/popu_geometry_frozen_v0.2.json)，逐行保留 `mask_strategy`、源文件、受试者、姿态、P2/P3 状态和代表帧索引。P4a 不得生成肩/腰/骨盆监督标签。

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-17 | 完成三策略全量比较；修复代表样本类别重叠并重跑。 |
| 2026-08-17 | 冻结 `largest_component`；接入 Geometry API 并生成 v0.2 全量产物。 |
| 2026-08-17 | 将单格点 PCA 从隐式 `NaN/OK` 修正为显式 `WARN`。 |
