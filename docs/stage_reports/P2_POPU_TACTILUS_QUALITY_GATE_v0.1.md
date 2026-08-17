# P2/R2 — PoPu Tactilus 记录级质量门报告 v0.1

## 1. 阶段结论

**状态：PARTIAL（自动质量扫描完成；94 条 WARN 的人工复核策略尚未冻结）。**

本阶段已逐条重读 P1 中的固定姿态记录，生成可追溯的数值/时序质量结果和样本画廊。没有读回失败或结构性损坏；但 `WARN` 只是相对同姿态分布的统计异常候选，不能直接等同于坏传感器、坏标签或应删除的数据。

## 2. 实际执行与范围

实际执行命令：

```bash
uv run pytest -q
uv run python scripts/quality_popu.py
```

自动测试结果：`11 passed`。

扫描范围：P1 中 `status=OK` 且标签属于 `empty/supine/prone/left/right` 的 5,100 条固定姿态记录。P1 的 60 条 `others.json` 因缺少固定姿态与 variation 标签，保留在结果中但标记为 `EXCLUDED`，没有被读取后伪装成训练数据。

## 3. 输入与产物

输入：

- [P1 Inventory](../../data/processed/popu/popu_tactilus_inventory_v0.1.csv)
- [P2 规则配置](../../configs/experiments/popu_quality_v0.1.json)

产物：

- [逐记录质量结果 CSV](../../outputs/metrics/popu_tactilus_quality_results_v0.1.csv)
- [质量汇总 JSON](../../outputs/reports/popu_tactilus_quality_summary_v0.1.json)
- [五姿态典型/低信号/高信号画廊](../../outputs/figures/popu_posture_gallery_v0.1.png)
- [WARN 候选画廊](../../outputs/figures/popu_abnormal_samples_v0.1.png)

## 4. 实际结果

| 结果 | 记录数 | 含义 |
|---|---:|---|
| `ACCEPT` | 5,006 | 在当前同姿态统计规则下没有异常候选 |
| `WARN` | 94 | 需人工查看的统计异常候选，不是确认的坏样本 |
| `REJECT` | 0 | 没有读回失败或结构性无效记录 |
| `EXCLUDED` | 60 | `others.json` 无固定姿态标签，未参与本阶段判定 |

WARN 分布：空床 7、仰卧 18、俯卧 24、左卧 27、右卧 18。

触发原因：

| 规则 | 候选数 |
|---|---:|
| 同姿态内的中位总信号 robust z > 4.5 | 54 |
| 同姿态内的帧间总信号变异系数 robust z > 4.5 | 40 |
| 同姿态内的活跃单元数 robust z > 4.5 | 1 |

其中 1 条同时触发总信号和活跃单元数规则，因此规则触发次数为 95、高于 WARN 记录数 94。

## 5. 规则与人工抽查

规则不是物理压力阈值：它对每种姿态分别计算 `median_total_signal`、`median_active_cells` 和 `temporal_total_cv` 的中位数与 MAD（median absolute deviation），当某条记录的任一指标 robust z 大于 4.5 时标为 `WARN`。

已检查五姿态画廊与最高优先级 WARN 候选画廊。画廊中可以看到：

- 空床 WARN 主要体现为极低背景信号中的短时波动或少量亮点；
- 卧姿 WARN 常体现为人体受力范围、强度或短时动作幅度不同；
- 这些现象与数据采集状态或个体差异相容，单凭此结果不能认定为文件损坏或标签错误。

## 6. 已验证、合理推断、尚未验证

### 已验证

- 5,100 条固定姿态记录均可被 P2 重新读取，未出现 `REJECT`。
- 每条结果均有来源文件、受试者、标签、variation、代表帧和三个质量指标，可追溯回 P1 CSV。
- `others.json` 没有混进固定姿态样本池。

### 合理推断

- PoPu 固定姿态数据可以继续用于接触区域、几何和特征研究；质量结果应作为后续过滤/敏感性分析的输入。
- 当前 94 条 WARN 更适合作为“模型前应关注的异质性样本”，而非立即删除。

### 尚未验证

- WARN 是否对应真实传感器噪声、动作、体型差异、姿态标签误差或数据采集条件；
- WARN 在保留、剔除、降权三种策略下对受试者隔离模型的影响；
- 自研硬件、真实矩阵密度、标签真值、舒适性和产品效果。

## 7. P2.1 决策与后续放行

P2 自动扫描完成，但 P2.1 仍需作出一项显式研究决策：**当前 94 条 WARN 暂时保留为可用候选，只在后续模型阶段同时报告“全量”与“仅 ACCEPT”两套受试者隔离结果；不在此阶段删除。**

这样既不把统计异常误删，也能后续验证它们是否影响泛化。确认这项策略后，P3 才能以明确的输入池推进。

## 8. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-17 | 首次运行 P2 自动质量门，生成结果、画廊和本报告。 |
