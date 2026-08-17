# P1/R1 — PoPu Tactilus 全量结构盘点报告 v0.1

## 1. 阶段结论

**状态：COMPLETE（结构盘点通过，带一个已识别的数据用途分流）。**

PoPu Tactilus 的全部 JSON 均可解析，传感矩阵形状一致，未发现结构性错误。固定姿态训练候选集与未标注连续序列已被明确分开；可以进入 P2 质量门，不应直接开始姿态模型训练。

## 2. 本阶段目标与实际命令

目标：遍历所有 PoPu Tactilus JSON，确认受试者、标签、variation、snapshot、矩阵形状、结构异常和稳定唯一标识。

实际执行命令：

```bash
uv run python scripts/inventory_popu.py --config configs/paths.local.json
```

运行配置：未使用 `--include-sha256`，因此本次不是逐文件内容哈希冻结。

## 3. 输入与产物

输入数据根目录：

```text
E:\TeamProjects\datasets\smart-topper\popu\PoPu_data\tactilus_data
```

本次产物：

- [逐记录 Inventory CSV](../../data/processed/popu/popu_tactilus_inventory_v0.1.csv)
- [机器可读汇总 JSON](../../outputs/reports/popu_tactilus_inventory_summary_v0.1.json)
- [姿态标签分布图](../../outputs/figures/popu_tactilus_label_distribution_v0.1.png)

## 4. 实际结果

| 项目 | 结果 |
|---|---:|
| JSON 记录数 | 5,160 |
| 受试者数 | 60 |
| 统一传感矩阵形状 | `64×27` |
| 结构错误 | 0 |
| 重复 `sample_id` | 0 |
| `OK` 记录 | 5,100 |
| `WARN` 记录 | 60 |
| 固定姿态有标签 snapshot | 51,000 |
| `others.json` 未标注 snapshot | 35,247 |

有标签记录分布：

| 标签 | JSON 记录 | snapshot |
|---|---:|---:|
| `supine` | 1,260 | 12,600 |
| `prone` | 1,260 | 12,600 |
| `left` | 1,260 | 12,600 |
| `right` | 1,260 | 12,600 |
| `empty` | 60 | 600 |

## 5. 60 条 WARN 的解释

所有 WARN 均来自每位受试者一条 `others.json`。这些记录矩阵结构正常、所有 snapshot 读数有效，但源 JSON 不包含 `position` 与 `variation` 字段；单条记录含 `341–914` 个 snapshot。

它们不是坏数据，也不能被假设为某个固定姿态标签。当前处理决定：保留原始数据与 Inventory 记录，但从固定姿态监督训练候选池中排除；后续可研究转身、过渡、异常或 `UNKNOWN`。

## 6. 结论边界

### 已验证

- Windows 本地 PoPu Tactilus 副本可被完整遍历和解析。
- 全部 5,160 条记录声明 `64×27` 矩阵，且本次检查没有发现读数长度不符或非有限值。
- 固定姿态有标签数据与无姿态标签的 `others.json` 可由来源字段明确区分。

### 合理推断

- 固定姿态数据足以进入样本质量和可视化研究；后续模型实验仍须按受试者隔离。
- `others.json` 可能对过渡/非固定状态研究有价值，但其具体语义尚未由独立标注验证。

### 尚未验证

- 图像质量、异常噪声、传感器漂移、坐标方向和标签正确性；
- 姿态识别性能、跨受试者泛化、置信度阈值、`UNKNOWN/REJECT`；
- 自研传感器、真实硬件密度、气囊闭环、舒适性、整夜或产品效果。

## 7. 放行至 P2 的条件与下一步

P1 放行：**通过**。

P2 的最小任务是基于有标签的 5,100 条记录，建立典型/边界/异常样本画廊，并对每条记录给出 `ACCEPT`、`WARN` 或 `REJECT`。P2 还需明确：

1. `empty` 是独立空床质量门还是姿态分类类别；
2. 同一受试者的所有记录必须进入同一数据划分；
3. `others.json` 继续隔离，直到任务定义和真值明确。

## 8. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-17 | 首次基于实际 P1 运行产物建立本报告。 |
