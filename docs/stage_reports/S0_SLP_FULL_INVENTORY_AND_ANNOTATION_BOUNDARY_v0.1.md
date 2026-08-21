# S0 — SLP 全量 Inventory 与标注边界审计 v0.1

## 1. 结论

**状态：COMPLETE_WITH_QUARANTINE — S1_ALIGNMENT_AUDIT_READY。**

SLP 本地数据已完成首轮全量结构扫描。109 名受试者、1,941 个 `subject × cover × modality` 组合中，1,939 组完整；2 组缺少全部 45 个 `depthRaw/cover2` 帧，应在后续跨模态配对中 quarantine。109 名受试者的 RGB/IR 14 节点标注、对齐矩阵和 danaLab 压力校准文件结构均通过。

SLP 没有身体区域像素真值。RGB/IR 节点属于 README 声明的人工原始标注；映射到 Depth/PM 的节点属于可能带偏差的派生标注。腰、臀等区域必须另建“节点几何 → 图像辅助 → 人工复核”证据链。

## 2. 实际执行

```powershell
uv run pytest -q tests/test_slp_inventory.py
uv run python scripts\inventory_slp.py --config configs\paths.local.json
```

测试：定向 `4 passed in 0.95s`；全仓回归 `524 passed, 14 warnings in 59.06s`。14 条均为既有 joblib/NumPy shape DeprecationWarning，不是 SLP 新增失败。

Inventory 命令按 fail-closed 设计，在发现结构错误时以非零状态结束，但仍完整写出 CSV/JSON，便于定位和 quarantine。扫描只枚举逐帧文件，并读取小型 MAT/NPY 标注、单应矩阵和校准文件；没有将整套 RGB/IR/Depth/PM 数据读入内存。

## 3. 输入与数据粒度

- 根目录：`E:\TeamProjects\datasets\smart-topper\SLP2022\SLP`（只读使用）。
- 粒度：`setting × subject × cover_condition × modality`。
- 场景：danaLab 102 人；simLab 7 人。
- 遮盖条件：`uncover / cover1 / cover2`。
- 期望覆盖：每组 45 帧。
- danaLab：PM、IR、IRraw、RGB、depth、depthRaw。
- simLab：IR、IRraw、RGB、depth、depthRaw；逐帧 PM 不属于其期望结构。

## 4. 产物

| 产物 | SHA-256 |
|---|---|
| `data/processed/slp/slp_modality_inventory_v0.1.csv` | `472F181F7369AA585D4E404C28FDA4684936A5862C702937495DE25F3F8B6706` |
| `data/processed/slp/slp_annotation_inventory_v0.1.csv` | `F0EC26DE0D399AAB1667ED0E640648F36D1E9F0748C60E30FA1FB29641E00E65` |
| `outputs/reports/slp_inventory_summary_v0.1.json` | `5F7DD1990083F59D63D3B44D257D5C4D832AF261D522E2E9D7B6281B0405E570` |

## 5. 数据质量结果

| 检查 | 结果 | 风险与处理 |
|---|---:|---|
| 受试者 | 109（102 + 7） | 与目录和 README 一致 |
| Inventory groups | 1,941 | 预期组合全部被扫描 |
| 完整 groups | 1,939（99.897%） | 可进入 S1 |
| 错误 groups | 2（0.103%） | 后续精确 quarantine |
| 扫描逐帧文件 | 87,255 | 不代表所有辅助文件总数 |
| RGB/IR 人工节点结构 | 109/109 通过 | shape `3×14×45` |
| 对齐矩阵结构 | 109/109 通过 | RGB/IR/Depth 均为 `3×3` |
| danaLab PM 校准结构 | 102/102 通过 | shape `3×45` |
| 区域真值 | 0/109 | 必须另建派生标签与人工复核流程 |

逐模态文件数：IR 14,715；IRraw 14,715；RGB 14,715；depth 14,715；depthRaw 14,625；PM 13,770。

### 5.1 精确异常

| setting | subject | cover | modality | 缺失 |
|---|---|---|---|---|
| simLab | 00003 | cover2 | depthRaw | 1–45 全部缺失 |
| simLab | 00004 | cover2 | depthRaw | 1–45 全部缺失 |

对应的非 raw `depth/cover2` 图片仍存在。后续不得用图片反推并伪造 depthRaw；这两组在 raw-depth 任务中 quarantine，在只使用发布图片的探索任务中可单独保留并注明来源。

## 6. 标注可信度分级

- **原始人工真值**：RGB/IR 的 14 节点。
- **派生节点标签**：通过 homography 投影到 PM/Depth 的节点；必须保存矩阵、方向和映射质量。
- **区域代理标签**：由关节和体型数据生成的粗区域。
- **自动伪标签**：由 OpenCV/图像信号细化的区域。
- **人工复核派生标签**：人工接受或修改后的区域；仍不宣称独立医学级解剖真值。

具体路线与人工复核合同见 `docs/SLP_RESEARCH_AND_REGION_ANNOTATION_ROUTE_v0.1.md`。

## 7. 已验证 / 合理推断 / 尚未验证

### 已验证

- 本地 SLP 根目录、109 个受试者目录和预期模态可读取。
- 87,255 个逐帧文件的目录覆盖和帧号完整性已扫描。
- 109 人的小型节点、单应矩阵和适用的压力校准 shape 已检查。
- 两个缺失 depthRaw 组合已精确定位。

### 合理推断

- 除两组 quarantine 外，数据结构足以进入 S1 跨模态配对与 overlay 审计。
- 关节可以为粗区域预标注提供几何种子，但不能单独形成腰/臀像素真值。

### 尚未验证

- 单应矩阵的实际方向、往返误差和映射后越界率。
- 每个图像/数组内容是否可解码、数值是否有限、PM 物理量恢复是否正确。
- RGB/IR 人工节点的标注一致性和遮挡字段分布。
- OpenCV 区域预标注的准确性、人工复核成本和审阅者一致性。
- SLP 的明确许可条款：本地数据根目录与 README 中未找到 LICENSE/LICENCE/COPYING 文件；许可未核实前仅限内部非商业研究。

## 8. 决策与下一步

1. 放行 S1：建立 `setting/subject/cover/frame` 主键并做跨模态一一配对。
2. 对 simLab 00003、00004 的 `cover2/depthRaw` 精确 quarantine。
3. 生成固定样本的 RGB/IR/Depth/PM 关节 overlay，验证 homography 方向和误差。
4. 冻结受试者拆分前，不运行任何 SLP 模型。
5. 区域线先做 12–20 名受试者 Pilot；不直接全量自动标注。

## 9. 不能得出的结论

- 不能声称 SLP 已有腰、臀等区域真值。
- 不能把 homography 投影节点称为无偏人工真值。
- 不能把 OpenCV 轮廓称为人体解剖分割。
- 不能因为目录完整就声称所有内容、模型或产品能力已验证。
