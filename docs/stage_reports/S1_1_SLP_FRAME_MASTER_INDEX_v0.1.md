# S1.1 SLP Frame Master Index v0.1

TASK-ID: `TASK-SLP-A03-FRAME-MASTER-INDEX-v0.1`

状态：`IMPLEMENTED_AND_REAL_RUN_COMPLETE — READY_FOR_REVIEW`

## 1. 阶段目标与完成判定

本阶段目标是在不修改 SLP 原始数据的前提下，按显式主键：

`setting / subject_id / cover_condition / frame_index`

建立逐帧跨模态 Master Index，使 RGB、IR、IRraw、depth、depthRaw、PM 的配对不依赖目录排序，也不使用 `zip()` 隐式对齐。

本阶段只验证文件级结构配对、缺失保留、重复/歧义 fail-closed 和可追溯 URI；不验证图像内容可解码性、数值范围、Homography 方向或模型性能。

## 2. 实现范围

新增：

- `src/topper_perception/io/slp_frame_index.py`
- `scripts/build_slp_frame_index.py`
- `tests/test_slp_frame_index.py`

核心实现：

- 生成稳定 `sample_id`；
- 使用显式 frame index join；
- raw 模态文件合同：`IRraw/depthRaw -> .npy`；
- 图像模态文件合同：`RGB/IR/depth/PM -> image_XXXXXX.png`；
- simLab 无逐帧 PM 记为 expected structural missing，不进入 quarantine；
- 缺失模态逐帧保留，不删除整名受试者；
- 同一 modality/frame 出现多个候选时 fail closed，不按排序选择；
- 不补造 raw depth，不做 silent imputation。

## 3. 本次实际执行的命令与结果

### 3.1 定向测试

```bash
uv run pytest -q tests/test_slp_frame_index.py
```

真实结果：

```text
6 passed in 1.79s
```

覆盖：

- 显式 frame-index 配对；
- simLab PM 结构性缺失；
- depthRaw 逐帧缺失保留；
- 错误扩展名不能冒充 raw slot；
- duplicate/ambiguous frame fail closed；
- full index 主键唯一性与 integrity summary。

### 3.2 真实 SLP 全量运行

```bash
uv run python scripts/build_slp_frame_index.py --config configs/paths.local.json
```

真实运行完成并生成：

- `data/processed/slp/slp_frame_index_v0.1.csv`
- `outputs/reports/slp_frame_index_summary_v0.1.json`

本地检查显示 CSV 约 4.5 MB，summary JSON 已落盘。上述派生产物未进入 Git 工作树。

### 3.3 Diff 检查

```bash
git diff --check
```

真实结果：无输出，PASS。

### 3.4 Whole-repo regression

曾执行：

```bash
uv run pytest -q
```

结果：未完成。当前独立 worktree 未安装 optional `neural` extra，8 个 neural test 在 collection 阶段因 `ModuleNotFoundError: No module named 'torch'` 中断。

该结果不能记为 whole-repo regression PASS，也不归因于 A03 功能失败。若 Reviewer 要求完整回归，应先安装仓库定义的 optional neural dependency 后另行执行。

## 4. 核心真实结果

| 指标 | 结果 |
|---|---:|
| `rows` | 14,715 |
| `unique_primary_keys` | 14,715 |
| `duplicate_primary_key_count` | 0 |
| `missing_modality_frame_counts.depthRaw` | 90 |
| `expected_missing_modality_frame_counts.PM` | 945 |
| `ambiguous_modality_frame_counts` | `{}` |
| `quarantine_rows` | 90 |
| `pairing_method` | `explicit_frame_index_join` |
| `silent_imputation` | `false` |

结构核对：

- `14,715 = 109 subjects × 3 cover conditions × 45 frames`；
- `945 = 7 simLab subjects × 3 cover conditions × 45 frames`，对应 simLab 无逐帧 PM 的预期结构；
- `90 = 2 known depthRaw-missing groups × 45 frames`，与 S0 已知 quarantine 边界一致。

## 5. 已验证

1. SLP 全量逐帧 Master Index 可由真实数据成功构建。
2. 主键在本次真实运行中完全唯一，无 duplicate primary key。
3. 已知两组 simLab `cover2/depthRaw` 缺失被精确保留为 90 个逐帧缺失，不被删除或补造。
4. simLab PM 缺失被编码为 expected structural missing，不触发 quarantine。
5. 未发现 modality/frame ambiguous pairing。
6. pairing method 为显式 frame index join，未使用目录顺序 `zip()`。
7. `silent_imputation=false`；原始数据未被写入或修改。
8. A03 定向测试在最终版本真实通过：`6 passed in 1.79s`。
9. `git diff --check` 真实通过。

## 6. 合理推断

A03 已提供足够稳定的文件级主键与跨模态 URI 合同，可作为 A04 Homography 审计、A05 Canonical Adapter 和后续 Pilot 抽样的结构输入。

该推断仅针对“结构输入已具备”，不代表坐标映射已正确。

## 7. 尚未验证

1. A02 内容 QA：逐文件 decode、shape、dtype、finite、数值范围仍未完成。
2. Homography 实际方向、矩阵合同、round-trip error、out-of-bounds rate 尚未验证。
3. RGB/IR/Depth/PM 的 overlay 视觉对齐尚未验证。
4. Whole-repo regression 在含 optional neural dependency 的完整环境中尚未完成。
5. SLP 许可与衍生标注使用边界仍由 A01 单独处理。

## 8. 限制与禁止结论

本阶段不能据此声称：

- SLP 各模态内容均可正常解码；
- Homography 方向正确；
- 关节点映射准确；
- 区域真值存在；
- OpenCV 预标注可直接作为 Ground Truth；
- 任何 CNN/区域模型性能已得到验证；
- 自研顶垫传感器、气囊闭环、舒适性或产品效果已验证。

## 9. Reviewer Gate

Reviewer 应至少检查：

1. PR 中只包含 A03 范围变更；
2. `sample_id` 与主键合同清晰且确定性；
3. modality-specific filename contract 不会把错误格式文件误配；
4. missing/expected-missing/ambiguous/quarantine 语义分离；
5. 真实结果与 S0 已知 90 个 depthRaw 缺失一致；
6. simLab PM 945 个 expected missing 不进入 quarantine；
7. 无 raw data 修改、无 silent imputation；
8. targeted tests 与 `git diff --check` 证据成立；
9. whole-repo regression 未完成的原因被明确保留，不伪造 PASS。

Reviewer `ACCEPT` 后，A03 才可从 `READY_FOR_REVIEW` 更新为 `DONE`，并允许 A04 Homography 数学与方向审计进入执行。A02 Content QA 可作为独立任务并行推进。
