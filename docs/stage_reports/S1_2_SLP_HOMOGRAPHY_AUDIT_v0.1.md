# S1.2 SLP Homography Math and Direction Audit v0.1

TASK-ID: `TASK-SLP-A04-HOMOGRAPHY-AUDIT-v0.1`

状态：`DIRECTION_CONFIRMED_BY_README_AND_AUDIT_AND_OVERLAY — READY_FOR_REVIEW`

## 1. 阶段目标与完成判定

本阶段目标是在 S0 全量 Inventory、S1.1 Frame Master Index 已完成的基础上：

1. 建立并验证 modality↔PM 坐标变换的**数学合同**；
2. 显式区分**数学可逆/数值稳定**与**语义方向是否正确**——前者由单元测试和审计证据负责，后者由 README 文档、审计 `direct_in_bounds_rate` 指标与人工 overlay 共同确认；
3. 不调用任何模型，不宣称区域真值，不修改原始 SLP 数据。

完成判定（Gate）：

- [x] 全部 327 个对齐矩阵通过可逆性、齐次除法、round-trip 误差检查；
- [x] 语义方向由 `E:\TeamProjects\datasets\smart-topper\SLP2022\SLP\README.md` 的 "Domain Alignment" 段明确给出 `align_PTr_<modality>.npy` maps source → PM reference；
- [x] 6 名 spread-out danaLab 受试者在 frame=1 / cover=uncover 上的 RGB/IR/Depth/PM 四联 overlay 已生成，可由人工直接复核；
- [x] 单元测试 27/27 通过；与 S0/S1.1 既有 SLP 测试合并 41/41 通过；
- [x] 全部修改路径限制在 SLP 文件边界，未触碰 A03、PoPu 或原始数据；
- [x] `git diff --check` PASS。

## 2. 文件边界

本阶段**新增 / 修改**：

| 路径 | 角色 |
|---|---|
| `pyproject.toml` | 增加 `opencv-python>=5.0.0.93` 显式依赖（commit `8782602`） |
| `uv.lock` | 同步解析 opencv-python 5.0.0.93 + numpy 依赖条目 |
| `tests/test_slp_homography.py` | Homography 数学单元测试 27 例（commit `1c2b7e8`） |
| `scripts/render_slp_homography_overlay.py` | 四联 overlay 渲染器（commit `6386a9d`） |
| `src/topper_perception/geometry/slp_homography.py` | 既有实现，本阶段未修改 |
| `src/topper_perception/io/slp_homography_audit.py` | 既有实现，本阶段未修改 |
| `scripts/audit_slp_homography.py` | 既有实现，本阶段未修改 |
| `docs/stage_reports/S1_2_SLP_HOMOGRAPHY_AUDIT_v0.1.md` | 本文件 |
| `docs/PROJECT_STATUS.md` | SLP 阶段看板更新 |
| `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md` | A04 状态从 `BLOCKED_BY_A03` 更新 |

本阶段**未触碰**：

- 原始 SLP 图像、`.mat`、`.npy`、`PMcali.npy` 等只读数据；
- PoPu 任何路径、任何文件；
- A03 Frame Master Index 的 `src/topper_perception/io/slp_frame_index.py`、`scripts/build_slp_frame_index.py`；
- `configs/paths.local.json`（已在 `.gitignore`）。

## 3. 真实运行的命令与产物

### 3.1 单元测试（Homography 数学合同）

```bash
uv run pytest -q tests/test_slp_homography.py
```

真实结果：

```text
27 passed in 9.78s
```

覆盖范围：

| 单元 | 验证点 |
|---|---|
| `validate_homography` | shape=3x3、dtype 强制 float64、非有限值拒绝 |
| `homography_diagnostics` | determinant、condition number、rank、invertible；奇异与近奇异分支 |
| `invert_homography` | `H @ H⁻¹ = I`、奇异失败 fail-closed |
| `apply_homography` | 恒等 / 平移 / 齐次缩放下点投影正确；零 / 近零齐次分母抛错 |
| `roundtrip_errors` | 良好条件矩阵下 `H → H⁻¹` 误差在 float64 精度内 |
| `in_bounds_mask` | 0-based 网格边界；边界、严格边界外、内部点 |
| `direction_hypothesis_metrics` | 返回诊断证据，**不**自动选方向；奇异矩阵返回 `BLOCKED_NON_INVERTIBLE` |

### 3.2 与 S0/S1.1 既有 SLP 测试合并

```bash
uv run pytest -q tests/test_slp_inventory.py tests/test_slp_frame_index.py \
                 tests/test_slp_region_annotation_schema.py \
                 tests/test_slp_homography.py
```

真实结果：

```text
41 passed in 49.00s
```

A03 Frame Master Index、S0 Inventory、S09 Region Schema 三套既有测试在新依赖下继续通过，未发现回归。

### 3.3 真实 SLP 全量 Homography 审计

```bash
uv run python scripts/audit_slp_homography.py \
    --data-root "E:/TeamProjects/datasets/smart-topper/SLP2022/SLP"
```

真实产物（位于 `outputs/`，已在 `.gitignore`）：

- `outputs/analysis/slp_homography_audit_v0.1.csv` — 327 行（109 受试者 × 3 模态）
- `outputs/reports/slp_homography_audit_summary_v0.1.json` — 汇总 JSON

汇总指标：

| 指标 | 数值 |
|---|---:|
| rows | 327 |
| expected_rows | 327 |
| invertible_matrices | 327 / 327 |
| direction_status_counts.UNRESOLVED_REQUIRES_DOCUMENT_AND_OVERLAY_REVIEW | 204 |
| direction_status_counts.UNRESOLVED_NO_ORIGINAL_DEPTH_J0 | 109 |
| direction_status_counts.UNRESOLVED_NO_PM_REFERENCE_IMAGE | 14 |
| error_counts | `{}` |
| max_probe_roundtrip_error | 4.547 × 10⁻¹³ |
| semantic_direction_auto_selected | `false` |
| coordinate_origin_auto_shifted | `false` |

按模态拆解的可逆性与 round-trip：

| 模态 | 受试者数 | invertible | max round-trip |
|---|---:|:---:|---:|
| RGB | 109 | 109 / 109 | 4.55 × 10⁻¹³ |
| IR | 109 | 109 / 109 | 8.64 × 10⁻¹⁴ |
| depth | 109 | 109 / 109 | 2.84 × 10⁻¹³ |

### 3.4 按模态的"方向假设"指标

下表数据由审计 CSV 在 danaLab 子集（102 / 109 受试者）上聚合：

| 模态 | direct in-bounds (H·source → PM) | inverse in-bounds (H⁻¹·source → PM) |
|---|---:|---:|
| RGB | mean **0.9928**（min 0.949 / max 1.000） | mean **0.0000** |
| IR | mean **0.9927**（min 0.957 / max 1.000） | mean **0.7369** |

释义：

- **direct = `apply_homography(joints_src, H_<modality>)`**：把 J0 原始关节投到 PM 网格的 (84, 192) 边界内。两种模态都接近满分 99.3 %，与 README "Domain Alignment" 段给出的 "to reference frame which is PM" 完全一致。
- **inverse = `apply_homography(joints_src, inv(H_<modality>))`**：把 H⁻¹ 当作另一种"反向"语义、作用于同一组 source 关节。RGB 全员 0.00 %，IR 平均 73.69 %——后者较高是因为 IR (120×160) 与 PM (84×192) 几何相近，仅是巧合而非方向证据；RGB 在 PM (84×192) 网格下严格 0 % 是 H 不可能把 source→source 反向的强证据。

审计模块刻意**不**自动选方向（`semantic_direction_auto_selected=false`），direction 决策由文档 + 人工 overlay 共同承担。

### 3.5 真实 SLP overlay 渲染

```bash
uv run python scripts/render_slp_homography_overlay.py \
    --data-root "E:/TeamProjects/datasets/smart-topper/SLP2022/SLP" \
    --subject-count 6 --frame-index 1 --cover uncover
```

真实产物（位于 `outputs/`，已在 `.gitignore`）：

- `outputs/figures/A04_overlay_samples/00001_composite.png` … `00102_composite.png` — 6 张四联 overlay
- `outputs/reports/slp_homography_overlay_manifest_v0.1.json` — 每名受试者的方向假设 + in-bounds 数

每张 composite 由 4 个 panel 组成（左上 RGB、右上 IR、左下 depth、右下 PM），panel 上方有标签条记录显示语义：RGB / IR panel 携带 J0 source 关节（蓝 / 黄圆点 + 骨架），depth panel 仅作参考（无 J0 真值），PM panel 携带由 H_RGB 与 H_IR 投影后的 J1 关节（红 / 黄圆点 + 骨架）。

按受试者的 in-bounds 率（覆盖 6 名 danaLab 抽样）：

| subject | RGB → PM (12-14/14 in-bounds) | IR → PM (12-14/14 in-bounds) |
|---|:---:|:---:|
| 00001 | 14/14 | 14/14 |
| 00021 | 12/14 | 14/14 |
| 00041 | 14/14 | 14/14 |
| 00061 | 14/14 | 14/14 |
| 00081 | 14/14 | 14/14 |
| 00102 | 14/14 | 13/14 |

视觉检查 6 张 composite，PM 面板的投影骨架与压力图上的人体位置、左右、头脚方向一致；RGB-derived（红）与 IR-derived（黄）两个骨架在身体中线高度重合，进一步印证 H_RGB、H_IR 都把 source 投到 PM 的同一坐标框架。

### 3.6 Git diff 边界检查

```bash
git diff --check
```

真实结果：无输出，PASS。

合并到 main 之后，A04 分支相对于本地 baseline 仅在 SLP 边界内新增 / 修改文件；与 A03、PoPu、原始数据完全分离。

## 4. 已验证

1. SLP 全部 109 名受试者 × 3 模态 = 327 个 `align_PTr_<modality>.npy` 矩阵 100 % 可逆；
2. round-trip 误差（5 点 probe 的 H → H⁻¹）最大值 4.55 × 10⁻¹³，处于 float64 精度；
3. 齐次除法在零 / 近零分母下抛错而非静默产生 infinity；
4. README 文档明确 `align_PTr_<modality>` maps source → PM；
5. 审计 direct-in-bounds 指标（danaLab RGB/IR 均值 99.3 %）与 README 描述一致；
6. 6 名抽样 danaLab 受试者的 PM overlay 视觉显示 H_RGB、H_IR 投影结果落在人体位置、左右、头脚方向均一致；
7. 单元测试 27/27、SLP 全套测试 41/41 通过；
8. `git diff --check` PASS；
9. 未触碰原始数据、未生成 region 训练集、未启动任何模型。

## 5. 合理推断

- 在 README 文档明确与 overlay 视觉证据一致的前提下，**danaLab 三模态（RGB/IR/depth）→ PM 的语义方向已收敛**：直接 = H，H⁻¹ 不用于正向数据流；J1 派生节点应仅作 `derived_homography_bias_possible` 处理。
- simLab（7 名受试者）没有逐帧 PM，因此无法通过 in-bounds 指标验证方向；但其矩阵数学性质与 danaLab 一致（invertible、round-trip 在 float64 精度），可暂时沿用相同方向契约，并在 S2 拆分前由 A05 Canonical Adapter 通过 reference image 或 PMcali 单独复核。
- audit `direct_in_bounds_rate` 在 danaLab RGB/IR 上极少低于 0.95；个案掉出 PM 网格的关节通常对应 J0 中标记为 occluded/uncertain 的极端坐标点，应在 A07 / A11 阶段与遮挡标志一起处理，不在 A04 阶段提前决策。

## 6. 尚未验证

1. **H 的真实逐坐标残差**：当前 probe 是网格 5 点；A05 Canonical Adapter 需要逐节点逐帧回归残差报告。
2. **simLab 方向单独验证**：simLab 无 PM 参考图像，需要靠 reference icon（`align_PM001.png/.npy` 等）或 PMcali 单独复核；本阶段只能沿用 danaLab 结论。
3. **Origin 与 origin offset**：当前 `coordinate_origin_status = UNRESOLVED_RAW_DATASET_COORDINATES_NO_OFFSET_APPLIED`，尚未确认 SLP 是否约定 (0,0) 起点与图像 (0,0) 一致；A05 应单独检验。
4. **跨 cover condition 的稳定性**：本阶段只覆盖 `cover=uncover`；`cover1` / `cover2` 的 overlay 与 in-bounds 应在 A07/A15 阶段补做。
5. **A02 内容 QA**：解码、shape、dtype、finite、数值范围仍未覆盖；不影响 A04 方向合同，但影响后续模型读取。
6. **Whole-repo regression**：仍受 `torch` 可选依赖缺席影响，未在含 neural extra 的环境下完成。

## 7. 限制与禁止结论

本阶段**不能**据此声称：

- A04 已确认任何 **simLab** 模态的语义方向；
- `align_PTr_<modality>.npy` 在所有 cover、所有受试者、所有帧上投影都精确无误（per-frame 残差属 A05）；
- SLP 各模态内容均可正常解码（属 A02）；
- 关节点映射到 PM/Depth 的 `J1` 节点是无偏真值（**只能**作为 `derived_homography_bias_possible` 的派生参考）；
- 任何 OpenCV 区域 proposal 已可用作区域训练真值；
- 任何 CNN/区域模型在 SLP 上已得到验证；
- 自研顶垫传感器、气囊闭环、舒适性或产品效果已验证。

## 8. Reviewer Gate

Reviewer 应至少检查：

1. `pyproject.toml` / `uv.lock` 中 `opencv-python>=5.0.0.93` 是否与 A11 计划中关于 `opencv-contrib` 是否需要 headless 保持一致；
2. `tests/test_slp_homography.py` 是否覆盖了**奇异 / 近奇异、零 / 近零分母、空输入、dtype 强制 float64** 等关键边界；
3. `scripts/audit_slp_homography.py` 与 `src/topper_perception/io/slp_homography_audit.py` 的 `direction_status` 是否仍在所有路径下保持 `UNRESOLVED_*` 或 `BLOCKED_*`，没有悄悄把 `direct = H` 写死到业务默认；
4. `outputs/figures/A04_overlay_samples/*.png` 中 PM 面板的投影骨架是否与压力图上的人体位置、左右、头脚方向一致；
5. `outputs/analysis/slp_homography_audit_v0.1.csv` 中 327 行无 `error_codes` 字段，且 `invertible` 全 True；
6. `outputs/reports/slp_homography_audit_summary_v0.1.json` 的 `semantic_direction_auto_selected=false`；
7. `git diff --check` PASS，且本次 commit 不包含 A03、PoPu、原始数据或 `paths.local.json` 的修改；
8. 本阶段没有把 R0/R1/J1 节点升级为任何训练真值。

Reviewer `ACCEPT` 后，A04 可从 `BLOCKED_BY_A03` 更新为 `DIRECTION_CONFIRMED`，并允许 A05 Canonical Sample / Adapter 启动。