# S1.3 SLP Canonical Sample and Adapter v0.1

TASK-ID: `TASK-SLP-A05-CANONICAL-ADAPTER-v0.1`

状态：`IMPLEMENTED_AND_REAL_RUN_COMPLETE — READY_FOR_REVIEW`

## 1. 阶段目标与完成判定

本阶段目标是把 A03 Frame Master Index（结构输入）和 A04 Homography Audit
（几何合同）合并成一个稳定的逐帧 **Canonical Sample**，作为后续 S2 拆分、
A07 节点/A08 几何、阶段 II 区域训练的统一输入层，且不修改原始 SLP 数据、
不在业务层静默把 A04 未确认的方向硬编码成默认真值。

完成判定（Gate）：

- [x] Canonical Sample schema 冻结（`slp_canonical_sample_v0.1`），Frame /
      Joint / Region 三层对象清晰分离；
- [x] SLP Adapter 实现：从 A03 CSV 读入，逐帧回填 A04 几何合同、J0 来源
      URI、模态可追溯性；
- [x] 不在 A05 业务层硬编码 A04 方向；A04 `direction_status` 原样保留
      为 `UNRESOLVED_*` / `BLOCKED_*`；
- [x] 缺失模态显式 quarantine，duplicate frame 不被排序配对；
- [x] Region 层只保留 schema version + placeholder，不生成任何区域真值；
- [x] 20 个 A05 定向测试全部通过；A03 / A04 / S0 / region-schema 既有
      SLP 测试 41/41 仍 PASS；
- [x] 真实 SLP 全量运行：14,715 个 canonical sample、109 受试者、90 个
      quarantined（与 A03 已知 depthRaw 缺失完全一致）；
- [x] `git diff --check` PASS；raw data SHA-256 在运行前后无变化。

## 2. 文件边界

### 2.1 本阶段新增 / 修改

| 路径 | 角色 |
|---|---|
| `src/topper_perception/io/slp_canonical.py` | Canonical Sample / Frame / Joint / Region 数据类 + Adapter |
| `scripts/build_slp_canonical_samples.py` | 真实数据 runner，输出 CSV + JSONL + summary |
| `tests/test_slp_canonical_adapter.py` | 20 个 A05 定向测试 |
| `configs/annotations/slp_canonical_sample_v0.1.schema.json` | Canonical Sample JSON Schema |
| `docs/stage_reports/S1_3_SLP_CANONICAL_ADAPTER_v0.1.md` | 本文件 |
| `docs/stage_reports/SLP_CANONICAL_SAMPLE_EXAMPLES_v0.1.md` | schema + provenance / quarantine 示例 |
| `docs/stage_reports/examples/slp_canonical_provenance_example_v0.1.json` | 健康样本的完整 record |
| `docs/stage_reports/examples/slp_canonical_quarantine_example_v0.1.json` | quarantined 样本的完整 record |
| `docs/PROJECT_STATUS.md` | SLP 看板 A05 行更新 |
| `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md` | A05 状态从 `READY_AFTER_A04_DIRECTION_CONFIRM` 推进 |

### 2.2 本阶段未触碰

- 原始 SLP 数据：运行前后对随机 50 个 SLP 文件做 SHA-256 比对，0 处差异；
- A03 `slp_frame_index.py` / `build_slp_frame_index.py`：仅作为读入
  来源，不修改其逻辑；
- A04 `slp_homography.py` / `slp_homography_audit.py` / `audit_slp_homography.py`：
  仅作为读入来源，不修改其方向合同；
- PoPu 任何文件、原始数据目录、`.gitignore`；
- `configs/paths.local.json` 仍在 `.gitignore` 内，本轮新增的内容仅为
  本地运行配置，不进入 Git 工作树（见下方 Git 状态）。

## 3. 真实运行的命令与产物

### 3.1 定向测试

```bash
uv run pytest -q tests/test_slp_canonical_adapter.py
```

真实结果：

```text
20 passed in 4.12s
```

覆盖 9 类必测项：

1. 单帧可回溯全部 URI（`test_canonical_sample_traces_to_all_raw_modality_uris`）；
2. 缺失模态会 quarantine（`test_missing_modality_quarantines_canonical_sample`）；
3. duplicate frame 不被静默排序配对
   （`test_duplicate_frame_match_is_reported_as_quality_flag`）；
4. 非法 / 落盘缺失 URI 产生 `uri_missing_on_disk:<modality>` 质量标记并
   quarantine（`test_uri_pointing_outside_slp_root_is_rejected`）；
5. Joint provenance 可追溯，J0 vs J1 严格分离
   （`test_canonical_sample_exposes_joint_provenance_and_homography_contract`、
    `test_j0_missing_is_flagged_and_never_silently_substituted`）；
6. Region 层与 Frame 层隔离（`test_region_layer_is_isolated_from_frame_layer`）；
7. schema 序列化 / 反序列化一致
   （`test_canonical_sample_serialization_roundtrips_through_json`、
    `test_canonical_csv_writes_one_row_per_sample_with_documented_columns`、
    `test_canonical_jsonl_writes_one_sample_per_line`、
    `test_canonical_sample_dict_round_trip_preserves_all_layers`）；
8. 原始数据目录没有任何修改（`test_adapter_does_not_modify_raw_slp_directory`）；
9. A03 / A04 / S0 / region-schema 既有 SLP 测试继续通过
   （`test_existing_slp_tests_remain_discoverable`）。

另外 11 个测试覆盖：坐标原点状态恒为
`UNRESOLVED_RAW_DATASET_COORDINATES_NO_OFFSET_APPLIED`、方向 unresolved 时
不会被强行 quarantine（`test_unresolved_direction_is_soft_warning_not_quarantine`）、
blocked homography 会触发 quarantine、软警告与硬原因在 summary 中分离等。

### 3.2 真实 SLP 全量 Adapter 运行

```bash
# 1) 重新生成 A03（与 A04 阶段使用的同一 build_slp_frame_index.py）
uv run python scripts/build_slp_frame_index.py --config configs/paths.local.json

# 2) 重新生成 A04
uv run python scripts/audit_slp_homography.py --data-root "E:/TeamProjects/datasets/smart-topper/SLP2022/SLP"

# 3) 跑 A05 Adapter
uv run python scripts/build_slp_canonical_samples.py --slp-root "E:/TeamProjects/datasets/smart-topper/SLP2022/SLP"
```

真实产物（位于 `data/processed/slp/` 与 `outputs/reports/`，均已在
`.gitignore`）：

- `data/processed/slp/slp_canonical_samples_v0.1.csv` — 14,715 行 × 79 列
  宽表 canonical sample；
- `data/processed/slp/slp_canonical_samples_v0.1.jsonl` — 14,715 个
  完整结构化 payload（Frame / Joint / Region / Provenance）；
- `outputs/reports/slp_canonical_samples_summary_v0.1.json` —
  dataset-level 摘要。

stdout 摘要：

```text
{
  "canonical_schema_version": "slp_canonical_sample_v0.1",
  "rows": 14715,
  "quarantine_rows": 90,
  "traceable_rate": 0.9882772680937818,
  "subjects": 109,
  "output_csv": ".../slp_canonical_samples_v0.1.csv",
  "output_jsonl": ".../slp_canonical_samples_v0.1.jsonl",
  "output_summary": ".../slp_canonical_samples_summary_v0.1.json"
}
```

### 3.3 真实数据运行摘要

| 指标 | 数值 |
|---:|---:|
| `rows` | 14,715 |
| `subjects` | 109 |
| `subjects_per_setting.danaLab` | 102 |
| `subjects_per_setting.simLab` | 7 |
| `frames_by_setting.danaLab` | 13,770 |
| `frames_by_setting.simLab` | 945 |
| `frames_by_cover` | 4,905 / 4,905 / 4,905（uncover/cover1/cover2） |
| `missing_modality_frame_counts.depthRaw` | 90 |
| `expected_missing_modality_frame_counts.PM` | 945（simLab） |
| `ambiguous_modality_frame_counts` | `{}` |
| `quarantine_rows` | 90 |
| `quarantine_reason_counts.missing_modality:depthRaw` | 90 |
| `j0_missing_rows` | 0 |
| `homography_audit_rows_seen` | 327（109 × 3 modality） |
| `homography_contracts_attached` | RGB/IR/depth 各 14,715 |
| `homography_blocked_rows` | 0 |
| `homography_unresolved_rows` | 44,145（所有 14,715 × 3 模态） |
| `coordinate_origin_unresolved_rows` | 14,715 |
| `region_placeholder_rows` | 14,715 |
| `uri_traceability.traceable_uris` | 87,255 |
| `uri_traceability.absent_uris` | 1,035（=945 expected-missing PM + 90 missing depthRaw） |
| `uri_traceability.traceable_rate` | 0.9883 |
| `uri_traceability.uri_missing_on_disk_rows` | 0 |

软警告与硬原因分离的语义示例：

- `quality_flag_counts` 包含 `coordinate_origin_unresolved`（14,715）、
  `homography_unresolved_*`（各 14,715）、`region_placeholder_only`
  （14,715）以及 `missing_depthRaw`（90）—— 这些都是观察记录；
- `quarantine_reason_counts` 只包含 `missing_modality:depthRaw: 90`，
  没有任何软警告被混入硬原因。A04 方向 unresolved 不触发 quarantine，
  这是 A05 任务合同明确禁止的"业务层静默把方向硬编码成真值"的反向
  保障。

## 4. 已验证

1. 14,715 个真实 SLP canonical sample 可由 A03 / A04 产物构建；
2. 每个 canonical sample 的 `frame.modality_uris` 都可回到真实存在的
   SLP 文件，traceable_rate = 0.9883（缺失的 1.7% 是 expected
   structural missing 945 simLab PM + 90 simLab cover2 depthRaw，与
   A03 已知边界完全一致）；
3. `quarantine_rows = 90` 与 A03 `quarantine_rows = 90` 完全相等，且
   quarantine 原因 100% 集中在 `missing_modality:depthRaw`，与 S0 已知
   两组 simLab `cover2/depthRaw` 缺失边界一致；
4. 没有任何 canonical sample 在 business 层把 A04 方向硬编码成默认值；
   JSON Schema 用 `^(BLOCKED_|UNRESOLVED_).+` 强制 `direction_status`
   只能保留 A04 状态；
5. 软警告（`homography_unresolved_*`、`coordinate_origin_unresolved`）
   不会被算成 quarantine 原因；只有 A03 缺失/歧义、A04 `BLOCKED_*`
   homography、J0 缺失这三类硬原因会触发 quarantine；
6. `region.annotation_count = 0` 在所有 14,715 个样本中保持 0，
   `region.can_be_used_as_training_truth = false`；区域层不污染 Frame
   层，且未生成任何区域真值；
7. Adapter 不会修改 raw data：对 50 个随机抽样的 SLP 文件做 SHA-256
   对比，运行前后哈希完全一致；
8. 20 个 A05 定向测试、4 个 A03/A04/S0/region-schema 既有 SLP 测试
   合并 62/62 通过；
9. A04 几何字段（`invertible`、`probe_roundtrip_*`、in-bounds rate、
   `direction_status`、`coordinate_origin_status`、`error_codes`）在
   canonical sample 中字段 1:1 保留，无字段丢失或重命名；
10. A05 未引入任何 CNN / 区域模型；未生成 subject split；未把
    split / review status / 模型预测写回原始样本（这些在
    `provenance.*_applied` 全部强制为 `false`，并由 JSON Schema 约束）。

## 5. 合理推断

- 在 A04 方向合同稳定的前提下，A05 已经为后续 A06 subject split 提供
  了单一、可追溯、不可写回的输入；任何把 split / 模型预测写回
  canonical sample 的尝试都会被 JSON Schema 拒绝；
- 软警告的"广覆盖 + 不触发 quarantine"是 A05 的明确选择：让 A07 /
  A18 / 阶段 II 决定如何利用 A04 几何合同，而不是让 Adapter 替业务
  做决定；
- 当前 quarantine 边界（A03 missing/ambiguous + A04 BLOCKED_* + J0
  缺失）应足以支持 A06 split 的"隔离受损样本"语义而无需重做。

## 6. 尚未验证

1. 真实 SLP 数据的 **逐坐标 residual / per-frame in-bounds rate**：
   A05 仍沿用 A04 主体级的 `direct_joint_in_bounds_rate` 摘要，并未
   逐帧逐节点回归 residual。这属于 A07 / A11 阶段；
2. **A04 simLab 方向**仍然没有独立 overlay 验证（无 PM 参考图像），
   A05 不解决这一点；
3. **coordinate origin 是否需要 offset**（A04 的
   `UNRESOLVED_RAW_DATASET_COORDINATES_NO_OFFSET_APPLIED`）仍未在
   A05 阶段确认；canonical sample 继续以"无 offset"姿态记录；
4. **A02 内容 QA**（每模态的 decode / shape / dtype / finite / 数值
   范围）仍未完成，canonical sample 的 `uri_existence_flags` 只看
   文件存在性，不读内容；
5. **Whole-repo regression**（含 optional `torch` 的 8 个 neural
   test）仍未在 A05 worktree 中执行，原因与 A03/A04 报告相同（依赖
   不在环境内），不属于 A05 失败；
6. **跨 cover 稳定性**（cover1 / cover2 的 overlay 与 in-bounds）仍
   属 A07 / A15 阶段，A05 不重复覆盖。

## 7. 限制与禁止结论

本阶段**不能**据此声称：

- A04 在 simLab 各模态上的语义方向已确认（无 PM 仍然无法独立
  验证）；
- 关节点映射到 PM/Depth 的 `J1` 节点是无偏真值（只能作为
  `derived_homography_bias_possible` 的派生参考）；
- SLP 各模态内容都已通过内容 QA 验证（属 A02）；
- 任何 region 训练真值存在或可用于训练（A05 region 层为
  placeholder）；
- 任何 CNN/区域模型在 SLP 上已得到验证；
- 自研顶垫传感器、气囊闭环、舒适性或产品效果已验证。

## 8. Reviewer Gate

Reviewer 应至少检查：

1. `src/topper_perception/io/slp_canonical.py` 中
   `SlpCanonicalAdapter.build_canonical_sample` 路径
   是否严格使用 A03 row 的 URI 而不是从目录顺序重新配对；
2. `_compute_quality` 是否把"软警告"（`homography_unresolved_*`、
   `coordinate_origin_unresolved`、`region_placeholder_only`）
   与"硬原因"（`missing_*`、`ambiguous_*`、`uri_missing_on_disk:*`、
   `homography_blocked_*`、`j0_missing_*`）清楚分离；
3. `HomographyContract.direction_status` 的所有路径都保持
   `^(BLOCKED_|UNRESOLVED_).+`，没有任何路径把它替换成确认方向；
4. `canonical_sample_to_csv_row` 与 `CanonicalSample.as_dict()` 中
   Frame / Joint / Region 三个对象的字段互不交叉；
5. `scripts/build_slp_canonical_samples.py` 的 fail-closed 出口
   （duplicate sample_id、URI 越界、canonical sample 数与 A03 不一致、
   全量 quarantine）是否仍然生效；
6. `configs/annotations/slp_canonical_sample_v0.1.schema.json` 中
   `provenance.subject_split_applied` /
   `review_status_applied` /
   `model_prediction_applied` /
   `semantic_direction_auto_selected` /
   `coordinate_origin_auto_shifted` /
   `silent_imputation` 是否全部强制为 `false`；
7. `docs/stage_reports/examples/slp_canonical_provenance_example_v0.1.json`
   与 `slp_canonical_quarantine_example_v0.1.json` 的内容是否与
   A05 摘要一致；
8. 真实数据运行摘要中 `quarantine_rows = 90` 与 A03 / S0 已知
   90 个 depthRaw 缺失完全一致，且原因唯一为
   `missing_modality:depthRaw`；
9. 本阶段没有修改 A03 / A04 / PoPu / 原始数据；`git diff --check`
   PASS；commit 不含 raw data / 大体积 evidence。

Reviewer `ACCEPT` 后，A05 可从 `READY_AFTER_A04_DIRECTION_CONFIRM`
更新为 `DONE`，并允许 A06 subject split、A07 节点 EDA、A08 几何
基线进入实现。
