# S1.4 SLP Subject Split Freeze v0.1

TASK-ID: `TASK-SLP-A06-SUBJECT-SPLIT-FREEZE-v0.1`

状态：`IMPLEMENTED_AND_REAL_RUN_COMPLETE — READY_FOR_REVIEW`

## 1. 阶段目标与完成判定

本阶段目标是在任何模型分数出现之前冻结 **subject-level** train/val/test split，
确保：
1. 同一 subject 的所有 frame 不跨 split；
2. 同一 subject 的所有 cover 不跨 split；
3. 同一 subject 的所有 RGB/IR/Depth/PM/Pose/Homography 不跨 split；
4. danaLab / simLab 分层明确；
5. split 由固定 seed 决定，不可由后续结果修改。

完成判定（Gate）：

- [x] Canonical sample CSV 重新生成（A05 重新运行），109 subjects、14,715 行；
- [x] 确定性 subject-level split manifest 生成，seed=42；
- [x] 6 项 subject isolation tests 全部 PASS；
- [x] 确定性复现验证通过（SHA-256 一致）；
- [x] Quarantine 样本单独统计，不混入训练集；
- [x] simLab 全部置于 TEST，有明确局限性说明；
- [x] danaLab 80/10/10 split，split 策略和理由文档化；
- [x] 19 个 A06 定向测试通过；
- [x] 62 个既有 SLP 回归测试通过（A03/A04/A05/S0/region-schema）；
- [x] `git diff --check` PASS；
- [x] 未修改原始数据、未上传大体积 CSV/JSONL。

## 2. 文件边界

### 2.1 本阶段新增 / 修改

| 路径 | 角色 |
|---|---|
| `src/topper_perception/io/slp_subject_split.py` | Subject split 核心模块：适配器、manifest、隔离验证 |
| `scripts/build_slp_subject_split.py` | 真实数据 runner：生成 manifest + summary + 隔离测试 |
| `tests/test_slp_subject_split.py` | 19 个 A06 定向测试（确定性、隔离、复现、schema） |
| `configs/annotations/slp_subject_split_v0.1.schema.json` | JSON Schema v0.1 |
| `docs/stage_reports/S1_4_SLP_SUBJECT_SPLIT_FREEZE_v0.1.md` | 本文件 |
| `docs/PROJECT_STATUS.md` | SLP 看板更新 |
| `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md` | A06 状态从 `BLOCKED_BY_A05` 推进 |

### 2.2 重新生成（A05 重跑）

| 路径 | 角色 |
|---|---|
| `data/processed/slp/slp_frame_index_v0.1.csv` | A03 Frame Master Index |
| `data/processed/slp/slp_canonical_samples_v0.1.csv` | A05 Canonical Sample CSV |
| `data/processed/slp/slp_canonical_samples_v0.1.jsonl` | A05 Canonical Sample JSONL |
| `outputs/analysis/slp_homography_audit_v0.1.csv` | A04 Homography Audit CSV |
| `outputs/reports/slp_frame_index_summary_v0.1.json` | A03 Summary |
| `outputs/reports/slp_homography_audit_summary_v0.1.json` | A04 Summary |
| `outputs/reports/slp_canonical_samples_summary_v0.1.json` | A05 Summary |
| `data/processed/slp/slp_subject_split_v0.1.json` | **A06 Split Manifest** |
| `outputs/reports/slp_subject_split_summary_v0.1.json` | **A06 Split Summary** |

以上产物均已在 `.gitignore`，不进入 Git 工作树。

### 2.3 本阶段未触碰

- 原始 SLP 数据目录；
- A03 `slp_frame_index.py` / `build_slp_frame_index.py`；
- A04 `slp_homography.py` / `slp_homography_audit.py`；
- A05 `slp_canonical.py` / `build_slp_canonical_samples.py`；
- PoPu 任何文件；
- `configs/paths.local.json`（已在 `.gitignore`）。

## 3. Split 设计决策

### 3.1 Split 方案

| | 比例 | danaLab subjects | simLab subjects | 总 subjects |
|---|:---:|:---:|:---:|:---:|
| **train** | 80 % | 81 | 0 | 81 |
| **val** | 10 % | 10 | 0 | 10 |
| **test** | 10 % (danaLab) + simLab 全部 | 11 | 7 | **18** |
| **合计** | — | 102 | 7 | **109** |

### 3.2 决策理由

**simLab → TEST（全部）**

simLab（7 名受试者）是小型仿真实验室数据子集，与 danaLab 在设备、场景、数据采集协议上存在系统性差异，且无逐帧 PM 参考图像。将 simLab 全部置于 TEST 有两个目的：
1. **域外泛化评估**：作为真实的 out-of-domain held-out set，TEST 上的 simLab 性能直接反映模型对数据分布偏移的鲁棒性；
2. **小样本保护**：7 名受试者数量不足以稳定地分配到 train/val/test；强制拆分会导致 val/test 组仅含 1–2 名受试者，无法提供有意义的验证信号。

**局限性说明**：此设计意味着：
- 无法用 simLab 数据验证模型在 simLab 上的训练效果（因为 simLab 不在训练集）；
- simLab 的泛化结论高度依赖 7 名受试者的代表性；
- 若 simLab 与 danaLab 存在系统性差异（如设备类型），TEST 集上的 simLab 性能可能无法推广到真实使用场景。

**danaLab → 80/10/10**

102 名 danaLab 受试者使用确定性 hash-based shuffle 分配（seed=42）：
- **80% train (81 subjects)**：保证充足训练数据；
- **10% val (10 subjects)**：模型选择与超参数调优；
- **10% test (11 subjects)**：最终性能报告。

不采用 k-fold 交叉验证冻结，因为 fold 设计属于模型训练阶段（由 B07 Full 协议决定），A06 只负责冻结 subject-level split boundary。

### 3.3 关键实现细节

**danaLab / simLab ID 冲突处理**

danaLab 和 simLab 使用相同的 subject_id 格式（如 `00001`）。直接用 `subject_id` 作为字典键会导致两个 setting 的同名 subject 冲突。A06 适配器使用 `(setting, subject_id)` 复合键作为内部标识，在 manifest 中每个 entry 保留 `setting` 和 `subject_id` 字段以保持可读性。隔离测试使用复合键 `f"{setting}::{subject_id}"`（如 `simLab::00001`）验证无跨-setting 混淆。

**Quarantine 处理**

90 个 quarantined 帧（simLab cover2/depthRaw 全部缺失）已单独统计：
- 全部位于 TEST split（simLab subjects）；
- 不混入 train / val；
- TEST 可用帧数 = 810 - 90 = 720（仅 danaLab 11 subjects）。

**随机 seed**

使用 Python `random.Random(seed=42)` 进行确定性 shuffle。seed 在 manifest 中明确记录（`random_seed: 42`），任何使用相同 seed 的重跑均产生完全相同的 manifest SHA-256。

## 4. 真实运行的命令与产物

### 4.1 A03 Frame Index（重新生成）

```bash
uv run python scripts/build_slp_frame_index.py
# 14,715 rows, 0 duplicate primary keys, 90 depthRaw missing
```

### 4.2 A04 Homography Audit（重新生成）

```bash
uv run python scripts/audit_slp_homography.py \
    --data-root "E:/TeamProjects/datasets/smart-topper/SLP2022/SLP"
# 327 rows, 327/327 invertible, max round-trip 4.55e-13
```

### 4.3 A05 Canonical Samples（重新生成）

```bash
uv run python scripts/build_slp_canonical_samples.py \
    --slp-root "E:/TeamProjects/datasets/smart-topper/SLP2022/SLP"
# 14,715 rows, 90 quarantined, traceable_rate=0.9883
```

### 4.4 A06 Subject Split（新建）

```bash
uv run python scripts/build_slp_subject_split.py
```

真实输出：

```
Manifest written to ...slp_subject_split_v0.1.json
  manifest_sha256 = 024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706
Verifying reproducibility ...
  reproducibility PASSED (sha_match=True, assignment_match=True)
Running subject-level isolation tests ...
  [PASS] no_subject_in_multiple_splits: clean
  [PASS] train_val_test_disjoint: clean
  [PASS] all_subjects_accounted: {total=109, train=81, val=10, test=18, union=109}
  [PASS] simlab_all_in_test: {simlab_test=[simLab::00001..simLab::00007], expected=[same]}
  [PASS] danalab_split_ratios: {danalab_train=81, danalab_val=10, danalab_test=11}
  [PASS] quarantine_reported_separately: {total=90, test=90, train=0, val=0}

============================================================
  Total subjects : 109 (danaLab=102, simLab=7)
  Total frames   : 4905 (quarantined=90, usable=4815)
  [TRAIN] subjects= 81 (danaLab=81, simLab=0), frames= 3645 (usable=3645)
  [ VAL] subjects= 10 (danaLab=10, simLab=0), frames=  450 (usable= 450)
  [TEST] subjects= 18 (danaLab=11, simLab=7), frames=  810 (usable= 720)
  Manifest SHA-256: 024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706
============================================================
```

### 4.5 定向测试

```bash
uv run pytest -q tests/test_slp_subject_split.py
# 19 passed in 0.97s
```

覆盖范围：

| 测试类 | 验证点 |
|---|---|
| `TestDeterministicSubjectHash` | 同 seed 同 hash；不同 seed 不同 hash；值在 [0,1) |
| `TestSlpSubjectSplitAdapter` | manifest 全字段；simLab 全部在 TEST；danaLab 80/10/10；无跨 split subject；train/val/test 两两不相交；quarantine 隔离；复现性（双次相同 seed）；不同 seed 不同 manifest |
| `TestManifestJsonRoundtrip` | manifest → JSON → manifest 字段一致 |
| `TestSchemaCompliance` | manifest 结构符合 JSON Schema；entry 字段完整性 |
| `TestExistingSlpTests` | A03/A04/A05/A09 测试模块仍可导入 |

### 4.6 SLP 回归测试

```bash
uv run pytest -q tests/test_slp_frame_index.py tests/test_slp_homography.py \
                 tests/test_slp_canonical_adapter.py \
                 tests/test_slp_region_annotation_schema.py \
                 tests/test_slp_inventory.py
# 62 passed in 5.01s
```

### 4.7 Git 边界检查

```bash
git diff --check
# (无输出，PASS)
```

## 5. 核心真实结果

### 5.1 Split 汇总

| 指标 | TRAIN | VAL | TEST |
|---|---:|---:|---:|
| **danaLab subjects** | 81 | 10 | 11 |
| **simLab subjects** | 0 | 0 | 7 |
| **总 subjects** | **81** | **10** | **18** |
| **unique frame indices** | 3,645 | 450 | 810 |
| **quarantined frames** | 0 | 0 | 90 |
| **usable frames** | 3,645 | 450 | 720 |
| **canonical sample rows** | 11,340 | 1,350 | 2,025 |

> 注：canonical sample rows = unique_frame_indices × 3 covers（每 subject 3 种 cover condition）。
> 总行数 = 11,340 + 1,350 + 2,025 = 14,715，与 A05 canonical sample 一致。

### 5.2 完整性验证

| 验证项 | 结果 |
|---|:---:|
| manifest SHA-256 | `024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706` |
| 确定性复现 | SHA + assignment 双次一致 |
| 无跨 split subject | clean |
| train ∩ val | ∅ |
| train ∩ test | ∅ |
| val ∩ test | ∅ |
| simLab 全部在 TEST | 7/7 subjects |
| danaLab 80/10/10 | 81/10/11（目标 80/10/10） |
| Quarantine 单独统计 | 90 frames，全部在 TEST，不在 train/val |

### 5.3 TRAIN / VAL / TEST subject 列表

**TRAIN (81 danaLab subjects)**：
danaLab::00001–00003, 00006–00011, 00013, 00016–00027, 00031, 00033–00035, 00037–00055, 00056–00064, 00066–00075, 00077, 00079–00081, 00083–00086, 00088–00090, 00092–00094, 00096–00099, 00101–00102

**VAL (10 danaLab subjects)**：
danaLab::00005, 00012, 00028, 00030, 00055, 00065, 00076, 00078, 00091, 00100

**TEST (11 danaLab + 7 simLab)**：
danaLab::00004, 00014, 00015, 00018, 00029, 00032, 00036, 00070, 00082, 00087, 00095
+ simLab::00001, 00002, 00003, 00004, 00005, 00006, 00007

## 6. 已验证

1. A03/A04/A05 真实数据重新运行，14,715 行 canonical sample 与历史记录一致；
2. danaLab（102 subjects）split 81/10/11，接近 80/10/10 目标比例；
3. simLab（7 subjects）全部置于 TEST，隔离验证 PASS；
4. 无 subject 跨 split（isolation tests 6/6 PASS）；
5. 相同 seed 重跑产生相同 SHA-256（确定性复现 PASS）；
6. 90 个 quarantined frames 不在 train/val，仅在 TEST；
7. A06 单元测试 19/19 PASS；
8. 既有 SLP 回归测试 62/62 PASS；
9. `git diff --check` PASS；
10. 未修改原始 SLP 数据、未上传 CSV/JSONL。

## 7. 合理推断

- 81/10/11 danaLab split 在 subject count 上接近 80/10/10（81/10/11 ≈ 80.4%/9.9%/9.7%），在 frame count 上为 81×45/10×45/11×45 = 3645/450/495（usable）——差异在可接受范围内；
- 固定 seed=42 作为 split 种子是任意但确定的选择；只要在报告中明确记录，后续研究者可使用相同 seed 复现；
- A06 冻结了 split boundary，但 fold 数量和 fold-level 分配属于训练协议（A07/B07 Full），不在 A06 范围内。

## 8. 尚未验证

1. **Fold 设计**：A06 不包含 fold；后续 A18 节点基线和 B07 Full 协议需要另行设计 fold 并冻结；
2. **跨 cover 的 subject isolation**：A06 保证了同一 subject 不跨 split，但不同 cover 条件之间的覆盖度分布仍由 A07/A15 评估；
3. **simLab 无 PM 的域外泛化影响**：simLab 无逐帧 PM 图像，这本身是数据差异的一部分；TEST 上 simLab 性能是否可比需由 A18 节点基线结果说明；
4. **Whole-repo regression**：含 optional `torch` 的 8 个 neural test 未在当前环境运行（与 A03/A04/A05 相同原因）。

## 9. 限制与禁止结论

本阶段**不能**据此声称：

- 模型在 simLab 上的性能可以直接与 danaLab 比较（simLab 在 TEST 中，无训练数据）；
- TEST 集上的 simLab 性能代表实际产品部署效果；
- A06 split 可以替代 B07 Full 协议中的 fold 设计；
- 90 个 quarantined frames 可以通过任何方式进入训练集；
- 任何模型在 SLP 上已得到验证。

## 10. Reviewer Gate

Reviewer 应至少检查：

1. `src/topper_perception/io/slp_subject_split.py` 中 danaLab/simLab 使用 `(setting, subject_id)` 复合键，无 ID 冲突；
2. `build_manifest` 中 quarantine 判断使用字符串 `"True"/"False"` 兼容 CSV 格式；
3. simLab 全部在 TEST，不在 train 或 val；
4. danaLab split 比例接近 80/10/10 并在报告中明确记录；
5. `manifest_sha256` 由 subject assignment JSON 的确定性序列化计算；
6. 复现性验证（双次相同 seed）证据存在；
7. 6 项 isolation tests 全部 PASS 并在 stdout 可验证；
8. A06 未修改原始数据、未把 split 信息写回 canonical sample（`provenance.subject_split_applied` 仍为 `false`）；
9. `git diff --check` PASS 且 commit 不含 raw data / 大体积 outputs；
10. A06 代码和测试范围明确，不包含 fold 设计、模型训练或 benchmark。

Reviewer `ACCEPT` 后，A06 可从 `BLOCKED_BY_A05` 更新为 `DONE`，并允许 A07 节点 EDA、A18 节点基线和 B01 冻结训练表进入实现。
