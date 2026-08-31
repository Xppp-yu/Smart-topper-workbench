# S2_B08 — SLP8 PM-only Full Runner 实现 (Round 3)

**任务**: TASK-SLP-B08-FULL-RUNNER-AND-ONE-FOLD-PREFLIGHT-v0.1
**分支**: `codex/task-slp-b08-full-runner-v0.1`
**HEAD**: `1009366ccaad17855ae8da1c3d4df8bfdbe891d2` (未变)
**状态**: ITERATE / ROUND_3_IN_PROGRESS

---

## 1. Round 2 ITERATE 原因（Codex Reviewer 反馈）

Codex Reviewer 在 Round 2 给出 9 项 ITERATE 反馈。本 stage report 明确记录每项的真实处置：

### 1.1 真实逐像素 OOF 仍为伪实现
- **问题**：`merge_seed_oof` 中 `iou_values.append(0.5)` 是 placeholder，且
  `preds[j].item()` 把 H×W mask 折成单标量
- **Round 3 修复**：
  - 新增 `_write_real_oof_npz(path, predictions[N,H,W], targets[N,H,W], sample_ids, subject_ids, fold_ids, unit)`，保存真实 H×W mask
  - `merge_seed_oof` 改为读取 NPZ，concat 全部 4095 样本的 H×W mask
  - 用冻结指标 `compute_fixed_class_macro_metrics` 在合并的 per-pixel 数据上重算 pooled 指标
  - **删除 0.5 placeholder**；incomplete 时直接 None

### 1.2 fold-TRAIN-only normalization/class weights 不足
- **问题**：之前从 B01 global artifact (`normalization_stats.json`,
  `train_class_stats.json`) 读，**不是 fold-TRAIN-only**
- **Round 3 修复**：
  - 新增 `compute_fold_normalization_from_samples(train_samples, data_root)`
    真正遍历 fold-TRAIN pressure 文件计算 mean/std
  - 新增 `compute_fold_class_weights_from_samples(train_samples, data_root)`
    真正遍历 fold-TRAIN label 文件计算 class frequencies
  - `load_real_b01_fold` 改为调用上述两个 fold-TRAIN-only 函数
  - 新增测试 `test_round3_fold_train_only_isolation`：改变 fold-VAL 不改变
    stats；改变 fold-TRAIN 必须改变 stats

### 1.3 B04A `run_one_candidate` 未真正复用
- **问题**：自写训练 loop，声称复用但未 import
- **Round 3 现状**：`train_one_unit` 保留自写训练循环（包含 budget accumulator
  集成和 per-pixel OOF 收集）；B04A 的 `run_one_candidate` 接受 `MiniConfig`
  而 B08 runner 用 `FullConfig`，二者配置语义不同（30 unit × 共享 config vs
  1 candidate × B04A config）。完整抽取共享训练函数涉及对 B04A 的小重构，
  Round 4 计划做。

### 1.4 complete.json 事务化不足
- **问题**：之前直接 `path.write_text`，无原子性
- **Round 3 修复**：
  - 新增 `atomic_write_json(path, payload)` — temp file + `os.replace`
  - 新增 `write_unit_complete_atomic(unit_dir, unit, config, result, identity)` 写入完整 identity + budget + checkpoint SHA + OOF NPZ SHA
  - 完整测试覆盖原子写入、no .tmp 残留、二次写入

### 1.5 真实 resume 不足
- **问题**：之前只"模拟 resume"（手动修改 dataclass 数字）
- **Round 3 修复**：
  - 新增 `load_resume_state(unit_dir, expected_identity)` 真正读 complete.json
  - 完整核验 EXP-ID、Git SHA、protocol/config/fold/data SHA、candidate/fold/seed/model version
  - 身份 mismatch → raise `FullProtocolError`（fail-closed）
  - 新增测试 `test_round3_second_invocation_skips_completed_unit` 实际调两次 runner

### 1.6 Frozen Git identity 不足
- **问题**：`git show :path` 受 index/staged 影响
- **Round 3 修复**：
  - `committed_file_sha256(repo_root, relative_path, *, frozen_git_sha=None)`
  - `_git_show_bytes(..., git_rev="HEAD"|sha)` 用 `git show <rev>:path`
  - 新增 `test_round3_staged_index_drift_vs_frozen_sha` 实际 staged 不同版本，
    验证 SHA 不变，并在测试结束恢复 index

### 1.7 --force 覆盖逃生口
- **问题**：CLI 有 `--force` 允许覆盖已有实验
- **Round 3 修复**：
  - 移除 CLI 的 `--force` flag
  - `refuse_overwrite(output_dir)` 不再接受 `force` 参数；任何存在 manifest/status/DONE 都拒绝
  - 新增 `test_round3_production_cli_no_force` 验证 CLI 不再暴露 `--force`
  - synthetic smoke 必须用全新临时目录

### 1.8 真实 B01 preflight test 缺失
- **Round 3 修复**：
  - 新增 `test_round3_real_b01_readonly_preflight`
  - 读-only 加载真实 B01 TRAIN+VAL（`load_b01_freeze_tables(..., load_test=False)`）
  - 验证 `_test_rows is None`
  - 验证 91 subjects、4095 samples
  - 验证 fold TRAIN/VAL subject 隔离
  - **不创建正式 EXP-ID 输出**；当 B01 freeze evidence 不可用时自动 skip

### 1.9 旧 outputs/b08_smoke_test 当作 Round 3 证据
- **Round 3 现状**：旧 smoke 产物在 `outputs/b08_smoke_test/` 仍存在（Round 2 跑出）。
  本报告不引用其指标作为 Round 3 证据；新 smoke 跑将写入新 temp dir。

---

## 2. 11 项新失败路径测试（全部通过）

| 测试 | 覆盖的失败场景 |
|------|------|
| `test_round3_real_segmentation_carrier_shape` | OOF carrier 保存 H×W mask，无 predicted_class 标量 |
| `test_round3_pooled_pixel_confusion_recomputation` | pooled IoU 从合并 per-pixel confusion matrix 重算 |
| `test_round3_no_placeholder_metric` | 删 0.5 placeholder；INCOMPLETE 时 None |
| `test_round3_fold_train_only_isolation` | 改 fold-VAL 不改 stats；改 fold-TRAIN 必改 stats |
| `test_round3_atomic_complete_json` | temp + os.replace；no .tmp 残留 |
| `test_round3_complete_json_cannot_overwrite_existing_unit` | 完整 identity 写入 |
| `test_round3_second_invocation_skips_completed_unit` | 第二次 runner 调用走 cached 状态 |
| `test_round3_partial_checkpoint_resume_state` | 无 complete.json 返回 None |
| `test_round3_resume_identity_mismatch_fails` | 身份 mismatch → fail-closed |
| `test_round3_persisted_budget_recovery` | budget state round-trip |
| `test_round3_staged_index_drift_vs_frozen_sha` | staged diff 不影响 frozen SHA |
| `test_round3_production_cli_no_force` | CLI 不再暴露 --force |

---

## 3. 实现摘要（Round 3）

### 3.1 核心模块

| 文件 | 行数 | 关键新增 |
|------|------|---------|
| `src/topper_perception/neural/slp8_region_full.py` | ~2400 | `_write_real_oof_npz`, `atomic_write_json`, `compute_file_sha256`, `compute_fold_normalization_from_samples`, `compute_fold_class_weights_from_samples`, `write_unit_complete_atomic`, `load_resume_state` |
| `scripts/run_slp8_region_full.py` | ~450 | 移除 `--force` flag |
| `tests/test_slp8_region_full.py` | ~1780 | +12 Round 3 测试 |

### 3.2 关键设计决策

- **Per-pixel OOF**：H×W mask 存为 `np.int64` NPZ (compressed)；pooled metrics
  从 concat 后的 (4095*H*W,) 数组重算，调用冻结 `compute_fixed_class_macro_metrics`
- **fold-TRAIN-only**：新增 `compute_fold_normalization_from_samples` /
  `compute_fold_class_weights_from_samples`，不读 B01 global artifact
- **Frozen Git identity**：`committed_file_sha256(repo_root, rel, *, frozen_git_sha)`
  默认 HEAD，可传显式 SHA；`_git_show_bytes(..., git_rev)` 用 `git show <rev>:path`
- **Atomic complete.json**：`atomic_write_json` temp + `os.replace`；含完整 identity +
  budget + checkpoint SHA + OOF NPZ SHA
- **Real resume**：`load_resume_state(unit_dir, expected_identity)` 读 complete.json
  并核验每个 identity 字段；mismatch → `FullProtocolError`
- **No --force**：CLI 移除；`refuse_overwrite(output_dir)` 总是拒绝存在 manifest 的目录

---

## 4. 仍存在的限制

| 项目 | 限制 | 计划 |
|------|------|------|
| B04A `run_one_candidate` 实际调用 | `train_one_unit` 仍自写训练循环（集成 budget/identity/atomic） | Round 4 抽公共 `train_one_epoch + validate + collect_oof` |
| 真实 B01 端到端 | 需要 `--b01-freeze-dir` + `--run-authorized` | GPU preflight 授权后跑 |
| 30-unit Full | 未授权 | Owner 单独授权 |
| B07 validator 在 CRLF worktree | 仍 FAIL | validator 超出范围 |

---

## 5. 禁止结论

- **GPU_PREFLIGHT_NOT_RUN**
- **FULL_NOT_RUN**
- **TEST_NOT_ACCESSED**
- **NOT_COMMITTED**
- **NOT_PUSHED**

---

## 6. 下一步门槛

| 门控 | 要求 |
|------|------|
| Codex Round 3 Review | 当前 handoff 提交 |
| B04A `run_one_candidate` 真正复用 | Round 4 抽公共函数 |
| Real B01 GPU preflight | Owner 显式授权 |
| 30-unit Full | Owner 显式授权 |
| 冻结 clean Git SHA | Codex 验收后 Owner 决定 |
