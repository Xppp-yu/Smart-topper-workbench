# S2_B08 — SLP8 PM-only Full Runner 实现

**任务**: TASK-SLP-B08-FULL-RUNNER-AND-ONE-FOLD-PREFLIGHT-v0.1
**分支**: `codex/task-slp-b08-full-runner-v0.1`
**HEAD**: `1009366ccaad17855ae8da1c3d4df8bfdbe891d2`
**状态**: ITERATE / REAL_RUNNER_INCOMPLETE

---

## 1. 本阶段做什么

实现 B07 冻结协议对应的 SLP8 PM-only Full Runner，通过本地 synthetic CPU smoke 验证调度语义。Runner 不得自行 preflight、commit、push。

---

## 2. 实现摘要

### 核心模块

| 文件 | 行数 | 职责 |
|------|------|------|
| `src/topper_perception/neural/slp8_region_full.py` | ~2400 | Full runner 实现：协议加载、synthetic/real 数据路径、fold 路由、单位调度、OOF 合并、预算管理、terminal 状态 |
| `scripts/run_slp8_region_full.py` | ~450 | CLI：argparse、run_full() 入口、validate-only/no-write 零文件模式、synthetic CPU smoke |
| `scripts/smoke_b08_full_runner.py` | ~150 | 独立 smoke 脚本 |
| `tests/test_slp8_region_full.py` | ~1350 | 54 个测试（44 原有 + 10 新增 ITERATE 失败路径） |

### 关键设计决策

**路径分离**：两个完全独立的数据路径，由 `config.synthetic_mode` 守卫：
- `synthetic_mode=True`：使用 `_build_synthetic_dataloader`，调用 `build_synthetic_fold_dataset()` 生成确定性数据
- `synthetic_mode=False`：使用 `Slp8RegionDataset`，调用 `load_b01_freeze_tables(..., load_test=False)` + `partition_records_for_fold()` 按 B07 fold subject 路由

**CRLF 处理**：runner 内部使用 `committed_file_sha256(repo_root, relative_path)` 调用 `git show :path` 读取 committed tree 内容 SHA，与工作树 CRLF 解耦。

**OOF 逐样本 predictions**：real B01 模式在验证循环收集 `sample_id / subject_id / fold_id / seed / candidate / predicted_class`，写 `unit_oof.csv`，`merge_seed_oof()` 读取 5 个 fold CSV 并拼接，pooled metrics 从逐样本数据重算。

**Unit 事务化**：`UnitResult` 携带 `oof_csv_path`，`write_terminal_state()` 在成功路径写入 `complete.json` 后才改 terminal state；已有 `DONE.json` 的目录拒绝覆盖（无 `--force` 逃生口）。

**Budget accumulator**：`BudgetAccumulatorState` 跨 unit 累积，`check_budget_and_update()` fail-closed，resume 时通过 `initial_state` 参数延续累计，不归零。

**validate-only/no-write**：`run_validate_only()` 在内存中完成所有验证（协议 SHA、fold manifest SHA、TEST=0、identity、30-unit 计划），stdout 输出，**零文件创建**。

---

## 3. 已验证（54 passed）

### 原有测试（44 passed）

协议验证（8）：
- `test_protocol_must_be_b07` ✅
- `test_protocol_must_be_accepted` ✅
- `test_execution_authorized_must_be_false` ✅
- `test_execution_authorized_cannot_be_null` ✅
- `test_test_access_must_be_exactly_denied` ✅
- `test_test_access_rows_must_be_zero` ✅
- `test_fold_manifest_sha_mismatch_fails_closed` ✅
- `test_fold_manifest_test_access_denied` ✅

Fold 覆盖（3）：
- `test_fold_subject_coverage_requires_91_subjects` ✅
- `test_fold_subject_duplicates_rejected` ✅
- `test_candidate_names_must_match_b07` ✅

执行计划（4）：
- `test_load_and_plan_exactly_30_units` ✅
- `test_seeds_must_match_b07` ✅
- `test_execution_matrix_must_be_30_units` ✅
- `test_execution_plan_no_duplicates` ✅
- `test_execution_plan_all_units_covered` ✅

数据分区（5）：
- `test_partition_routes_by_subject` ✅
- `test_partition_rejects_test_record` ✅
- `test_partition_rejects_train_val_overlap` ✅
- `test_partition_requires_exact_val_coverage` ✅
- `test_partition_rejects_unsupported_split` ✅

OOF 验证（5）：
- `test_oof_exact_coverage_passes` ✅
- `test_oof_rejects_test_row` ✅
- `test_oof_rejects_duplicate_sample` ✅
- `test_oof_rejects_wrong_sample_count` ✅
- `test_oof_rejects_wrong_subject_count` ✅

Synthetic 数据（2）：
- `test_synthetic_fold_dataset_structure` ✅
- `test_synthetic_fold_dataset_reproducibility` ✅

Budget（6）：
- `test_budget_accumulator_initialization` ✅
- `test_budget_accumulator_updates` ✅
- `test_budget_fails_on_wall_exceed` ✅
- `test_budget_fails_on_cuda_exceed` ✅
- `test_budget_report_structure` ✅

OOF 合并（2）：
- `test_merge_seed_oof_complete` ✅
- `test_merge_seed_oof_incomplete_on_failed_fold` ✅

聚合与选择（3）：
- `test_aggregate_candidate_results_complete` ✅
- `test_apply_selection_rule_picks_winner` ✅
- `test_apply_selection_rule_near_tie_uses_tiebreak` ✅

输出保护（3）：
- `test_refuse_overwrite_on_existing_dones` ✅
- `test_refuse_overwrite_allows_force` ✅
- `test_refuse_overwrite_new_directory_allowed` ✅

集成 smoke（2）：
- `test_synthetic_full_run_smoke` ✅
- `test_synthetic_no_write_mode` ✅

单元属性（1）：
- `test_full_unit_properties` ✅

### 新增 ITERATE 失败路径测试（10 passed）

| 测试 | 覆盖的失败场景 |
|------|------|
| `test_real_path_never_calls_synthetic_helper` | synthetic helper 在 real 模式不被调用 |
| `test_real_path_rejects_test_row_in_b01_data` | real 路径拒绝注入 TEST row |
| `test_real_fold_respects_subject_partition` | fold TRAIN/VAL subject overlap=0 |
| `test_oof_rejects_duplicate_sample_across_folds` | 跨 fold 重复 sample → INCOMPLETE |
| `test_oof_requires_valid_subject_count` | subject 数量不对 → rejected |
| `test_refuse_overwrite_raises_on_existing_done_json` | 已有 DONE.json 拒绝覆盖 |
| `test_validate_only_requires_config_arg_subprocess` | CLI 缺 --config 参数时明确报错 |
| `test_validate_only_no_files_created_subprocess` | validate-only 零文件（subprocess 验证） |
| `test_staged_index_cannot_replace_frozen_head_sha` | git show 读取 committed tree，不受 staged/index 影响 |
| `test_budget_accumulator_persists_across_multiple_units` | budget 跨 unit 累积，不归零 |

---

## 4. 推断为真（已审查，未独立运行）

- **Real B01 路径**：`load_b01_freeze_tables(..., load_test=False)` + `_test_rows is None` 断言 + `partition_records_for_fold()` 按 B07 fold subject 路由，逻辑链路完整
- **OOF 逐样本**：`val_loop` 收集 `oof_predictions` → `_write_real_oof_csv()` → `merge_seed_oof()` 拼接，字段完整
- **Synthetic/real 守卫**：`train_one_unit()` 中 `synthetic_mode=True` 路径只调用 `_build_synthetic_dataloader`；`synthetic_mode=False` 路径只调用 `Slp8RegionDataset` + `build_dataloader`
- **resume identity**：`FullUnit` 携带全部 identity 字段，runner 在启动时核验，mismatch → fail closed
- **Budget not reset**：状态通过参数传入，累积逻辑 `+=`，不归零

---

## 5. 未验证（限制）

| 项目 | 限制原因 |
|------|------|
| 真实 B01 TRAIN+VAL 端到端训练 | 需要 `--b01-freeze-dir` + `--run-authorized`，GPU preflight 未授权 |
| RTX 4090 单 fold GPU preflight | `GPU_PREFLIGHT_NOT_AUTHORIZED` |
| 30-unit Full run | 未授权 |
| B07 validator（工作树 CRLF） | validator 恢复为 HEAD 后使用 working-tree SHA，在 CRLF worktree 下预期失败；runner 内部用 `committed_file_sha256()` 规避 |
| B04/B04A 回归测试 | 需要主工作区 `main` 分支 context |

---

## 6. 已知限制

- **CRLF worktree**：runner 用 `git show :path` 读取 committed content 规避；`validate_b07_protocol.py`（超出范围，恢复为 HEAD）在 CRLF worktree 下预期失败
- **隔离 worktree 无 B01 freeze evidence**：`slp8_training_tables_v0.1/freeze_manifest.json` 在主工作区 `.gitignore`，runner 对 synthetic 模式无影响，real 模式需要显式 `--b01-freeze-dir` 参数
- **RTX 4090 preflight**：需要 Owner 单独授权 EXP-ID
- **30-unit Full**：需要 Owner 单独授权 EXP-ID

---

## 7. 禁止结论

- GPU preflight 未运行 → **GPU_PREFLIGHT_NOT_RUN**
- 30-unit Full 未运行 → **FULL_NOT_RUN**
- TEST 数据从未访问 → **TEST_NOT_ACCESSED**
- 代码未 commit/push → **NOT_COMMITTED / NOT_PUSHED**

---

## 8. 下一步门槛

| 门控 | 要求 |
|------|------|
| Synthetic smoke | 30/30 units DONE（已验证） |
| B07 validator | 需要 `--b07-freeze-dir` 或修复 CRLF（超出范围） |
| B04/B04A 回归 | 需要主工作区 context |
| Real B01 GPU preflight | 需要 Owner 显式授权 RTX 4090 EXP-ID |
| 30-unit Full | 需要 Owner 显式授权 |
| 冻结 clean Git SHA | 需要 Codex 验收后由 Owner 决定 |
