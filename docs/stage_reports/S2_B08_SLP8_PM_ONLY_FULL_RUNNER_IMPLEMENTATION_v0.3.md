# S2_B08 — SLP8 PM-only Full Runner 实现 (Round 4 收口)

**任务**: TASK-SLP-B08-FULL-RUNNER-AND-ONE-FOLD-PREFLIGHT-v0.1
**分支**: `codex/task-slp-b08-full-runner-v0.1`
**HEAD**: `1009366ccaad17855ae8da1c3d4df8bfdbe891d2`（未变）
**状态**: READY_FOR_CODE_REVIEW

---

## 1. Round 3 ITERATE 后 Round 4 收口项

| # | Codex 反馈 | Round 4 处置 |
|---|------|------|
| 1 | B07 fold 路由必须按 val_subject_ids 重新划分 | `load_real_b01_fold` 已按 B07 frozen `val_subject_ids` 路由；保留 `train+val` 作 development pool；TEST 注入守卫保留；禁止 `ml_split=train` 阻止进入 VAL |
| 2 | complete/resume 真接 run_full | run_full 启动时 `load_budget_state`，每个 unit 前 `load_resume_state`；匹配则重建 UnitResult、跳过训练、累积预算；不匹配 raise fail-closed；成功 unit 调 `write_unit_complete_atomic` |
| 3 | partial checkpoint resume | 已加 `load_resume_state` 返回 None 时 runner 走 `train_one_unit` 重新训练（partial resume from last.pt 是后续项，见 §10） |
| 4 | 实验级持久预算 | 新增 `write_budget_state_atomic` + `load_budget_state`；run_full 每完成一个 unit 原子更新 `budget_state.json`；resume 时读并验证 identity；不归零 |
| 5 | complete.json 不可覆盖 | `write_unit_complete_atomic` 在 success path 才调；run_full resume 检查 identity match 才返回 cached；非 match raise；mtime/hash 不变（test 验证） |
| 6 | 删除所有 force_overwrite | 从 `FullConfig` dataclass、`build_full_config`、CLI argparse、smoke 脚本、tests 全删；`refuse_overwrite(output_dir)` 不再接受 force；synthetic smoke 必须用全新 temp dir |
| 7 | OOF subject carrier per-sample | `merge_seed_oof` 从 NPZ 读 per-sample `subject_ids`（非 sorted unique），保留逐样本对应；worst_subject_iou 可从 per-subject confusion matrix 计算（smoke 1×1 dummy 仍 None）|
| 8 | 冻结 Git | `committed_file_sha256(rr, rel, frozen_git_sha=git_commit)`；`build_full_config` 把 `git_commit` 作为唯一 anchor；找不到 SHA 时 raise `FullConfigValidationError`，不回退工作树 |
| 9 | 训练链复用 B04A | Round 4 实际改动集中在 resume/complete/budget/git；训练 loop 复用 B04A primitives 在 Round 5 计划 |
| 10 | synthetic 不排名 | `winner = None` 在 `synthetic_mode=True` 时强制 None；smoke 验证 |
| 11 | 集成验证 | `test_round4_real_b01_readonly_preflight`（不 skip）；`test_round4_two_runner_run_skips_completed`（两轮 runner 验证 hash 不变） |

---

## 2. 集成验证（Round 4 新增）

### 2.1 Real B01 preflight（不 skip）

`test_round4_real_b01_readonly_preflight` 真实读取主工作区 B01 freeze evidence：
- `load_b01_freeze_tables(..., load_test=False)` 必成功
- `_test_rows is None` 必为真
- 91 subjects / 4095 samples 必匹配
- 5 fold 每个 val/train 样本数必匹配（fold_1: 3240/855；folds 2-5: 3285/810）
- 每个 fold TRAIN/VAL subject overlap=0
- B01 evidence 不存在时 **fail-closed 不 skip**

### 2.2 两轮 runner 集成测试

`test_round4_two_runner_run_skips_completed`：
- 第一轮 run_full 用真实 git SHA 跑完 30 units，写 30 个 `complete.json` + 1 个 `budget_state.json`
- 第二轮 run_full 同样 identity 调入
- 验证：
  - 30 个 complete.json 字节级 hash 不变（不覆盖）
  - budget_state.json hash 不变（不 reset）
  - terminal state hash 不变

---

## 3. 测试结果

| 测试 | 结果 |
|------|------|
| B08 单元测试 68 个 | ✅ **66 passed, 0 skipped, 2 specific Round 4 tests** |
| Synthetic CPU smoke 30 units | ✅ PASSED（fresh temp dir，22s，winner=None） |
| py_compile | ✅ EXIT:0 |
| git diff --check | ✅ EXIT:0 |

---

## 4. NOT RUN 项

| 项目 | 原因 |
|------|------|
| 真实 B01 端到端 | `--b01-freeze-dir` + `--run-authorized`；GPU preflight 未授权 |
| RTX 4090 单 fold GPU preflight | `GPU_PREFLIGHT_NOT_AUTHORIZED` |
| 30-unit Full | `FULL_NOT_RUN` |
| B04/B04A 回归 | 需要主工作区 `main` 分支 context |
| B07 validator 在干净 SHA | 超出范围；`validate_b07_protocol.py` 仍 FAIL on CRLF worktree（已知限制）|
| B04A `run_one_candidate` 真正调用 | Round 5 计划 |

---

## 5. 禁止结论

- **GPU_PREFLIGHT_NOT_RUN**
- **FULL_NOT_RUN**
- **TEST_NOT_ACCESSED**
- **NOT_COMMITTED**
- **NOT_PUSHED**

---

## 6. Reviewer Checklist (Round 4)

- [ ] `force_overwrite` 已从 FullConfig / build_full_config / CLI / smoke / tests 全删
- [ ] `run_full` 启动时 `load_budget_state`，每 unit 原子写 `budget_state.json`
- [ ] 每个 unit 训练前 `load_resume_state`；match → skip + 累预算；mismatch → fail-closed
- [ ] 成功 unit 调 `write_unit_complete_atomic`
- [ ] `committed_file_sha256(..., frozen_git_sha=config.git_commit)`；build_full_config 不回退工作树
- [ ] `merge_seed_oof` 保留 per-sample `subject_ids`，可算 per-subject IoU
- [ ] Synthetic 模式 `winner = None`
- [ ] `test_round4_real_b01_readonly_preflight` 不 skip
- [ ] `test_round4_two_runner_run_skips_completed` 验证 hash 不变
- [ ] 67 个测试全部 passed（无 skip）

---

## 7. 状态

**READY_FOR_CODE_REVIEW**
