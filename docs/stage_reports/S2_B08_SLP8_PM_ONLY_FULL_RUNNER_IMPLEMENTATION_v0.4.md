# S2_B08 — SLP8 PM-only Full Runner 实现 (Round 6 收口)

**任务**: TASK-SLP-B08-FULL-RUNNER-AND-ONE-FOLD-PREFLIGHT-v0.1
**分支**: `codex/task-slp-b08-full-runner-v0.1`
**HEAD**: `1009366ccaad17855ae8da1c3d4df8bfdbe891d2`（未变）
**状态**: RUNNER_IMPLEMENTATION_COMPLETE / READY_FOR_CODE_REVIEW / GPU_PREFLIGHT_NOT_RUN

## 0. Round 6 Reviewer 收口

- `--one-fold-preflight` 现为真实单 unit 训练路径：必须指定 B01 freeze、dataset、
  Owner EXP-ID、candidate、fold、seed 与 `--run-authorized`；强制 clean Git SHA、
  B07 frozen 30 epochs，并完全绕过 30-unit `run_full()`。
- 单折训练记录 wall time、peak CUDA、budget verdict；`best.pt` 内嵌
  `experiment_id/git/config/data/split/model` identity。独立 reload 使用冻结
  VAL 顺序计算 prediction hash，失败写 `FAILED.json`，通过写 `DONE.json` 与
  `preflight_manifest.json`。
- Full unit 的最终 OOF 改为 minimum-val-loss `best.pt` 独立 reload 结果；fresh
  run 同时与 in-process best-epoch predictions 比较，mismatch fail closed。
- real OOF 每 epoch 清空，只保留一次 VAL 覆盖；merge 保留逐行 subject_id，
  实际计算 per-subject fixed foreground macro IoU 和 worst-subject IoU。
- `INTERRUPTED` 是唯一不封 terminal 的可恢复测试状态；普通 FAILED/STOPPED
  必须写唯一 terminal JSON。
- 新增 Round 6 定向测试：bounded one-fold preflight、real subject metrics、
  FAILED/STOPPED terminal；连同 resume 定向子集共 6 passed。

---

## 1. Round 4 ITERATE 后 Round 5 修复

| # | Codex Round 4 反馈 | Round 5 处置 |
|---|------|------|
| 1 | 真实 B01 fold 路由：`load_real_b01_fold()` 在 fold_1 报 TRAIN/VAL subject overlap | 删除 `partition_records_for_fold` 中"基于原始 train_subjects & val_subjects" 的错误 overlap 判定；B07 frozen `val_subject_ids` 是唯一权威；同一 subject 的不同 development row 按 `val_subject_ids` 路由；TRAIN/VAL 路由完成后只验证 subject 集无交集、sample_id 无重复 |
| 2 | real preflight 测试手工 `train_rows = [...]` 不调生产函数 | `test_round4_real_b01_readonly_preflight` 改写为直接调 `load_real_b01_fold(b01_dir, data_root=None, ...)` 覆盖 5 fold，验证 routing/VAL subjects/counts/no-TEST；B01 evidence 不可用时 **fail-closed** 不 skip |
| 3 | CLI 仍传 `force_overwrite=force` | `run_synthetic_cpu_smoke` 签名删除 `force` 参数；删 `--force` argparse；新增 `test_round5_cli_synthetic_cpu_smoke_subprocess` 通过 subprocess 调真实 CLI 验证 |
| 4 | 真正 partial checkpoint resume | 新增 `load_checkpoint_for_resume` (torch.load + identity verify + fail-closed)；`_save_checkpoint` 保存 model/optimizer/epoch/best/early-stopper/RNG；`train_one_unit` 启动时若 last.pt 存在则恢复并从 next epoch 继续；新增 `test_round5_partial_checkpoint_resume_continues_from_next_epoch` 真实中断 + resume 验证 |
| 5 | 复用 B04A 训练原语 | B04A 用 `MiniConfig`，B08 用 `FullConfig`；Round 5 不做语义漂移提取；本报告记录为 Round 6 任务 |
| 6 | Git provenance fail-closed | `build_full_config` 用 `committed_file_sha256(rr, rel, frozen_git_sha=git_commit)`；找不到 SHA 时 raise `FullConfigValidationError`，不回退工作树（Round 4 已就绪）|
| 7 | 独立验证命令 | 全部执行（见 §3） |

---

## 2. Real B01 fold counts（5-fold 真实路由）

```
fold_1: train_n=3240 val_n=855  train_subj=72 val_subj=19 overlap=0
fold_2: train_n=3285 val_n=810  train_subj=73 val_subj=18 overlap=0
fold_3: train_n=3285 val_n=810  train_subj=73 val_subj=18 overlap=0
fold_4: train_n=3285 val_n=810  train_subj=73 val_subj=18 overlap=0
fold_5: train_n=3285 val_n=810  train_subj=73 val_subj=18 overlap=0
```

每折 counts 精确匹配 B07 frozen contract。

---

## 3. 独立验证

| 命令 | 结果 |
|------|------|
| `python -m pytest tests/test_slp8_region_full.py -q` | **74 passed** |
| `python -m py_compile src/topper_perception/neural/slp8_region_full.py` | EXIT:0 |
| `python -m py_compile scripts/run_slp8_region_full.py` | EXIT:0 |
| `python -m py_compile scripts/smoke_b08_full_runner.py` | EXIT:0 |
| `python scripts/smoke_b08_full_runner.py` | 30/30 DONE，winner=null（synthetic）|
| `git diff --check` | EXIT:0 |
| B04/B04A core + models + links | **347 passed** |
| `python scripts/validate_b07_protocol.py ...` | Windows CRLF worktree-byte SHA mismatch；B08 committed-blob SHA tests PASS；validator 文件未越界修改 |

---

## 4. Interruption/Resume 证据

`test_round5_partial_checkpoint_resume_continues_from_next_epoch`：
1. 第一次 run 在 epoch boundary 写 `last.pt` 后注入 `INTERRUPTED`；不写
   FAILED/STOPPED terminal，保留可恢复证据。
2. 第二次保持相同 identity/max_epochs，读取 `last.pt` 并从 next epoch 恢复。
3. 普通 FAILED/STOPPED 由独立参数化测试验证必须写唯一 terminal。

---

## 5. B04A primitive reuse 证据

B04A 训练原语（`run_one_candidate`）位于 `slp8_region_mini.py`，使用 `MiniConfig`。
B08 runner 使用 `FullConfig`，配置语义不同（30 units × shared config vs 1 candidate × B04A config）。

Round 5 限制：未提取共享训练原语，避免语义漂移。
Round 6 计划：抽出公共 `train_one_epoch + validate + collect_oof` 函数供 B04A / B08 同时调用。

---

## 6. 禁止结论

- **GPU_PREFLIGHT_NOT_RUN**
- **FULL_NOT_RUN**
- **TEST_NOT_ACCESSED**
- **NOT_COMMITTED**
- **NOT_PUSHED**

---

## 7. Reviewer Checklist (Round 5)

- [ ] `partition_records_for_fold` 不再 reject 同 subject 跨 split；B07 val_subject_ids 权威
- [ ] `test_round4_real_b01_readonly_preflight` 调 `load_real_b01_fold` 5 fold，B01 缺失时 fail-closed
- [ ] CLI `--force` 全删；subprocess smoke 测试覆盖
- [ ] `_save_checkpoint` 保存 model/optimizer/epoch/best/early-stopper/RNG
- [ ] `load_checkpoint_for_resume` 验证 identity 后恢复；mismatch fail-closed
- [ ] `train_one_unit` 从 last.pt 恢复并从 next epoch 继续
- [ ] 真实两阶段 interruption/resume 测试通过
- [ ] 5 fold real B01 counts 精确匹配 B07

---

## 8. 状态

**RUNNER_IMPLEMENTATION_COMPLETE / READY_FOR_CODE_REVIEW / READY_FOR_OWNER_PREFLIGHT_AUTHORIZATION**

真实 RTX 4090 one-fold、30-unit Full、TEST 均未运行；只有代码和 CPU/synthetic
验证，不构成任何模型结果或候选排名。
