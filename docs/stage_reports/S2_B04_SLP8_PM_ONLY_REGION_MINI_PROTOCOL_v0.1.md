# Stage Report: S2-B04 — SLP8 PM-only Region Mini 协议与 Runner (R03)

**TASK-ID:** `TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-AND-RUNNER-v0.1`
**Stage:** S2-B04
**日期:** 2026-08-27 (R03)
**状态:** `MINI_PROTOCOL_AND_RUNNER_READY_FOR_REVIEW` — 协议、Runner、模型、指标、测试、中文文档已就绪；**未执行真实 Mini**，**未读取 TEST**，**未进入 B07**
**Base:** `main @ 6e19374`
**Branch:** `codex/task-slp-b04-pm-only-region-mini-protocol-v0.1`
**前序 commit:** `7f210fb` (R02)
**本 commit:** `R03`

---

## 1. R03 摘要

R02 之后 Code Reviewer 再次提出多项硬化要求，R03 在 R02 基础上完成：

1. **CLI 真正接通 `result.terminal_state`**：DONE/FAILED/STOPPED 三状态机在 CLI 主入口（`--validate-config` / `--synthetic-cpu-smoke` / `--run-authorized real-b01` 三条路径）实际接线；CLI 退出码 0 / 1 严格对应；`status.json` 与终端文件一致。
2. **真实 B01 输入合同接入 `_run_real_b01`**：构造 `B01FreezeSnapshot` → `verify_b01_contract`（fail-closed）→ `check_freeze_manifest_file_consistency`（fail-closed）；删除原 WARNING 分支；显式区分"freeze 结构上有 TEST 495"与"我们的 loader 永不读取 TEST 495"，构造 snapshot 时 test_rows 强制 None。
3. **可操作的 Resume**：CLI 暴露 `--resume-from <output_dir>`；自动检测 `checkpoints/<candidate>/last.pt`；拒绝 DONE；保存/恢复 early-stopper 当前 `current_patience`；保存/恢复历史、best epoch、best metric、checkpoint、optimizer、RNG、**预算累计状态**（`BudgetAccumulatorState`）；config SHA 必须使用实际 config 文件 SHA（拒绝空 identity）；所有 identity 不一致 fail-closed。
4. **CUDA 确定性 fail-closed**：`torch.use_deterministic_algorithms(warn_only=False)` 在 CUDA 可用时；`CUBLAS_WORKSPACE_CONFIG=":4096:8"` 与 `CUBLASLT_WORKSPACE_CONFIG` 在 CUDA 初始化前导出；当前实验环境为 CPU，记录 `run_mode="cpu_synthetic_reproducible"`；CUDA determinism 仍 **NOT RUN**。
5. **流程文档修正**：删除"B04 Real Mini 在 B07 协议下执行"等措辞。正确流程是 B04 Runner Reviewer → Owner 授权 B04 Real Mini → Experiment Runner 执行 → Codex 审核 → 通过后解锁 B07。

**关键禁止结论（仍生效）：**

* ❌ 不宣称 Mini 已完成
* ❌ 不宣称 B04 任何候选"优于 B02"
* ❌ 不宣称候选已保留 / 排除
* ❌ 不宣告 B07 解锁
* ❌ 不读 TEST
* ❌ 不跑真实 GPU Mini

---

## 2. 正确流程（R03 明确）

```
B04 Runner Reviewer 验收
  → Owner 单独授权 B04 Real Mini
  → Experiment Runner 执行 B04 Mini（带 --run-authorized + 真实 B01 路径）
  → Codex 审核真实结果
  → 通过后才解锁 B07
```

**B04 Real Mini 不再"属于 B07 协议"；它由 Owner 显式授权 + Experiment Runner 执行 + Codex 审核完成。**

---

## 3. 完整文件清单

### 新增文件
- `src/topper_perception/neural/slp8_region_b01_contract.py` (R02) — `B01FreezeSnapshot` + `verify_b01_contract` + `check_freeze_manifest_file_consistency`
- `src/topper_perception/neural/slp8_region_budget.py` (R02 + R03 强化) — `BudgetAccumulatorState` 新增；`ResourceBudgetState.snapshot()/restore()` 新增
- `src/topper_perception/neural/slp8_region_determinism.py` (R02 + R03 强化) — `_export_cublas_workspace_configs()`、`warn_only=not cuda_available()`、`run_mode` 字段
- `src/topper_perception/neural/slp8_region_resume.py` (R02 + R03 强化) — `EarlyStopperState.current_patience` 新增；resume 校验 + DONE 拒绝
- `src/topper_perception/neural/slp8_region_mini.py` — B04 Mini 核心 runner（R01-R03 累计）
- `configs/experiments/slp8_pm_region_mini_v0.1.json` — 冻结配置
- `scripts/run_slp8_region_mini.py` — CLI Runner（R01-R03 累计）
- `tests/test_slp8_region_mini.py` (R03 强化) — 147 个 B04 测试
- `docs/stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md`
- `docs/deliverables/SLP/B04_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md`

### 修改文件（R03）
- `src/topper_perception/neural/slp8_region_mini.py`:
  - `_build_checkpoint_payload` 新增 `budget_state: BudgetAccumulatorState | None`
  - `run_one_candidate` 在 resume 路径调用 `budget_state.restore(...)`
  - `run_one_candidate` 在保存 checkpoint 前调用 `budget_state.snapshot()`
  - `run_mini` 用真实 `config.config_path` 计算 `CheckpointIdentity.config_sha256`；若 config_path 为空则 `MiniProtocolError`（拒绝空 identity）
- `src/topper_perception/neural/slp8_region_resume.py`:
  - `EarlyStopperState` 新增 `current_patience: int`
  - `snapshot()` / `restore()` 包含 current_patience
  - `restore()` 校验 `0 <= current_patience <= patience`，越界 fail-closed
- `src/topper_perception/neural/slp8_region_budget.py`:
  - 新增 `BudgetAccumulatorState` dataclass
  - `ResourceBudgetState.snapshot()/restore()`
- `src/topper_perception/neural/slp8_region_determinism.py`:
  - 新增 `_export_cublas_workspace_configs()`
  - `apply_settings` 在 CUDA 可用时 `warn_only=False`
  - `DeterminismSettings.run_mode` 字段（`"cpu_synthetic_reproducible"` / `"cuda_determinism_unverified"`）
  - `environment_payload()` 包含 `CUBLAS_WORKSPACE_CONFIG`、`CUBLASLT_WORKSPACE_CONFIG`、`run_mode`
- `scripts/run_slp8_region_mini.py`:
  - `--resume-from` CLI flag
  - `_run_synthetic_cpu_smoke` 与 `_run_real_b01` 读取 `result.terminal_state`，写入 `DONE.json` / `FAILED.json` / `STOPPED.json` 互斥
  - CLI 退出码：DONE → 0；FAILED / STOPPED → 1；碰撞 / 授权缺失 → 2
  - `_run_real_b01` 移除 WARNING 分支，构造 `B01FreezeSnapshot` + `verify_b01_contract` + `check_freeze_manifest_file_consistency`，失败 → 抛 `B01ContractError`（整体 FAILED 路径或 CLI exit 1）
  - `_auto_detect_resume_candidates()` 解析 `--resume-from` 路径下的 `checkpoints/<candidate>/last.pt`
  - 测试专用 `B04_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS` env var 触发 STOPPED（仅供 CLI 集成测试）
  - 删除 `verify_subject_isolation` 副本（已 `from ... import`）
- `tests/test_slp8_region_mini.py` (R03 强化):
  - 新增 `TestCLITerminalStateDone` / `TestCLITerminalStateStopped` / `TestCLITerminalStateFailed`（真实 CLI 子进程）
  - 新增 `TestB01ContractEntryLevel`（5 个 fake-freeze 负向测试）
  - 新增 `TestResumeEquivalence`（interrupted+resume vs uninterrupted 完全一致）
  - 新增 `TestDeterminismConfigR03`（CUBLAS workspace、CUDA fail-closed、run_mode）
  - 新增 `TestDeterminismSubprocessR03`（两独立 subprocess 跑 byte-identical `predictions_manifest.csv` + `centroid_errors.csv`）
  - 修复 `test_early_stopper_state_round_trip` 与新加 `current_patience` 字段一致
  - 删除 PytestRemovedIn10Warning（旧的 class-scoped `test_only_tiny_budget` 调用 → 改用 inlined `ResourceBudget` 构造）

---

## 4. 数据合同（与 R02 一致）

| 字段 | 值 |
|---|---|
| TRAIN samples / subjects | 3,645 / 81 |
| VAL samples / subjects | 450 / 10 |
| TEST samples / subjects | **0 / 0**（loader 永不读取 TEST rows） |
| Pressure | float64 → float32，raw passthrough |
| raw_semantics | `raw_pmarray_response`（NOT kPa） |
| Provenance | `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED` |
| source_review_status | `NOT_REVIEWED` |
| A06 split SHA | `024f5abe...` |
| B01 input contract | counts/subjects/A06/provenance/setting/cover/freeze-manifest-SHA — 任一不一致 → `B01ContractError`（fail-closed） |

---

## 5. 资源预算（R03 强化执行）

| 阈值 | 值 |
|---|---|
| `max_wall_minutes_per_candidate` | 45 |
| `max_total_wall_minutes` | 90 |
| `max_peak_cuda_mb` | 12,288 |

**执行机制**：
- `BudgetAccumulatorState` 在每个 checkpoint 中保存 `candidate_seconds_consumed` 与 `last_candidate_peak_cuda_mb`。
- Resume 路径 `budget_state.restore(saved_state)` 恢复累计预算，避免双计时间。
- `ResourceBudgetState.begin_candidate()` 在每个候选开始时启动 `time.monotonic()`。
- `ResourceBudgetState.check()` 在每个验证 epoch 完成后调用；超预算 → 候选 `STOPPED`。
- 真实 Mini 启动时（CUDA 不可用）立即 fail-closed。

---

## 6. 状态机（R03 强化）

| 条件 | 终端状态 | 写入文件 | CLI exit |
|---|---|---|---|
| 全部候选正常完成 | DONE | `DONE.json` | 0 |
| 任一候选 FAILED | FAILED | `FAILED.json` | 1 |
| 任一候选 STOPPED（无 FAILED） | STOPPED | `STOPPED.json` | 1 |
| 输出目录碰撞 | n/a | 不写任何文件 | 2 |
| 无 `--run-authorized` 但传了 B01 路径 | n/a | 不创建 output 目录 | 2 |
| 真实 B01 输入合同 fail | n/a | `FAILED.json` | 1 |

三个终端文件 **互斥**（写一个自动删另两个）。

`status.json` 与终端文件 **强一致**：DONE 写 `status="DONE"`，FAILED 写 `status="FAILED"`，STOPPED 写 `status="STOPPED"`。

---

## 7. Resume 契约（R03 完整化）

* **CLI 暴露** `--resume-from <output_dir>`：自动检测 `checkpoints/<candidate>/last.pt`。
* **拒绝 resume** DONE 状态（`refuse_resume_for_done_run` 抛 `ResumeRefusedError`）。
* **保存/恢复**：`model.state_dict` + `optimizer.state_dict` + `early_stopper.snapshot()`（含 `current_patience`）+ `train_loss_history` / `val_loss_history` / `epoch_metrics` + `input_manifest_hashes` + `rng_state`（Python/NumPy/torch/CUDA）+ `budget_state.snapshot()`（`BudgetAccumulatorState`）。
* **Config SHA**：`CheckpointIdentity.config_sha256 = file_sha256(Path(config.config_path))`；空 config_path 立即 `MiniProtocolError`。
* **Identity 校验**：`verify_resume_identity(saved, requested)` 逐字段比对；任一不一致 → `ResumeIdentityError`。
* **current_patience 校验**：恢复时 `0 <= current_patience <= patience`；越界 fail-closed。

**等价性测试** (`TestResumeEquivalence::test_interrupted_then_resume_equals_uninterrupted`)：跑 N=6 epochs 完整 run A；跑 K=2 epochs 部分 run B（保留 last.pt）；从 B 恢复跑完 N；断言 A 与恢复后的 C 在 `feasibility`, `best_epoch`, `best_val_loss`, `best_prediction_hash`, per-region 指标签名上完全一致。

---

## 8. CUDA 确定性（R03 fail-closed）

```python
# 任何 CUDA 操作前：
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUBLASLT_WORKSPACE_CONFIG"] = ":4096:8"

# apply_settings(seed=42, cpu_threads=1) 内部：
torch.use_deterministic_algorithms(True, warn_only=not cuda_available)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)
```

`run_mode` 字段：
- 当前实验环境（CPU） → `"cpu_synthetic_reproducible"`
- 若有 CUDA 设备 → `"cuda_determinism_unverified"`（NOT RUN，**未端到端验证**）

---

## 9. 测试覆盖（R03）

`tests/test_slp8_region_mini.py` — **147 个 B04 测试**，全部通过。

新增（28 个）：
- `TestCLITerminalStateDone`（1）：CLI 跑合成烟雾 → `DONE.json` + exit 0
- `TestCLITerminalStateStopped`（1）：CLI 通过 `B04_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS=0.001` env var 触发 `STOPPED.json` + exit 非零
- `TestCLITerminalStateFailed`（1）：CLI 通过 monkey-patched `run_mini` 返回 `terminal_state="FAILED"` → `FAILED.json` + exit 非零（**未**直接调用 `write_status_files` 冒充 CLI）
- `TestB01ContractEntryLevel`（5）：fake B01 freeze + 直接构造 snapshot
  - correct freeze passes
  - bad A06 SHA rejected fail-closed
  - bad provenance rejected fail-closed
  - bad setting/cover rejected fail-closed
  - train count mismatch rejected fail-closed
  - TEST rows stay unloaded（`_test_rows is None`，`freeze.test_rows` 抛 `TestLeakageError`）
- `TestResumeEquivalence`（3）：
  - interrupted K + resume N == uninterrupted N（predictions hash、metrics、best epoch 完全一致）
  - resume for DONE run refused
  - partial checkpoints detected by `_auto_detect_resume_candidates`
- `TestDeterminismConfigR03`（3）：CUBLAS workspace 在 `apply_settings` 之前/期间导出；`warn_only` 取决于 CUDA 可用性；`environment_payload` 记录 `run_mode`
- `TestDeterminismSubprocessR03`（2）：两独立 subprocess 跑 `--synthetic-cpu-smoke` 产生 byte-identical `predictions_manifest.csv` + `centroid_errors.csv`
- 修复 R02 的 `current_patience` 字段
- 删除 PytestRemovedIn10Warning

联合回归（不带真实数据）：**1313 passed, 4 skipped**（与本任务无关的 pre-existing skip）。

---

## 10. 实际命令

- `uv sync --frozen --group dev --extra neural`
- `pytest tests/test_slp8_region_mini.py` — **147 passed**
- `pytest tests/ --ignore=tests/test_slp_8region_pressure_dataset.py` — **1313 passed, 4 skipped**
- 手动 `python scripts/run_slp8_region_mini.py --config … --synthetic-cpu-smoke` — `terminal_state=DONE`，20+1 产物
- 手动 `B04_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS=0.001` — `terminal_state=STOPPED`
- `git diff --check` — clean

---

## 11. Git 信息

* **Base:** `main @ 6e19374`
* **Branch:** `codex/task-slp-b04-pm-only-region-mini-protocol-v0.1`
* **前序 commit:** `7f210fb`（R02）
* **Working tree status:** clean（R03 待 commit）

---

## 12. Reviewer Checklist

- [x] CLI `result.terminal_state` 真正接通：DONE → DONE.json+exit 0；FAILED → FAILED.json+exit 1；STOPPED → STOPPED.json+exit 1
- [x] `status.json` 与终端文件一致
- [x] Synthetic CPU smoke 跑 20+1 产物齐全
- [x] 真实 B01 输入合同 `_run_real_b01` fail-closed（fake-freeze 负向测试覆盖 5 种违反）
- [x] WARNING 分支删除
- [x] 真实 B01 loader 永不读 TEST rows（结构有 495，loader 拒绝）
- [x] `--resume-from` CLI 暴露；自动检测 `checkpoints/<candidate>/last.pt`
- [x] resume 拒 DONE（`refuse_resume_for_done_run`）
- [x] `current_patience` 保存/恢复 + 越界 fail-closed
- [x] `BudgetAccumulatorState` 保存/恢复
- [x] 真实 `config_sha256` 拒绝空 identity
- [x] interrupted K + resume N ≡ uninterrupted N（predictions hash、metrics、best epoch 完全一致）
- [x] `CUBLAS_WORKSPACE_CONFIG` 在 CUDA 初始化前导出
- [x] `warn_only` 取决于 CUDA 可用性（CUDA fail-closed）
- [x] `run_mode` 字段记录当前实验状态
- [x] 两独立 subprocess 跑 byte-identical `predictions_manifest.csv` + `centroid_errors.csv`
- [x] 中文文档流程修正：Runner Reviewer → Owner 授权 → Experiment Runner → Codex 审核 → 解锁 B07
- [x] 删除"B04 Real Mini 属于 B07 协议"等措辞
- [x] CUDA determinism 仍为 NOT RUN（明确声明）
- [x] 147 个 B04 定向测试 + 1313 个联合回归测试通过

---

**阶段交付版本：** v0.1-R03
**生成时间：** 2026-08-27
**维护者：** Mavis (MiniMax Code)

---

## R04 收口记录（2026-08-27）

本轮不运行真实 B01、GPU 或 TEST；目标是把 R03 留下的 freeze 合同与中断续跑路径收紧到可审计状态。

### 已验证

- B01 合同不再补默认值：TRAIN/VAL 实际行、`core.splits`、A06 SHA、唯一的 setting/cover/provenance/review-status，以及 B01 `core` 的规范化 SHA-256 都必须同时匹配；TEST 仅读取 `core.splits.test` 的结构计数（495 样本 / 11 subject），不加载 TEST 行。
- 冻结配置显式固化 B01 `core` SHA 和 TEST 结构计数；缺任一字段时配置验证拒绝。
- 合成数据不再使用 Python 进程随机化的 `hash()`；跨进程生成的 class-weight identity 因而稳定。
- 极小预算真实 CLI smoke：两个候选均进入 `STOPPED`，各自落盘 `last.pt` 与 `best.pt`。
- 在全新的输出目录执行 `--resume-from <STOPPED 输出>`：退出码 0、`DONE.json` 与 `status.json` 均为 `DONE`；两个候选完成且为 `NOT_FEASIBLE`。恢复时会复制并校验此前的 `best.pt`，避免“最佳 epoch 在中断前”时丢失证据。
- 本轮定向回归：`pytest -q tests/test_slp8_region_mini.py -k "B01Contract or CLITerminalStateStopped or DeterminismConfigR03"` → **31 passed, 127 deselected**；`py_compile` 通过；`git diff --check` 通过。
- CUDA 路径若无法启用确定性算法或设置 cuDNN 确定性，现抛错拒绝执行；CPU smoke 保留兼容性行为。

### 推断

- 无。R04 只证明协议、合同与合成 CLI 生命周期；不构成真实 Mini 的模型效果结论。

### 未验证

- 真实 B01 Mini、GPU/CUDA 12 GB 峰值、真实数据上的恢复，以及任何 TEST 读取：均未运行。

### 限制

- 当前设备无 CUDA；合成样本仅验证工程链路，不能同 B02 指标比较。
- 本阶段仍是 `MINI_PROTOCOL_AND_RUNNER_READY_FOR_REVIEW`，不代表 Mini 完成、候选保留或 B07 解锁。

### 下一 Gate

1. Codex/Reviewer 验收冻结合同和 R04 回归；
2. Owner 明确授权并提供满足 CUDA 12 GB 峰值的环境；
3. 才能运行真实 B01 Mini，随后由 Codex 审核真实产物；
4. 仅在该审核通过后，才讨论 B07 Full 协议。

### Reviewer 最终验收

- `pytest -q tests/test_slp8_region_mini.py`：**158 passed**（340.11 秒）。
- `pytest -q tests/ --ignore=tests/test_slp_8region_pressure_dataset.py`：**1342 passed, 4 skipped**（1249.08 秒）；4 个 skip 均为缺少真实 evidence/data 或 A05 CSV 的既有环境条件。
- `py_compile` 与 `git diff --check`：通过。
- 结论：`TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-AND-RUNNER-v0.1` 通过 Reviewer Gate；状态为 `PROTOCOL_AND_RUNNER_ACCEPTED`。
- 未验证边界不变：真实 B01 Mini、CUDA/GPU 与 TEST 均为 `NOT RUN`。

### Verified

- 协议、Runner、冻结合同、合成生命周期与软件回归已经 Reviewer 独立验证。

### Inferred

- 无模型效果推断。

### Unverified

- 真实 B01 Mini、CUDA/GPU、TEST 和真实数据恢复均未运行。

### Limitations

- 合成 smoke 只证明工程链路，不能与 B02 比较，也不能形成产品或硬件结论。

### Next Gate

- Owner 明确授权真实 B01 Mini 并提供 CUDA 12 GB peak 环境；真实结果经 Reviewer 接受前，B07 保持阻塞。
