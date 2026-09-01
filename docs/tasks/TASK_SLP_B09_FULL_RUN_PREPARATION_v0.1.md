# TASK-SLP-B09-FULL-RUN-PREPARATION-v0.1

状态：`ACCEPTED / CLI_BRIDGE_ACCEPTED_AT_8b3ebda / GPU_FULL_NOT_AUTHORIZED / TEST_DENIED`

> R02 修正：当前 B08 runner CLI 不暴露 30-unit real B01 入口（`main()` 显式拒绝
> "Real B01 run not executed by this task"），因此本任务最高只能达到
> `ITERATE / CLI_BRIDGE_REQUIRED`；下一任务必须是独立的
> `TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1`。CLI bridge 验收并 commit 前，Owner
> 不得授权 30-unit Full；本任务的 §14.1 / §14.2 已改为
> "DO NOT RUN — CLI BRIDGE NOT IMPLEMENTED" 的拟议命令模板，仅供 bridge 任务
> 实现时参考。
>
> R03 修正：Codex R02 review 构造了三个直接反例（删除 OOF/TEST carrier；
> 篡改 DONE.json identity；peak CUDA 99999 with budget_ok=true）均错误地
> 返回 0 errors。本轮 fail-open → fail-closed 修复：
>
> - 完整审计 `DONE.json`（非空 JSON、experiment_id 匹配 CLI、git_commit 匹配
>   当前 HEAD、git_dirty 严格 False、4 个 frozen hash 与 DONE/manifest/status
>   /complete.json/input_manifest_hashes.json 一致）。
> - OOF evidence fail-closed：`oof_metrics_summary.json` 必须存在，2 candidates
>   × 3 seeds × 91 subjects/4095 samples 全部覆盖，不允许丢 unit，不允许
>   fold-average 当 primary；缺失 carrier 显式标记
>   `TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1` 附带 blocker。
> - TEST=0 evidence 语义正确：必须 4 个 frozen hash + `test_access == False`
>   + `test_rows/labels/onehot/predictions/metrics` 严格 int 0；缺一 ERR。
> - 完整 CUDA budget 检查：total / per-candidate / per-unit wall 与 peak
>   CUDA，全部 finite non-negative，不超 B07 上限，budget_ok 必须与重算一致。
> - A06 split 单源：`--a06-split-manifest` 参数**已删除**；A06 split SHA 现在
>   单一来源是 `--b01-freeze-manifest` 的 `core.a06_split_sha256`，避免要求一个
>   实际不存在的独立文件。
>
> R04 修正：Codex R03 review 构造五个直接反例（budget map 全空 / TEST=0
> evidence 只有 `{test_rows:0}` / validator fixture 与真实 runner schema
> 不一致 / A06 split 三方未绑定 / 任务书残留 `--a06-split-manifest` 与
> 旧 Gate 字样）仍被 validator 错误地返回 0 errors 或掩盖 bridge 缺失。
> 本轮 fail-open → fail-closed 修复：
>
> - **R04-#1 budget audit**：`per_unit_wall_seconds` 必须含精确 30 个 unit key
>   （2 candidates × 5 folds × 3 seeds），`per_candidate_wall_seconds` /
>   `peak_cuda_mb_per_candidate` 必须含精确 2 个 candidate key；任何 missing /
>   extra key ERR。从 30 个 `units/<uid>/complete.json` 重新计算
>   per-candidate wall、total wall、per-candidate peak CUDA。由于真实 writer
>   `build_budget_report()` 会把这些数值舍入到两位小数，浮点容差为
>   `RECOMPUTE_TOLERANCE = 0.005001`；与 `budget_report.json` 超出该 writer
>   序列化精度的任何不一致
>   即 ERR；`budget_ok=true` 不能替代重算。
> - **R04-#2 TEST=0 evidence**：`input_manifest_hashes.json` 必须同时含
>   `test_access` (strict bool False) + `test_rows` / `test_labels` /
>   `test_onehot` / `test_predictions` / `test_metrics` (strict int 0)；
>   任何缺失、字符串 "0"、bool 1、null、非零、负数、未知 `test_*` 字段均 ERR。
>   Runner 当前未写完整 6 字段 — 此项被明确登记为
>   `TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1` 的 artifact-schema blocker，
>   validator 在 audit-only 模式下用 `CLI_BRIDGE_ARTIFACT_SCHEMA_INCOMPLETE`
>   显式报错。
> - **R05 schema alignment correction**：Runner 在所有计划 unit 达到 terminal 后
>   会通过 `write_terminal_state()` 写入 `DONE.json`，并已写入 unit-level
>   `unit_oof.npz`、四个 budget limits 和每 unit `peak_cuda_mb`；这些不是 bridge
>   blocker。仍需 bridge 补齐的是：30-unit CLI 入口、`status.json` 的 frozen
>   hashes、`input_manifest_hashes.json` 的六个 TEST=0 carrier，以及（若保持 B09
>   audit 的 91-subject 断言）candidate seed carrier 的 `total_subjects`。现有
>   DONE.json 若缺完整 frozen identity，validator 也会显式
>   `CLI_BRIDGE_ARTIFACT_SCHEMA_INCOMPLETE`，绝不静默通过。
> - **R04-#4 A06 三方绑定**：A06 split SHA 必须三方一致 — B01 freeze
>   `core.a06_split_sha256`、B07 protocol `data_contract.a06_split_sha256`、
>   B07 fold `source_a06_split_sha256`；`core.a06_split_identifier` 必须
>   等于 `slp_subject_split_v0.1`；任一缺失 / 漂移 / 非 64 位小写 hex ERR。
> - **R04-#5 任务书清理**：彻底移除 `--a06-split-manifest` 命令行参数
>   引用（旧 `--a06-split-manifest` 与旧 A06 split 路径在所有执行命令 /
>   reviewer checklist 中已替换为"由 B01 freeze `core.a06_split_sha256`
>   单一来源供给"）；将当前 Gate 文本统一到
>   `ITERATE / CLI_BRIDGE_REQUIRED / GPU_FULL_NOT_AUTHORIZED / TEST_DENIED`，
>   `RUN_PREPARATION_READY_FOR_REVIEW` 仅保留在"未来 bridge 验收并 commit
>   后的升级条件"上下文，并明确它**不是当前 Gate**。
>
>
> - **完整审计 DONE.json**：必须解析为 JSON object、status ∈ {DONE, PREFLIGHT_PASSED}、
>   identity（experiment_id / git_commit / git_dirty / 4 frozen hashes）必须存在且
>   与 CLI 与 manifest.json / status.json / input_manifest_hashes.json 全部一致。
> - **OOF evidence 必填**：`oof_metrics_summary.json` + `candidates/<cand>/candidate_decision.json`
>   缺失即 ERR；每个 candidate × 每个 seed 的 `total_samples` 必须严格 = 4095；
>   `candidates` 块必须包含两个 frozen candidate 且 status=DONE。
> - **TEST=0 evidence 必填**：`input_manifest_hashes.json` 缺失即 ERR；4 个 frozen
>   identity hash 必须存在；仅接受 `test_access=false` / `test_rows=0` /
>   `test_labels=0` / `test_onehot=0` / `test_predictions=0` / `test_metrics=0` 六个
>   安全字段，true / 非零 / 字符串 / null 全部 ERR。
> - **完整 CUDA budget 检查**：逐 unit wall ≤ 900 s、逐 candidate wall ≤ 13500 s、
>   total wall ≤ 27000 s、逐 candidate peak CUDA ≤ 8192 MiB、逐 unit peak CUDA
>   ≤ 8192 MiB；所有数值必须为 finite non-negative number；`budget_ok` 必须与
>   重算结果一致。
> - **A06 split 单源**：`--a06-split-manifest` 参数**已删除**；A06 split SHA 现在
>   单一来源是 `--b01-freeze-manifest` 的 `core.a06_split_sha256`，避免要求一个
>   实际不存在的独立文件。
>
> 范围说明：本任务只交付"30-unit PM-only Full 的运行准备合同 + validator + 测试 + AutoDL
> 命令合同 + 资源估算 + 治理更新"，不得实际运行 30-unit Full，不得访问 TEST，不得 commit
> 或 push。最终 Git SHA 在 Codex Review 与正式 commit 后再冻结；本文件以
> `TO_BE_FROZEN_AFTER_CODE_REVIEW_AND_COMMIT` 占位。

## 0. 起点

- 分支：`codex/task-slp-b09-full-run-preparation-v0.1`
- 起点 HEAD：`8f0c2c3475f3669920871ccb4ca62af1baaab1d6`（B08 R03 接受 commit；与
  `codex/task-slp-b08-full-runner-v0.1` worktree 共享 commit）
- main / origin/main 相对 HEAD：HEAD 领先 1 commit（`0 1`）
- 工作区状态：clean，无 untracked
- 远端：`https://github.com/Xppp-yu/Smart-topper-workbench.git`
- 本机 GPU 状态：RTX 4060 Laptop GPU，本机 CPU only torch；本任务不运行任何 GPU 训练
- B08 前置 Gate：`ACCEPT / R03_PREFLIGHT_PASSED / TEST_DENIED`；30-unit Full 与 TEST 仍未授权
- B07 前置 Gate：`PROTOCOL_ACCEPTED / COMPUTE_NOT_RUN / TEST_DENIED`
- 当前在途训练任务：无

## 1. 目标

为 B09 的 30-unit PM-only Full 公平比较建立可独立复核的运行准备合同。所有内容必须满足：

1. 30-unit execution matrix（2 candidates × 5 folds × 3 seeds = 30）已在 B07 冻结；本任务只
   复述与机器可校验，不重写。
2. 提供一个 `scripts/validate_b09_full_run_preparation.py`，fail-closed 校验
   B07 协议 / fold manifest / TEST=0 / budget / identity / 输出覆盖 / terminal 互斥 /
   完成 EXP-ID 不可重跑 / `--run-authorized` 强制等。
3. 提供 `tests/test_b09_full_run_preparation.py`，覆盖 14 类异常/正路径场景；测试只
   使用临时目录或 synthetic fixtures，不得写真实 outputs，不得读 TEST。
4. 评估现有 B08 runner 能否承载 30-unit Full，输出**最小 blocker 列表 + 最小复现
   测试**；不擅自重构 runner，不扩大本 TASK。
5. 提供唯一 AutoDL 命令合同（**不执行**），含上传、SHA256 校验、clean checkout、
   环境检查、validate-only、Owner 授权启动、恢复、监控、artifact 审计、evidence
   打包、scp 下载。
6. 基于 B08 R03 实测给出 30-unit Full 的资源估算（**仅 estimate**），保留 B07
   正式上限 450 min total / 15 min per unit / 8192 MiB peak CUDA。
7. 同步更新 `docs/PROJECT_STATUS.md` 与 `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`
   状态口径到 `ITERATE / CLI_BRIDGE_REQUIRED / GPU_FULL_NOT_AUTHORIZED /
   TEST_DENIED`；不得写 `ACCEPT / FULL_AUTHORIZED / FULL_RUNNING /
   FULL_COMPLETE / TEST_AUTHORIZED / RUN_PREPARATION_READY_FOR_REVIEW`（后者
   仅在 bridge 验收并 commit 之后才能启用，不属于当前 Gate）。

## 2. 非目标（In-Scope / Out-of-Scope）

明确不做：

- 不运行 30-unit Full、单 unit 真实训练、CUDA/GPU 训练。
- 不访问 TEST，不调用 `enable_test_access`，不设 `load_test=True`，不读 TEST
  predictions / labels / onehot / statistics。
- 不修改原始数据，不写入 `E:\TeamProjects\datasets\smart-topper`。
- 不提交 archive、checkpoint、OOF、数据文件。
- 不覆盖 B08 R01/R02/R03 evidence。
- 不 commit，不 push，不 merge，不修改 main。
- 不使用 `git add .` 或 `git add -A`。
- 不为了"看起来有代码量"而重构 B08 runner。
- 不擅自放宽 B07 budget、不擅自修改 B07 protocol / fold manifest。
- 不写新的 Mini/Mid protocol，不重开 B04A 候选。
- 不把 B08 任何 R01/R02/R03 失败证据改写或重新标记。

## 3. 允许修改的文件（Files allowed to change）

> 本节为 fail-closed 文件白名单；任何越界必须先回到本任务做一次更新。

新增：

- `docs/tasks/TASK_SLP_B09_FULL_RUN_PREPARATION_v0.1.md`（本文件）
- `scripts/validate_b09_full_run_preparation.py`
- `tests/test_b09_full_run_preparation.py`

窄范围修改：

- `docs/PROJECT_STATUS.md`：只更新 S2_B09 一行 + 当前一句话状态；保留所有历史数字。
- `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`：只更新 `TASK-SLP-B09` 段落与必要总览。

禁止修改（与本任务无关，但显式声明以避免误改）：

- `configs/experiments/slp8_pm_full_protocol_v0.1.json`
- `configs/experiments/slp8_pm_full_folds_v0.1.json`
- `src/topper_perception/neural/slp8_region_full.py`
- `scripts/run_slp8_region_full.py`
- `scripts/validate_b07_protocol.py`
- `tests/test_b07_protocol_validator.py`
- `tests/test_slp8_region_full.py`
- `docs/stage_reports/S2_B08_SLP8_PM_ONLY_ONE_FOLD_PREFLIGHT_RESULT_v0.1.md`
- `docs/tasks/TASK_SLP_B07_FULL_PROTOCOL_FREEZE_v0.1.md`
- `docs/tasks/TASK_SLP_B08_ONE_FOLD_PREFLIGHT_RUN_v0.1.md`
- `docs/SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md`
- `docs/EXPERIMENT_GOVERNANCE_AND_GPU_EXECUTION_PLAN_v0.1.md`
- `docs/COLLABORATION_WORKFLOW.md` / `AGENTS.md`（如有更新需求应另立 TASK）

## 4. 冻结的 30-unit Execution Matrix

> 直接复述 B07 协议 `configs/experiments/slp8_pm_full_protocol_v0.1.json` 与 fold
> manifest `configs/experiments/slp8_pm_full_folds_v0.1.json`；本任务**不重写、不调
> 参、不扩展**。validator 通过 committed-content byte SHA 校验后才放行。

- candidates（**仅 2 个**，按 B07 顺序冻结）：
  - `slp8_deeplabv3plus_lite_v0.1`（model_version 相同；`exact_parameter_count = 53449`）
  - `slp8_resunet_lite_v0.1`（model_version 相同；`exact_parameter_count = 120809`）
- folds（**5**，按 B07 fold manifest 顺序冻结）：
  - `fold_1`（19 VAL subjects / 855 VAL samples；TRAIN 72 / 3240）
  - `fold_2`（18 / 810；TRAIN 73 / 3285）
  - `fold_3`（18 / 810；TRAIN 73 / 3285）
  - `fold_4`（18 / 810；TRAIN 73 / 3285）
  - `fold_5`（18 / 810；TRAIN 73 / 3285）
- seeds（**3**）：`42`、`123`、`2026`
- total units：精确 `2 × 5 × 3 = 30`（不允许出现 30 之外的任何 unit）
- 每 unit `max_epochs = 30`（与 B07 训练合同一致；不得缩短或扩展）
- 候选顺序敏感：必须保持 `slp8_deeplabv3plus_lite_v0.1` 在前、
  `slp8_resunet_lite_v0.1` 在后（与 B07 manifest 顺序一致）

## 5. proposed EXP-ID 规则

> 30-unit Full 作为单一 30-unit 实验；不按 unit 拆 EXP-ID。每 unit 的
> identity 走 `complete.json` + `checkpoint` payload 内的 `model_version /
> candidate / fold_id / seed` 字段。

proposed 模板：

```text
EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-RNN
```

- `RNN` 从 `R01` 起；B09T 一次性 TEST 用独立 `EXP-SLP-B09T-PM-FULL-TEST-<YYYYMMDD>-AUTODL-RNN`，
  不复用 B09 EXP-ID。
- 任何重跑必须使用新 `RNN`；`QUEUED` 后 `EXP-ID`、SHA、config、manifest、split 全部
  不可变（与 COLLABORATION_WORKFLOW §5 一致）。
- EXP-ID 必须作为 `--experiment-id` CLI 显式传入；runner 不接受任何隐式或合成
  sentinel。

## 6. Git SHA / config / data / fold / split hash 冻结规则

每个 unit 的 `complete.json`、checkpoint payload 与顶层 manifest 必须内嵌以下
identity（与 B07 `identity_contract.required_fields` 一致）：

| 字段 | 来源 | 何时冻结 |
|---|---|---|
| `experiment_id` | Owner 提供的 EXP-ID | Owner 授权前冻结 |
| `git_commit` | 当前 clean worktree 的 HEAD | `git rev-parse HEAD` 在启动前一刻 |
| `git_dirty` | `git status --porcelain` 是否空 | 必须 `false` |
| `config_sha256` | B07 协议 committed-content SHA | 已冻结（见 §11.1） |
| `data_manifest_sha256` | B01 freeze manifest SHA | 已冻结（见 §11.1） |
| `fold_manifest_sha256` | B07 fold manifest SHA | 已冻结（见 §11.1） |
| `split_sha256` | A06 split SHA | 已冻结（见 §11.1） |
| `model_version` | 与 candidate 同名 | 与 candidate 冻结同步 |
| `candidate` | `slp8_deeplabv3plus_lite_v0.1` 或 `slp8_resunet_lite_v0.1` | 与 B07 同步 |
| `fold_id` | `fold_1..fold_5` | 与 B07 fold manifest 同步 |
| `seed` | `42 / 123 / 2026` | 与 B07 同步 |

启动前 validator 必须重新计算并精确匹配以下 4 个 committed-input hash（不得伪造）：

| 资源 | committed-content SHA-256 |
|---|---|
| `configs/experiments/slp8_pm_full_protocol_v0.1.json` | `98314e70590094496418c0c8a43bb8b62497841a9b2437b9306f3d247e382c83` |
| `configs/experiments/slp8_pm_full_folds_v0.1.json` | `0ac344c9bb89cc71757c796096a8e2c63e8b4bb1cf9eeea2cab875fd2add8b2b` |
| `data/processed/slp8_training_tables_v0.1/freeze_manifest.json` | `42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04`（Windows 路径 `E:\TeamProjects\smarttopper-team-workbench\data\processed\slp8_training_tables_v0.1\freeze_manifest.json`） |
| A06 subject split manifest | `024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706` |

> 注：B08 R03 阶段报告 `docs/stage_reports/S2_B08_SLP8_PM_ONLY_ONE_FOLD_PREFLIGHT_RESULT_v0.1.md`
> 已在 R03 现场 re-verify 通过；本任务不重复检查，但 validator 必须独立重算以排除 CRLF
> / 行尾漂移。

本任务的最终 Git commit SHA（指承载 B09 准备变更的提交）目前尚未产生；任务书、Handoff
与 validator 输出中所有引用本任务 commit SHA 的位置必须使用占位符
`TO_BE_FROZEN_AFTER_CODE_REVIEW_AND_COMMIT`，并在 `Codex Review + 正式 commit +
clean checkout` 之后由下一次状态升级（`TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1`
验收并 commit 后的 handoff 升级）替换/冻结。
任何把占位符当真实 SHA 提交的行为按"伪造证据"处理。

## 7. TEST = 0 守卫（fail-closed）

- `protocol.test_access` 必须等于 `{allowed: false, load_test: false, expected_rows: 0,
  expected_labels: 0, expected_onehot: 0}`。
- `fold_manifest.test_access` 必须等于字符串 `"DENIED"`。
- `fold_manifest.invariants.test_subjects_in_any_fold` 必须等于 `0`。
- `fold_manifest.development_subject_count == 91` 且 `development_sample_count == 4095`。
- 任何对 `enable_test_access(purpose=...)` 的调用必须 fail-closed：仅
  `purpose="final_evaluation"` 可被接受，其他 purpose 抛 `TestLeakageError`；本任务
  validator 必须验证 30-unit Full runner 代码路径**不引用** `enable_test_access` /
  `load_test=True`。
- 任何 `partition_records_for_fold` 收到 `ml_split == "test"` 必须 fail-closed。
- 任何 `validate_oof_rows` 收到 `ml_split == "test"` 必须 fail-closed。
- OOF 写入路径必须不出现 `test/` 路径；本任务 validator 通过静态 import 解析 + 关键
  字符串模式扫描确认 `slp8_region_full` 路径上不出现 `enable_test_access` /
  `load_test=True` 的可达调用。

## 8. 输出目录不可覆盖

- `--output-dir` 在 `--run-authorized` 启动时由 runner 调
  `refuse_overwrite(output_dir)` 拒绝任何 pre-existing 产物；`manifest.json /
  status.json / DONE.json / FAILED.json / STOPPED.json` 任一存在即拒绝。
- 例外：**仅当** `output_dir` 内**仅有**一个 `STOPPED.json`、其 `experiment_id` 与
  本次 EXP-ID 一致、其 `git_commit` 与 `git_dirty=false` 与启动时一致时，可视为合法
  `INTERRUPTED` 状态，并允许从 `units/<unit_id>/complete.json` 恢复
  `DONE`-status 单元；其它单元仍按 `run_full` 顺序补跑。
- 任何 `DONE.json` 存在即拒绝；任何 `FAILED.json` 存在即拒绝；任何 `STOPPED.json` 的
  identity 与本次不匹配也拒绝。
- 任何 `--force` / `--overwrite` 标志在 B08 已**删除**；本任务 validator 复核 CLI 不
  接受任何覆盖逃生口。

## 9. terminal JSON 规则（DONE/FAILED/STOPPED 互斥）

- 单元粒度：每个 unit 在 `output_dir/units/<unit_id>/` 下写 `complete.json`
  （DONE）或 `unit_failed.json`（FAILED），互斥，且只有 DONE 的 unit 才有 `checkpoint
  / OOF`。
- 顶层 `output_dir` 下最终只能存在**正好一个** `DONE.json` / `FAILED.json` /
  `STOPPED.json`；多个则 validator 拒绝视为可恢复运行。
- `DONE.json` 出现条件：30 个 unit 全部 DONE，且 OOF 合并 `pooled_*` 指标已
  写入，`winner` 字段已写入。
- `FAILED.json` 出现条件：任何 unit 失败（budget 超限、NaN/Inf、subject leak、identity
  不一致、reload mismatch 等）。
- `STOPPED.json` 出现条件：人工/外部中断（runner 收到 SIGINT/资源不足自动停止）；仅
  `INTERRUPTED` 状态才允许 resume。

## 10. checkpoint / resume 规则

- 每个 unit 写 `units/<unit_id>/checkpoints/last.pt`（每 epoch 后）和
  `checkpoints/best.pt`（按 `val_loss` 最小）。
- checkpoint payload 内嵌 identity block：
  `{experiment_id, git_commit, git_dirty, config_sha256, data_manifest_sha256,
  fold_manifest_sha256, split_sha256, model_version, candidate, fold_id, seed}`。
- `load_checkpoint_for_resume(checkpoint_path, expected_identity)` 必须在 identity 任
  一字段不匹配时 fail-closed。
- 单元级 resume：存在 `units/<unit_id>/complete.json` 且 identity 完全匹配 → 跳过该
  unit，复用预算；否则视为未完成。
- 预算级 resume：`output_dir/budget_state.json` 持久化 `BudgetAccumulatorState`；恢复
  时 `load_budget_state` 不得重复累计已 DONE 单元的 wall。
- 任何 `units/<unit_id>/` 出现 `FAILED` / `STOPPED` 但无 `complete.json` 必须 fail-
  closed：拒绝 resume，要求 fresh output_dir。

## 11. failure / STOPPED / interruption 保留规则

- 任何 `FAILED.json` / `STOPPED.json` / `unit_failed.json` 一律保留，**禁止覆写**。
- 历史 EXP-ID（`EXP-SLP-B08-...-R01/R02/R03`）的 output 必须继续可读；新 EXP-ID 不得
  复用旧 EXP-ID 的 output 路径。
- validator 对任何"已存在 EXP-ID"必须校验其 terminal 与 30-unit 完成度：若 terminal
  已是 `DONE` 且 unit 数 = 30 → 拒绝重跑；若是 `FAILED/STOPPED` 且非 `INTERRUPTED` →
  拒绝覆盖。
- interruption 流程：`Ctrl-C` / `SIGTERM` 触发 `STOPPED.json` 写入；已写
  `complete.json` 的 unit 不重训；其余 unit 视为未完成。
- 资源超限：wall 超 15 min/unit 或 450 min total、peak CUDA > 8192 MiB → 立即
  `FAILED.json`，保留已 DONE 的 unit complete。

### 11.1 已冻结的 committed-input SHA 速查

| 资源 | committed-content SHA-256 |
|---|---|
| B07 protocol | `98314e70590094496418c0c8a43bb8b62497841a9b2437b9306f3d247e382c83` |
| B07 fold manifest | `0ac344c9bb89cc71757c796096a8e2c63e8b4bb1cf9eeea2cab875fd2add8b2b` |
| B01 freeze manifest | `42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04` |
| A06 split | `024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706` |

## 12. wall / CUDA / disk budget

| 项目 | 预算 | 来源 |
|---|---|---|
| 单 unit wall | ≤ 15 min | B07 `resource_budget.max_wall_minutes_per_fold_seed_unit` |
| 单 candidate wall | ≤ 225 min | B07 |
| 30-unit total wall | ≤ 450 min | B07 |
| 单 unit peak CUDA | ≤ 8192 MiB | B07 `max_peak_cuda_mb` |
| 单 unit best epoch | ≤ 30 | B07 `max_epochs` |
| 单 unit batch size | 16 | runner 默认，与 B08 R03 一致 |
| AutoDL 数据盘 | ≥ 50 GB 免费；按需扩到 200 GB | 治理文档 §9 |
| AutoDL 实例 | RTX 4090 24GB（与 B08 R03 相同） | 治理文档 §9.1 模板 A |
| 本地 evidence 备份盘 | ≥ 30 GB | B08 R01+R02+R03 压缩包 ≈ 数十 MB，预留余量 |

> B07 上限为 hard fail-closed gate；本任务的资源**估算**（见 §15）不得替换上限，
> 只能用作 sanity check。

## 13. AutoDL 环境预检（Owner 授权前必跑）

```bash
# 0. clean checkout（必须）：HEAD 与 §6 占位符替换后的 frozen SHA 一致
cd /root/workspace/smarttopper-team-workbench-b09
git status --porcelain
git rev-parse HEAD
test -z "$(git status --porcelain)" || { echo "DIRTY"; exit 1; }

# 1. Python / uv / torch / CUDA / RTX 4090 检查
uv --version
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; t=torch.tensor([1.0], device='cuda'); torch.use_deterministic_algorithms(True, warn_only=False); x=torch.randn(8,8, device='cuda', requires_grad=True); y=(x*2).sum(); y.backward(); print('STRICT_DETERMINISTIC_CUDA_OK')"

# 2. 输入存在 + hash 匹配（见 §11.1）
ls -la /root/autodl-tmp/data/processed/slp8_training_tables_v0.1/freeze_manifest.json
ls -la /root/autodl-tmp/datasets/SLP_8Region_Pressure_VAL_v1.1
sha256sum /root/autodl-tmp/data/processed/slp8_training_tables_v0.1/freeze_manifest.json

# 3. validate-only / no-write（不创建 output dir）
python scripts/run_slp8_region_full.py \
  --config configs/experiments/slp8_pm_full_protocol_v0.1.json \
  --output-dir /root/autodl-tmp/outputs/EXP-SLP-B09-30-UNIT-VALIDATE \
  --validate-only

# 4. 本任务准备 validator（不进入训练）
python scripts/validate_b09_full_run_preparation.py \
  --protocol configs/experiments/slp8_pm_full_protocol_v0.1.json \
  --fold-manifest configs/experiments/slp8_pm_full_folds_v0.1.json \
  --b01-freeze-manifest /root/autodl-tmp/data/processed/slp8_training_tables_v0.1/freeze_manifest.json \
  --output-dir /root/autodl-tmp/outputs/EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01 \
  --experiment-id EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01
```

## 14. 拟议命令模板（CURRENTLY BLOCKED — CLI BRIDGE NOT IMPLEMENTED）

> ⚠️ **DO NOT RUN — CLI BRIDGE NOT IMPLEMENTED** ⚠️
>
> 本节所有命令都是**拟议模板**，供下一个任务
> `TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1` 实现 30-unit real B01 CLI 入口时
> 参考；当前 B08 runner 的 `scripts/run_slp8_region_full.py` 在收到
> `--b01-freeze-dir` + `--run-authorized`（无 `--one-fold-preflight`）时会
> 显式打印 "Real B01 run not executed by this task" 并 `return 1`。
>
> **未在 CLI bridge 验收并提交前，Owner 不得授权 30-unit Full；这些命令当前
> 必然失败，CI 也不会以这些命令为基础。**
>
> 下一任务 (`TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1`) 的实现必须满足：
>
> 1. 新增 `--run-full` 标志；与 `--one-fold-preflight` 互斥。
> 2. 调用 `run_full()` 路径（已存在；可处理 30-unit real B01 `FullConfig`）。
> 3. 复用现有 `validate_b07_protocol` / `load_frozen_full_protocol` /
>    `refuse_overwrite` / git identity / output-dir 覆盖检查。
> 4. 强制要求显式 `--run-authorized`；无授权时拒绝并 `return 1`。
> 5. 强制要求 `--experiment-id` 匹配 `EXP-SLP-B09-PM-FULL-30-UNIT-\d{8}-AUTODL-R\d{2}`。
> 6. 禁止 `python -c "from topper_perception.neural.slp8_region_full import run_full; …"`
>    形式的旁路；CLI bridge 必须是 `scripts/run_slp8_region_full.py` 内的分支。
> 7. 不修改 B07 协议 / fold manifest / B01 freeze / A06 split。
> 8. 接受现有 review 流程后再正式 commit。

### 14.1 拟议正式启动命令模板（DO NOT RUN — CLI BRIDGE NOT IMPLEMENTED）

```text
# When the bridge is implemented, the following template applies.
# It is NOT executable today.
```

```bash
# === DO NOT RUN — CLI BRIDGE NOT IMPLEMENTED ===
# This template is the target contract for TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1.
# Running it now will print "Real B01 run not executed by this task" and exit non-zero.

cd /root/workspace/smarttopper-team-workbench-b09
mkdir -p /root/autodl-tmp/outputs

# sanity: ensure unique output_dir
test ! -e /root/autodl-tmp/outputs/EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01 \
  || { echo "OUTPUT_DIR_EXISTS"; exit 1; }

# freeze git identity at run time
GIT_HEAD=$(git rev-parse HEAD)
test -z "$(git status --porcelain)" || { echo "DIRTY_AT_START"; exit 1; }

# launch 30-unit real B01 Full via nohup
nohup python scripts/run_slp8_region_full.py \
  --config configs/experiments/slp8_pm_full_protocol_v0.1.json \
  --output-dir /root/autodl-tmp/outputs/EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01 \
  --b01-freeze-dir /root/autodl-tmp/data/processed/slp8_training_tables_v0.1 \
  --dataset-root /root/autodl-tmp/datasets/SLP_8Region_Pressure_VAL_v1.1 \
  --experiment-id EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01 \
  --run-full \
  --run-authorized \
  > /root/autodl-tmp/outputs/EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01.run.log 2>&1 &

RUN_PID=$!
echo "RUN_PID=$RUN_PID"
```

> 关键差异（相对 §14.1 R01 模板）：新增 `--run-full` 标志，禁止在没有
> `--run-authorized` 时静默通过。bridge 实现必须保留 `--one-fold-preflight`
> 路径不变。

### 14.2 拟议中断恢复命令模板（DO NOT RUN — CLI BRIDGE NOT IMPLEMENTED）

> 适用条件（bridge 实现后）：上次运行因人工 `Ctrl-C` / 平台回收留下**唯一**
> `STOPPED.json` 与 identity 完全匹配的 budget_state；其它 terminal 一律拒绝
> resume。STOPPED.json 必须携带完整 identity 块
> `{status: INTERRUPTED, experiment_id, git_commit, git_dirty: false,
> config_sha256, data_manifest_sha256, fold_manifest_sha256, split_sha256,
> model_version, candidate, fold_id, seed}`，且与启动时一致。

```bash
# === DO NOT RUN — CLI BRIDGE NOT IMPLEMENTED ===
# 1. 复核 INTERRUPTED 状态合法（validator 在 preparation 模式下要求：
#    - output_dir 仅有 STOPPED.json
#    - 完整 identity 块 + 与当前 HEAD 一致 + 与冻结 SHA 一致
#    - 任一已 DONE 的 unit complete.json identity 与顶层一致
#    - 不存在 unit_failed.json / 其它非 INTERRUPTED 终态
#    - 满足 B07 budget / candidates / folds / seeds / TEST=0）

python scripts/validate_b09_full_run_preparation.py \
  --protocol configs/experiments/slp8_pm_full_protocol_v0.1.json \
  --fold-manifest configs/experiments/slp8_pm_full_folds_v0.1.json \
  --b01-freeze-manifest /root/autodl-tmp/data/processed/slp8_training_tables_v0.1/freeze_manifest.json \
  --output-dir /root/autodl-tmp/outputs/EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01 \
  --experiment-id EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01

# 2. 重启（同 EXP-ID，runner 自动从 units/<unit_id>/complete.json 恢复已完成 unit）
# 启动命令与 14.1 模板完全一致；runner 在第二轮会跳过 DONE 单元、仅补未完成单元。
nohup python scripts/run_slp8_region_full.py \
  --config configs/experiments/slp8_pm_full_protocol_v0.1.json \
  --output-dir /root/autodl-tmp/outputs/EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01 \
  --b01-freeze-dir /root/autodl-tmp/data/processed/slp8_training_tables_v0.1 \
  --dataset-root /root/autodl-tmp/datasets/SLP_8Region_Pressure_VAL_v1.1 \
  --experiment-id EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01 \
  --run-full \
  --run-authorized \
  > /root/autodl-tmp/outputs/EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01.resume.log 2>&1 &
```

### 14.3 进程 / GPU / 日志 / 状态 监控

```bash
# 进程
ps -ef | grep run_slp8_region_full | grep -v grep

# GPU
watch -n 5 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

# 实时日志
tail -n 200 -F /root/autodl-tmp/outputs/EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01/logs/run.log

# status snapshot
cat /root/autodl-tmp/outputs/EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01/status.json
cat /root/autodl-tmp/outputs/EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01/budget_report.json

# unit-level progress
for u in /root/autodl-tmp/outputs/EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01/units/*/; do
  id=$(basename "$u")
  if [ -f "$u/complete.json" ]; then
    echo "DONE  $id"
  elif [ -f "$u/unit_failed.json" ]; then
    echo "FAIL  $id"
  else
    echo "PEND  $id"
  fi
done
```

### 14.4 完成后 artifact 审计（Owner 授权后执行；含 audit-only validator）

> 审计在 `EXP-SLP-B09-*` 终端为 `DONE.json` 后进行；本节命令基于真实 B08
> `write_run_artifacts` 写出的 schema（`manifest.json` / `status.json` /
> `input_manifest_hashes.json` / `budget_report.json` / `candidate_decision.json` /
> `oof_metrics_summary.json` / `units/<unit_id>/complete.json` / `units/<unit_id>/status.json` /
> `candidates/<cand>/candidate_decision.json` / `per_subject_metrics.csv` /
> `per_region_metrics.csv` / `per_posture_metrics.csv`），不含
> `oof_predictions.csv`（B08 实际写 `units/<unit_id>/oof/unit_oof.npz`）；如 bridge
> 任务扩展 schema，必须同步更新本节命令。

```bash
# === DO NOT RUN — only after Full has actually completed AND the bridge task is merged ===
OUT=/root/autodl-tmp/outputs/EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01
PROTOCOL=configs/experiments/slp8_pm_full_protocol_v0.1.json
FOLD=configs/experiments/slp8_pm_full_folds_v0.1.json
B01=/root/autodl-tmp/data/processed/slp8_training_tables_v0.1/freeze_manifest.json

# 1. validator --audit-only：read-only 全量检查
python scripts/validate_b09_full_run_preparation.py \
  --protocol "$PROTOCOL" \
  --fold-manifest "$FOLD" \
  --b01-freeze-manifest "$B01" \
  --output-dir "$OUT" \
  --experiment-id EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01 \
  --audit-only

# 2. 补充的 shell-only 抽检（验证关键 schema 字段确实存在并满足 B07 上限）
test -f "$OUT/DONE.json"  || { echo "MISSING DONE.json"; exit 1; }
test ! -f "$OUT/FAILED.json"
test ! -f "$OUT/STOPPED.json"
test "$(jq -r '.terminal_state'  "$OUT/manifest.json")" = "DONE"
test "$(jq -r '.unit_count_done'  "$OUT/manifest.json")" = 30
test "$(jq -r '.unit_count_failed'"$OUT/manifest.json")" = 0
test "$(jq -r '.unit_count_stopped'"$OUT/manifest.json")" = 0
test "$(jq -r '.git_dirty'        "$OUT/manifest.json")" = "false"
test "$(jq -r '.winner'           "$OUT/candidate_decision.json")" \
  = "slp8_deeplabv3plus_lite_v0.1" -o \
  "$(jq -r '.winner'           "$OUT/candidate_decision.json")" \
  = "slp8_resunet_lite_v0.1"
python - <<'PY'
import json, sys
m = json.load(open("$OUT/manifest.json"))
assert m["config_sha256"] == "98314e70590094496418c0c8a43bb8b62497841a9b2437b9306f3d247e382c83"
assert m["data_manifest_sha256"] == "42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04"
assert m["fold_manifest_sha256"] == "0ac344c9bb89cc71757c796096a8e2c63e8b4bb1cf9eeea2cab875fd2add8b2b"
assert m["split_sha256"] == "024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706"
b = json.load(open("$OUT/budget_report.json"))
assert b.get("budget_ok") is True
assert b["total_wall_seconds"] <= 15 * 5 * 3 * 2 * 60  # ≤ 450 min × 60 s
PY
```

### 14.5 evidence 打包 + SHA256（Owner 授权后执行）

```bash
OUT=/root/autodl-tmp/outputs/EXP-SLP-B09-PM-FULL-30-UNIT-<YYYYMMDD>-AUTODL-R01
TS=$(date -u +%Y%m%dT%H%M%SZ)
PKG=EXP-SLP-B09-PM-FULL-30-UNIT-EVIDENCE-${TS}.tar.gz
tar -czf "/root/autodl-tmp/$PKG" \
  --exclude='checkpoints/last.pt' \
  --exclude='*.pt' \
  -C "$(dirname "$OUT")" "$(basename "$OUT")"
sha256sum "/root/autodl-tmp/$PKG" | tee "/root/autodl-tmp/$PKG.sha256"
```

### 14.6 scp 下载到 Windows

```powershell
# DO NOT RUN WITHOUT OWNER GPU FULL AUTHORIZATION
scp root@<autodl-host>:/root/autodl-tmp/EXP-SLP-B09-PM-FULL-30-UNIT-EVIDENCE-<TS>.tar.gz `
  E:\TeamProjects\B09_EVIDENCE\
scp root@<autodl-host>:/root/autodl-tmp/EXP-SLP-B09-PM-FULL-30-UNIT-EVIDENCE-<TS>.tar.gz.sha256 `
  E:\TeamProjects\B09_EVIDENCE\
# 本地重算 SHA-256 复核
Get-FileHash E:\TeamProjects\B09_EVIDENCE\EXP-SLP-B09-PM-FULL-30-UNIT-EVIDENCE-<TS>.tar.gz -Algorithm SHA256
```

## 15. 资源估算（estimate，B07 上限不替换）

> 全部数据点均为 estimate；**不是 Full 实测**。正式上限保留 B07 §12。

锚点：B08 R03 单 unit 实测（`docs/stage_reports/S2_B08_SLP8_PM_ONLY_ONE_FOLD_PREFLIGHT_RESULT_v0.1.md`）

- wall：155.32885087199975 s ≈ 2.59 min（ResUNet-lite, fold_1, seed 42, best epoch 22, max_epochs 30）
- peak CUDA：368.764416 MiB
- best epoch：22（剩余 8 epoch 为 early-stop patience 区间）
- TRAIN：3,240 samples / 72 subjects；VAL：855 samples / 19 subjects

推算 30-unit Full（per-seed arithmetic mean × 2 candidates）：

| 项 | estimate | 说明 |
|---|---:|---|
| 单 unit mean wall | 2.59 min | 与 R03 同架构 / 同 fold-best-epoch 假设（仅供参考） |
| 单 unit 90th-percentile wall | ≤ 5 min | R03 用 30 epoch，best 22；early-stop 正常时 wall 显著低于上限 |
| 30-unit mean total wall | ≈ 78 min | 30 × 2.59（linear extrapolation） |
| 30-unit 90th-percentile total wall | ≤ 150 min | 30 × 5 |
| 30-unit worst-case total wall | ≤ 450 min | B07 hard upper bound |
| 单 unit mean peak CUDA | 369 MiB | 与 R03 相同；DeepLabV3+-lite 预计更低（53k vs 121k params） |
| 30-unit max peak CUDA | ≤ 8192 MiB | B07 hard upper bound |
| 单 unit best epoch range | 18–28 | B04A R03 历史 18–27 区间；本 Full 不可外推（仅 sanity） |
| 单 unit VAL samples | 855（fold_1）/ 810（其余） | 与 fold manifest 一致 |
| 单 unit OOF rows | 855 / 810 | 同上 |
| 30-unit OOF rows per seed | 4095 | 与 B07 一致 |
| 30-unit OOF rows per candidate | 4095 | per-candidate OOF；不重复 |

> 注：以上 mean/90th 假设全部 unit 跑满早停候选区间，不代表 Full 已运行。Full 实际
> wall 取决于 early-stop 行为、数据加载抖动和平台调度，最坏情况下达到 B07 上限 450
> min。

## 16. B08 Runner 30-unit 能力评估

经直接阅读 `scripts/run_slp8_region_full.py` 与
`src/topper_perception/neural/slp8_region_full.py`：

| 能力 | 已具备？ | 证据 |
|---|---|---|
| 30-unit plan（2×5×3） | ✅ | `build_execution_plan` / `test_execution_plan_no_duplicates` |
| 单元 TEST=0 守卫 | ✅ | `load_real_b01_fold` 强 assert `_test_rows is None` |
| checkpoint / resume | ✅ | `load_checkpoint_for_resume` + `last.pt`/`best.pt` + identity 校验 |
| unit 级别 identity 持久化 | ✅ | `units/<unit_id>/complete.json` 含 identity block |
| unit 级别 budget 累计 | ✅ | `BudgetAccumulatorState` 持久化到 `budget_state.json` |
| 30-unit OOF 合并（pooled） | ✅ | `merge_seed_oof` + `validate_oof_rows` |
| DONE/FAILED/STOPPED 互斥 | ✅ | `write_run_artifacts` 互斥写 terminal |
| `refuse_overwrite` 输出覆盖 | ✅ | `refuse_overwrite` 检 manifest/status/DONE/FAILED/STOPPED |
| `--run-authorized` 强制 | ✅（仅 one-fold preflight 路径） | `scripts/run_slp8_region_full.py:main()` |
| `run_full()` 30-unit real B01 入口 | ⚠️ | `run_full()` 函数本身可处理 30-unit，但 `main()` 对 30-unit real B01 路径**显式拒绝**（"Real B01 run not executed by this task"） |
| CLI 暴露 30-unit real B01 入口 | ❌ | 当前 `--one-fold-preflight` 仅 1 unit；不存在 `--run-full` 或类似标志 |
| 30-unit 输出 schema 包含 30 unique units / unit_count_done / winner / OOF coverage / `candidates/<cand>/candidate_decision.json` | ✅ | `write_run_artifacts` + `units/<unit_id>/status.json` + `units/<unit_id>/complete.json` |
| audit-only 模式（read-only 审计） | ✅（本任务 B09 validator 自身实现） | `scripts/validate_b09_full_run_preparation.py --audit-only` |

**关键 blocker（CLI bridge）**：

1. CLI 不暴露 30-unit real B01 入口：`scripts/run_slp8_region_full.py` 的 `main()`
   在收到 `--b01-freeze-dir` + `--run-authorized`（无 `--one-fold-preflight`）时打印
   "Real B01 run not executed by this task" 并 `return 1`。`run_full()` 函数本身可
   接受 30-unit real B01 `FullConfig`，但目前没有 CLI 路径调用它。

   - 最小复现：执行 `python scripts/run_slp8_region_full.py --config <B07>
     --output-dir <new> --b01-freeze-dir <...> --dataset-root <...>
     --experiment-id <EXP> --run-authorized` → 看到 "Real B01 run not executed by
     this task"，`return 1`。

2. 该 blocker 不在本任务范围内解决；本任务**只记录**，不在 B09 任务中擅自添加 CLI
   入口或重构 runner。**下一任务必须是独立的**
   `TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1`，由 Owner 单独签发。bridge 任务必
   须满足：

   - 新增 `--run-full` 标志；与 `--one-fold-preflight` 互斥。
   - 调用 `run_full()` 路径并复用现有 `validate_b07_protocol` /
     `load_frozen_full_protocol` / `refuse_overwrite` / git identity / EXP-ID 检
     查。
   - 强制要求显式 `--run-authorized`；无授权时拒绝并 `return 1`。
   - 强制要求 `--experiment-id` 匹配
     `EXP-SLP-B09-PM-FULL-30-UNIT-\d{8}-AUTODL-R\d{2}`。
   - 禁止 `python -c "from topper_perception.neural.slp8_region_full import run_full; …"`
     形式的旁路；CLI bridge 必须是 `scripts/run_slp8_region_full.py` 内的分支。
   - 不修改 B07 协议 / fold manifest / B01 freeze / A06 split。

3. validator 需静态复核 CLI 仍不暴露未授权的 30-unit real B01 入口；本任务
   validator 已实现此静态检查。

**当前 Gate 必须保持 `ITERATE` 的理由**：

- 本任务**准备**已具备（合同 / validator / 测试 / AutoDL 命令 / 资源估算 / 治理
  同步），但**当前 B08 runner 仍拒绝执行 30-unit real B01**。
- 30-unit Full 仍为 `GPU_FULL_NOT_AUTHORIZED` / `TEST_DENIED`。
- Bridge 任务之前不能使用 `RUN_PREPARATION_READY_FOR_REVIEW`（该状态暗示 Owner 可
  立即授权 Full，与 CLI 现实不符）。
- 这是治理 / 安全保留，不应在本任务内绕过。
- 当且仅当 `TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1` 验收并 commit 后，本任务
  的 handoff 才允许升级到 `RUN_PREPARATION_READY_FOR_REVIEW`。

## 17. Reviewer checklist

Codex 独立复核本任务时必须逐项勾选：

- [ ] `scripts/validate_b09_full_run_preparation.py` 是 fail-closed：committed-content
      SHA、30-unit 唯一性、fold 隔离、TEST=0、load_test=False、identity 必填、
      git_dirty==false、output-dir 状态（preparation 模式严格 fail-closed，仅允许
      不存在 / 唯一 STOPPED + 完整 identity 匹配）、terminal 互斥、--run-authorized
      守卫、自身不训练/不读 TEST/不创建 output-dir。
- [ ] `--audit-only` 模式在 B09 validator 内部真实实现（不是占位符）：要求
      output_dir 已存在、唯一 DONE.json、identity 一致、30 unique unit complete、
      budget_ok、wall/CUDA 不超 B07 上限、winner ∈ frozen candidates、TEST=0。
- [ ] B01 freeze manifest 必须通过 `--b01-freeze-manifest` 实际文件路径传入并独立
      重算 SHA256；A06 split 由 B01 freeze `core.a06_split_sha256` 单一来源供给，
      validator 在 B01 freeze 缺失或 SHA 不匹配时 fail-closed（不写
      "not found, will be re-verified"作为 OK）。
- [ ] 30-unit plan 与 B07 fold manifest SHA 完全一致；`committed-content SHA` 重算
      与 §11.1 严格匹配（CRLF-safe via `git show HEAD:...`）。
- [ ] `git_dirty_must_be == False` 在 validator 中已硬编码。
- [ ] `test_access` 全 0 + fold `test_access == "DENIED"` 已硬编码。
- [ ] `enable_test_access` / `load_test=True` 在 Full runner 代码路径上不可达（静
      态扫描 + 动态导入校验）。
- [ ] output_dir 覆盖检查覆盖 manifest/status/DONE/FAILED/STOPPED；STOPPED 仅在
      完整 identity + 当前 HEAD 一致 + 任一已 DONE unit complete identity 一致 +
      不存在 unit_failed.json 时才接受 resume。
- [ ] DONE/FAILED/STOPPED 互斥；多 terminal 即 fail-closed。
- [ ] `--run-authorized` 在 CLI 中是 30-unit 入口的 hard requirement（bridge 任务
      验收后）；本任务 validator 静态确认当前 CLI 仍拒绝未授权的 30-unit real B01。
- [ ] validator 不创建 output_dir、不调 `train_one_unit`、不调 `run_full`、不读
      TEST；通过 AST 解析测试覆盖。
- [ ] `test_check_markdown_links.py` 通过；新合同 / 更新文档的相对链接有效。
- [ ] `git status --short` 只列出本任务产生的变更；`git diff --check` 通过；
      `py_compile` 通过；无 staged。
- [ ] B07 validator 在主 worktree 仍 12/12 通过；b09 worktree 上 2 项 CRLF / freeze
      manifest 路径漂移是 worktree 间的预先存在差异，**不是 B09 改动造成的回归**；
      本任务未尝试修复（按文件白名单）。
- [ ] B08 runner 关键子集 + 慢子集 + synthetic full run smoke 全部通过。
- [ ] B09 30 项定向测试（normal / missing / duplicate / invalid candidate/fold/seed /
      budget drift / dirty Git / identity missing / output empty / output DONE /
      output FAILED / multi-terminal / STOPPED missing id / STOPPED missing
      git_commit / STOPPED wrong git_commit / STOPPED missing git_dirty / STOPPED
      git_dirty=true / STOPPED missing frozen hash / STOPPED mismatched frozen hash /
      STOPPED with unit_failed / valid INTERRUPTED resume / audit-only missing
      output / audit-only valid DONE / audit-only FAILED / audit-only terminal
      conflict / audit-only unit count < 30 / audit-only duplicate / audit-only
      identity mismatch / audit-only budget failure / audit-only TEST evidence /
      missing B01 freeze / wrong B01 SHA / missing A06 split / wrong A06 SHA /
      no-write / no-train / no-TEST）全部通过；负向测试用 `CheckLog` 或
      `capture_output` 抓取具体错误消息，不只 `assert rc != 0`。
- [ ] `docs/PROJECT_STATUS.md` 与 `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md` 已更新到
      `ITERATE / CLI_BRIDGE_REQUIRED / GPU_FULL_NOT_AUTHORIZED / TEST_DENIED`；
      没有 `ACCEPT / FULL_AUTHORIZED / FULL_RUNNING / FULL_COMPLETE /
      TEST_AUTHORIZED / RUN_PREPARATION_READY_FOR_REVIEW` 字样。
- [ ] Handoff 中明确列：本任务 commit SHA = `TO_BE_FROZEN_AFTER_CODE_REVIEW_AND_COMMIT`，
      不得伪造或提前冻结；下一任务是 `TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1`。

## 18. 禁止结论（prohibited conclusions）

本任务交付**不得**声称、声明或暗示以下任何结论：

- 30-unit Full 已被授权、已运行、已成功、已失败、已完成。
- 任何 `EXP-SLP-B09-*` EXP-ID 已 `QUEUED` / `RUNNING` / `SUCCEEDED` / `FAILED` /
  `REVIEWED` / `ACCEPTED`。
- 任何候选已"被选为最终研究候选"；B07 选择规则（`selection_rule`）尚未触发。
- B08 R03 的单 unit 指标可代表 30-unit Full。
- TEST 已读、已计算、已报告。
- SLP8 GT 的真值等级已超越 `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED / NOT_REVIEWED`。
- SLP8 结果已外推到产品、硬件、舒适性、医学、整夜或气囊控制。
- 在 bridge 验收前提前声称 "RUN_PREPARATION_READY_FOR_REVIEW"；该历史禁令已由
  `main@8b3ebda` 的 bridge 验收满足，不再是当前 blocker。

## 19. 当前 Gate（bridge 验收后更新）

- `TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1` 已独立验收并合入
  `main@8b3ebdaa021405790b6137bff581acc490d8a024`。
- 当前状态：`B09_RUN_PREPARATION_ACCEPTED / FULL_RUNNER_CLI_BRIDGE_ACCEPTED /
  GPU_FULL_NOT_AUTHORIZED / TEST_DENIED`。
- 30-unit Full 与 TEST 仍需 Owner 分别独立授权；当前没有正式 EXP-ID 进入
  `QUEUED` 或 `RUNNING`。

## 20. 下一 Gate

- Owner 复核 `TASK-SLP-B09-FULL-RUN-AUTHORIZATION-PREPARATION-v0.1`，并决定
  `AUTHORIZE / ITERATE / STOP`。
- 只有 Owner 明确 `AUTHORIZE` 后，才可将冻结的
  `EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01` 置为 `QUEUED` 并执行。
- Full 完成后进入 B10（UNKNOWN/REJECT）、B11（候选冻结）；B09T 一次性 TEST 仍需
  独立任务独立授权。
