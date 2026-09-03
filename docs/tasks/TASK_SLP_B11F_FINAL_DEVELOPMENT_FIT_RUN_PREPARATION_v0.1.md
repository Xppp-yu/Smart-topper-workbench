# TASK-SLP-B11F-FINAL-DEVELOPMENT-FIT-RUN-PREPARATION-v0.1

状态：`DRAFT_COMPLETE / READY_FOR_INDEPENDENT_REVIEW / GPU_NOT_AUTHORIZED / TEST_DENIED`

日期：2026-09-03
Proposed EXP-ID：`EXP-SLP-B11F-PM-FINAL-FIT-20260903-AUTODL-R01`

## 1. 目标与授权边界

为已验收的 B11F 最终开发集拟合 runner 建立可独立复核的运行准备合同，冻结候选
EXP-ID、已推送 runner SHA、输入 hash、训练计划、资源上限、AutoDL no-training
preflight 和正式命令。

Owner 的“继续推进下去”指令授权创建、验证、提交并推送本运行准备文档，只允许修改：

- `docs/tasks/TASK_SLP_B11F_FINAL_DEVELOPMENT_FIT_RUN_PREPARATION_v0.1.md`
- `docs/stage_reports/S2_B11F_SLP8_FINAL_DEVELOPMENT_FIT_RUN_PREPARATION_v0.1.md`
- `docs/PROJECT_STATUS.md`
- `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`

本任务不授权 GPU/CUDA preflight、GPU final fit、TEST、实验输出目录写入、代码或配置
修改。只有独立审查 `ACCEPT`、AutoDL no-training preflight 通过且 Owner 再次明确授权
下述精确对象后，Experiment Runner 才可启动训练。

## 2. 已满足的前置 Gate

- B11 candidate contract：`CANDIDATE_CONTRACT_ACCEPTED / FINAL_FIT_NOT_RUN / TEST_DENIED`。
- B11F implementation R05：独立只读复审 `ACCEPT`，无 P0/P1/P2。
- release commit 已推送：`main == origin/main == 0f3fb936653460a4fb81093f0e95cc1014e366db`。
- B11F 定向 `31 passed`；B11F+B11+B08/B09 联合回归 `114 passed`；Markdown
  links `6 passed`；validator、validate-only、`py_compile`、`git diff --check` 通过。
- 所有验证均为 TEST=0；GPU final fit `NOT RUN`。

## 3. 待审查的冻结对象

| 字段 | 冻结值 |
|---|---|
| TASK-ID | `TASK-SLP-B11F-FINAL-DEVELOPMENT-FIT-RUN-PREPARATION-v0.1` |
| Proposed EXP-ID | `EXP-SLP-B11F-PM-FINAL-FIT-20260903-AUTODL-R01` |
| Runner Git SHA | `0f3fb936653460a4fb81093f0e95cc1014e366db` |
| Git dirty | 必须为 `false` |
| Config | `configs/experiments/slp8_pm_final_development_fit_v0.1.json` |
| Config SHA-256（Git/LF、AutoDL runtime） | `8c1f93e27f7547831cf5ac6da945257f1de653c0d9054bd0b4dee59647992621` |
| Candidate contract | `configs/experiments/slp8_pm_research_candidate_v0.1.json` |
| Candidate SHA-256（Git/LF、AutoDL runtime） | `34f0fcf45d07920b99b7baf6d595f61297f086ff3187c9ec9b3bd69400b2cd4b` |
| B01 freeze manifest SHA-256 | `42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04` |
| A06 split SHA-256 | `024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706` |
| Model | `slp8_deeplabv3plus_lite_v0.1` only |
| Data | B01 TRAIN+VAL only；3,645+450=4,095 samples；81+10=91 subjects |
| Seeds / fixed epochs | `42→15`、`123→20`、`2026→12` |
| Optimizer | AdamW；batch 16；lr 0.001；weight decay 0.0001；shuffle true |
| Selection | 无 early stopping；无 validation selection；training loss 仅为运行诊断 |
| Outputs | 三个 `final.pt` 全部独立重载并审计后才允许根 `DONE.json` |
| TEST | denied；`load_test=False`；`_test_rows is None`；所有 carrier 为 0 |

Windows 当前工作树的 candidate 文件可能因 CRLF materialization 得到诊断 hash
`839c9482c69cf34d3c91c3acb3c7a36cb4d199117d0d6eb2ceb7906bac52b994`；它不是
AutoDL Linux 运行身份。AutoDL 必须从上述 release SHA checkout，并以 Git/LF hash
`34f0fcf4...` 为准。任何 hash 漂移都 fail closed，不得现场改文件后沿用 EXP-ID。

## 4. 资源与停止条件

- 目标：AutoDL RTX 4090 24 GB，至少 8 vCPU / 32 GB RAM。
- peak CUDA memory 硬上限：8,192 MiB；runner 逐 batch 与重载后检查。
- 启动前输出卷可用空间至少 1 GiB；runner fail closed 检查。
- proposed total wall budget：45 分钟；正式命令由外部
  `timeout --signal=INT --kill-after=2m 45m` 约束，另留最多 2 分钟只用于 terminal
  落盘/强制回收。收到 SIGINT 后应形成 `STOPPED.json`，不得误报 DONE；若清理失败被
  SIGKILL，遗留 RUNNING 必须按显式恢复合同审计，不能视为完成。
- OOM、NaN/Inf、超显存、磁盘不足、identity/hash 漂移、TEST 非零、checkpoint
  重载不一致、缺任一 seed completion 或多根 terminal，均停止并保留证据。
- 若 45 分钟耗尽，同一 EXP-ID 不得增加预算继续；变更预算必须新 EXP-ID 和重新授权。

## 5. AutoDL no-training preflight（本任务不执行）

以下命令只允许在本任务独立审查 `ACCEPT` 后执行。不得使用正式 EXP-ID 输出目录作为
preflight 目录，不得加载 B01 TEST rows/labels/onehot。

```bash
set -euo pipefail
B11F_REPO=/root/autodl-tmp/smarttopper-team-workbench
B11F_FREEZE_DIR=/root/autodl-tmp/data/processed/slp8_training_tables_v0.1
B11F_DATA_ROOT=/root/autodl-tmp/datasets/SLP_8Region_Pressure_VAL_v1.1
B11F_CONFIG="$B11F_REPO/configs/experiments/slp8_pm_final_development_fit_v0.1.json"
B11F_CANDIDATE="$B11F_REPO/configs/experiments/slp8_pm_research_candidate_v0.1.json"
B11F_EXP_ID=EXP-SLP-B11F-PM-FINAL-FIT-20260903-AUTODL-R01
B11F_GIT_SHA=0f3fb936653460a4fb81093f0e95cc1014e366db
B11F_CONFIG_SHA=8c1f93e27f7547831cf5ac6da945257f1de653c0d9054bd0b4dee59647992621
B11F_CANDIDATE_SHA=34f0fcf45d07920b99b7baf6d595f61297f086ff3187c9ec9b3bd69400b2cd4b
B11F_FREEZE_SHA=42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04

cd "$B11F_REPO"
git fetch origin main
git checkout --detach "$B11F_GIT_SHA"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$B11F_GIT_SHA"
git merge-base --is-ancestor "$B11F_GIT_SHA" origin/main
test "$(sha256sum "$B11F_CONFIG" | cut -d' ' -f1)" = "$B11F_CONFIG_SHA"
test "$(sha256sum "$B11F_CANDIDATE" | cut -d' ' -f1)" = "$B11F_CANDIDATE_SHA"
test "$(sha256sum "$B11F_FREEZE_DIR/freeze_manifest.json" | cut -d' ' -f1)" = "$B11F_FREEZE_SHA"
test -d "$B11F_DATA_ROOT"
test ! -e "$B11F_REPO/outputs/experiments/$B11F_EXP_ID"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)" = "NVIDIA GeForce RTX 4090"
uv run python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "cudnn": torch.backends.cudnn.version(),
       "gpu": torch.cuda.get_device_name(0)})
PY
uv run python scripts/validate_slp8_b11f_final_fit_preparation.py "$B11F_CONFIG"
uv run python scripts/run_slp8_region_final_fit.py --config "$B11F_CONFIG" --validate-only
```

Pass 要求：checkout clean 且 exact SHA；release SHA 是 origin/main ancestor；三项文件 hash 一致；GPU 身份
符合；目标 EXP-ID 目录不存在；validator 与 validate-only 明确输出 TEST=0；没有训练、
checkpoint、RUNNING/DONE/FAILED 或正式实验目录写入。完整 transcript 必须回传独立审查。

## 6. Owner 授权记录

```text
Run-preparation independent review: PENDING
AutoDL no-training preflight review: PENDING
Owner authorization: PENDING
Authorization timestamp: PENDING
Final EXP-ID: EXP-SLP-B11F-PM-FINAL-FIT-20260903-AUTODL-R01
Runner Git SHA: 0f3fb936653460a4fb81093f0e95cc1014e366db
Git dirty: false required
Config SHA-256: 8c1f93e27f7547831cf5ac6da945257f1de653c0d9054bd0b4dee59647992621
Candidate SHA-256: 34f0fcf45d07920b99b7baf6d595f61297f086ff3187c9ec9b3bd69400b2cd4b
B01 freeze manifest SHA-256: 42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04
AutoDL instance / GPU / PyTorch / CUDA / cuDNN: PENDING
Peak CUDA budget: 8192 MiB
Total wall budget: 45 minutes
TEST access: denied / 0
Exact launch transcript: PENDING
```

## 7. 冻结正式命令（禁止在本任务执行）

只有 §6 全部完成且 Owner 明确授权精确 EXP-ID/SHA/环境/预算/命令后，才可执行：

```bash
set -euo pipefail
B11F_REPO=/root/autodl-tmp/smarttopper-team-workbench
B11F_FREEZE_DIR=/root/autodl-tmp/data/processed/slp8_training_tables_v0.1
B11F_DATA_ROOT=/root/autodl-tmp/datasets/SLP_8Region_Pressure_VAL_v1.1
B11F_CONFIG="$B11F_REPO/configs/experiments/slp8_pm_final_development_fit_v0.1.json"
B11F_CANDIDATE="$B11F_REPO/configs/experiments/slp8_pm_research_candidate_v0.1.json"
B11F_EXP_ID=EXP-SLP-B11F-PM-FINAL-FIT-20260903-AUTODL-R01
B11F_OUTPUT="$B11F_REPO/outputs/experiments/$B11F_EXP_ID"
B11F_CONFIG_SHA=8c1f93e27f7547831cf5ac6da945257f1de653c0d9054bd0b4dee59647992621
B11F_CANDIDATE_SHA=34f0fcf45d07920b99b7baf6d595f61297f086ff3187c9ec9b3bd69400b2cd4b
B11F_FREEZE_SHA=42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04

cd "$B11F_REPO"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "0f3fb936653460a4fb81093f0e95cc1014e366db"
test "$(sha256sum "$B11F_CONFIG" | cut -d' ' -f1)" = "$B11F_CONFIG_SHA"
test "$(sha256sum "$B11F_CANDIDATE" | cut -d' ' -f1)" = "$B11F_CANDIDATE_SHA"
test "$(sha256sum "$B11F_FREEZE_DIR/freeze_manifest.json" | cut -d' ' -f1)" = "$B11F_FREEZE_SHA"
test -d "$B11F_DATA_ROOT"
test ! -e "$B11F_OUTPUT"
timeout --signal=INT --kill-after=2m 45m uv run python scripts/run_slp8_region_final_fit.py \
  --config "$B11F_CONFIG" \
  --output-dir "$B11F_OUTPUT" \
  --b01-freeze-dir "$B11F_FREEZE_DIR" \
  --dataset-root "$B11F_DATA_ROOT" \
  --experiment-id "$B11F_EXP_ID" \
  --run-authorized
```

仅当同一 identity 的非终态 `RUNNING.json` 或 `STOPPED.json` 存在，且尚有原 45 分钟
预算余额、环境文件 hash 未漂移并取得显式恢复授权时，才允许：

```bash
set -euo pipefail
B11F_REPO=/root/autodl-tmp/smarttopper-team-workbench
B11F_FREEZE_DIR=/root/autodl-tmp/data/processed/slp8_training_tables_v0.1
B11F_DATA_ROOT=/root/autodl-tmp/datasets/SLP_8Region_Pressure_VAL_v1.1
B11F_CONFIG="$B11F_REPO/configs/experiments/slp8_pm_final_development_fit_v0.1.json"
B11F_CANDIDATE="$B11F_REPO/configs/experiments/slp8_pm_research_candidate_v0.1.json"
B11F_EXP_ID=EXP-SLP-B11F-PM-FINAL-FIT-20260903-AUTODL-R01
B11F_OUTPUT="$B11F_REPO/outputs/experiments/$B11F_EXP_ID"
B11F_CONFIG_SHA=8c1f93e27f7547831cf5ac6da945257f1de653c0d9054bd0b4dee59647992621
B11F_CANDIDATE_SHA=34f0fcf45d07920b99b7baf6d595f61297f086ff3187c9ec9b3bd69400b2cd4b
B11F_FREEZE_SHA=42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04
B11F_REMAINING_MINUTES=REPLACE_WITH_OWNER_AUTHORIZED_INTEGER_1_TO_44

cd "$B11F_REPO"
test "$B11F_REMAINING_MINUTES" -ge 1
test "$B11F_REMAINING_MINUTES" -le 44
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "0f3fb936653460a4fb81093f0e95cc1014e366db"
test "$(sha256sum "$B11F_CONFIG" | cut -d' ' -f1)" = "$B11F_CONFIG_SHA"
test "$(sha256sum "$B11F_CANDIDATE" | cut -d' ' -f1)" = "$B11F_CANDIDATE_SHA"
test "$(sha256sum "$B11F_FREEZE_DIR/freeze_manifest.json" | cut -d' ' -f1)" = "$B11F_FREEZE_SHA"
test -d "$B11F_DATA_ROOT"
test ! -e "$B11F_OUTPUT/DONE.json"
test ! -e "$B11F_OUTPUT/FAILED.json"
if test -e "$B11F_OUTPUT/RUNNING.json" && test -e "$B11F_OUTPUT/STOPPED.json"; then exit 1; fi
test -e "$B11F_OUTPUT/RUNNING.json" || test -e "$B11F_OUTPUT/STOPPED.json"
timeout --signal=INT --kill-after=2m "${B11F_REMAINING_MINUTES}m" \
  uv run python scripts/run_slp8_region_final_fit.py \
  --config "$B11F_CONFIG" \
  --output-dir "$B11F_OUTPUT" \
  --b01-freeze-dir "$B11F_FREEZE_DIR" \
  --dataset-root "$B11F_DATA_ROOT" \
  --experiment-id "$B11F_EXP_ID" \
  --run-authorized --resume-authorized
```

恢复命令中的 timeout 必须改为原预算实际剩余值，不能重新获得 45 分钟。DONE/FAILED、
身份不一致、环境漂移或预算耗尽均禁止恢复；不得使用 `--force` 或覆盖已有文件。

## 8. Reviewer checklist 与下一 Gate

- [ ] Proposed EXP-ID 在本地、AutoDL 和既有记录中均未使用。
- [ ] Runner SHA 为已推送 clean commit，config/candidate/freeze hash 三方一致。
- [ ] B01 只加载 TRAIN+VAL；`load_test=False`、`_test_rows is None`、TEST=0。
- [ ] 模型、seeds/epochs、AdamW 与全部超参数和 B11/B09 冻结合同一致。
- [ ] no-training preflight 不创建正式输出、不训练、不读取 TEST。
- [ ] 正式命令具备外部 wall timeout、显存/磁盘门禁与唯一 EXP-ID 输出路径。
- [ ] Owner 授权时间早于 RUNNING，且精确覆盖 SHA/EXP-ID/环境/预算/命令。
- [ ] 三个 checkpoint 与 DONE 的 identity/SHA/reload/terminal 审计完整后才进入 B09T。

本准备记录不产生模型性能证据，也不验证真实 wall/peak、跨进程恢复或三个 checkpoint。

下一 Gate：`B11F_RUN_PREPARATION_INDEPENDENT_REVIEW / GPU_NOT_AUTHORIZED / TEST_DENIED`。
