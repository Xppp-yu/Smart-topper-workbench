# TASK-SLP-B04A-R03-RUN-PREPARATION-v0.1

**Status:** `PREPARATION_COMPLETE / READY_FOR_AUTODL_PREFLIGHT / GPU_R03_NOT_AUTHORIZED`
**Stage:** S2-B04A corrected formal GPU Mini R03 preparation
**Date:** 2026-08-30
**Proposed EXP-ID:** `EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-AUTODL-R03`

> This task freezes R03 identity, inputs, commands, and gates. It does not
> authorize training. After Codex reviews a passing AutoDL no-training
> Preflight, the Owner must separately authorize the final EXP-ID, clean
> released Git SHA, machine, budget, and exact command.

## Objective and current gate

Create a new auditable R03 record after the accepted reload and experiment
identity fixes. R01/R02 remain failed historical archives and must not be
resumed, overwritten, or used for advancement.

At task start, `main` and `origin/main` were clean and synchronized at
`bdd4c3d5a6a6e205bc1af9676be68843b587e2d3` (ahead 0 / behind 0; no running
job). This is the candidate code baseline. Because this preparation document
is not yet committed, the final run SHA remains `PENDING_TASK_RELEASE`; after
release it must be re-frozen to the resulting clean commit. A dirty worktree
must never be presented as the final run identity.

## Frozen identity and inputs

| Item | Frozen value |
|---|---|
| Proposed EXP-ID | `EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-AUTODL-R03` |
| Candidate code SHA | `bdd4c3d5a6a6e205bc1af9676be68843b587e2d3` |
| Final clean released SHA | `PENDING_TASK_RELEASE` |
| Git dirty at run | `false` required |
| Config | `configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json` |
| Config SHA-256 (Git/LF bytes; AutoDL runtime) | `74230e146cdde4b980c5f3e8308e2a6ad6f176ac1b16941243a6e8a6b8aab3fa` |
| B01 archive SHA-256 | `23b32395238130437c1ec1b0771fbc793b8be74578b0a1acb2ca237c1913269a` |
| `freeze_manifest.json` file SHA-256 | `42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04` |
| Freeze manifest core SHA-256 | `3c78999551580fc46ce15229e053798b5e4c9464a5bab27e05130cb319090b1e` |
| A06 split SHA-256 | `024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706` |
| Data | TRAIN 3,645 / VAL 450 / TEST 0; danaLab; uncover only |
| Candidates | SmallUNet, ResUNet-lite, DeepLabV3+-lite frozen v0.1 registrations |
| Seeds | `42`, `123`, `2026` |
| Budget | 45 cumulative min/candidate; 135 total; peak CUDA <= 8192 MB |
| Device | AutoDL CUDA; target RTX 4090 24 GB; exact identity pending Preflight |

The config hash is frozen from the committed Git blob and the LF checkout used
on AutoDL. A Windows checkout may materialize CRLF bytes and therefore produce
`f5c4ed124fe5eef80b57345d56d73c4c3efaab491b84804dd96051a1103e6a5a`;
that Windows worktree hash is diagnostic only and must not be used as the Linux
run identity. All hashes must be recomputed on AutoDL. Any other mismatch is
fail-closed and requires a new preparation record, not an in-run repair.

## Boundaries

This task may change only this task, its stage report, and narrow B04A status
entries in `docs/PROJECT_STATUS.md` and `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`.
The future Runner may write only `outputs/experiments/<EXP-ID>/`. Raw SLP8,
B01 freeze inputs, frozen config, and R01/R02 outputs stay read-only. This task
does not authorize package installation, data upload, config/code edits,
commit/push, TEST access, real training, or B07/Full.

## AutoDL no-training Preflight

Bind the paths to the actual AutoDL instance. Do not use the real R03 output
directory as the Preflight directory.

```bash
set -euo pipefail
B04A_REPO=/root/autodl-tmp/smarttopper-team-workbench
B04A_ARCHIVE=/root/autodl-tmp/slp8_training_tables_v0.1.tar.gz
B04A_FREEZE_DIR=/root/autodl-tmp/data/processed/slp8_training_tables_v0.1
B04A_DATA_ROOT=/root/autodl-tmp/datasets/SLP_8Region_Pressure_VAL_v1.1
B04A_CONFIG="$B04A_REPO/configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json"
B04A_PREFLIGHT="$B04A_REPO/outputs/preflight/B04A_R03_20260830"
B04A_EXP_ID=EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-AUTODL-R03
B04A_FINAL_GIT_SHA=PENDING_TASK_RELEASE
B04A_CONFIG_SHA=74230e146cdde4b980c5f3e8308e2a6ad6f176ac1b16941243a6e8a6b8aab3fa
B04A_ARCHIVE_SHA=23b32395238130437c1ec1b0771fbc793b8be74578b0a1acb2ca237c1913269a
B04A_FREEZE_SHA=42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04

cd "$B04A_REPO"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$B04A_FINAL_GIT_SHA"
git rev-list --left-right --count origin/main...HEAD
test "$(sha256sum "$B04A_CONFIG" | cut -d' ' -f1)" = "$B04A_CONFIG_SHA"
test "$(sha256sum "$B04A_ARCHIVE" | cut -d' ' -f1)" = "$B04A_ARCHIVE_SHA"
if test ! -d "$B04A_FREEZE_DIR"; then
  mkdir -p "$(dirname "$B04A_FREEZE_DIR")"
  tar -xzf "$B04A_ARCHIVE" -C "$(dirname "$B04A_FREEZE_DIR")"
fi
test "$(sha256sum "$B04A_FREEZE_DIR/freeze_manifest.json" | cut -d' ' -f1)" = "$B04A_FREEZE_SHA"
test ! -e "$B04A_PREFLIGHT"
test ! -e "$B04A_REPO/outputs/experiments/$B04A_EXP_ID"
uv run python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0),
       "capability": torch.cuda.get_device_capability(0)})
PY
uv run python scripts/validate_b04a_protocol.py "$B04A_CONFIG"
uv run python scripts/smoke_b04a_implementation.py --no-write
uv run python scripts/smoke_b04a_runner_integration.py --no-write
uv run python scripts/run_slp8_region_mini.py \
  --config "$B04A_CONFIG" --output-dir "$B04A_PREFLIGHT" --validate-config
```

Pass requires: clean final released SHA; exact hashes above; intended
ahead/behind state; actual CUDA forward/backward smoke for all three candidates;
validator `30 OKs / 0 errors`; runner-integration no-write PASS; validate-only
identity bundle only; unused R03 output; no B01 table, raw sample, or TEST label
read. Preserve the Preflight transcript for Codex review.

## Owner authorization record

```text
Codex Preflight review: PENDING
Owner authorization: PENDING
Authorization timestamp: PENDING
Final EXP-ID: EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-AUTODL-R03
Final clean released Git SHA: PENDING_TASK_RELEASE
Git dirty: false required
Config SHA-256: 74230e146cdde4b980c5f3e8308e2a6ad6f176ac1b16941243a6e8a6b8aab3fa
B01 freeze manifest file SHA-256: 42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04
B01 freeze manifest core SHA-256: 3c78999551580fc46ce15229e053798b5e4c9464a5bab27e05130cb319090b1e
A06 split SHA-256: 024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706
AutoDL instance/GPU: PENDING
PyTorch/CUDA: PENDING
Peak CUDA budget: 8192 MB
Total wall budget: 135 cumulative minutes
TEST access: denied / 0
Exact launch command transcript: PENDING
```

## Frozen launch and resume commands

Do not execute before the record above is complete and Owner authorization
explicitly names this exact invocation.

```bash
set -euo pipefail
B04A_REPO=/root/autodl-tmp/smarttopper-team-workbench
B04A_FREEZE_DIR=/root/autodl-tmp/data/processed/slp8_training_tables_v0.1
B04A_DATA_ROOT=/root/autodl-tmp/datasets/SLP_8Region_Pressure_VAL_v1.1
B04A_CONFIG="$B04A_REPO/configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json"
B04A_EXP_ID=EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-AUTODL-R03
B04A_FINAL_GIT_SHA=PENDING_TASK_RELEASE
B04A_CONFIG_SHA=74230e146cdde4b980c5f3e8308e2a6ad6f176ac1b16941243a6e8a6b8aab3fa
B04A_FREEZE_SHA=42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04

cd "$B04A_REPO"
B04A_OUTPUT="$B04A_REPO/outputs/experiments/$B04A_EXP_ID"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$B04A_FINAL_GIT_SHA"
test "$(sha256sum "$B04A_CONFIG" | cut -d' ' -f1)" = "$B04A_CONFIG_SHA"
test "$(sha256sum "$B04A_FREEZE_DIR/freeze_manifest.json" | cut -d' ' -f1)" = "$B04A_FREEZE_SHA"
test ! -e "$B04A_OUTPUT"
uv run python scripts/run_slp8_region_mini.py \
  --config "$B04A_CONFIG" \
  --output-dir "$B04A_OUTPUT" \
  --b01-freeze-dir "$B04A_FREEZE_DIR" \
  --dataset-root "$B04A_DATA_ROOT" \
  --experiment-id "$B04A_EXP_ID" \
  --run-authorized
```

Only a non-terminal interrupted R03 may resume the same immutable identity:

```bash
uv run python scripts/run_slp8_region_mini.py \
  --config "$B04A_CONFIG" --output-dir "$B04A_OUTPUT" \
  --b01-freeze-dir "$B04A_FREEZE_DIR" --dataset-root "$B04A_DATA_ROOT" \
  --resume-from "$B04A_OUTPUT" --experiment-id "$B04A_EXP_ID" --run-authorized
```

Never resume `DONE`, reuse R01/R02, or use `--force`.

## Reviewer checklist and next Gate

- [ ] Task release commit exists and final clean SHA is recorded everywhere.
- [ ] EXP-ID is unused locally and remotely.
- [ ] Git/config/archive/freeze/split identities match.
- [ ] AutoDL CUDA and all three candidate CUDA smokes pass.
- [ ] Preflight performed no training and accessed no TEST.
- [ ] Owner authorization timestamp precedes the run.
- [ ] Exact command includes both `--experiment-id` and `--run-authorized`.
- [ ] TRAIN=3,645 / VAL=450 / TEST=0 and resource guards remain frozen.
- [ ] R03 evidence is independently accepted before B07 is unblocked.

Any missing/drifting identity, dirty Git, CUDA failure, existing output, TEST
access, budget breach, checkpoint mismatch, or multiple terminal files stops
the run while preserving evidence.

Preparation proves no model performance. R03 cannot validate cover conditions,
TEST subjects, self-developed hardware, calibrated pressure, comfort, medical,
overnight, or airbag-control conclusions.

**Next Gate:** release this task and freeze its clean SHA; run only AutoDL
no-training Preflight; Codex reviews evidence; Owner separately authorizes the
exact R03 command. Until then `GPU_R03_NOT_AUTHORIZED`; B07 remains
`BLOCKED_BY_B04A`.
