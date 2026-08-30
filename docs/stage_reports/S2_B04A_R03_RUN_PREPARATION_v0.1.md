# S2 B04A R03 Run Preparation v0.1

**TASK-ID:** `TASK-SLP-B04A-R03-RUN-PREPARATION-v0.1`
**Status:** `PREPARATION_COMPLETE / READY_FOR_AUTODL_PREFLIGHT / GPU_R03_NOT_AUTHORIZED`
**Date:** 2026-08-30

## Outcome

R03 now has a distinct preparation contract that freezes its proposed EXP-ID,
candidate code SHA, config/archive/freeze/split hashes, AutoDL no-training
Preflight, and exact launch/resume commands. It retains separate Codex review
and Owner authorization Gates. GPU Mini, TEST, commit, and push were `NOT RUN`.

## Files changed

- `docs/tasks/TASK_SLP_B04A_R03_RUN_PREPARATION_v0.1.md`
- `docs/stage_reports/S2_B04A_R03_RUN_PREPARATION_v0.1.md`
- narrow B04A entries in `docs/PROJECT_STATUS.md` and
  `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`

## Commands and results

- Project snapshot: `main` at `bdd4c3d5a6a6...`, clean, ahead 0 / behind 0,
  no running jobs at task start.
- SHA-256 verified: config `f5c4ed...e6a5a`; B01 archive
  `23b323...3269a`; archived freeze manifest `42e3cb...f3e04`.
- Runner CLI inspected: real B01 requires explicit `--experiment-id`,
  `--run-authorized`, freeze directory, and dataset root.
- Publication review made Preflight fail-closed against the exact released
  SHA and hashes, added non-overwriting archive restoration, and made the real
  launch block self-contained for a fresh AutoDL shell.
- Markdown relative-link check: 4 files, 0 errors; Bash blocks: syntax OK.
- Documentation/protocol tests: 56 passed; full B04A runner integration:
  129 passed; protocol validator: 30 OKs / 0 errors; `git diff --check`: passed.
- AutoDL CUDA Preflight: `NOT RUN`; research training: `NOT RUN`; TEST: `0`.

## Verified, inferred, and unverified

Verified: accepted fixes are merged; local archive/hash exists; proposed R03
output does not exist locally; frozen command explicitly carries EXP-ID and
preserves TEST=0.

Inferred: corrected Runner can proceed to AutoDL Preflight, subject to clean
task release and remote evidence.

Unverified: final clean release SHA; AutoDL machine/PyTorch/CUDA and remote
hashes; CUDA smoke; training, metrics, and resources.

## Limitations and next Gate

SLP8 remains danaLab/uncover-only,
`V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`, `source_review_status=NOT_REVIEWED`,
raw PMarray semantics, and non-medical/non-product reference GT. Preparation is
not experiment evidence.

Release this task and record the resulting clean SHA. Then run only the
no-training AutoDL Preflight. After Codex review, the Owner may separately
authorize the exact R03 command. B07 remains `BLOCKED_BY_B04A`.
