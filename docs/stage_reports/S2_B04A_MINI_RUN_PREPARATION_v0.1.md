# S2 B04A Mini Run Preparation v0.1

**TASK-ID:** `TASK-SLP-B04A-MINI-RUN-PREPARATION-v0.1`
**Status:** `READY_FOR_REVIEW / GPU_MINI_NOT_AUTHORIZED`
**Date:** 2026-08-30

## Outcome

The real B04A Mini has an executable task contract, proposed EXP-ID, local
input inventory, fail-closed preflight, real-run command template, resume
boundary, expected artifact list, and Reviewer checklist. No GPU Mini, TEST,
package installation, commit, push, or merge was performed by this preparation
task.

## Files changed

- `docs/tasks/TASK_SLP_B04A_MINI_RUN_v0.1.md`: new real-run contract and
  terminal command sequence.
- `docs/tasks/TASK_SLP_B04A_PM_ARCHITECTURE_EXPANSION_MINI_v0.1.md`: umbrella
  route status reconciled with completed Protocol/Implementation/Runner work.
- `docs/PROJECT_STATUS.md`: stale top summary and S1/B04A next-step text
  reconciled.
- `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`: Mini Run preparation and input
  restoration Gate recorded.
- `docs/stage_reports/S2_B04A_MINI_RUN_PREPARATION_v0.1.md`: this report.

## Local inventory

### Present

- Clean preparation parent: `236f14e` on `origin/main` at inventory time.
- Frozen B04A config and accepted three-candidate Runner.
- Raw SLP8 dataset:
  `E:\TeamProjects\datasets\smart-topper\SLP2022\SLP\SLP_8Region_Pressure_VAL_v1.1`.
- B01 transfer archive:
  `E:\TeamProjects\autodl-transfer\slp8_training_tables_v0.1.tar.gz`.
- Archive contents include the complete B01 freeze bundle: TRAIN/VAL/TEST
  manifests, freeze manifest, normalization, class statistics, and dataset card.
- Archive SHA-256:
  `23B32395238130437C1EC1B0771FBC793B8BE74578B0A1ACB2CA237C1913269A`.
- The archived freeze manifest declares A06 split SHA
  `024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706`,
  TRAIN/VAL/TEST counts 3,645/450/495, and TEST statistics as structural-only.

### Not present / not ready

- Extracted `data/processed/slp8_training_tables_v0.1/` was not present.
- A standalone `slp_subject_split_v0.1.json` was not found under
  `E:\TeamProjects`; the B01 freeze manifest remains the Runner input and must
  carry/validate the expected A06 split identity.
- The observed repository `uv` environment used a CPU-only PyTorch build;
  real CUDA availability is unverified.
- Final released run SHA, config hash, machine identity, and Owner authorization
  remain pending.

## Commands actually run

- Read all repository-mandated governance/status/SLP route documents.
- Read the accepted Runner Integration task and CLI help.
- Inspected the frozen config's training, determinism, metric, resource,
  artifact, identity, and authorization sections.
- Located the raw dataset and B01 transfer archive.
- Listed archive contents and computed its SHA-256.
- Read the archived `freeze_manifest.json` directly without extracting it.
- Searched for an extracted B01 freeze directory and standalone A06 split.
- Markdown relative-link check over all five changed documents: 5 files,
  0 errors.
- Frozen B04A protocol validator: 30 OKs / 0 errors.
- Focused documentation/protocol tests: 56 passed.
- `git diff --check`: passed.

Research training: `NOT RUN`.

## Verified

- Code/Runner baseline was accepted before this task.
- The raw dataset and complete B01 transfer archive exist locally.
- The archive hash and members listed above are observable.
- Frozen protocol requires CUDA, 3 candidates x 3 seeds, TEST=0, and immutable
  artifact identity.
- Repository status text had two stale references claiming Runner Integration
  had not started; this task reconciles them without changing research results.
- The preparation documents and unchanged frozen protocol pass the checks listed
  above.

## Inferred

- The B01 freeze artifacts can be restored from the transfer archive without
  rebuilding labels or touching raw data, subject to post-extraction identity
  validation by the real Runner.
- The small model sizes likely fit the 8192 MB cap, but actual CUDA peak and
  135-minute cumulative wall budget remain empirical preflight/run facts.

## Unverified

- CUDA-enabled PyTorch and GPU behavior on the user's chosen terminal environment.
- Final clean release SHA and config hash after this documentation task is
  reviewed and published.
- Real B01 TRAIN/VAL load, all nine training units, metrics, resource use, and
  candidate ranking.

## Limitations

The SLP8 reference remains `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED` with
`source_review_status=NOT_REVIEWED`, danaLab/uncover only, raw PMarray response
semantics, and TEST inaccessible. Preparation evidence does not imply model
performance, B07 readiness, product/hardware validity, comfort, medical,
overnight, or airbag-control validity.

## Next Gate

Reviewer checks this preparation diff. After it is released, the user runs the
no-training preflight in `TASK_SLP_B04A_MINI_RUN_v0.1.md`. Only a passing CUDA
preflight plus a separately recorded Owner authorization may move the proposed
EXP-ID into `QUEUED`. B07 remains `BLOCKED_BY_B04A`.
