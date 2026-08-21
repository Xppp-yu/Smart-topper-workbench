# Repository Agent Rules

## Read first

Before changing this repository, read:

1. `docs/PROJECT_STATUS.md`
2. `docs/VALIDATION_WORKFLOW_MASTER.md`
3. `docs/EXPERIMENT_GOVERNANCE_AND_GPU_EXECUTION_PLAN_v0.1.md`
4. For SLP work: `docs/SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md`
5. For SLP work: `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`

## Scope and evidence

- Work on exactly one declared TASK-ID at a time.
- Do not modify raw datasets under `E:/TeamProjects/datasets/smart-topper`.
- Never commit local path configs, credentials, raw RGB/IR/depth/pressure data, archives, or large checkpoints.
- Code or a plan is not a completed experiment. Report commands actually run and preserve artifacts.
- Public-data results are research evidence, not product, hardware, comfort, medical, overnight, or airbag-control validation.
- PoPu, PMD, SLP and PressurePose are independent datasets. Never row-pair unrelated subjects or frames.

## SLP truth boundary

- RGB/IR 14-joint labels are original manual joint ground truth (`J0`).
- Homography-mapped joints are derived references (`J1`), not unbiased ground truth.
- Geometry/OpenCV region proposals (`R0/R1`) are pseudo-labels.
- Only human-reviewed, QC-passed labels (`R2/R3`) may be used as the default SLP region training reference.
- Keep posture, joints, coarse regions, pixel segmentation and product zones as separate tasks and metrics.

## Implementation and testing

- Preserve unrelated user changes in a dirty worktree.
- Use deterministic subject-level splits and fit preprocessing on training subjects only.
- Add focused tests for each behavior and run the relevant suite.
- Do not run Mini/Full GPU experiments without an explicit frozen protocol and authorization.
- Never overwrite completed EXP-ID artifacts; failure paths must remain auditable.
- End every stage report with verified, inferred, unverified, limitations and next Gate.

## Handoff

Each handoff must include TASK-ID, files changed, commands run, test results, generated artifacts, known failures, prohibited conclusions and reviewer checklist.
