# Repository Agent Rules

## Read first

Before changing this repository, read:

1. `COLLABORATION_WORKFLOW.md`
2. `docs/PROJECT_STATUS.md`
3. `docs/VALIDATION_WORKFLOW_MASTER.md`
4. `docs/EXPERIMENT_GOVERNANCE_AND_GPU_EXECUTION_PLAN_v0.1.md`
5. For SLP work: `docs/SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md`
6. For SLP work: `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`

## Collaboration roles

- The Owner sets direction, priority, budget, run authorization and final acceptance.
- Web GPT works from the pushed GitHub baseline for strategy, task drafting and second review; it must not claim knowledge of local dirty files, raw data or ignored outputs.
- Claude Code is the main coding agent for one declared TASK-ID in one bounded worktree.
- Codex is the local controller/reviewer and verifies the actual worktree, data, outputs and stage evidence before a formal commit.
- Experiment Runner executes only frozen experiments.
- GitHub is the committed and pushed shared baseline, not the latest local state.

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
- Geometry/OpenCV region proposals (`R0/R1`) are pseudo-labels. **They are NOT the project reference GT.**
- **SLP 8-region pressure-only GT** (`SLP_8Region_Pressure_VAL_v1.1`, 4,590 samples, 102 danaLab subjects) is the **PROJECT_ACCEPTED_REFERENCE_GT** for SLP8 region segmentation. Its provenance is `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`, `source_review_status=NOT_REVIEWED`. **Not human pixel-level semantic masks. Not medical, skin-interface stress, or product GT.**
- The 10-region polygon+review ontology (`slp_region_annotation_v0.1`, `R0-R3` tiers) is **historical internal governance**; it is **NOT the current training contract** and must NOT be conflated with the 8-region dataset.
- Only human-reviewed, QC-passed labels (`R2/R3`) may be used as the default polygon-route training reference. The SLP8 pressure GT is the current default training reference.
- Keep posture, joints, coarse regions, pixel segmentation and product zones as separate tasks and metrics.

## Implementation and testing

- Preserve unrelated user changes in a dirty worktree.
- Before editing, capture branch, HEAD, dirty/untracked state, active TASK, running jobs, relevant outputs and ahead/behind status. Use `scripts/project_status_snapshot.py` when helpful.
- Separate parallel PoPu and SLP work with different branches/worktrees or explicit non-overlapping file boundaries.
- Use deterministic subject-level splits and fit preprocessing on training subjects only.
- Add focused tests for each behavior and run the relevant suite.
- Do not run Mini/Full GPU experiments without an explicit frozen protocol and authorization.
- Never overwrite completed EXP-ID artifacts; failure paths must remain auditable.
- End every stage report with verified, inferred, unverified, limitations and next Gate.

## Handoff

Each handoff must include TASK-ID, files changed, commands run, test results, generated artifacts, known failures, prohibited conclusions, reviewer checklist and current Git status. Mark unexecuted checks as `NOT RUN`. Do not commit or push unless the TASK explicitly authorizes it.
