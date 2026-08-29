# TASK-SLP-B04A-RELOAD-CONSISTENCY-FIX-v0.1

**Status:** `ACCEPTED / R02_PRESERVED_FAILED`
**Stage:** S2-B04A defect correction before a new Mini run
**Branch:** `codex/task-slp-b04a-reload-consistency-fix-v0.1`
**Date:** 2026-08-30

## Objective

Correct the false reload-consistency failures observed in
`EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-AUTODL-R02` without
changing the frozen B04A candidates, seeds, threshold, augmentation,
resource budget, data split, or any R01/R02 artifact.

The corrected runner must compare an independently reloaded
`best.pt` model with logits captured from the same best-epoch model
state. It must never compare `best.pt` with the final in-memory model
when `best_epoch < final_epoch`.

## Incident evidence

The immutable local evidence archive is outside the repository:

```text
E:/TeamProjects/autodl-transfer/B04A_R01_R02_EVIDENCE_20260830.tar.gz
SHA-256: 75b9cd09fbf7214ddca9d0511991419a30cb404f709512a54f3f173029cb6494
```

R02 completed all 3 candidates x 3 seeds and wrote nine checkpoint
pairs. It ended `FAILED` because the runner compared the final model
with the reloaded best checkpoint. Every affected seed recorded
`hash_match=true`; failures aligned with `best_epoch < final_epoch`.
R02 remains a preserved failed experiment and is not rewritten.

## In scope

- Persist a deterministic, bounded best-epoch reload probe in
  `best.pt`.
- Compare the independently reloaded best model against that
  best-epoch probe on CPU.
- Preserve the existing full-validation prediction hash comparison.
- Add regression coverage where the model changes after the best
  checkpoint is written.
- Add corruption/fail-closed coverage for a mismatched reload probe.
- Record the incident, correction, test evidence, and R03 gate.

## Out of scope

- Reclassifying or editing R01/R02 outputs.
- Post-hoc promotion of the R02 `advanced` field to an accepted result.
- Changing B04A metrics, gates, candidates, seeds, early stopping, or
  resource budgets.
- Running GPU Mini, Full, or TEST.
- Fixing interruption lifecycle/resume-output collision in this task;
  those are separate defects and must not expand this correction.

## Files allowed to change

- `src/topper_perception/neural/slp8_region_mini.py`
- `tests/test_slp8_region_mini.py`
- `tests/test_b04a_runner_integration.py` only if B04A-specific
  orchestration coverage is required
- `docs/tasks/TASK_SLP_B04A_RELOAD_CONSISTENCY_FIX_v0.1.md`
- `docs/stage_reports/S2_B04A_RELOAD_CONSISTENCY_FIX_v0.1.md`
- `docs/PROJECT_STATUS.md`
- `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`

## Hard contract

1. Reload identity remains `best.pt` plus the frozen checkpoint
   identity block.
2. The reference probe must be produced from the best-epoch model
   state before serialization, using a fresh CPU model and a fixed
   bounded validation subset.
3. The probe must include its sample indices and logits in the best
   checkpoint; the reloaded model must use the same indices.
4. Missing, malformed, non-finite, or shape-inconsistent probe data
   fails closed.
5. `hash_match` remains independently computed over the full saved
   best-validation predictions.
6. Candidate feasibility requires both numeric probe consistency and
   full prediction-hash consistency.
7. Historical B04 behavior, B04A decision rules, and TEST=0 remain
   unchanged.

## Required tests

- A best checkpoint reloaded after the live model has advanced to a
  different final state remains reload-consistent.
- Corrupting the reloaded model state while retaining the stored probe
  is detected as inconsistent.
- Missing/malformed probe data fails closed.
- Existing B04 and B04A focused suites remain green.
- Protocol validator remains `30 OKs / 0 errors`.
- `git diff --check` is clean.

## Prohibited conclusions

- R02 is not `DONE`, `ACCEPTED`, or a formal candidate selection.
- DeepLabV3+-lite is not yet frozen for B07 from R02.
- No R03 GPU run is authorized by this code task.
- No TEST, product, hardware, comfort, medical, overnight, or airbag
  conclusion is permitted.

## Next Gate

After implementation review, complete a separate B04A experiment-identity
carrier correction. R02 used the task-derived value
`TASK-SLP-B04A-PROTOCOL-FREEZE-v0.1::<run>::seed=-` instead of the
Owner-authorized EXP-ID and copied the config SHA into
`data_manifest_sha256`; those defects are outside this task and block R03.
Only after both fixes are independently accepted may the Owner freeze a new
Git SHA and authorize an immutable R03 EXP-ID. B07 remains
`BLOCKED_BY_B04A` until the corrected Mini is complete and reviewed.

## Reviewer acceptance

Codex independently reviewed the implementation and reran the required focused,
historical B04, B04A integration, protocol, smoke, compile, link, and diff checks.
The correction is accepted on 2026-08-30. This acceptance covers only the
reload-consistency defect; it does not authorize R03 and does not accept R02.
