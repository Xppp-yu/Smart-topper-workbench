# S2 B04A Reload Consistency Fix v0.1

**TASK-ID:** `TASK-SLP-B04A-RELOAD-CONSISTENCY-FIX-v0.1`
**Status:** `ACCEPTED / R02_PRESERVED_FAILED / R03_BLOCKED_BY_IDENTITY`
**Date:** 2026-08-30

## Outcome

The R02 terminal failure was reproduced from the immutable local evidence
archive and traced to a best-versus-final comparison error. The runner loaded
`best.pt` but compared its logits with the live model after training had
continued beyond the best epoch. Seeds with `best_epoch < final_epoch` were
therefore falsely marked `FAILED` even though their full validation prediction
hashes matched after reload.

The correction captures a deterministic CPU probe from the best-epoch model
state before checkpoint serialization, stores its sample indices and logits in
`best.pt`, and compares the independently reloaded best model with that probe.
The existing full-validation prediction-hash comparison remains mandatory.
Missing, malformed, non-finite, shape-inconsistent, or numerically mismatched
probe data fails closed.

## Evidence boundary

The external archive remains read-only and was not extracted into or modified
by the repository:

```text
E:/TeamProjects/autodl-transfer/B04A_R01_R02_EVIDENCE_20260830.tar.gz
size: 26,025,173 bytes
SHA-256: 75b9cd09fbf7214ddca9d0511991419a30cb404f709512a54f3f173029cb6494
```

Independent archive inspection confirmed:

- R02 Git SHA `e8cac7be53593be3151864f27101addef2e17ed2` and `git_dirty=false`;
- terminal state `FAILED`, all 3 candidates x 3 seeds attempted;
- affected seeds reported `hash_match=true` with non-zero best-versus-final
  `max_abs_diff`;
- DeepLabV3+-lite was directionally feasible with reported
  `macro_iou_mean=0.5044011036077056`, but R02 is not accepted and its
  `advanced` field is not a formal B07 admission decision.

## Files changed

- `src/topper_perception/neural/slp8_region_mini.py`
- `tests/test_slp8_region_mini.py`
- `docs/tasks/TASK_SLP_B04A_RELOAD_CONSISTENCY_FIX_v0.1.md`
- `docs/stage_reports/S2_B04A_RELOAD_CONSISTENCY_FIX_v0.1.md`
- `docs/PROJECT_STATUS.md`
- `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`

## Commands run

```text
python -m pytest tests/test_slp8_region_mini.py -k BestEpochReloadProbe -q
  5 passed, 162 deselected

python -m pytest tests/test_slp8_region_mini.py -q
  167 passed

python -m pytest tests/test_b04a_runner_integration.py
  tests/test_b04a_implementation.py
  tests/test_b04a_protocol_validator.py
  tests/test_slp8_region_models.py -q
  247 passed

python scripts/validate_b04a_protocol.py
  configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json
  30 OKs / 0 errors

python scripts/smoke_b04a_runner_integration.py --no-write
  terminal_state=DONE; candidates=3; seeds=3; TEST=0
```

No GPU Mini, Full, or TEST was run by this task.

## Additional blocker found during review

R02's run-level identity carrier does not satisfy the frozen governance Gate:

- `experiment_id` is
  `TASK-SLP-B04A-PROTOCOL-FREEZE-v0.1::<run>::seed=-`, not the
  Owner-authorized R02 EXP-ID;
- `data_manifest_sha256` equals the config SHA instead of a data/freeze
  manifest identity;
- run-level `model_version` is empty.

These are independent of the reload-consistency defect. They are not changed
in this task and require a separate bounded task before R03 authorization.

## Verified

- R01/R02 archive hash and required directory structure.
- R02 false-failure pattern and the exact best-versus-final code path.
- Best-epoch probe survives later live-model updates.
- Corrupted reloaded state is detected.
- Missing and malformed probes fail closed.
- Historical B04 and focused B04A regression suites pass.
- Frozen B04A config, thresholds, candidates, seeds, and TEST=0 are unchanged.

## Inferred

- The R02 SmallUNet and ResUNet candidate decisions are invalidated by the
  reload-check implementation defect, not demonstrated model infeasibility.
- R02 remains useful as defect reproduction and directional evidence only.

## Unverified

- Corrected behavior on a real CUDA/B01 run.
- Formal R03 candidate feasibility and advancement.
- Any B07 Full result or TEST result.

## Limitations

- Legacy checkpoints without the new reload probe fail closed under the
  corrected audit; R02 is intentionally not resumed or rewritten.
- Interruption lifecycle and resume/output-directory collision remain separate
  defects and are not addressed here.
- Experiment identity carrier defects remain open.

## Next Gate

1. Independent review of this reload correction.
2. Separate experiment-identity carrier correction and review.
3. Freeze the resulting Git SHA and run-preparation record.
4. Owner separately authorizes a fresh immutable R03 EXP-ID.
5. R03 must complete and pass Reviewer audit before B07 can open.

B07 remains `BLOCKED_BY_B04A`.

## Reviewer decision

`ACCEPT` for `TASK-SLP-B04A-RELOAD-CONSISTENCY-FIX-v0.1` on 2026-08-30.
The acceptance is limited to the reload-consistency correction and its tests.
R02 remains `FAILED`; R03 remains unauthorized until the separate experiment
identity carrier correction is implemented and accepted.
