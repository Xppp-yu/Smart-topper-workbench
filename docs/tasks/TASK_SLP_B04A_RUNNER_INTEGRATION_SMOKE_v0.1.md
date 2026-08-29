# TASK-SLP-B04A-RUNNER-INTEGRATION-SMOKE-v0.1

**Status:** `RUNNER_INTEGRATION_ACCEPTED / GPU_MINI_NOT_AUTHORIZED`
**Stage:** S2-B04A (runner integration + CPU synthetic smoke)
**Branch:** `codex/task-slp-b04a-runner-integration-smoke-v0.1`
**Date:** 2026-08-29

> ⚠ This task advances the B04A stage from
> `IMPLEMENTATION_SMOKE_ACCEPTED` to `RUNNER_INTEGRATION_ACCEPTED` after Codex Reviewer R03 independent acceptance.
> It does **not** authorize a real GPU Mini, does **not** open the B07
> Full protocol, and does **not** read TEST rows.  The next gate
> (`B04A-MINI-RUN`) requires a separate Owner authorization.

## Objective

Make the existing B04 PM-only Region Mini runner accept the frozen
B04A architecture expansion Mini config and execute its 3 candidates
× 3 seeds orchestration path on a small synthetic CPU dataset,
**without breaking the historical B04 protocol**, **without
modifying the B04A frozen config**, and **without** changing the
B04A R03 protocol contract (threshold, seeds, augmentation, budget,
candidate list).

## Why now

`TASK-SLP-B04A-IMPLEMENTATION-SMOKE-v0.1` (R03 2026-08-29) left the
B04A orchestrator integration explicitly as a follow-up:

> 现有 B04 runner 拒绝 B04A config（实测 `ConfigValidationError`：
> `B04_CANDIDATE_NAMES` 不含 B04A 候选名 + `task_id` 不匹配），因此
> runner integration **未完成**，属于下一个独立 TASK。

This task closes that gap.  The B04A implementation is now wired
into the same Mini runner with a **clear protocol/profile
dispatch**, so B04 and B04A share the runner infrastructure but
each has its own contract, validator, orchestrator, and per-seed
hard gates.

## Prerequisites

* B01 freeze tables (A06 SHA `024f5abe...`, freeze manifest core
  SHA `3c789995...`) — **read-only**; not loaded by this task.
* B02 baseline `0.205644` (history unchanged).
* B04 R05 results `SmallUNet=0.439625` / `TinyFCN=0.051631`
  (history unchanged).
* B04A R03 protocol (frozen) — config, candidate set, seeds,
  threshold, augmentation, budget.
* B04A implementation smoke (Implementation+Smoke, 2026-08-29)
  — three candidates registered in `MODEL_REGISTRY`, parameter
  counts frozen at 118,121 / 120,809 / 53,449.
* `tests/test_b04a_implementation.py` (79/79 passing) and
  `tests/test_b04a_protocol_validator.py` (50/50 passing) — both
  unchanged by this task.

## Hard implementation contract

### 1. B04 / B04A configuration identity separation

* Do not simply delete the existing `TASK_ID` check.
* Establish a clear protocol / profile dispatch: the validator
  inspects `config_version` and routes to the B04 or B04A
  protocol-specific contract.
* B04 configs continue to accept only the historical B04
  candidates (`slp8_tiny_fcn_v0.1`, `slp8_small_unet_v0.1`).
* B04A configs accept only the three frozen active candidates
  (`slp8_small_unet_v0.1`, `slp8_resunet_lite_v0.1`,
  `slp8_deeplabv3plus_lite_v0.1`).
* Unknown `task_id`, mixed B04/B04A candidates, duplicate
  candidates, missing active candidates, and forbidden
  candidates (TinyFCN, SegFormer-B0) all **fail closed**.
* Do not modify the B04A frozen JSON config to accommodate the
  old runner.

### 2. B04A seeds

* Read the frozen `[42, 123, 2026]` exactly.
* No silent collapse to a single seed.
* No seeds outside the registered set.
* Synthetic integration smoke may use a reduced per-candidate
  epoch / sample budget, but the three-seed orchestration
  semantics are preserved.

### 3. `all_seeds_must_succeed`

* Any per-seed FAILED / STOPPED / non-finite metric / class
  collapse / worst-subject hard gate / per-region hard gate
  violation flips the entire candidate to INFEASIBLE.
* Computing `macro_iou_mean` over only the surviving seeds
  (e.g. 2 of 3) is forbidden.
* `macro_iou_mean` is computed only when all three registered
  seeds pass every hard gate.

### 4. Candidate-level decision (B04A)

| Feasible | Action |
| --- | --- |
| 0 | `MINI_NOT_FEASIBLE` (no candidate advances) |
| 1 | Advance the single feasible candidate |
| 2 | Advance both feasible candidates; B04A does not pick a champion |
| 3 | Advance top 2 by `macro_iou_mean`; near-tie tiebreak (prefer simpler when `\|diff\| < 0.02`) |

The smoke must not produce a real performance ranking.

### 5. Resource budget

* Per-candidate budget (cumulative over 3 seeds) = 45 min.
* Total budget (cumulative over 3 candidates × 3 seeds) = 135 min.
* Resume must restore the candidate-level and total-level
  accumulators; double-counting wall time after restart is
  forbidden.

### 6. Identity, checkpoint, and output

* At minimum the artifacts carry: `task_id`, `EXP-ID`,
  `config_sha256`, `git_commit`, `candidate`, `model_version`,
  `seed`.
* Checkpoint / reload / resume must verify the identity.
* No overwrite of an existing successful or failed artifact.
* Failure paths must leave an auditable state.
* DONE / FAILED / STOPPED must be mutually exclusive.
* Identity fields in JSON, CSV, checkpoint, and log must agree.

### 7. TEST = 0

* No TEST row appears in the runner's training objects.
* `enable_test_access(...)` is never called.
* The static declaration of `test_access.kind = "declarative_policy"`
  is not a runtime counter; it is a contract statement.
* The synthetic smoke never reads real B01 freeze tables or any
  raw data.

### 8. Backward compatibility

* The historical B04 config and B04 mini tests must continue to
  pass without modification.
* Do not rewrite B04 R05 historical `EXP-ID`, numerical
  results, candidate decision, or outputs.
* Do not modify the B04A-frozen threshold, seeds, augmentation,
  budget, or candidate list.
* Do not modify the already-accepted SmallUNet, ResUNet-lite,
  or DeepLabV3+-lite model structure or parameter counts.

## Files allowed to change

* `src/topper_perception/neural/slp8_region_mini.py` (the runner
  module: protocol dispatch, B04A orchestrator, B04A bundle
  writer, B04A hard gates, identity helpers).
* `scripts/run_slp8_region_mini.py` (add
  `--synthetic-cpu-smoke-b04a` and `--no-write` flags; keep B04
  paths unchanged).
* `scripts/smoke_b04a_runner_integration.py` (new dedicated
  B04A smoke with `--no-write` / `--force` / `--output` /
  `--output-dir` / `--budget-override-seconds`).
* `tests/test_b04a_runner_integration.py` (new focused test
  module).
* `docs/tasks/TASK_SLP_B04A_RUNNER_INTEGRATION_SMOKE_v0.1.md` (this
  file).
* `docs/stage_reports/S2_B04A_RUNNER_INTEGRATION_SMOKE_v0.1.md`
  (new stage report).
* `docs/PROJECT_STATUS.md` (B04A row update only).
* `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md` (B04A section + next
  Gate).

## Out of scope

* Real B01 GPU Mini run (`B04A-MINI-RUN`).
* B07 Full protocol (still `BLOCKED_BY_B04A`).
* Any modification of the B04A R03 protocol, the B04A JSON
  config, the B01 freeze tables, or the B04 historical outputs.
* Modifying the historical B04A implementation smoke
  (`scripts/smoke_b04a_implementation.py`).
* Modifying `tests/test_b04a_implementation.py` or
  `tests/test_b04a_protocol_validator.py` (those pin the
  implementation + protocol; this task pins the **runner
  integration** on top).
* Modifying the `MODEL_REGISTRY` or the candidate classes
  (Smoke uses the registered builder names; new candidate
  entries are forbidden).

## Required tests

Focused tests in `tests/test_b04a_runner_integration.py`
covering at minimum:

1. B04 config still accepted by the dispatched validator.
2. B04A config now accepted by the dispatched validator.
3. Unknown `config_version` rejected fail-closed.
4. B04/B04A candidate mix-up rejected fail-closed.
5. `slp8_tiny_fcn_v0.1` in the B04A active set rejected
   fail-closed.
6. `slp8_segformer_b0_v0.1` promoted out of `DEFERRED` in B04A
   rejected fail-closed.
7. Exact three seeds `B04A_SEEDS = (42, 123, 2026)` enforced
   (length, contents, no silent collapse to 1 seed).
8. Any per-seed failure (FAILED / STOPPED / non-finite /
   class collapse / worst-subject floor / per-region floor)
   flips the candidate to INFEASIBLE; the
   `partial_seed_mean` is forbidden.
9. 0 / 1 / 2 / 3-feasible advance decision rules; near-tie
   tiebreak (`|diff| < 0.02`) prefers the simpler model.
10. Resource budget constants (45 / 135 / 8192); budget
    accumulator `restore` carries the candidate-level seconds
    forward after restart.
11. Identity mismatch on resume raises `ResumeIdentityError`;
    matching identity resumes cleanly.
12. Existing output directory (with `DONE.json` / `FAILED.json` /
    `STOPPED.json` or any non-`.gitkeep` file) refuses to be
    overwritten; the three terminal files are mutually exclusive.
13. TEST = 0: the B04A mini runner and the smoke script never
    import or call `enable_test_access` / `TestLeakageError` /
    `load_b01_freeze_tables` / `compute_class_stats(ml_split="test")`.
14. B04 backward compatibility: the historical B04 mini tests
    (`tests/test_slp8_region_mini.py` 15+ tests covering
    `TestSmallUnetArchitecture` / `TestCandidateRegistry` /
    `build_synthetic_dataset` / `TestPredict` and
    `tests/test_slp8_region_models.py`) all keep passing.

## Smoke policy

* CPU only; torch `2.13.0+cpu` is the host build.
* Synthetic / tiny dataset (`n_train_samples=4`, `n_val_samples=2`,
  `seed=42`).
* Per-candidate epoch budget = 1 (protocol budget remains 30).
* `all_seeds_attempted=True` after the smoke (every
  candidate × seed attempted and recorded).
* `any_seed_feasible=False` for the synthetic smoke (the
  synthetic data is too small to beat the B04A threshold
  `0.355644`; the smoke verifies the orchestration path, not
  the model performance).
* The script supports `--no-write` (prints one line,
  `B04A_SMOKE_NO_WRITE ...`) for the no-write audit, and
  `--force` / `--output` / `--output-dir` for the writing path
  (refuses to overwrite an existing populated directory by
  default).

## Required commands

```powershell
uv run python -m pytest tests/test_b04a_runner_integration.py -q
uv run python -m pytest tests/test_slp8_region_mini.py -q
uv run python -m pytest tests/test_b04a_implementation.py tests/test_b04a_protocol_validator.py tests/test_slp8_region_models.py -q
uv run python scripts/validate_b04a_protocol.py configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json
uv run python scripts/smoke_b04a_runner_integration.py --no-write
uv run python scripts/check_markdown_links.py docs/tasks/TASK_SLP_B04A_RUNNER_INTEGRATION_SMOKE_v0.1.md docs/stage_reports/S2_B04A_RUNNER_INTEGRATION_SMOKE_v0.1.md
uv run python -m py_compile src/topper_perception/neural/slp8_region_mini.py scripts/run_slp8_region_mini.py scripts/smoke_b04a_runner_integration.py tests/test_b04a_runner_integration.py
git diff --check
```

## Acceptance criteria

* The B04A config `configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json`
  passes `validate_mini_config` and yields a `MiniConfig` with
  `protocol="B04A"`, `seeds=(42, 123, 2026)`, and
  `candidates=(slp8_small_unet_v0.1, slp8_resunet_lite_v0.1, slp8_deeplabv3plus_lite_v0.1)`.
* The B04 config keeps the historical `protocol="B04"`,
  `seeds=(42,)`, `candidates=(slp8_tiny_fcn_v0.1, slp8_small_unet_v0.1)`.
* The 58 focused runner-integration tests pass.
* The B04 mini smoke (`tests/test_slp8_region_mini.py`) keeps
  its historical pass count.
* The protocol validator (`scripts/validate_b04a_protocol.py`)
  still returns 30 OKs / 0 errors for the B04A config (the
  runner did not touch the protocol contract).
* The B04A smoke script reports a single
  `B04A_SMOKE_NO_WRITE ...` line in `--no-write` mode and writes
  the full run-level bundle (`manifest.json`,
  `resolved_config.json`, `input_manifest_hashes.json`,
  `environment.json`, `status.json`,
  `candidate_decision.json`, `budget_report.json`,
  `logs/run.log`, per-seed `checkpoints/<candidate>/seed_<seed>/{last,best}.pt`)
  in the writing path.
* The terminal file is one of `DONE.json` / `FAILED.json` /
  `STOPPED.json` (mutually exclusive).
* `git diff --check` is clean.
* Before Reviewer acceptance, the B04A stage name in the handoff is
  `RUNNER_INTEGRATION_READY_FOR_REVIEW`; after Codex Reviewer R03 acceptance,
  the repository status is `RUNNER_INTEGRATION_ACCEPTED / GPU_MINI_NOT_AUTHORIZED`.
* The implementation handoff itself makes no acceptance claim; only the Reviewer
  may record acceptance. No one may claim `GPU_MINI_AUTHORIZED`, `MINI_COMPLETE`,
  or `B07_READY` from this task.

## Fail-closed conditions

* The runner refuses to overwrite a populated output directory
  unless `--force` is set; the collision check runs **before**
  any file is written.
* The B04A mini refuses to load `slp8_tiny_fcn_v0.1` or
  `slp8_segformer_b0_v0.1` in the active candidate set, even
  if the config erroneously sets `role="new_candidate"`.
* The B04A mini refuses to use any seed outside
  `B04A_SEEDS = (42, 123, 2026)`.
* The B04A mini refuses to compute `macro_iou_mean` over a
  partial seed subset.
* The CLI refuses any config whose `config_version` is not
  `slp8_region_mini_v0.1` (B04) or
  `slp8_pm_architecture_expansion_mini_v0.1` (B04A).

## Prohibited conclusions

* ❌ "B04A Mini complete" or "B04A candidates compared".
* ❌ "B04A advances to B07" or "B07 ready".
* ❌ "GPU Mini authorized".
* ❌ "Test result X is better than Y" (smoke is not a benchmark).
* ❌ "Architecture A beats architecture B" (no real Mini).
* ❌ "Suitable for product / hardware / comfort / medical /
  overnight / airbag" (GT is `NOT_REVIEWED` synthetic).
* ❌ Modifying the frozen B04A config, threshold, seeds,
  augmentation, budget, or candidate list.
* ❌ Re-writing B04 R05 historical values or outputs.

## Reviewer checklist

* [ ] B04 / B04A protocol dispatch is implemented (no scattered
  `if/elif` on `task_id`).
* [ ] B04 config still passes the validator unchanged.
* [ ] B04A config now passes the validator and builds the
  B04A `MiniConfig` with frozen seeds and candidates.
* [ ] Unknown `config_version` is rejected.
* [ ] Mixed B04/B04A candidates, TinyFCN in B04A, duplicate
  candidates, and missing active candidates are all rejected.
* [ ] `all_seeds_must_succeed` is enforced per candidate.
* [ ] 0 / 1 / 2 / 3-feasible decision rules are applied; near-tie
  tiebreak prefers the simpler model.
* [ ] Resource budget is 45 / 135 min; resume restores both
  accumulators.
* [ ] Identity mismatch raises `ResumeIdentityError`.
* [ ] Output directory collision refused; terminal files
  mutually exclusive.
* [ ] TEST = 0 (no `enable_test_access` / `TestLeakageError` /
  real B01 calls in the smoke path).
* [ ] B04 mini regression suite still passes.
* [ ] `git diff --check` is clean.
* [ ] Pre-review handoff stage name is `RUNNER_INTEGRATION_READY_FOR_REVIEW`;
  Reviewer may record `RUNNER_INTEGRATION_ACCEPTED / GPU_MINI_NOT_AUTHORIZED`.
* [ ] No `GPU_MINI_AUTHORIZED` / `MINI_COMPLETE` / `B07_READY`
  claims.

## Next Gate

`B04A-MINI-RUN` is **BLOCKED** until:

* Owner authorizes a real B01 GPU Mini with explicit
  `EXP-ID` and a separate `TASK-ID`; and
* Codex Reviewer accepts the runner-integration stage as
  `RUNNER_INTEGRATION_ACCEPTED / GPU_MINI_NOT_AUTHORIZED`.

`B07` (Full protocol) remains `BLOCKED_BY_B04A` until the
authorized Mini run completes and at most 1–2 candidates are
advanced by the candidate-level decision rule.
