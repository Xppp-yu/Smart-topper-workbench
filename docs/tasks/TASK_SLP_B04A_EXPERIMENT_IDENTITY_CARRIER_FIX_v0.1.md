# TASK-SLP-B04A-EXPERIMENT-IDENTITY-CARRIER-FIX-v0.1

**Status:** `READY_FOR_IMPLEMENTATION / GPU_NOT_AUTHORIZED`
**Stage:** S2-B04A defect correction before R03 run preparation
**Owner:** Project Owner
**Implementer:** MiniMax Code
**Reviewer:** Codex
**Branch:** `codex/task-slp-b04a-experiment-identity-carrier-fix-v0.1`
**Date:** 2026-08-30

## Objective

Correct the B04A run-level experiment identity carrier defects observed in the
preserved failed experiment
`EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-AUTODL-R02`.

The runner must carry the Owner-authorized EXP-ID, the exact B01 freeze manifest
file SHA-256, and a truthful non-empty model-version identity through every
formal run artifact. Invalid, absent, inconsistent, or resumed-with-different
identity must fail closed before training and, where possible, before creating
the requested output directory.

This task changes identity plumbing and tests only. It does not change any
candidate, seed, metric, threshold, data split, augmentation, optimizer,
resource budget, reload-consistency behavior, or R01/R02 artifact.

## Incident evidence

The immutable evidence archive is outside the repository:

```text
E:/TeamProjects/autodl-transfer/B04A_R01_R02_EVIDENCE_20260830.tar.gz
size: 26,025,173 bytes
SHA-256: 75b9cd09fbf7214ddca9d0511991419a30cb404f709512a54f3f173029cb6494
```

R02 run-level artifacts incorrectly recorded:

```text
experiment_id = TASK-SLP-B04A-PROTOCOL-FREEZE-v0.1::<run>::seed=-
data_manifest_sha256 = 74230e...aab3fa  # equals config_sha256
model_version = ""
```

R02 remains `FAILED` and must not be rewritten, resumed, promoted, or used as a
formal B07 admission decision.

## Frozen identity semantics

### 1. Experiment ID

- Add an explicit CLI input named `--experiment-id`.
- A real B01 run with `--run-authorized` requires a non-empty Owner-supplied
  EXP-ID. No TASK-ID-derived fallback is allowed.
- Every run-, candidate-, and seed-level artifact must carry the exact same
  supplied EXP-ID. Candidate and seed remain separate identity fields; do not
  append them to `experiment_id`.
- Resume requires exact EXP-ID equality with the saved run/checkpoint identity.
- Missing, blank, malformed, or mismatched EXP-ID fails closed.

### 2. Data manifest SHA-256

- For a real B01 run, `data_manifest_sha256` is the lowercase SHA-256 of the
  exact on-disk `freeze_manifest.json` file already validated by the B01 input
  gate (`freeze_manifest_file_sha256`).
- Preserve the separate B01 core-manifest consistency check and its expected
  core SHA. Do not conflate the file hash with the core hash.
- Never fall back to `config_sha256`, TASK-ID, split SHA, or an empty string.
- For synthetic smoke only, use a deterministic SHA-256 of the canonical
  synthetic dataset-manifest payload and keep the run explicitly marked
  synthetic. Synthetic identity must never be accepted as a real B01 identity.

### 3. Model version

- Candidate- and seed-level artifacts use the candidate builder's exact model
  version, as today.
- Run-level multi-candidate artifacts use this deterministic string grammar in
  frozen config order:

```text
multi_candidate[<candidate_1>,<candidate_2>,...,<candidate_n>]
```

- An empty `model_version` is prohibited at every carrier.
- Candidate order must come from the validated frozen config and must not be
  sorted or inferred from completed results.

## Required carrier coverage

Verify the seven frozen identity fields wherever required by the B04A protocol:

```text
experiment_id
git_commit
git_dirty
config_sha256
data_manifest_sha256
split_sha256
model_version
```

Coverage includes at minimum:

- run-level JSON files and the first line of `logs/run.log`;
- candidate aggregate JSON files;
- per-seed JSON and identity sidecars;
- checkpoint `identity` dictionaries in both `best.pt` and `last.pt`;
- per-seed log first lines;
- CSV identity sidecars;
- resume identity checks.

All carriers for one real run must agree on EXP-ID, Git SHA/dirty state, config
SHA, data-manifest SHA, and split SHA. Model version differs only as explicitly
defined above for run-level multi-candidate versus candidate/seed-level scope.

## Fail-closed and lifecycle rules

1. Validate `--experiment-id` before any real B01 data read or training.
2. An invalid authorization/identity request must not create or mutate the
   requested output directory.
3. Existing output collision behavior remains unchanged and auditable.
4. Resume must reject EXP-ID, config SHA, data-manifest SHA, split SHA, Git SHA,
   candidate, seed, or model-version drift.
5. `TEST=0` remains mandatory; do not add or enable TEST access.
6. Do not add `--force`, overwrite, post-hoc repair, or legacy fallback paths.
7. Do not modify the accepted reload-probe correction from PR #23 except for
   narrowly required identity plumbing around its checkpoints.

## Files allowed to change

- `scripts/run_slp8_region_mini.py`
- `src/topper_perception/neural/slp8_region_mini.py`
- `tests/test_b04a_runner_integration.py`
- `tests/test_slp8_region_mini.py` only when checkpoint/resume identity coverage
  genuinely requires it
- `scripts/validate_b04a_protocol.py` and its focused test only if the frozen
  identity Gate cannot otherwise be validated
- `docs/tasks/TASK_SLP_B04A_EXPERIMENT_IDENTITY_CARRIER_FIX_v0.1.md`
- `docs/stage_reports/S2_B04A_EXPERIMENT_IDENTITY_CARRIER_FIX_v0.1.md`
- `docs/PROJECT_STATUS.md`
- `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`

Do not modify the frozen B04A candidate/config values merely to make tests pass.
Any required protocol-text clarification must be reported to the Reviewer before
editing the frozen config.

## Required tests

- Real B01 CLI refuses missing/blank `--experiment-id` without output mutation.
- Supplied EXP-ID is identical across run, candidate, seed, checkpoint, sidecar,
  and log carriers.
- Real `data_manifest_sha256` equals the exact freeze-manifest file hash and is
  demonstrably different from config SHA in the fixture.
- Synthetic smoke has a deterministic synthetic-manifest hash and cannot be
  mistaken for real B01.
- Run-level multi-candidate model version follows the frozen grammar and config
  order; candidate/seed model versions remain exact builder versions.
- Empty model version fails closed.
- Resume rejects EXP-ID/data-manifest/model-version drift.
- Existing output-collision and no-authorization no-write tests remain green.
- Accepted best-epoch reload-probe regression remains green.
- Historical B04 and focused B04A suites remain green.
- Protocol validator remains `30 OKs / 0 errors` unless a separately reviewed
  contract update intentionally changes the frozen count.
- No-write B04A smoke completes with `TEST=0`.
- `python -m py_compile`, Markdown link check, and `git diff --check` pass.

## Required handoff

MiniMax Code must report:

- TASK-ID and branch/worktree;
- exact files changed;
- design of identity propagation and why no fallback remains;
- commands actually run and exact pass/fail counts;
- generated synthetic artifacts inspected and identity consistency evidence;
- checks not run marked `NOT RUN`;
- known limitations and prohibited conclusions;
- `git status --short --branch` and `git diff --check`;
- no commit, push, PR, GPU run, TEST access, or merge unless separately
  authorized by the Owner.

## Prohibited conclusions

- This task does not make R02 valid or accepted.
- This task does not authorize R03, GPU Mini, B07 Full, or TEST.
- No model advances from this code correction.
- R02 directional metrics remain non-formal evidence only.
- No product, hardware, comfort, medical, overnight, or airbag-control claim.

## Next Gate

Codex independently reviews the exact worktree, reruns targeted and regression
tests, inspects generated identity carriers, and returns `ACCEPT` or `ITERATE`.
Only after acceptance and merge may a new Git SHA be frozen and a separate R03
run-preparation/Owner-authorization record be created. B07 remains
`BLOCKED_BY_B04A`.
