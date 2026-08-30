# TASK-SLP-B04A-EXPERIMENT-IDENTITY-CARRIER-FIX-v0.1

**Status:** `IDENTITY_FIX_ACCEPTED / GPU_R03_NOT_AUTHORIZED`
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

### 4. Git object ID

- A real Git object ID is required at every formal B04A identity carrier.
- The frozen B04A identity contract pins the Git object ID at run start
  (CLI dispatch time); the writer MUST NOT re-resolve.
- `_b04a_identity_block` rejects empty, whitespace, the legacy
  `unresolvable_git_commit` sentinel, non-hex, and wrong-length (not 40 or
  64 hex characters) `git_commit` values (R04 ITERATE: fail-closed).
- `_resolve_git_identity` raises `MiniProtocolError` when the repository's
  Git HEAD cannot be resolved, when the working-tree state cannot be read,
  or when the resolved value is not a real Git object ID (R04 ITERATE).

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
8. The CLI freezes the run identity context at dispatch time
   (R04 ITERATE: single run identity source).  All carriers — normal
   result, checkpoint, run bundle, post-validation FAILED / STOPPED —
   must use that frozen context.  The exception handler MUST NOT
   re-resolve Git identity at exception time.
9. `_resolve_git_identity` raises `MiniProtocolError` when the
   repository's Git HEAD cannot be resolved (R04 ITERATE: no sentinel
   fallback).  When the resolver fails BEFORE the dispatch, the CLI
   returns 2 and does not write any file in the output directory.

## Files allowed to change

- `scripts/run_slp8_region_mini.py`
- `scripts/smoke_b04a_runner_integration.py` — synthetic carrier
  propagation through the smoke integration script; the R02 ITERATE proved
  that synthetic checkpoint identity must reach the same `run_mini_b04a`
  path the real B01 use, and the smoke script is the only place where the
  synthetic identity is constructed.
- `src/topper_perception/neural/slp8_region_mini.py`
- `src/topper_perception/neural/slp8_region_resume.py` — checkpoint/resume
  identity schema; `CheckpointIdentity` and `identity_from_dict` live here,
  and resume must reject drift on the seven frozen fields including
  `git_commit` / `git_dirty` / `split_sha256`. The R01/R02 iterations
  required adding those fields to the dataclass and the fail-closed loader
  in this module, so this file is now an explicit part of the contract.
- `tests/test_b04a_runner_integration.py`
- `tests/test_slp8_region_mini.py` only when checkpoint/resume identity coverage
  genuinely requires it
- `scripts/validate_b04a_protocol.py` and its focused test only if the frozen
  identity Gate cannot otherwise be validated
- `docs/tasks/TASK_SLP_B04A_EXPERIMENT_IDENTITY_CARRIER_FIX_v0.1.md`
- `docs/stage_reports/S2_B04A_EXPERIMENT_IDENTITY_CARRIER_FIX_v0.1.md`
- `docs/PROJECT_STATUS.md`
- `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`

The two explicit additions (`slp8_region_resume.py` and
`smoke_b04a_runner_integration.py`) are the R04 ITERATE scope-extension
declared by the Reviewer: without them the identity plumbing that the
contract demands cannot be implemented in a single coherent module, and
the Reviewer would otherwise correctly reject the R03 handoff as having
modified contract-out-of-scope files.

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
- `_b04a_identity_block` rejects empty / whitespace / sentinel / non-hex /
  wrong-length `git_commit` (R04 ITERATE: 5 dedicated test cases).
- `_resolve_git_identity` raises `MiniProtocolError` on a non-resolvable
  repository (R04 ITERATE).
- The CLI freezes the run identity context at dispatch time; the
  post-validation FAILED carrier uses the dispatch-time frozen value
  rather than a re-resolved value (R04 ITERATE).
- The CLI returns 2 and writes no output file when the resolver fails
  before the dispatch (R04 ITERATE).
- Resume rejects EXP-ID/data-manifest/model-version drift.
- Real B01 post-validation failure test: a structurally complete B01 freeze
  fixture loaded with `load_test=False` (TEST labels remain inaccessible), an
  explicit Owner EXP-ID, a
  post-training exception injected AFTER the data contract has been
  verified, and assertions on `FAILED.json` / `status.json` (R04
  ITERATE: same seven required identity fields; `data_manifest_sha256`
  equals the on-disk `freeze_manifest.json` file SHA; `git_commit`
  equals the dispatch-time frozen value; `TEST=0`).
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
- the actual task-contract diff (especially the
  `Files allowed to change` additions for `slp8_region_resume.py`
  and `smoke_b04a_runner_integration.py`);
- exact files changed;
- design of identity propagation and why no fallback remains;
- an explicit TEST=0 invariant: how the real B01 post-validation
  failure test never reads any TEST labels, never enables
  `enable_test_access`, and never calls
  `load_b01_freeze_tables(..., load_test=True)`;
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

Codex independently accepted R05 after rerunning the targeted and regression
suites and auditing a written synthetic DONE bundle. This code correction may
now be merged. After merge, freeze the new `main` Git SHA and create a separate
R03 run-preparation/Owner-authorization record with a new EXP-ID. This acceptance
does not authorize GPU execution or TEST access. B07 remains `BLOCKED_BY_B04A`.
