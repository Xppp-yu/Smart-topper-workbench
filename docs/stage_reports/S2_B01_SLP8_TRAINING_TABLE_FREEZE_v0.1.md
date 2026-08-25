# S2_B01_SLP8_TRAINING_TABLE_FREEZE_v0.1

**TASK-ID**: `TASK-SLP-B01-SLP8-TRAINING-TABLE-FREEZE-v0.1`
**Branch**: `codex/task-slp-b01-training-table-freeze-v0.1`
**HEAD**: `725d0aaafdd9f0c502e22fa6c75ff700bdf25bc0` (at task start; B01 changes are uncommitted)
**Date**: 2026-08-24
**Status**: `DONE_WITH_LIMITATIONS` — implementation, unit tests, real-data
build+validator, and Codex independent review have completed.

---

## 1. Scope and contract

This task freezes the B01 training tables for the SLP8 pressure-only region
segmentation route.  It does **not** train a model, run GPU/Mini/Full
experiments, or modify the original SLP8 dataset.  It accepts the
SLP_8Region_Pressure_VAL_v1.1 (4,590 samples, 102 danaLab, uncover only,
8-region) as the project reference GT and the A06 subject-level split
(81/10/11 danaLab subjects → 3,645/450/495 samples) as the subject-level
assignment, and produces:

* `train_manifest.csv` + `train_manifest.jsonl` (3,645 samples)
* `val_manifest.csv` + `val_manifest.jsonl` (450 samples)
* `test_manifest.csv` + `test_manifest.jsonl` (495 samples)
* `freeze_manifest.json` (top-level freeze contract, content-addressed)
* `normalization_stats.json` (TRAIN-only pressure normalisation)
* `train_class_stats.json` / `val_class_stats.json` (per-split class coverage,
  TEST kept structural-only)
* `dataset_card.md` (human-readable provenance + usage limits)

Every manifest uses **relative paths only**; no absolute Windows / POSIX
path or `..` segment is allowed in any artifact, and every path is verified
to be strictly inside the dataset root (no same-prefix sibling escapes).

## 2. Files added / modified

| Path | Change | Purpose |
|---|---|---|
| `src/topper_perception/io/slp8_training_table_freeze.py` | NEW | B01 module: A06 / source-manifest loaders, freeze row builder, manifest CSV/JSONL I/O, TRAIN-only normalization, TEST-access guard, class-stats, freeze manifest, dataset card, full orchestrator (`Slp8TrainingTableFreezer`) |
| `scripts/build_slp8_training_tables.py` | NEW | CLI builder (no hard-coded paths; dataset root and A06 split are CLI flags) |
| `scripts/validate_slp8_training_tables.py` | NEW | CLI validator (11 sections; fail-closed; deterministic rebuild) |
| `tests/test_slp8_training_table_freeze.py` | NEW | 82 unit + integration tests (2 real-data tests gated on env vars) |
| `data/processed/slp8_training_tables_v0.1/` | NEW (gitignored) | Output directory holding the frozen artifacts |
| `docs/stage_reports/S2_B01_SLP8_TRAINING_TABLE_FREEZE_v0.1.md` | NEW | This report |
| `docs/PROJECT_STATUS.md` | UPDATED | S2 SLP table updated; data-boundary section cross-links B01 |
| `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md` | UPDATED | B01 accepted as `DONE_WITH_LIMITATIONS`; B02/B03 released to `READY` |
| `docs/SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md` | UPDATED | B01 cross-link and a new B01 ↔ B02/B03 dependency note |
| `COLLABORATION_WORKFLOW.md` | UPDATED | Test access policy and B01 contract are referenced |

## 3. Provenance and limitations (mirrored in `dataset_card.md`)

* `annotation_provenance = V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`
* `source_review_status  = NOT_REVIEWED`
* **NOT** human pixel-level semantic masks
* **NOT** medical, skin-interface stress, or product ground truth
* Pressure values are **raw PMarray response semantics**, NOT kPa
* **danaLab only**, **uncover only**; do not extrapolate to cover1/cover2
* Do not use for self-developed topper product effect claims, comfort,
  medical, or hardware validation

These are recorded in every B01 manifest row (`annotation_provenance`,
`source_review_status`) and re-stated in the dataset card and the
freeze manifest's `expected_provenance` / `expected_review_status` fields.
The 10-region polygon schema (`slp_region_annotation_v0.1`) is not used
in this task and is not mixed with the 8-region data.

## 4. Determinism and content-addressing

* Manifest rows are sorted by `sample_id` before writing (CSV, JSONL,
  and SHA-256 hashing) so byte-level content is stable.
* SHA-256 digests are computed over canonical JSON
  (`json.dumps(..., sort_keys=True, separators=(",", ":"))`).
* The `freeze_manifest.json` is split into `core` (content-addressed;
  includes the per-split manifest SHAs and the normalization stats SHA)
  and `meta` (observational; build timestamp, builder version, Git SHA,
  platform, Python version).
* The `normalization_stats.json` includes a `stats_sha256` that excludes
  `fitted_at_utc` so the contract hash is reproducible.

## 5. TEST access policy

A module-level guard blocks any read of TEST label/onehot or computation
of TEST class statistics in development mode.  The guard is opt-in:

```python
from topper_perception.io.slp8_training_table_freeze import (
    enable_test_access, disable_test_access, TestLeakageError,
)

# Development (default): TEST label/onehot access is denied.
try:
    compute_class_stats(test_rows, dataset_root, ml_split="test")
except TestLeakageError:
    pass  # expected

# Opt-in for a final-evaluation runner:
enable_test_access(purpose="final_evaluation")
stats = compute_class_stats(test_rows, dataset_root, ml_split="test")
disable_test_access()
```

Only the literal purpose `final_evaluation` is accepted; any other
string raises `TestLeakageError`.

**Object-level isolation:** ``load_b01_freeze_tables`` by default does **not**
read the TEST manifest CSV — the returned ``B01FreezeTables`` object has
``_test_rows = None``.  Enabling ``enable_test_access(purpose="final_evaluation")``
does not retroactively grant TEST access to an already-loaded object;
the caller must reload with an explicit ``load_test=True`` call to get
TEST rows::

```python
tables = load_b01_freeze_tables(output_dir)   # default: no TEST rows
enable_test_access(purpose="final_evaluation")
try:
    tables.test_rows          # raises TestLeakageError: "TEST rows are not present"
    tables.all_rows_with_test_opt_in()  # raises TestLeakageError: "reload with load_test=True"
finally:
    disable_test_access()

# Correct pattern:
enable_test_access(purpose="final_evaluation")
try:
    tables = load_b01_freeze_tables(output_dir, load_test=True)
    rows = tables.test_rows   # OK
finally:
    disable_test_access()
```

Structural checks on TEST (sample count, subject count, sample_id uniqueness,
path format, file existence, hash / contract consistency) remain available
via the ``freeze_manifest`` without loading row objects.  No TEST IoU, Dice,
per-class pixel distribution, or any model metric is computed anywhere in B01.

## 6. TRAIN-only normalization

Pressure normalization statistics are fitted on TRAIN subjects only.
The fit walks every TRAIN row, loads the pressure array with
`np.load(..., allow_pickle=False)`, and computes:

* sample count, total pixel count
* finite / non-finite pixel counts (must be 0 for SLP8 v1.1)
* global min, max, mean, std

The method recorded in the artifact is
`raw_passthrough_with_minmax_reference` and the semantic is
`raw_pmarray_response` (never kPa; a misspelled legacy alias
is accepted on read for backward compatibility but is no
longer written by the B01 module).  Epsilon is fixed at
`1e-12`.  Normalization fits on VAL or TEST are rejected with
`NormalizationContractError`.  NaN/Inf pressure values are rejected
during the fit.

The stats file is content-addressed by its own
`stats_sha256`, and that SHA is recorded in the freeze manifest under
`normalization_stats_sha256`.

## 7. Path safety (fail-closed)

Every path field in every manifest row is checked:

* No absolute Windows path (e.g. `D:\foo`)
* No absolute POSIX path (e.g. `/etc/passwd`)
* No UNC path (e.g. `\\server\share`)
* No `..` segment
* `is_path_within(child, parent)` uses
  `child.resolve().relative_to(parent.resolve())` and explicitly rejects
  the same-prefix sibling case (e.g. `dataset_evil` is not within
  `dataset`).

The validator reproduces all four checks on every row and reports
PASS / FAIL with counts.

## 8. A06 split integration

The A06 split (`slp_subject_split_v0.1.json`) is loaded with the
following contract:

* `schema_version` must equal `slp_subject_split_v0.1`
* The embedded `manifest_sha256` is the SHA-256 of the JSON dump of
  `(subject_id, setting, split)` entries (sorted by `subject_id`,
  `json.dumps(..., sort_keys=True, ensure_ascii=False)`) — the
  A06 generator's own contract.  B01 re-computes it from the parsed
  JSON and refuses to accept the file if the re-computed value
  differs.
* The re-computed SHA must match the canonical A06 SHA
  `024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706`.
* danaLab subject counts must be 81 / 10 / 11 (total 102).
* Each `subject_id` is mapped to exactly one ML split; subjects with
  `ml_split` outside `{train, val, test}` are rejected.
* Subjects in the SLP8 dataset that are not in the A06 split (or vice
  versa) cause fail-closed `SubjectMappingError`.

The B01 freeze manifest records the A06 SHA under
`core.a06_split_sha256`; the validator re-computes it on every
re-validate and aborts on mismatch.

## 9. Commands actually run

> Note: paths prefixed with `E:\TeamProjects\` are local machine paths
> passed to the CLI; they are **not** written into any artifact, the
> dataset card, or the freeze manifest.  The CLI flag-based design lets
> the same code run on any machine without modification.

```powershell
# 1. Confirm worktree / branch / baseline
cd E:\TeamProjects\smarttopper-slp-b01-training-table-freeze
git status
git rev-parse HEAD            # 725d0aaafdd9f0c502e22fa6c75ff700bdf25bc0
git rev-list --left-right --count origin/main...HEAD  # 0 0

# 2. Build frozen training tables on real data
uv run python scripts\build_slp8_training_tables.py `
  --dataset-root "E:\TeamProjects\datasets\smart-topper\SLP2022\SLP\SLP_8Region_Pressure_VAL_v1.1" `
  --a06-split "E:\TeamProjects\smarttopper-slp-a06\data\processed\slp\slp_subject_split_v0.1.json"
# 3645 train / 450 val / 495 test rows written

# 3. Full fail-closed validator (11 sections, including deterministic rebuild)
uv run python scripts\validate_slp8_training_tables.py `
  --dataset-root "E:\TeamProjects\datasets\smart-topper\SLP2022\SLP\SLP_8Region_Pressure_VAL_v1.1" `
  --a06-split "E:\TeamProjects\smarttopper-slp-a06\data\processed\slp\slp_subject_split_v0.1.json"
# ALL CHECKS PASSED in 56.0s

# 4. New B01 unit + integration tests
$env:SLP8_DATASET_ROOT = "E:\TeamProjects\datasets\smart-topper\SLP2022\SLP\SLP_8Region_Pressure_VAL_v1.1"
$env:A06_SPLIT_PATH    = "E:\TeamProjects\smarttopper-slp-a06\data\processed\slp\slp_subject_split_v0.1.json"
$env:B01_FREEZE_OUTPUT_DIR = "E:\TeamProjects\smarttopper-slp-b01-training-table-freeze\data\processed\slp8_training_tables_v0.1"
uv run pytest -q tests\test_slp8_training_table_freeze.py
# 82 passed

# 5. Regression: A09R dataset tests
$env:SLP_A06_SPLIT_PATH = "E:\TeamProjects\smarttopper-slp-a06\data\processed\slp\slp_subject_split_v0.1.json"
uv run pytest -q tests\test_slp_8region_pressure_dataset.py
# 66 passed in 1.46s

# 6. Regression: SLP region annotation schema + canonical adapter + pressure infrastructure + subject split
uv run pytest -q tests\test_slp_region_annotation_schema.py tests\test_slp_canonical_adapter.py tests\test_slp_pressure_infrastructure.py tests\test_slp_subject_split.py
# 221 passed, 1 skipped in 46.74s

# 7. Final whitespace check
git diff --check
# (no whitespace-only errors)
```

## 10. Real-data build result

```
[B01] WROTE 3645 train / 450 val / 495 test rows
[B01]   train manifest: data\processed\slp8_training_tables_v0.1\train_manifest.csv
[B01]   val   manifest: data\processed\slp8_training_tables_v0.1\val_manifest.csv
[B01]   test  manifest: data\processed\slp8_training_tables_v0.1\test_manifest.csv
[B01]   freeze manifest: data\processed\slp8_training_tables_v0.1\freeze_manifest.json
[B01]   normalization stats: data\processed\slp8_training_tables_v0.1\normalization_stats.json
[B01]   dataset card: data\processed\slp8_training_tables_v0.1\dataset_card.md

[B01]   a06 split SHA-256      = 024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706
[B01]   source manifest SHA-256= 59042df9ad4cba9bda644f48ff63849ba109a5e45a880ce9cf43e9c19e19deca
[B01]   train manifest SHA-256 = 4801f0c4130bc52a49245b4930bfa7e871bbbce3f8ec5bc38fed2f238736d044
[B01]   val   manifest SHA-256 = 800c8262049a2694024ddb0c8ffe7a5f1c2d5a79321d95f0923653c2feb7ea71
[B01]   test  manifest SHA-256 = 6c1d7726602dfd7287019d1b455f246f5604bfc352d90bcd908479fbff899c53
[B01]   normalization SHA-256  = 0b1ef18b4769f8b1b47d077cfc4c06c8310c8fff5877a6e44afcd0df2f466c59
[B01]   freeze   manifest core SHA-256 = 3c78999551580fc46ce15229e053798b5e4c9464a5bab27e05130cb319090b1e
```

| Section | Result |
|---|---|
| 1. Manifest structural (4,590 / 4,590 unique sample_ids, 102 subjects, 45 frames each) | PASS |
| 2. Split counts (3,645 / 450 / 495) | PASS |
| 3. Path safety (0 absolute / 0 `..` / 0 escapes) | PASS |
| 4. Source integrity (A06 SHA matches canonical; SLP8 source manifest SHA matches file; per-split manifest SHAs stable across re-reads) | PASS |
| 5. Provenance / review status (all 4,590 rows: `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED` / `NOT_REVIEWED`) | PASS |
| 6. Normalization stats (TRAIN-only fit, method, semantics, dtype, finite count, SHA matches freeze manifest) | PASS |
| 7. Freeze manifest (task_id, source_dataset_id, A06 SHA, source SHA, per-split SHAs, normalization SHA, TEST structural-only, test access policy, meta) | PASS |
| 8. Output contract completeness (CSV/JSONL parity; TRAIN/VAL class stats) | PASS |
| 9. Dataset card (8-region, provenance, NOT_REVIEWED, danaLab only, uncover only, raw PMarray response, NOT kPa, prohibited conclusions) | PASS |
| 10. TEST access policy (default deny, wrong purpose rejected, only `final_evaluation` allowed) | PASS |
| 11. Deterministic rebuild (rebuild train manifest SHA matches; freeze manifest core SHA stable) | PASS |

Total: **ALL CHECKS PASSED in 56.0s** (deterministic rebuild included).

## 11. Normalization statistics summary

| Field | Value |
|---|---|
| n_samples | 3,645 |
| n_pixels | 3,645 × 192 × 84 = 58,786,560 |
| finite_pixel_count | 58,786,560 (all finite) |
| non_finite_pixel_count | 0 |
| global_min | computed from real data (raw PMarray response) |
| global_max | computed from real data |
| global_mean | computed from real data |
| global_std | computed from real data |
| method | `raw_passthrough_with_minmax_reference` |
| epsilon | `1e-12` |
| raw_dtype | `float64` |
| raw_semantics | `raw_pmarray_response` (NOT kPa) |
| fit_split | `train` |
| subject_count | 81 |
| per_subject_count_min / max | 45 / 45 |
| stats_sha256 | `0b1ef18b4769f8b1b47d077cfc4c06c8310c8fff5877a6e44afcd0df2f466c59` |

The exact numeric min / max / mean / std are written to
`normalization_stats.json` (and only to that file) under
`data/processed/slp8_training_tables_v0.1/normalization_stats.json`.
They are gitignored.

## 12. Test results (per suite)

| Suite | Result |
|---|---|
| `tests/test_slp8_training_table_freeze.py` (B01) | 82 passed |
| `tests/test_slp_8region_pressure_dataset.py` (A09R) | 66 passed |
| `tests/test_slp_region_annotation_schema.py` | passed (part of 221) |
| `tests/test_slp_canonical_adapter.py` | passed (part of 221) |
| `tests/test_slp_pressure_infrastructure.py` | passed (part of 221) |
| `tests/test_slp_subject_split.py` | passed (1 skipped: A05 CSV not present in this worktree) |
| `git diff --check` | clean |

The combined executed suite is 369 tests passed: 82 B01 tests plus
287 related regressions (66 A09R + 221 schema/adapter/infra/split),
with 1 skipped (A05 CSV not present in this worktree).

## 13. Verified

| Item | Evidence |
|---|---|
| 4,590 samples fully mapped to A06 splits (3,645/450/495) | Section 1 + 2 of full validator; `n_train`/`n_val`/`n_test` in `FreezeResult` |
| 102 danaLab subjects, 45 frames each, 0 cross-split overlap | Section 1 of full validator; `unique_subjects = 102`, subject-disjointness checks |
| All paths relative + contained + no `..` | Section 3 of full validator; `is_path_within` strictness |
| A06 split SHA matches canonical `024f5abe…` | Section 4 of full validator; re-computed hash from subject-assignment JSON |
| SLP8 source manifest SHA matches file | Section 4 + 7 of full validator |
| Every row has `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED` and `NOT_REVIEWED` | Section 5 of full validator |
| TRAIN-only normalization fit | Section 6 of full validator; `fit_split = "train"` recorded in `normalization_stats.json` |
| Pressure values are raw PMarray response (NOT kPa) | `raw_semantics` field in stats; dataset card |
| Deterministic rebuild produces identical freeze manifest core SHA | Section 10 of full validator; train manifest SHA matches after rebuild |
| TEST access is denied by default and accepted only with `purpose="final_evaluation"` | Section 9 of full validator; module-level `enable_test_access` test |
| TEST class statistics are NOT computed in development mode | `compute_class_stats` raises `TestLeakageError` for `ml_split="test"` without opt-in (test: `TestClassStats::test_compute_test_class_stats_blocked`) |
| `np.load` always uses `allow_pickle=False` | AST scan test `test_all_npload_uses_allow_pickle_false` |
| No hard-coded local paths in the B01 module | AST scan test `test_no_hard_coded_local_paths` |
| All A09R regression tests still pass | 66 / 66 `test_slp_8region_pressure_dataset.py` |
| 221 / 221 schema + adapter + infra + split tests pass | section 12 above |

## 14. Inferred

| Item | Basis |
|---|---|
| A06 subject split is stable across the 102 danaLab subjects (the embedded `manifest_sha256` re-computes to `024f5abe…`) | Re-computation uses the same JSON dump logic the A06 generator uses |
| `dataset_root` and `a06_split_path` can be passed on any machine; the frozen artifact is portable because all paths are dataset-root-relative | All four relative-path checks pass; no absolute paths appear in any B01 output |
| The 2 real-data integration tests pass on the canonical inputs | They read the same freeze manifest the CLI build produces and assert sample/subject counts |
| The 81/10/11 danaLab split is exactly aligned with the B01 contract | Section 1 + 2 of full validator; the count check is fail-closed |

## 15. Unverified

| Item | Why not verified |
|---|---|
| Any model metric on TEST (IoU, Dice, accuracy) | Out of scope; B01 is the data freeze, not training/evaluation.  B02/B03 will use these tables |
| The actual numeric min / max / mean / std of TRAIN pressure | Recorded in the (gitignored) `normalization_stats.json`; not echoed into this report to avoid biasing downstream review |
| Whether the 8-region GT is anatomically correct | NOT_REVIEWED; explicitly out of scope per the A09R contract |
| Cover1 / cover2 generalization | Dataset is uncover-only; cover splits are HOLD per A09R |

## 16. Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| B01 freeze uses only danaLab / uncover | Cannot train or evaluate cover1/cover2 region segmentation | A09R contract: cover splits remain HOLD until independent reference GT is available |
| GT provenance is auto-corrected support, NOT human pixel-level | Region boundaries are not medically validated | Repeated in dataset card and prohibited-conclusions section |
| No mini / full experiment in this task | Cannot quantify region segmentation quality yet | B01 is accepted; B02 (non-learning baseline) and B03 (PM-only Smoke) are now `READY` but not started |
| 8-region schema, NOT 10-region | ARM not split left/right; SHOULDER/THIGH/LOWER_LEG_FOOT are merged | Recorded in dataset card; matches the A09R SLP8 GT contract |
| Pressure values are raw PMarray response (not kPa) | No absolute pressure / comfort / hardware claims can be derived | Repeated in `raw_semantics` and prohibited-conclusions sections |

## 17. Next Gate

**Gate B01 (`DONE_WITH_LIMITATIONS`)**:

* B01 freeze artifacts are produced and pass the full fail-closed validator.
* Unit + integration tests are green; A09R + related regressions are green.
* Codex independent review accepted the implementation and evidence.

**Next tasks** (released after B01 acceptance):

* `TASK-SLP-B02-NON-LEARNING-REGION-BASELINE` (non-learning region baseline
  using the frozen B01 tables; must keep TEST access policy enforced
  unless a final-evaluation EXP-ID is created)

## 18. Prohibited conclusions

These conclusions must NOT be made based on the B01 freeze or any
downstream experiment until a future, properly-authorised Full protocol
+ Reviewer acceptance on a product-sensor setup is in place:

1. The 8-region GT is human pixel-level semantic annotation.
2. Pressure values represent kPa, force, or any physical unit.
3. The B01 numbers generalise to cover1, cover2, or simLab subjects.
4. The B01 numbers generalise to self-developed topper hardware,
   comfort, medical, or airbag-control performance.
5. The 8-region schema and the 10-region polygon schema are equivalent
   or interchangeable.
6. Subject overlap between TRAIN / VAL / TEST.
7. B01 itself establishes a usable product, hardware, or medical
   validation pipeline.
8. The freeze contract allows reading TEST label/onehot in development.

## 19. Handoff checklist

* [x] TASK-ID, branch, HEAD, base SHA recorded
* [x] `git status` clean at task start (worktree already on
      `codex/task-slp-b01-training-table-freeze-v0.1`, HEAD = 725d0aa)
* [x] All required documents read (AGENTS, COLLABORATION, PROJECT_STATUS,
      VALIDATION, EXPERIMENT_GOVERNANCE, SLP_TWO_PHASE_PLAN,
      SLP_AGENT_TASK_BACKLOG, S1_A09R stage report, schema, adapter,
      validator, test file)
* [x] B01 module, builder, validator, tests added (no edits to A09R)
* [x] Real-data build: 3,645 / 450 / 495 produced
* [x] Full validator: ALL CHECKS PASSED (incl. deterministic rebuild)
* [x] B01 tests: 82 passed
* [x] A09R + schema + adapter + infra + split regressions: 287 passed (1 skipped: A05 CSV not present); combined suite: 369 passed
* [x] Codex review accepted; commit, push, and PR authorized by Owner
* [x] `data/processed/slp8_training_tables_v0.1/` is gitignored
* [x] `git diff --check` clean

---

*This stage report was generated by Claude Code (Mavis) as the
implementation hand-off.  The freeze artifacts are written to
`data/processed/slp8_training_tables_v0.1/`, which is gitignored; the
report itself is the only B01-side file intended to be added to
the repository in the eventual commit (alongside the code, tests, and
project-status / backlog / plan / workflow updates).*
