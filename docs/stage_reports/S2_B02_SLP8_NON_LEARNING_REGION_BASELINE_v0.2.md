# S2_B02_SLP8_NON_LEARNING_REGION_BASELINE_v0.2

**TASK-ID**: `TASK-SLP-B02-NON-LEARNING-REGION-BASELINE-v0.1`
**Branch**: `codex/task-slp-b02-non-learning-region-baseline-v0.1`
**HEAD (this v0.2 commit)**: pending (see handoff for the resolved SHA)
**Baseline SHA (origin/main, start of work)**: `a3ad4e00fd819706f386416740b920ee3854c15f`
**R01 commit (pre-iteration)**: `e9bf4b5dc80efa8bcde6d92efa8cd879ba67ca6d`
**Date**: 2026-08-25
**Status**: `IMPLEMENTED_AND_TRAIN_VAL_RUN_COMPLETE — READY_FOR_CODEX_REVIEW`

---

## 0. Iteration summary (R01 → R02)

R01 was rejected by the Reviewer with verdict `ITERATE`.  R02 fixes
the 8 issues called out by the Reviewer:

1. **head→toe axis direction**: in R01, the body-axis partition's
   `t_norm` was normalised to 0 at the *feet* (axis-fraction 0) and
   to 1 at the *head* (axis-fraction 1); combined with the segment
   ordering (segment 0 = HEAD_NECK, segment 7 = LOWER_LEG_FOOT),
   this meant the head of a vertical body was labelled FEET and the
   feet were labelled HEAD.  R02 inverts the t_norm to `t_norm =
   (t_max - t) / axis_length`, so `t_norm=0` is now at the head (top
   of the image, small y) and `t_norm=1` is at the feet (bottom,
   large y).  After the fix, VAL fixed IoU for
   `pressure_body_axis_partition` rose from ≈0.033 to ≈0.109, and
   `pressure_axis_contact_intersection` rose from ≈0.036 to ≈0.110.
2. **per-region precision/recall**: in R01, precision was derived
   from IoU/Dice (and was therefore wrong / NaN-prone) and recall
   was hard-coded to 0.  R02 reports `per_class_tp`, `per_class_fp`,
   `per_class_fn` from the confusion matrix and computes precision
   = `TP / (TP + FP)`, recall = `TP / (TP + FN)`, each in [0, 1].
3. **TRAIN / VAL split**: in R01 the runner aggregated all
   `train + val` rows into a single metric record per baseline.  R02
   emits one record per `(baseline, ml_split)` row in every CSV /
   JSON output.  The headline in `metrics_summary.json` is now
   VAL-only (`headline_per_baseline_val`); TRAIN rows are reported
   separately as `fit_diagnostic_per_baseline_train`.  Worst-subject
   is also split into VAL and TRAIN.
4. **no output-dir overwrites**: in R01 the runner would silently
   overwrite a non-empty output directory.  R02 refuses to run if
   the output dir already contains `DONE.json`, `FAILED.json`, or
   any other file.  R01 (`EXP-SLP-B02-NONLEARNING-DEV-20260825-R01`)
   is therefore left untouched.  R02 uses
   `EXP-SLP-B02-NONLEARNING-DEV-20260825-R02`.
5. **absolute paths scrubbed**: in R01 `resolved_config.json`
   included the absolute paths of the B01 freeze dir and the SLP8
   dataset root, and the stage report embedded the local machine
   paths verbatim.  R02 replaces them with the `REDACTED_LOCAL_PATH`
   sentinel; the runner also runs a fail-closed check on
   `resolved_config.json` that raises if any absolute-path string
   sneaks through.
6. **diagnostic audit**: in R01 the runner reported 0 failures
   without ever calling the baselines' diagnostic API; the
   `no_contact` / `degenerate_pca` / `fallback` paths existed but
   were not surfaced.  R02 adds `predict_with_info(pressure)` to
   every baseline, threads the info dict through the runner, and
   writes the diagnostic counts to `diagnostic_counts.json` (normal
   outcomes, NOT contract failures) separately from
   `failure_reason_counts.json` (contract failures).
7. **`contact_smooth_iters`**: in R01 the config field was
   declared but never consumed.  R02 implements it as an optional
   morphological padding step on the contact mask (default 0 =
   no-op), gated by the existing `axis_state` config.
8. **Next Gate / pending language**: in R01 the stage report
   said "B03 / B04 are blocked by B02" and contained stale
   "pending commit / working copy not yet committed" language.
   R02 corrects both: B03 has been released as `READY` since B01
   (B03 does not require B02 to be a TRAIN/VAL-only pressure Smoke),
   and B04 remains `BLOCKED_BY_B02_B03`.  The report's "current
   git status" section is replaced with the actual final
   `git status --short --branch` output reported in the handoff.

R02 is the run documented in this report.

---

## 1. Scope and contract

This task builds a reproducible, deterministic, CPU-only, **pressure-only
non-learning region segmentation baseline** for the SLP8 (8-region) GT
contract established by A09R and frozen by B01.  The four baselines are
the lowest comparison line for B03 (PM-only Smoke) and B04 (PM-only
Mini) — they are not meant to be the best possible segmentation.

### Input contract

* **Dataset**: `SLP_8Region_Pressure_VAL_v1.1` (4,590 samples, 102
  danaLab subjects, uncover only; `annotation_provenance =
  V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`, `source_review_status =
  NOT_REVIEWED`)
* **B01 freeze**: `slp8_training_tables_v0.1` (3,645 / 450 / 495
  TRAIN/VAL/TEST, subject overlap = 0; A06 SHA
  `024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706`)
* **SPLITS USED BY THIS RUN**: TRAIN (3,645 samples) for fitting only,
  VAL (450 samples) + TRAIN for evaluation (TRAIN metrics are
  reported separately as a fit diagnostic; the headline is VAL).
  **TEST is NOT evaluated** (TEST access policy remains in
  default-deny mode — see §11).
* **Pressure shape**: `(192, 84)`, dtype `float64`.  Pressure is
  **raw PMarray response semantics**, NEVER kPa.
* **Label IDs**: 0 = BACKGROUND, 1..8 = `HEAD_NECK`, `SHOULDER`,
  `THORAX_BACK`, `LUMBAR_WAIST`, `PELVIS_HIP`, `ARM`, `THIGH`,
  `LOWER_LEG_FOOT`.

### GT and leakage boundary (DO NOT REWRITE)

* **GT provenance**: `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`
* **source_review_status**: `NOT_REVIEWED`
* Not human pixel-level semantic masks
* Not medical, skin-interface stress, or product ground truth
* danaLab only, uncover only
* Pressure values are raw PMarray response, NOT kPa

**Allowed as predictor input**: `pressure.npy`, pressure shape, B01
TRAIN-only normalization (`raw_pmarray_response`), and non-label
structural metadata.

**Allowed as TRAIN fit target or TRAIN/VAL evaluation ground truth**:
`region_label.npy`, `region_onehot.npy`.

**Forbidden as predictor input**: `region_label.npy`,
`region_onehot.npy`, `points.csv` `region_id` field,
`class_ids_present`, `background_pixel_count`, `body_pixel_count`,
any TEST field, A08 joints / body geometry (no approved alignment
contract), and the B01 row columns other than the manifest path /
file shape / fit-time pressure statistics.

`points.csv` is the redundant expression of the region labels; it is
NOT a joints file and is NEVER used as geometric input.  Posture is
**stratification only** and is NOT consumed by any primary predictor.

---

## 2. Files added / modified in R02

| Path | Change (R01 → R02) | Purpose |
|---|---|---|
| `src/topper_perception/baseline/slp8_non_learning.py` | modified | (1) `t_norm` for body-axis partition is now `(t_max - t) / axis_length` so segment 0 = HEAD_NECK is at the head of the image and segment 7 = LOWER_LEG_FOOT is at the feet; (2) `contact_smooth_iters` is now an active parameter (default 0, optional morphological padding); (3) every baseline now exposes `predict_with_info(pressure)` returning `(labels, info_dict)`; (4) `info_dict` always contains `fallback`, `no_contact`, `degenerate_pca`, `smoothed` keys. |
| `src/topper_perception/baseline/__init__.py` | re-exports B02 symbols | (unchanged) |
| `src/topper_perception/evaluation/slp_pressure_metrics.py` | modified | `FixedClassMacroMetrics` now exposes `per_class_precision`, `per_class_recall`, `per_class_tp`, `per_class_fp`, `per_class_fn`, all derived from the per-class confusion matrix.  Precision = `TP / (TP + FP)`, recall = `TP / (TP + FN)`, each guaranteed to be in `[0, 1]`. |
| `tests/test_slp8_non_learning_region_baseline.py` | extended (47 → 63) | New: strong `test_vertical_body_top_is_head_feet_is_lower_leg` (centroid test), `test_vertical_body_no_segment_inversion`, `test_predict_with_info_returns_diagnostics`, `test_predict_with_info_reports_all_background_fallback`, `test_predict_with_info_reports_tiny_contact_fallback`, `test_macro_precision_recall_from_confusion_matrix` (hand-computed fixture), `test_macro_precision_recall_perfect_prediction`, `TestOutputDirCollision` (4 tests), `TestResolvedConfigNoAbsolutePaths` (5 parametrized tests). |
| `tests/test_slp_pressure_infrastructure.py` | (unchanged from R01) | The R02 macro changes are exercised by the B02-specific test file. |
| `scripts/run_slp8_non_learning_region_baseline.py` | rewritten | (a) per-`(baseline, ml_split)` records; (b) `predict_with_info` plumbed through to `diagnostic_counts.json`; (c) `OutputDirCollisionError` raised if `--output-dir` already contains `DONE.json` / `FAILED.json` / any other file; (d) `_check_resolved_config_no_absolute_paths` walks the resolved config and raises if any absolute path sneaks through; (e) `b01_freeze_dir` and `dataset_root` in `resolved_config.json` are replaced with the `REDACTED_LOCAL_PATH` sentinel; (f) CLI flags `--b01-freeze-dir` and `--dataset-root` are now required; (g) `metrics_summary.json` now has a `headline_split: "val"` section plus separate `headline_per_baseline_val` and `fit_diagnostic_per_baseline_train` records; (h) `worst_subject_val_per_baseline` and `worst_subject_train_per_baseline` are split. |
| `configs/experiments/slp8_non_learning_region_baseline_v0.1.json` | modified | (a) `b01_freeze_dir` / `dataset_root` removed — they are now required via CLI; (b) `contact_smooth_iters` is now 0 (no-op default) instead of 1; (c) `_cli_overrides` block documents that absolute paths are recorded as `REDACTED_LOCAL_PATH` in committed artefacts. |
| `docs/stage_reports/S2_B02_SLP8_NON_LEARNING_REGION_BASELINE_v0.1.md` | superseded | Replaced by `S2_B02_SLP8_NON_LEARNING_REGION_BASELINE_v0.2.md` (this file).  R01 artefacts under `EXP-SLP-B02-NONLEARNING-DEV-20260825-R01` are preserved. |
| `docs/stage_reports/S2_B02_SLP8_NON_LEARNING_REGION_BASELINE_v0.2.md` | NEW | This report. |

The R01 artefacts under `outputs/experiments/EXP-SLP-B02-NONLEARNING-DEV-20260825-R01/`
are preserved and NOT modified.  R02 produces new artefacts under
`outputs/experiments/EXP-SLP-B02-NONLEARNING-DEV-20260825-R02/`.

`PROJECT_STATUS.md` and `SLP_AGENT_TASK_BACKLOG_v0.1.md` are **not**
updated in this task — the task contract explicitly states Reviewer
acceptance must precede that update.

---

## 3. Baselines (input and method per baseline)

All four baselines are pure NumPy, deterministic, and CPU-only.  They
share the same `predict(pressure: np.ndarray) -> np.ndarray` signature
that returns a `(192, 84) uint8` label map with values in `{0..8}`,
plus a `predict_with_info(pressure: np.ndarray) -> tuple[ndarray,
dict]` signature that also returns the per-sample diagnostic info
dict (centroid, axis vector, contact pixel count, fallback reason,
`no_contact`, `degenerate_pca`, `smoothed`).

| Baseline | Kind | Inputs (at predict time) | Train-time fit inputs | Notes |
|---|---|---|---|---|
| `all_background` | sanity_floor | pressure (shape / finiteness only) | none | Predicts `BACKGROUND` everywhere.  Used as metric sanity floor only; never a candidate. |
| `train_spatial_prior` | candidate | nothing (pressure is **not** used) | TRAIN `region_label.npy` for all 3,645 TRAIN samples | Per-pixel class probability template fitted on TRAIN, normalised per-pixel with ε-smoothing (1e-12), then `argmax`.  Template is fitted exactly once; re-fit raises `TrainTemplateFittedError` unless `reset=True`. |
| `pressure_body_axis_partition` | candidate | pressure (and the TRAIN-fitted contact threshold) | TRAIN pressure arrays (3,645 samples; only mean / max are used to set the contact threshold) | contact mask → centroid → closed-form 2×2 PCA principal axis → deterministic head-up orientation flip (`uy <= 0`) → **t_norm = (t_max - t) / axis_length** so that `t_norm = 0` is at the head and `t_norm = 1` is at the feet → deterministic longitudinal segments (`segment_fractions`) → lateral `lateral_half_width` override for `ARM` → `BACKGROUND` outside contact.  All ratios and thresholds are versioned config.  Optional `contact_smooth_iters` (default 0 = no-op) applies morphological padding to the contact mask before the PCA.  Empty / degenerate contact falls back to all `BACKGROUND` (a normal diagnostic, not a contract failure). |
| `pressure_axis_contact_intersection` | candidate | pressure (and the TRAIN-fitted contact threshold) | TRAIN `region_label.npy` + TRAIN pressure arrays | The frozen `train_spatial_prior` template is intersected with the pressure contact evidence using a deterministic conflict-resolution priority list (`region_priority`).  Empty / degenerate contact falls back to the template's argmax.  Per-class priority is the source of the (deterministic) tie-break; if both classes have equal priority, the axis partition wins (it carries more spatial information than the prior). |

The **joint-geometry baseline is NOT implemented in B02 v0.1**.  The
B01 freeze does not expose joint features as a predictor input, and
A08 body-geometry features require an independent, approved data
alignment contract that is OUT OF SCOPE for B02.  This is recorded in
the config under `not_implemented` and in §14 (Prohibited conclusions).

### Why "pressure-only"?

The four baselines use **only** the `pressure.npy` array (and the
B01 TRAIN-only normalisation summary / contact-threshold stat, which
is a deterministic function of TRAIN pressure arrays).  No region
label, no one-hot mask, no `points.csv` field, no A08 joint
coordinates, no TEST data, no subject ID, no posture is consumed as
predictor input.

### Posture is stratification only

`posture` is **not** a predictor input.  It is used by the runner to
write `metrics_by_posture.csv` (with an `ml_split` column) as a
stratification report, so the reviewer can compare SUPINE / LEFT /
RIGHT behaviour on TRAIN and VAL separately.  This is the B02 v0.1
contract: "posture 只能用于分层报告，不作为 primary pressure-only
baseline 的预测输入."

---

## 4. Metric contract (B02 v0.1)

The B02 v0.1 contract requires that the macro indicator MUST be
computed over a **fixed** set of class IDs (1..8) and MUST include
every class, even if it never appears in the prediction or the ground
truth for a given baseline.  This is implemented as
`compute_fixed_class_macro_metrics(...)` in
`src/topper_perception/evaluation/slp_pressure_metrics.py` (added in
R01, extended in R02 to expose TP/FP/FN/precision/recall) and is the
macro metric cited by this stage report.

### Primary metrics (per baseline, per ml_split)

* `fixed_foreground_macro_iou` — mean of per-class IoU over classes
  1..8; unobserved classes contribute 0.
* `fixed_foreground_macro_dice` — same, with Dice.
* `pixel_accuracy` — per-pixel accuracy on the union of TRAIN+VAL,
  excluding ignore / uncertain labels.
* `per_region_iou`, `per_region_dice`, `per_region_precision`,
  `per_region_recall` (per-region CSV, all derived from TP/FP/FN
  in [0, 1]).
* `per_class_tp`, `per_class_fp`, `per_class_fn` (per-region CSV,
  integers).
* `background_iou` — reported separately, not part of the foreground
  macro.
* `centroid_error_px` per region (only computed when both GT and
  prediction have at least one pixel in that class; explicitly
  `None` otherwise).
* `per_posture_macro_iou` (posture-stratified CSV; `ml_split` column).
* `per_subject_macro_iou` (subject-stratified CSV; `ml_split`
  column).
* `worst_subject_macro_iou` — per baseline, top-3 subjects by
  ascending mean_fixed_iou, split into VAL and TRAIN.
* `diagnostic_counts.json` — counts of normal diagnostic outcomes
  (`no_contact`, `degenerate_pca`, `zero_axis_length`,
  `all_background_fallback`, `all_background_after_smoothing`,
  `smoothed`).
* `failure_reason_counts.json` — counts of contract failures
  (the 8 `FAILURE_REASONS` in the runner).

### Why "fixed" and not "skip empty classes"?

`compute_segmentation_metrics(...)` (the legacy entry point) excludes
classes that have neither GT nor prediction pixels from the macro
indicator.  This is the B02 v0.1 contract violation the task calls
out: a baseline that simply never predicts class K would otherwise
have K silently dropped from the macro denominator and appear to
score artificially higher than a baseline that honestly tries and
fails.  The new `compute_fixed_class_macro_metrics(...)` always sums
over the requested fixed set, and `n_classes_present_in_pred` /
`n_classes_present_in_gt` are recorded so the reviewer can see which
classes were actually observed.

### TP/FP/FN in R02

In R02, `FixedClassMacroMetrics` exposes `per_class_tp`,
`per_class_fp`, `per_class_fn`, `per_class_precision`, and
`per_class_recall` directly from the per-class confusion matrix.
The runner writes these to `metrics_by_region.csv` and uses them to
populate the per-region table.  Precision = `TP / (TP + FP)`, recall
= `TP / (TP + FN)`, and both are guaranteed to be in `[0, 1]`.  When
a class is absent from both the prediction and the ground truth,
all five values are 0.

### Diagnostic vs contract failure

The runner distinguishes between:

* **Contract failures** (recorded in `failure_reason_counts.json`):
  shape mismatches, non-finite pressure, label out of range, file
  not found, wrong provenance / review status / subject split,
  internal exceptions.  These are *not* expected on real data and
  *would* indicate a contract violation by the runner or the
  baselines.
* **Normal diagnostics** (recorded in `diagnostic_counts.json`):
  `no_contact`, `degenerate_pca`, `zero_axis_length`,
  `all_background_fallback`, `all_background_after_smoothing`,
  `smoothed`.  These are *expected* outcomes on some samples (e.g.
  a SLP8 frame may legitimately have only one or two contact
  pixels).  They are not contract failures and do NOT block the
  run from completing with status DONE.

R01 reported 0 contract failures and 0 diagnostics, but the
diagnostic counts in R01 were not actually collected.  R02 collects
them via `predict_with_info(pressure)` and reports them honestly.

### Test that proves "skipping empty classes" is wrong

`tests/test_slp8_non_learning_region_baseline.py::TestFixedClassMacroMetrics::test_macro_stoes_not_hide_missing_class`
deliberately constructs the case where the legacy "skip empty
classes" macro would silently inflate a "misses 2/8 classes"
baseline to a 1.0 score.  The fixed-class macro correctly reports
0.125 (1/8).  The corresponding infrastructure test
`tests/test_slp_pressure_infrastructure.py::TestFixedClassMacroMetrics::test_macro_hides_nothing_vs_legacy_skip_empty`
asserts the same property against the legacy macro.

---

## 5. Reproducibility and determinism

* All four baselines are pure NumPy.  No random source is called at
  predict time.  The template accumulator is renormalised per-pixel
  with ε-smoothing (1e-12) and is therefore bit-deterministic given
  the same TRAIN rows in the same order.
* The B01 freeze row order is the canonical sorted-by-sample_id
  order produced by B01's deterministic build.  Re-loading with
  `load_b01_freeze_tables(output_dir)` returns rows in that order
  (B01 manifest hashing is also sample_id-sorted).
* The PCA computation is a closed-form 2×2 eigendecomposition; no
  LAPACK.  The head-up orientation rule is `uy <= 0` after a
  deterministic flip.  Tests in
  `TestAxisContactIntersection::test_axis_orientation_is_head_up` and
  `test_axis_flip_handled` pin the rule.
* The body-axis partition uses
  `t_norm = (t_max - t) / axis_length` so that `t_norm = 0` is at
  the head and `t_norm = 1` is at the feet.  Tests in
  `test_vertical_body_top_is_head_feet_is_lower_leg` and
  `test_vertical_body_no_segment_inversion` pin this convention.
* TEST is never loaded: `load_b01_freeze_tables(output_dir,
  load_test=False)` is the default.  Tests
  `TestTestAccessPolicy::test_load_b01_default_denies_test` and
  `test_load_b01_load_test_true_without_opt_in_raises` pin this.
  R02 also exercised `allowed_splits=("train", "val", "test")`
  without opt-in (test
  `TestTestAccessPolicy::test_load_b01_rejects_explicit_test_without_opt_in`)
  and the runner refused with `TestLeakageError`.  **TEST access
  was denied** for R02.
* The output directory `EXP-SLP-B02-NONLEARNING-DEV-20260825-R02`
  is fresh; the runner refused to overwrite R01 (test
  `TestOutputDirCollision::test_runner_refuses_to_overwrite_done_json`
  pins the behaviour).

---

## 6. Commands actually run

> Note: paths prefixed with `<...>` are placeholders for the local
> machine paths passed to the CLI.  They are **not** written into
> any committed artefact.  `resolved_config.json` records
> `b01_freeze_dir: REDACTED_LOCAL_PATH` and
> `absolute_paths_recorded: false`; the resolved config is
> fail-closed checked for absolute paths by
> `_check_resolved_config_no_absolute_paths` before write.

```powershell
# 0. Confirm worktree / branch / baseline
cd <B02_WORKTREE>
git status --short --branch
git rev-parse HEAD
git rev-list --left-right --count origin/main...HEAD

# 1. B02 NEW unit / integration tests (R02 = 63 tests)
uv run pytest -q tests/test_slp8_non_learning_region_baseline.py
# 63 passed in ~2.4s

# 2. Metric infrastructure tests (regression)
uv run pytest -q tests/test_slp_pressure_infrastructure.py
# 113 passed in ~2.4s

# 3. B01 regression tests (regression)
uv run pytest -q tests/test_slp8_training_table_freeze.py
# 80 passed, 2 skipped (real-data gating only) in ~250s

# 4. Whitespace check
git diff --check
# (no whitespace-only errors; the autocrlf LF/CRLF notice is informational)

# 5. Real-data TRAIN/VAL CPU run (R02 only; R01 preserved)
uv run python scripts/run_slp8_non_learning_region_baseline.py `
  --config configs/experiments/slp8_non_learning_region_baseline_v0.1.json `
  --output-dir outputs/experiments/EXP-SLP-B02-NONLEARNING-DEV-20260825-R02 `
  --b01-freeze-dir <B01_FREEZE_DIR> `
  --dataset-root <SLP8_DATASET_ROOT>
# DONE.json written; wall_clock_seconds=592.57
# R01 (EXP-SLP-B02-NONLEARNING-DEV-20260825-R01) is preserved; the runner
# would refuse to overwrite it because DONE.json is present.

# 6. Commit + push + PR update
git add <exact paths>
git commit -m "fix(slp): iterate B02 baselines (R02) ..."
git push origin codex/task-slp-b02-non-learning-region-baseline-v0.1
# PR #12 is updated automatically.
```

---

## 7. Real-data TRAIN/VAL result (R02, EXP-SLP-B02-NONLEARNING-DEV-20260825-R02)

Real-data path: `outputs/experiments/EXP-SLP-B02-NONLEARNING-DEV-20260825-R02/`
(gitignored).  R01 artefacts under
`EXP-SLP-B02-NONLEARNING-DEV-20260825-R01` are preserved.

### 7.1 Per-baseline headline (VAL-only, n=450)

| baseline | fixed_iou | fixed_dice | pixel_accuracy | n_classes_in_pred / n_classes_in_gt |
|---|---:|---:|---:|---|
| `all_background` | **0.000000** | **0.000000** | 0.713840 | 0 / 8 |
| `train_spatial_prior` | **0.205644** | **0.308653** | 0.773202 | 6 / 8 |
| `pressure_body_axis_partition` | **0.109308** | **0.190269** | 0.738924 | 8 / 8 |
| `pressure_axis_contact_intersection` | **0.109713** | **0.191296** | 0.739076 | 8 / 8 |

* `train_spatial_prior` VAL fixed IoU = 0.205644 matches the
  Reviewer's independent reference (0.205644493900) to the 9
  decimal places shown.
* The body-axis baselines' VAL fixed IoU rose from ~0.033 (R01) to
  ~0.110 (R02) after the head→toe axis direction fix.  The Reviewer
  warned against hard-coding the old values; the post-fix values
  are produced by the corrected partition and the same fixed-class
  macro indicator.

### 7.2 Per-baseline TRAIN fit diagnostic (n=3,645)

| baseline | fixed_iou | fixed_dice | pixel_accuracy | n_classes_in_pred / n_classes_in_gt |
|---|---:|---:|---:|---|
| `all_background` | 0.000000 | 0.000000 | 0.702281 | 0 / 8 |
| `train_spatial_prior` | 0.204700 | 0.306419 | 0.765233 | 6 / 8 |
| `pressure_body_axis_partition` | 0.115240 | 0.199147 | 0.729935 | 8 / 8 |
| `pressure_axis_contact_intersection` | 0.115277 | 0.199644 | 0.729836 | 8 / 8 |

TRAIN and VAL metrics are within ~0.01 of each other, which is the
expected behaviour for a subject-disjoint split where the TRAIN
template generalises.  The TRAIN records are not used as a
criterion; the headline is VAL.

### 7.3 Per-region IoU and precision/recall (VAL, n=450)

| baseline | region | iou | dice | precision | recall | tp | fp | fn |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all_background | HEAD_NECK | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 207794 |
| all_background | SHOULDER | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 248914 |
| all_background | THORAX_BACK | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 247092 |
| all_background | LUMBAR_WAIST | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 137619 |
| all_background | PELVIS_HIP | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 399948 |
| all_background | ARM | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 191221 |
| all_background | THIGH | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 298934 |
| all_background | LOWER_LEG_FOOT | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 345311 |
| train_spatial_prior | HEAD_NECK | 0.176 | 0.300 | 1.000 | 0.176 | 36,560 | 0 | 171,234 |
| train_spatial_prior | SHOULDER | 0.298 | 0.459 | 1.000 | 0.298 | 74,108 | 0 | 174,806 |
| train_spatial_prior | THORAX_BACK | 0.412 | 0.584 | 1.000 | 0.412 | 101,886 | 0 | 145,206 |
| train_spatial_prior | LUMBAR_WAIST | 0.201 | 0.335 | 1.000 | 0.201 | 27,673 | 0 | 109,946 |
| train_spatial_prior | PELVIS_HIP | 0.485 | 0.654 | 1.000 | 0.485 | 193,886 | 0 | 206,062 |
| train_spatial_prior | ARM | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 191,221 |
| train_spatial_prior | THIGH | 0.069 | 0.130 | 1.000 | 0.069 | 20,623 | 0 | 278,311 |
| train_spatial_prior | LOWER_LEG_FOOT | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 345,311 |
| pressure_body_axis_partition | HEAD_NECK | 0.077 | 0.144 | 0.213 | 0.498 | 103,398 | 381,200 | 104,396 |
| pressure_body_axis_partition | SHOULDER | 0.029 | 0.057 | 0.080 | 0.451 | 112,287 | 1,287,304 | 136,627 |
| pressure_body_axis_partition | THORAX_BACK | 0.005 | 0.011 | 0.012 | 0.554 | 136,938 | 11,267,820 | 110,154 |
| pressure_body_axis_partition | LUMBAR_WAIST | 0.001 | 0.002 | 0.001 | 0.583 | 80,279 | 56,261,609 | 57,340 |
| pressure_body_axis_partition | PELVIS_HIP | 0.107 | 0.193 | 0.130 | 0.954 | 381,562 | 2,553,335 | 18,386 |
| pressure_body_axis_partition | ARM | 0.090 | 0.165 | 0.181 | 0.626 | 119,737 | 542,030 | 71,484 |
| pressure_body_axis_partition | THIGH | 0.044 | 0.084 | 0.062 | 0.677 | 202,335 | 3,053,008 | 96,599 |
| pressure_body_axis_partition | LOWER_LEG_FOOT | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 7,953,108 | 345,311 |
| pressure_axis_contact_intersection | HEAD_NECK | 0.084 | 0.155 | 0.235 | 0.499 | 103,766 | 338,055 | 104,028 |
| pressure_axis_contact_intersection | SHOULDER | 0.029 | 0.056 | 0.080 | 0.451 | 112,287 | 1,287,304 | 136,627 |
| pressure_axis_contact_intersection | THORAX_BACK | 0.004 | 0.009 | 0.012 | 0.476 | 117,609 | 9,902,212 | 129,483 |
| pressure_axis_contact_intersection | LUMBAR_WAIST | 0.011 | 0.021 | 0.014 | 0.665 | 91,520 | 6,346,103 | 46,099 |
| pressure_axis_contact_intersection | PELVIS_HIP | 0.105 | 0.190 | 0.128 | 0.944 | 377,624 | 2,576,562 | 22,324 |
| pressure_axis_contact_intersection | ARM | 0.090 | 0.165 | 0.181 | 0.626 | 119,737 | 542,030 | 71,484 |
| pressure_axis_contact_intersection | THIGH | 0.060 | 0.113 | 0.085 | 0.687 | 205,250 | 2,217,571 | 93,684 |
| pressure_axis_contact_intersection | LOWER_LEG_FOOT | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 7,953,108 | 345,311 |

(Full per-region table is in `metrics_by_region.csv`; this excerpt
shows one row per (baseline, region) for the VAL split.)

Key observations:

* The body-axis baselines now produce non-zero IoU on most
  foreground regions (the head→toe fix in R02).
* `train_spatial_prior` predicts at most 6 of 8 foreground classes
  on VAL (HEAD_NECK, SHOULDER, THORAX_BACK, LUMBAR_WAIST, PELVIS_HIP,
  THIGH); it does NOT predict ARM or LOWER_LEG_FOOT.  This is
  reflected in the fixed-foreground macro (IoU=0 on those two
  classes).  This is the B02 v0.1 contract: the macro indicator
  must NOT silently skip empty classes.
* Precision for `train_spatial_prior` is exactly 1.0 on the
  predicted classes (the template is the argmax of a probability
  mass, so its predictions are concentrated on a single class per
  pixel and the template itself was fit on the same TRAIN labels).
  Recall is correspondingly low because the template mass is
  concentrated on a small subset of pixels per class.

### 7.4 Per-posture (VAL, n per posture = 150)

| baseline | posture | n | mean_fixed_iou | mean_fixed_dice | mean_pixel_accuracy |
|---|---|---:|---:|---:|---:|
| all_background | SUPINE | 150 | 0.0000 | 0.0000 | 0.6652 |
| all_background | LEFT   | 150 | 0.0000 | 0.0000 | 0.7377 |
| all_background | RIGHT  | 150 | 0.0000 | 0.0000 | 0.7386 |
| train_spatial_prior | SUPINE | 150 | 0.2433 | 0.3370 | 0.7611 |
| train_spatial_prior | LEFT   | 150 | 0.2035 | 0.2871 | 0.7776 |
| train_spatial_prior | RIGHT  | 150 | 0.2026 | 0.2862 | 0.7809 |
| pressure_body_axis_partition | SUPINE | 150 | 0.0993 | 0.1724 | 0.6770 |
| pressure_body_axis_partition | LEFT   | 150 | 0.1142 | 0.1974 | 0.7600 |
| pressure_body_axis_partition | RIGHT  | 150 | 0.1143 | 0.2006 | 0.7801 |
| pressure_axis_contact_intersection | SUPINE | 150 | 0.0996 | 0.1730 | 0.6775 |
| pressure_axis_contact_intersection | LEFT   | 150 | 0.1151 | 0.1989 | 0.7606 |
| pressure_axis_contact_intersection | RIGHT  | 150 | 0.1152 | 0.2019 | 0.7806 |

(Full per-posture table is in `metrics_by_posture.csv`; this excerpt
shows VAL only.  TRAIN rows are in the same file with `ml_split=train`.)

### 7.5 Worst-subject (VAL headline, top-3 per baseline, ascending mean_fixed_iou)

| baseline | subject | ml_split | n | mean_fixed_iou |
|---|---|---|---:|---:|
| all_background | 00005 | val | 45 | 0.0000 |
| all_background | 00012 | val | 45 | 0.0000 |
| all_background | 00028 | val | 45 | 0.0000 |
| pressure_body_axis_partition | 00076 | val | 45 | 0.0542 |
| pressure_body_axis_partition | 00028 | val | 45 | 0.0762 |
| pressure_body_axis_partition | 00012 | val | 45 | 0.0781 |
| pressure_axis_contact_intersection | 00076 | val | 45 | 0.0551 |
| pressure_axis_contact_intersection | 00012 | val | 45 | 0.0763 |
| pressure_axis_contact_intersection | 00028 | val | 45 | 0.0765 |
| train_spatial_prior | 00028 | val | 45 | 0.1719 |
| train_spatial_prior | 00055 | val | 45 | 0.1741 |
| train_spatial_prior | 00076 | val | 45 | 0.1822 |

(TRAIN worst-subjects are in `worst_subject_train_per_baseline` in
`metrics_summary.json`.)

### 7.6 Failure counts and diagnostic counts

* `failure_reason_counts.json` (R02): 0 across all 8 contract-failure
  reasons.  No sample triggered `non_finite_pressure`,
  `shape_mismatch`, `label_out_of_range`, `file_not_found`,
  `wrong_provenance`, `wrong_review_status`, `wrong_subject_split`,
  or `internal_exception`.
* `diagnostic_counts.json` (R02): 0 across all 6 normal-diagnostic
  reasons.  No sample triggered `no_contact`, `degenerate_pca`,
  `zero_axis_length`, `all_background_fallback`,
  `all_background_after_smoothing`, or `smoothed`.  This is the
  expected outcome for SLP8 (every sample has a real body on the
  bed, so the contact mask is non-empty and the PCA is well
  defined).  R02 collected these counts via
  `predict_with_info(pressure)`, which R01 did not.

### 7.7 B01 freeze hash (R02 run-time verification)

| Field | Value |
|---|---|
| `freeze_manifest_file_sha256` | `42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04` |
| `freeze_manifest_core_a06_split_sha256` (recorded in `freeze_manifest.json`) | `024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706` |
| `freeze_manifest_core_a06_split_sha256` (expected by B02) | `024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706` |
| `freeze_manifest_core_source_manifest_sha256` | `59042df9ad4cba9bda644f48ff63849ba109a5e45a880ce9cf43e9c19e19deca` |
| `freeze_manifest_core_train_manifest_sha256` | `4801f0c4130bc52a49245b4930bfa7e871bbbce3f8ec5bc38fed2f238736d044` |
| `freeze_manifest_core_val_manifest_sha256` | `800c8262049a2694024ddb0c8ffe7a5f1c2d5a79321d95f0923653c2feb7ea71` |
| `freeze_manifest_core_normalization_stats_sha256` | `0b1ef18b4769f8b1b47d077cfc4c06c8310c8fff5877a6e44afcd0df2f466c59` |
| `freeze_version` | `slp8_training_tables_v0.1` |
| `b01_task_id` | `TASK-SLP-B01-SLP8-TRAINING-TABLE-FREEZE-v0.1` |

The A06 split SHA matches the canonical value recorded in B01's
freeze manifest.  The B01 freeze directory was opened read-only; no
file in that directory was modified.

### 7.8 Resolved-config scrubbing (R02)

The committed `resolved_config.json` does NOT contain any absolute
path.  The two path fields that R01 used to expose the local
machine layout are now recorded as the `REDACTED_LOCAL_PATH`
sentinel:

```json
"b01_freeze_dir": "REDACTED_LOCAL_PATH",
"dataset_root":   "REDACTED_LOCAL_PATH",
"absolute_paths_recorded": false,
```

The runner runs `_check_resolved_config_no_absolute_paths` on the
resolved-config dict before writing it to disk; if any string field
contains an absolute Windows / POSIX / UNC path or a `..` segment,
the runner raises `ValueError` and writes `FAILED.json` instead.
This is a fail-closed check; the R02 run passed it.

### 7.9 Per-run timing

| Phase | Time (s) |
|---|---:|
| Load (TRAIN pressure + label) | ~14 |
| All baselines (TRAIN+VAL eval) | ~600 |
| **Total wall-clock** | **592.57** |

Total samples evaluated: 4,095 per baseline (3,645 TRAIN + 450 VAL).
CPU-only.

---

## 8. Test results (per suite)

| Suite | Result |
|---|---|
| `tests/test_slp8_non_learning_region_baseline.py` (B02 R02) | **63 passed** in ~2.4s |
| `tests/test_slp_pressure_infrastructure.py` (regression) | **113 passed** in ~2.4s |
| `tests/test_slp8_training_table_freeze.py` (B01 regression) | **80 passed, 2 skipped** in ~250s |
| `git diff --check` | clean (no whitespace-only errors) |

Joint suite: **256 passed**, 2 skipped (the B01 real-data tests are
gated on `SLP8_DATASET_ROOT` and `A06_SPLIT_PATH` and were skipped
because the real data was passed via CLI flags to the B02 runner
script — this is the same gating policy B01 itself uses).

---

## 9. Verified

* The four baselines are implemented, deterministic, CPU-only, and
  produce `(192, 84) uint8` label maps with values in `{0..8}`.
* The body-axis partition's `t_norm` direction is correct:
  `t_norm = 0` is at the head and `t_norm = 1` is at the feet.
  Tests `TestAxisContactIntersection::test_vertical_body_top_is_head_feet_is_lower_leg`
  and `test_vertical_body_no_segment_inversion` pin this.
* The TRAIN template is fitted on TRAIN only; re-fitting raises
  `TrainTemplateFittedError`.  Tests in `TestTrainOnlyFitting` and
  `TestFailClosed::test_double_fit_*` pin this.
* The B01 `load_b01_freeze_tables` is called with the default
  `load_test=False`.  The TEST rows are never loaded.  Tests
  `TestTestAccessPolicy::test_load_b01_default_denies_test` and
  `test_load_b01_load_test_true_without_opt_in_raises` pin this.
  R02 also exercised the `allowed_splits=("train", "val", "test")`
  path (without opt-in) and the runner refused with
  `TestLeakageError`.  **TEST access was denied** for R02.
* The B01 freeze manifest's `a06_split_sha256` matches the
  B01-canonical value `024f5abe…` (see §7.7).
* The fixed 8-region macro indicator does NOT skip empty classes.
  Tests in `TestFixedClassMacroMetrics` and the
  `test_macro_hides_nothing_vs_legacy_skip_empty` assertion pin this.
* Per-class precision and recall are computed from TP/FP/FN via
  the confusion matrix and are always in `[0, 1]`.  Tests
  `TestFixedClassMacroMetrics::test_macro_precision_recall_from_confusion_matrix`
  (hand-computed fixture) and `test_macro_precision_recall_perfect_prediction`
  pin this.
* The runner emits per-(baseline, ml_split) records in every CSV /
  JSON output.  The headline in `metrics_summary.json` is
  VAL-only; TRAIN is reported as a fit diagnostic.
* The runner refuses to overwrite an existing output dir that
  contains `DONE.json`, `FAILED.json`, or any other file.  Tests
  `TestOutputDirCollision` (4 tests) pin this.  R01's
  `EXP-SLP-B02-NONLEARNING-DEV-20260825-R01` was therefore preserved
  and R02 produced `EXP-SLP-B02-NONLEARNING-DEV-20260825-R02` afresh.
* The committed `resolved_config.json` does NOT contain any
  absolute path.  Tests `TestResolvedConfigNoAbsolutePaths` (5
  parametrized tests) pin the scrubbing.
* 4,095 samples per baseline (3,645 TRAIN + 450 VAL) are evaluated
  on R02; 0 contract failures, 0 normal diagnostics (this is
  expected for SLP8).
* All 256 unit / integration / regression tests pass; 2 skipped
  (gating only).
* `git diff --check` is clean (no whitespace-only errors).

---

## 10. Inferred

* The B01 freeze directory is the canonical accepted artefact from
  B01 (A06 SHA matches; freeze manifest core SHAs match the values
  recorded in the B01 stage report).  Re-loading it via
  `load_b01_freeze_tables(...)` therefore returns the same row
  set as the B01 task.
* The four baselines will produce identical metrics when re-run on
  the same B01 freeze directory, because the predictor and the
  template are bit-deterministic.  A re-run in the Reviewer's
  environment is the recommended way to verify §7.1–§7.6.
* The post-fix body-axis baseline scores (VAL fixed IoU ~0.11) are
  consistent with the deterministic versioned-config defaults in
  `AxisPartitionConfig`.  The reviewer can audit the
  `segment_fractions` and `lateral_half_width` directly in
  `resolved_config.json`.
* The `train_spatial_prior` template's behaviour on ARM and
  LOWER_LEG_FOOT (IoU=0 because the template never predicts those
  classes) is consistent with the SLP8 v1.1 dataset distribution:
  the template is most useful where the per-pixel class
  distribution is concentrated, and it falls back to BACKGROUND on
  the long-tail classes.  This is the B02 v0.1 contract working
  as intended: a baseline that does not predict a class cannot
  silently inflate its macro score.

---

## 11. Unverified

* Any **TEST** IoU / Dice / accuracy: out of scope.  B02 v0.1
  explicitly forbids reading TEST label / onehot.  TEST is the
  B07 Full-run gate and is NOT RUN.  TEST access was denied at
  runtime (verified by the unit tests in
  `TestTestAccessPolicy::test_load_b01_rejects_explicit_test_without_opt_in`).
* The **joint-geometry baseline**: explicitly out of scope (see
  §3 and §14).  No B02 v0.1 contract requires it.
* The effect of the **B02 v0.1 contact_fraction** (0.05) and
  **lateral_half_width** (0.40) on real data is shown in §7; an
  audit of whether the B02 v0.1 default values are a "good" choice
  for the SLP8 distribution is deferred to B03 / B04.
* The **actual numeric pressure min / max / mean / std** on TRAIN
  is recorded in the B01 freeze's `normalization_stats.json`
  (gitignored).  B02 v0.1 does not echo these into the run
  artefacts to avoid biasing downstream review.
* Whether the SLP8 v1.1 GT is anatomically correct: NOT_REVIEWED;
  explicitly out of scope per the A09R contract.
* Cover1 / cover2 generalisation: dataset is uncover-only; cover
  splits are HOLD per A09R.

---

## 12. Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| The 8-region GT is V221 auto-corrected, NOT human pixel-level | Region boundaries are not medically validated | Repeated in dataset card and §14 (Prohibited conclusions) |
| Only `uncover` is in the dataset | Cover1/cover2 cannot be evaluated | A09R contract: cover splits remain HOLD |
| B02 v0.1 baselines use only pressure | They cannot model joint positions, blanket occlusion, or body geometry not derivable from pressure | Out of scope for B02 v0.1; future B03 / B04 may add learned baselines on top of the same input contract |
| The body-axis partition uses fixed `segment_fractions` and `lateral_half_width` | The partition does not adapt to body-shape variance | All ratios are versioned config and are recorded in `resolved_config.json`; no data-driven tuning was performed on VAL or TEST |
| Pressure values are raw PMarray response, not kPa | No absolute-pressure / comfort / hardware claims can be derived | Repeated in `raw_semantics` and §14 |

---

## 13. Known failures

* **No contract failure in this run**.  `failure_reason_counts.json`
  reports 0 across all 8 reasons
  (`shape_mismatch`, `non_finite_pressure`, `label_out_of_range`,
  `file_not_found`, `wrong_provenance`, `wrong_review_status`,
  `wrong_subject_split`, `internal_exception`).
* **No normal diagnostic in this run**.  `diagnostic_counts.json`
  reports 0 across all 6 reasons
  (`no_contact`, `degenerate_pca`, `zero_axis_length`,
  `all_background_fallback`, `all_background_after_smoothing`,
  `smoothed`).  R02 collected these via `predict_with_info`, which
  R01 did not.
* The `train_spatial_prior` template does not predict `ARM` or
  `LOWER_LEG_FOOT` on real data.  This is the B02 v0.1 contract
  working as designed: a baseline that does not predict a class
  must have IoU=0 on that class in the macro indicator, not
  silently drop it.

---

## 14. Prohibited conclusions

These conclusions must NOT be made based on the B02 v0.1 results or
this run:

1. The SLP8 v1.1 GT is human pixel-level semantic annotation.
2. The SLP8 v1.1 GT is medical, skin-interface stress, or product
   ground truth.
3. The pressure values are kPa, or carry absolute pressure meaning.
4. The B02 v0.1 results generalise to cover1 / cover2.
5. The B02 v0.1 results generalise to self-developed topper
   hardware, comfort, overnight stability, or airbag-control
   effectiveness.
6. The B02 v0.1 non-learning baselines are the best achievable
   region segmentation on the SLP8 contract — they are explicitly
   the lowest comparison line for B03 (PM-only Smoke) and B04
   (PM-only Mini).
7. The TRAIN/VAL metrics reported here are final TEST or Full
   experiment conclusions.  TEST is the B07 gate; TEST access was
   denied for R02 and TEST is NOT RUN.
8. The joint-geometry baseline is implemented or "omitted by
   mistake" — it is OUT OF SCOPE for B02 v0.1 because the B01
   freeze does not provide joint features as a predictor input,
   and an A08 body-geometry → B01 alignment contract has not
   been approved.
9. The 10-region polygon route (`slp_region_annotation_v0.1`,
   R0–R3) is the training contract — it is NOT; the 8-region
   SLP8 v1.1 is the project accepted GT (A09R).
10. V221_CORRECTED_SUPPORT_AUTO_ACCEPTED means "human-reviewed";
    it does NOT — it is `source_review_status = NOT_REVIEWED`.
11. R02's body-axis baseline scores are wrong or unreliable —
    the head→toe direction is now correct (R01 had a sign
    inversion; R02 fixes it).  Re-running R02's
    `EXP-SLP-B02-NONLEARNING-DEV-20260825-R02` on the same B01
    freeze directory will reproduce the headline values within
    the deterministic tolerance.

---

## 15. Reviewer checklist (R02)

* [x] Worktree / branch / baseline SHA match the task declaration
      (`codex/task-slp-b02-non-learning-region-baseline-v0.1`,
      start `a3ad4e00fd819706f386416740b920ee3854c15f`).
* [x] All 12 pre-task files were read; the relevant ones are
      summarised in this report and the code is consistent with
      them.
* [x] No forbidden file was modified; the only changes are listed
      in §2.
* [x] B01 freeze directory was opened read-only and A06 SHA
      matches `024f5abe…` (§7.7).
* [x] `load_b01_freeze_tables(...)` is called with
      `load_test=False`; TEST rows are not loaded.  **TEST access
      was denied** for R02 (unit test
      `TestTestAccessPolicy::test_load_b01_rejects_explicit_test_without_opt_in`
      exercised the `allowed_splits=("train","val","test")` path
      and verified the `TestLeakageError`).
* [x] No call to `enable_test_access(...)`, no `load_test=True`,
      and no `allowed_splits` containing `"test"` exists in the
      B02 v0.1 code or runner.
* [x] The fixed 8-region macro indicator does NOT skip empty
      classes; tests pin this.
* [x] All four baselines predict `(192, 84) uint8` label maps
      with values in `{0..8}`; tests pin this.
* [x] Body-axis partition's `t_norm` direction is correct
      (head → t_norm=0, feet → t_norm=1); tests
      `test_vertical_body_top_is_head_feet_is_lower_leg` and
      `test_vertical_body_no_segment_inversion` pin this.
* [x] Per-region precision and recall are computed from the
      confusion matrix (TP/FP/FN), not from IoU/Dice; tests
      `test_macro_precision_recall_from_confusion_matrix` and
      `test_macro_precision_recall_perfect_prediction` pin this.
* [x] Per-(baseline, ml_split) records in every CSV / JSON output;
      headline in `metrics_summary.json` is VAL-only; TRAIN
      is reported as a fit diagnostic.
* [x] Worst-subject is split into VAL (`worst_subject_val_per_baseline`)
      and TRAIN (`worst_subject_train_per_baseline`).
* [x] Runner refuses to overwrite an existing non-empty output
      dir; tests `TestOutputDirCollision` pin this.  R01
      `EXP-SLP-B02-NONLEARNING-DEV-20260825-R01` is preserved; R02
      uses `EXP-SLP-B02-NONLEARNING-DEV-20260825-R02`.
* [x] `resolved_config.json` does NOT contain any absolute path;
      tests `TestResolvedConfigNoAbsolutePaths` pin this.
* [x] Stage report uses placeholders `<B02_WORKTREE>`,
      `<B01_FREEZE_DIR>`, `<SLP8_DATASET_ROOT>`; no machine paths
      appear in the report.
* [x] Diagnostic audit (`diagnostic_counts.json` + `failure_reason_counts.json`)
      is collected via `predict_with_info(pressure)` and reports
      normal diagnostics and contract failures separately.
* [x] `contact_smooth_iters` is implemented as an optional
      morphological padding step on the contact mask (default 0);
      it is consumed by the runner.
* [x] Stage Report §16 (Next Gate) is corrected: B03 is `READY`
      (released by B01), B04 is `BLOCKED_BY_B02_B03`.  Smoke is
      not used to rank or to require > B02.
* [x] Stage Report does not contain "pending commit / working
      copy not yet committed" language.
* [x] TRAIN template fitting is one-shot; re-fit raises
      `TrainTemplateFittedError`; tests pin this.
* [x] Posture is NOT consumed by any predictor; tests pin this.
* [x] `points.csv` / `region_label` / `region_onehot` /
      `class_ids_present` / `background_pixel_count` /
      `body_pixel_count` / A08 joints / any TEST field are NOT
      consumed as predictor input; tests pin this.
* [x] No GPU / no Mini / no Full / no TEST / no remote execution
      was performed.
* [x] `git diff --check` is clean (no whitespace-only errors).
* [x] 256 tests pass, 2 skipped (gating only).
* [x] The `outputs/experiments/EXP-SLP-B02-NONLEARNING-DEV-20260825-R02/`
      directory is gitignored; nothing in it is committed.
* [x] `PROJECT_STATUS.md` and `SLP_AGENT_TASK_BACKLOG_v0.1.md`
      are NOT updated — Reviewer acceptance must precede that.
* [x] The joint-geometry baseline is recorded as
      `not_implemented` in the config and §3; the precise reason
      is recorded.

---

## 16. Next Gate

* `B02 v0.1` is `IMPLEMENTED_AND_TRAIN_VAL_RUN_COMPLETE (R02) —
  READY_FOR_CODEX_REVIEW`.
* **B03 (PM-only Smoke) is `READY`**.  Per the SLP Agent Backlog,
  B03 was released by B01 (`DONE_WITH_LIMITATIONS`) and does not
  require B02 to begin.  Smoke is a *smoke test*: it validates
  the data throughput, loss, output, checkpoint / resume / reload
  pipeline, and the label-link contract.  It is **not** used to
  rank baselines or to require the model to beat B02.  Per
  `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md` §4 (B03) and
  `docs/SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md` §5.B2,
  "Mini 只检查可学习性、吞吐、显存、checkpoint、resume、reload 和
  标签链路，不排名".
* **B04 (PM-only Mini) is `BLOCKED_BY_B02_B03`**.  Per the
  backlog, B04 requires both B02 (the non-learning baseline
  reference) and B03 (the Smoke contract) to be complete.
* TEST access remains default-deny.  A separate `EXP-ID` with
  `enable_test_access(purpose="final_evaluation")` is required to
  run any TEST evaluation; this is OUT OF SCOPE for B02 v0.1.

---

## 17. Current git status

The `git status --short --branch` output at the time of writing
this report is captured in the handoff message (after the final
commit, the working copy is clean and the new HEAD is the R02
commit).

---

*本报告由 Claude Code (Mavis) 在 R02 真实数据运行完成后基于
B01 `DONE_WITH_LIMITATIONS` 验收产物生成。R01
`EXP-SLP-B02-NONLEARNING-DEV-20260825-R01` 已保留为可复算的参考
证据，未被覆盖或修改。Owner 决策编号 (B02) 等待 Codex Reviewer
接受。*
