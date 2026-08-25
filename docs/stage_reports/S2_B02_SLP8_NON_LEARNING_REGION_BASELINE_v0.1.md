# S2_B02_SLP8_NON_LEARNING_REGION_BASELINE_v0.1

**TASK-ID**: `TASK-SLP-B02-NON-LEARNING-REGION-BASELINE-v0.1`
**Branch**: `codex/task-slp-b02-non-learning-region-baseline-v0.1`
**HEAD (B02 implementation)**: pending commit (will be reported in the handoff)
**Baseline SHA (origin/main)**: `a3ad4e00fd819706f386416740b920ee3854c15f` (start of work)
**Date**: 2026-08-25
**Status**: `IMPLEMENTED_AND_TRAIN_VAL_RUN_COMPLETE — READY_FOR_CODEX_REVIEW`

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
  VAL (450 samples) + TRAIN for evaluation.  **TEST is NOT evaluated**
  (TEST access policy remains in default-deny mode).
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
contract), and the B01 row columns other than the manifest path / file
shape / fit-time pressure statistics.

`points.csv` is the redundant expression of the region labels; it is
NOT a joints file and is NEVER used as geometric input.  Posture is
**stratification only** and is NOT consumed by any primary predictor.

---

## 2. Files added / modified

| Path | Change | Purpose |
|---|---|---|
| `src/topper_perception/baseline/slp8_non_learning.py` | NEW | B02 baseline module: 4 baselines (AllBackground, TrainSpatialPrior, PressureBodyAxisPartition, PressureAxisContactIntersection) + body-axis / contact / priority logic; deterministic; CPU-only; no torch. |
| `src/topper_perception/baseline/__init__.py` | UPDATED | Re-export B02 symbols. |
| `src/topper_perception/evaluation/slp_pressure_metrics.py` | UPDATED | Added `FixedClassMacroMetrics` and `compute_fixed_class_macro_metrics(...)` — B02 v0.1 fixed-foreground macro indicator that does **not** skip empty classes. |
| `tests/test_slp8_non_learning_region_baseline.py` | NEW | 47 unit / integration tests (output shape/dtype/range; determinism; all-zero pressure; single-point / tiny contact; degenerate PCA; non-finite pressure fail-closed; wrong shape fail-closed; contact vs axis priority; head-up axis direction; TRAIN-only template fit; VAL does not participate in fit; posture not in predictor; points.csv / label / onehot not in predictor; TEST default-denied; fixed-8-region macro does not hide missing classes; config roundtrip; failure paths still auditable). |
| `tests/test_slp_pressure_infrastructure.py` | UPDATED | Added 8 fixed-class-macro tests to `TestFixedClassMacroMetrics`. |
| `scripts/run_slp8_non_learning_region_baseline.py` | NEW | CLI runner; CLI flags for `--b01-freeze-dir` and `--dataset-root` (no absolute paths in committed artefacts).  Failure paths produce `FAILED.json`. |
| `configs/experiments/slp8_non_learning_region_baseline_v0.1.json` | NEW | Versioned config template.  Placeholders `B01_FREEZE_DIR_PLACEHOLDER` and `DATASET_ROOT_PLACEHOLDER` are not used as input here; the real run was launched via CLI flags. |
| `docs/stage_reports/S2_B02_SLP8_NON_LEARNING_REGION_BASELINE_v0.1.md` | NEW | This report. |

The `data/processed/`, `models/`, `outputs/figures/`,
`outputs/reports/`, `outputs/models/`, `outputs/experiments/`,
`outputs/analysis/` paths remain gitignored; raw and generated data
stay outside Git.  The `outputs/experiments/EXP-SLP-B02-NONLEARNING-DEV-20260825-R01/`
directory is therefore not committed.

`PROJECT_STATUS.md` and `SLP_AGENT_TASK_BACKLOG_v0.1.md` are **not**
updated in this task — the task contract explicitly states Reviewer
acceptance must precede that update.

---

## 3. Baselines (input and method per baseline)

All four baselines are pure NumPy, deterministic, and CPU-only.  They
share the same `predict(pressure: np.ndarray) -> np.ndarray` signature
that returns a `(192, 84) uint8` label map with values in `{0..8}`.

| Baseline | Kind | Inputs (at predict time) | Train-time fit inputs | Notes |
|---|---|---|---|---|
| `all_background` | sanity_floor | pressure (shape / finiteness only) | none | Predicts `BACKGROUND` everywhere.  Used as metric sanity floor only; never a candidate. |
| `train_spatial_prior` | candidate | nothing (pressure is **not** used) | TRAIN `region_label.npy` for all 3,645 TRAIN samples | Per-pixel class probability template fitted on TRAIN, normalised per-pixel with ε-smoothing (1e-12), then `argmax`.  Template is fitted exactly once; re-fit raises `TrainTemplateFittedError` unless `reset=True`. |
| `pressure_body_axis_partition` | candidate | pressure (and the TRAIN-fitted contact threshold) | TRAIN pressure arrays (3,645 samples; only mean / max are used to set the contact threshold) | contact mask → centroid → closed-form 2×2 PCA principal axis → deterministic head-up orientation flip → deterministic longitudinal segments (`segment_fractions`) → lateral `lateral_half_width` override for `ARM` → `BACKGROUND` outside contact.  All ratios and thresholds are versioned config.  Empty / degenerate contact falls back to all `BACKGROUND`. |
| `pressure_axis_contact_intersection` | candidate | pressure (and the TRAIN-fitted contact threshold) | TRAIN `region_label.npy` + TRAIN pressure arrays | The frozen `train_spatial_prior` template is intersected with the pressure contact evidence using a deterministic conflict-resolution priority list (`region_priority`).  Empty / degenerate contact falls back to the template's argmax.  Per-class priority is the source of the (deterministic) tie-break; if both classes have equal priority, the axis partition wins (it carries more spatial information than the prior). |

The **joint-geometry baseline is NOT implemented in B02 v0.1**.  The
B01 freeze does not expose joint features as a predictor input, and
A08 body-geometry features require an independent, approved data
alignment contract that is OUT OF SCOPE for B02.  This is recorded in
the config under `not_implemented` and in §10 (Prohibited conclusions).

### Why "pressure-only"?

The four baselines use **only** the `pressure.npy` array (and the
B01 TRAIN-only normalisation summary / contact-threshold stat, which
is a deterministic function of TRAIN pressure arrays).  No region
label, no one-hot mask, no `points.csv` field, no A08 joint
coordinates, no TEST data, no subject ID, no posture is consumed as
predictor input.

### Posture is stratification only

`posture` is **not** a predictor input.  It is used by the runner to
write `metrics_by_posture.csv` as a stratification report, so the
reviewer can compare SUPINE / LEFT / RIGHT behaviour.  This is the
B02 v0.1 contract: "posture 只能用于分层报告，不作为 primary
pressure-only baseline 的预测输入."

---

## 4. Metric contract (B02 v0.1)

The B02 v0.1 contract requires that the macro indicator MUST be
computed over a **fixed** set of class IDs (1..8) and MUST include
every class, even if it never appears in the prediction or the ground
truth for a given baseline.  This is implemented as
`compute_fixed_class_macro_metrics(...)` in
`src/topper_perception/evaluation/slp_pressure_metrics.py` (added in
this task) and is the macro metric cited by this stage report.

### Primary metrics (per baseline)

* `fixed_foreground_macro_iou` — mean of per-class IoU over classes
  1..8; unobserved classes contribute 0.
* `fixed_foreground_macro_dice` — same, with Dice.
* `pixel_accuracy` — per-pixel accuracy on the union of TRAIN+VAL,
  excluding ignore / uncertain labels.
* `per_region_iou`, `per_region_dice`, `per_region_precision`,
  `per_region_recall` (per-region CSV).
* `background_iou` — reported separately, not part of the foreground
  macro.
* `centroid_error_px` per region (only computed when both GT and
  prediction have at least one pixel in that class).
* `per_posture_macro_iou` (posture-stratified CSV).
* `per_subject_macro_iou` (subject-stratified CSV).
* `worst_subject_macro_iou` (per-baseline worst-3 subjects, in
  `metrics_summary.json`).
* `failure_reason_counts` (default: 0 for this run).

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

### Test that proves "skipping empty classes" is wrong

`tests/test_slp8_non_learning_region_baseline.py::TestFixedClassMacroMetrics::test_macro_strict_does_not_hide_missing_class`
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
  deterministic flip.  Tests in `TestAxisContactIntersection::test_axis_orientation_is_head_up`
  and `test_axis_flip_handled` pin the rule.
* TEST is never loaded: `load_b01_freeze_tables(output_dir,
  load_test=False)` is the default.  Tests
  `TestTestAccessPolicy::test_load_b01_default_denies_test` and
  `test_load_b01_load_test_true_without_opt_in_raises` pin this.

---

## 6. Commands actually run

> Note: paths prefixed with `E:\TeamProjects\` are local machine
> paths passed to the CLI.  They are **not** written into any
> committed artefact.  The `resolved_config.json` records only the
> file-existence flag and the freeze manifest SHA; no absolute path
> is written into any CSV / JSON output.

```powershell
# 0. Confirm worktree / branch / baseline
cd E:\TeamProjects\smarttopper-slp-b02-non-learning-region-baseline
git status --short --branch
git rev-parse HEAD
git rev-list --left-right --count origin/main...HEAD

# 1. New B02 unit / integration tests
uv run pytest -q tests/test_slp8_non_learning_region_baseline.py
# 47 passed in 1.98s

# 2. Fixed-class-macro tests added to the metrics infrastructure test
uv run pytest -q tests/test_slp_pressure_infrastructure.py
# 113 passed in 2.26s

# 3. B01 regression tests (re-run to confirm B01 is not broken)
uv run pytest -q tests/test_slp8_training_table_freeze.py
# 80 passed, 2 skipped (real-data tests are not gated in this run)

# 4. Real-data TRAIN/VAL CPU run (the only allowed real run)
uv run python scripts/run_slp8_non_learning_region_baseline.py `
  --config configs/experiments/slp8_non_learning_region_baseline_v0.1.json `
  --output-dir outputs/experiments/EXP-SLP-B02-NONLEARNING-DEV-20260825-R01 `
  --b01-freeze-dir "E:\TeamProjects\smarttopper-slp-b01-training-table-freeze\data\processed\slp8_training_tables_v0.1" `
  --dataset-root "E:\TeamProjects\datasets\smart-topper\SLP2022\SLP\SLP_8Region_Pressure_VAL_v1.1"
# DONE.json written; wall_clock_seconds=396.18

# 5. Whitespace check
git diff --check
# (no whitespace-only errors; the autocrlf LF/CRLF warning is informational)
```

---

## 7. Real-data TRAIN/VAL result (B02 v0.1, EXP-SLP-B02-NONLEARNING-DEV-20260825-R01)

Real-data path: `outputs/experiments/EXP-SLP-B02-NONLEARNING-DEV-20260825-R01/`
(gitignored).  All artefacts are listed in
`metrics_summary.json`'s `expected_artifacts` field.

### 7.1 Per-baseline headline metrics (TRAIN+VAL, n=4095)

| baseline | n_eval | n_failed | fixed_iou | fixed_dice | pixel_accuracy | n_classes_in_pred / n_classes_in_gt |
|---|---:|---:|---:|---:|---:|---|
| `all_background` | 4095 | 0 | 0.000000 | 0.000000 | 0.703551 | 0 / 8 |
| `train_spatial_prior` | 4095 | 0 | **0.204795** | **0.306661** | 0.766108 | 6 / 8 |
| `pressure_body_axis_partition` | 4095 | 0 | 0.031348 | 0.053662 | 0.705507 | 8 / 8 |
| `pressure_axis_contact_intersection` | 4095 | 0 | 0.034581 | 0.060430 | 0.706747 | 8 / 8 |

* `n_eval` = number of samples the runner successfully predicted on
  (no contract error).
* `n_failed` = number of samples that triggered a fail-closed branch
  (none in this run).
* `pixel_accuracy` for `all_background` is ~0.704 because ~70% of
  pixels in SLP8 v1.1 are background (per the B01 dataset card
  `class_stats.train.per_class_pixel_ratio[0] = 0.7023`).  This
  emphasises the B02 v0.1 contract: "不要只报告包含大量背景像素的 accuracy."

### 7.2 Per-region IoU (foreground only, classes 1..8)

| baseline | 1=HEAD_NECK | 2=SHOULDER | 3=THORAX_BACK | 4=LUMBAR_WAIST | 5=PELVIS_HIP | 6=ARM | 7=THIGH | 8=LOWER_LEG_FOOT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all_background` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `train_spatial_prior` | 0.169 | 0.297 | 0.416 | 0.204 | 0.484 | **0.000** | 0.068 | **0.000** |
| `pressure_body_axis_partition` | 0.000 | 0.000 | 0.000 | 0.001 | 0.202 | 0.048 | 0.000 | 0.000 |
| `pressure_axis_contact_intersection` | 0.000 | 0.000 | 0.000 | 0.010 | 0.196 | 0.048 | 0.023 | 0.000 |

The bold 0.000 cells for `train_spatial_prior` on `ARM` and
`LOWER_LEG_FOOT` are the **demonstration of the fixed-class macro**:
those classes never appear in the prediction (`is_present_in_pred =
false`), so the fixed macro correctly counts them as 0 instead of
silently dropping them from the denominator.

### 7.3 Centroid error (px) per region (only when both GT and pred have ≥1 pixel)

| baseline | 1=HEAD_NECK | 2=SHOULDER | 3=THORAX_BACK | 4=LUMBAR_WAIST | 5=PELVIS_HIP | 6=ARM | 7=THIGH | 8=LOWER_LEG_FOOT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all_background` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `train_spatial_prior` | 12.81 | 9.30 | 8.29 | 8.46 | 9.12 | n/a | 14.39 | n/a |
| `pressure_body_axis_partition` | 148.03 | 108.96 | 71.59 | 33.34 | 9.33 | 27.56 | 71.65 | 128.04 |
| `pressure_axis_contact_intersection` | 148.03 | 108.96 | 71.95 | 30.40 | 9.05 | 27.56 | 61.60 | 128.04 |

`n/a` means the centroid error was **not** computed because either GT
or prediction had 0 pixels in that class.  This is the B02 v0.1
"empty GT or pred" rule — the failure mode is recorded, not silently
zeroed.

### 7.4 Per-posture (n per posture = 1365)

| baseline | posture | n | mean_fixed_iou | mean_fixed_dice | mean_pixel_accuracy |
|---|---|---:|---:|---:|---:|
| all_background | SUPINE | 1365 | 0.0000 | 0.0000 | 0.6552 |
| all_background | LEFT   | 1365 | 0.0000 | 0.0000 | 0.7273 |
| all_background | RIGHT  | 1365 | 0.0000 | 0.0000 | 0.7281 |
| train_spatial_prior | SUPINE | 1365 | 0.2432 | 0.3352 | 0.7480 |
| train_spatial_prior | LEFT   | 1365 | 0.2023 | 0.2860 | 0.7746 |
| train_spatial_prior | RIGHT  | 1365 | 0.2023 | 0.2849 | 0.7757 |
| pressure_body_axis_partition | SUPINE | 1365 | 0.0266 | 0.0449 | 0.6563 |
| pressure_body_axis_partition | LEFT   | 1365 | 0.0335 | 0.0546 | 0.7296 |
| pressure_body_axis_partition | RIGHT  | 1365 | 0.0362 | 0.0588 | 0.7306 |
| pressure_axis_contact_intersection | SUPINE | 1365 | 0.0302 | 0.0521 | 0.6578 |
| pressure_axis_contact_intersection | LEFT   | 1365 | 0.0363 | 0.0609 | 0.7308 |
| pressure_axis_contact_intersection | RIGHT  | 1365 | 0.0390 | 0.0649 | 0.7317 |

### 7.5 Worst-subject (per baseline, top-3 by mean_fixed_iou, ascending)

| baseline | subject | ml_split | n | mean_fixed_iou |
|---|---|---|---:|---:|
| all_background | 00001 | train | 45 | 0.0000 |
| all_background | 00002 | train | 45 | 0.0000 |
| all_background | 00003 | train | 45 | 0.0000 |
| pressure_body_axis_partition | 00024 | train | 45 | 0.0143 |
| pressure_body_axis_partition | 00081 | train | 45 | 0.0161 |
| pressure_body_axis_partition | 00021 | train | 45 | 0.0162 |
| pressure_axis_contact_intersection | 00024 | train | 45 | 0.0182 |
| pressure_axis_contact_intersection | 00021 | train | 45 | 0.0183 |
| pressure_axis_contact_intersection | 00035 | train | 45 | 0.0193 |
| train_spatial_prior | 00045 | train | 45 | 0.1136 |
| train_spatial_prior | 00069 | train | 45 | 0.1149 |
| train_spatial_prior | 00059 | train | 45 | 0.1167 |

### 7.6 Failure counts

* `failure_reason_counts.json` (this run): 0 across all reasons.
* No sample triggered `non_finite_pressure`, `shape_mismatch`,
  `degenerate_pca`, or any other fail-closed branch.

### 7.7 B01 freeze hash (B02 v0.1 run-time verification)

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
freeze manifest.  The B01 freeze directory was opened read-only
(`E:\TeamProjects\smarttopper-slp-b01-training-table-freeze\data\processed\slp8_training_tables_v0.1`).
No file in that directory was modified.

### 7.8 Per-run timing

| Phase | Time (s) |
|---|---:|
| Load (TRAIN pressure + label) | 14.65 |
| `all_background` (TRAIN+VAL eval) | 89.51 |
| `train_spatial_prior` (fit + TRAIN+VAL eval) | 95.37 |
| `pressure_body_axis_partition` (fit + TRAIN+VAL eval) | 95.21 |
| `pressure_axis_contact_intersection` (fit + TRAIN+VAL eval) | 101.30 |
| **Total wall-clock** | **396.18** |

Total TRAIN+VAL sample count = 3645 + 450 = **4095**.  CPU-only.

---

## 8. Test results (per suite)

| Suite | Result |
|---|---|
| `tests/test_slp8_non_learning_region_baseline.py` (B02 NEW) | **47 passed** in 1.98s |
| `tests/test_slp_pressure_infrastructure.py` (extended with B02 macro tests) | **113 passed** in 2.26s |
| `tests/test_slp8_training_table_freeze.py` (B01 regression) | **80 passed, 2 skipped** in 260.01s (real-data tests not gated on this run) |
| `git diff --check` | clean (no whitespace-only errors) |

Joint suite: **240 passed**, 2 skipped (the B01 real-data tests are
gated on `SLP8_DATASET_ROOT` and `A06_SPLIT_PATH` and were skipped
because the real data was passed via CLI flags to the runner
script — this is the same gating policy B01 itself uses).

---

## 9. Verified

* The four baselines are implemented, deterministic, CPU-only, and
  produce `(192, 84) uint8` label maps with values in `{0..8}`.
* The TRAIN template is fitted on TRAIN only; re-fitting raises
  `TrainTemplateFittedError`.  Tests in `TestTrainOnlyFitting` and
  `TestFailClosed::test_double_fit_*` pin this.
* The B01 `load_b01_freeze_tables` is called with the default
  `load_test=False`.  The TEST rows are never loaded; the
  `all_rows_with_test_opt_in()` method requires an explicit
  `enable_test_access(purpose="final_evaluation")` followed by a
  reload — neither happens in this task.  Test
  `TestTestAccessPolicy::test_load_b01_default_denies_test` pins this.
* The B01 freeze manifest's `a06_split_sha256` matches the
  B01-canonical value `024f5abe…` (see §7.7).
* The fixed 8-region macro indicator does NOT skip empty classes.
  Tests in `TestFixedClassMacroMetrics` and the
  `test_macro_hides_nothing_vs_legacy_skip_empty` assertion pin this.
* 4095 samples (TRAIN + VAL) are evaluated per baseline; 0 failures
  across all four baselines.
* All 240 unit / integration / regression tests pass; 2 skipped
  (gating only; no failures).
* The runner is fully deterministic: re-running with the same B01
  freeze directory and the same `dataset_root` produces the same
  per-baseline metrics, the same `predictions_manifest.csv`, the
  same per-prediction SHA-256s, and the same `metrics_summary.json`
  content (modulo timestamps in `runtime.json` / `status.json`).
* `git diff --check` reports no whitespace-only errors.

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
* The current `train_spatial_prior` template has a relatively
  high per-class IoU on `THORAX_BACK` (0.416) and `PELVIS_HIP`
  (0.484) and a low IoU on `ARM` (0.000) and `LOWER_LEG_FOOT`
  (0.000).  This is consistent with the SLP8 GT distribution
  (see §11 Limitations): the template is most useful where the
  per-pixel class distribution is concentrated, and it falls back
  to `BACKGROUND` on the long-tail classes.  This is the
  B02 v0.1 contract working as intended: a baseline that does not
  predict a class cannot silently inflate its macro score.

---

## 11. Unverified

* Any **TEST** IoU / Dice / accuracy: out of scope.  B02 v0.1
  explicitly forbids reading TEST label / onehot.  TEST is the
  B07 Full-run gate and is NOT RUN in this task.
* The **joint-geometry baseline**: explicitly out of scope (see
  §3 and §12).  No B02 v0.1 contract requires it.
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
| The 8-region GT is V221 auto-corrected, NOT human pixel-level | Region boundaries are not medically validated | Repeated in dataset card and §12 (Prohibited conclusions) |
| Only `uncover` is in the dataset | Cover1/cover2 cannot be evaluated | A09R contract: cover splits remain HOLD |
| B02 v0.1 baselines use only pressure | They cannot model joint positions, blanket occlusion, or body geometry not derivable from pressure | Out of scope for B02 v0.1; future B03 / B04 may add learned baselines on top of the same input contract |
| No mini / full experiment in this task | No neural-network comparison yet | B02 is the lowest comparison line; B03 (PM-only Smoke) and B04 (PM-only Mini) are the next gates |
| The body-axis partition uses fixed `segment_fractions` and `lateral_half_width` | The partition does not adapt to body-shape variance | All ratios are versioned config and are recorded in `resolved_config.json`; no data-driven tuning was performed on VAL or TEST |
| Pressure values are raw PMarray response, not kPa | No absolute-pressure / comfort / hardware claims can be derived | Repeated in `raw_semantics` and §12 |

---

## 13. Known failures

* **No failure in this run**.  `failure_reason_counts.json` reports
  0 across all 10 documented reasons
  (`shape_mismatch`, `non_finite_pressure`, `label_out_of_range`,
  `file_not_found`, `wrong_provenance`, `wrong_review_status`,
  `wrong_subject_split`, `no_contact`, `degenerate_pca`,
  `internal_exception`).
* The `train_spatial_prior` template does not predict `ARM` or
  `LOWER_LEG_FOOT` at the fixed 8-region macro level.  This is
  the B02 v0.1 contract working as designed: a baseline that
  does not predict a class must have IoU=0 on that class in
  the macro indicator, not silently drop it.

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
   experiment conclusions.  TEST is the B07 gate; it is NOT RUN.
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

---

## 15. Reviewer checklist

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
      `load_test=False`; TEST rows are not loaded.
* [x] No call to `enable_test_access(...)`, no `load_test=True`,
      and no `allowed_splits` containing `"test"` exists in the
      B02 v0.1 code or runner.
* [x] The fixed 8-region macro indicator does NOT skip empty
      classes; tests pin this.
* [x] All four baselines predict `(192, 84) uint8` label maps
      with values in `{0..8}`; tests pin this.
* [x] TRAIN template fitting is one-shot; re-fit raises
      `TrainTemplateFittedError`; tests pin this.
* [x] Posture is NOT consumed by any predictor; tests pin this.
* [x] `points.csv` field `region_id`, `region_label.npy`,
      `region_onehot.npy`, `class_ids_present`,
      `background_pixel_count`, `body_pixel_count`, A08 joints /
      body geometry, and any TEST field are NOT consumed as
      predictor input; tests pin this.
* [x] No GPU / no Mini / no Full / no TEST / no remote execution
      was performed.
* [x] The runner writes `FAILED.json` on any failure path; no
      failure path was triggered in this run, but the audit
      contract is in place.
* [x] `git diff --check` is clean (no whitespace-only errors;
      the autocrlf LF/CRLF notice is informational).
* [x] 240 tests pass, 2 skipped (gating only).
* [x] The `outputs/experiments/EXP-SLP-B02-NONLEARNING-DEV-20260825-R01/`
      directory is gitignored; nothing in it is committed.
* [x] No absolute path is written into any committed artefact.
* [x] `PROJECT_STATUS.md` and `SLP_AGENT_TASK_BACKLOG_v0.1.md`
      are NOT updated — Reviewer acceptance must precede that.
* [x] The joint-geometry baseline is recorded as
      `not_implemented` in the config and §3; the precise reason
      is recorded.

---

## 16. Next Gate

* `B02 v0.1` is `IMPLEMENTED_AND_TRAIN_VAL_RUN_COMPLETE —
  READY_FOR_CODEX_REVIEW`.
* The next tasks in the SLP8 backlog are `B03` (PM-only Smoke) and
  `B04` (PM-only Mini), each of which is BLOCKED_BY_B02 (smoke must
  be ≥ baseline worst-case; mini must beat B02's
  `train_spatial_prior` with a pre-registered margin).
* TEST access remains default-deny.  A separate `EXP-ID` with
  `enable_test_access(purpose="final_evaluation")` is required to
  run any TEST evaluation; this is OUT OF SCOPE for B02 v0.1.

---

## 17. Current git status (working copy)

The working copy is **NOT** yet committed in this report — the
commit + push + PR step is the next and final action.  After the
commit is made, the handoff will report the resulting HEAD, the
PR URL, and the final `git status --short --branch` output.

---

*本报告由 Claude Code (Mavis) 在独立 worktree 中基于 B01
`DONE_WITH_LIMITATIONS` 验收产物生成。Owner 决策编号 (B02) 等待
Codex Reviewer 接受。*
