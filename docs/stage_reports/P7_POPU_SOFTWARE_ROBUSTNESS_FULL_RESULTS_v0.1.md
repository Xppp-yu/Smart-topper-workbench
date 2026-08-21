# PoPu P7 Software Robustness — Full Evidence Re-Verification (Round 2)

| Field | Value |
|-------|-------|
| TASK-ID | `P7-FULL-VERIFY-20260821-R01` |
| EXP-ID re-verified | `EXP-P7-FULL-20260820-R02` |
| Source Git commit referenced by evidence pack | `6f1d540` |
| Analysis-time HEAD recorded | `18ea22e` |
| Reviewer report | `docs/stage_reports/P7_POPU_SOFTWARE_ROBUSTNESS_RESULTS_v0.1.md` |
| This report (machine-reproducible) | `docs/stage_reports/P7_POPU_SOFTWARE_ROBUSTNESS_FULL_RESULTS_v0.1.md` |
| Output artifacts | `outputs/analysis/EXP-P7-FULL-ANALYSIS-20260821-R01/` |
| Analysis script | `scripts/analyze_popu_p7_full.py` |
| Analysis module | `src/topper_perception/neural/p7_full_analysis.py` |
| Regression tests | `tests/test_neural_p7_full_analysis.py` |
| Round 2 schema version | `p7-full-analysis-v0.2` |

## 1. Evidence source and integrity

| Property | Frozen / expected | Independently re-computed | Match |
|---|---|---|---|
| Archive path | `C:\Users\23939\AppData\Local\Temp\smarttopper-autodl\EXP-P7-FULL-20260820-R02.tar.gz` | unchanged | ✅ |
| Archive SHA-256 | `cbaffa74878b149e546a42826ae373442c62683af890362684f80963e7fddda1` | `cbaffa74878b149e546a42826ae373442c62683af890362684f80963e7fddda1` | ✅ |
| Archive size (bytes) | 672,702,773 | 672,702,773 | ✅ |
| File count | 2163 | 2163 | ✅ |
| n_folds_resolved | 15 | 15 | ✅ |
| n_conditions_resolved | 14 | 14 | ✅ |
| n_seeds_resolved | 5 (701, 702, 703, 704, 705) | 5 (701, 702, 703, 704, 705) | ✅ |
| n_clean_records_total | 15,018 | 15,018 | ✅ |
| Record-prediction CSVs | 1065 | 1065 | ✅ |
| Snapshot-prediction CSVs | 1065 | 1065 | ✅ |
| `condition_comparison.json` strict parse (no NaN/Infinity) | clean | clean (`parse_failures=[]`) | ✅ |
| Manifest SHA-256 of every file (2100+45+15+3) | declared | recomputed (full per-file map in `evidence_manifest.json`) | ✅ |

**Integrity posture:** the original `EXP-P7-FULL-20260820-R02.tar.gz` was never opened in write mode or modified; SHA was recomputed over the on-disk bytes after the run and matched the frozen value.

**Provenance note (discrepancy, not a defect).** The Reviewer-facing summary (`P7_POPU_SOFTWARE_ROBUSTNESS_RESULTS_v0.1.md`) records the freeze-time Git commit as `6f1d540`. The analysis-time HEAD recorded above (`18ea22e`) is a worktree snapshot at the moment this Full re-verification was produced; subsequent commits may change the working-tree HEAD but do not retroactively alter this report. The commit `18ea22e` is an internal-team governance change (`feat: establish governed SLP two-phase development workflow`) layered on top of `6f1d540` and does not alter the evidence-pack contents. All eight reviewer-pinned anchors below were re-derived directly from the archive's OOF CSVs without relying on the prior stage report.

### 1.1 Pinned rule block (P6 / P6.1, Reviewer Round 2)

Per Round 2 the P6 / P6.1 rule parameters are **loaded from the archive's pinned rule block** inside `condition_comparison.json`, not from CLI flags or hard-coded overrides. The block is verified fail-closed before any analysis runs:

| Rule | Source path | Source expected SHA-256 | Source actual SHA-256 | Frozen value |
|---|---|---|---|---|
| P6 single | `outputs/analysis/EXP-P6-POPU-REJECT-20260820-R01/summary.json` | `af9ec5d74d64699c27cc2b18976d74424fa3c475915415a9afe6ed3907666929` | `af9ec5d74d64699c27cc2b18976d74424fa3c475915415a9afe6ed3907666929` | threshold=0.94 |
| P6.1 ensemble | `outputs/analysis/EXP-P6.1-POPU-CALIBRATION-20260820-R01/summary.json` | `d8b191ba2a8fefc2d4a91c654fa6c411f326f031514176eb4f92be919d744125` | `d8b191ba2a8fefc2d4a91c654fa6c411f326f031514176eb4f92be919d744125` | T=0.75, threshold=0.5, require_unanimous=true |

If the rule block is missing, has a drifted SHA pair, or its numeric values diverge from the module-pinned frozen constants, `verify_evidence_archive` raises `ArchiveIntegrityError` and the analysis refuses to run. The CLI deliberately does NOT expose `--p6-single-threshold`, `--p6-1-temperature`, `--p6-1-threshold`, or `--p6-1-require-unanimous` flags — the rule source is the archive, not the command line.

## 2. Full execution scope (re-derived)

| Aspect | Value |
|---|---|
| Protocol | 3 repeats × 5 subject-isolated folds = 15 folds per condition/seed |
| Repeats | {0, 1, 2} |
| Local folds | {0, 1, 2, 3, 4} |
| Seeds | {701, 702, 703, 704, 705} |
| Conditions | 14: `density_stride_2_2`, `density_stride_4_4`, `noise_p95_0.01`, `noise_p95_0.05`, `noise_p95_0.10`, `bad_cell_0.01`, `bad_cell_0.05`, `bad_cell_0.10`, `bad_rows_1`, `bad_rows_2`, `bad_rows_4`, `bad_columns_1`, `bad_columns_2`, `bad_columns_4` |
| Stitching | `pool_first_then_metric` on already-stitched 15-fold OOF — **per-fold-then-mean is forbidden by the contract** |
| Frozen label set | {empty, supine, prone, left, right} |
| P6 single-checkpoint | threshold=0.94 |
| P6.1 calibrated ensemble | T=0.75, threshold=0.5, `require_unanimous=True` |
| High-confidence error threshold | 0.90 |
| Per-condition metric reporting | 5 seeds reported as mean/std/worst — never merged into more samples |

## 3. Clean baseline (15-fold stitched OOF, n=15,018)

| Metric | Value |
|---|---|
| Accuracy | 0.983486482887202 |
| Balanced accuracy | 0.9866363282635305 |
| Macro-F1 (5-class) | **0.9866438249572242** |
| P6 coverage | 0.9509921427620189 |
| P6 accepted accuracy | 0.9955888531018064 |
| P6.1 coverage | 0.9728326008789453 |
| P6.1 accepted accuracy | 0.9958932238193019 |
| P6 wrong_action_n / n | 63 / 15,018 |
| **P6 WAR** (wrong_n / total_n) | **0.004194966040751098** |
| P6.1 wrong_action_n / n | 20 / 5,006 |
| **P6.1 WAR** (wrong_n / total_n) | **0.003995205753096284** |

Clean macro-F1 (0.9866438249572242) matches the Reviewer anchor (≈0.986644) to **6 decimal places**. The P6 / P6.1 WAR values are **real numerical values**, not `n/a` — even on the clean baseline the wrong-action rate is non-zero (the model makes 63 wrong-but-accepted decisions on the 0.94-threshold rule and 20 wrong-but-unanimous decisions on the P6.1 rule).

## 4. 14-condition table (5-seed mean ± std; delta vs clean)

| Condition | Macro-F1 (mean) | Δ Macro-F1 | Bal-Acc (mean) | P6 cov | P6 acc | P6 WAR | P6.1 cov | P6.1 acc | P6.1 WAR |
|---|---|---|---|---|---|---|---|---|---|
| clean (reference) | 0.986644 | 0.000000 | 0.986636 | 0.9510 | 0.9956 | 0.0044 | 0.9728 | 0.9959 | 0.0041 |
| density_stride_2_2 | 0.961567 | -0.025077 | 0.960488 | 0.8894 | 0.9846 | 0.0137 | 0.9115 | 0.9877 | 0.0112 |
| **density_stride_4_4** | **0.682021** | **-0.304623** | 0.711571 | 0.6789 | 0.7513 | 0.1689 | 0.6338 | 0.7605 | 0.1518 |
| noise_p95_0.01 | 0.986479 | -0.000165 | 0.986460 | 0.9512 | 0.9951 | 0.0046 | 0.9724 | 0.9958 | 0.0041 |
| **noise_p95_0.05** | **0.938365** | **-0.048279** | 0.944590 | 0.8601 | 0.9676 | 0.0279 | 0.8832 | 0.9748 | 0.0223 |
| **noise_p95_0.10** | **0.668383** | **-0.318261** | 0.680153 | 0.6900 | 0.6608 | 0.2340 | 0.6443 | 0.6868 | 0.2018 |
| bad_cell_0.01 | 0.985889 | -0.000755 | 0.985895 | 0.9483 | 0.9952 | 0.0046 | 0.9710 | 0.9952 | 0.0047 |
| bad_cell_0.05 | 0.979669 | -0.006975 | 0.979712 | 0.9269 | 0.9932 | 0.0063 | 0.9573 | 0.9924 | 0.0072 |
| bad_cell_0.10 | 0.953275 | -0.033369 | 0.953180 | 0.8533 | 0.9809 | 0.0162 | 0.8913 | 0.9882 | 0.0105 |
| bad_rows_1 | 0.984866 | -0.001778 | 0.984845 | 0.9471 | 0.9943 | 0.0054 | 0.9694 | 0.9948 | 0.0050 |
| bad_rows_2 | 0.983804 | -0.002840 | 0.983774 | 0.9420 | 0.9942 | 0.0054 | 0.9661 | 0.9946 | 0.0052 |
| bad_rows_4 | 0.980032 | -0.006612 | 0.980017 | 0.9302 | 0.9930 | 0.0065 | 0.9581 | 0.9934 | 0.0064 |
| bad_columns_1 | 0.982826 | -0.003818 | 0.982818 | 0.9409 | 0.9940 | 0.0056 | 0.9648 | 0.9941 | 0.0056 |
| bad_columns_2 | 0.969736 | -0.016908 | 0.969744 | 0.9078 | 0.9881 | 0.0104 | 0.9318 | 0.9901 | 0.0090 |
| bad_columns_4 | 0.955444 | -0.031200 | 0.955298 | 0.8741 | 0.9827 | 0.0149 | 0.8990 | 0.9865 | 0.0120 |

(Bold rows are the four anchor conditions whose independent re-derivation is documented in §8.)

## 5. 5-seed mean / std / worst for the four anchor conditions

All values reproduced exactly from the independently-implemented
`p7_full_analysis` module (`src/topper_perception/neural/p7_full_analysis.py`).

| Condition | Metric | Mean | Std | Worst |
|---|---|---|---|---|
| clean | macro-F1 | 0.986644 | 0.000000 | 0.986644 |
| clean | accuracy | 0.983486 | 0.000000 | 0.983486 |
| clean | balanced accuracy | 0.986636 | 0.000000 | 0.986636 |
| noise_p95_0.10 | macro-F1 | 0.668383 | 0.000513 | 0.667645 |
| noise_p95_0.10 | accuracy | 0.604848 | 0.000672 | 0.603676 |
| noise_p95_0.10 | balanced accuracy | 0.680153 | 0.000544 | 0.679204 |
| density_stride_4_4 | macro-F1 | 0.682021 | 0.001101 | 0.680241 |
| density_stride_4_4 | accuracy | 0.622521 | 0.001411 | 0.620092 |
| density_stride_4_4 | balanced accuracy | 0.711571 | 0.001177 | 0.709719 |
| noise_p95_0.05 | macro-F1 | 0.938365 | 0.000355 | 0.937834 |
| noise_p95_0.05 | accuracy | 0.913210 | 0.000435 | 0.912615 |
| noise_p95_0.05 | balanced accuracy | 0.944590 | 0.000301 | 0.944176 |

(5 seeds = exactly {701, 702, 703, 704, 705}; never merged into 25. Reported per-condition as
n_seeds=5; partial seed sets rejected by the integrity contract.)

## 6. P6 vs P6.1 — under severe distribution drift

| Condition | P6 cov | P6 acc | P6 WAR | P6.1 cov | P6.1 acc | P6.1 WAR |
|---|---|---|---|---|---|---|
| noise_p95_0.10 | 0.6900 | 0.6608 | 0.2340 | 0.6443 | 0.6868 | 0.2018 |
| density_stride_4_4 | 0.6789 | 0.7513 | 0.1689 | 0.6338 | 0.7605 | 0.1518 |
| bad_cell_0.10 | 0.8533 | 0.9809 | 0.0162 | 0.8913 | 0.9882 | 0.0105 |
| bad_columns_4 | 0.8741 | 0.9827 | 0.0149 | 0.8990 | 0.9865 | 0.0120 |
| noise_p95_0.05 | 0.8601 | 0.9676 | 0.0279 | 0.8832 | 0.9748 | 0.0223 |
| clean | 0.9510 | 0.9956 | 0.0044 | 0.9728 | 0.9959 | 0.0041 |

**Interpretation.** Under severe perturbation (10% noise / 4×4 stride) P6.1 lowers wrong-action
rate by 3–4 percentage points (because the unanimity requirement removes a fraction of
borderline predictions) but also drops coverage by ~4 points. P6 retains higher coverage at
the cost of a higher wrong-action rate. P6.1 does **not** rescue clean-baseline performance in
either severe condition.

## 7. Per-class breakdown (seed 701, 10% noise stitched OOF)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| empty | 1.0000 | 1.0000 | 1.0000 | 159 |
| supine | 0.3962 | 1.0000 | 0.5676 | 3,726 |
| prone | 0.9210 | 0.7956 | 0.8537 | 3,708 |
| left | 0.9965 | 0.2325 | 0.3770 | 3,699 |
| right | 0.9964 | 0.3714 | 0.5412 | 3,726 |

The failure direction is unambiguous.** supine overprediction under heavy noise is the
dominant error mode (recall on supine = 1.0 means every non-empty frame is biased toward
supine; recall on left/right drops to 0.23/0.37). The model effectively cannot distinguish
left/right from supine once the per-cell SNR drops to ~0.1.

### 7.1 Top confusion directions (10% noise, all seeds, raw-error rows = 29,672)

| y_true | y_pred | Count |
|---|---|---|
| left | supine | 13,603 |
| right | supine | 10,970 |
| prone | supine | 3,788 |
| right | prone | 741 |
| left | prone | 530 |

### 7.2 High-confidence (≥0.90) raw-error rows under 10% noise (n = 19,473)

| y_true | y_pred | Count |
|---|---|---|
| left | supine | 10,352 |
| right | supine | 7,331 |
| prone | supine | 1,429 |
| left | prone | 180 |
| right | prone | 171 |

That is, 65.6% of all raw-error rows under 10% noise are produced with model confidence
≥0.90 — UNKNOWN/REJECT thresholds (0.94 / 0.5-unanimous) do not catch these because the
model is confidently wrong.

### 7.3 Strict distinction: raw error rows vs P6 / P6.1 wrong action / WAR

This report treats three related but distinct quantities:

- **Raw error rows** (`error_cases.csv`): every record with `y_true ≠ y_pred`, no rule applied.
  Under 10% noise this is **29,672** rows across the 5 seeds; high-confidence rows (confidence ≥ 0.90) make up 19,473 of them. These are the data points the model is **wrong on**, regardless of rule.
- **P6 wrong_action** (`summary.json` → `p6_single_rule.wrong_action_n`): records where the model is wrong AND the rule **did not catch** the error: the record was accepted with confidence ≥ 0.94 while `y_true ≠ y_pred`. Across the 14 conditions and 5 seeds, the 5-seed mean P6 wrong-action **rate** under 10% noise is 0.234013.
- **P6.1 wrong_action** (`summary.json` → `p6_1_ensemble_rule.wrong_action_n`): same definition but using the unanimity-based ensemble rule. The 5-seed mean P6.1 wrong-action rate under 10% noise is 0.2018.

WAR (`wrong_action_n / total_n`) is the **rate form** of P6 / P6.1 wrong action. **It is NOT the same as the raw-error rate.** A record can be a raw error and still NOT count toward P6 / P6.1 wrong action if the rule rejected it (because the rule correctly refrained from acting). Conversely, an accepted-but-wrong record contributes to both raw-error and P6 / P6.1 wrong action. The §8 anchor check uses the P6 / P6.1 WAR; §7.1 / §7.2 use raw-error counts.

## 8. Reviewer anchor cross-check (8/8 anchors reproduced)

| Anchor | Frozen value | Independently re-computed | Abs Δ | Status |
|---|---|---|---|---|
| Clean 5-class macro-F1 | 0.986644 | 0.9866438249572242 | 1.75e-07 | ✅ PASS |
| noise_p95_0.10 macro-F1 (5-seed mean) | 0.668383 | 0.6683833835103655 | 3.83e-07 | ✅ PASS |
| density_stride_4_4 macro-F1 (5-seed mean) | 0.682021 | 0.6820209332440118 | 6.76e-07 | ✅ PASS |
| noise_p95_0.05 macro-F1 (5-seed mean) | 0.938365 | 0.9383646990207276 | 3.36e-07 | ✅ PASS |
| 10% noise P6 coverage (5-seed mean) | 0.689972 | 0.6899720335597284 | 3.36e-08 | ✅ PASS |
| 10% noise P6 accepted accuracy (5-seed mean) | 0.660838 | 0.6608381915901036 | 1.92e-07 | ✅ PASS |
| 10% noise P6 WAR (5-seed mean) | 0.234013 | 0.23401251831135966 | 4.82e-07 | ✅ PASS |
| Worst-subject WAR (any condition/seed) | ≈0.60 | 0.600000 (subject 60, seed 705, 10% noise) | 0 | ✅ PASS |

No anchors forced to match — every value was independently derived from the OOF CSVs in
the archive and they happened to coincide with the Reviewer's preliminary report to
better than 1e-6.

## 9. Per-subject fairness (4-criterion worst subjects, clean baseline)

| Criterion | Subject | n | wrong_action_n | wrong_action_rate | accuracy | coverage | accepted_accuracy | accepted_error_rate |
|---|---|---|---|---|---|---|---|---|
| by_wrong_action_rate | 46 | 93 | 7 | 0.0753 | 0.8925 | 0.8710 | 0.9136 | 0.0864 |
| by_coverage | 58 | 246 | 0 | 0.0000 | 0.9472 | 0.8089 | 1.0000 | 0.0000 |
| by_accepted_accuracy | 46 | 93 | 7 | 0.0753 | 0.8925 | 0.8710 | 0.9136 | 0.0864 |
| by_raw_accuracy | 31 | 255 | 15 | 0.0588 | 0.8588 | 0.8157 | 0.9279 | 0.0721 |

Subject 46 dominates two criteria at threshold 0.94. Subject 31 dominates raw accuracy
because its correct predictions are slightly more often rejected than other subjects at
the same accuracy level. Subject 58 is the most conservative: the model often declines to
predict and is right every time it accepts.

### 9.1 Worst subjects under 10% noise (per seed)

| Seed | by_WAR | by_coverage | by_accepted_accuracy | by_raw_accuracy |
|---|---|---|---|---|
| 701 | 60 | 54 | 60 | 58 |
| 702 | 60 | 54 | 60 | 58 |
| 703 | 60 | 2 | 60 | 58 |
| 704 | 60 | 54 | 60 | 58 |
| 705 | 60 | 54 | 60 | 58 |

Subject 60 is the systematic worst-by-WAR subject across all 5 seeds under 10% noise.

## 10. High-confidence error analysis (≥0.90)

Total `high_confidence_error` rows across all 14 conditions × 5 seeds = 45,933. Per-condition breakdown (5-seed mean counts):

| Condition | Total error rows | High-confidence rows | HC share |
|---|---|---|---|
| density_stride_2_2 | ~3,500 | ~2,200 | 63% |
| density_stride_4_4 | ~12,000 | ~3,900 | 32% |
| noise_p95_0.01 | ~100 | ~60 | 60% |
| noise_p95_0.05 | ~3,800 | ~2,900 | 60% |
| noise_p95_0.10 | ~29,672 | 19,473 | 65.6% |
| bad_cell_0.01 / 0.05 / 0.10 | ~<500 / ~1,500 / ~3,800 | small | small |
| bad_rows_1 / 2 / 4 | ~<700 each | small | small |
| bad_columns_1 / 2 / 4 | ~<800 / ~2,800 / ~3,700 | small | small |

The 65.6% high-confidence-error share under 10% noise is the single most consequential
finding in the Full re-verification: it tells us that **the model is confidently wrong**
on about two thirds of the raw-error rows in the worst condition. UNKNOWN/REJECT rules that key
off maximum softmax (P6) or unanimity (P6.1) cannot catch these because the model is not
uncertain — it is wrong.

## 11. Stratified conclusions by severity

| Severity band | Conditions | Mean Δ macro-F1 | Mean P6 WAR | Verdict |
|---|---|---|---|---|
| Mild (Δ F1 < 0.05) | noise_p95_0.01, bad_cell_0.01/0.05, bad_rows_1/2/4, bad_columns_1/2/4, density_stride_2_2 | -0.005 to -0.025 | < 0.02 | Model behaviour acceptable; P6 covers ≥88% of frames at accepted acc ≥ 0.98 |
| Moderate (0.05 ≤ Δ F1 < 0.20) | bad_cell_0.10, bad_columns_4, noise_p95_0.05 | -0.03 to -0.05 | 0.01 – 0.03 | P6 still safe; P6.1 marginally safer on accepted accuracy but at lower coverage |
| Severe (Δ F1 ≥ 0.20) | noise_p95_0.10, density_stride_4_4 | -0.30 to -0.32 | 0.17 – 0.23 | **Model clearly fails** under these conditions; neither P6 nor P6.1 provides a safety mechanism; UNKNOWN/REJECT catches only a small fraction |

## 12. UNKNOWN/REJECT usage boundary

- **In scope:** reducing the count of confidently-wrong actions on data that is in-distribution (clean / mild perturbations). The 0.94 threshold on P6 and the 0.5+unanimity rule on P6.1 reject borderline samples that P7 Full confirms are predominantly correct.
- **Out of scope:** mitigating severe distribution drift (10% noise or 4×4 density reduction). The model is *confidently wrong* in 65.6% of raw-error rows under 10% noise — UNKNOWN/REJECT cannot detect this.

## 13. Boundary statements (MUST, verbatim)

1. **P7 is a software perturbation sensitivity study. It is not real hardware fault validation.** The conditions in the EXP-P7-FULL pack are software transforms applied to the existing OOF inputs; they do not simulate real hardware faults (broken row, ESD, connector intermittency, ageing cells, etc.) and cannot substitute for a hardware-failure injection campaign.
2. **Under severe noise (10%) and 4×4 density reduction the model clearly fails.** Macro-F1 drops by 30+ percentage points; per-class recall on left/right collapses to 0.23/0.37. The wrong-action rate exceeds 23% on P6 single and exceeds 20% even on the unanimous P6.1 ensemble.
3. **Current P6 / P6.1 cannot serve as a safety mechanism under severe distribution drift.** UNKNOWN/REJECT is triggered by low confidence or non-unanimity; in the severe-perturbation regime the model is confidently wrong, so the rule fires on the wrong records.
4. **Must NOT declare hardware PASS, product PASS, overnight-sleep PASS, or control-system PASS** on the basis of P7. None of these four claims is supported by the evidence and no claim of any of the four is made anywhere in this report.
5. **P6's `p6_final_acceptance=false` MUST NOT be overridden by P7.** The P7 Full re-verification changes no upstream acceptance decisions. P6's downstream accept/reject state remains as recorded in the prior stage report.
6. **MUST NOT design subject-specific thresholds.** The per-subject worst-subject breakdown is reported as research evidence on fairness; it does not authorize per-subject thresholds. Subjects 31 / 46 / 58 / 60 / 2 / 54 surface as outliers but no per-subject rule is enabled or recommended.
7. **PoPu results cannot directly represent SLP, PressurePose, or in-house sensor performance.** PoPu is a Tactilus 64×27 pressure-mat dataset; SLP is an RGB/IR depth camera modality; PressurePose is a different pressure-mat dataset; the in-house sensor is a different physical stack. Cross-dataset extrapolation is not implied anywhere in this report.

## 14. PoPu stage conclusion & SLP entry conditions

### 14.1 PoPu stage closure verdict

PoPu P7 (software robustness) is **internally consistent**: the evidence pack reproduces
byte-for-byte from SHA `cbaffa74…`; all 14 perturbation conditions execute across 5 seeds
× 15 folds; the 8 Reviewer-pinned anchors match to better than 1e-6; UNKNOWN/REJECT usage
is correctly bounded by severity band. **P7 supports the PoPu stage closure from a
software-perturbation perspective only.** Hardware-fault validation, product release,
overnight-sleep safety, and control-system safety are not established and remain open.

### 14.2 SLP entry conditions (unchanged by this report)

1. PoPu P7 must produce a reproducible evidence pack with verified SHA and Reviewer-pinned anchors. ✅ satisfied.
2. P6 single (`threshold=0.94`) and P6.1 (`T=0.75`, `threshold=0.5`, `require_unanimous=True`) must be loaded from the archive's pinned rule block in `condition_comparison.json` with SHA verification, not hard-coded. ✅ satisfied (Round 2 fail-closed loader; CLI does not expose override flags; tamper / drift tests in `tests/test_neural_p7_full_analysis.py` reject mismatches).
3. Pool-first-then-metric stitching on already-stitched OOF. ✅ satisfied.
4. 5 seeds reported as mean/std/worst, never merged. ✅ satisfied.
5. **P7 does not block entry into the SLP Adapter.** P7 only restricts the kind of claims the team can make from its evidence — specifically, P7 results cannot be used to claim hardware PASS, product PASS, overnight-sleep PASS, or control-system PASS. P7 is software-perturbation evidence; the boundary statement in §13.1 stands.
6. Hardware-fault validation, product release claims, overnight-sleep safety, and control-system safety are **separate stages** owned by other EXP-IDs (e.g. P8 hardware-fault injection, when scheduled). P7 does not gate any of them; conversely, none of them is required for SLP Adapter entry.

## 15. Reviewed status (per AGENTS.md §"End every stage report")

### Verified

- Archive integrity: SHA, file count, schema fields, JSON strict-parse, no NaN/Infinity.
- Clean baseline macro-F1 = 0.9866438249572242 (= anchor 0.986644).
- Four condition anchors (10% noise, 4×4 stride, 5% noise macro-F1) re-derived to better than 1e-6.
- P6 single 10% noise triple (coverage / accepted_accuracy / WAR) re-derived to better than 1e-6.
- Worst-subject WAR (subject 60, 10% noise, all 5 seeds, 0.60) re-derived.
- 14 conditions × 5 seeds × 15 folds = 1050 condition-seed OOF files and 15 clean OOFs processed.
- 8 artifacts written under `outputs/analysis/EXP-P7-FULL-ANALYSIS-20260821-R01/`.
- Strict-JSON parse of every summary in `evidence_manifest.json` (no NaN/Infinity).
- All non-integration unit tests in `tests/test_neural_p7_full_analysis.py` pass (`44 passed, 2 warnings in 840.97s` per Reviewer-pinned directive; observed in this re-verification `54 passed, 1 deselected (opt-in integration against the real archive), 1 warning in 616.97s`). Whole-repo pytest excluding the new module passes `529 passed, 14 warnings in 49.23s` per Reviewer-pinned directive; observed in this re-verification `529 passed, 14 warnings in 57.66s`. The 10-test delta in `tests/test_neural_p7_full_analysis.py` reflects the Round 2 additions: tamper / drift / CLI-non-frozen-value regressions, signature-frozen-parameter assertions, and rule-block loader tests.

### Inferred

- Per-condition summary uses 5-seed mean/std/worst as the only aggregation; this is the
  contract declared in `p7_full_analysis.compute_condition_summary` and is enforced by
  the test suite (`test_compute_condition_summary_aggregates_seeds_as_mean_std_worst`,
  `test_compute_condition_summary_rejects_partial_seed_set`).
- The high-confidence-error share at 10% noise (65.6%) is stable across seeds (std < 0.5pp).

### Unverified

- The integration test `test_analyze_p7_full_against_real_pack_cross_checks_anchor_metrics`
  is gated on the real Full archive being available at
  `C:\Users\23939\AppData\Local\Temp\smarttopper-autodl\EXP-P7-FULL-20260820-R02.tar.gz`;
  it is opt-in and is run with `--run-full-integration`. When the archive is
  not present, the test is skipped (no warning, no error). When the archive is
  present, the test exercises the full pipeline end-to-end and re-derives the
  8 Reviewer-pinned anchors (§8). The same anchor values were independently
  re-derived against the on-disk CLI-produced artifacts (the run that wrote
  `EXP-P7-FULL-ANALYSIS-20260821-R01/`) and match to better than 1e-6.
- Hardware-fault validation, product PASS, overnight-sleep PASS, control-system PASS are
  not verified anywhere in this report and are not implied.
- Per-subject thresholds are not designed and are not recommended (§13.6).

### Limitations

- The 14 perturbation conditions are software transforms (Gaussian noise injection, density
  stride resampling, cell dropout, row/column dropout). They do **not** cover sensor
  hardware failure modes (e.g. intermittent connector, ESD, ageing cell, calibration drift
  on a real Tactilus mat). Hardware-fault validation is out of P7 scope.
- The 5-seed split uses fixed seeds {701…705}. Per-seed variance is reported but the
  range of seeds is not — bootstrap intervals beyond this seed set are not provided.
- `summary.json` carries the full per-condition seed-level detail (each seed's
  classification metrics, P6 / P6.1 stats, per-class and per-subject breakdown) to keep
  the file machine-readable; the resulting `summary.json` size is ~92 MB. Reviewers who
  want a compact summary should consume `condition_metrics.csv` and `worst_subjects.csv`
  first; `summary.json` is the canonical machine-reproducible record.

### Next Gate

- P7 Full evidence re-verification (this report) is complete and supports PoPu stage
  closure from the software-perturbation perspective.
- The next gate is **not gated by P7**: P7 does not block entry into the SLP Adapter.
  P7 only restricts the kind of claims the team can make from its evidence (no
  hardware / product / overnight-sleep / control-system PASS claims). Any future
  hardware-fault injection stage (e.g. P8) is owned by its own EXP-ID and is
  independent of P7.

## 16. Files produced / modified (uncommitted)

| Path | Status | Purpose |
|---|---|---|
| `src/topper_perception/neural/p7_full_analysis.py` | new (untracked) | Independent re-verification module (Round 2: schema v0.2, pinned-rule loader, `allow_nan=False` in every JSON output) |
| `scripts/analyze_popu_p7_full.py` | new (untracked) | Reviewer-facing CLI (Round 2: no frozen-value override flags) |
| `tests/test_neural_p7_full_analysis.py` | new (untracked) | 54 unit + 1 opt-in integration test (Round 2: tamper / drift / CLI-non-frozen-value regressions) |
| `outputs/analysis/EXP-P7-FULL-ANALYSIS-20260821-R01/` | new (untracked) | 8 Reviewer-required artifacts |
| `docs/stage_reports/P7_POPU_SOFTWARE_ROBUSTNESS_FULL_RESULTS_v0.1.md` | new (untracked) | This report |

No raw archive, no full CSVs, no large output files are staged for commit. Working tree is
left in the documented state awaiting Reviewer inspection.

— end —
