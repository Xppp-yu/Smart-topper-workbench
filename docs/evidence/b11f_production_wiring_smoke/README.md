# B11F production-wiring smoke R07 evidence

This directory preserves the sanitized JSON summary from local smoke
`SMOKE-SLP-B11F-PRODUCTION-WIRING-20260904-R07` so the pushed Git baseline retains
the evidence reviewed locally.

- `R07_summary.json` is byte-identical to the generated ignored artifact
  `outputs/analysis/b11f_production_wiring_smoke_20260904_r07.json` at evidence capture.
- The JSON contains no absolute dataset path, credential, raw array, model checkpoint or formal EXP-ID.
- It records CPU-only one-microbatch wiring evidence, not a final fit or validation metric.
- TEST rows/labels/onehot remain zero; GPU was not run; AutoDL was not connected.
- `SHA256SUMS` binds the committed evidence files.

The source task and limitations are documented in
`docs/tasks/TASK_SLP_B11F_PRODUCTION_WIRING_SMOKE_v0.1.md` and
`docs/stage_reports/S2_B11F_PRODUCTION_WIRING_SMOKE_v0.1.md`.
