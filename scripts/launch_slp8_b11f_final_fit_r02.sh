#!/usr/bin/env bash
set -euo pipefail

readonly B11F_MODE="${1:-}"
readonly B11F_REPO="/root/autodl-tmp/smarttopper-b11f-preflight-a6a5d8e-r03"
readonly B11F_FREEZE_DIR="/root/autodl-tmp/data/processed/slp8_training_tables_v0.1"
readonly B11F_DATA_ROOT="/root/autodl-tmp/datasets/SLP_8Region_Pressure_VAL_v1.1"
readonly B11F_CONFIG="$B11F_REPO/configs/experiments/slp8_pm_final_development_fit_v0.1.json"
readonly B11F_CANDIDATE="$B11F_REPO/configs/experiments/slp8_pm_research_candidate_v0.1.json"
readonly B11F_EXP_ID="EXP-SLP-B11F-PM-FINAL-FIT-20260904-AUTODL-R02"
readonly B11F_OUTPUT="$B11F_REPO/outputs/experiments/$B11F_EXP_ID"
readonly B11F_BUNDLE="/root/autodl-tmp/smarttopper-b11f-main-a6a5d8e.bundle"
readonly B11F_RUNNER_SHA="a6a5d8e6f8db003149169ee48f71d6e41e445a80"
readonly B11F_BUNDLE_SHA="5e9d855397face954cac18e3dbadb26449129f828f77d45412b3c4f30d8e6bb2"
readonly B11F_CONFIG_SHA="a6590d6f068644d98fa5340ec3d4a2e02171b529ec22ab092efb54a298925a43"
readonly B11F_CANDIDATE_SHA="34f0fcf45d07920b99b7baf6d595f61297f086ff3187c9ec9b3bd69400b2cd4b"
readonly B11F_FREEZE_SHA="42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04"
readonly B11F_AUTHORIZED_ENV_SHA="a5a9342b18d00b614355e63ce056a7edd92dd80358d8aead5ef6e8e0ba045669"
readonly B11F_MAX_SECONDS=2700

if [[ "$B11F_MODE" != "run" && "$B11F_MODE" != "resume" ]]; then
  printf '%s\n' "usage: $0 {run|resume}" >&2
  exit 2
fi

cd "$B11F_REPO"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$B11F_RUNNER_SHA"
test "$(git remote get-url origin)" = "$B11F_BUNDLE"
test "$(git rev-parse origin/main)" = "$B11F_RUNNER_SHA"
test "$(sha256sum "$B11F_BUNDLE" | cut -d' ' -f1)" = "$B11F_BUNDLE_SHA"
test "$(sha256sum "$B11F_CONFIG" | cut -d' ' -f1)" = "$B11F_CONFIG_SHA"
test "$(sha256sum "$B11F_CANDIDATE" | cut -d' ' -f1)" = "$B11F_CANDIDATE_SHA"
test "$(sha256sum "$B11F_FREEZE_DIR/freeze_manifest.json" | cut -d' ' -f1)" = "$B11F_FREEZE_SHA"
test -d "$B11F_DATA_ROOT"
test "${#B11F_AUTHORIZED_ENV_SHA}" -eq 64

readonly B11F_COMMON_ARGS=(
  --config "$B11F_CONFIG"
  --output-dir "$B11F_OUTPUT"
  --b01-freeze-dir "$B11F_FREEZE_DIR"
  --dataset-root "$B11F_DATA_ROOT"
  --experiment-id "$B11F_EXP_ID"
  --authorized-environment-sha256 "$B11F_AUTHORIZED_ENV_SHA"
  --run-authorized
)

if [[ "$B11F_MODE" == "run" ]]; then
  test ! -e "$B11F_OUTPUT"
  exec timeout --signal=INT --kill-after=2m "${B11F_MAX_SECONDS}s" \
    uv run --extra neural python scripts/run_slp8_region_final_fit.py \
    "${B11F_COMMON_ARGS[@]}"
fi

test -d "$B11F_OUTPUT"
test ! -e "$B11F_OUTPUT/DONE.json"
test ! -e "$B11F_OUTPUT/FAILED.json"
if test -e "$B11F_OUTPUT/RUNNING.json" && test -e "$B11F_OUTPUT/STOPPED.json"; then
  exit 1
fi
test -e "$B11F_OUTPUT/RUNNING.json" || test -e "$B11F_OUTPUT/STOPPED.json"

readonly B11F_REMAINING_SECONDS="$(
  uv run --extra neural python - "$B11F_OUTPUT/budget.json" "$B11F_MAX_SECONDS" <<'PY'
import json
import math
import sys
import time

with open(sys.argv[1], encoding="utf-8") as handle:
    budget = json.load(handle)
maximum = int(sys.argv[2])
remaining = math.floor(float(budget["deadline_utc_epoch_seconds"]) - time.time())
if not 1 <= remaining <= maximum:
    raise SystemExit("no valid EXP wall budget remains")
print(remaining)
PY
)"

exec timeout --signal=INT --kill-after=2m "${B11F_REMAINING_SECONDS}s" \
  uv run --extra neural python scripts/run_slp8_region_final_fit.py \
  "${B11F_COMMON_ARGS[@]}" --resume-authorized
