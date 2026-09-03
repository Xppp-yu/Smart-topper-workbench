#!/usr/bin/env bash
set -euo pipefail

# This script is preparation-only until the Owner authorizes this exact file,
# runner SHA and bundle SHA. It probes the environment but performs no training.
readonly B11F_BUNDLE="/root/autodl-tmp/smarttopper-b11f-main-a6a5d8e.bundle"
readonly B11F_REPO="/root/autodl-tmp/smarttopper-b11f-preflight-a6a5d8e"
readonly B11F_FREEZE_DIR="/root/autodl-tmp/data/processed/slp8_training_tables_v0.1"
readonly B11F_DATA_ROOT="/root/autodl-tmp/datasets/SLP_8Region_Pressure_VAL_v1.1"
readonly B11F_GIT_SHA="a6a5d8e6f8db003149169ee48f71d6e41e445a80"
readonly B11F_BUNDLE_SHA="5e9d855397face954cac18e3dbadb26449129f828f77d45412b3c4f30d8e6bb2"
readonly B11F_CONFIG_SHA="a6590d6f068644d98fa5340ec3d4a2e02171b529ec22ab092efb54a298925a43"
readonly B11F_CANDIDATE_SHA="839c9482c69cf34d3c91c3acb3c7a36cb4d199117d0d6eb2ceb7906bac52b994"
readonly B11F_FREEZE_SHA="42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04"

test -f "$B11F_BUNDLE"
test "$(sha256sum "$B11F_BUNDLE" | cut -d' ' -f1)" = "$B11F_BUNDLE_SHA"
test ! -e "$B11F_REPO"
git clone "$B11F_BUNDLE" "$B11F_REPO"
cd "$B11F_REPO"
git checkout --detach "$B11F_GIT_SHA"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$B11F_GIT_SHA"
test "$(git remote get-url origin)" = "$B11F_BUNDLE"
test "$(git rev-parse origin/main)" = "$B11F_GIT_SHA"
git merge-base --is-ancestor "$B11F_GIT_SHA" origin/main

readonly B11F_CONFIG="$B11F_REPO/configs/experiments/slp8_pm_final_development_fit_v0.1.json"
readonly B11F_CANDIDATE="$B11F_REPO/configs/experiments/slp8_pm_research_candidate_v0.1.json"
test "$(sha256sum "$B11F_CONFIG" | cut -d' ' -f1)" = "$B11F_CONFIG_SHA"
test "$(sha256sum "$B11F_CANDIDATE" | cut -d' ' -f1)" = "$B11F_CANDIDATE_SHA"
test "$(sha256sum "$B11F_FREEZE_DIR/freeze_manifest.json" | cut -d' ' -f1)" = "$B11F_FREEZE_SHA"
test -d "$B11F_DATA_ROOT"

# A fresh bundle checkout may contain only the tracked .gitkeep. No formal
# B11F experiment directory may exist before or after this environment probe.
if test -d "$B11F_REPO/outputs/experiments"; then
  test -z "$(find "$B11F_REPO/outputs/experiments" -mindepth 1 -maxdepth 1 ! -name .gitkeep -print -quit)"
fi

test "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)" = "NVIDIA GeForce RTX 4090"
uv run python - <<'PY'
import json
import torch

assert torch.cuda.is_available(), "CUDA unavailable"
payload = {
    "cuda_available": True,
    "cuda_device_count": int(torch.cuda.device_count()),
    "cuda_runtime": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_training_run": False,
    "torch_version": torch.__version__,
}
assert payload["gpu_name"] == "NVIDIA GeForce RTX 4090"
print(json.dumps(payload, allow_nan=False, sort_keys=True))
PY

uv run python scripts/validate_slp8_b11f_final_fit_preparation.py "$B11F_CONFIG"
uv run python scripts/run_slp8_region_final_fit.py --config "$B11F_CONFIG" --validate-only
uv run python scripts/run_slp8_region_final_fit.py --config "$B11F_CONFIG" --environment-preflight

if test -d "$B11F_REPO/outputs/experiments"; then
  test -z "$(find "$B11F_REPO/outputs/experiments" -mindepth 1 -maxdepth 1 ! -name .gitkeep -print -quit)"
fi
test -z "$(git status --porcelain)"
printf '%s\n' "B11F_AUTODL_NO_TRAINING_PREFLIGHT_PASSED TEST=0 TRAINING_NOT_STARTED"
