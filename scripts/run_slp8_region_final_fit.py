"""CLI for governed B11F final development fit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topper_perception.neural.slp8_region_final_fit import (
    FinalFitError,
    environment_preflight_payload,
    load_protocol,
    run_final_fit,
)


def git_identity() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return sha, dirty


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/experiments/slp8_pm_final_development_fit_v0.1.json")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--environment-preflight", action="store_true")
    p.add_argument("--run-authorized", action="store_true")
    p.add_argument("--resume-authorized", action="store_true")
    p.add_argument("--authorized-environment-sha256")
    p.add_argument("--experiment-id")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--b01-freeze-dir", type=Path)
    p.add_argument("--dataset-root", type=Path)
    args = p.parse_args()
    try:
        protocol = load_protocol(args.config.resolve(), ROOT)
        if args.validate_only:
            print("B11F_FINAL_FIT_PREPARATION_VALIDATION_PASSED TEST=0 GPU_NOT_AUTHORIZED")
            return 0
        if args.environment_preflight:
            print(json.dumps(environment_preflight_payload(), allow_nan=False, sort_keys=True))
            print("B11F_ENVIRONMENT_PREFLIGHT_PASSED TEST=0 GPU_TRAINING_NOT_RUN")
            return 0
        if not args.run_authorized: raise FinalFitError("real final fit requires --run-authorized")
        if not all((args.experiment_id, args.output_dir, args.b01_freeze_dir, args.dataset_root, args.authorized_environment_sha256)): raise FinalFitError("real final fit requires experiment/output/freeze/dataset/authorized-environment arguments")
        sha, dirty = git_identity()
        run_final_fit(protocol=protocol, freeze_dir=args.b01_freeze_dir, data_root=args.dataset_root, output_dir=args.output_dir, experiment_id=args.experiment_id, git_commit=sha, git_dirty=dirty, authorized_environment_sha256=args.authorized_environment_sha256, resume=args.resume_authorized)
        return 0
    except FinalFitError as exc:
        print(f"B11F_REJECTED: {exc}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
