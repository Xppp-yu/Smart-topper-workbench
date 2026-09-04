"""B09T no-TEST validator and synthetic smoke CLI; no real TEST path exists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.validate_slp8_b09t_protocol import validate
from topper_perception.evaluation.slp8_b09t_evaluator import synthetic_smoke_payload


class B09TRunnerError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/experiments/slp8_pm_b09t_final_test_protocol_v0.1.json")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    try:
        errors = validate(args.protocol.resolve())
        if errors:
            raise B09TRunnerError("protocol rejected: " + "; ".join(errors))
        if args.validate_only:
            if args.output_dir or args.experiment_id:
                raise B09TRunnerError("validate-only refuses output and experiment arguments")
            print("B09T_EVALUATOR_VALIDATION_PASSED TEST=0 GPU_NOT_RUN EXECUTION_NOT_AUTHORIZED")
            return 0
        if not args.output_dir or not args.experiment_id:
            raise B09TRunnerError("synthetic smoke requires --output-dir and --experiment-id")
        if not args.experiment_id.startswith("SMOKE-B09T-"):
            raise B09TRunnerError("synthetic experiment-id must start with SMOKE-B09T-")
        if args.output_dir.exists():
            raise B09TRunnerError("output path already exists")
        payload = synthetic_smoke_payload()
        payload["experiment_id"] = args.experiment_id
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "synthetic_summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        print("B09T_SYNTHETIC_SMOKE_PASSED TEST=0 GPU_NOT_RUN EXECUTION_NOT_AUTHORIZED")
        return 0
    except (B09TRunnerError, ValueError, OSError) as exc:
        print(f"B09T_REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
