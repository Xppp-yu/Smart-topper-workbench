"""Governed one-fold timing preflight for the frozen P5.2-C Full runner.

This command trains all three neural candidates on exactly one frozen outer
fold.  It estimates Full wall time and records Git/CUDA/config provenance, but
it does not create or execute the formal Full EXP-ID.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from topper_perception.experiments.artifacts import (
    atomic_write_json,
    capture_git_info,
    capture_system_info,
    compute_config_hash,
)
from topper_perception.experiments.contracts import DirtyWorktreeError, validate_experiment_config
from topper_perception.neural.full import PROJECT_ROOT, run_one_fold_preflight
from topper_perception.neural.full_protocol import validate_full_config


DEFAULT_CONFIG = Path("configs/experiments/popu_neural_full_v0.1.json")
DEFAULT_OUTPUT = Path("outputs/experiments/EXP-P5.2-C-FULL-PREFLIGHT-20260820-R01")


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_clean_git() -> dict:
    git = capture_git_info(PROJECT_ROOT)
    if not (
        git.get("repo") is True
        and isinstance(git.get("sha"), str)
        and bool(git.get("sha"))
        and git.get("dirty") is False
    ):
        raise DirtyWorktreeError(
            "Full preflight requires a clean committed worktree; "
            f"got repo={git.get('repo')!r}, sha={git.get('sha')!r}, "
            f"dirty={git.get('dirty')!r}."
        )
    return git


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--local-fold", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _project_path(args.config)
    output_dir = _project_path(args.output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parsed = validate_experiment_config(config)
    validate_full_config(config)
    git = _require_clean_git()
    started = _utcnow()
    try:
        result = run_one_fold_preflight(
            parsed.parameters,
            parsed.seed,
            output_dir,
            repeat=args.repeat,
            local_fold=args.local_fold,
        )
        manifest = {
            "state": "SUCCEEDED",
            "scope": "one_fold_timing_preflight",
            "formal_full_not_run": True,
            "config_path": str(config_path),
            "config_hash": compute_config_hash(config),
            "git": git,
            "system": capture_system_info(),
            "started_at_utc": started,
            "ended_at_utc": _utcnow(),
            "result": result,
        }
        atomic_write_json(output_dir / "preflight_manifest.json", manifest)
    except BaseException as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            output_dir / "FAILED.json",
            {
                "state": "FAILED",
                "formal_full_not_run": True,
                "error": f"{type(exc).__name__}: {exc}",
                "started_at_utc": started,
                "ended_at_utc": _utcnow(),
            },
        )
        traceback.print_exc()
        return 1
    print(
        "PREFLIGHT_SUCCEEDED "
        f"dir={output_dir} observed_seconds={result['observed_seconds']} "
        f"estimated_full_seconds={result['estimated_full_seconds']} "
        f"within_frozen_budget={result['within_frozen_budget']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
