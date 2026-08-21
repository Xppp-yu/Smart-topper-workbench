"""PoPu P7 software-robustness CLI.

This script is a thin wrapper around :mod:`topper_perception.neural.p7_runner`.
It loads the frozen P7 analysis config (``configs/analysis/popu_p7_robustness_v0.1.json``),
locates the frozen P5.2-C Full evidence pack, derives the Full cohort loading
parameters from ``configs/experiments/popu_neural_full_v0.1.json``, and either
runs the CPU Smoke (1 repeat × 1 fold × a few records × clean + one light
perturbation per class) or the full P7 sweep across all 15 small_resnet
fold checkpoints.

The script never:

- retrains a model;
- mutates the P5.2-C Full evidence pack;
- fabricates perturbation results from existing OOF probabilities;
- calls a perturbation a hardware PASS.

After every successful run it writes:

- ``experiment_dir/scope.json`` (repeats/folds/conditions/device);
- ``experiment_dir/config_used.json`` (frozen P7 config snapshot);
- ``experiment_dir/folds/repeat_R/fold_F/{clean,condition_name/seed_S}/...``
  with snapshot and record CSVs, summary JSON, and the OOF cross-check;
- ``experiment_dir/condition_comparison.json`` with per-condition seed mean/std
  and worst case.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from topper_perception.neural.p7_runner import run_popu_p7_robustness  # noqa: E402

DEFAULT_P7_CONFIG = REPO_ROOT / "configs" / "analysis" / "popu_p7_robustness_v0.1.json"
DEFAULT_FULL_CONFIG = REPO_ROOT / "configs" / "experiments" / "popu_neural_full_v0.1.json"
DEFAULT_PATHS_CONFIG = REPO_ROOT / "configs" / "paths.local.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p7-config",
        type=Path,
        default=DEFAULT_P7_CONFIG,
        help="Path to the frozen P7 analysis config JSON.",
    )
    parser.add_argument(
        "--full-config",
        type=Path,
        default=DEFAULT_FULL_CONFIG,
        help="Path to the frozen P5.2-C Full experiment config JSON "
        "(used to derive full_cohort_parameters).",
    )
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=DEFAULT_PATHS_CONFIG,
        help="Optional paths.local.json to override the PoPu data root.",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help="Override the frozen evidence-pack root (defaults to the path "
        "encoded in p7_runner.DEFAULT_EVIDENCE_ROOT).",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Where to write the per-fold / per-condition outputs.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Restrict to 1 repeat × 1 fold × 5 records × clean + one light "
        "perturbation per class for CPU Smoke validation.",
    )
    parser.add_argument(
        "--smoke-max-records",
        type=int,
        default=5,
        help="How many records to retain in --smoke mode (default: 5).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        nargs="+",
        default=None,
        help="Restrict to a subset of repeats (e.g. --repeats 0).",
    )
    parser.add_argument(
        "--local-folds",
        type=int,
        nargs="+",
        default=None,
        help="Restrict to a subset of local folds (e.g. --local-folds 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260820,
        help="Reserved outer seed recorded in the manifest "
        "(the perturbation itself is deterministic from the frozen "
        "per-condition + per-fold seeds).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Resolve device via the training module.",
    )
    return parser.parse_args()


def _load_full_cohort_parameters(full_config_path: Path) -> dict:
    """Extract the Full cohort loading parameters from the frozen P5.2-C config.

    The Full runner already pins the PoPu quality manifest and SHA-256; we
    mirror only the fields ``load_full_cohort`` needs. Drift here is a
    hard fail at cohort load time.
    """
    config = json.loads(full_config_path.read_text(encoding="utf-8"))
    parameters = config.get("parameters", {})
    return {
        "data_root": parameters.get("data_root"),
        "paths_config": parameters.get("paths_config"),
        "quality_manifest": parameters["quality_manifest"],
        "quality_manifest_sha256": parameters["quality_manifest_sha256"],
        "cohort": parameters.get("cohort", "primary"),
        "dataset": parameters.get("dataset", "popu_tactilus"),
    }


def _merge_paths_overlay(full_params: dict, paths_config: Path | None) -> None:
    """If a paths.local.json is provided, overlay ``data_root`` onto the params."""
    if paths_config is None or not paths_config.is_file():
        return
    overlay = json.loads(paths_config.read_text(encoding="utf-8"))
    popu_root = overlay.get("popu_tactilus_root") or overlay.get("data_root")
    if popu_root and not full_params.get("data_root"):
        full_params["data_root"] = popu_root


def build_p7_parameters(
    p7_config: Path,
    full_config: Path,
    paths_config: Path | None,
    evidence_root: Path | None,
    smoke: bool,
    smoke_max_records: int,
    repeats: list[int] | None,
    local_folds: list[int] | None,
    device: str,
) -> dict:
    config = json.loads(p7_config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "p7-robustness-v0.1":
        raise ValueError(
            f"Unexpected P7 schema_version: {config.get('schema_version')!r}."
        )
    # Validate the FROZEN fields of the config BEFORE any CLI narrowing.
    # CLI overrides only narrow repeats/local_folds for Smoke / clean-only
    # modes; the underlying frozen contract must still be intact.
    _validate_frozen_p7_contract(config)
    full_params = _load_full_cohort_parameters(full_config)
    _merge_paths_overlay(full_params, paths_config)
    config["full_cohort_parameters"] = full_params
    if evidence_root is not None:
        config["evidence_root"] = str(evidence_root.resolve())
    if smoke:
        config["smoke_max_records"] = int(smoke_max_records)
    if repeats:
        # CLI narrowing: store under a dedicated key so the frozen values
        # remain intact for downstream contract validation.
        config["__narrowed_repeats"] = [int(r) for r in repeats]
    if local_folds:
        config["__narrowed_local_folds"] = [int(f) for f in local_folds]
    config["device"] = device
    return config


def _validate_frozen_p7_contract(config: dict) -> None:
    """Validate the frozen P7 contract fields of the loaded JSON config.

    This is the Script-side guard that runs against the *un-overridden*
    frozen config — before any CLI narrowing of repeats/local_folds is
    applied. The runner-side :func:`p7_runner._validate_p7_config` enforces
    the same contract at execution time, but only after CLI overrides have
    narrowed the fold set for Smoke / clean-only modes.
    """
    from topper_perception.neural.p7_runner import (
        FULL_LOCAL_FOLDS,
        FULL_REPEATS,
        MODEL_FAMILY,
        SCHEMA_VERSION,
    )

    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"P7 config schema_version must be {SCHEMA_VERSION!r}, "
            f"got {config.get('schema_version')!r}."
        )
    if config.get("model_family") != MODEL_FAMILY:
        raise ValueError(
            f"P7 config model_family must be {MODEL_FAMILY!r}, "
            f"got {config.get('model_family')!r}."
        )
    repeats = config.get("repeats")
    if not isinstance(repeats, list) or set(repeats) != set(FULL_REPEATS):
        raise ValueError(
            f"P7 frozen config repeats must be exactly {list(FULL_REPEATS)}; "
            f"got {sorted(repeats) if isinstance(repeats, list) else repeats!r}."
        )
    local_folds = config.get("local_folds")
    if not isinstance(local_folds, list) or set(local_folds) != set(FULL_LOCAL_FOLDS):
        raise ValueError(
            f"P7 frozen config local_folds must be exactly {list(FULL_LOCAL_FOLDS)}; "
            f"got {sorted(local_folds) if isinstance(local_folds, list) else local_folds!r}."
        )


def main() -> int:
    args = _parse_args()
    parameters = build_p7_parameters(
        p7_config=args.p7_config,
        full_config=args.full_config,
        paths_config=args.paths_config,
        evidence_root=args.evidence_root,
        smoke=args.smoke,
        smoke_max_records=args.smoke_max_records,
        repeats=args.repeats,
        local_folds=args.local_folds,
        device=args.device,
    )
    experiment_dir: Path = args.experiment_dir.expanduser().resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    summary = run_popu_p7_robustness(
        parameters=parameters,
        seed=int(args.seed),
        experiment_dir=experiment_dir,
    )
    elapsed = time.perf_counter() - started

    scope_payload = json.loads((experiment_dir / "scope.json").read_text(encoding="utf-8"))
    output = {
        "ok": True,
        "experiment_dir": str(experiment_dir),
        "elapsed_seconds": elapsed,
        "scope": scope_payload,
        "runner_summary": summary,
    }
    print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())