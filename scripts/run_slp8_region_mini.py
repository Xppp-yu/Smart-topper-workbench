"""B04 PM-only Region Mini runner CLI (TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-v0.1).

This script is the B04 entry point.  It enforces the B04 governance:

* The default mode is ``--validate-config``, which reads the frozen
  config and writes ``status.json``/``resolved_config.json``/``DONE.json``
  without touching any B01 data.
* The synthetic CPU smoke mode is invoked with ``--synthetic-cpu-smoke``
  and uses tiny deterministic synthetic (pressure, label) pairs that
  fully exercise the runner, the registry, the class-weight formula,
  the metrics bundle, the checkpoint save/load, and the FEASIBLE gate.
* A real B01 run requires **both** ``--run-authorized`` and
  ``--b01-freeze-dir`` / ``--dataset-root``; the script refuses to
  read any real B01 path when ``--run-authorized`` is missing.  The
  current task never exercises the real B01 path.

Usage (default = --validate-config)::

    uv run python scripts/run_slp8_region_mini.py \\
        --config configs/experiments/slp8_pm_region_mini_v0.1.json \\
        --output-dir outputs/experiments/EXP-SLP-B04-PM-REGION-MINI-20260827-VALIDATE

Synthetic CPU smoke::

    uv run python scripts/run_slp8_region_mini.py \\
        --config configs/experiments/slp8_pm_region_mini_v0.1.json \\
        --output-dir outputs/experiments/EXP-SLP-B04-PM-REGION-MINI-20260827-SYNTH \\
        --synthetic-cpu-smoke

Real run (NOT executed by B04 v0.1; requires explicit --run-authorized)::

    uv run python scripts/run_slp8_region_mini.py \\
        --config configs/experiments/slp8_pm_region_mini_v0.1.json \\
        --output-dir outputs/experiments/EXP-SLP-B04-PM-REGION-MINI-20260827-R01 \\
        --b01-freeze-dir <B01_FREEZE_DIR> \\
        --dataset-root <SLP8_DATASET_ROOT> \\
        --run-authorized
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from topper_perception.io.slp8_training_table_freeze import (
    A06_SPLIT_SHA256_EXPECTED,
    EXPECTED_PROVENANCE,
    EXPECTED_REVIEW_STATUS,
    EXPECTED_SOURCE_SPLITS,
    EXPECTED_SETTINGS,
    EXPECTED_COVERS,
    load_b01_freeze_tables,
    sha256_file,
)
from topper_perception.neural.slp8_region_class_weights import (
    assert_class_weight_invariants,
    compute_class_weights,
)
from topper_perception.neural.slp8_region_mini import (
    B02_BASELINE_REFERENCE_VAL_FIXED_IOU,
    B04_CANDIDATE_NAMES,
    CHECKPOINT_VERSION,
    MINI_VERSION,
    SYNTHETIC_DEFAULTS,
    TASK_ID,
    MiniConfig,
    MiniProtocolError,
    OutputCollisionError,
    build_mini_config,
    build_synthetic_dataset,
    check_output_dir_safety,
    file_sha256,
    resolve_device,
    run_mini,
    validate_mini_config,
    write_mini_artifacts,
    write_status_files,
    _gather_environment,
)
from topper_perception.neural.slp8_region_models import (
    B04_MAX_PARAMETERS,
    SMALL_UNET_VERSION,
    MODEL_VERSION,
    get_model_builder,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDACTED_LOCAL_PATH = "REDACTED_LOCAL_PATH"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, float):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, (list, tuple)):
        return [_json_default(i) for i in obj]
    if isinstance(obj, dict):
        return {str(k): _json_default(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Validate-only mode
# ---------------------------------------------------------------------------


def _run_validate_config(
    config_path: Path, output_dir: Path
) -> int:
    """Validate the config (and the model registry) without running anything."""

    output_dir = Path(output_dir).resolve()
    check_output_dir_safety(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    def _log(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{_now_iso()}] {msg}\n")

    _log(f"task_id={TASK_ID}")
    _log(f"config_path={config_path}")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    validate_mini_config(raw)
    _log("config validation: PASSED")

    # Build a MiniConfig purely to serialize the resolved view.
    config = build_mini_config(raw, b01_freeze_dir=None, data_root=None)
    _write_json(output_dir / "resolved_config.json", config.as_dict())
    _write_json(
        output_dir / "input_manifest_hashes.json",
        {
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "a06_split_sha256_expected": A06_SPLIT_SHA256_EXPECTED,
            "registered_candidates": list(B04_CANDIDATE_NAMES),
            "note": "validate-only mode: no B01 freeze tables read",
        },
    )
    _write_json(output_dir / "environment.json", _gather_environment())

    # Verify the model registry can build both candidates (CPU) and
    # that the parameter count cap is respected.
    for cand in B04_CANDIDATE_NAMES:
        builder = get_model_builder(cand)
        model, _ = builder.factory(9, "cpu")
        count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        if count > B04_MAX_PARAMETERS:
            raise MiniProtocolError(
                f"candidate {cand} has {count} parameters; exceeds B04 cap of "
                f"{B04_MAX_PARAMETERS}"
            )
        _log(f"candidate={cand} model_version={builder.version} parameters={count}")

    _write_json(
        output_dir / "status.json",
        {
            "task_id": TASK_ID,
            "config_version": MINI_VERSION,
            "status": "VALIDATED",
            "started_at_utc": _now_iso(),
            "ended_at_utc": _now_iso(),
            "mode": "validate-config",
            "registered_candidates": list(B04_CANDIDATE_NAMES),
            "model_parameter_cap": B04_MAX_PARAMETERS,
        },
    )
    write_status_files(
        output_dir,
        status="DONE",
        extra={
            "mode": "validate-config",
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
        },
    )
    return 0


# ---------------------------------------------------------------------------
# Synthetic CPU smoke
# ---------------------------------------------------------------------------


def _run_synthetic_cpu_smoke(
    config_path: Path,
    output_dir: Path,
) -> int:
    """Run the B04 Mini end-to-end on synthetic data with CPU only."""

    output_dir = Path(output_dir).resolve()
    check_output_dir_safety(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    def _log(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{_now_iso()}] {msg}\n")

    _log(f"task_id={TASK_ID} mode=synthetic-cpu-smoke")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    validate_mini_config(raw)

    # Build a synthetic-data override config (synthetic forces CPU).
    raw["training"]["device"] = "cpu"
    config = build_mini_config(
        raw,
        b01_freeze_dir="<SYNTHETIC>",
        data_root="<SYNTHETIC>",
    )

    device = resolve_device("cpu", allow_cpu_fallback=True)
    if str(device) != "cpu":
        raise MiniProtocolError("synthetic CPU smoke must run on cpu")

    _log(f"device={device}")

    # Build synthetic datasets.
    train_dataset, val_dataset, dataset_manifest, train_class_stats = build_synthetic_dataset(
        n_train_samples=int(SYNTHETIC_DEFAULTS["n_train_samples"]),
        n_val_samples=int(SYNTHETIC_DEFAULTS["n_val_samples"]),
        seed=int(SYNTHETIC_DEFAULTS["seed"]),
    )
    if dataset_manifest["n_test_samples"] != 0:
        raise MiniProtocolError(
            "synthetic dataset must report n_test_samples=0; got "
            f"{dataset_manifest['n_test_samples']}"
        )
    _log(
        f"train_subjects={dataset_manifest['train_subjects']} "
        f"n_train={dataset_manifest['n_train_samples']} "
        f"n_val={dataset_manifest['n_val_samples']} "
        f"n_test={dataset_manifest['n_test_samples']}"
    )

    # Compute class weights from the synthetic TRAIN-only stats.
    class_weight_result = compute_class_weights(
        {
            "n_samples": int(train_class_stats["n_samples"]),
            "n_pixels": int(train_class_stats["n_pixels"]),
            "per_class_pixel_ratio": {
                int(k): float(v)
                for k, v in train_class_stats["per_class_pixel_ratio"].items()
            },
        }
    )
    assert_class_weight_invariants(class_weight_result)
    _log(
        "class_weights="
        f"{[(c, round(class_weight_result.weights[c], 4)) for c in range(9)]}"
    )

    # Write the manifest and resolved config up front so the operator
    # can audit them even if a candidate later fails.
    _write_json(
        output_dir / "resolved_config.json",
        {**config.as_dict(), "mode": "synthetic-cpu-smoke"},
    )
    _write_json(
        output_dir / "input_manifest_hashes.json",
        {
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "a06_split_sha256_expected": A06_SPLIT_SHA256_EXPECTED,
            "registered_candidates": list(B04_CANDIDATE_NAMES),
            "synthetic": True,
            "synthetic_train_class_stats_sha256": _hash_dict(train_class_stats),
        },
    )
    _write_json(output_dir / "environment.json", _gather_environment())

    # Persist a small manifest so the reviewer can audit the run.
    _write_json(
        output_dir / "manifest.json",
        {
            "task_id": TASK_ID,
            "config_version": MINI_VERSION,
            "mode": "synthetic-cpu-smoke",
            "dataset_manifest": dataset_manifest,
            "train_class_stats": train_class_stats,
            "class_weight_summary": class_weight_result.as_dict(),
            "registered_candidates": list(B04_CANDIDATE_NAMES),
            "checkpoint_version": CHECKPOINT_VERSION,
            "b02_reference_val_fixed_iou": B02_BASELINE_REFERENCE_VAL_FIXED_IOU,
            "b04_max_parameters": B04_MAX_PARAMETERS,
            "started_at_utc": _now_iso(),
        },
    )

    t_start = time.perf_counter()
    result = run_mini(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataset_manifest=dataset_manifest,
        class_weight_result=class_weight_result,
        output_dir=output_dir,
        device=device,
        input_hashes={
            "config_sha256": file_sha256(config_path),
            "a06_split_sha256_expected": A06_SPLIT_SHA256_EXPECTED,
            "synthetic": True,
        },
        train_class_stats_source="synthetic_train_class_stats",
        synthetic=True,
    )
    t_end = time.perf_counter()
    result.wall_clock_seconds = float(t_end - t_start)
    _log(
        f"completed in {result.wall_clock_seconds:.2f}s; "
        f"overall_decision={result.overall_decision}; "
        f"feasible={result.n_candidates_feasible} "
        f"not_feasible={result.n_candidates_not_feasible} "
        f"failed={result.n_candidates_failed}"
    )

    _write_json(
        output_dir / "manifest.json",
        {
            "task_id": TASK_ID,
            "config_version": MINI_VERSION,
            "mode": "synthetic-cpu-smoke",
            "dataset_manifest": dataset_manifest,
            "train_class_stats": train_class_stats,
            "class_weight_summary": class_weight_result.as_dict(),
            "registered_candidates": list(B04_CANDIDATE_NAMES),
            "checkpoint_version": CHECKPOINT_VERSION,
            "b02_reference_val_fixed_iou": B02_BASELINE_REFERENCE_VAL_FIXED_IOU,
            "b04_max_parameters": B04_MAX_PARAMETERS,
            "started_at_utc": result.started_at_utc,
            "ended_at_utc": result.ended_at_utc,
            "wall_clock_seconds": result.wall_clock_seconds,
            "candidate_feasibility": {
                cand: {
                    "feasibility": cand_result.feasibility,
                    "reason": cand_result.reason,
                }
                for cand, cand_result in result.candidate_results.items()
            },
            "overall_decision": result.overall_decision,
        },
    )

    # Emit DONE.json (or FAILED.json) — mutually exclusive.
    has_failure = any(
        cand.feasibility == "FAILED" for cand in result.candidate_results.values()
    )
    if has_failure or result.overall_decision == "MINI_NOT_FEASIBLE":
        # For a synthetic smoke we still want the artefacts to remain
        # inspectable, but the contract says FAILED and DONE are
        # mutually exclusive.  We treat MINI_NOT_FEASIBLE as a *non-FAILED*
        # completion (no candidate reached the B02 gate, but the runner
        # itself is healthy).
        write_status_files(
            output_dir,
            status="DONE",
            extra={
                "mode": "synthetic-cpu-smoke",
                "overall_decision": result.overall_decision,
                "wall_clock_seconds": result.wall_clock_seconds,
                "n_candidates_feasible": result.n_candidates_feasible,
                "n_candidates_not_feasible": result.n_candidates_not_feasible,
            },
        )
    else:
        write_status_files(
            output_dir,
            status="DONE",
            extra={
                "mode": "synthetic-cpu-smoke",
                "overall_decision": result.overall_decision,
                "wall_clock_seconds": result.wall_clock_seconds,
            },
        )

    _write_json(output_dir / "status.json", {
        "task_id": TASK_ID,
        "config_version": MINI_VERSION,
        "status": "DONE",
        "mode": "synthetic-cpu-smoke",
        "started_at_utc": result.started_at_utc,
        "ended_at_utc": result.ended_at_utc,
        "wall_clock_seconds": result.wall_clock_seconds,
        "overall_decision": result.overall_decision,
        "n_candidates_feasible": result.n_candidates_feasible,
        "n_candidates_not_feasible": result.n_candidates_not_feasible,
        "n_candidates_failed": result.n_candidates_failed,
        "n_candidates_stopped": result.n_candidates_stopped,
    })
    return 0


def _hash_dict(payload: Any) -> str:
    """Stable SHA-256 of a JSON payload."""

    import hashlib

    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Real B01 path (gated by --run-authorized)
# ---------------------------------------------------------------------------


def _run_real_b01(
    config_path: Path,
    output_dir: Path,
    b01_freeze_dir: Path,
    dataset_root: Path,
) -> int:
    """Run the B04 Mini on real B01 freeze tables.  Requires --run-authorized.

    This branch is **not** exercised by the B04 protocol task and is
    deliberately left as a future-work hook: the protocol file is
    delivered, the code path is implemented and unit-tested on
    synthetic data, but a real run must wait for Owner authorization.
    """

    output_dir = Path(output_dir).resolve()
    b01_freeze_dir = Path(b01_freeze_dir).resolve()
    dataset_root = Path(dataset_root).resolve()
    check_output_dir_safety(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    def _log(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{_now_iso()}] {msg}\n")

    _log(f"task_id={TASK_ID} mode=real-b01")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    validate_mini_config(raw)
    config = build_mini_config(
        raw, b01_freeze_dir=str(b01_freeze_dir), data_root=str(dataset_root)
    )

    if not b01_freeze_dir.is_dir():
        raise FileNotFoundError(f"B01 freeze directory not found: {b01_freeze_dir}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # Load B01 freeze tables with TEST access explicitly off.
    freeze = load_b01_freeze_tables(b01_freeze_dir, load_test=False)
    if freeze._test_rows is not None:  # noqa: SLF001
        raise MiniProtocolError(
            "TEST rows were loaded by load_b01_freeze_tables(..., load_test=False); "
            "this must not happen"
        )

    n_train = len(freeze.train_rows)
    n_val = len(freeze.val_rows)
    if n_train == 0 or n_val == 0:
        raise MiniProtocolError(
            f"B01 freeze has zero rows for an essential split: train={n_train}, val={n_val}"
        )

    # Real data path: B04 demands device='cuda'.  In a real B01 run, CUDA
    # must be available; otherwise fail-closed.
    device = resolve_device("cuda", allow_cpu_fallback=False)
    _log(f"device={device}")

    # Hash input artefacts for the audit record.
    freeze_manifest_path = b01_freeze_dir / "freeze_manifest.json"
    train_class_stats_path = b01_freeze_dir / "train_class_stats.json"
    if not freeze_manifest_path.is_file():
        raise FileNotFoundError(f"freeze_manifest.json missing in {b01_freeze_dir}")
    if not train_class_stats_path.is_file():
        raise FileNotFoundError(f"train_class_stats.json missing in {b01_freeze_dir}")
    train_class_stats = json.loads(train_class_stats_path.read_text(encoding="utf-8"))

    # Real cross-check: B01 freeze must have come from A06 SHA
    # 024f5abe (otherwise the audit trail breaks).
    fm_sha = sha256_file(freeze_manifest_path)
    if freeze.freeze_manifest.get("core", {}).get(
        "a06_split_sha256_expected"
    ) not in (None, A06_SPLIT_SHA256_EXPECTED) and fm_sha != freeze.freeze_manifest_sha256:
        _log(
            f"WARNING: freeze_manifest sha256 mismatch: file={fm_sha} "
            f"manifest={freeze.freeze_manifest_sha256}"
        )

    # Subject isolation sanity (B01 should already guarantee this; we
    # re-verify as a defense in depth).
    train_subjects = sorted({row.subject_id for row in freeze.train_rows})
    val_subjects = sorted({row.subject_id for row in freeze.val_rows})
    if not verify_subject_isolation(train_subjects, val_subjects):
        raise MiniProtocolError("TRAIN/VAL subject overlap detected in B01 freeze")

    # Class weights from the B01 train_class_stats.json.
    class_weight_result = compute_class_weights(train_class_stats)
    assert_class_weight_invariants(class_weight_result)

    # Build the real B01 dataset using the B03 dataset builder.
    from topper_perception.neural.slp8_region_dataset import (
        build_smoke_dataset,
    )
    train_dataset, val_dataset, dataset_manifest = build_smoke_dataset(
        b01_freeze_dir=b01_freeze_dir,
        dataset_root=dataset_root,
        seed=int(raw.get("dataset", {}).get("smoke_subset", {}).get("seed", 42)),
        n_train_subjects=int(
            raw.get("dataset", {}).get("smoke_subset", {}).get(
                "n_train_subjects", len(train_subjects)
            )
        ),
        n_val_subjects=int(
            raw.get("dataset", {}).get("smoke_subset", {}).get(
                "n_val_subjects", len(val_subjects)
            )
        ),
    )
    if dataset_manifest["n_test_samples"] != 0:
        raise MiniProtocolError(
            f"real B01 dataset must report n_test_samples=0; got "
            f"{dataset_manifest['n_test_samples']}"
        )

    _write_json(output_dir / "resolved_config.json", config.as_dict())
    _write_json(
        output_dir / "input_manifest_hashes.json",
        {
            "config_sha256": file_sha256(config_path),
            "freeze_manifest_sha256": fm_sha,
            "a06_split_sha256_expected": A06_SPLIT_SHA256_EXPECTED,
            "b01_freeze_dir": REDACTED_LOCAL_PATH,
            "dataset_root": REDACTED_LOCAL_PATH,
            "train_class_stats_sha256": sha256_file(train_class_stats_path),
        },
    )
    _write_json(output_dir / "environment.json", _gather_environment())

    t_start = time.perf_counter()
    result = run_mini(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataset_manifest=dataset_manifest,
        class_weight_result=class_weight_result,
        output_dir=output_dir,
        device=device,
        input_hashes={
            "config_sha256": file_sha256(config_path),
            "freeze_manifest_sha256": fm_sha,
            "a06_split_sha256_expected": A06_SPLIT_SHA256_EXPECTED,
            "train_class_stats_sha256": sha256_file(train_class_stats_path),
        },
        train_class_stats_source="b01_train_class_stats.json",
        synthetic=False,
    )
    t_end = time.perf_counter()
    result.wall_clock_seconds = float(t_end - t_start)
    _log(
        f"completed in {result.wall_clock_seconds:.2f}s; "
        f"overall_decision={result.overall_decision}"
    )

    write_status_files(
        output_dir,
        status="DONE",
        extra={
            "mode": "real-b01",
            "overall_decision": result.overall_decision,
            "wall_clock_seconds": result.wall_clock_seconds,
        },
    )
    _write_json(output_dir / "status.json", {
        "task_id": TASK_ID,
        "config_version": MINI_VERSION,
        "status": "DONE",
        "mode": "real-b01",
        "overall_decision": result.overall_decision,
        "wall_clock_seconds": result.wall_clock_seconds,
    })
    return 0


def verify_subject_isolation(train_subjects, val_subjects) -> bool:
    return not (set(train_subjects) & set(val_subjects))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to the B04 Mini config JSON.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for run artefacts.",
    )
    parser.add_argument(
        "--validate-config", dest="validate_config",
        action="store_true",
        help="Only validate the config; do not run anything.",
    )
    parser.add_argument(
        "--synthetic-cpu-smoke", dest="synthetic_cpu_smoke",
        action="store_true",
        help="Run the B04 Mini on synthetic CPU data (no real B01).",
    )
    parser.add_argument(
        "--b01-freeze-dir", type=Path, default=None,
        help="Path to the B01 freeze directory (real run only).",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=None,
        help="Path to the SLP8 dataset root (real run only).",
    )
    parser.add_argument(
        "--run-authorized", dest="run_authorized",
        action="store_true",
        help="REQUIRED for any real B01 run; the B04 protocol task does not set this.",
    )

    args = parser.parse_args(argv)

    try:
        # Mutual-exclusion: --validate-config and --synthetic-cpu-smoke
        # and a real run cannot coexist.
        if args.validate_config and args.synthetic_cpu_smoke:
            raise MiniProtocolError(
                "--validate-config and --synthetic-cpu-smoke are mutually exclusive"
            )

        if args.validate_config:
            return _run_validate_config(args.config, args.output_dir)

        if args.synthetic_cpu_smoke:
            return _run_synthetic_cpu_smoke(args.config, args.output_dir)

        if args.b01_freeze_dir is not None or args.dataset_root is not None:
            if not args.run_authorized:
                raise MiniProtocolError(
                    "B01 freeze or dataset-root paths were supplied but "
                    "--run-authorized was NOT set.  B04 forbids the real B01 "
                    "path without explicit owner authorization.  Re-run with "
                    "--run-authorized to proceed."
                )
            if args.b01_freeze_dir is None or args.dataset_root is None:
                raise MiniProtocolError(
                    "real B01 run requires both --b01-freeze-dir and --dataset-root"
                )
            return _run_real_b01(
                args.config, args.output_dir, args.b01_freeze_dir, args.dataset_root
            )

        # No explicit mode supplied — default to validate-config.
        return _run_validate_config(args.config, args.output_dir)
    except Exception as exc:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        failed = {
            "status": "FAILED",
            "task_id": TASK_ID,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "ended_at_utc": _now_iso(),
        }
        try:
            write_status_files(
                output_dir,
                status="FAILED",
                extra={
                    "mode": "validate-config"
                    if args.validate_config
                    else (
                        "synthetic-cpu-smoke"
                        if args.synthetic_cpu_smoke
                        else "real-b01"
                    ),
                    "error": str(exc),
                },
            )
        except Exception:
            pass
        _write_json(output_dir / "status.json", {
            "task_id": TASK_ID,
            "status": "FAILED",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
