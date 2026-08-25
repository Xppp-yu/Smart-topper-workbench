"""Run SLP8 B03 PM-only Region Segmentation Smoke Test.

CLI for the SLP8 pressure-only region segmentation smoke test
(TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1).  The script is CPU-only,
deterministic, and never reads TEST data.

Usage
-----

::

    uv run python scripts/run_slp8_region_smoke.py \\
        --config configs/experiments/slp8_pm_region_smoke_v0.1.json \\
        --output-dir outputs/experiments/<EXP-ID> \\
        --b01-freeze-dir <B01_FREEZE_DIR> \\
        --dataset-root <SLP8_DATASET_ROOT> \\
        --device cpu

Inputs
------
* ``--config`` — path to a JSON config.
* ``--output-dir`` — directory for run artifacts.
* ``--b01-freeze-dir`` — the B01 freeze directory.
* ``--dataset-root`` — the SLP8 dataset root.
* ``--device`` — device to run on (cpu or cuda).

Outputs
-------
* ``status.json`` — overall status (DONE / FAILED)
* ``manifest.json`` — run manifest with dataset and model info
* ``metrics_summary.json`` — training and validation metrics
* ``reload_consistency.json`` — checkpoint verification results
* ``checkpoints/initial_epoch.pt`` — checkpoint after initial training
* ``checkpoints/resumed_epoch.pt`` — checkpoint after resume
* ``DONE.json`` or ``FAILED.json``

The script refuses to overwrite an existing output directory that
contains ``DONE.json``, ``FAILED.json``, or any other files.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure ``src`` is on the import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topper_perception.io.slp8_training_table_freeze import (
    EXPECTED_PROVENANCE,
    FREEZE_VERSION,
    TASK_ID as B01_TASK_ID,
    A06_SPLIT_SHA256_EXPECTED,
    load_b01_freeze_tables,
    sha256_hex,
)
from topper_perception.neural.slp8_region_dataset import (
    build_smoke_dataset,
    verify_subject_isolation,
)
from topper_perception.neural.slp8_region_smoke import (
    TASK_ID,
    SMOKE_VERSION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_INITIAL_EPOCHS,
    DEFAULT_LR,
    DEFAULT_RESUME_EPOCHS,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    SmokeConfig,
    run_smoke_test,
    write_smoke_artifacts,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDACTED_LOCAL_PATH = "REDACTED_LOCAL_PATH"

# Required config fields
REQUIRED_CONFIG_FIELDS = (
    "config_version",
    "task_id",
    "smoke_version",
    "provenance",
    "raw_semantics",
    "model_version",
    "n_classes",
    "epochs",
    "lr",
    "weight_decay",
    "batch_size",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OutputDirCollisionError(RuntimeError):
    """Raised when output directory already contains artifacts."""


class ConfigValidationError(RuntimeError):
    """Raised when config validation fails."""


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def _validate_config(cfg: dict[str, Any]) -> None:
    """Validate the config dictionary."""
    missing = [f for f in REQUIRED_CONFIG_FIELDS if f not in cfg]
    if missing:
        raise ConfigValidationError(f"config missing required fields: {missing}")

    if cfg["task_id"] != TASK_ID:
        raise ConfigValidationError(
            f"config task_id {cfg['task_id']!r} != expected {TASK_ID!r}"
        )

    if cfg["provenance"] != EXPECTED_PROVENANCE:
        raise ConfigValidationError(
            f"config provenance must be {EXPECTED_PROVENANCE!r}"
        )

    if cfg["raw_semantics"] != "raw_pmarray_response":
        raise ConfigValidationError(
            "config raw_semantics must be 'raw_pmarray_response'"
        )


# ---------------------------------------------------------------------------
# Output directory safety
# ---------------------------------------------------------------------------


def _check_output_dir_safety(output_dir: Path) -> None:
    """Refuse to run if output directory already contains artifacts."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise OutputDirCollisionError(
            f"output path exists but is not a directory: {output_dir}"
        )

    # Check for sentinel files
    sentinel_files = ("DONE.json", "FAILED.json")
    for sentinel in sentinel_files:
        if (output_dir / sentinel).is_file():
            raise OutputDirCollisionError(
                f"output directory already contains {sentinel}; refusing to "
                f"overwrite.  Choose a fresh --output-dir.  ({output_dir})"
            )

    # Check for any other files
    contents = list(output_dir.iterdir())
    non_keep = [p for p in contents if p.name != ".gitkeep"]
    if non_keep:
        raise OutputDirCollisionError(
            f"output directory is not empty ({len(non_keep)} entries); "
            f"refusing to overwrite.  ({output_dir})"
        )


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    def json_default(obj: Any) -> Any:
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
            return [json_default(i) for i in obj]
        if isinstance(obj, dict):
            return {str(k): json_default(v) for k, v in obj.items()}
        return str(obj)

    path.write_text(
        json.dumps(payload, indent=2, default=json_default, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for artifacts.",
    )
    parser.add_argument(
        "--b01-freeze-dir",
        type=Path,
        required=True,
        help="B01 freeze directory.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="SLP8 dataset root directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    b01_freeze_dir = Path(args.b01_freeze_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()

    # Validate config
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    _validate_config(cfg)

    # Check output directory safety
    _check_output_dir_safety(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write initial status
    status = {
        "task_id": TASK_ID,
        "smoke_version": SMOKE_VERSION,
        "status": "RUNNING",
        "started_at_utc": _now_iso(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    _write_json(output_dir / "status.json", status)

    try:
        # Verify B01 freeze directory
        if not b01_freeze_dir.is_dir():
            raise FileNotFoundError(f"B01 freeze directory not found: {b01_freeze_dir}")

        freeze_manifest_path = b01_freeze_dir / "freeze_manifest.json"
        if freeze_manifest_path.exists():
            freeze_manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
            freeze_manifest_sha = sha256_hex(freeze_manifest_path)
        else:
            freeze_manifest = None
            freeze_manifest_sha = None

        # Verify dataset root
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

        # Build datasets and verify
        train_dataset, val_dataset, dataset_manifest = build_smoke_dataset(
            b01_freeze_dir=b01_freeze_dir,
            dataset_root=dataset_root,
            seed=cfg.get("seed", DEFAULT_SEED),
            n_train_subjects=2,
            n_val_subjects=1,
        )

        # Verify subject isolation
        if not verify_subject_isolation(
            dataset_manifest["train_subjects"],
            dataset_manifest["val_subjects"],
        ):
            raise ValueError("TRAIN/VAL subject overlap detected")

        # Build smoke config
        smoke_config = SmokeConfig(
            seed=cfg.get("seed", DEFAULT_SEED),
            batch_size=cfg.get("batch_size", DEFAULT_BATCH_SIZE),
            initial_epochs=cfg.get("epochs", {}).get("initial", DEFAULT_INITIAL_EPOCHS),
            resume_epochs=cfg.get("epochs", {}).get("resume", DEFAULT_RESUME_EPOCHS),
            lr=cfg.get("lr", DEFAULT_LR),
            weight_decay=cfg.get("weight_decay", DEFAULT_WEIGHT_DECAY),
            device=args.device,
        )

        # Run smoke test
        result = run_smoke_test(
            b01_freeze_dir=b01_freeze_dir,
            dataset_root=dataset_root,
            output_dir=output_dir,
            config=smoke_config,
            seed=smoke_config.seed,
            batch_size=smoke_config.batch_size,
            initial_epochs=smoke_config.initial_epochs,
            resume_epochs=smoke_config.resume_epochs,
            device=smoke_config.device,
        )

        # Write resolved config
        resolved_config = {
            "config_version": cfg.get("config_version"),
            "task_id": TASK_ID,
            "smoke_version": SMOKE_VERSION,
            "b01_freeze_dir": REDACTED_LOCAL_PATH,
            "dataset_root": REDACTED_LOCAL_PATH,
            "seed": smoke_config.seed,
            "batch_size": smoke_config.batch_size,
            "device": smoke_config.device,
            "lr": smoke_config.lr,
            "weight_decay": smoke_config.weight_decay,
            "initial_epochs": smoke_config.initial_epochs,
            "resume_epochs": smoke_config.resume_epochs,
            "absolute_paths_recorded": False,
        }
        _write_json(output_dir / "resolved_config.json", resolved_config)

        # Write input manifest hashes
        input_manifest_hashes = {
            "freeze_manifest_sha256": freeze_manifest_sha,
            "freeze_manifest_core_a06_split_sha256": A06_SPLIT_SHA256_EXPECTED,
            "normalization_stats_sha256": dataset_manifest.get("normalization_stats_sha256"),
        }
        _write_json(output_dir / "input_manifest_hashes.json", input_manifest_hashes)

        # Write runtime info
        runtime = {
            "wall_clock_seconds": result.training_time_seconds,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "started_at_utc": status["started_at_utc"],
            "ended_at_utc": _now_iso(),
        }
        _write_json(output_dir / "runtime.json", runtime)

        # Write artifacts
        model_config = {
            "model_version": cfg.get("model_version"),
            "n_classes": cfg.get("n_classes"),
        }
        write_smoke_artifacts(
            output_dir=output_dir,
            result=result,
            config=smoke_config,
            dataset_manifest=dataset_manifest,
            model_config=model_config,
        )

        # Update status
        status["status"] = "DONE" if result.success else "FAILED"
        status["ended_at_utc"] = _now_iso()
        status["verification_failures"] = result.verification_failures
        _write_json(output_dir / "status.json", status)

        # Print summary
        print(f"[B03 Smoke] Status: {'PASS' if result.success else 'FAIL'}")
        print(f"[B03 Smoke] Training time: {result.training_time_seconds:.2f}s")
        print(f"[B03 Smoke] TRAIN loss (initial): {result.train_loss_initial}")
        print(f"[B03 Smoke] VAL loss (initial): {result.val_loss_initial}")
        print(f"[B03 Smoke] TRAIN IoU: {result.train_metrics_initial.get('fixed_foreground_macro_iou', 'N/A')}")
        print(f"[B03 Smoke] VAL IoU: {result.val_metrics_initial.get('fixed_foreground_macro_iou', 'N/A')}")

        if result.verification_failures:
            print(f"[B03 Smoke] Verification failures:")
            for failure in result.verification_failures:
                print(f"  - {failure}")

        if result.success:
            print(f"[B03 Smoke] DONE.json written")
            return 0
        else:
            print(f"[B03 Smoke] FAILED.json written")
            return 1

    except Exception as exc:
        # Write FAILED.json
        failed_status = {
            "status": "FAILED",
            "task_id": TASK_ID,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "ended_at_utc": _now_iso(),
        }
        _write_json(output_dir / "FAILED.json", failed_status)

        # Update status
        status["status"] = "FAILED"
        status["error"] = str(exc)
        status["ended_at_utc"] = _now_iso()
        _write_json(output_dir / "status.json", status)

        print(f"[B03 Smoke] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
