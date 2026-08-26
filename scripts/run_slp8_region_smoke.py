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
* ``resolved_config.json`` — the parsed config
* ``input_manifest_hashes.json`` — input file hashes
* ``runtime.json`` — wall-clock timing
* ``metrics_summary.json`` — training and validation metrics
* ``metrics_by_region.csv`` — per-region metrics
* ``predictions_manifest.csv`` — per-prediction manifest
* ``failure_cases.csv`` — failure case list
* ``reload_consistency.json`` — checkpoint reload consistency result
* ``logs/run.log`` — run-time log
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
import time
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
    A06_SPLIT_SHA256_EXPECTED,
    sha256_file,
)
from topper_perception.neural.slp8_region_dataset import (
    N_CLASSES,
    build_smoke_dataset,
    verify_subject_isolation,
)
from topper_perception.neural.slp8_region_models import (
    MODEL_VERSION as MODEL_VERSION_CONST,
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

ALLOWED_DEVICES: tuple[str, ...] = ("cpu", "cuda")

REQUIRED_TOP_LEVEL_CONFIG_FIELDS: tuple[str, ...] = (
    "config_version",
    "task_id",
    "smoke_version",
    "provenance",
    "raw_semantics",
    "model",
    "training",
    "dataset",
)

REQUIRED_MODEL_FIELDS: tuple[str, ...] = (
    "n_classes",
    "input_shape",
)

REQUIRED_TRAINING_FIELDS: tuple[str, ...] = (
    "seed",
    "device",
    "batch_size",
    "lr",
    "weight_decay",
    "epochs",
)

REQUIRED_EPOCHS_FIELDS: tuple[str, ...] = (
    "initial",
    "resume",
)

REQUIRED_DATASET_FIELDS: tuple[str, ...] = (
    "smoke_subset",
    "normalization",
)

REQUIRED_SMOKE_SUBSET_FIELDS: tuple[str, ...] = (
    "n_train_subjects",
    "n_val_subjects",
    "seed",
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


def _expect_keys(parent: str, dct: dict[str, Any], required: tuple[str, ...]) -> None:
    missing = [k for k in required if k not in dct]
    if missing:
        raise ConfigValidationError(
            f"{parent} missing required fields: {missing}"
        )


def _expect_type(parent: str, value: Any, expected_type: type, field_name: str) -> None:
    if not isinstance(value, expected_type):
        raise ConfigValidationError(
            f"{parent}.{field_name} must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


def _validate_config(cfg: dict[str, Any]) -> None:
    """Validate the config dictionary using the nested structure.

    Required structure:
        cfg["task_id"]
        cfg["smoke_version"]
        cfg["provenance"]
        cfg["raw_semantics"]
        cfg["model"]["n_classes"]
        cfg["training"]["seed"]
        cfg["training"]["device"]
        cfg["training"]["batch_size"]
        cfg["training"]["lr"]
        cfg["training"]["weight_decay"]
        cfg["training"]["epochs"]["initial"]
        cfg["training"]["epochs"]["resume"]
        cfg["dataset"]["smoke_subset"]["n_train_subjects"]
        cfg["dataset"]["smoke_subset"]["n_val_subjects"]
        cfg["dataset"]["smoke_subset"]["seed"]
    """
    # Top-level fields
    _expect_keys("config", cfg, REQUIRED_TOP_LEVEL_CONFIG_FIELDS)

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

    # model
    model_cfg = cfg["model"]
    _expect_keys("config.model", model_cfg, REQUIRED_MODEL_FIELDS)
    _expect_type("config.model", model_cfg["n_classes"], int, "n_classes")
    if model_cfg["n_classes"] != N_CLASSES:
        raise ConfigValidationError(
            f"config.model.n_classes must be {N_CLASSES}, "
            f"got {model_cfg['n_classes']}"
        )
    _expect_type("config.model", model_cfg["input_shape"], list, "input_shape")
    if tuple(model_cfg["input_shape"]) != (192, 84):
        raise ConfigValidationError(
            f"config.model.input_shape must be [192, 84], "
            f"got {model_cfg['input_shape']}"
        )

    # training
    training_cfg = cfg["training"]
    _expect_keys("config.training", training_cfg, REQUIRED_TRAINING_FIELDS)
    _expect_type("config.training", training_cfg["seed"], int, "seed")
    _expect_type("config.training", training_cfg["device"], str, "device")
    if training_cfg["device"] not in ALLOWED_DEVICES:
        raise ConfigValidationError(
            f"config.training.device must be one of {ALLOWED_DEVICES}, "
            f"got {training_cfg['device']!r}"
        )
    _expect_type("config.training", training_cfg["batch_size"], int, "batch_size")
    if training_cfg["batch_size"] <= 0:
        raise ConfigValidationError(
            f"config.training.batch_size must be positive, "
            f"got {training_cfg['batch_size']}"
        )
    _expect_type("config.training", training_cfg["lr"], (int, float), "lr")
    _expect_type(
        "config.training", training_cfg["weight_decay"], (int, float), "weight_decay"
    )

    # training.epochs
    epochs_cfg = training_cfg["epochs"]
    _expect_keys("config.training.epochs", epochs_cfg, REQUIRED_EPOCHS_FIELDS)
    _expect_type(
        "config.training.epochs", epochs_cfg["initial"], int, "initial"
    )
    _expect_type(
        "config.training.epochs", epochs_cfg["resume"], int, "resume"
    )
    if epochs_cfg["initial"] < 1:
        raise ConfigValidationError(
            f"config.training.epochs.initial must be >= 1, "
            f"got {epochs_cfg['initial']}"
        )
    if epochs_cfg["resume"] < 1:
        raise ConfigValidationError(
            f"config.training.epochs.resume must be >= 1, "
            f"got {epochs_cfg['resume']}"
        )

    # dataset
    dataset_cfg = cfg["dataset"]
    _expect_keys("config.dataset", dataset_cfg, REQUIRED_DATASET_FIELDS)
    smoke_subset_cfg = dataset_cfg["smoke_subset"]
    _expect_keys(
        "config.dataset.smoke_subset", smoke_subset_cfg, REQUIRED_SMOKE_SUBSET_FIELDS
    )
    _expect_type(
        "config.dataset.smoke_subset",
        smoke_subset_cfg["n_train_subjects"],
        int,
        "n_train_subjects",
    )
    _expect_type(
        "config.dataset.smoke_subset",
        smoke_subset_cfg["n_val_subjects"],
        int,
        "n_val_subjects",
    )
    if smoke_subset_cfg["n_train_subjects"] != 2:
        raise ConfigValidationError(
            f"config.dataset.smoke_subset.n_train_subjects must be 2, "
            f"got {smoke_subset_cfg['n_train_subjects']}"
        )
    if smoke_subset_cfg["n_val_subjects"] != 1:
        raise ConfigValidationError(
            f"config.dataset.smoke_subset.n_val_subjects must be 1, "
            f"got {smoke_subset_cfg['n_val_subjects']}"
        )
    _expect_type(
        "config.dataset.smoke_subset",
        smoke_subset_cfg["seed"],
        int,
        "seed",
    )

    # dataset.normalization
    normalization_cfg = dataset_cfg.get("normalization", {})
    if normalization_cfg.get("method") != "raw_passthrough_with_minmax_reference":
        raise ConfigValidationError(
            "config.dataset.normalization.method must be "
            "'raw_passthrough_with_minmax_reference'"
        )
    if normalization_cfg.get("fit_split") != "train":
        raise ConfigValidationError(
            "config.dataset.normalization.fit_split must be 'train'"
        )
    if normalization_cfg.get("raw_semantics") != "raw_pmarray_response":
        raise ConfigValidationError(
            "config.dataset.normalization.raw_semantics must be "
            "'raw_pmarray_response'"
        )


def _build_smoke_config(cfg: dict[str, Any], device_override: str) -> SmokeConfig:
    """Build SmokeConfig from the validated nested config.

    The subset parameters (``n_train_subjects``, ``n_val_subjects``,
    ``subset_seed``) are read from the config and passed to the
    SmokeConfig, so the runner cannot silently fall back to hard-coded
    defaults.
    """
    return SmokeConfig(
        seed=int(cfg["training"]["seed"]),
        batch_size=int(cfg["training"]["batch_size"]),
        initial_epochs=int(cfg["training"]["epochs"]["initial"]),
        resume_epochs=int(cfg["training"]["epochs"]["resume"]),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        device=device_override,
        n_train_subjects=int(cfg["dataset"]["smoke_subset"]["n_train_subjects"]),
        n_val_subjects=int(cfg["dataset"]["smoke_subset"]["n_val_subjects"]),
        subset_seed=int(cfg["dataset"]["smoke_subset"]["seed"]),
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

    sentinel_files = ("DONE.json", "FAILED.json")
    for sentinel in sentinel_files:
        if (output_dir / sentinel).is_file():
            raise OutputDirCollisionError(
                f"output directory already contains {sentinel}; refusing to "
                f"overwrite.  Choose a fresh --output-dir.  ({output_dir})"
            )

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
        default=None,
        help="Device override; if not provided, taken from config.training.device.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    b01_freeze_dir = Path(args.b01_freeze_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()

    # Validate config (fail-closed)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    _validate_config(cfg)

    # Determine device: CLI override > config
    device = args.device if args.device is not None else cfg["training"]["device"]
    if device not in ALLOWED_DEVICES:
        raise ConfigValidationError(
            f"device must be one of {ALLOWED_DEVICES}, got {device!r}"
        )

    # Check output directory safety
    _check_output_dir_safety(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    def _log(msg: str) -> None:
        line = f"[{_now_iso()}] {msg}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
        print(msg, flush=True)

    _log(f"task_id={TASK_ID}")
    _log(f"smoke_version={SMOKE_VERSION}")
    _log(f"device={device}")
    _log(f"output_dir={output_dir}")
    _log(f"b01_freeze_dir={b01_freeze_dir}")
    _log(f"dataset_root={dataset_root}")

    # Write initial status
    status: dict[str, Any] = {
        "task_id": TASK_ID,
        "smoke_version": SMOKE_VERSION,
        "status": "RUNNING",
        "started_at_utc": _now_iso(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    _write_json(output_dir / "status.json", status)

    t_start = time.perf_counter()

    try:
        # Verify B01 freeze directory
        if not b01_freeze_dir.is_dir():
            raise FileNotFoundError(f"B01 freeze directory not found: {b01_freeze_dir}")

        freeze_manifest_path = b01_freeze_dir / "freeze_manifest.json"
        if freeze_manifest_path.exists():
            freeze_manifest = json.loads(
                freeze_manifest_path.read_text(encoding="utf-8")
            )
            freeze_manifest_sha = sha256_file(freeze_manifest_path)
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
            seed=int(cfg["dataset"]["smoke_subset"]["seed"]),
            n_train_subjects=int(cfg["dataset"]["smoke_subset"]["n_train_subjects"]),
            n_val_subjects=int(cfg["dataset"]["smoke_subset"]["n_val_subjects"]),
        )

        if not verify_subject_isolation(
            dataset_manifest["train_subjects"],
            dataset_manifest["val_subjects"],
        ):
            raise ValueError("TRAIN/VAL subject overlap detected")

        _log(
            f"train_subjects={dataset_manifest['train_subjects']} "
            f"n_train={dataset_manifest['n_train_samples']}"
        )
        _log(
            f"val_subjects={dataset_manifest['val_subjects']} "
            f"n_val={dataset_manifest['n_val_samples']}"
        )
        _log(f"n_test={dataset_manifest['n_test_samples']} (must be 0)")

        # Build smoke config from validated nested config
        smoke_config = _build_smoke_config(cfg, device_override=device)

        # Run smoke test
        result = run_smoke_test(
            b01_freeze_dir=b01_freeze_dir,
            dataset_root=dataset_root,
            output_dir=output_dir,
            config=smoke_config,
        )

        # Write resolved config (path fields redacted)
        resolved_config = {
            "config_version": cfg.get("config_version"),
            "task_id": TASK_ID,
            "smoke_version": SMOKE_VERSION,
            "model_version": cfg["model"].get("architecture", MODEL_VERSION_CONST),
            "checkpoint_version": cfg.get("checkpoint_version"),
            "freeze_version": cfg.get("freeze_version"),
            "b01_freeze_dir": REDACTED_LOCAL_PATH,
            "dataset_root": REDACTED_LOCAL_PATH,
            "device": device,
            "training": {
                "seed": smoke_config.seed,
                "batch_size": smoke_config.batch_size,
                "lr": smoke_config.lr,
                "weight_decay": smoke_config.weight_decay,
                "epochs": {
                    "initial": smoke_config.initial_epochs,
                    "resume": smoke_config.resume_epochs,
                },
            },
            "dataset": {
                "smoke_subset": {
                    "n_train_subjects": int(
                        cfg["dataset"]["smoke_subset"]["n_train_subjects"]
                    ),
                    "n_val_subjects": int(
                        cfg["dataset"]["smoke_subset"]["n_val_subjects"]
                    ),
                    "seed": int(cfg["dataset"]["smoke_subset"]["seed"]),
                },
            },
            "absolute_paths_recorded": False,
        }
        _write_json(output_dir / "resolved_config.json", resolved_config)

        # Write input manifest hashes
        input_manifest_hashes = {
            "freeze_manifest_sha256": freeze_manifest_sha,
            "freeze_manifest_core_a06_split_sha256": A06_SPLIT_SHA256_EXPECTED,
            "normalization_stats_sha256": dataset_manifest.get(
                "normalization_stats_sha256"
            ),
        }
        _write_json(output_dir / "input_manifest_hashes.json", input_manifest_hashes)

        t_end = time.perf_counter()
        wall_clock = t_end - t_start

        # Write runtime info
        runtime = {
            "wall_clock_seconds": wall_clock,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "started_at_utc": status["started_at_utc"],
            "ended_at_utc": _now_iso(),
        }
        _write_json(output_dir / "runtime.json", runtime)

        # Write artifacts
        model_config = {
            "model_version": cfg["model"].get("architecture", MODEL_VERSION_CONST),
            "n_classes": int(cfg["model"]["n_classes"]),
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

        _log(
            f"status={status['status']} wall_clock={wall_clock:.2f}s"
        )
        if result.train_loss_initial is not None:
            _log(
                f"train_loss_initial={result.train_loss_initial:.6f} "
                f"val_loss_initial={result.val_loss_initial:.6f}"
            )
        if result.train_loss_resumed is not None:
            _log(
                f"train_loss_resumed={result.train_loss_resumed:.6f} "
                f"val_loss_resumed={result.val_loss_resumed:.6f}"
            )
        if result.train_metrics_initial:
            _log(
                f"train_iou={result.train_metrics_initial.get('fixed_foreground_macro_iou')}"
            )
        if result.val_metrics_initial:
            _log(
                f"val_iou={result.val_metrics_initial.get('fixed_foreground_macro_iou')}"
            )
        if result.checkpoint_sha_initial:
            _log(
                f"checkpoint_initial_sha256={result.checkpoint_sha_initial[:16]}..."
            )
        if result.checkpoint_sha_resumed:
            _log(
                f"checkpoint_resumed_sha256={result.checkpoint_sha_resumed[:16]}..."
            )
        _log(
            f"param_changed_after_initial={result.param_changed_after_initial} "
            f"param_changed_after_resume={result.param_changed_after_resume}"
        )
        _log(f"reload_consistent={result.reload_consistent}")

        if result.verification_failures:
            for failure in result.verification_failures:
                _log(f"verification_failure: {failure}")

        if result.success:
            return 0
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

        _log(f"FAILED: {exc}")
        _log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
