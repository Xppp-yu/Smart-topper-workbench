"""Run the SLP8 B02 v0.1 non-learning region baseline.

CLI for the SLP8 pressure-only non-learning region baseline (TASK-SLP-
B02-NON-LEARNING-REGION-BASELINE-v0.1).  The script is CPU-only,
deterministic, and never reads TEST label/onehot.

Usage
-----

::

    uv run python scripts/run_slp8_non_learning_region_baseline.py \\
        --config configs/experiments/slp8_non_learning_region_baseline_v0.1.json \\
        --output-dir outputs/experiments/EXP-SLP-B02-NONLEARNING-DEV-20260825-R01

Inputs
------
* ``--config`` — path to a JSON config (see
  ``configs/experiments/slp8_non_learning_region_baseline_v0.1.json``)
* ``--output-dir`` — directory where the run artefacts will be written
* ``--b01-freeze-dir`` — optional override for the B01 freeze directory;
  defaults to the value in the config
* ``--dataset-root`` — optional override for the SLP8 dataset root
  (must be a real path; never written into the config)

Outputs
-------
The runner writes the following artefacts to ``--output-dir``:

* ``status.json`` — overall status (DONE / FAILED)
* ``resolved_config.json`` — config as actually used (no absolute paths)
* ``input_manifest_hashes.json`` — SHA-256 of the B01 freeze manifest
  and the source manifest
* ``metrics_summary.json`` — overall metric summary
* ``metrics_by_baseline.csv`` — per-baseline metrics
* ``metrics_by_region.csv`` — per-region metrics
* ``metrics_by_subject.csv`` — per-subject metrics
* ``metrics_by_posture.csv`` — per-posture metrics
* ``predictions_manifest.csv`` — per-prediction summary
* ``failure_cases.csv`` — list of failure cases (if any)
* ``failure_reason_counts.json`` — counts of failure reasons
* ``runtime.json`` — wall-clock timing info
* ``DONE.json`` or ``FAILED.json``

A ``predictions/<sample_id>.npz`` archive (gzip-compressed) is written
for each prediction; it is gitignored.  The archive contains the
predicted label map and a SHA-256.  The raw pressure array is never
written.

The runner is intentionally side-effect-free outside of ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Ensure ``src`` is on the import path so the script can be invoked
# directly via ``uv run python scripts/run_...``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topper_perception.baseline.slp8_non_learning import (  # noqa: E402
    AllBackgroundBaseline,
    AxisPartitionConfig,
    AxisPartitionState,
    BASELINE_VERSION,
    BACKGROUND_ID,
    PRESSURE_SHAPE,
    PressureAxisContactIntersectionBaseline,
    PressureBodyAxisPartitionBaseline,
    REGION_IDS,
    REGION_ID_TO_NAME,
    REGION_NAMES,
    TrainSpatialPriorBaseline,
    fit_axis_partition_config,
)
from topper_perception.evaluation.slp_pressure_metrics import (  # noqa: E402
    compute_fixed_class_macro_metrics,
)
from topper_perception.io.slp8_training_table_freeze import (  # noqa: E402
    EXPECTED_PROVENANCE,
    EXPECTED_REVIEW_STATUS,
    FREEZE_VERSION,
    TASK_ID as B01_TASK_ID,
    A06_SPLIT_SHA256_EXPECTED,
    B01FreezeTables,
    FreezeRow,
    ML_SPLITS,
    load_b01_freeze_tables,
    sha256_hex,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_ID: str = "TASK-SLP-B02-NON-LEARNING-REGION-BASELINE-v0.1"

#: Default config version recorded in every artefact.
DEFAULT_CONFIG_VERSION: str = "slp8_non_learning_v0.1"

#: Maximum predictions to keep on disk per baseline (gitignored).
MAX_PREDICTIONS_PER_BASELINE: int = 1024

#: Failure reason taxonomy (stable strings; used in CSV and counts).
FAILURE_REASONS: tuple[str, ...] = (
    "shape_mismatch",
    "non_finite_pressure",
    "label_out_of_range",
    "file_not_found",
    "wrong_provenance",
    "wrong_review_status",
    "wrong_subject_split",
    "no_contact",
    "degenerate_pca",
    "internal_exception",
)


# ---------------------------------------------------------------------------
# Config schema (lightweight; we keep the JSON config explicit and
# versioned, and we validate it field-by-field).
# ---------------------------------------------------------------------------


REQUIRED_CONFIG_FIELDS: tuple[str, ...] = (
    "config_version",
    "task_id",
    "freeze_version",
    "b01_freeze_dir",
    "b01_a06_split_sha256_expected",
    "baselines",
    "metrics",
    "provenance",
    "raw_semantics",
    "fit_split",
)


def _validate_config(cfg: dict[str, Any]) -> None:
    """Validate the resolved config dictionary."""
    missing = [f for f in REQUIRED_CONFIG_FIELDS if f not in cfg]
    if missing:
        raise ValueError(f"config missing required fields: {missing}")
    if cfg["task_id"] != TASK_ID:
        raise ValueError(
            f"config task_id {cfg['task_id']!r} != expected {TASK_ID!r}"
        )
    if cfg["fit_split"] != "train":
        raise ValueError(
            f"config fit_split must be 'train' (got {cfg['fit_split']!r})"
        )
    if cfg["provenance"] != EXPECTED_PROVENANCE:
        raise ValueError(
            f"config provenance must be {EXPECTED_PROVENANCE!r}"
        )
    if cfg["raw_semantics"] != "raw_pmarray_response":
        raise ValueError(
            f"config raw_semantics must be 'raw_pmarray_response'"
        )
    for bl in cfg["baselines"]:
        if bl["name"] not in (
            "all_background",
            "train_spatial_prior",
            "pressure_body_axis_partition",
            "pressure_axis_contact_intersection",
        ):
            raise ValueError(f"unknown baseline name: {bl['name']!r}")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v):
            return None
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


def _per_class_pixel_counts(label: np.ndarray) -> list[int]:
    counts = np.bincount(label.flatten().astype(np.int64), minlength=9)
    return [int(c) for c in counts]


def _region_centroid_xy(label: np.ndarray, class_id: int) -> tuple[float | None, int]:
    """Return (centroid_xy_in_pixels, n_pixels) for a single class.

    If the class is absent, return (None, 0)."""
    mask = label == int(class_id)
    n = int(mask.sum())
    if n == 0:
        return None, 0
    ys, xs = np.nonzero(mask)
    cx = float(xs.mean())
    cy = float(ys.mean())
    return (cx, cy), n


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class SamplePrediction:
    """One prediction for one sample by one baseline."""

    sample_id: str
    ml_split: str
    subject_id: str
    posture: str
    baseline: str
    pred_label_path: str
    pred_label_sha256: str
    pred_class_pixel_counts: list[int]
    pred_class_present: list[bool]
    failure_reason: str | None
    runtime_ms: float


def _predict_with_failure_capture(
    *,
    baseline_obj: Any,
    baseline_name: str,
    pressure: np.ndarray,
) -> tuple[np.ndarray, str | None, float]:
    """Call baseline.predict(pressure); on any contract error, return
    a zero label map and a failure reason.  Any unexpected exception
    is captured with a generic reason."""
    t0 = time.perf_counter()
    try:
        pred = baseline_obj.predict(pressure)
        return pred, None, (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # noqa: BLE001 - we want to capture any
        # contract violation; the failure must be auditable.
        label = np.zeros(PRESSURE_SHAPE, dtype=np.uint8)
        if isinstance(exc, (TypeError, ValueError)) and "shape" in str(exc).lower():
            reason = "shape_mismatch"
        elif "non-finite" in str(exc).lower() or "non_finite" in str(exc).lower():
            reason = "non_finite_pressure"
        elif "label range" in str(exc).lower() or "label_map values" in str(exc).lower():
            reason = "label_out_of_range"
        else:
            reason = "internal_exception"
        return label, reason, (time.perf_counter() - t0) * 1000.0


def run(config: dict[str, Any], output_dir: Path) -> int:
    """Execute the run; return 0 on success, 1 on failure."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    status: dict[str, Any] = {
        "task_id": TASK_ID,
        "baseline_version": BASELINE_VERSION,
        "status": "RUNNING",
        "started_at_utc": _now_iso(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    _write_json(output_dir / "status.json", status)

    t_start = time.perf_counter()
    failure_counts: Counter[str] = Counter()
    failure_cases: list[dict[str, Any]] = []
    predictions_manifest: list[dict[str, Any]] = []
    per_region_records: list[dict[str, Any]] = []
    per_baseline_records: list[dict[str, Any]] = []
    per_subject_records: list[dict[str, Any]] = []
    per_posture_records: list[dict[str, Any]] = []

    try:
        b01_dir = Path(config["b01_freeze_dir"]).resolve()
        if not b01_dir.is_dir():
            raise FileNotFoundError(
                f"b01_freeze_dir does not exist: {b01_dir}"
            )

        # Input manifest hash
        fm_path = b01_dir / "freeze_manifest.json"
        if not fm_path.is_file():
            raise FileNotFoundError(f"B01 freeze manifest missing: {fm_path}")
        fm_bytes = fm_path.read_bytes()
        fm_sha = sha256_hex(fm_bytes)
        # Cross-check A06 SHA recorded in the freeze manifest core.
        fm = json.loads(fm_bytes.decode("utf-8"))
        a06_sha_recorded = fm["core"]["a06_split_sha256"]
        if a06_sha_recorded != A06_SPLIT_SHA256_EXPECTED:
            raise ValueError(
                f"A06 split SHA mismatch: freeze manifest has "
                f"{a06_sha_recorded!r}, expected {A06_SPLIT_SHA256_EXPECTED!r}"
            )
        # Source manifest SHA
        source_manifest_path = b01_dir / "train_manifest.csv"
        source_manifest_sha = sha256_hex(source_manifest_path.read_bytes()) if source_manifest_path.is_file() else None
        # Also check val manifest exists.
        val_manifest_sha = sha256_hex((b01_dir / "val_manifest.csv").read_bytes()) if (b01_dir / "val_manifest.csv").is_file() else None

        _write_json(
            output_dir / "input_manifest_hashes.json",
            {
                "freeze_manifest_file_sha256": fm_sha,
                "freeze_manifest_core_a06_split_sha256": a06_sha_recorded,
                "freeze_manifest_core_a06_split_sha256_expected": A06_SPLIT_SHA256_EXPECTED,
                "freeze_manifest_core_source_manifest_sha256": fm["core"].get("source_manifest_sha256"),
                "freeze_manifest_core_train_manifest_sha256": fm["core"]["splits"]["train"].get("manifest_sha256"),
                "freeze_manifest_core_val_manifest_sha256": fm["core"]["splits"]["val"].get("manifest_sha256"),
                "freeze_manifest_core_normalization_stats_sha256": fm["core"].get("normalization_stats_sha256"),
                "train_manifest_file_sha256": source_manifest_sha,
                "val_manifest_file_sha256": val_manifest_sha,
                "freeze_version": FREEZE_VERSION,
                "b01_task_id": B01_TASK_ID,
                "raw_semantics": "raw_pmarray_response",
                "annotation_provenance": EXPECTED_PROVENANCE,
                "source_review_status": EXPECTED_REVIEW_STATUS,
                "test_access": "denied (this run)",
            },
        )

        # Resolved config (no absolute paths; record only file existence).
        resolved_config = {
            **config,
            "b01_freeze_dir_exists": True,
            "b01_freeze_dir_freeze_manifest_sha256": fm_sha,
            "resolved_at_utc": _now_iso(),
            "absolute_paths_recorded": False,
        }
        _write_json(output_dir / "resolved_config.json", resolved_config)

        # Load B01 freeze tables (TRAIN + VAL only; default load_test=False).
        tables = load_b01_freeze_tables(b01_dir, load_test=False)
        train_rows = list(tables.train_rows)
        val_rows = list(tables.val_rows)
        all_dev_rows = train_rows + val_rows

        # Fit the baselines on TRAIN only.
        # Iterate once to compute the contact threshold / template
        # state; we need actual pressure + label arrays.
        from topper_perception.io.slp_8region_pressure_dataset import (
            Slp8RegionDatasetAdapter,
        )
        # We use the raw A09R adapter to read individual samples.  The
        # B01 freeze row already contains the relative paths.  We
        # resolve them through the dataset root, which the runner gets
        # from the freeze manifest's ``source_dataset_id`` (no absolute
        # path is recorded).

        # We don't have the dataset root as an absolute path; it must
        # be passed via the config or inferred from the manifest.
        dataset_root_str = config.get("dataset_root")
        if dataset_root_str is None:
            raise ValueError(
                "config.dataset_root is required to read pressure / label arrays"
            )
        dataset_root = Path(dataset_root_str).resolve()
        if not (dataset_root / "manifest" / "val_manifest.csv").is_file():
            raise FileNotFoundError(
                f"SLP8 dataset_root missing manifest/val_manifest.csv: {dataset_root}"
            )

        adapter = Slp8RegionDatasetAdapter(dataset_root, validate_on_load=True)

        # Map sample_id → loaded arrays (TRAIN only — kept in memory
        # because the template fit needs the entire TRAIN set).
        # The TRAIN set is 3645 samples × 192 × 84 × 8 bytes = ~470 MB;
        # this is acceptable for a CPU-only dev run.
        t_load0 = time.perf_counter()
        train_pressures: list[np.ndarray] = []
        train_labels: list[np.ndarray] = []
        train_subjects: list[str] = []
        train_postures: list[str] = []
        train_sids: list[str] = []
        for r in train_rows:
            try:
                loaded = adapter.load_sample(r.sample_id)
            except Exception as exc:  # noqa: BLE001
                failure_counts["file_not_found"] += 1
                failure_cases.append(
                    {
                        "sample_id": r.sample_id,
                        "ml_split": r.ml_split,
                        "subject_id": r.subject_id,
                        "posture": r.posture,
                        "baseline": "load_train",
                        "reason": "file_not_found",
                        "detail": str(exc),
                    }
                )
                continue
            train_pressures.append(loaded.pressure)
            train_labels.append(loaded.region_label)
            train_subjects.append(r.subject_id)
            train_postures.append(r.posture)
            train_sids.append(r.sample_id)
        t_load = time.perf_counter() - t_load0

        # Fit the baselines.
        baselines_state: dict[str, Any] = {}
        for bl_cfg in config["baselines"]:
            name = bl_cfg["name"]
            if name == "all_background":
                baselines_state[name] = AllBackgroundBaseline()
            elif name == "train_spatial_prior":
                tsp = TrainSpatialPriorBaseline()
                tsp.fit(
                    train_labels,
                    subject_ids=train_subjects,
                )
                baselines_state[name] = tsp
            elif name == "pressure_body_axis_partition":
                ap_cfg = AxisPartitionConfig(**{
                    k: v for k, v in bl_cfg.items()
                    if k in {
                        "contact_fraction",
                        "contact_smooth_iters",
                        "segment_fractions",
                        "lateral_half_width",
                        "region_priority",
                    }
                })
                bap = PressureBodyAxisPartitionBaseline()
                bap.fit(train_pressures, config=ap_cfg)
                baselines_state[name] = bap
            elif name == "pressure_axis_contact_intersection":
                ap_cfg = AxisPartitionConfig(**{
                    k: v for k, v in bl_cfg.items()
                    if k in {
                        "contact_fraction",
                        "contact_smooth_iters",
                        "segment_fractions",
                        "lateral_half_width",
                        "region_priority",
                    }
                })
                ib = PressureAxisContactIntersectionBaseline()
                ib.fit(
                    train_labels,
                    train_pressures,
                    config=ap_cfg,
                    subject_ids=train_subjects,
                )
                baselines_state[name] = ib
            else:
                raise ValueError(f"unknown baseline: {name!r}")

        # Per-baseline evaluation: TRAIN + VAL.
        for bl_name, bl_obj in baselines_state.items():
            t_bl0 = time.perf_counter()
            all_preds: list[np.ndarray] = []
            all_gts: list[np.ndarray] = []
            per_subj_metrics: dict[str, dict[str, list[float]]] = {}
            per_posture_metrics: dict[str, dict[str, list[float]]] = {}
            for r in all_dev_rows:
                try:
                    loaded = adapter.load_sample(r.sample_id)
                except Exception as exc:  # noqa: BLE001
                    failure_counts["file_not_found"] += 1
                    failure_cases.append(
                        {
                            "sample_id": r.sample_id,
                            "ml_split": r.ml_split,
                            "subject_id": r.subject_id,
                            "posture": r.posture,
                            "baseline": bl_name,
                            "reason": "file_not_found",
                            "detail": str(exc),
                        }
                    )
                    continue
                pred, reason, rt_ms = _predict_with_failure_capture(
                    baseline_obj=bl_obj,
                    baseline_name=bl_name,
                    pressure=loaded.pressure,
                )
                if reason is not None:
                    failure_counts[reason] += 1
                    failure_cases.append(
                        {
                            "sample_id": r.sample_id,
                            "ml_split": r.ml_split,
                            "subject_id": r.subject_id,
                            "posture": r.posture,
                            "baseline": bl_name,
                            "reason": reason,
                            "detail": "see exception log",
                        }
                    )
                # Persist prediction (gzip-compressed npz).
                pred_label_path = predictions_dir / f"{_safe_filename(r.sample_id)}__{_safe_filename(bl_name)}.npz"
                np.savez_compressed(
                    pred_label_path,
                    pred=pred,
                    sample_id=np.array(r.sample_id),
                    baseline=np.array(bl_name),
                    ml_split=np.array(r.ml_split),
                    subject_id=np.array(r.subject_id),
                    posture=np.array(r.posture),
                )
                pred_sha = sha256_hex(pred_label_path.read_bytes())

                # Manifest
                predictions_manifest.append(
                    {
                        "sample_id": r.sample_id,
                        "ml_split": r.ml_split,
                        "subject_id": r.subject_id,
                        "posture": r.posture,
                        "baseline": bl_name,
                        "pred_label_path": str(pred_label_path.relative_to(output_dir)),
                        "pred_label_sha256": pred_sha,
                        "pred_class_pixel_counts": _per_class_pixel_counts(pred),
                        "pred_class_present": [
                            int(_per_class_pixel_counts(pred)[c]) > 0
                            for c in range(9)
                        ],
                        "failure_reason": reason,
                        "runtime_ms": rt_ms,
                    }
                )

                if reason is None:
                    all_preds.append(pred)
                    all_gts.append(loaded.region_label)

                    per_subj_metrics.setdefault(
                        r.subject_id,
                        {
                            "fixed_iou": [],
                            "fixed_dice": [],
                            "pixel_accuracy": [],
                        },
                    )
                    m1 = compute_fixed_class_macro_metrics(
                        loaded.region_label, pred,
                        class_ids=REGION_IDS, n_classes=9,
                    )
                    per_subj_metrics[r.subject_id]["fixed_iou"].append(m1.fixed_iou)
                    per_subj_metrics[r.subject_id]["fixed_dice"].append(m1.fixed_dice)
                    per_subj_metrics[r.subject_id]["pixel_accuracy"].append(m1.pixel_accuracy)

                    per_posture_metrics.setdefault(
                        r.posture,
                        {
                            "fixed_iou": [],
                            "fixed_dice": [],
                            "pixel_accuracy": [],
                        },
                    )
                    per_posture_metrics[r.posture]["fixed_iou"].append(m1.fixed_iou)
                    per_posture_metrics[r.posture]["fixed_dice"].append(m1.fixed_dice)
                    per_posture_metrics[r.posture]["pixel_accuracy"].append(m1.pixel_accuracy)

            # Aggregate per-baseline metrics.
            if all_preds:
                m = compute_fixed_class_macro_metrics(
                    all_gts, all_preds,
                    class_ids=REGION_IDS, n_classes=9,
                )
                # Centroid error (per region) vs GT centroid.
                centroid_errors: dict[str, float | None] = {}
                for cid in REGION_IDS:
                    errs: list[float] = []
                    for gt, pr in zip(all_gts, all_preds):
                        gt_c, gt_n = _region_centroid_xy(gt, cid)
                        pr_c, pr_n = _region_centroid_xy(pr, cid)
                        if gt_n == 0 or pr_n == 0:
                            continue
                        dx = float(gt_c[0] - pr_c[0])  # type: ignore[index]
                        dy = float(gt_c[1] - pr_c[1])  # type: ignore[index]
                        errs.append(float(np.sqrt(dx * dx + dy * dy)))
                    if errs:
                        centroid_errors[str(cid)] = float(np.mean(errs))
                    else:
                        centroid_errors[str(cid)] = None
                # Region coverage: how many GT regions were predicted at all.
                coverage_per_region = {
                    str(cid): int(sum(1 for gt in all_gts if (gt == cid).any())) for cid in REGION_IDS
                }
                pred_coverage_per_region = {
                    str(cid): int(sum(1 for pr in all_preds if (pr == cid).any())) for cid in REGION_IDS
                }
                record = {
                    "baseline": bl_name,
                    "n_samples_evaluated": len(all_preds),
                    "n_samples_failed": sum(1 for p in predictions_manifest if p["baseline"] == bl_name and p["failure_reason"] is not None),
                    "fixed_iou": m.fixed_iou,
                    "fixed_dice": m.fixed_dice,
                    "pixel_accuracy": m.pixel_accuracy,
                    "n_classes_present_in_pred": m.n_classes_present_in_pred,
                    "n_classes_present_in_gt": m.n_classes_present_in_gt,
                    "per_class_iou": {str(k): v for k, v in m.per_class_iou.items()},
                    "per_class_dice": {str(k): v for k, v in m.per_class_dice.items()},
                    "per_class_pred_count": {str(k): v for k, v in m.per_class_pred_count.items()},
                    "per_class_gt_count": {str(k): v for k, v in m.per_class_gt_count.items()},
                    "per_class_present_in_gt": {str(k): v for k, v in m.per_class_present_in_gt.items()},
                    "per_class_present_in_pred": {str(k): v for k, v in m.per_class_present_in_pred.items()},
                    "centroid_error_px": centroid_errors,
                    "region_coverage_gt": coverage_per_region,
                    "region_coverage_pred": pred_coverage_per_region,
                    "runtime_seconds": time.perf_counter() - t_bl0,
                }
            else:
                record = {
                    "baseline": bl_name,
                    "n_samples_evaluated": 0,
                    "n_samples_failed": len([p for p in predictions_manifest if p["baseline"] == bl_name]),
                    "fixed_iou": 0.0,
                    "fixed_dice": 0.0,
                    "pixel_accuracy": 0.0,
                    "n_classes_present_in_pred": 0,
                    "n_classes_present_in_gt": 0,
                    "per_class_iou": {str(k): 0.0 for k in REGION_IDS},
                    "per_class_dice": {str(k): 0.0 for k in REGION_IDS},
                    "per_class_pred_count": {str(k): 0 for k in REGION_IDS},
                    "per_class_gt_count": {str(k): 0 for k in REGION_IDS},
                    "per_class_present_in_gt": {str(k): False for k in REGION_IDS},
                    "per_class_present_in_pred": {str(k): False for k in REGION_IDS},
                    "centroid_error_px": {str(k): None for k in REGION_IDS},
                    "region_coverage_gt": {str(k): 0 for k in REGION_IDS},
                    "region_coverage_pred": {str(k): 0 for k in REGION_IDS},
                    "runtime_seconds": time.perf_counter() - t_bl0,
                }
            per_baseline_records.append(record)

            # Per-region CSV records.
            for cid in REGION_IDS:
                per_region_records.append(
                    {
                        "baseline": bl_name,
                        "region_id": int(cid),
                        "region_name": REGION_ID_TO_NAME[int(cid)],
                        "iou": record["per_class_iou"][str(cid)],
                        "dice": record["per_class_dice"][str(cid)],
                        "precision": float(
                            record["per_class_iou"][str(cid)]
                            / (2 * record["per_class_iou"][str(cid)] - record["per_class_dice"][str(cid)])
                            if (2 * record["per_class_iou"][str(cid)] - record["per_class_dice"][str(cid)]) > 0
                            else 0.0
                        ),
                        "recall": 0.0,
                        "pred_count": record["per_class_pred_count"][str(cid)],
                        "gt_count": record["per_class_gt_count"][str(cid)],
                        "is_present_in_pred": record["per_class_present_in_pred"][str(cid)],
                        "is_present_in_gt": record["per_class_present_in_gt"][str(cid)],
                        "centroid_error_px": record["centroid_error_px"][str(cid)],
                    }
                )

            # Per-subject records.
            for subj, d in per_subj_metrics.items():
                per_subject_records.append(
                    {
                        "baseline": bl_name,
                        "subject_id": subj,
                        "ml_split": next(
                            (r.ml_split for r in all_dev_rows if r.subject_id == subj),
                            "train",
                        ),
                        "n_samples": len(d["fixed_iou"]),
                        "mean_fixed_iou": float(np.mean(d["fixed_iou"])),
                        "mean_fixed_dice": float(np.mean(d["fixed_dice"])),
                        "mean_pixel_accuracy": float(np.mean(d["pixel_accuracy"])),
                    }
                )

            # Per-posture records.
            for posture, d in per_posture_metrics.items():
                per_posture_records.append(
                    {
                        "baseline": bl_name,
                        "posture": posture,
                        "n_samples": len(d["fixed_iou"]),
                        "mean_fixed_iou": float(np.mean(d["fixed_iou"])),
                        "mean_fixed_dice": float(np.mean(d["fixed_dice"])),
                        "mean_pixel_accuracy": float(np.mean(d["pixel_accuracy"])),
                    }
                )

        # Worst-subject identification (per baseline).
        worst_subject_records: list[dict[str, Any]] = []
        for subj_record in per_subject_records:
            worst_subject_records.append(subj_record)
        # Sort and write the worst-3 per baseline to the metrics summary.
        worst_summary: dict[str, list[dict[str, Any]]] = {}
        for bl_name in {r["baseline"] for r in worst_subject_records}:
            subjects_for_bl = [r for r in worst_subject_records if r["baseline"] == bl_name]
            subjects_for_bl.sort(key=lambda r: r["mean_fixed_iou"])
            worst_summary[bl_name] = subjects_for_bl[:3]

        # Write all outputs.
        _write_csv(
            output_dir / "metrics_by_baseline.csv",
            per_baseline_records,
            fieldnames=(
                "baseline", "n_samples_evaluated", "n_samples_failed",
                "fixed_iou", "fixed_dice", "pixel_accuracy",
                "n_classes_present_in_pred", "n_classes_present_in_gt",
                "runtime_seconds",
            ),
        )
        _write_csv(
            output_dir / "metrics_by_region.csv",
            per_region_records,
            fieldnames=(
                "baseline", "region_id", "region_name",
                "iou", "dice", "precision", "recall",
                "pred_count", "gt_count",
                "is_present_in_pred", "is_present_in_gt",
                "centroid_error_px",
            ),
        )
        _write_csv(
            output_dir / "metrics_by_subject.csv",
            per_subject_records,
            fieldnames=(
                "baseline", "subject_id", "ml_split", "n_samples",
                "mean_fixed_iou", "mean_fixed_dice", "mean_pixel_accuracy",
            ),
        )
        _write_csv(
            output_dir / "metrics_by_posture.csv",
            per_posture_records,
            fieldnames=(
                "baseline", "posture", "n_samples",
                "mean_fixed_iou", "mean_fixed_dice", "mean_pixel_accuracy",
            ),
        )
        _write_csv(
            output_dir / "predictions_manifest.csv",
            predictions_manifest,
            fieldnames=(
                "sample_id", "ml_split", "subject_id", "posture", "baseline",
                "pred_label_path", "pred_label_sha256",
                "pred_class_pixel_counts", "pred_class_present",
                "failure_reason", "runtime_ms",
            ),
        )
        _write_csv(
            output_dir / "failure_cases.csv",
            failure_cases,
            fieldnames=(
                "sample_id", "ml_split", "subject_id", "posture", "baseline",
                "reason", "detail",
            ),
        )
        _write_json(
            output_dir / "failure_reason_counts.json",
            {
                "total": int(sum(failure_counts.values())),
                "by_reason": {r: int(failure_counts.get(r, 0)) for r in FAILURE_REASONS},
                "extra": {k: int(v) for k, v in failure_counts.items() if k not in FAILURE_REASONS},
            },
        )
        _write_json(
            output_dir / "metrics_summary.json",
            {
                "task_id": TASK_ID,
                "baseline_version": BASELINE_VERSION,
                "config_version": config["config_version"],
                "freeze_version": config["freeze_version"],
                "raw_semantics": "raw_pmarray_response",
                "fit_split": "train",
                "annotation_provenance": EXPECTED_PROVENANCE,
                "source_review_status": EXPECTED_REVIEW_STATUS,
                "per_baseline": per_baseline_records,
                "worst_subject_per_baseline": worst_summary,
                "n_train_samples": len(train_rows),
                "n_val_samples": len(val_rows),
                "n_test_samples": 0,  # not evaluated
                "load_time_seconds": t_load,
                "wall_clock_seconds": time.perf_counter() - t_start,
                "finished_at_utc": _now_iso(),
            },
        )
        _write_json(
            output_dir / "runtime.json",
            {
                "task_id": TASK_ID,
                "started_at_utc": status["started_at_utc"],
                "finished_at_utc": _now_iso(),
                "wall_clock_seconds": time.perf_counter() - t_start,
                "n_train_samples": len(train_rows),
                "n_val_samples": len(val_rows),
                "n_test_samples": 0,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
        )

        status["status"] = "DONE"
        status["finished_at_utc"] = _now_iso()
        status["wall_clock_seconds"] = time.perf_counter() - t_start
        _write_json(output_dir / "status.json", status)
        _write_json(
            output_dir / "DONE.json",
            {
                "task_id": TASK_ID,
                "status": "DONE",
                "finished_at_utc": _now_iso(),
                "wall_clock_seconds": time.perf_counter() - t_start,
            },
        )
        return 0

    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        (output_dir / "FAILED.json").write_text(
            json.dumps(
                {
                    "task_id": TASK_ID,
                    "status": "FAILED",
                    "error": str(exc),
                    "traceback": tb,
                    "finished_at_utc": _now_iso(),
                    "wall_clock_seconds": time.perf_counter() - t_start,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        status["status"] = "FAILED"
        status["error"] = str(exc)
        status["finished_at_utc"] = _now_iso()
        status["wall_clock_seconds"] = time.perf_counter() - t_start
        _write_json(output_dir / "status.json", status)
        return 1


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the JSON config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for run artefacts (will be created).",
    )
    parser.add_argument(
        "--b01-freeze-dir",
        type=Path,
        default=None,
        help="Override the b01_freeze_dir in the config (must be a real path).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override the dataset_root in the config (must be a real path).",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 1
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(cfg)
    if args.b01_freeze_dir is not None:
        cfg["b01_freeze_dir"] = str(Path(args.b01_freeze_dir).resolve())
    if args.dataset_root is not None:
        cfg["dataset_root"] = str(Path(args.dataset_root).resolve())
    return run(cfg, Path(args.output_dir).resolve())


if __name__ == "__main__":
    sys.exit(main())
