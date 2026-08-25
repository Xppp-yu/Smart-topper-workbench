"""Run the SLP8 B02 v0.1 non-learning region baseline.

CLI for the SLP8 pressure-only non-learning region baseline (TASK-SLP-
B02-NON-LEARNING-REGION-BASELINE-v0.1).  The script is CPU-only,
deterministic, and never reads TEST label/onehot.

Usage
-----

::

    uv run python scripts/run_slp8_non_learning_region_baseline.py \\
        --config configs/experiments/slp8_non_learning_region_baseline_v0.1.json \\
        --output-dir outputs/experiments/<EXP-ID> \\
        --b01-freeze-dir <B01_FREEZE_DIR> \\
        --dataset-root <SLP8_DATASET_ROOT>

The CLI flags for ``--b01-freeze-dir`` and ``--dataset-root`` are
required and are recorded into the run only as logical names; the
absolute paths are NOT written into any committed artefact.

Inputs
------
* ``--config`` — path to a JSON config (see
  ``configs/experiments/slp8_non_learning_region_baseline_v0.1.json``).
* ``--output-dir`` — directory where the run artefacts will be written.
  Must be empty (the runner refuses to overwrite an existing non-empty
  output directory, especially one containing ``DONE.json`` or
  ``FAILED.json``).
* ``--b01-freeze-dir`` — the B01 freeze directory.
* ``--dataset-root`` — the SLP8 dataset root.

Outputs
-------
The runner writes the following artefacts to ``--output-dir``:

* ``status.json`` — overall status (DONE / FAILED)
* ``resolved_config.json`` — config as actually used (no absolute paths)
* ``input_manifest_hashes.json`` — SHA-256 of the B01 freeze manifest
  and the source manifest
* ``metrics_summary.json`` — overall metric summary (TRAIN + VAL
  per-baseline records; VAL-only headline)
* ``metrics_by_baseline.csv`` — per-(baseline, ml_split) metrics
* ``metrics_by_region.csv`` — per-(baseline, ml_split, region) metrics
* ``metrics_by_subject.csv`` — per-(baseline, ml_split, subject) metrics
* ``metrics_by_posture.csv`` — per-(baseline, ml_split, posture) metrics
* ``predictions_manifest.csv`` — per-prediction summary
* ``failure_cases.csv`` — list of contract-failure cases
* ``diagnostic_counts.json`` — fallback / diagnostic counts
* ``failure_reason_counts.json`` — counts of contract failures
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
    load_b01_freeze_tables,
    sha256_hex,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_ID: str = "TASK-SLP-B02-NON-LEARNING-REGION-BASELINE-v0.1"

#: Sentinel value used in ``resolved_config.json`` and other artefacts
#: to indicate that an absolute path is intentionally omitted.  The
#: corresponding logical key (``b01_freeze_dir`` / ``dataset_root``)
#: is replaced by this string.
REDACTED_LOCAL_PATH: str = "REDACTED_LOCAL_PATH"

#: Failure reason taxonomy for contract failures (stable strings; used
#: in CSV and counts).  These are *contract* failures, distinct from
#: the *fallback* diagnostics (which are normal outcomes on some samples
#: and are counted in ``diagnostic_counts.json``).
FAILURE_REASONS: tuple[str, ...] = (
    "shape_mismatch",
    "non_finite_pressure",
    "label_out_of_range",
    "file_not_found",
    "wrong_provenance",
    "wrong_review_status",
    "wrong_subject_split",
    "internal_exception",
)

#: Diagnostic (non-failure) counters.  These are normal outcomes on
#: some samples and are not contract failures.
DIAGNOSTIC_REASONS: tuple[str, ...] = (
    "no_contact",
    "degenerate_pca",
    "zero_axis_length",
    "all_background_fallback",
    "all_background_after_smoothing",
    "smoothed",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OutputDirCollisionError(RuntimeError):
    """Raised when the output directory already contains run artefacts."""


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


REQUIRED_CONFIG_FIELDS: tuple[str, ...] = (
    "config_version",
    "task_id",
    "freeze_version",
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


def _region_centroid_xy(label: np.ndarray, class_id: int) -> tuple[tuple[float, float] | None, int]:
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


def _classify_failure_reason(exc: BaseException) -> str:
    """Map an exception to a stable failure-reason string."""
    msg = str(exc).lower()
    if "non-finite" in msg or "non_finite" in msg:
        return "non_finite_pressure"
    if "shape" in msg:
        return "shape_mismatch"
    if "label_map values" in msg or "label range" in msg:
        return "label_out_of_range"
    if "provenance" in msg:
        return "wrong_provenance"
    if "review" in msg:
        return "wrong_review_status"
    if "subject" in msg and "split" in msg:
        return "wrong_subject_split"
    return "internal_exception"


def _classify_diagnostic(info: dict[str, Any]) -> str | None:
    """Map a baseline's diagnostic info dict to a stable diagnostic
    string, or None if the baseline ran cleanly with no fallback."""
    if not info:
        return None
    fallback = info.get("fallback", "none")
    if fallback == "none":
        return None
    if fallback in {"all_background", "all_background_after_smoothing"}:
        return fallback
    if fallback == "zero_axis_length":
        return "zero_axis_length"
    return str(fallback)


# ---------------------------------------------------------------------------
# Output directory safety
# ---------------------------------------------------------------------------


def _check_output_dir_safety(output_dir: Path) -> None:
    """Refuse to run if the output directory already contains artefacts.

    Raises
    ------
    OutputDirCollisionError
        If the directory already exists and contains files (especially
        DONE.json or FAILED.json) that would be overwritten.
    """
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
                f"overwrite.  Choose a fresh --output-dir or remove the "
                f"existing one manually.  ({output_dir})"
            )
    # Any other existing files / sub-directories also block the run.
    contents = list(output_dir.iterdir())
    if contents:
        # Allow the .gitkeep file in fresh sub-directories of the
        # outputs tree, but anything else is a collision.
        non_keep = [p for p in contents if p.name != ".gitkeep"]
        if non_keep:
            raise OutputDirCollisionError(
                f"output directory is not empty ({len(non_keep)} entries); "
                f"refusing to overwrite.  ({output_dir})"
            )


# ---------------------------------------------------------------------------
# Per-baseline evaluation
# ---------------------------------------------------------------------------


@dataclass
class _SampleRecord:
    """Accumulator for per-sample metrics within one (baseline, ml_split)."""

    pred: np.ndarray
    gt: np.ndarray
    info: dict[str, Any]
    subject_id: str
    posture: str
    sample_id: str
    failure_reason: str | None
    diagnostic: str | None
    runtime_ms: float


def _predict_with_failure_capture(
    *,
    baseline_obj: Any,
    pressure: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any] | None, str | None, float]:
    """Call ``baseline_obj.predict_with_info(pressure)``; on any
    contract error, return a zero label map and a failure reason."""
    t0 = time.perf_counter()
    try:
        pred, info = baseline_obj.predict_with_info(pressure)
    except Exception as exc:  # noqa: BLE001 - capture all contract errors
        label = np.zeros(PRESSURE_SHAPE, dtype=np.uint8)
        return label, None, _classify_failure_reason(exc), (time.perf_counter() - t0) * 1000.0
    return pred, info, None, (time.perf_counter() - t0) * 1000.0


def _aggregate_metrics(
    records: list[_SampleRecord],
    baseline_name: str,
    ml_split: str,
) -> dict[str, Any]:
    """Aggregate per-sample records into one per-(baseline, ml_split)
    metric record.  Returns a dict compatible with the CSV / summary
    schema."""
    good_records = [r for r in records if r.failure_reason is None]
    failed_records = [r for r in records if r.failure_reason is not None]
    n_eval = len(good_records)
    n_failed = len(failed_records)
    if n_eval > 0:
        all_preds = [r.pred for r in good_records]
        all_gts = [r.gt for r in good_records]
        m = compute_fixed_class_macro_metrics(
            all_gts, all_preds,
            class_ids=REGION_IDS, n_classes=9,
        )
        # Centroid error per region (only when both GT and pred have ≥1 pixel).
        centroid_errors: dict[str, float | None] = {}
        for cid in REGION_IDS:
            errs: list[float] = []
            for r in good_records:
                gt_c, gt_n = _region_centroid_xy(r.gt, cid)
                pr_c, pr_n = _region_centroid_xy(r.pred, cid)
                if gt_n == 0 or pr_n == 0:
                    continue
                dx = float(gt_c[0] - pr_c[0])  # type: ignore[index]
                dy = float(gt_c[1] - pr_c[1])  # type: ignore[index]
                errs.append(float(np.sqrt(dx * dx + dy * dy)))
            centroid_errors[str(cid)] = float(np.mean(errs)) if errs else None
        coverage_per_region = {
            str(cid): int(sum(1 for r in good_records if (r.gt == cid).any()))
            for cid in REGION_IDS
        }
        pred_coverage_per_region = {
            str(cid): int(sum(1 for r in good_records if (r.pred == cid).any()))
            for cid in REGION_IDS
        }
        record = {
            "baseline": baseline_name,
            "ml_split": ml_split,
            "n_samples_evaluated": n_eval,
            "n_samples_failed": n_failed,
            "fixed_iou": m.fixed_iou,
            "fixed_dice": m.fixed_dice,
            "pixel_accuracy": m.pixel_accuracy,
            "n_classes_present_in_pred": m.n_classes_present_in_pred,
            "n_classes_present_in_gt": m.n_classes_present_in_gt,
            "per_class_iou": {str(k): v for k, v in m.per_class_iou.items()},
            "per_class_dice": {str(k): v for k, v in m.per_class_dice.items()},
            "per_class_precision": {str(k): v for k, v in m.per_class_precision.items()},
            "per_class_recall": {str(k): v for k, v in m.per_class_recall.items()},
            "per_class_tp": {str(k): v for k, v in m.per_class_tp.items()},
            "per_class_fp": {str(k): v for k, v in m.per_class_fp.items()},
            "per_class_fn": {str(k): v for k, v in m.per_class_fn.items()},
            "per_class_pred_count": {str(k): v for k, v in m.per_class_pred_count.items()},
            "per_class_gt_count": {str(k): v for k, v in m.per_class_gt_count.items()},
            "per_class_present_in_gt": {str(k): v for k, v in m.per_class_present_in_gt.items()},
            "per_class_present_in_pred": {str(k): v for k, v in m.per_class_present_in_pred.items()},
            "centroid_error_px": centroid_errors,
            "region_coverage_gt": coverage_per_region,
            "region_coverage_pred": pred_coverage_per_region,
        }
    else:
        record = {
            "baseline": baseline_name,
            "ml_split": ml_split,
            "n_samples_evaluated": 0,
            "n_samples_failed": n_failed,
            "fixed_iou": 0.0,
            "fixed_dice": 0.0,
            "pixel_accuracy": 0.0,
            "n_classes_present_in_pred": 0,
            "n_classes_present_in_gt": 0,
            "per_class_iou": {str(k): 0.0 for k in REGION_IDS},
            "per_class_dice": {str(k): 0.0 for k in REGION_IDS},
            "per_class_precision": {str(k): 0.0 for k in REGION_IDS},
            "per_class_recall": {str(k): 0.0 for k in REGION_IDS},
            "per_class_tp": {str(k): 0 for k in REGION_IDS},
            "per_class_fp": {str(k): 0 for k in REGION_IDS},
            "per_class_fn": {str(k): 0 for k in REGION_IDS},
            "per_class_pred_count": {str(k): 0 for k in REGION_IDS},
            "per_class_gt_count": {str(k): 0 for k in REGION_IDS},
            "per_class_present_in_gt": {str(k): False for k in REGION_IDS},
            "per_class_present_in_pred": {str(k): False for k in REGION_IDS},
            "centroid_error_px": {str(k): None for k in REGION_IDS},
            "region_coverage_gt": {str(k): 0 for k in REGION_IDS},
            "region_coverage_pred": {str(k): 0 for k in REGION_IDS},
        }
    return record


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(config: dict[str, Any], output_dir: Path) -> int:
    """Execute the run; return 0 on success, 1 on failure."""
    output_dir = Path(output_dir).resolve()
    _check_output_dir_safety(output_dir)
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
    diagnostic_counts: Counter[str] = Counter()
    failure_cases: list[dict[str, Any]] = []
    predictions_manifest: list[dict[str, Any]] = []
    per_baseline_records: list[dict[str, Any]] = []
    per_region_records: list[dict[str, Any]] = []
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
        # Source manifest SHAs (file bytes) and content-addressed SHAs
        # (from the freeze manifest core).
        source_manifest_path = b01_dir / "train_manifest.csv"
        source_manifest_sha = (
            sha256_hex(source_manifest_path.read_bytes())
            if source_manifest_path.is_file()
            else None
        )
        val_manifest_sha = (
            sha256_hex((b01_dir / "val_manifest.csv").read_bytes())
            if (b01_dir / "val_manifest.csv").is_file()
            else None
        )

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

        # Resolved config — NEVER include absolute paths.  Replace
        # dataset_root and b01_freeze_dir with the REDACTED_LOCAL_PATH
        # sentinel so the committed artefacts do not leak any local
        # machine layout.
        resolved_config = {
            k: v
            for k, v in config.items()
            if k not in {"b01_freeze_dir", "dataset_root"}
        }
        resolved_config["b01_freeze_dir"] = REDACTED_LOCAL_PATH
        resolved_config["dataset_root"] = REDACTED_LOCAL_PATH
        resolved_config["b01_freeze_dir_freeze_manifest_sha256"] = fm_sha
        resolved_config["resolved_at_utc"] = _now_iso()
        # ``absolute_paths_recorded`` is True iff every potentially
        # path-like field is either REDACTED or non-path.  The check
        # is exhaustive: it fails fast if any absolute-path string
        # sneaks through.
        _check_resolved_config_no_absolute_paths(resolved_config)
        resolved_config["absolute_paths_recorded"] = False
        _write_json(output_dir / "resolved_config.json", resolved_config)

        # Load B01 freeze tables (TRAIN + VAL only; default load_test=False).
        tables = load_b01_freeze_tables(b01_dir, load_test=False)
        train_rows = list(tables.train_rows)
        val_rows = list(tables.val_rows)
        all_dev_rows = train_rows + val_rows

        # We don't have the dataset root as an absolute path; it must
        # be passed via the config.  We open the SLP8 dataset via the
        # A09R adapter, which reads manifest/val_manifest.csv.
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

        from topper_perception.io.slp_8region_pressure_dataset import (
            Slp8RegionDatasetAdapter,
        )

        adapter = Slp8RegionDatasetAdapter(dataset_root, validate_on_load=True)

        # Load TRAIN pressure + label once; the template fit needs the
        # entire TRAIN set.
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

        # Fit the baselines on TRAIN only.
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

        # Per-baseline evaluation: TRAIN + VAL, separately.
        rows_by_split: dict[str, list[Any]] = {
            "train": list(train_rows),
            "val": list(val_rows),
        }

        for bl_name, bl_obj in baselines_state.items():
            t_bl0 = time.perf_counter()
            per_split_records: dict[str, list[_SampleRecord]] = {
                "train": [],
                "val": [],
            }
            for ml_split, split_rows in rows_by_split.items():
                for r in split_rows:
                    try:
                        loaded = adapter.load_sample(r.sample_id)
                    except Exception as exc:  # noqa: BLE001
                        failure_counts["file_not_found"] += 1
                        failure_cases.append(
                            {
                                "sample_id": r.sample_id,
                                "ml_split": ml_split,
                                "subject_id": r.subject_id,
                                "posture": r.posture,
                                "baseline": bl_name,
                                "reason": "file_not_found",
                                "detail": str(exc),
                            }
                        )
                        continue
                    pred, info, failure_reason, rt_ms = _predict_with_failure_capture(
                        baseline_obj=bl_obj,
                        pressure=loaded.pressure,
                    )
                    diagnostic = _classify_diagnostic(info or {})
                    if diagnostic is not None:
                        diagnostic_counts[diagnostic] += 1
                    if info and info.get("smoothed"):
                        diagnostic_counts["smoothed"] += 1
                    if failure_reason is not None:
                        failure_counts[failure_reason] += 1
                        failure_cases.append(
                            {
                                "sample_id": r.sample_id,
                                "ml_split": ml_split,
                                "subject_id": r.subject_id,
                                "posture": r.posture,
                                "baseline": bl_name,
                                "reason": failure_reason,
                                "detail": "see exception log",
                            }
                        )
                    # Persist prediction (gzip-compressed npz).
                    pred_label_path = (
                        predictions_dir
                        / f"{_safe_filename(r.sample_id)}__{_safe_filename(bl_name)}__split_{ml_split}.npz"
                    )
                    np.savez_compressed(
                        pred_label_path,
                        pred=pred,
                        sample_id=np.array(r.sample_id),
                        baseline=np.array(bl_name),
                        ml_split=np.array(ml_split),
                        subject_id=np.array(r.subject_id),
                        posture=np.array(r.posture),
                        diagnostic=np.array(diagnostic or "none"),
                    )
                    pred_sha = sha256_hex(pred_label_path.read_bytes())

                    # Manifest
                    predictions_manifest.append(
                        {
                            "sample_id": r.sample_id,
                            "ml_split": ml_split,
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
                            "diagnostic": diagnostic,
                            "failure_reason": failure_reason,
                            "runtime_ms": rt_ms,
                        }
                    )

                    if failure_reason is None:
                        per_split_records[ml_split].append(
                            _SampleRecord(
                                pred=pred,
                                gt=loaded.region_label,
                                info=info or {},
                                subject_id=r.subject_id,
                                posture=r.posture,
                                sample_id=r.sample_id,
                                failure_reason=None,
                                diagnostic=diagnostic,
                                runtime_ms=rt_ms,
                            )
                        )

            # Aggregate per (baseline, ml_split).
            for ml_split, records in per_split_records.items():
                agg = _aggregate_metrics(records, bl_name, ml_split)
                agg["runtime_seconds"] = float(time.perf_counter() - t_bl0)
                per_baseline_records.append(agg)

                # Per-region CSV records.
                for cid in REGION_IDS:
                    per_region_records.append(
                        {
                            "baseline": bl_name,
                            "ml_split": ml_split,
                            "region_id": int(cid),
                            "region_name": REGION_ID_TO_NAME[int(cid)],
                            "iou": agg["per_class_iou"][str(cid)],
                            "dice": agg["per_class_dice"][str(cid)],
                            "precision": agg["per_class_precision"][str(cid)],
                            "recall": agg["per_class_recall"][str(cid)],
                            "tp": agg["per_class_tp"][str(cid)],
                            "fp": agg["per_class_fp"][str(cid)],
                            "fn": agg["per_class_fn"][str(cid)],
                            "pred_count": agg["per_class_pred_count"][str(cid)],
                            "gt_count": agg["per_class_gt_count"][str(cid)],
                            "is_present_in_pred": agg["per_class_present_in_pred"][str(cid)],
                            "is_present_in_gt": agg["per_class_present_in_gt"][str(cid)],
                            "centroid_error_px": agg["centroid_error_px"][str(cid)],
                        }
                    )

                # Per-subject records.
                per_subject: dict[str, list[_SampleRecord]] = {}
                for r in records:
                    per_subject.setdefault(r.subject_id, []).append(r)
                for subj, recs in per_subject.items():
                    ifs = [
                        compute_fixed_class_macro_metrics(
                            [r.gt], [r.pred],
                            class_ids=REGION_IDS, n_classes=9,
                        )
                        for r in recs
                    ]
                    per_subject_records.append(
                        {
                            "baseline": bl_name,
                            "ml_split": ml_split,
                            "subject_id": subj,
                            "n_samples": len(recs),
                            "mean_fixed_iou": float(np.mean([m.fixed_iou for m in ifs])),
                            "mean_fixed_dice": float(np.mean([m.fixed_dice for m in ifs])),
                            "mean_pixel_accuracy": float(np.mean([m.pixel_accuracy for m in ifs])),
                        }
                    )

                # Per-posture records.
                per_posture: dict[str, list[_SampleRecord]] = {}
                for r in records:
                    per_posture.setdefault(r.posture, []).append(r)
                for posture, recs in per_posture.items():
                    ifs = [
                        compute_fixed_class_macro_metrics(
                            [r.gt], [r.pred],
                            class_ids=REGION_IDS, n_classes=9,
                        )
                        for r in recs
                    ]
                    per_posture_records.append(
                        {
                            "baseline": bl_name,
                            "ml_split": ml_split,
                            "posture": posture,
                            "n_samples": len(recs),
                            "mean_fixed_iou": float(np.mean([m.fixed_iou for m in ifs])),
                            "mean_fixed_dice": float(np.mean([m.fixed_dice for m in ifs])),
                            "mean_pixel_accuracy": float(np.mean([m.pixel_accuracy for m in ifs])),
                        }
                    )

        # Worst-subject: per baseline, per ml_split, top-3 (ascending by mean_fixed_iou).
        worst_summary: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for ml_split in ("val", "train"):
            for bl_name in {r["baseline"] for r in per_subject_records if r["ml_split"] == ml_split}:
                subjects_for_bl = [
                    r for r in per_subject_records
                    if r["baseline"] == bl_name and r["ml_split"] == ml_split
                ]
                subjects_for_bl.sort(key=lambda r: r["mean_fixed_iou"])
                worst_summary.setdefault(ml_split, {})[bl_name] = subjects_for_bl[:3]

        # Headline: VAL-only per-baseline records.
        val_baseline_records = [r for r in per_baseline_records if r["ml_split"] == "val"]
        train_baseline_records = [r for r in per_baseline_records if r["ml_split"] == "train"]

        # Write all outputs.
        _write_csv(
            output_dir / "metrics_by_baseline.csv",
            per_baseline_records,
            fieldnames=(
                "baseline", "ml_split", "n_samples_evaluated", "n_samples_failed",
                "fixed_iou", "fixed_dice", "pixel_accuracy",
                "n_classes_present_in_pred", "n_classes_present_in_gt",
                "runtime_seconds",
            ),
        )
        _write_csv(
            output_dir / "metrics_by_region.csv",
            per_region_records,
            fieldnames=(
                "baseline", "ml_split", "region_id", "region_name",
                "iou", "dice", "precision", "recall",
                "tp", "fp", "fn",
                "pred_count", "gt_count",
                "is_present_in_pred", "is_present_in_gt",
                "centroid_error_px",
            ),
        )
        _write_csv(
            output_dir / "metrics_by_subject.csv",
            per_subject_records,
            fieldnames=(
                "baseline", "ml_split", "subject_id", "n_samples",
                "mean_fixed_iou", "mean_fixed_dice", "mean_pixel_accuracy",
            ),
        )
        _write_csv(
            output_dir / "metrics_by_posture.csv",
            per_posture_records,
            fieldnames=(
                "baseline", "ml_split", "posture", "n_samples",
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
                "diagnostic", "failure_reason", "runtime_ms",
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
            output_dir / "diagnostic_counts.json",
            {
                "total": int(sum(diagnostic_counts.values())),
                "by_reason": {r: int(diagnostic_counts.get(r, 0)) for r in DIAGNOSTIC_REASONS},
                "extra": {k: int(v) for k, v in diagnostic_counts.items() if k not in DIAGNOSTIC_REASONS},
                "notes": (
                    "diagnostic counts are NOT contract failures.  They "
                    "are normal outcomes on some samples (e.g. an SLP8 "
                    "frame may have only one or two contact pixels, "
                    "which is expected for some postures and is not a "
                    "test failure)."
                ),
            },
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
                "headline_split": "val",
                "headline_per_baseline_val": val_baseline_records,
                "fit_diagnostic_per_baseline_train": train_baseline_records,
                "worst_subject_val_per_baseline": worst_summary.get("val", {}),
                "worst_subject_train_per_baseline": worst_summary.get("train", {}),
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

        # A contract failure (n_failed > 0) does NOT block DONE; the
        # metrics on the unaffected samples are still valid.  But any
        # unexpected exception in the runner itself (caught below)
        # will write FAILED.json instead.
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

    except OutputDirCollisionError as exc:
        # The output-dir collision is a user-facing error, not a run
        # failure.  We write a small FAILED.json and return 1 so the
        # caller knows.
        (output_dir / "FAILED.json").write_text(
            json.dumps(
                {
                    "task_id": TASK_ID,
                    "status": "FAILED",
                    "error": str(exc),
                    "error_kind": "output_dir_collision",
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
        status["error_kind"] = "output_dir_collision"
        status["finished_at_utc"] = _now_iso()
        status["wall_clock_seconds"] = time.perf_counter() - t_start
        _write_json(output_dir / "status.json", status)
        return 1
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


def _check_resolved_config_no_absolute_paths(payload: Any) -> None:
    """Walk the resolved-config dict and raise if any string field
    looks like an absolute Windows / POSIX / UNC path.  Skips the
    ``REDACTED_LOCAL_PATH`` sentinel and any string starting with
    ``"sha256:"`` / ``"metadata:"``."""
    def _walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, f"{path}[{i}]")
        elif isinstance(obj, str):
            if obj == REDACTED_LOCAL_PATH:
                return
            s = obj
            if s.startswith("/") or s.startswith("\\"):
                raise ValueError(
                    f"resolved_config: absolute path not allowed at {path}: {s!r}"
                )
            if len(s) >= 2 and s[1] == ":":
                raise ValueError(
                    f"resolved_config: Windows absolute path not allowed at {path}: {s!r}"
                )
            if s.startswith("\\\\"):
                raise ValueError(
                    f"resolved_config: UNC path not allowed at {path}: {s!r}"
                )
            if ".." in Path(s).parts:
                raise ValueError(
                    f"resolved_config: '..' segment not allowed at {path}: {s!r}"
                )

    _walk(payload, "resolved_config")


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
        help="Output directory for run artefacts.  Must not already contain run artefacts.",
    )
    parser.add_argument(
        "--b01-freeze-dir",
        type=Path,
        required=True,
        help="The B01 freeze directory.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="The SLP8 dataset root.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 1
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(cfg)
    cfg["b01_freeze_dir"] = str(Path(args.b01_freeze_dir).resolve())
    cfg["dataset_root"] = str(Path(args.dataset_root).resolve())
    return run(cfg, Path(args.output_dir).resolve())


if __name__ == "__main__":
    sys.exit(main())
