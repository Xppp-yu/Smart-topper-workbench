"""Record-level, streaming quality checks for labeled PoPu Tactilus data.

The gate is deliberately conservative: malformed/unreadable source records are
``REJECT``; statistically unusual but structurally valid records are ``WARN``
for human review; records without a fixed posture label are ``EXCLUDED`` from
this stage, not silently mislabeled or discarded.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from topper_perception.io.popu import POPU_POSTURES, load_tactilus_record


QUALITY_COLUMNS = (
    "sample_id",
    "source_file",
    "subject_id",
    "posture",
    "variation",
    "p1_status",
    "frame_count",
    "representative_frame_index",
    "median_total_signal",
    "minimum_total_signal",
    "maximum_total_signal",
    "median_active_cells",
    "minimum_active_cells",
    "maximum_active_cells",
    "median_peak_value",
    "maximum_peak_value",
    "temporal_total_cv",
    "quality_status",
    "quality_reasons",
    "maximum_robust_z",
)

METRICS_FOR_REFERENCE = (
    "median_total_signal",
    "median_active_cells",
    "temporal_total_cv",
)


@dataclass(frozen=True, slots=True)
class RecordMetrics:
    """Compact per-record metrics; source pressure frames are not retained."""

    values: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {column: self.values.get(column, "") for column in QUALITY_COLUMNS}


def _number(value: object) -> float:
    return float(value) if value not in ("", None) else float("nan")


def compute_record_metrics(inventory_row: dict[str, str]) -> RecordMetrics:
    """Read one eligible JSON record and calculate temporal/numeric summaries."""
    base: dict[str, object] = {
        "sample_id": inventory_row.get("sample_id", ""),
        "source_file": inventory_row.get("source_file", ""),
        "subject_id": inventory_row.get("subject_id", ""),
        "posture": inventory_row.get("posture", ""),
        "variation": inventory_row.get("variation", ""),
        "p1_status": inventory_row.get("status", ""),
        **{column: "" for column in QUALITY_COLUMNS[6:]},
    }

    posture = str(base["posture"])
    if base["p1_status"] != "OK" or posture not in POPU_POSTURES:
        base.update(
            {
                "quality_status": "EXCLUDED",
                "quality_reasons": "missing_fixed_posture_label_or_p1_warning",
            }
        )
        return RecordMetrics(base)

    try:
        frames = load_tactilus_record(Path(str(base["source_file"])))
    except (OSError, ValueError) as exc:
        base.update(
            {
                "quality_status": "REJECT",
                "quality_reasons": f"unreadable_or_invalid_source:{type(exc).__name__}",
            }
        )
        return RecordMetrics(base)

    totals = np.asarray([float(np.sum(frame.values)) for frame in frames])
    active_cells = np.asarray([int(np.count_nonzero(frame.values)) for frame in frames])
    peaks = np.asarray([float(np.max(frame.values)) for frame in frames])
    median_total = float(np.median(totals))
    representative_frame_index = int(np.argmin(np.abs(totals - median_total)))
    mean_total = float(np.mean(totals))
    temporal_total_cv = float(np.std(totals) / mean_total) if mean_total > 0 else 0.0

    base.update(
        {
            "frame_count": len(frames),
            "representative_frame_index": representative_frame_index,
            "median_total_signal": median_total,
            "minimum_total_signal": float(np.min(totals)),
            "maximum_total_signal": float(np.max(totals)),
            "median_active_cells": float(np.median(active_cells)),
            "minimum_active_cells": int(np.min(active_cells)),
            "maximum_active_cells": int(np.max(active_cells)),
            "median_peak_value": float(np.median(peaks)),
            "maximum_peak_value": float(np.max(peaks)),
            "temporal_total_cv": temporal_total_cv,
        }
    )
    return RecordMetrics(base)


def build_reference_profiles(rows: Iterable[RecordMetrics]) -> dict[str, dict[str, dict[str, float]]]:
    """Build per-label robust references without mixing posture distributions."""
    values_by_posture: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in rows:
        row = record.as_dict()
        if row["quality_status"] in ("EXCLUDED", "REJECT"):
            continue
        posture = str(row["posture"])
        for metric in METRICS_FOR_REFERENCE:
            value = _number(row[metric])
            if math.isfinite(value):
                values_by_posture[posture][metric].append(value)

    profiles: dict[str, dict[str, dict[str, float]]] = {}
    for posture, metrics in values_by_posture.items():
        profiles[posture] = {}
        for metric, values in metrics.items():
            array = np.asarray(values, dtype=float)
            median = float(np.median(array))
            mad = float(np.median(np.abs(array - median)))
            profiles[posture][metric] = {
                "count": float(array.size),
                "median": median,
                "mad": mad,
                "minimum": float(np.min(array)),
                "maximum": float(np.max(array)),
            }
    return profiles


def _robust_z(value: float, profile: dict[str, float]) -> float:
    mad = profile["mad"]
    if not math.isfinite(value) or mad <= 0:
        return 0.0
    return 0.6745 * abs(value - profile["median"]) / mad


def assess_quality(
    records: Sequence[RecordMetrics],
    *,
    robust_z_threshold: float,
) -> tuple[list[RecordMetrics], dict[str, dict[str, dict[str, float]]]]:
    """Apply provisional per-posture outlier rules and retain all decisions."""
    profiles = build_reference_profiles(records)
    assessed: list[RecordMetrics] = []
    for record in records:
        row = record.as_dict()
        existing_status = str(row["quality_status"])
        if existing_status in ("EXCLUDED", "REJECT"):
            assessed.append(RecordMetrics(row))
            continue

        posture = str(row["posture"])
        reasons: list[str] = []
        maximum_z = 0.0
        for metric in METRICS_FOR_REFERENCE:
            z = _robust_z(_number(row[metric]), profiles[posture][metric])
            maximum_z = max(maximum_z, z)
            if z > robust_z_threshold:
                reasons.append(f"{metric}_robust_z_gt_{robust_z_threshold:g}")
        row["quality_status"] = "WARN" if reasons else "ACCEPT"
        row["quality_reasons"] = ";".join(reasons)
        row["maximum_robust_z"] = maximum_z
        assessed.append(RecordMetrics(row))
    return assessed, profiles


def build_quality_summary(
    records: Sequence[RecordMetrics],
    *,
    profiles: dict[str, dict[str, dict[str, float]]],
    robust_z_threshold: float,
) -> dict[str, Any]:
    """Return report-ready counts and thresholds without presenting them as labels."""
    rows = [record.as_dict() for record in records]
    status_counts = Counter(str(row["quality_status"]) for row in rows)
    reason_counts = Counter(
        reason
        for row in rows
        for reason in str(row["quality_reasons"]).split(";")
        if reason
    )
    return {
        "dataset": "PoPu",
        "sensor_layer": "tactilus",
        "scope": "P1-OK fixed-posture records; P1 WARN records are retained as EXCLUDED",
        "records": len(rows),
        "quality_status_counts": dict(sorted(status_counts.items())),
        "quality_reason_counts": dict(sorted(reason_counts.items())),
        "provisional_rule": {
            "method": "per-posture median absolute deviation robust z-score",
            "metrics": list(METRICS_FOR_REFERENCE),
            "warn_when_robust_z_gt": robust_z_threshold,
            "note": "WARN marks candidates for visual review; it is not a ground-truth defect label.",
        },
        "reference_profiles": profiles,
    }
