"""Deterministic, label-free per-snapshot feature extraction for PoPu Tactilus.

Every feature derives ONLY from the raw pressure matrix, the frozen P3.1
``largest_component`` contact mask, and a documented grid partition.  Subject
identity, posture, variation and source filename are kept strictly outside the
feature columns so the table can feed a subject-isolated posture baseline
without label leakage.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage

from topper_perception.geometry.popu import describe_geometry
from topper_perception.io.popu import PopuTactilusFrame


DATASET_ID = "popu"
MASK_RULE_VERSION = "largest_component@frozen_v0.2"
FEATURE_SCHEMA_VERSION = "v0.1"
MATRIX_ORIENTATION = (
    "row-major origin=upper-left; rows=64 columns=27 as restored by "
    "topper_perception.io.popu.load_tactilus_record"
)

# Columns that identify a row or carry the label; never model inputs.
METADATA_COLUMNS = (
    "sample_id",
    "dataset_id",
    "source_relative_path",
    "subject_id",
    "posture",
    "variation",
    "snapshot_index",
    "snapshot_key",
    "quality_status",
    "cohort",
    "rows",
    "columns",
    "matrix_orientation",
    "mask_rule_version",
    "feature_schema_version",
    "feature_status",
    "feature_reason",
)

RAW_INTENSITY_FEATURES = (
    "intensity_sum",
    "intensity_mean",
    "intensity_std",
    "intensity_min",
    "intensity_max",
    "intensity_p25",
    "intensity_p50",
    "intensity_p75",
    "intensity_p90",
    "intensity_p95",
    "intensity_p99",
    "nonzero_cell_count",
    "nonzero_fraction",
    "positive_mean",
)

MASK_GEOMETRY_FEATURES = (
    "mask_threshold_raw",
    "mask_cell_count",
    "mask_fraction",
    "component_count",
    "bbox_row_min",
    "bbox_row_max",
    "bbox_column_min",
    "bbox_column_max",
    "bbox_height",
    "bbox_width",
    "bbox_area",
    "centroid_row_fraction",
    "centroid_column_fraction",
    "cop_row_fraction",
    "cop_column_fraction",
    "principal_axis_degrees",
    "principal_axis_anisotropy",
    "contact_signal_sum",
)

SHAPE_FEATURES = (
    "bbox_aspect_ratio",
    "mask_extent",
    "mask_compactness",
)


def grid_zone_names(row_bands: int, column_bands: int) -> tuple[str, ...]:
    """Return the ordered grid-statistic column names for one partition."""
    return tuple(
        f"zone_{statistic}_r{row}c{column}"
        for row in range(row_bands)
        for column in range(column_bands)
        for statistic in ("sum", "fraction", "peak")
    )


def feature_column_names(row_bands: int, column_bands: int) -> tuple[str, ...]:
    """Return the full ordered feature-column list, labels excluded by design."""
    return RAW_INTENSITY_FEATURES + MASK_GEOMETRY_FEATURES + SHAPE_FEATURES + grid_zone_names(
        row_bands, column_bands
    )


def _to_float_or_nan(value: object) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def _raw_intensity_features(values: np.ndarray) -> dict[str, float]:
    flat = values.ravel()
    positive = flat[flat > 0]
    percentiles = np.percentile(flat, [25, 50, 75, 90, 95, 99])
    return {
        "intensity_sum": float(flat.sum()),
        "intensity_mean": float(flat.mean()),
        "intensity_std": float(flat.std()),
        "intensity_min": float(flat.min()),
        "intensity_max": float(flat.max()),
        "intensity_p25": float(percentiles[0]),
        "intensity_p50": float(percentiles[1]),
        "intensity_p75": float(percentiles[2]),
        "intensity_p90": float(percentiles[3]),
        "intensity_p95": float(percentiles[4]),
        "intensity_p99": float(percentiles[5]),
        "nonzero_cell_count": float(int(np.count_nonzero(flat))),
        "nonzero_fraction": float(np.count_nonzero(flat) / flat.size),
        "positive_mean": float(positive.mean()) if positive.size else 0.0,
    }


def _grid_features(
    values: np.ndarray, *, row_bands: int, column_bands: int
) -> dict[str, float]:
    rows, columns = values.shape
    total = float(values.sum())
    row_edges = np.linspace(0, rows, row_bands + 1).astype(int)
    column_edges = np.linspace(0, columns, column_bands + 1).astype(int)
    features: dict[str, float] = {}
    for row in range(row_bands):
        for column in range(column_bands):
            zone = values[
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
            ]
            zone_sum = float(zone.sum())
            features[f"zone_sum_r{row}c{column}"] = zone_sum
            features[f"zone_fraction_r{row}c{column}"] = (
                zone_sum / total if total > 0 else 0.0
            )
            features[f"zone_peak_r{row}c{column}"] = (
                float(zone.max()) if zone.size else 0.0
            )
    return features


def _mask_perimeter(mask: np.ndarray) -> float:
    """Estimate the 4-connected perimeter as the boundary-ring cell count."""
    if not mask.any():
        return 0.0
    eroded = ndimage.binary_erosion(mask)
    return float(np.count_nonzero(mask != eroded))


def _shape_features(mask: np.ndarray, geometry: dict[str, object]) -> dict[str, float]:
    height = _to_float_or_nan(geometry.get("bbox_height", ""))
    width = _to_float_or_nan(geometry.get("bbox_width", ""))
    area = height * width if math.isfinite(height) and math.isfinite(width) else float("nan")
    cell_count = float(mask.sum())
    bbox_aspect_ratio = height / width if math.isfinite(width) and width > 0 else float("nan")
    mask_extent = cell_count / area if math.isfinite(area) and area > 0 else float("nan")
    perimeter = _mask_perimeter(mask)
    mask_compactness = (
        perimeter * perimeter / (4.0 * math.pi * cell_count)
        if perimeter > 0 and cell_count > 0
        else float("nan")
    )
    return {
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "mask_extent": mask_extent,
        "mask_compactness": mask_compactness,
    }


def extract_feature_vector(
    values: np.ndarray,
    *,
    strategy: str,
    positive_percentile: float,
    minimum_raw_threshold: float,
    minimum_component_cells: int,
    minimum_component_fraction_of_largest: float,
    row_bands: int,
    column_bands: int,
) -> tuple[dict[str, float], str, str]:
    """Return ``(features, feature_status, feature_reason)`` for one matrix.

    ``features`` contains only numeric feature columns; the frozen geometry
    rule is applied through :func:`describe_geometry`.  Undefined per-frame
    quantities (for example a principal axis on a single-cell mask) surface as
    ``NaN`` with ``feature_status="WARN"`` rather than as an invented number.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Pressure matrix must be finite and two-dimensional.")
    if values.shape[0] < row_bands or values.shape[1] < column_bands:
        raise ValueError("Grid partition must fit inside the pressure matrix.")

    geometry, mask = describe_geometry(
        values,
        strategy=strategy,
        positive_percentile=positive_percentile,
        minimum_raw_threshold=minimum_raw_threshold,
        minimum_component_cells=minimum_component_cells,
        minimum_component_fraction_of_largest=minimum_component_fraction_of_largest,
    )

    feature_status = "OK" if geometry["geometry_status"] == "OK" else "WARN"
    feature_reason = "" if geometry["geometry_status"] == "OK" else str(geometry["geometry_reason"])

    rows, columns = values.shape
    centroid_row = _to_float_or_nan(geometry.get("centroid_row", ""))
    centroid_column = _to_float_or_nan(geometry.get("centroid_column", ""))
    mask_features: dict[str, float] = {
        "mask_threshold_raw": float(geometry["mask_threshold_raw"]),
        "mask_cell_count": float(geometry["mask_cell_count"]),
        "mask_fraction": float(geometry["mask_fraction"]),
        "component_count": float(geometry["component_count"]),
        "bbox_row_min": _to_float_or_nan(geometry.get("bbox_row_min", "")),
        "bbox_row_max": _to_float_or_nan(geometry.get("bbox_row_max", "")),
        "bbox_column_min": _to_float_or_nan(geometry.get("bbox_column_min", "")),
        "bbox_column_max": _to_float_or_nan(geometry.get("bbox_column_max", "")),
        "bbox_height": _to_float_or_nan(geometry.get("bbox_height", "")),
        "bbox_width": _to_float_or_nan(geometry.get("bbox_width", "")),
        "bbox_area": _to_float_or_nan(geometry.get("bbox_height", ""))
        * _to_float_or_nan(geometry.get("bbox_width", "")),
        "centroid_row_fraction": (
            centroid_row / max(rows - 1, 1) if math.isfinite(centroid_row) else float("nan")
        ),
        "centroid_column_fraction": (
            centroid_column / max(columns - 1, 1)
            if math.isfinite(centroid_column)
            else float("nan")
        ),
        "cop_row_fraction": _to_float_or_nan(geometry.get("cop_row_fraction", "")),
        "cop_column_fraction": _to_float_or_nan(geometry.get("cop_column_fraction", "")),
        "principal_axis_degrees": _to_float_or_nan(geometry.get("principal_axis_degrees", "")),
        "principal_axis_anisotropy": _to_float_or_nan(geometry.get("principal_axis_anisotropy", "")),
        "contact_signal_sum": _to_float_or_nan(geometry.get("contact_signal_sum", "")),
    }

    features: dict[str, float] = {}
    features.update(_raw_intensity_features(values))
    features.update(mask_features)
    features.update(_shape_features(mask, geometry))
    features.update(_grid_features(values, row_bands=row_bands, column_bands=column_bands))
    return features, feature_status, feature_reason


def _cohort_for(quality_status: str) -> str:
    return "primary" if quality_status == "ACCEPT" else "warn"


def extract_row(
    frame: PopuTactilusFrame,
    *,
    source_relative_path: str,
    snapshot_index: int,
    quality_status: str,
    mask_rule: Mapping[str, Any],
    row_bands: int,
    column_bands: int,
    schema_version: str,
) -> dict[str, object]:
    """Build one full, traceable feature row for a single snapshot frame."""
    values = np.asarray(frame.values, dtype=np.float64)
    features, feature_status, feature_reason = extract_feature_vector(
        values,
        row_bands=row_bands,
        column_bands=column_bands,
        **mask_rule,
    )
    rows, columns = values.shape
    row: dict[str, object] = {
        "sample_id": f"popu-tactilus::{source_relative_path}#frame={snapshot_index}",
        "dataset_id": DATASET_ID,
        "source_relative_path": source_relative_path,
        "subject_id": frame.subject_id,
        "posture": frame.posture or "",
        "variation": frame.variation or "",
        "snapshot_index": snapshot_index,
        "snapshot_key": frame.snapshot_key,
        "quality_status": quality_status,
        "cohort": _cohort_for(quality_status),
        "rows": rows,
        "columns": columns,
        "matrix_orientation": MATRIX_ORIENTATION,
        "mask_rule_version": MASK_RULE_VERSION,
        "feature_schema_version": schema_version,
        "feature_status": feature_status,
        "feature_reason": feature_reason,
    }
    row.update(features)
    return row
