"""Transparent contact-mask and geometry calculations for PoPu pressure maps."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

from .mask_strategies import build_strategy_mask


GEOMETRY_COLUMNS = (
    "sample_id",
    "source_file",
    "subject_id",
    "posture",
    "variation",
    "p2_quality_status",
    "representative_frame_index",
    "mask_strategy",
    "geometry_status",
    "geometry_reason",
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
    "centroid_row",
    "centroid_column",
    "cop_row",
    "cop_column",
    "cop_row_fraction",
    "cop_column_fraction",
    "principal_axis_degrees",
    "principal_axis_anisotropy",
    "contact_signal_sum",
)


def build_contact_mask(
    matrix: np.ndarray,
    *,
    positive_percentile: float = 50.0,
    minimum_raw_threshold: float = 1.0,
    minimum_component_cells: int = 3,
    minimum_component_fraction_of_largest: float = 0.02,
) -> tuple[np.ndarray, float, int]:
    """Create a conservative mask from one raw map without smoothing values.

    The threshold is relative to each frame's positive readings, so this is a
    shape/geometry candidate mask rather than a calibrated physical-pressure
    boundary. Tiny disconnected islands are removed, while separated body
    regions remain available to the geometry descriptor. Components must also
    be at least a small fraction of the largest retained component, preventing
    scattered edge noise from expanding the bounding box.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Pressure matrix must be finite and two-dimensional.")
    positive = values[values > 0]
    if positive.size == 0:
        return np.zeros(values.shape, dtype=bool), float(minimum_raw_threshold), 0

    threshold = max(float(np.percentile(positive, positive_percentile)), minimum_raw_threshold)
    raw_mask = values >= threshold
    labels, count = ndimage.label(raw_mask)
    if count == 0:
        return raw_mask, threshold, 0
    component_sizes = np.bincount(labels.ravel())
    largest_component = int(component_sizes[1:].max())
    adaptive_minimum = max(
        minimum_component_cells,
        int(math.ceil(largest_component * minimum_component_fraction_of_largest)),
    )
    keep = component_sizes >= adaptive_minimum
    keep[0] = False
    mask = keep[labels]
    _, retained_count = ndimage.label(mask)
    return mask, threshold, int(retained_count)


def _empty_geometry(
    *,
    strategy: str,
    threshold: float,
    component_count: int,
) -> dict[str, object]:
    return {
        "mask_strategy": strategy,
        "geometry_status": "WARN",
        "geometry_reason": "empty_contact_mask",
        "mask_threshold_raw": threshold,
        "mask_cell_count": 0,
        "mask_fraction": 0.0,
        "component_count": component_count,
        **{column: "" for column in GEOMETRY_COLUMNS[14:]},
    }


def describe_geometry(
    matrix: np.ndarray,
    *,
    strategy: str = "relative_filtered",
    positive_percentile: float = 50.0,
    minimum_raw_threshold: float = 1.0,
    minimum_component_cells: int = 3,
    minimum_component_fraction_of_largest: float = 0.02,
) -> tuple[dict[str, object], np.ndarray]:
    """Return mask, bbox, centroid, centre of pressure and PCA-axis geometry."""
    values = np.asarray(matrix, dtype=np.float64)
    mask, threshold = build_strategy_mask(
        values,
        strategy=strategy,
        positive_percentile=positive_percentile,
        minimum_raw_threshold=minimum_raw_threshold,
        minimum_component_cells=minimum_component_cells,
        minimum_component_fraction_of_largest=minimum_component_fraction_of_largest,
    )
    _, component_count = ndimage.label(mask)
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return _empty_geometry(
            strategy=strategy,
            threshold=threshold,
            component_count=int(component_count),
        ), mask

    row_min, column_min = coordinates.min(axis=0)
    row_max, column_max = coordinates.max(axis=0)
    centroid_row, centroid_column = coordinates.mean(axis=0)
    weights = values[mask]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return _empty_geometry(
            strategy=strategy,
            threshold=threshold,
            component_count=int(component_count),
        ), mask
    cop_row = float(np.average(coordinates[:, 0], weights=weights))
    cop_column = float(np.average(coordinates[:, 1], weights=weights))

    geometry_status = "OK"
    geometry_reason = ""
    axis_degrees: float | str = ""
    anisotropy: float | str = ""
    if coordinates.shape[0] >= 2:
        centered = coordinates.astype(float) - np.asarray([cop_row, cop_column])
        covariance = np.cov(centered, rowvar=False, aweights=weights)
        if np.isfinite(covariance).all():
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            major_value, minor_value = float(eigenvalues[-1]), float(eigenvalues[0])
            major_vector = eigenvectors[:, -1]
            axis_degrees = float(np.degrees(np.arctan2(major_vector[0], major_vector[1])))
            # A line has no direction: normalize its angle to [-90, 90).
            axis_degrees = ((axis_degrees + 90.0) % 180.0) - 90.0
            anisotropy = (major_value - minor_value) / major_value if major_value > 0 else 0.0
        else:
            geometry_status = "WARN"
            geometry_reason = "non_finite_principal_axis"
    else:
        geometry_status = "WARN"
        geometry_reason = "insufficient_cells_for_principal_axis"

    rows, columns = values.shape
    result: dict[str, object] = {
        "mask_strategy": strategy,
        "geometry_status": geometry_status,
        "geometry_reason": geometry_reason,
        "mask_threshold_raw": threshold,
        "mask_cell_count": int(mask.sum()),
        "mask_fraction": float(mask.mean()),
        "component_count": int(component_count),
        "bbox_row_min": int(row_min),
        "bbox_row_max": int(row_max),
        "bbox_column_min": int(column_min),
        "bbox_column_max": int(column_max),
        "bbox_height": int(row_max - row_min + 1),
        "bbox_width": int(column_max - column_min + 1),
        "centroid_row": float(centroid_row),
        "centroid_column": float(centroid_column),
        "cop_row": cop_row,
        "cop_column": cop_column,
        "cop_row_fraction": cop_row / max(rows - 1, 1),
        "cop_column_fraction": cop_column / max(columns - 1, 1),
        "principal_axis_degrees": axis_degrees,
        "principal_axis_anisotropy": anisotropy,
        "contact_signal_sum": weight_sum,
    }
    return result, mask


def summarise_geometry(rows: list[dict[str, object]]) -> dict[str, Any]:
    """Return compact report statistics for geometry rows."""
    summary: dict[str, Any] = {"records": len(rows), "status_counts": {}, "by_posture": {}}
    for row in rows:
        status = str(row["geometry_status"])
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
    for posture in sorted({str(row["posture"]) for row in rows}):
        selected = [row for row in rows if row["posture"] == posture and row["geometry_status"] == "OK"]
        if not selected:
            continue
        summary["by_posture"][posture] = {
            "records": len(selected),
            "median_mask_fraction": float(np.median([float(row["mask_fraction"]) for row in selected])),
            "median_cop_row_fraction": float(np.median([float(row["cop_row_fraction"]) for row in selected])),
            "median_cop_column_fraction": float(np.median([float(row["cop_column_fraction"]) for row in selected])),
            "median_axis_degrees": float(np.median([float(row["principal_axis_degrees"]) for row in selected])),
        }
    return summary
