"""Comparable, explicit candidate contact-mask strategies.

These are signal-processing candidates for P3.1.  None is an anatomical
segmentation or a calibrated pressure-contact boundary.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


MASK_STRATEGIES = ("relative_filtered", "largest_component", "relative_closed")


def _raw_mask(matrix: np.ndarray, *, positive_percentile: float, minimum_raw_threshold: float) -> tuple[np.ndarray, float]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Pressure matrix must be finite and two-dimensional.")
    positive = values[values > 0]
    if positive.size == 0:
        return np.zeros(values.shape, dtype=bool), float(minimum_raw_threshold)
    threshold = max(float(np.percentile(positive, positive_percentile)), minimum_raw_threshold)
    return values >= threshold, threshold


def _keep_size_filtered(mask: np.ndarray, *, minimum_component_cells: int, minimum_component_fraction_of_largest: float) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    minimum_size = max(
        minimum_component_cells,
        int(math.ceil(int(sizes[1:].max()) * minimum_component_fraction_of_largest)),
    )
    keep = sizes >= minimum_size
    keep[0] = False
    return keep[labels]


def build_strategy_mask(
    matrix: np.ndarray,
    *,
    strategy: str,
    positive_percentile: float = 50.0,
    minimum_raw_threshold: float = 1.0,
    minimum_component_cells: int = 3,
    minimum_component_fraction_of_largest: float = 0.02,
) -> tuple[np.ndarray, float]:
    """Build one named candidate mask from one pressure matrix."""
    if strategy not in MASK_STRATEGIES:
        raise ValueError(f"Unknown mask strategy: {strategy}")
    raw, threshold = _raw_mask(
        matrix,
        positive_percentile=positive_percentile,
        minimum_raw_threshold=minimum_raw_threshold,
    )
    if strategy == "relative_filtered":
        return _keep_size_filtered(
            raw,
            minimum_component_cells=minimum_component_cells,
            minimum_component_fraction_of_largest=minimum_component_fraction_of_largest,
        ), threshold
    if strategy == "relative_closed":
        closed = ndimage.binary_closing(raw, structure=np.ones((3, 3), dtype=bool))
        return _keep_size_filtered(
            closed,
            minimum_component_cells=minimum_component_cells,
            minimum_component_fraction_of_largest=minimum_component_fraction_of_largest,
        ), threshold

    labels, count = ndimage.label(raw)
    if count == 0:
        return raw, threshold
    sizes = np.bincount(labels.ravel())
    largest_label = int(np.argmax(sizes[1:]) + 1)
    return labels == largest_label, threshold


def mask_summary(mask: np.ndarray, values: np.ndarray) -> dict[str, float | int]:
    """Return simple dimensions for stability comparisons, not body features."""
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return {"mask_cell_count": 0, "mask_fraction": 0.0, "bbox_area_fraction": 0.0, "component_count": 0, "cop_row_fraction": float("nan"), "cop_column_fraction": float("nan")}
    labels, component_count = ndimage.label(mask)
    row_min, column_min = coordinates.min(axis=0)
    row_max, column_max = coordinates.max(axis=0)
    weights = np.asarray(values, dtype=np.float64)[mask]
    weight_sum = float(weights.sum())
    cop_row, cop_column = np.average(coordinates, axis=0, weights=weights) if weight_sum > 0 else coordinates.mean(axis=0)
    rows, columns = mask.shape
    return {
        "mask_cell_count": int(mask.sum()),
        "mask_fraction": float(mask.mean()),
        "bbox_area_fraction": float((row_max - row_min + 1) * (column_max - column_min + 1) / mask.size),
        "component_count": int(component_count),
        "cop_row_fraction": float(cop_row / max(rows - 1, 1)),
        "cop_column_fraction": float(cop_column / max(columns - 1, 1)),
    }


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Return overlap for two binary masks; two empty masks are not informative."""
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else float("nan")
