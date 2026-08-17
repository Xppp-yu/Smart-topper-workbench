"""Comparable, explicit candidate contact-mask strategies.

These are signal-processing candidates for P3.1.  None is an anatomical
segmentation or a calibrated pressure-contact boundary.
"""

from __future__ import annotations

import math
from typing import Sequence

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


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the inclusive ``(row_min, row_max, column_min, column_max)`` of a mask.

    ``None`` for an empty mask so temporal comparisons can skip empty frames
    instead of fabricating a zero-area box.
    """
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return None
    row_min, column_min = coordinates.min(axis=0)
    row_max, column_max = coordinates.max(axis=0)
    return int(row_min), int(row_max), int(column_min), int(column_max)


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return the ``(row, column)`` centre of an inclusive bounding box."""
    return (bbox[0] + bbox[1]) / 2.0, (bbox[2] + bbox[3]) / 2.0


def bbox_iou(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    """Intersection-over-union for two inclusive axis-aligned boxes."""
    row_min = max(left[0], right[0])
    row_max = min(left[1], right[1])
    column_min = max(left[2], right[2])
    column_max = min(left[3], right[3])
    intersection = max(0, row_max - row_min + 1) * max(0, column_max - column_min + 1)
    area_left = (left[1] - left[0] + 1) * (left[3] - left[2] + 1)
    area_right = (right[1] - right[0] + 1) * (right[3] - right[2] + 1)
    union = area_left + area_right - intersection
    return float(intersection / union) if union > 0 else float("nan")


def bbox_center_shift(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    """Euclidean distance between two bbox centres, in sensor-cell units."""
    left_row, left_column = bbox_center(left)
    right_row, right_column = bbox_center(right)
    return float(np.hypot(right_row - left_row, right_column - left_column))


def consecutive_bbox_stability(masks: Sequence[np.ndarray]) -> dict[str, float]:
    """Aggregate bbox stability across consecutive frames of one record.

    IoU and centre shift are computed only on adjacent non-empty pairs; size
    statistics use every non-empty frame. A fully empty sequence yields NaN
    for every metric rather than inventing a zero-size box.
    """
    boxes = [mask_bbox(mask) for mask in masks]
    ious = [
        bbox_iou(left, right)
        for left, right in zip(boxes, boxes[1:])
        if left is not None and right is not None
    ]
    shifts = [
        bbox_center_shift(left, right)
        for left, right in zip(boxes, boxes[1:])
        if left is not None and right is not None
    ]
    widths = [box[3] - box[2] + 1 for box in boxes if box is not None]
    heights = [box[1] - box[0] + 1 for box in boxes if box is not None]
    return {
        "mean_consecutive_bbox_iou": float(np.mean(ious)) if ious else float("nan"),
        "mean_bbox_center_shift": float(np.mean(shifts)) if shifts else float("nan"),
        "median_bbox_width": float(np.median(widths)) if widths else float("nan"),
        "median_bbox_height": float(np.median(heights)) if heights else float("nan"),
        "bbox_width_iqr": (
            float(np.percentile(widths, 75) - np.percentile(widths, 25)) if widths else float("nan")
        ),
        "bbox_height_iqr": (
            float(np.percentile(heights, 75) - np.percentile(heights, 25)) if heights else float("nan")
        ),
    }
