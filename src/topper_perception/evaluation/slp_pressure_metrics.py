"""Segmentation metrics for SLP Pressure-only region segmentation.

This module provides metrics for evaluating region segmentation models:
- mIoU (mean Intersection over Union)
- Macro-F1
- Per-region IoU
- Accuracy, Precision, Recall
- Confusion matrix
- Ignore/uncertain label handling
- Empty/missing class handling

Pure NumPy implementation, no torch dependency for unit testing.

Design rules:
* Metrics are computed over fixed region classes (from external schema).
* Ignore labels are excluded from metric computation.
* Uncertain labels are flagged but may be included or excluded based on config.
* Empty classes (no predictions or no ground truth) are handled gracefully.
* Zero-division returns 0, not NaN.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default ignore label value.
DEFAULT_IGNORE_LABEL = -1

#: Default uncertain label value.
DEFAULT_UNCERTAIN_LABEL = -2

#: Label values that should never be treated as valid regions.
SPECIAL_LABELS: frozenset[int] = frozenset({DEFAULT_IGNORE_LABEL, DEFAULT_UNCERTAIN_LABEL, -100})


@dataclass(frozen=True, slots=True)
class RegionMetrics:
    """Per-region IoU, precision, recall, F1 for one region class."""
    region_id: str
    iou: float
    precision: float
    recall: float
    f1: float
    intersection: int
    union: int
    pred_count: int
    gt_count: int
    is_empty_pred: bool  # True if no predictions for this class
    is_empty_gt: bool    # True if no ground truth for this class
    is_ignored: bool      # True if this region was ignored

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "intersection": self.intersection,
            "union": self.union,
            "pred_count": self.pred_count,
            "gt_count": self.gt_count,
            "is_empty_pred": self.is_empty_pred,
            "is_empty_gt": self.is_empty_gt,
            "is_ignored": self.is_ignored,
        }


@dataclass(frozen=True, slots=True)
class SegmentationMetrics:
    """Aggregate segmentation metrics for a region segmentation task."""

    # Overall metrics
    mIoU: float  # Mean IoU over all valid regions
    mIoU_strict: float  # Mean IoU excluding empty classes
    macro_f1: float
    macro_precision: float
    macro_recall: float
    pixel_accuracy: float  # Per-pixel accuracy
    mean_pixel_accuracy: float  # Per-class accuracy averaged

    # Per-region metrics
    per_region: tuple[RegionMetrics, ...]

    # Confusion matrix (n_classes × n_classes)
    confusion_matrix: np.ndarray  # C[true][pred]

    # Class information
    n_classes: int
    n_valid_classes: int  # Non-empty classes in either pred or gt
    n_empty_pred: int
    n_empty_gt: int
    n_ignored: int

    # Sample counts
    n_samples: int
    n_ignored_samples: int  # Samples with all-ignore masks

    # Coverage
    valid_pixel_fraction: float  # Fraction of pixels not ignored

    def as_dict(self) -> dict[str, Any]:
        return {
            "mIoU": self.mIoU,
            "mIoU_strict": self.mIoU_strict,
            "macro_f1": self.macro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "pixel_accuracy": self.pixel_accuracy,
            "mean_pixel_accuracy": self.mean_pixel_accuracy,
            "per_region": [r.as_dict() for r in self.per_region],
            "confusion_matrix": self.confusion_matrix.tolist(),
            "n_classes": self.n_classes,
            "n_valid_classes": self.n_valid_classes,
            "n_empty_pred": self.n_empty_pred,
            "n_empty_gt": self.n_empty_gt,
            "n_ignored": self.n_ignored,
            "n_samples": self.n_samples,
            "n_ignored_samples": self.n_ignored_samples,
            "valid_pixel_fraction": self.valid_pixel_fraction,
        }


# ---------------------------------------------------------------------------
# Core Metric Functions
# ---------------------------------------------------------------------------


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    ignore_label: int = DEFAULT_IGNORE_LABEL,
    uncertain_label: int = DEFAULT_UNCERTAIN_LABEL,
) -> np.ndarray:
    """Compute confusion matrix for segmentation.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth label map (H × W).
    y_pred : np.ndarray
        Predicted label map (H × W).
    n_classes : int
        Number of region classes.
    ignore_label : int
        Label value to ignore in computation.
    uncertain_label : int
        Label value treated as uncertain (excluded from strict metrics).

    Returns
    -------
    np.ndarray
        Confusion matrix C[true][pred] with shape (n_classes, n_classes).
    """
    # Flatten
    true_flat = y_true.ravel()
    pred_flat = y_pred.ravel()

    # Create mask for valid pixels (not ignore, not uncertain)
    valid_mask = np.ones_like(true_flat, dtype=bool)
    valid_mask = valid_mask & (true_flat != ignore_label)
    valid_mask = valid_mask & (pred_flat != ignore_label)
    valid_mask = valid_mask & (true_flat != uncertain_label)
    valid_mask = valid_mask & (pred_flat != uncertain_label)

    # Apply mask
    true_valid = true_flat[valid_mask]
    pred_valid = pred_flat[valid_mask]

    # Compute confusion matrix
    # C[i][j] = count of pixels with true=i and pred=j
    cm = np.bincount(
        true_valid * n_classes + pred_valid,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)

    return cm.astype(np.int64)


def compute_region_iou(
    confusion_matrix: np.ndarray,
    class_idx: int,
    ignore_label: int = DEFAULT_IGNORE_LABEL,
) -> tuple[float, int, int, int]:
    """Compute IoU for one region class.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        C[true][pred] confusion matrix.
    class_idx : int
        Class index to compute IoU for.
    ignore_label : int
        Ignore label value.

    Returns
    -------
    tuple[float, int, int, int]
        (iou, intersection, union, gt_count)
    """
    # Intersection = diagonal (TP)
    intersection = int(confusion_matrix[class_idx, class_idx])

    # Union = TP + FP + FN
    # TP = diagonal
    # FP = column sum - diagonal
    # FN = row sum - diagonal
    pred_count = int(confusion_matrix[:, class_idx].sum())
    gt_count = int(confusion_matrix[class_idx, :].sum())
    fp = pred_count - intersection
    fn = gt_count - intersection
    union = intersection + fp + fn

    # Compute IoU
    if union == 0:
        iou = 0.0
    else:
        iou = intersection / union

    return iou, intersection, union, gt_count


def compute_region_precision_recall(
    confusion_matrix: np.ndarray,
    class_idx: int,
) -> tuple[float, float]:
    """Compute precision and recall for one region class.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        C[true][pred] confusion matrix.
    class_idx : int
        Class index.

    Returns
    -------
    tuple[float, float]
        (precision, recall)
    """
    # TP = diagonal
    tp = confusion_matrix[class_idx, class_idx]

    # FP = column sum - TP
    fp = confusion_matrix[:, class_idx].sum() - tp

    # FN = row sum - TP
    fn = confusion_matrix[class_idx, :].sum() - tp

    # Precision = TP / (TP + FP)
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    # Recall = TP / (TP + FN)
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    return precision, recall


def compute_segmentation_metrics(
    y_true: np.ndarray | Sequence[np.ndarray],
    y_pred: np.ndarray | Sequence[np.ndarray],
    region_ids: Sequence[str],
    *,
    ignore_label: int = DEFAULT_IGNORE_LABEL,
    uncertain_label: int = DEFAULT_UNCERTAIN_LABEL,
    strict_mode: bool = False,  # If True, exclude uncertain samples
) -> SegmentationMetrics:
    """Compute comprehensive segmentation metrics.

    This is the main entry point for computing region segmentation metrics.

    Parameters
    ----------
    y_true : np.ndarray or Sequence[np.ndarray]
        Ground truth label map(s). Can be a single (H, W) array or a list.
    y_pred : np.ndarray or Sequence[np.ndarray]
        Predicted label map(s). Must match y_true structure.
    region_ids : Sequence[str]
        Region ID strings for each class index.
    ignore_label : int
        Label value to completely ignore in all computations.
    uncertain_label : int
        Label value treated as uncertain.
    strict_mode : bool
        If True, exclude uncertain samples from all computations.

    Returns
    -------
    SegmentationMetrics
        Comprehensive metrics object.

    Raises
    ------
    ValueError
        If inputs are invalid or mismatched.
    """
    # Normalize inputs to sequences
    if isinstance(y_true, np.ndarray):
        y_true = [y_true]
    if isinstance(y_pred, np.ndarray):
        y_pred = [y_pred]

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length, "
            f"got {len(y_true)} and {len(y_pred)}"
        )

    if len(y_true) == 0:
        raise ValueError("At least one sample is required.")

    n_classes = len(region_ids)

    # Aggregate confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    # Track ignored samples
    n_ignored_samples = 0
    valid_pixel_total = 0
    correct_pixel_total = 0
    total_pixel = 0

    for i, (gt, pred) in enumerate(zip(y_true, y_pred)):
        # Validate shapes
        if gt.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch at sample {i}: "
                f"y_true {gt.shape} vs y_pred {pred.shape}"
            )

        # Count total pixels
        total_pixel += gt.size

        # Create masks
        ignore_mask = (gt == ignore_label) | (pred == ignore_label)
        uncertain_mask = (gt == uncertain_label) | (pred == uncertain_label)

        # In strict mode, uncertain pixels are excluded
        if strict_mode:
            exclude_mask = ignore_mask | uncertain_mask
        else:
            exclude_mask = ignore_mask

        # Count ignored samples (all pixels excluded)
        if exclude_mask.all():
            n_ignored_samples += 1

        # Compute valid pixels
        valid_mask = ~exclude_mask
        valid_pixel_total += valid_mask.sum()

        # Count correct pixels
        correct_mask = valid_mask & (gt == pred)
        correct_pixel_total += correct_mask.sum()

        # Compute per-sample confusion matrix
        sample_cm = compute_confusion_matrix(
            gt, pred, n_classes,
            ignore_label=ignore_label,
            uncertain_label=uncertain_label if not strict_mode else ignore_label,
        )
        cm += sample_cm

    # Compute per-region metrics
    per_region: list[RegionMetrics] = []
    ious = []
    f1s = []
    precisions = []
    recalls = []
    n_empty_pred = 0
    n_empty_gt = 0

    for class_idx, region_id in enumerate(region_ids):
        iou, intersection, union, gt_count = compute_region_iou(
            cm, class_idx, ignore_label=ignore_label
        )
        precision, recall = compute_region_precision_recall(cm, class_idx)

        # F1 = 2 * precision * recall / (precision + recall)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        # Check for empty classes
        pred_count = int(cm[:, class_idx].sum())
        is_empty_pred = pred_count == 0
        is_empty_gt = gt_count == 0
        is_ignored = False  # Currently no mechanism to mark a region as ignored

        if is_empty_pred:
            n_empty_pred += 1
        if is_empty_gt:
            n_empty_gt += 1

        per_region.append(RegionMetrics(
            region_id=region_id,
            iou=iou,
            precision=precision,
            recall=recall,
            f1=f1,
            intersection=intersection,
            union=union,
            pred_count=pred_count,
            gt_count=gt_count,
            is_empty_pred=is_empty_pred,
            is_empty_gt=is_empty_gt,
            is_ignored=is_ignored,
        ))

        # Accumulate for macro averages (only non-empty classes)
        if not is_empty_pred and not is_empty_gt:
            ious.append(iou)
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)

    # Compute aggregate metrics
    if len(ious) > 0:
        mIoU = float(np.mean(ious))
        mIoU_strict = mIoU
        macro_f1 = float(np.mean(f1s))
        macro_precision = float(np.mean(precisions))
        macro_recall = float(np.mean(recalls))
    else:
        # All classes empty
        mIoU = 0.0
        mIoU_strict = 0.0
        macro_f1 = 0.0
        macro_precision = 0.0
        macro_recall = 0.0

    # Pixel accuracy
    if valid_pixel_total > 0:
        pixel_accuracy = correct_pixel_total / valid_pixel_total
    else:
        pixel_accuracy = 0.0

    # Mean pixel accuracy (per-class)
    if len(per_region) > 0:
        class_accuracies = []
        for r in per_region:
            total = r.pred_count + r.gt_count - r.intersection
            if total > 0:
                class_accuracies.append(r.intersection / total)
            else:
                class_accuracies.append(0.0)
        mean_pixel_accuracy = float(np.mean(class_accuracies)) if class_accuracies else 0.0
    else:
        mean_pixel_accuracy = 0.0

    # Valid pixel fraction
    if total_pixel > 0:
        valid_fraction = valid_pixel_total / total_pixel
    else:
        valid_fraction = 0.0

    return SegmentationMetrics(
        mIoU=mIoU,
        mIoU_strict=mIoU_strict,
        macro_f1=macro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        pixel_accuracy=pixel_accuracy,
        mean_pixel_accuracy=mean_pixel_accuracy,
        per_region=tuple(per_region),
        confusion_matrix=cm,
        n_classes=n_classes,
        n_valid_classes=len(ious),
        n_empty_pred=n_empty_pred,
        n_empty_gt=n_empty_gt,
        n_ignored=0,  # TODO: implement if needed
        n_samples=len(y_true),
        n_ignored_samples=n_ignored_samples,
        valid_pixel_fraction=valid_fraction,
    )


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def iou_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_idx: int,
    ignore_label: int = DEFAULT_IGNORE_LABEL,
) -> float:
    """Compute IoU score for a single class.

    Simplified interface for binary-like evaluation.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth.
    y_pred : np.ndarray
        Predictions.
    class_idx : int
        Class to evaluate.
    ignore_label : int
        Label to ignore.

    Returns
    -------
    float
        IoU score in [0, 1].
    """
    # Mask for valid pixels
    valid_mask = (y_true != ignore_label) & (y_pred != ignore_label)

    # Create binary masks for this class
    true_binary = (y_true == class_idx) & valid_mask
    pred_binary = (y_pred == class_idx) & valid_mask

    intersection = np.sum(true_binary & pred_binary)
    union = np.sum(true_binary | pred_binary)

    if union == 0:
        return 0.0

    return intersection / union


def dice_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_idx: int,
    ignore_label: int = DEFAULT_IGNORE_LABEL,
) -> float:
    """Compute Dice (F1) score for a single class.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth.
    y_pred : np.ndarray
        Predictions.
    class_idx : int
        Class to evaluate.
    ignore_label : int
        Label to ignore.

    Returns
    -------
    float
        Dice score in [0, 1].
    """
    # Mask for valid pixels
    valid_mask = (y_true != ignore_label) & (y_pred != ignore_label)

    # Create binary masks for this class
    true_binary = (y_true == class_idx) & valid_mask
    pred_binary = (y_pred == class_idx) & valid_mask

    intersection = np.sum(true_binary & pred_binary)
    total = np.sum(true_binary) + np.sum(pred_binary)

    if total == 0:
        return 0.0

    return 2 * intersection / total


def pixel_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ignore_label: int = DEFAULT_IGNORE_LABEL,
    uncertain_label: int = DEFAULT_UNCERTAIN_LABEL,
) -> float:
    """Compute per-pixel accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth.
    y_pred : np.ndarray
        Predictions.
    ignore_label : int
        Label to ignore.
    uncertain_label : int
        Label to treat as uncertain (excluded).

    Returns
    -------
    float
        Fraction of valid pixels that match.
    """
    valid_mask = (
        (y_true != ignore_label)
        & (y_pred != ignore_label)
        & (y_true != uncertain_label)
        & (y_pred != uncertain_label)
    )

    if valid_mask.sum() == 0:
        return 0.0

    return np.mean((y_true == y_pred)[valid_mask])


def create_synthetic_segmentation(
    shape: tuple[int, int],
    n_classes: int,
    seed: int = 42,
    empty_class_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic ground truth and predictions for testing.

    Parameters
    ----------
    shape : tuple[int, int]
        (height, width) of the segmentation maps.
    n_classes : int
        Number of region classes.
    seed : int
        Random seed for reproducibility.
    empty_class_fraction : float
        Fraction of classes to make empty (no pixels).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (y_true, y_pred) synthetic label maps.
    """
    rng = np.random.default_rng(seed)

    # Create random ground truth
    y_true = rng.integers(0, n_classes, size=shape)

    # Create predictions with some noise
    y_pred = y_true.copy()
    noise_mask = rng.random(shape) < 0.3  # 30% noise
    y_pred[noise_mask] = rng.integers(0, n_classes, size=noise_mask.sum())

    # Make some classes empty if requested
    if empty_class_fraction > 0:
        n_empty = int(n_classes * empty_class_fraction)
        empty_classes = rng.choice(n_classes, size=n_empty, replace=False)
        for cls in empty_classes:
            y_true[y_true == cls] = 0  # Merge into class 0
            y_pred[y_pred == cls] = 0

    return y_true.astype(np.int32), y_pred.astype(np.int32)


def create_mock_labels(
    shape: tuple[int, int],
    region_ids: Sequence[str],
    class_weights: Sequence[float] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Create a mock ground truth label map for testing.

    Parameters
    ----------
    shape : tuple[int, int]
        (height, width) of the label map.
    region_ids : Sequence[str]
        Region IDs defining class order.
    class_weights : Sequence[float] | None
        Relative weights for class distribution.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Label map with class indices.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(region_ids)

    if class_weights is None:
        # Uniform distribution
        probs = np.ones(n_classes) / n_classes
    else:
        # Weighted distribution
        probs = np.array(class_weights)
        probs = probs / probs.sum()

    # Create flat labels
    flat_probs = np.tile(probs, (shape[0] * shape[1], 1))
    flat_labels = rng.choice(n_classes, size=shape[0] * shape[1], p=probs)

    return flat_labels.reshape(shape).astype(np.int32)


def rasterize_polygon(
    polygon: np.ndarray,
    shape: tuple[int, int],
    fill_value: int = 1,
) -> np.ndarray:
    """Rasterize a polygon into a binary mask.

    Simple scanline rasterization for convex polygons.
    For non-convex polygons, use a proper library.

    Parameters
    ----------
    polygon : np.ndarray
        Vertices as (N, 2) array.
    shape : tuple[int, int]
        Output mask shape (height, width).
    fill_value : int
        Value to fill inside the polygon.

    Returns
    -------
    np.ndarray
        Binary mask with the polygon filled.
    """
    from matplotlib.path import Path as MplPath

    mask = np.zeros(shape, dtype=np.uint8)

    if polygon.shape[0] < 3:
        return mask

    # Create matplotlib path (x, y) format
    vertices = polygon[:, :2]  # Take x, y only
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 2) + [MplPath.CLOSEPOLY]
    path = MplPath(vertices, codes)

    # Create grid
    y_grid, x_grid = np.mgrid[0:shape[0], 0:shape[1]]
    points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # Check which points are inside
    inside = path.contains_points(points).reshape(shape)

    mask[inside] = fill_value

    return mask


def rasterize_polygons(
    polygons: Sequence[tuple[str, np.ndarray]],
    shape: tuple[int, int],
    class_indices: dict[str, int],
    ignore_label: int = DEFAULT_IGNORE_LABEL,
) -> np.ndarray:
    """Rasterize multiple polygons into a label map.

    Parameters
    ----------
    polygons : Sequence[tuple[str, np.ndarray]]
        List of (region_id, polygon_vertices) pairs.
    shape : tuple[int, int]
        Output label map shape.
    class_indices : dict[str, int]
        Mapping from region_id to class index.
    ignore_label : int
        Value for pixels not covered by any polygon.

    Returns
    -------
    np.ndarray
        Label map with class indices.
    """
    label_map = np.full(shape, ignore_label, dtype=np.int32)

    for region_id, polygon in polygons:
        if region_id not in class_indices:
            continue

        class_idx = class_indices[region_id]
        mask = rasterize_polygon(polygon, shape, fill_value=1)

        # Assign class to masked pixels
        label_map[mask > 0] = class_idx

    return label_map
