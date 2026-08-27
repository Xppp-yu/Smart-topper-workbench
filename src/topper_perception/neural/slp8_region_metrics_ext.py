"""Extended SLP8 region segmentation metrics (B04 v0.1).

B04 requires metrics beyond the B02 ``fixed_class_macro_metrics`` set:

* **Per-region** IoU / Dice / precision / recall / TP / FP / FN
  (the B02 helper already returns the per-class numbers; this module
  reshapes them into the B04 per-region table format).
* **Per-posture** breakdown of the same per-region metrics, indexed by
  the canonical ``SUPINE / LEFT / RIGHT`` posture string from
  :mod:`topper_perception.neural.slp8_region_dataset`.
* **Per-subject** breakdown of the same per-region metrics, indexed by
  subject_id.  The worst subject is identified by the lowest mean
  foreground IoU.
* **Centroid error** between ground-truth and predicted regions.  For
  every sample and every foreground class:

    * If the GT region is absent and the predicted region is absent
      the class does not contribute to the per-region average.
    * If the GT region is present and the predicted region is absent
      the normalized centroid error is recorded as 1.0 (the maximum).
    * If the GT region is present and the predicted region is present
      the Euclidean distance between the two centroids is divided by
      the image diagonal length and recorded.

  The image diagonal is computed as ``sqrt(H^2 + W^2)`` for the input
  spatial shape (192, 84) so the error is in ``[0, 1]``.

All metrics are computed from raw ``(H, W)`` label maps and the
accompanying subject / posture metadata; the module has **no torch
dependency** and is unit-testable in isolation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from topper_perception.evaluation.slp_pressure_metrics import (
    DEFAULT_FOREGROUND_CLASS_IDS,
    compute_fixed_class_macro_metrics,
)
from topper_perception.neural.slp8_region_dataset import N_CLASSES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical foreground class IDs (1..8) for B04 metrics.
FOREGROUND_CLASS_IDS: tuple[int, ...] = DEFAULT_FOREGROUND_CLASS_IDS

#: Default spatial shape used to compute the image diagonal.
DEFAULT_IMAGE_SHAPE: tuple[int, int] = (192, 84)

#: Centroid-error label for GT-present / pred-missing samples.
CENTROID_ERROR_MAX: float = 1.0

#: Metric version tag.
METRICS_VERSION: str = "slp8_region_metrics_ext_v0.1"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MetricsExtError(ValueError):
    """Raised when the B04 extended-metric contract is violated."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_diagonal(shape: tuple[int, int]) -> float:
    h, w = shape
    if h <= 0 or w <= 0:
        raise MetricsExtError(f"image shape must be positive, got {shape}")
    return float(math.sqrt(float(h) ** 2 + float(w) ** 2))


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Return the (row, col) centroid of ``mask`` or ``None`` if empty.

    ``mask`` is a boolean array; ``None`` is returned when no ``True``
    pixels exist so the caller can distinguish empty regions.
    """
    if mask.size == 0 or not mask.any():
        return None
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return None
    centroid = coords.mean(axis=0)
    return float(centroid[0]), float(centroid[1])


def _centroid_distance(
    gt_centroid: tuple[float, float] | None,
    pred_centroid: tuple[float, float] | None,
    diagonal: float,
) -> float:
    """Compute the per-sample, per-region normalized centroid error.

    The B04 contract is:

    * Both missing → the caller is expected to skip this entry.
    * GT present, pred missing → 1.0 (max).
    * Otherwise → Euclidean distance / diagonal.
    """
    if gt_centroid is None and pred_centroid is None:
        raise MetricsExtError(
            "_centroid_distance called with both centroids missing; "
            "the caller must filter this case before calling"
        )
    if pred_centroid is None:
        return float(CENTROID_ERROR_MAX)
    gt_r, gt_c = gt_centroid
    pr_r, pr_c = pred_centroid
    dist = float(math.sqrt((gt_r - pr_r) ** 2 + (gt_c - pr_c) ** 2))
    if diagonal <= 0:
        raise MetricsExtError(f"image diagonal must be positive, got {diagonal}")
    return dist / diagonal


# ---------------------------------------------------------------------------
# Per-region metrics
# ---------------------------------------------------------------------------


def build_per_region_table(
    class_ids: Sequence[int],
    iou: Mapping[int, float],
    dice: Mapping[int, float],
    precision: Mapping[int, float],
    recall: Mapping[int, float],
    tp: Mapping[int, int],
    fp: Mapping[int, int],
    fn: Mapping[int, int],
) -> list[dict[str, Any]]:
    """Assemble the per-region table required by B04 ``metrics_by_region.csv``."""

    table: list[dict[str, Any]] = []
    for cid in class_ids:
        if cid not in iou or cid not in dice:
            raise MetricsExtError(
                f"per-region metric for class {cid} missing from input"
            )
        table.append({
            "class_id": int(cid),
            "iou": float(iou[cid]),
            "dice": float(dice[cid]),
            "precision": float(precision.get(cid, 0.0)),
            "recall": float(recall.get(cid, 0.0)),
            "tp": int(tp.get(cid, 0)),
            "fp": int(fp.get(cid, 0)),
            "fn": int(fn.get(cid, 0)),
        })
    return table


# ---------------------------------------------------------------------------
# Per-posture / per-subject breakdown
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Subset:
    """Indices into the global sample list for one (split, posture/subject)."""

    label_indices: list[int]
    pred_indices: list[int]


def _split_indices_by_key(
    keys: Sequence[str],
    indices: Sequence[int],
) -> dict[str, list[int]]:
    """Group sample indices by the corresponding key string."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for key, idx in zip(keys, indices):
        grouped[str(key)].append(int(idx))
    return dict(grouped)


def _per_key_metrics(
    labels: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    indices: Sequence[int],
) -> dict[str, Any]:
    """Run the fixed-class macro metric on the selected subset."""

    if not indices:
        return {
            "n_samples": 0,
            "fixed_foreground_macro_iou": 0.0,
            "fixed_foreground_macro_dice": 0.0,
            "pixel_accuracy": 0.0,
            "per_class": {},
        }
    sub_labels = [labels[i] for i in indices]
    sub_preds = [predictions[i] for i in indices]
    fixed = compute_fixed_class_macro_metrics(
        sub_labels,
        sub_preds,
        class_ids=FOREGROUND_CLASS_IDS,
        n_classes=N_CLASSES,
    )
    return {
        "n_samples": int(fixed.n_samples),
        "fixed_foreground_macro_iou": float(fixed.fixed_iou),
        "fixed_foreground_macro_dice": float(fixed.fixed_dice),
        "pixel_accuracy": float(fixed.pixel_accuracy),
        "per_class": {
            "iou": {int(k): float(v) for k, v in fixed.per_class_iou.items()},
            "dice": {int(k): float(v) for k, v in fixed.per_class_dice.items()},
            "precision": {int(k): float(v) for k, v in fixed.per_class_precision.items()},
            "recall": {int(k): float(v) for k, v in fixed.per_class_recall.items()},
            "tp": {int(k): int(v) for k, v in fixed.per_class_tp.items()},
            "fp": {int(k): int(v) for k, v in fixed.per_class_fp.items()},
            "fn": {int(k): int(v) for k, v in fixed.per_class_fn.items()},
        },
    }


def compute_per_posture_metrics(
    labels: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    postures: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Compute per-posture metric blocks keyed by posture string.

    Returns
    -------
    dict
        Mapping ``posture -> { n_samples, fixed_foreground_macro_iou, ...,
        per_class: {...} }``.  An "ALL" key is included so the runner can
        emit a single consistent table.
    """

    if len(labels) != len(predictions) or len(labels) != len(postures):
        raise MetricsExtError(
            "labels/predictions/postures must have equal length; got "
            f"{len(labels)}/{len(predictions)}/{len(postures)}"
        )
    grouped = _split_indices_by_key(postures, list(range(len(labels))))
    out: dict[str, dict[str, Any]] = {"ALL": _per_key_metrics(
        labels, predictions, list(range(len(labels)))
    )}
    for posture, idxs in grouped.items():
        out[posture] = _per_key_metrics(labels, predictions, idxs)
    return out


def compute_per_subject_metrics(
    labels: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    subject_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Compute per-subject metric blocks keyed by subject_id."""

    if len(labels) != len(predictions) or len(labels) != len(subject_ids):
        raise MetricsExtError(
            "labels/predictions/subject_ids must have equal length; got "
            f"{len(labels)}/{len(predictions)}/{len(subject_ids)}"
        )
    grouped = _split_indices_by_key(subject_ids, list(range(len(labels))))
    out: dict[str, dict[str, Any]] = {"ALL": _per_key_metrics(
        labels, predictions, list(range(len(labels)))
    )}
    for subject, idxs in grouped.items():
        out[subject] = _per_key_metrics(labels, predictions, idxs)
    return out


# ---------------------------------------------------------------------------
# Worst subject
# ---------------------------------------------------------------------------


def find_worst_subject(
    per_subject: Mapping[str, Mapping[str, Any]],
    *,
    exclude: Iterable[str] = ("ALL",),
) -> dict[str, Any] | None:
    """Return the subject with the lowest fixed foreground macro IoU.

    Returns
    -------
    dict or None
        ``None`` if no real subject is present (only ``ALL`` or empty).
        Otherwise the subject_id, the macro IoU, and the per-class
        breakdown.
    """

    excluded = set(exclude)
    candidates = [
        (subject, metrics)
        for subject, metrics in per_subject.items()
        if subject not in excluded and int(metrics.get("n_samples", 0)) > 0
    ]
    if not candidates:
        return None
    worst_subject, worst_metrics = min(
        candidates, key=lambda item: float(item[1].get("fixed_foreground_macro_iou", 0.0))
    )
    return {
        "subject_id": str(worst_subject),
        "n_samples": int(worst_metrics.get("n_samples", 0)),
        "fixed_foreground_macro_iou": float(worst_metrics.get("fixed_foreground_macro_iou", 0.0)),
        "fixed_foreground_macro_dice": float(worst_metrics.get("fixed_foreground_macro_dice", 0.0)),
        "pixel_accuracy": float(worst_metrics.get("pixel_accuracy", 0.0)),
        "selection_rule": "argmin(fixed_foreground_macro_iou) over subjects with n_samples>0",
    }


# ---------------------------------------------------------------------------
# Centroid error
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CentroidErrorRecord:
    """One per-sample, per-region centroid error record."""

    sample_index: int
    subject_id: str
    region_id: int
    error: float
    both_missing: bool  # True iff both GT and pred regions are absent

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_index": int(self.sample_index),
            "subject_id": str(self.subject_id),
            "region_id": int(self.region_id),
            "error": float(self.error),
            "both_missing": bool(self.both_missing),
        }


def compute_centroid_errors(
    labels: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    subject_ids: Sequence[str],
    *,
    class_ids: Sequence[int] = FOREGROUND_CLASS_IDS,
    image_shape: tuple[int, int] = DEFAULT_IMAGE_SHAPE,
) -> list[CentroidErrorRecord]:
    """Compute the per-sample, per-region centroid-error list."""

    if len(labels) != len(predictions) or len(labels) != len(subject_ids):
        raise MetricsExtError(
            "labels/predictions/subject_ids must have equal length; got "
            f"{len(labels)}/{len(predictions)}/{len(subject_ids)}"
        )
    diagonal = _image_diagonal(image_shape)
    records: list[CentroidErrorRecord] = []
    for i, (lab, pred, subj) in enumerate(zip(labels, predictions, subject_ids)):
        if lab.shape != image_shape or pred.shape != image_shape:
            raise MetricsExtError(
                f"sample {i} has unexpected shape "
                f"(label={lab.shape}, pred={pred.shape}); expected {image_shape}"
            )
        for cid in class_ids:
            gt_mask = (lab == cid)
            pr_mask = (pred == cid)
            gt_centroid = _centroid(gt_mask)
            pr_centroid = _centroid(pr_mask)
            if gt_centroid is None and pr_centroid is None:
                # Both missing — record but mark so the per-region average
                # can exclude it (the per-region aggregator will skip
                # ``both_missing=True`` rows).
                records.append(CentroidErrorRecord(
                    sample_index=i,
                    subject_id=str(subj),
                    region_id=int(cid),
                    error=0.0,
                    both_missing=True,
                ))
                continue
            error = _centroid_distance(gt_centroid, pr_centroid, diagonal)
            records.append(CentroidErrorRecord(
                sample_index=i,
                subject_id=str(subj),
                region_id=int(cid),
                error=float(error),
                both_missing=False,
            ))
    return records


@dataclass(frozen=True)
class CentroidErrorSummary:
    """Centroid-error summary across a candidate Mini run."""

    overall_mean: float
    per_region_mean: dict[int, float]
    per_region_count: dict[int, int]
    n_records: int
    n_both_missing: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall_mean": float(self.overall_mean),
            "per_region_mean": {str(k): float(v) for k, v in self.per_region_mean.items()},
            "per_region_count": {str(k): int(v) for k, v in self.per_region_count.items()},
            "n_records": int(self.n_records),
            "n_both_missing": int(self.n_both_missing),
        }


def summarize_centroid_errors(
    records: Sequence[CentroidErrorRecord],
    *,
    class_ids: Sequence[int] = FOREGROUND_CLASS_IDS,
) -> CentroidErrorSummary:
    """Aggregate centroid-error records into a per-region and overall summary.

    The per-region average excludes ``both_missing=True`` records, per the
    B04 contract.  The overall mean is the simple average of the
    included (non-both-missing) records, or 0.0 when no records are
    included.
    """

    included = [r for r in records if not r.both_missing]
    n_both_missing = int(len(records) - len(included))
    per_region_sum: dict[int, float] = {int(cid): 0.0 for cid in class_ids}
    per_region_count: dict[int, int] = {int(cid): 0 for cid in class_ids}
    for r in included:
        per_region_sum[r.region_id] += float(r.error)
        per_region_count[r.region_id] += 1
    per_region_mean: dict[int, float] = {}
    for cid, total in per_region_sum.items():
        count = per_region_count[cid]
        per_region_mean[cid] = float(total / count) if count else 0.0

    if not included:
        overall_mean = 0.0
    else:
        overall_mean = float(sum(r.error for r in included) / len(included))

    return CentroidErrorSummary(
        overall_mean=overall_mean,
        per_region_mean=per_region_mean,
        per_region_count=per_region_count,
        n_records=len(records),
        n_both_missing=n_both_missing,
    )


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def build_confusion_matrix(
    labels: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    n_classes: int = N_CLASSES,
) -> np.ndarray:
    """Build a ``(n_classes, n_classes)`` confusion matrix over all samples.

    Rows are the ground-truth class, columns are the predicted class.
    Pixels outside ``[0, n_classes)`` are excluded from the matrix.
    """

    if len(labels) != len(predictions):
        raise MetricsExtError(
            f"labels and predictions must have equal length, got "
            f"{len(labels)} and {len(predictions)}"
        )
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for lab, pred in zip(labels, predictions):
        if lab.shape != pred.shape:
            raise MetricsExtError(
                f"shape mismatch (label={lab.shape}, pred={pred.shape})"
            )
        gt = lab.astype(np.int64, copy=False).ravel()
        pr = pred.astype(np.int64, copy=False).ravel()
        valid = (gt >= 0) & (gt < n_classes) & (pr >= 0) & (pr < n_classes)
        cm += np.bincount(
            gt[valid] * n_classes + pr[valid], minlength=n_classes * n_classes
        ).reshape(n_classes, n_classes).astype(np.int64)
    return cm


# ---------------------------------------------------------------------------
# Top-level convenience: extended metrics bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtendedMetricsBundle:
    """All B04 extended metrics for one (split, candidate) block."""

    fixed_foreground_macro_iou: float
    fixed_foreground_macro_dice: float
    pixel_accuracy: float
    background_iou: float
    per_region: list[dict[str, Any]]
    per_posture: dict[str, dict[str, Any]]
    per_subject: dict[str, dict[str, Any]]
    worst_subject: dict[str, Any] | None
    confusion_matrix: np.ndarray
    centroid_error_summary: CentroidErrorSummary
    n_samples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fixed_foreground_macro_iou": float(self.fixed_foreground_macro_iou),
            "fixed_foreground_macro_dice": float(self.fixed_foreground_macro_dice),
            "pixel_accuracy": float(self.pixel_accuracy),
            "background_iou": float(self.background_iou),
            "per_region": list(self.per_region),
            "per_posture": {k: v for k, v in self.per_posture.items()},
            "per_subject": {k: v for k, v in self.per_subject.items()},
            "worst_subject": dict(self.worst_subject) if self.worst_subject is not None else None,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "centroid_error_summary": self.centroid_error_summary.as_dict(),
            "n_samples": int(self.n_samples),
            "metrics_version": METRICS_VERSION,
        }


def compute_extended_metrics(
    labels: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    subject_ids: Sequence[str],
    postures: Sequence[str],
    *,
    class_ids: Sequence[int] = FOREGROUND_CLASS_IDS,
    image_shape: tuple[int, int] = DEFAULT_IMAGE_SHAPE,
) -> ExtendedMetricsBundle:
    """One-shot computation of every B04 extended metric for a sample set.

    The B04 contract is fully enforced here:

    * ``fixed_foreground_macro_iou`` is averaged over the supplied
      ``class_ids`` with no class skipping (B02 fixed-class contract).
    * per-posture / per-subject blocks all use the same metric.
    * worst-subject is the subject with the lowest macro IoU.
    * centroid error records follow the documented handling for both
      missing / GT-only / both-present cases.
    * confusion matrix covers all ``N_CLASSES``.
    """

    if not labels:
        raise MetricsExtError("compute_extended_metrics: at least one sample required")
    if len(predictions) != len(labels):
        raise MetricsExtError(
            f"labels and predictions length mismatch: {len(labels)} vs {len(predictions)}"
        )
    if len(subject_ids) != len(labels):
        raise MetricsExtError(
            f"subject_ids length mismatch: {len(subject_ids)} vs {len(labels)}"
        )
    if len(postures) != len(labels):
        raise MetricsExtError(
            f"postures length mismatch: {len(postures)} vs {len(labels)}"
        )

    fixed = compute_fixed_class_macro_metrics(
        labels, predictions, class_ids=list(class_ids), n_classes=N_CLASSES
    )
    cm = build_confusion_matrix(labels, predictions, n_classes=N_CLASSES)
    background_tp = int(cm[0, 0])
    background_row = int(cm[0, :].sum())
    background_col = int(cm[:, 0].sum())
    background_union = background_row + background_col - background_tp
    background_iou = (
        float(background_tp / background_union) if background_union > 0 else 0.0
    )

    per_region_table = build_per_region_table(
        class_ids=class_ids,
        iou=fixed.per_class_iou,
        dice=fixed.per_class_dice,
        precision=fixed.per_class_precision,
        recall=fixed.per_class_recall,
        tp=fixed.per_class_tp,
        fp=fixed.per_class_fp,
        fn=fixed.per_class_fn,
    )

    per_posture = compute_per_posture_metrics(labels, predictions, postures)
    per_subject = compute_per_subject_metrics(labels, predictions, subject_ids)
    worst_subject = find_worst_subject(per_subject)

    centroid_records = compute_centroid_errors(
        labels, predictions, subject_ids, class_ids=class_ids, image_shape=image_shape
    )
    centroid_summary = summarize_centroid_errors(centroid_records, class_ids=class_ids)

    return ExtendedMetricsBundle(
        fixed_foreground_macro_iou=float(fixed.fixed_iou),
        fixed_foreground_macro_dice=float(fixed.fixed_dice),
        pixel_accuracy=float(fixed.pixel_accuracy),
        background_iou=background_iou,
        per_region=per_region_table,
        per_posture=per_posture,
        per_subject=per_subject,
        worst_subject=worst_subject,
        confusion_matrix=cm,
        centroid_error_summary=centroid_summary,
        n_samples=int(fixed.n_samples),
    )
