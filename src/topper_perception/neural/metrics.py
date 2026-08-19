"""Classification metrics for the PoPu neural path (P5.2-B Mini).

Pure NumPy, no ``torch`` dependency, so the metrics can be unit-tested and
reused across the smoke/mini/full runners without pulling in the optional
``neural`` stack. All metrics are computed over the *frozen* label order
(``FROZEN_LABELS``), never over whichever labels happen to appear in one batch,
so ``macro_*`` and ``balanced_accuracy`` stay comparable across models and runs.

Zero-division conventions (fixed, not adaptive):

- a class with ``precision + recall == 0`` gets ``f1 = 0``;
- a class with ``tp + fp == 0`` gets ``precision = 0``;
- a class with ``tp + fn == 0`` gets ``recall = 0``;

so ``balanced_accuracy`` and ``macro_*`` are means over all ``K`` classes,
including any class with zero support in a given split (that class contributes
0). This mirrors a ``zero_division=0``-style, fixed-``K`` macro.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Per-class precision/recall/F1/support for one frozen label."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Aggregate classification metrics plus per-class and confusion-matrix detail."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    per_class: tuple[ClassMetrics, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    n_samples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "per_class": [item.as_dict() for item in self.per_class],
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "n_samples": self.n_samples,
        }


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str],
) -> ClassificationMetrics:
    """Compute accuracy/balanced-accuracy/macro metrics and per-class detail.

    ``y_true`` and ``y_pred`` are integer class indices into ``labels`` (order
    fixed by the caller, normally ``FROZEN_LABELS``). The confusion matrix uses
    the sklearn convention ``C[i][j] = count(true=i, predicted=j)`` — rows are
    true labels, columns are predictions.
    """
    labels = tuple(labels)
    if not labels:
        raise ValueError("At least one label is required.")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique.")

    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.ndim != 1 or pred.ndim != 1 or true.shape != pred.shape:
        raise ValueError("y_true and y_pred must be 1-D arrays of equal length.")
    if true.size == 0:
        raise ValueError("Metrics require at least one sample.")
    k = len(labels)
    if bool(((true < 0) | (true >= k) | (pred < 0) | (pred >= k)).any()):
        raise ValueError(f"Label index out of range for {k} labels.")

    flat = true * k + pred
    confusion = np.bincount(flat, minlength=k * k).reshape(k, k).astype(np.int64)

    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp
    support = confusion.sum(axis=1).astype(np.int64)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    accuracy = float((true == pred).mean())
    per_class = tuple(
        ClassMetrics(
            label=labels[i],
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            support=int(support[i]),
        )
        for i in range(k)
    )
    return ClassificationMetrics(
        accuracy=accuracy,
        balanced_accuracy=float(recall.mean()),
        macro_f1=float(f1.mean()),
        macro_precision=float(precision.mean()),
        macro_recall=float(recall.mean()),
        per_class=per_class,
        confusion_matrix=tuple(tuple(int(v) for v in row) for row in confusion),
        n_samples=int(true.size),
    )
