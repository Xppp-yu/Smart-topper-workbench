"""Subject-isolated posture baseline for PoPu fixed-posture snapshots.

The module owns the deterministic, label-safe pieces of the P5/R5 evaluation:
cohort filtering, subject splitting, model construction, within-fold metric
computation and model selection.  It never reads the raw dataset directly; the
orchestration (cross-validation, final test evaluation and artifact writing)
lives in ``scripts/baseline_popu.py`` so this module stays unit-testable without
real data.

Design guarantees:

- ``subject_id`` is the only grouping key; no snapshot is ever split across
  train/test or across CV folds.
- Every preprocessing step (median imputation for the ``principal_axis`` NaNs,
  standardization) is packaged inside a ``Pipeline`` and therefore fitted only
  on the training fold, never on the full table.
- Only the 71 P4a feature columns enter ``X``; labels and metadata are excluded
  by construction via :func:`feature_columns`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from topper_perception.features.popu import METADATA_COLUMNS
from topper_perception.io.popu import POPU_POSTURES


# Canonical label order for reports and confusion matrices.
POSTURE_LABELS: tuple[str, ...] = POPU_POSTURES

DEFAULT_RANDOM_STATE = 42
SUPPORTED_MODELS: tuple[str, ...] = ("dummy", "logreg", "rf", "knn")

IMPUTER_STEP = "imputer_median"
SCALER_STEP = "standard_scaler"
CLASSIFIER_STEP = "classifier"


@dataclass
class ModelSpec:
    """A named candidate model plus its frozen version string and estimator."""

    name: str
    version: str
    estimator: Any


def feature_columns(frame_columns: Iterable[str]) -> list[str]:
    """Return columns that are numeric features, in original order.

    Every label/metadata column is dropped by name so subject identity, posture,
    variation, path and snapshot keys can never leak into the model inputs.
    """
    excluded = set(METADATA_COLUMNS)
    return [column for column in frame_columns if column not in excluded]


def sort_subjects_numeric(subjects: Iterable[str]) -> list[str]:
    """Sort subject ids by numeric value, not lexicographically."""
    return sorted({str(subject) for subject in subjects}, key=lambda value: int(value))


def split_subjects(
    subjects: Iterable[str], *, held_out: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Split a subject set into ``(dev, test)`` around a frozen held-out set.

    Raises ``ValueError`` if any held-out subject is not present, so a typo in
    the frozen protocol fails loudly instead of silently shrinking the test set.
    """
    subject_set = {str(subject) for subject in subjects}
    held = [str(subject) for subject in held_out]
    missing = [subject for subject in held if subject not in subject_set]
    if missing:
        raise ValueError(f"Held-out subjects not present in data: {missing}")
    held_set = set(held)
    dev = [subject for subject in subject_set if subject not in held_set]
    return sort_subjects_numeric(dev), sort_subjects_numeric(held_set)


def filter_cohort(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Filter a feature table to one cohort, returning a copy."""
    if cohort == "primary":
        return df[df["cohort"] == "primary"].copy()
    if cohort == "combined":
        return df[df["cohort"].isin(("primary", "warn"))].copy()
    raise ValueError(f"Unknown cohort: {cohort!r}")


def build_model(name: str, random_state: int = DEFAULT_RANDOM_STATE) -> ModelSpec:
    """Construct one candidate model with frozen, documented parameters.

    Imputation and scaling are always inside the pipeline so they are fitted
    per training fold during cross-validation and never on the full table.
    """
    if name == "dummy":
        estimator = DummyClassifier(strategy="stratified", random_state=random_state)
        return ModelSpec(name, "dummy@stratified", estimator)

    if name == "logreg":
        estimator = Pipeline(
            [
                (IMPUTER_STEP, SimpleImputer(strategy="median")),
                (SCALER_STEP, StandardScaler()),
                (
                    CLASSIFIER_STEP,
                    LogisticRegression(
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        return ModelSpec(name, "logreg@multinomial", estimator)

    if name == "rf":
        estimator = Pipeline(
            [
                (IMPUTER_STEP, SimpleImputer(strategy="median")),
                (
                    CLASSIFIER_STEP,
                    RandomForestClassifier(
                        n_estimators=200, n_jobs=-1, random_state=random_state
                    ),
                ),
            ]
        )
        return ModelSpec(name, "rf@n200", estimator)

    if name == "knn":
        estimator = Pipeline(
            [
                (IMPUTER_STEP, SimpleImputer(strategy="median")),
                (SCALER_STEP, StandardScaler()),
                (CLASSIFIER_STEP, KNeighborsClassifier(n_neighbors=5)),
            ]
        )
        return ModelSpec(name, "knn@k5", estimator)

    raise ValueError(f"Unknown model: {name!r}")


def predict(estimator: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(predicted_labels, confidence)`` where confidence is max proba."""
    predicted = np.asarray(estimator.predict(x))
    probabilities = estimator.predict_proba(x)
    confidence = np.max(probabilities, axis=1)
    return predicted, confidence


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = POSTURE_LABELS,
) -> dict[str, Any]:
    """Compute the P5 metric set: accuracy, balanced accuracy, macro-F1, per-class.

    ``per_class`` maps each label to ``precision``/``recall``/``f1``/``support``;
    ``confusion`` is an ``(n_labels, n_labels)`` integer matrix in ``labels``
    order.  Absent classes score 0 (``zero_division=0``) instead of raising.
    """
    ordered = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ordered, zero_division=0
    )
    per_class = {
        str(ordered[index]): {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in range(len(ordered))
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion": confusion_matrix(y_true, y_pred, labels=ordered),
        "labels": ordered,
    }


def per_subject_metrics(
    subject_ids: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = POSTURE_LABELS,
) -> list[dict[str, Any]]:
    """Return one metric row per subject (accuracy, error count, macro-F1)."""
    ordered = list(labels)
    frame = pd.DataFrame(
        {
            "subject_id": [str(subject) for subject in subject_ids],
            "y_true": list(y_true),
            "y_pred": list(y_pred),
        }
    )
    rows: list[dict[str, Any]] = []
    for subject_id, group in frame.groupby("subject_id", sort=False):
        correct = int((group["y_true"] == group["y_pred"]).sum())
        total = len(group)
        _, _, f1, _ = precision_recall_fscore_support(
            group["y_true"], group["y_pred"], labels=ordered, zero_division=0
        )
        rows.append(
            {
                "subject_id": str(subject_id),
                "n_samples": total,
                "n_correct": correct,
                "n_errors": total - correct,
                "accuracy": float(correct / total) if total else float("nan"),
                "macro_f1": float(np.mean(f1)),
            }
        )
    return rows


def select_best_model(
    rows: Sequence[dict[str, Any]],
    *,
    cohort: str = "primary",
    criterion: str = "macro_f1",
    exclude: Sequence[str] = ("dummy",),
) -> str:
    """Pick the best non-excluded model by its development-set criterion.

    ``rows`` is the model-comparison table where ``split == "dev"`` marks the
    out-of-fold estimate.  Only the frozen cohort is considered, so a test-split
    row can never influence selection.
    """
    excluded = set(exclude)
    candidates = [
        row
        for row in rows
        if row.get("cohort") == cohort
        and row.get("split") == "dev"
        and row.get("model") not in excluded
    ]
    if not candidates:
        raise ValueError(f"No selectable model rows for cohort={cohort!r} split=dev")
    return str(max(candidates, key=lambda row: float(row[criterion]))["model"])
