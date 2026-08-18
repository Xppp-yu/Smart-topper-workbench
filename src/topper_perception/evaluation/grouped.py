"""Dataset-agnostic subject-grouped cross-validation evaluator.

This module owns the fold construction, out-of-fold prediction and metric
aggregation for repeated subject-grouped cross-validation (P5.1's model-ranking
protocol).  It never imports PoPu, never reads CSV files and never writes
figures or knows file paths: callers hand it ``X``, ``y``, ``groups``,
``sample_ids``, a model spec and evaluation settings, and script layers own all
I/O.

Evaluation-protocol guarantees:

- A group (subject) is never split across train/validation within a fold.
- Every sample is validated exactly once per repeat (out-of-fold).
- Same seed + same config reproduce identical folds.
- Model selection consumes only :class:`OofResult` objects, which contain
  validation/OOF predictions only -- a historical held-out test can never
  influence ranking here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)

# Prefix for per-class probability columns in out-of-fold predictions and in the
# aggregated record-level table, e.g. ``proba__empty`` / ``proba__supine``.
PROBA_PREFIX = "proba__"


@dataclass(frozen=True)
class GroupFolds:
    """Per-fold index pairs ``(train_indices, val_indices)`` into one dataset."""

    folds: tuple[tuple[np.ndarray, np.ndarray], ...]

    @property
    def n_splits(self) -> int:
        return len(self.folds)


@dataclass
class OofResult:
    """Out-of-fold predictions plus per-fold metrics for one candidate model."""

    model_name: str
    model_version: str
    predictions: pd.DataFrame
    per_fold_metrics: list[dict[str, Any]]
    labels: tuple[str, ...]


def _seed_for(seed: int | None, repeat: int) -> int:
    if seed is None:
        raise ValueError("A seed is required for reproducible folds.")
    return int(seed) * 1000 + repeat


def generate_group_folds(
    groups: Sequence[str],
    *,
    n_splits: int,
    shuffle: bool = False,
    seed: int | None = None,
) -> GroupFolds:
    """Generate one subject-grouped fold set.

    Every group lands in exactly one validation chunk and all other groups in
    training, so no subject leaks across train/validation within a fold.  With
    ``shuffle=True`` a seed is required; different seeds yield different fold
    assignments, which is what makes repeated cross-validation meaningful.
    """
    unique = sorted({str(group) for group in groups})
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if n_splits > len(unique):
        raise ValueError(
            f"n_splits={n_splits} exceeds unique group count {len(unique)}"
        )

    groups_array = np.asarray(list(groups), dtype=object)
    order = np.asarray(unique, dtype=object)
    if shuffle:
        if seed is None:
            raise ValueError("A seed is required when shuffle=True.")
        order = np.random.RandomState(seed).permutation(order)

    split_groups: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index in range(n_splits):
        chunk = order[fold_index::n_splits]
        train_chunk = np.setdiff1d(order, chunk, assume_unique=True)
        train_idx = np.flatnonzero(np.isin(groups_array, train_chunk))
        val_idx = np.flatnonzero(np.isin(groups_array, chunk))
        split_groups.append((train_idx, val_idx))

    folds = GroupFolds(folds=tuple(split_groups))
    validate_group_folds(folds, groups_array)
    return folds


def generate_repeated_group_folds(
    groups: Sequence[str],
    *,
    n_splits: int,
    shuffle: bool = True,
    seed: int,
    n_repeats: int,
) -> list[GroupFolds]:
    """Generate ``n_repeats`` fold sets with distinct, seeded shuffles."""
    if seed is None:
        raise ValueError("A seed is required for reproducible repeated folds.")
    return [
        generate_group_folds(
            groups,
            n_splits=n_splits,
            shuffle=shuffle,
            seed=_seed_for(seed, repeat),
        )
        for repeat in range(n_repeats)
    ]


def validate_group_folds(folds: GroupFolds, groups: Sequence[str]) -> None:
    """Verify no group appears in both training and validation of any fold."""
    groups_array = np.asarray(list(groups), dtype=object)
    for fold_id, (train_idx, val_idx) in enumerate(folds.folds):
        train_groups = set(groups_array[train_idx])
        val_groups = set(groups_array[val_idx])
        overlap = train_groups & val_groups
        if overlap:
            raise ValueError(
                f"Group leak in fold {fold_id}: {sorted(overlap)} appear in "
                "both train and validation."
            )


def _align_proba(raw_proba: np.ndarray, classes_: Sequence[Any], labels: Sequence[str]) -> np.ndarray:
    """Align an estimator's probability matrix to the frozen label order.

    ``classes_`` indexes the estimator's probability columns.  A class the
    estimator predicts but that is NOT in the frozen label set is label drift
    and fails loudly.  A frozen label absent from ``classes_`` (e.g. a training
    fold that never saw ``empty``) is zero-filled so column order never depends
    on which classes a particular fold happened to see.
    """
    class_index = {str(label): index for index, label in enumerate(classes_)}
    frozen = list(labels)
    unexpected = [str(label) for label in classes_ if str(label) not in frozen]
    if unexpected:
        raise ValueError(
            "Estimator predicted classes not in the frozen label set: "
            f"{unexpected} (frozen labels: {frozen})"
        )
    aligned = np.zeros((raw_proba.shape[0], len(frozen)), dtype=float)
    for column, label in enumerate(frozen):
        if str(label) in class_index:
            aligned[:, column] = raw_proba[:, class_index[str(label)]]
    return aligned


def predict(
    estimator: Any, x: np.ndarray, labels: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(y_pred, proba, confidence)`` aligned to the frozen label order.

    The raw probability matrix must be finite and each row must sum to 1.0; a
    broken ``predict_proba`` fails loudly here instead of silently corrupting
    downstream confidence and record aggregation.
    """
    raw_proba = np.asarray(estimator.predict_proba(x), dtype=float)
    if not np.isfinite(raw_proba).all():
        raise ValueError("predict_proba returned non-finite probabilities")
    if not np.allclose(raw_proba.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("predict_proba rows do not sum to 1.0")
    proba = _align_proba(raw_proba, estimator.classes_, labels)
    predicted = np.asarray([labels[int(index)] for index in proba.argmax(axis=1)])
    confidence = proba.max(axis=1)
    return predicted, proba, confidence


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Compute accuracy, balanced accuracy, macro-F1 and per-class metrics."""
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
        "labels": ordered,
    }


def per_group_metrics(
    group_ids: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    """Return one metric row per group (accuracy, error count, macro-F1)."""
    ordered = list(labels)
    frame = pd.DataFrame(
        {
            "group_id": [str(group) for group in group_ids],
            "y_true": list(y_true),
            "y_pred": list(y_pred),
        }
    )
    rows: list[dict[str, Any]] = []
    for group_id, group in frame.groupby("group_id", sort=False):
        correct = int((group["y_true"] == group["y_pred"]).sum())
        total = len(group)
        _, _, f1, _ = precision_recall_fscore_support(
            group["y_true"], group["y_pred"], labels=ordered, zero_division=0
        )
        rows.append(
            {
                "group_id": str(group_id),
                "n_samples": total,
                "n_correct": correct,
                "n_errors": total - correct,
                "accuracy": float(correct / total) if total else float("nan"),
                "macro_f1": float(np.mean(f1)),
            }
        )
    return rows


def evaluate_grouped_oof(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    folds: Sequence[GroupFolds] | GroupFolds,
    *,
    sample_ids: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
) -> OofResult:
    """Produce out-of-fold predictions for one candidate over grouped folds.

    ``model`` may be a :class:`~topper_perception.models.registry.RegisteredModel`
    (preferred) or any object exposing ``estimator``/``name``/``version``, or a
    bare sklearn estimator.  Every fold fits a fresh clone on that fold's
    training subjects, so any preprocessing inside the estimator's Pipeline is
    fitted per fold and never on the full table.
    """
    if hasattr(model, "estimator") and hasattr(model, "name"):
        model_name = str(model.name)
        model_version = str(getattr(model, "version", ""))
        estimator = model.estimator
    else:
        model_name = "model"
        model_version = ""
        estimator = model

    n = len(x)
    if len(groups) != n:
        raise ValueError(
            f"groups length {len(groups)} does not match x length {n}; "
            "pass one group id per sample"
        )
    if sample_ids is not None and len(sample_ids) != n:
        raise ValueError(
            f"sample_ids length {len(sample_ids)} does not match x length {n}"
        )
    labels_tuple = tuple(labels) if labels is not None else tuple(sorted({str(v) for v in y}))
    sample_ids_arr = (
        np.asarray(list(sample_ids), dtype=object)
        if sample_ids is not None
        else np.asarray([f"sample_{index}" for index in range(n)], dtype=object)
    )
    group_ids_arr = np.asarray(list(groups), dtype=object)

    fold_sets = [folds] if isinstance(folds, GroupFolds) else list(folds)

    rows: list[dict[str, Any]] = []
    per_fold: list[dict[str, Any]] = []
    global_fold = 0
    for repeat, group_folds in enumerate(fold_sets):
        for local_fold, (train_idx, val_idx) in enumerate(group_folds.folds):
            fold_estimator = clone(estimator)
            fold_estimator.fit(x[train_idx], y[train_idx])
            y_pred, proba, confidence = predict(
                fold_estimator, x[val_idx], labels=labels_tuple
            )
            metrics = compute_metrics(y[val_idx], y_pred, labels=labels_tuple)
            per_fold.append(
                {
                    "repeat": repeat,
                    "local_fold": local_fold,
                    "fold_id": global_fold,
                    "n_samples": int(len(val_idx)),
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )
            proba_columns = {
                f"{PROBA_PREFIX}{label}": proba[:, column]
                for column, label in enumerate(labels_tuple)
            }
            for index, position in enumerate(val_idx):
                row: dict[str, Any] = {
                    "sample_id": str(sample_ids_arr[position]),
                    "group_id": str(group_ids_arr[position]),
                    "fold_id": global_fold,
                    "repeat": repeat,
                    "y_true": str(y[position]),
                    "y_pred": str(y_pred[index]),
                    "confidence": float(confidence[index]),
                }
                for column_name, values in proba_columns.items():
                    row[column_name] = float(values[index])
                rows.append(row)
            global_fold += 1

    return OofResult(
        model_name=model_name,
        model_version=model_version,
        predictions=pd.DataFrame(rows),
        per_fold_metrics=per_fold,
        labels=labels_tuple,
    )


def oof_summary(oof: OofResult) -> dict[str, Any]:
    """Aggregate one OOF result: overall, per-group, fold stats, worst group."""
    preds = oof.predictions
    y_true = preds["y_true"].to_numpy()
    y_pred = preds["y_pred"].to_numpy()
    overall = compute_metrics(y_true, y_pred, labels=oof.labels)
    per_group = per_group_metrics(
        preds["group_id"].to_numpy(), y_true, y_pred, oof.labels
    )

    fold_macro_f1 = [fold["macro_f1"] for fold in oof.per_fold_metrics]
    fold_accuracy = [fold["accuracy"] for fold in oof.per_fold_metrics]

    finite_groups = [row for row in per_group if row["accuracy"] == row["accuracy"]]
    worst_group = min(finite_groups, key=lambda row: row["accuracy"]) if finite_groups else None

    return {
        "model_name": oof.model_name,
        "n_samples": int(len(preds)),
        "accuracy": overall["accuracy"],
        "balanced_accuracy": overall["balanced_accuracy"],
        "macro_f1": overall["macro_f1"],
        "per_class": overall["per_class"],
        "per_group": per_group,
        "fold_macro_f1_mean": (
            float(np.mean(fold_macro_f1)) if fold_macro_f1 else float("nan")
        ),
        "fold_macro_f1_std": (
            float(np.std(fold_macro_f1)) if fold_macro_f1 else float("nan")
        ),
        "fold_accuracy_mean": (
            float(np.mean(fold_accuracy)) if fold_accuracy else float("nan")
        ),
        "worst_group": worst_group,
    }


def select_best_model(
    oof_results: Mapping[str, OofResult],
    *,
    criterion: str = "macro_f1",
    exclude: Sequence[str] = (),
    tie_break: str = "balanced_accuracy",
) -> str:
    """Pick the best non-excluded candidate by its grouped-CV OOF criterion.

    Only :class:`OofResult` values are consulted, which by construction contain
    validation/OOF predictions only -- a historical held-out test cannot affect
    selection here.
    """
    excluded = set(exclude)
    candidates = {
        name: result for name, result in oof_results.items() if name not in excluded
    }
    if not candidates:
        raise ValueError("No selectable model results (all excluded or none provided).")
    scores = {name: oof_summary(result)[criterion] for name, result in candidates.items()}
    ties = {name: oof_summary(result)[tie_break] for name, result in candidates.items()}
    best = max(candidates, key=lambda name: (scores[name], ties[name]))
    return str(best)


def snapshot_metrics_per_repeat(oof: OofResult) -> list[dict[str, Any]]:
    """Return one metric row per repeat over the snapshot-level OOF predictions.

    Repeats are never pooled: each row is computed on that repeat's own
    out-of-fold samples, and ``reduce_repeat_metrics`` turns them into
    mean/std across repeats.
    """
    preds = oof.predictions
    rows: list[dict[str, Any]] = []
    for repeat in sorted(int(value) for value in preds["repeat"].unique()):
        frame = preds[preds["repeat"] == repeat]
        metrics = compute_metrics(
            frame["y_true"].to_numpy(), frame["y_pred"].to_numpy(), labels=oof.labels
        )
        rows.append(
            {
                "repeat": repeat,
                "n_samples": int(len(frame)),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    return rows


def reduce_repeat_metrics(
    per_repeat_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str] = ("accuracy", "balanced_accuracy", "macro_f1"),
) -> dict[str, dict[str, float]]:
    """Summarize per-repeat metric rows into ``{metric: {mean, std}}``."""
    return {
        metric: {
            "mean": float(np.mean([row[metric] for row in per_repeat_rows])),
            "std": float(np.std([row[metric] for row in per_repeat_rows])),
        }
        for metric in metrics
    }


def repeated_subject_metrics(
    frame: pd.DataFrame,
    *,
    labels: Sequence[str],
    repeat_col: str = "repeat",
    group_col: str = "group_id",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> list[dict[str, Any]]:
    """Per-subject accuracy/macro-F1 mean+std across repeats; worst subject flagged.

    Callers pass the frame they want subject metrics over (snapshot-level OOF
    or record-level aggregations) with the repeat column, so subject-level
    stability always mirrors the same caliber it is reported against.
    """
    if repeat_col not in frame.columns or group_col not in frame.columns:
        raise ValueError(
            f"frame must contain {repeat_col!r} and {group_col!r} columns for "
            "repeated subject metrics"
        )
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for repeat in sorted(int(value) for value in frame[repeat_col].unique()):
        rows = frame[frame[repeat_col] == repeat]
        for row in per_group_metrics(
            rows[group_col].to_numpy(),
            rows[y_true_col].to_numpy(),
            rows[y_pred_col].to_numpy(),
            labels,
        ):
            by_subject[str(row["group_id"])].append(row)

    subjects: list[dict[str, Any]] = []
    for subject_id, rows in by_subject.items():
        accuracies = [row["accuracy"] for row in rows]
        macro_f1s = [row["macro_f1"] for row in rows]
        subjects.append(
            {
                "subject_id": subject_id,
                "n_repeats": len(rows),
                "n_samples_mean": float(np.mean([row["n_samples"] for row in rows])),
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies)),
                "macro_f1_mean": float(np.mean(macro_f1s)),
                "macro_f1_std": float(np.std(macro_f1s)),
            }
        )
    if subjects:
        worst_id = min(subjects, key=lambda row: row["accuracy_mean"])["subject_id"]
        for row in subjects:
            row["is_worst"] = row["subject_id"] == worst_id
    return subjects


def select_best_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    criterion: str,
    tie_break: str,
    worst_subject_criterion: str,
    complexity_order: Sequence[str],
    exclude: Sequence[str] = (),
    margin: float = 0.0,
) -> str:
    """Pick the winning candidate from record-level metric summary rows.

    Only candidates whose primary score lies within ``margin`` of the top
    primary score compete; candidates outside the margin are dominated.  The
    winner is then ranked by ``(tie_break, worst_subject_criterion, -complexity)``:
    a slightly-lower primary score is treated as tied within the margin, the
    fixed tie-break ladder decides, and only a true tie falls through to
    preferring the earliest candidate in ``complexity_order``.  ``exclude``
    removes lower-bound/reference candidates (e.g. ``dummy``).
    """
    excluded = set(exclude)
    candidates = [row for row in rows if str(row["model"]) not in excluded]
    if not candidates:
        raise ValueError("No selectable candidates (all excluded or none provided).")
    complexity_index = {str(name): index for index, name in enumerate(complexity_order)}
    missing = [
        str(row["model"]) for row in candidates if str(row["model"]) not in complexity_index
    ]
    if missing:
        raise ValueError(f"Candidates missing from complexity_order: {missing}")
    for key in (criterion, tie_break, worst_subject_criterion):
        if any(key not in row for row in candidates):
            raise ValueError(f"Candidate summary rows missing required metric {key!r}")

    top_primary = max(float(row[criterion]) for row in candidates)
    within = [
        row for row in candidates if top_primary - float(row[criterion]) <= margin
    ]
    winner = max(
        within,
        key=lambda row: (
            float(row[tie_break]),
            float(row[worst_subject_criterion]),
            -complexity_index[str(row["model"])],
        ),
    )
    return str(winner["model"])
