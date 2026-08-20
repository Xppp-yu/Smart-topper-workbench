"""Bounded P6.1 calibration and OOF ensemble-consistency analysis."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np
import pandas as pd

from topper_perception.neural.data import FROZEN_LABELS
from topper_perception.neural.p6_reject import PROBA_COLUMNS


def deterministic_subject_split(
    subjects: Sequence[str], *, seed: int, evaluation_count: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Create a stable subject-level development/evaluation split."""
    unique = sorted({str(value) for value in subjects})
    if not 0 < evaluation_count < len(unique):
        raise ValueError("evaluation_count must leave non-empty development and evaluation sets.")
    ranked = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )
    evaluation = tuple(sorted(ranked[:evaluation_count]))
    development = tuple(sorted(set(unique) - set(evaluation)))
    return development, evaluation


def aggregate_repeat_ensemble(frame: pd.DataFrame) -> pd.DataFrame:
    """Average the three independently trained OOF probability vectors per record."""
    rows: list[dict[str, Any]] = []
    for record_id, group in frame.groupby("record_id", sort=True):
        if len(group) != 3 or set(group["repeat"].astype(int)) != {0, 1, 2}:
            raise ValueError(f"Record {record_id!r} must have exactly one row in each repeat.")
        if group["y_true"].nunique() != 1 or group["subject_id"].astype(str).nunique() != 1:
            raise ValueError(f"Record {record_id!r} has inconsistent provenance across repeats.")
        probabilities = group.loc[:, PROBA_COLUMNS].to_numpy(dtype=np.float64)
        mean_probability = probabilities.mean(axis=0)
        repeat_predictions = group.sort_values("repeat")["y_pred"].astype(str).tolist()
        prediction_index = int(mean_probability.argmax())
        rows.append({
            "record_id": record_id,
            "subject_id": str(group["subject_id"].iloc[0]),
            "y_true": str(group["y_true"].iloc[0]),
            "y_pred": FROZEN_LABELS[prediction_index],
            "repeat_predictions": "|".join(repeat_predictions),
            "unanimous": len(set(repeat_predictions)) == 1,
            "repeat_error_count": int((group["y_pred"].astype(str) != group["y_true"].astype(str)).sum()),
            **{column: float(value) for column, value in zip(PROBA_COLUMNS, mean_probability)},
        })
    return pd.DataFrame(rows)


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling to serialized probabilities."""
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive.")
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(FROZEN_LABELS):
        raise ValueError("probabilities must have shape [n, n_classes].")
    logits = np.log(np.clip(values, 1e-15, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def multiclass_nll(probabilities: np.ndarray, labels: Sequence[str]) -> float:
    indices = np.asarray([FROZEN_LABELS.index(str(label)) for label in labels], dtype=np.int64)
    return float(-np.log(np.clip(probabilities[np.arange(len(indices)), indices], 1e-15, 1.0)).mean())


def select_temperature(frame: pd.DataFrame, grid: Sequence[float]) -> tuple[float, pd.DataFrame]:
    probabilities = frame.loc[:, PROBA_COLUMNS].to_numpy(dtype=np.float64)
    rows = []
    for value in grid:
        scaled = temperature_scale(probabilities, float(value))
        rows.append({"temperature": float(value), "development_nll": multiclass_nll(scaled, frame["y_true"])})
    table = pd.DataFrame(rows).sort_values(["development_nll", "temperature"])
    return float(table.iloc[0]["temperature"]), table.sort_values("temperature")


def calibrated_frame(frame: pd.DataFrame, temperature: float) -> pd.DataFrame:
    result = frame.copy()
    scaled = temperature_scale(result.loc[:, PROBA_COLUMNS].to_numpy(dtype=np.float64), temperature)
    result.loc[:, PROBA_COLUMNS] = scaled
    indices = scaled.argmax(axis=1)
    result["y_pred"] = np.asarray(FROZEN_LABELS)[indices]
    result["confidence"] = scaled.max(axis=1)
    result["correct"] = result["y_pred"].astype(str) == result["y_true"].astype(str)
    return result


def selective_metrics(
    frame: pd.DataFrame, threshold: float, *, require_unanimous: bool = False
) -> dict[str, Any]:
    accepted = frame["confidence"] >= threshold
    if require_unanimous:
        accepted &= frame["unanimous"].astype(bool)
    correct = frame["correct"].astype(bool)
    accepted_n = int(accepted.sum())
    wrong_n = int((accepted & ~correct).sum())
    return {
        "threshold": float(threshold), "require_unanimous": require_unanimous,
        "n": len(frame), "accepted_n": accepted_n, "coverage": accepted_n / len(frame),
        "reject_rate": 1.0 - accepted_n / len(frame),
        "accepted_accuracy": float(correct[accepted].mean()) if accepted_n else None,
        "wrong_action_rate": wrong_n / len(frame),
        "wrong_action_n": wrong_n,
    }


def select_threshold(
    frame: pd.DataFrame,
    grid: Sequence[float],
    constraints: dict[str, float],
    *,
    require_unanimous: bool = False,
) -> tuple[float, pd.DataFrame]:
    table = pd.DataFrame([
        selective_metrics(frame, value, require_unanimous=require_unanimous) for value in grid
    ])
    feasible = table[
        (table["wrong_action_rate"] <= constraints["max_wrong_action_rate"])
        & (table["accepted_accuracy"].fillna(0) >= constraints["min_accepted_accuracy"])
        & (table["coverage"] >= constraints["min_coverage"])
    ]
    if feasible.empty:
        raise ValueError("No P6.1 threshold satisfies development constraints.")
    return float(feasible.sort_values("threshold").iloc[0]["threshold"]), table


def per_subject_metrics(
    frame: pd.DataFrame, threshold: float, *, require_unanimous: bool = False
) -> pd.DataFrame:
    rows = []
    for subject, group in frame.groupby("subject_id", sort=True):
        metric = selective_metrics(group, threshold, require_unanimous=require_unanimous)
        rows.append({"subject_id": subject, **metric})
    return pd.DataFrame(rows)

