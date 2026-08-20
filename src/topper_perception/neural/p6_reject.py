"""P6 UNKNOWN/REJECT analysis for frozen record-level OOF predictions.

This module deliberately operates on record-level probabilities emitted by the
P5.2 Full runner. Threshold selection is separated from evaluation by repeat
so the reported operating point is not selected on the same repeat on which it
is judged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from topper_perception.neural.data import FROZEN_LABELS

PROBA_COLUMNS = tuple(f"proba__{label}" for label in FROZEN_LABELS)
REQUIRED_COLUMNS = {
    "model", "repeat", "outer_seed", "local_fold", "record_id", "subject_id",
    "y_true", "y_pred", "confidence", "n_snapshots", *PROBA_COLUMNS,
}


@dataclass(frozen=True, slots=True)
class RejectRule:
    """A record is accepted only when its confidence is at least this value."""

    confidence_threshold: float
    name: str = "max_probability"


def load_record_oof(paths: Iterable[str | Any]) -> pd.DataFrame:
    """Load and validate Small ResNet record OOF CSVs."""
    path_list = [str(path) for path in paths]
    if not path_list:
        raise ValueError("At least one OOF CSV path is required.")
    frame = pd.concat([pd.read_csv(path) for path in path_list], ignore_index=True)
    validate_record_oof(frame)
    return add_uncertainty_columns(frame)


def validate_record_oof(frame: pd.DataFrame) -> None:
    """Fail closed on schema, probability, coverage, and provenance errors."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"OOF is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("OOF must not be empty.")
    if set(frame["model"].astype(str).unique()) != {"small_resnet"}:
        raise ValueError("P6 input must contain only small_resnet record predictions.")
    labels = set(FROZEN_LABELS)
    for column in ("y_true", "y_pred"):
        if not set(frame[column].astype(str).unique()).issubset(labels):
            raise ValueError(f"{column} contains labels outside frozen label order.")
    if not np.all(frame["n_snapshots"].to_numpy(dtype=np.int64) == 10):
        raise ValueError("Every record must contain exactly 10 snapshots.")
    probabilities = frame.loc[:, PROBA_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("OOF probabilities must be finite and in [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError("OOF probability rows must sum to 1 within 1e-6.")
    expected_pred = np.asarray(FROZEN_LABELS)[probabilities.argmax(axis=1)]
    if not np.array_equal(expected_pred, frame["y_pred"].astype(str).to_numpy()):
        raise ValueError("y_pred does not match probability argmax.")
    if not np.allclose(frame["confidence"].to_numpy(float), probabilities.max(axis=1), atol=1e-6, rtol=0):
        raise ValueError("confidence does not match max probability.")
    key = frame["repeat"].astype(str) + "::" + frame["record_id"].astype(str)
    if key.duplicated().any():
        raise ValueError("A record occurs more than once within a repeat.")
    if set(frame["repeat"].unique()) != {0, 1, 2}:
        raise ValueError("P6 Full OOF must contain repeats 0, 1, and 2.")
    counts = frame.groupby("repeat")["record_id"].nunique()
    if not (counts == 5006).all():
        raise ValueError(f"Each repeat must contain 5006 records, observed {counts.to_dict()}.")


def add_uncertainty_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add top-1 confidence, top-2 margin, and normalized entropy diagnostics."""
    result = frame.copy()
    probabilities = result.loc[:, PROBA_COLUMNS].to_numpy(dtype=np.float64)
    ordered = np.sort(probabilities, axis=1)
    result["top1_probability"] = ordered[:, -1]
    result["top2_probability"] = ordered[:, -2]
    result["top2_margin"] = ordered[:, -1] - ordered[:, -2]
    result["normalized_entropy"] = (
        -(probabilities * np.log(np.clip(probabilities, 1e-15, 1.0))).sum(axis=1)
        / np.log(len(FROZEN_LABELS))
    )
    result["correct"] = result["y_true"].astype(str) == result["y_pred"].astype(str)
    return result


def apply_rule(frame: pd.DataFrame, rule: RejectRule) -> pd.DataFrame:
    """Return per-record accept/reject and action columns."""
    if not 0.0 <= rule.confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1].")
    result = frame.copy()
    result["accepted"] = result["top1_probability"] >= rule.confidence_threshold
    result["action"] = np.where(result["accepted"], result["y_pred"], "UNKNOWN/REJECT")
    result["wrong_action"] = result["accepted"] & ~result["correct"]
    return result


def threshold_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Compute coverage, conditional accuracy, and wrong-action rate."""
    scored = apply_rule(frame, RejectRule(threshold))
    accepted = scored["accepted"]
    n = len(scored)
    accepted_n = int(accepted.sum())
    wrong_n = int(scored["wrong_action"].sum())
    return {
        "threshold": float(threshold),
        "n": n,
        "accepted_n": accepted_n,
        "coverage": accepted_n / n,
        "reject_rate": 1.0 - accepted_n / n,
        "accepted_accuracy": float(scored.loc[accepted, "correct"].mean()) if accepted_n else None,
        "wrong_action_rate": wrong_n / n,
        "accepted_error_rate": wrong_n / accepted_n if accepted_n else None,
        "wrong_action_n": wrong_n,
    }


def threshold_metrics_for(
    frame: pd.DataFrame, metric: str, threshold: float, *, accept_when: str = "ge"
) -> dict[str, Any]:
    """Compute reject metrics for confidence, margin, or entropy."""
    if metric == "max_probability":
        score = frame["top1_probability"]
    elif metric == "top2_margin":
        score = frame["top2_margin"]
    elif metric == "normalized_entropy":
        score = frame["normalized_entropy"]
    else:
        raise ValueError(f"Unsupported P6 metric: {metric}")
    accepted = score >= threshold if accept_when == "ge" else score <= threshold
    correct = frame["correct"]
    n = len(frame)
    accepted_n = int(accepted.sum())
    wrong_n = int((accepted & ~correct).sum())
    return {
        "metric": metric, "threshold": float(threshold), "n": n,
        "accepted_n": accepted_n, "coverage": accepted_n / n,
        "reject_rate": 1.0 - accepted_n / n,
        "accepted_accuracy": float(correct[accepted].mean()) if accepted_n else None,
        "wrong_action_rate": wrong_n / n,
        "accepted_error_rate": wrong_n / accepted_n if accepted_n else None,
        "wrong_action_n": wrong_n,
    }


def threshold_table_for(
    frame: pd.DataFrame, metric: str, thresholds: Sequence[float], *, accept_when: str = "ge"
) -> pd.DataFrame:
    return pd.DataFrame([threshold_metrics_for(frame, metric, value, accept_when=accept_when) for value in thresholds])


def threshold_table(frame: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    """Return threshold metrics in the supplied deterministic order."""
    return pd.DataFrame([threshold_metrics(frame, value) for value in thresholds])


def choose_threshold(
    development: pd.DataFrame,
    thresholds: Sequence[float],
    *,
    max_wrong_action_rate: float = 0.005,
    min_accepted_accuracy: float = 0.995,
    min_coverage: float = 0.50,
) -> tuple[RejectRule, dict[str, Any], str]:
    """Choose the lowest threshold meeting safety and coverage constraints."""
    table = threshold_table(development, thresholds)
    feasible = table[
        (table["wrong_action_rate"] <= max_wrong_action_rate)
        & (table["accepted_accuracy"].fillna(0) >= min_accepted_accuracy)
        & (table["coverage"] >= min_coverage)
    ]
    if feasible.empty:
        raise ValueError("No threshold meets the frozen P6 operating constraints.")
    # Lowest feasible threshold maximizes coverage; ties are deterministic.
    selected = feasible.sort_values(["threshold"], ascending=True).iloc[0].to_dict()
    rule = RejectRule(float(selected["threshold"]))
    rationale = (
        f"lowest threshold satisfying WAR <= {max_wrong_action_rate}, "
        f"accepted accuracy >= {min_accepted_accuracy}, coverage >= {min_coverage}"
    )
    return rule, selected, rationale


def grouped_metrics(frame: pd.DataFrame, by: str, threshold: float) -> pd.DataFrame:
    """Report coverage, accepted accuracy, and WAR by class or subject."""
    scored = apply_rule(frame, RejectRule(threshold))
    rows: list[dict[str, Any]] = []
    for key, group in scored.groupby(by, sort=True):
        accepted = group["accepted"]
        accepted_n = int(accepted.sum())
        wrong_n = int(group["wrong_action"].sum())
        rows.append({
            by: key,
            "n": len(group),
            "coverage": accepted_n / len(group),
            "accepted_n": accepted_n,
            "accepted_accuracy": float(group.loc[accepted, "correct"].mean()) if accepted_n else None,
            "wrong_action_rate": wrong_n / len(group),
            "accepted_error_rate": wrong_n / accepted_n if accepted_n else None,
        })
    return pd.DataFrame(rows)


def error_cases(frame: pd.DataFrame, threshold: float, *, high_confidence: float = 0.90) -> pd.DataFrame:
    """Return all errors with derived uncertainty and high-confidence flag."""
    scored = apply_rule(frame, RejectRule(threshold))
    errors = scored.loc[~scored["correct"]].copy()
    errors["high_confidence_error"] = errors["top1_probability"] >= high_confidence
    return errors.sort_values(["high_confidence_error", "top1_probability"], ascending=[False, False])


def confusion_matrix(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return accepted-action confusion counts, with UNKNOWN/REJECT as a column."""
    scored = apply_rule(frame, RejectRule(threshold))
    columns = (*FROZEN_LABELS, "UNKNOWN/REJECT")
    matrix = pd.crosstab(scored["y_true"], scored["action"]).reindex(
        index=FROZEN_LABELS, columns=columns, fill_value=0
    )
    matrix.index.name = "y_true"
    return matrix.reset_index()
