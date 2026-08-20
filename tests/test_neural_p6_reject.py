"""Unit tests for the P6 reject rule and uncertainty diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from topper_perception.neural.p6_reject import (
    add_uncertainty_columns,
    confusion_matrix,
    choose_threshold,
    threshold_metrics,
    threshold_metrics_for,
    validate_record_oof,
)


def _frame() -> pd.DataFrame:
    rows = []
    labels = ["empty", "supine", "prone", "left", "right"]
    for repeat in range(3):
        for i, label in enumerate(labels):
            probs = np.full(5, 0.01)
            probs[i] = 0.96
            rows.append({
                "model": "small_resnet", "repeat": repeat, "outer_seed": 11 + repeat,
                "local_fold": 0, "record_id": f"r{repeat}-{i}", "subject_id": str(i),
                "y_true": label, "y_pred": label, "confidence": 0.96,
                "n_snapshots": 10, **{f"proba__{name}": value for name, value in zip(labels, probs)},
            })
    return pd.DataFrame(rows)


def test_uncertainty_and_threshold_metrics() -> None:
    frame = add_uncertainty_columns(_frame())
    assert frame["top2_margin"].iloc[0] == pytest.approx(0.95)
    assert frame["normalized_entropy"].iloc[0] < 0.2
    metrics = threshold_metrics(frame, 0.97)
    assert metrics["coverage"] == 0.0
    assert metrics["wrong_action_rate"] == 0.0


def test_threshold_selection_uses_lowest_feasible_threshold() -> None:
    frame = add_uncertainty_columns(_frame())
    rule, selected, _ = choose_threshold(frame[frame["repeat"] < 2], [0.5, 0.95, 0.97], min_coverage=0.5)
    assert rule.confidence_threshold == 0.5
    assert selected["coverage"] == 1.0


def test_validation_rejects_probability_argmax_mismatch() -> None:
    frame = _frame()
    frame.loc[0, "y_pred"] = "right"
    with pytest.raises(ValueError, match="argmax"):
        validate_record_oof(frame)


def test_confusion_matrix_includes_reject_column() -> None:
    frame = add_uncertainty_columns(_frame())
    matrix = confusion_matrix(frame, 0.97)
    assert "UNKNOWN/REJECT" in matrix.columns
    assert int(matrix["UNKNOWN/REJECT"].sum()) == len(frame)


def test_alternative_metrics_have_expected_direction() -> None:
    frame = add_uncertainty_columns(_frame())
    assert threshold_metrics_for(frame, "top2_margin", 0.9)["coverage"] == 1.0
    assert threshold_metrics_for(frame, "normalized_entropy", 0.0, accept_when="le")["coverage"] == 0.0
