"""Tests for bounded P6.1 calibration and ensemble analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from topper_perception.neural.p6_1 import (
    aggregate_repeat_ensemble,
    deterministic_subject_split,
    temperature_scale,
)


def test_subject_split_is_deterministic_and_disjoint() -> None:
    first = deterministic_subject_split([str(i) for i in range(10)], seed=6061, evaluation_count=3)
    second = deterministic_subject_split([str(i) for i in reversed(range(10))], seed=6061, evaluation_count=3)
    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert len(first[1]) == 3


def test_temperature_scale_preserves_argmax_and_normalizes() -> None:
    probabilities = np.asarray([[0.8, 0.1, 0.05, 0.03, 0.02]])
    scaled = temperature_scale(probabilities, 2.0)
    assert scaled.argmax(axis=1).tolist() == [0]
    assert scaled.sum(axis=1) == pytest.approx([1.0])
    assert scaled[0, 0] < probabilities[0, 0]


def test_repeat_ensemble_averages_probabilities() -> None:
    rows = []
    for repeat, confidence in enumerate((0.8, 0.7, 0.6)):
        rows.append({
            "record_id": "1/left1.json", "subject_id": "1", "repeat": repeat,
            "y_true": "left", "y_pred": "left", "proba__empty": 0.0,
            "proba__supine": 0.0, "proba__prone": 1.0 - confidence,
            "proba__left": confidence, "proba__right": 0.0,
        })
    result = aggregate_repeat_ensemble(pd.DataFrame(rows))
    assert len(result) == 1
    assert result.iloc[0]["proba__left"] == pytest.approx(0.7)
    assert bool(result.iloc[0]["unanimous"])
