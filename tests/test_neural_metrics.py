"""Unit tests for P5.2-B classification metrics (pure NumPy, no torch)."""

from __future__ import annotations

import pytest

from topper_perception.neural.metrics import compute_classification_metrics

LABELS = ("empty", "supine", "prone", "left", "right")


def test_perfect_classification() -> None:
    y = [0, 1, 2, 3, 4]
    metrics = compute_classification_metrics(y, y, LABELS)

    assert metrics.n_samples == 5
    assert metrics.accuracy == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.macro_precision == 1.0
    assert metrics.macro_recall == 1.0
    assert metrics.confusion_matrix == (
        (1, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
    )
    assert [item.support for item in metrics.per_class] == [1, 1, 1, 1, 1]


def test_mixed_classification_matches_manual_sklearn_convention() -> None:
    # Rows are true labels, columns are predictions (sklearn convention).
    y_true = [0, 0, 0, 1, 1, 2, 3, 4]
    y_pred = [0, 0, 1, 1, 1, 2, 3, 4]
    metrics = compute_classification_metrics(y_true, y_pred, LABELS)

    assert metrics.accuracy == pytest.approx(7 / 8)
    assert metrics.confusion_matrix == (
        (2, 1, 0, 0, 0),
        (0, 2, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
    )
    assert [item.support for item in metrics.per_class] == [3, 2, 1, 1, 1]
    assert [item.precision for item in metrics.per_class] == pytest.approx(
        [1.0, 2 / 3, 1.0, 1.0, 1.0]
    )
    assert [item.recall for item in metrics.per_class] == pytest.approx(
        [2 / 3, 1.0, 1.0, 1.0, 1.0]
    )
    assert [item.f1 for item in metrics.per_class] == pytest.approx(
        [0.8, 0.8, 1.0, 1.0, 1.0]
    )
    assert metrics.balanced_accuracy == pytest.approx((2 / 3 + 4) / 5)
    assert metrics.macro_f1 == pytest.approx((0.8 + 0.8 + 3) / 5)


def test_zero_support_class_contributes_zero() -> None:
    # Only two of five labels appear; missing labels contribute 0 (fixed-K macro).
    metrics = compute_classification_metrics([0, 1], [0, 1], LABELS)

    assert metrics.accuracy == 1.0
    assert metrics.balanced_accuracy == pytest.approx(2 / 5)
    assert metrics.macro_f1 == pytest.approx(2 / 5)
    assert metrics.macro_precision == pytest.approx(2 / 5)
    assert metrics.macro_recall == pytest.approx(2 / 5)
    assert [item.support for item in metrics.per_class] == [1, 1, 0, 0, 0]


def test_raises_on_empty_input() -> None:
    with pytest.raises(ValueError):
        compute_classification_metrics([], [], LABELS)


def test_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        compute_classification_metrics([0, 1], [0], LABELS)


def test_raises_on_out_of_range_label() -> None:
    with pytest.raises(ValueError):
        compute_classification_metrics([0, 5], [0, 0], LABELS)


def test_raises_on_duplicate_labels() -> None:
    with pytest.raises(ValueError):
        compute_classification_metrics([0], [0], ("a", "a"))
