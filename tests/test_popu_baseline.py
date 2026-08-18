"""Focused unit tests for P5/R5 subject-isolated posture baseline evaluation.

These tests cover the pure, deterministic pieces of the baseline module: cohort
filtering, subject splitting, model construction, within-fold imputation,
metric computation and model selection.  They do not touch the raw PoPu dataset;
a real run is exercised separately by ``scripts/baseline_popu.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from topper_perception.baseline.popu import (
    POSTURE_LABELS,
    ModelSpec,
    build_model,
    compute_metrics,
    feature_columns,
    filter_cohort,
    per_subject_metrics,
    predict,
    select_best_model,
    sort_subjects_numeric,
    split_subjects,
)
from topper_perception.features.popu import METADATA_COLUMNS


def test_feature_columns_exclude_metadata_and_labels() -> None:
    columns = list(METADATA_COLUMNS) + ["intensity_sum", "mask_cell_count"]
    result = feature_columns(columns)

    assert result == ["intensity_sum", "mask_cell_count"]
    assert set(METADATA_COLUMNS).isdisjoint(result)


def test_sort_subjects_numeric_handles_lexicographic_order() -> None:
    assert sort_subjects_numeric(["10", "2", "1"]) == ["1", "2", "10"]


def test_split_subjects_is_disjoint_and_exhaustive() -> None:
    subjects = [str(i) for i in range(1, 11)]
    dev, test = split_subjects(subjects, held_out=["5", "10"])

    assert dev == ["1", "2", "3", "4", "6", "7", "8", "9"]
    assert test == ["5", "10"]
    assert set(dev).isdisjoint(test)
    assert set(dev) | set(test) == set(subjects)


def test_split_subjects_rejects_unknown_held_out() -> None:
    with pytest.raises(ValueError):
        split_subjects(["1", "2", "3"], held_out=["99"])


def test_filter_cohort_primary_and_combined() -> None:
    df = pd.DataFrame(
        {
            "cohort": ["primary", "warn", "warn", "primary"],
            "sample_id": ["a", "b", "c", "d"],
        }
    )

    primary = filter_cohort(df, "primary")
    assert list(primary["sample_id"]) == ["a", "d"]

    combined = filter_cohort(df, "combined")
    assert list(combined["sample_id"]) == ["a", "b", "c", "d"]

    with pytest.raises(ValueError):
        filter_cohort(df, "unknown")


def test_build_model_returns_expected_models() -> None:
    for name in ("dummy", "logreg", "rf", "knn"):
        spec = build_model(name, random_state=0)
        assert isinstance(spec, ModelSpec)
        assert spec.name == name
        assert spec.version
        assert hasattr(spec.estimator, "fit")
        assert hasattr(spec.estimator, "predict")
        assert hasattr(spec.estimator, "predict_proba")

    with pytest.raises(ValueError):
        build_model("does_not_exist")


def test_pipeline_imputer_fits_only_on_training_fold() -> None:
    """The median imputer must learn from the training fold, never full data."""
    estimator = build_model("logreg", random_state=0).estimator
    x_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, np.nan]])
    y_train = np.array(["empty", "supine", "empty"])

    estimator.fit(x_train, y_train)
    imputer = estimator.named_steps["imputer_median"]

    assert imputer.statistics_[0] == pytest.approx(3.0)  # median of [1, 3, 5]
    assert imputer.statistics_[1] == pytest.approx(3.0)  # median of [2, 4]


def test_predict_handles_nan_and_returns_confidence_in_unit_interval() -> None:
    estimator = build_model("logreg", random_state=0).estimator
    x_train = np.vstack(
        [
            np.tile([1.0, 1.0], (4, 1)),
            np.tile([5.0, 5.0], (4, 1)),
        ]
    )
    y_train = np.array(["empty"] * 4 + ["supine"] * 4)
    x_test = np.array([[1.0, np.nan], [5.0, 5.0], [1.0, 1.0]])

    estimator.fit(x_train, y_train)
    y_pred, confidence = predict(estimator, x_test)

    assert len(y_pred) == len(x_test)
    assert len(confidence) == len(x_test)
    assert all(str(p) in set(POSTURE_LABELS) for p in y_pred)
    assert np.all(confidence >= 0.0)
    assert np.all(confidence <= 1.0)


def test_compute_metrics_perfect_classification() -> None:
    y_true = np.array(["empty", "supine", "prone", "left", "right"] * 2)
    y_pred = y_true.copy()

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    for label in POSTURE_LABELS:
        assert metrics["per_class"][label]["f1"] == pytest.approx(1.0)
        assert metrics["per_class"][label]["support"] == 2


def test_compute_metrics_balanced_accuracy_equals_mean_recall() -> None:
    y_true = np.array(["empty", "empty", "empty", "supine", "supine", "supine"])
    y_pred = np.array(["empty", "empty", "supine", "supine", "supine", "empty"])

    metrics = compute_metrics(y_true, y_pred, labels=("empty", "supine"))

    recalls = [metrics["per_class"][c]["recall"] for c in ("empty", "supine")]
    assert metrics["balanced_accuracy"] == pytest.approx(float(np.mean(recalls)))
    assert metrics["macro_f1"] == pytest.approx(
        float(np.mean([metrics["per_class"][c]["f1"] for c in ("empty", "supine")]))
    )
    assert metrics["confusion"].sum() == len(y_true)


def test_per_subject_metrics_isolates_subjects() -> None:
    subject_ids = ["1", "1", "1", "2", "2"]
    y_true = np.array(["empty", "empty", "supine", "supine", "supine"])
    y_pred = np.array(["empty", "empty", "empty", "supine", "supine"])

    rows = per_subject_metrics(subject_ids, y_true, y_pred)

    by_subject = {row["subject_id"]: row for row in rows}
    assert by_subject["1"]["n_errors"] == 1
    assert by_subject["1"]["accuracy"] == pytest.approx(2 / 3)
    assert by_subject["2"]["n_errors"] == 0
    assert by_subject["2"]["accuracy"] == pytest.approx(1.0)


def test_select_best_model_excludes_dummy_and_picks_max() -> None:
    rows = [
        {"cohort": "primary", "split": "dev", "model": "dummy", "macro_f1": 0.24},
        {"cohort": "primary", "split": "dev", "model": "rf", "macro_f1": 0.91},
        {"cohort": "primary", "split": "dev", "model": "logreg", "macro_f1": 0.88},
        {"cohort": "primary", "split": "test", "model": "rf", "macro_f1": 0.90},
    ]

    assert select_best_model(rows) == "rf"

    # No non-dummy dev rows -> error rather than silently picking dummy.
    with pytest.raises(ValueError):
        select_best_model([{"cohort": "primary", "split": "dev", "model": "dummy", "macro_f1": 0.24}])
