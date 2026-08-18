"""Tests for snapshot-to-record probability aggregation.

A PoPu JSON record carries 10 highly correlated snapshots.  Aggregation reduces
per-snapshot predictions to one record-level prediction by mean probability and
must reject records whose snapshots disagree on the label or the subject/group.
"""

from __future__ import annotations

import pandas as pd
import pytest

from topper_perception.evaluation import aggregation


def _snapshot_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["r1", "r1", "r1", "r2", "r2", "r2"],
            "group_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "y_true": ["A", "A", "A", "B", "B", "B"],
            "A": [0.9, 0.8, 0.7, 0.1, 0.2, 0.3],
            "B": [0.1, 0.2, 0.3, 0.9, 0.8, 0.7],
        }
    )


def test_record_aggregation_uses_mean_probability() -> None:
    result = aggregation.aggregate_record_predictions(
        _snapshot_predictions(),
        record_id_col="record_id",
        group_id_col="group_id",
        y_true_col="y_true",
        label_columns=["A", "B"],
    )

    assert set(result["record_id"]) == {"r1", "r2"}

    row_1 = result[result["record_id"] == "r1"].iloc[0]
    assert row_1["A"] == pytest.approx(0.8)  # mean of 0.9, 0.8, 0.7
    assert row_1["B"] == pytest.approx(0.2)
    assert row_1["y_pred"] == "A"
    assert row_1["confidence"] == pytest.approx(0.8)
    assert row_1["n_snapshots"] == 3
    assert row_1["group_id"] == "s1"
    assert row_1["y_true"] == "A"

    row_2 = result[result["record_id"] == "r2"].iloc[0]
    assert row_2["y_pred"] == "B"
    assert row_2["confidence"] == pytest.approx(0.8)


def test_record_aggregation_conflicting_labels_error() -> None:
    df = _snapshot_predictions()
    df.loc[1, "y_true"] = "B"  # conflict inside record r1

    with pytest.raises(ValueError, match="label"):
        aggregation.aggregate_record_predictions(
            df,
            record_id_col="record_id",
            group_id_col="group_id",
            y_true_col="y_true",
            label_columns=["A", "B"],
        )


def test_record_aggregation_conflicting_group_error() -> None:
    df = _snapshot_predictions()
    df.loc[2, "group_id"] = "other"  # snapshots of r1 claim two subjects

    with pytest.raises(ValueError, match="group"):
        aggregation.aggregate_record_predictions(
            df,
            record_id_col="record_id",
            group_id_col="group_id",
            y_true_col="y_true",
            label_columns=["A", "B"],
        )


def test_record_id_from_sample_id_parses_p4a_contract() -> None:
    record = aggregation.record_id_from_sample_id(
        "popu-tactilus::popu/s5/r1/f2.json#frame=7"
    )
    assert record == "popu/s5/r1/f2.json"


def test_record_id_from_sample_id_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        aggregation.record_id_from_sample_id("not-a-popu-sample-id")


def test_record_id_from_source_path_is_the_path() -> None:
    assert aggregation.record_id_from_source_path("popu/s1/r1/f1.json") == "popu/s1/r1/f1.json"


def test_record_aggregation_groups_by_repeat_and_record_id() -> None:
    df = pd.DataFrame(
        {
            "repeat": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "record_id": ["r1", "r1", "r1", "r2", "r2", "r2", "r1", "r1", "r1", "r2", "r2", "r2"],
            "group_id": ["s1", "s1", "s1", "s2", "s2", "s2", "s1", "s1", "s1", "s2", "s2", "s2"],
            "y_true": ["A", "A", "A", "B", "B", "B", "A", "A", "A", "B", "B", "B"],
            "proba__A": [0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.6, 0.6, 0.6, 0.2, 0.2, 0.2],
            "proba__B": [0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8],
        }
    )
    result = aggregation.aggregate_record_predictions(
        df, record_id_col="record_id", group_id_col="group_id", y_true_col="y_true",
        label_columns=["proba__A", "proba__B"], repeat_id_col="repeat",
    )
    # 2 repeats x 2 records; the snapshots of r1 in repeat 0 must NOT be mixed
    # with the snapshots of r1 in repeat 1.
    assert len(result) == 4
    assert "repeat" in result.columns
    r1_r0 = result[(result["repeat"] == 0) & (result["record_id"] == "r1")].iloc[0]
    assert r1_r0["proba__A"] == pytest.approx(0.8)  # mean of 0.9, 0.8, 0.7
    assert r1_r0["y_pred"] == "A"
    r2_r1 = result[(result["repeat"] == 1) & (result["record_id"] == "r2")].iloc[0]
    assert r2_r1["proba__B"] == pytest.approx(0.8)
    assert r2_r1["y_pred"] == "B"


def test_record_aggregation_supports_proba_prefixed_columns_without_repeat() -> None:
    df = pd.DataFrame(
        {
            "record_id": ["r1", "r1"],
            "group_id": ["s1", "s1"],
            "y_true": ["A", "A"],
            "proba__A": [0.7, 0.9],
            "proba__B": [0.3, 0.1],
        }
    )
    result = aggregation.aggregate_record_predictions(
        df, record_id_col="record_id", group_id_col="group_id", y_true_col="y_true",
        label_columns=["proba__A", "proba__B"],
    )
    row = result.iloc[0]
    assert row["proba__A"] == pytest.approx(0.8)
    assert row["y_pred"] == "A"
    assert row["proba__B"] == pytest.approx(0.2)
