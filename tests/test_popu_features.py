"""Focused unit tests for P4a per-snapshot label-free feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from topper_perception.features.popu import (
    MASK_RULE_VERSION,
    extract_feature_vector,
    extract_row,
    feature_column_names,
)
from topper_perception.io.popu import PopuTactilusFrame


MASK_RULE = {
    "strategy": "largest_component",
    "positive_percentile": 50.0,
    "minimum_raw_threshold": 1.0,
    "minimum_component_cells": 3,
    "minimum_component_fraction_of_largest": 0.02,
}


def _synthetic_matrix() -> np.ndarray:
    """A deterministic 8x6 matrix with a clear main-contact cluster.

    The 4x4 block uses a uniform value so the frozen 50th-percentile relative
    threshold keeps the whole block; the corner cell is a disconnected island
    that ``largest_component`` must discard.
    """
    values = np.zeros((8, 6), dtype=float)
    values[2:6, 1:5] = 10.0
    values[0, 0] = 10.0
    return values


def _frame(values: np.ndarray, *, subject_id: str, posture: str) -> PopuTactilusFrame:
    return PopuTactilusFrame(
        source_file=Path(f"E:/fake/tactilus_data/{subject_id}/x.json"),
        subject_id=subject_id,
        posture=posture,
        variation="1",
        snapshot_key="0",
        snapshot_id="0",
        values=values.astype(np.float32),
    )


def _extract(values: np.ndarray) -> tuple[dict[str, float], str, str]:
    return extract_feature_vector(
        values,
        row_bands=4,
        column_bands=3,
        **MASK_RULE,
    )


def test_feature_column_names_are_deterministic_and_label_free() -> None:
    names = feature_column_names(row_bands=4, column_bands=3)
    assert len(names) == len(set(names))
    forbidden = {
        "subject_id",
        "posture",
        "variation",
        "sample_id",
        "source_relative_path",
        "snapshot_index",
        "snapshot_key",
        "dataset_id",
        "quality_status",
        "cohort",
    }
    assert forbidden.isdisjoint(set(names))
    assert feature_column_names(4, 3) == names


def test_extract_feature_vector_returns_finite_numeric_features() -> None:
    features, status, reason = _extract(_synthetic_matrix())

    assert status == "OK"
    assert reason == ""
    assert features
    for name, value in features.items():
        assert isinstance(value, float), f"{name} is not a float"
        assert np.isfinite(value), f"{name} is not finite"


def test_frozen_rule_discards_disconnected_island() -> None:
    features, _, _ = _extract(_synthetic_matrix())

    # The single disconnected corner cell must not enter the largest component.
    assert features["component_count"] == 1
    assert features["mask_cell_count"] == 16


def test_grid_features_sum_to_total_and_fractions_normalize() -> None:
    values = _synthetic_matrix()
    features, _, _ = _extract(values)

    zone_sums = [features[f"zone_sum_r{r}c{c}"] for r in range(4) for c in range(3)]
    zone_fractions = [
        features[f"zone_fraction_r{r}c{c}"] for r in range(4) for c in range(3)
    ]
    assert sum(zone_sums) == pytest.approx(float(values.sum()))
    assert sum(zone_fractions) == pytest.approx(1.0)


def test_extract_feature_vector_is_reproducible() -> None:
    values = _synthetic_matrix()
    left, left_status, _ = _extract(values)
    right, right_status, _ = _extract(values.copy())

    assert left_status == right_status
    assert left == right


def test_feature_values_do_not_depend_on_subject_or_posture_labels() -> None:
    values = _synthetic_matrix()
    row_a = extract_row(
        _frame(values, subject_id="1", posture="left"),
        source_relative_path="1/left1_0.json",
        snapshot_index=0,
        quality_status="ACCEPT",
        mask_rule=MASK_RULE,
        row_bands=4,
        column_bands=3,
        schema_version="v0.1",
    )
    row_b = extract_row(
        _frame(values, subject_id="99", posture="supine"),
        source_relative_path="99/supine1_0.json",
        snapshot_index=0,
        quality_status="ACCEPT",
        mask_rule=MASK_RULE,
        row_bands=4,
        column_bands=3,
        schema_version="v0.1",
    )

    feature_names = set(feature_column_names(4, 3))
    assert {name: row_a[name] for name in feature_names} == {
        name: row_b[name] for name in feature_names
    }
    assert row_a["subject_id"] != row_b["subject_id"]
    assert row_a["posture"] != row_b["posture"]
    assert row_a["sample_id"] != row_b["sample_id"]


def test_single_cell_mask_marks_feature_warn_with_nan_axis() -> None:
    values = np.zeros((4, 4), dtype=float)
    values[2, 1] = 10.0

    features, status, reason = _extract(values)

    assert status == "WARN"
    assert reason == "insufficient_cells_for_principal_axis"
    assert np.isnan(features["principal_axis_degrees"])
    assert np.isnan(features["principal_axis_anisotropy"])
    for name in ("intensity_sum", "mask_cell_count", "bbox_height", "mask_extent"):
        assert np.isfinite(features[name]), f"{name} should remain finite"


def test_nonfinite_matrix_is_rejected() -> None:
    values = _synthetic_matrix()
    values[0, 0] = np.nan

    with pytest.raises(ValueError):
        _extract(values)


def test_mask_rule_version_is_a_frozen_constant() -> None:
    assert MASK_RULE_VERSION.startswith("largest_component")
