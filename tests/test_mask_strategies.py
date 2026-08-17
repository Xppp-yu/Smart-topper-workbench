from __future__ import annotations

import numpy as np
import pytest

from topper_perception.geometry.mask_strategies import (
    bbox_center_shift,
    bbox_iou,
    build_strategy_mask,
    consecutive_bbox_stability,
    mask_bbox,
)


def test_largest_component_keeps_only_largest_signal_island() -> None:
    values = np.zeros((7, 7), dtype=float)
    values[1:4, 1:4] = 10
    values[5, 5] = 10

    mask, threshold = build_strategy_mask(values, strategy="largest_component", positive_percentile=10)

    assert threshold == 10
    assert mask.sum() == 9
    assert not mask[5, 5]


def test_relative_closed_can_join_a_one_cell_gap() -> None:
    values = np.zeros((7, 8), dtype=float)
    values[2:5, 1:3] = 10
    values[2:5, 4:6] = 10

    filtered, _ = build_strategy_mask(values, strategy="relative_filtered", positive_percentile=10, minimum_component_cells=1)
    closed, _ = build_strategy_mask(values, strategy="relative_closed", positive_percentile=10, minimum_component_cells=1)

    assert filtered.sum() == 12
    assert closed.sum() >= filtered.sum()


def test_mask_bbox_returns_none_for_empty_and_inclusive_box_otherwise() -> None:
    assert mask_bbox(np.zeros((4, 4), dtype=bool)) is None

    mask = np.zeros((5, 6), dtype=bool)
    mask[1:4, 2:5] = True

    assert mask_bbox(mask) == (1, 3, 2, 4)


def test_bbox_iou_full_disjoint_and_partial() -> None:
    assert bbox_iou((0, 3, 0, 3), (0, 3, 0, 3)) == 1.0
    assert bbox_iou((0, 1, 0, 1), (3, 4, 3, 4)) == 0.0
    # Two 2x2 boxes sharing a 1x2 strip: intersection 2, union 6.
    assert bbox_iou((0, 1, 0, 1), (0, 1, 1, 2)) == pytest.approx(1 / 3)


def test_bbox_center_shift_measures_centre_movement() -> None:
    assert bbox_center_shift((0, 3, 0, 3), (0, 3, 0, 3)) == 0.0
    assert bbox_center_shift((0, 3, 0, 3), (1, 4, 0, 3)) == pytest.approx(1.0)


def test_consecutive_bbox_stability_reports_identity_shift_and_size() -> None:
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 1:4] = True
    shifted = np.zeros((6, 6), dtype=bool)
    shifted[2:5, 1:4] = True

    identity = consecutive_bbox_stability([mask, mask])
    assert identity["mean_consecutive_bbox_iou"] == pytest.approx(1.0)
    assert identity["mean_bbox_center_shift"] == pytest.approx(0.0)
    assert identity["median_bbox_width"] == pytest.approx(3.0)
    assert identity["median_bbox_height"] == pytest.approx(3.0)
    assert identity["bbox_width_iqr"] == pytest.approx(0.0)

    moved = consecutive_bbox_stability([mask, shifted])
    assert moved["mean_bbox_center_shift"] == pytest.approx(1.0)
    assert moved["bbox_height_iqr"] == pytest.approx(0.0)


def test_consecutive_bbox_stability_returns_nan_for_all_empty() -> None:
    empty = np.zeros((4, 4), dtype=bool)

    stability = consecutive_bbox_stability([empty, empty])

    assert np.isnan(stability["mean_consecutive_bbox_iou"])
    assert np.isnan(stability["mean_bbox_center_shift"])
    assert np.isnan(stability["median_bbox_width"])
