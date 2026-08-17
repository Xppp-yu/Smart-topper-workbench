from __future__ import annotations

import numpy as np

from topper_perception.geometry.mask_strategies import build_strategy_mask


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
