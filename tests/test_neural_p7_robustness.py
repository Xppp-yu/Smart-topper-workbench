"""Tests for deterministic P7 software perturbations."""

from __future__ import annotations

import numpy as np
import pytest

from topper_perception.neural.p7_robustness import (
    add_relative_gaussian_noise,
    downsample_nearest,
    inject_bad_cells,
    inject_bad_lines,
)


def test_density_reconstruction_preserves_shape_and_sampled_cells() -> None:
    matrix = np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
    result = downsample_nearest(matrix, 2, 2)
    assert result.shape == matrix.shape
    assert result[0, 0] == matrix[0, 0]
    assert result[2, 2] == matrix[2, 2]


def test_noise_is_deterministic_and_non_negative() -> None:
    matrix = np.ones((2, 4, 3), dtype=np.float32)
    first = add_relative_gaussian_noise(matrix, 0.1, seed=7)
    second = add_relative_gaussian_noise(matrix, 0.1, seed=7)
    assert np.array_equal(first, second)
    assert float(first.min()) >= 0
    assert np.array_equal(matrix, np.ones_like(matrix))


def test_bad_cell_mask_is_fixed_across_frames() -> None:
    matrix = np.ones((3, 10, 10), dtype=np.float32)
    result, mask = inject_bad_cells(matrix, 0.1, seed=9)
    assert int(mask.sum()) == 10
    assert np.all(result[:, mask] == 0)


def test_bad_lines_are_zero_and_deterministic() -> None:
    matrix = np.ones((2, 8, 6), dtype=np.float32)
    result, (rows, columns) = inject_bad_lines(matrix, bad_rows=2, bad_columns=1, seed=11)
    assert len(rows) == 2 and len(columns) == 1
    assert np.all(result[:, rows, :] == 0)
    assert np.all(result[:, :, columns] == 0)


def test_invalid_pressure_fails_closed() -> None:
    with pytest.raises(ValueError):
        add_relative_gaussian_noise(np.asarray([[np.nan]], dtype=np.float32), 0.1, seed=1)

