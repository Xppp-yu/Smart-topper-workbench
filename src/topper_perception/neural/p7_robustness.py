"""Deterministic software perturbations for PoPu P7 robustness evaluation."""

from __future__ import annotations

import numpy as np


def _matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim not in (2, 3):
        raise ValueError("pressure input must be [rows, cols] or [frames, rows, cols].")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("pressure input must be finite and non-negative.")
    return array.copy()


def downsample_nearest(values: np.ndarray, row_stride: int, column_stride: int) -> np.ndarray:
    """Simulate lower spatial density and reconstruct to the original grid."""
    array = _matrix(values)
    if row_stride < 1 or column_stride < 1:
        raise ValueError("density strides must be positive integers.")
    rows, columns = array.shape[-2:]
    kept_rows = np.arange(0, rows, row_stride)
    kept_columns = np.arange(0, columns, column_stride)
    nearest_rows = kept_rows[np.abs(np.arange(rows)[:, None] - kept_rows).argmin(axis=1)]
    nearest_columns = kept_columns[np.abs(np.arange(columns)[:, None] - kept_columns).argmin(axis=1)]
    return array[..., nearest_rows[:, None], nearest_columns]


def add_relative_gaussian_noise(values: np.ndarray, sigma_fraction: float, *, seed: int) -> np.ndarray:
    """Add zero-mean noise scaled by the positive-value p95, clipping below zero."""
    array = _matrix(values)
    if sigma_fraction < 0:
        raise ValueError("sigma_fraction must be non-negative.")
    positive = array[array > 0]
    scale = float(np.percentile(positive, 95)) if positive.size else 0.0
    noise = np.random.default_rng(seed).normal(0.0, sigma_fraction * scale, size=array.shape)
    return np.maximum(array + noise.astype(np.float32), 0.0)


def inject_bad_cells(values: np.ndarray, fraction: float, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Set a fixed random fraction of sensor cells to zero across all frames."""
    array = _matrix(values)
    if not 0 <= fraction <= 1:
        raise ValueError("bad-cell fraction must be in [0, 1].")
    rows, columns = array.shape[-2:]
    count = int(round(rows * columns * fraction))
    mask = np.zeros(rows * columns, dtype=bool)
    if count:
        indices = np.random.default_rng(seed).choice(rows * columns, size=count, replace=False)
        mask[indices] = True
    mask = mask.reshape(rows, columns)
    array[..., mask] = 0.0
    return array, mask


def inject_bad_lines(
    values: np.ndarray, *, bad_rows: int = 0, bad_columns: int = 0, seed: int
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Set fixed random rows and columns to zero across all frames."""
    array = _matrix(values)
    rows, columns = array.shape[-2:]
    if not 0 <= bad_rows <= rows or not 0 <= bad_columns <= columns:
        raise ValueError("bad row/column counts exceed matrix shape.")
    rng = np.random.default_rng(seed)
    row_indices = np.sort(rng.choice(rows, size=bad_rows, replace=False))
    column_indices = np.sort(rng.choice(columns, size=bad_columns, replace=False))
    if row_indices.size:
        array[..., row_indices, :] = 0.0
    if column_indices.size:
        array[..., :, column_indices] = 0.0
    return array, (row_indices, column_indices)

