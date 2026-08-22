"""Density Transform module for SLP sensor density testing.

This module implements theoretical density changes for Pressure Map data.
It supports different density levels (100%, 50%, 25%, 12.5%) and layouts
(uniform grid, sparse grid, local high-density).

Design rules:
* Output shape contract is maintained (same dimensions as input).
* Retained sensor positions are recorded for traceability.
* Resize is NOT treated as real hardware verification.
* Only transform correctness is validated.

Note: This module implements theoretical density transformations.
It does NOT simulate real hardware limitations or physical sensor constraints.
The transformations are for model robustness testing only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Constants and Types
# ---------------------------------------------------------------------------

DENSITY_VERSION = "slp_density_transform_v0.1"

#: Valid density levels as fractions of original.
DENSITY_LEVELS: frozenset[float] = frozenset({1.0, 0.5, 0.25, 0.125})


class DensityLevel(str, Enum):
    """Predefined density levels."""
    FULL = "100%"      # 100% = 1.0
    HALF = "50%"       # 50% = 0.5
    QUARTER = "25%"    # 25% = 0.25
    EIGHTH = "12.5%"   # 12.5% = 0.125


class DensityLayout(str, Enum):
    """Spatial layout of retained sensors."""
    UNIFORM = "uniform"           # Evenly distributed grid
    SPARSE = "sparse"            # Sparse with gaps
    LOCAL_HIGH_DENSITY = "local_high_density"  # Clustered in certain areas


@dataclass(frozen=True, slots=True)
class DensityTransformConfig:
    """Configuration for a density transform."""
    density_level: float  # Fraction of original sensors retained
    layout: str  # Spatial layout strategy
    original_shape: tuple[int, int]  # (height, width)
    output_shape: tuple[int, int]  # (height, width) - same as input
    seed: int  # For reproducibility
    method: str  # Interpolation/decimation method
    fill_value: float  # Value for missing regions

    def as_dict(self) -> dict[str, Any]:
        return {
            "density_level": self.density_level,
            "layout": self.layout,
            "original_shape": list(self.original_shape),
            "output_shape": list(self.output_shape),
            "seed": self.seed,
            "method": self.method,
            "fill_value": self.fill_value,
        }


@dataclass(frozen=True, slots=True)
class SensorPosition:
    """Position of a retained sensor."""
    row: int
    col: int
    is_active: bool
    original_value: float  # Value at this position before transform


@dataclass(frozen=True, slots=True)
class DensityTransformResult:
    """Result of a density transformation."""
    data: np.ndarray  # Transformed pressure map
    config: DensityTransformConfig
    retained_positions: tuple[SensorPosition, ...]
    n_active_sensors: int
    n_total_positions: int
    active_fraction: float
    interpolation_used: str
    timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "data": self.data.tolist(),
            "config": self.config.as_dict(),
            "retained_positions": [
                {"row": p.row, "col": p.col, "is_active": p.is_active}
                for p in self.retained_positions
            ],
            "n_active_sensors": self.n_active_sensors,
            "n_total_positions": self.n_total_positions,
            "active_fraction": self.active_fraction,
            "interpolation_used": self.interpolation_used,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Sensor Selection Functions
# ---------------------------------------------------------------------------


def select_uniform_positions(
    shape: tuple[int, int],
    density_level: float,
    seed: int,
) -> np.ndarray:
    """Select positions uniformly to achieve target density.

    Parameters
    ----------
    shape : tuple[int, int]
        (height, width) of the grid.
    density_level : float
        Fraction of positions to retain (0-1).
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Boolean mask of retained positions.
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    # Calculate grid stride to achieve target density
    # For density d, we need stride s where (1/s^2) ≈ d
    # s ≈ 1/sqrt(d)
    target_density = density_level

    if target_density >= 1.0:
        # Keep all
        return np.ones(shape, dtype=bool)

    # Calculate approximate stride
    stride = max(1, int(round(1.0 / np.sqrt(target_density))))

    # Create uniform mask
    mask = np.zeros(shape, dtype=bool)

    for r in range(0, h, stride):
        for c in range(0, w, stride):
            mask[r, c] = True

    # Adjust for exact density
    current_density = mask.sum() / mask.size
    n_to_add = int((target_density - current_density) * mask.size)

    if n_to_add > 0:
        # Add random positions
        zero_positions = np.argwhere(~mask)
        if len(zero_positions) > 0:
            indices = rng.choice(len(zero_positions), size=min(n_to_add, len(zero_positions)), replace=False)
            for idx in indices:
                r, c = zero_positions[idx]
                mask[r, c] = True

    elif n_to_add < 0:
        # Remove random positions
        one_positions = np.argwhere(mask)
        indices = rng.choice(len(one_positions), size=min(-n_to_add, len(one_positions)), replace=False)
        for idx in indices:
            r, c = one_positions[idx]
            mask[r, c] = False

    return mask


def select_sparse_positions(
    shape: tuple[int, int],
    density_level: float,
    seed: int,
    gap_size: int = 3,
) -> np.ndarray:
    """Select positions with sparse gaps.

    Creates a pattern with intentional gaps to simulate
    sparse sensor deployment or sensor failure patterns.

    Parameters
    ----------
    shape : tuple[int, int]
        (height, width) of the grid.
    density_level : float
        Fraction of positions to retain.
    seed : int
        Random seed.
    gap_size : int
        Size of gap regions (in grid units).

    Returns
    -------
    np.ndarray
        Boolean mask of retained positions.
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    # Calculate base stride
    base_stride = max(1, int(round(1.0 / np.sqrt(density_level))))

    # Create sparse mask
    mask = np.zeros(shape, dtype=bool)

    # Fill with regular pattern
    for r in range(h):
        for c in range(w):
            # Check if position should be active based on stride
            is_active = (r % base_stride == 0) and (c % base_stride == 0)
            if is_active:
                # Randomly skip some positions to add variation
                if rng.random() > 0.3:
                    mask[r, c] = True

    # Add sparse gaps (larger regions with no sensors)
    n_gaps = max(1, int(gap_size * density_level))
    for _ in range(n_gaps):
        gap_r = rng.integers(0, max(1, h // (gap_size * 2)))
        gap_c = rng.integers(0, max(1, w // (gap_size * 2)))

        # Create gap region
        for dr in range(gap_size):
            for dc in range(gap_size):
                nr, nc = gap_r + dr, gap_c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    mask[nr, nc] = False

    # Adjust density
    current_density = mask.sum() / mask.size
    if current_density < density_level:
        # Add more positions
        zero_positions = np.argwhere(~mask)
        n_to_add = int((density_level - current_density) * mask.size)
        if len(zero_positions) > 0 and n_to_add > 0:
            indices = rng.choice(len(zero_positions), size=min(n_to_add, len(zero_positions)), replace=False)
            for idx in indices:
                r, c = zero_positions[idx]
                mask[r, c] = True

    return mask


def select_local_high_density_positions(
    shape: tuple[int, int],
    density_level: float,
    seed: int,
    n_clusters: int = 4,
) -> np.ndarray:
    """Select positions clustered in high-density regions.

    Simulates scenarios where sensors are concentrated
    in certain areas (e.g., under high-pressure zones).

    Parameters
    ----------
    shape : tuple[int, int]
        (height, width) of the grid.
    density_level : float
        Fraction of positions to retain.
    seed : int
        Random seed.
    n_clusters : int
        Number of high-density cluster regions.

    Returns
    -------
    np.ndarray
        Boolean mask of retained positions.
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    mask = np.zeros(shape, dtype=bool)

    # Define cluster centers
    cluster_centers = []
    margin = max(h, w) // 4
    for _ in range(n_clusters):
        # Ensure valid range for random integers
        cr_low = min(margin, h - margin - 1) if h > 2 * margin else 0
        cc_low = min(margin, w - margin - 1) if w > 2 * margin else 0
        cr_high = max(margin + 1, h - margin) if h > 2 * margin else h
        cc_high = max(margin + 1, w - margin) if w > 2 * margin else w

        cr = rng.integers(cr_low, cr_high)
        cc = rng.integers(cc_low, cc_high)
        cluster_centers.append((cr, cc))

    # Calculate cluster radius to achieve target density
    cluster_radius = int(np.sqrt((h * w * density_level) / (np.pi * n_clusters)))
    cluster_radius = max(3, cluster_radius)

    # Fill clusters
    for center_r, center_c in cluster_centers:
        for dr in range(-cluster_radius, cluster_radius + 1):
            for dc in range(-cluster_radius, cluster_radius + 1):
                nr, nc = center_r + dr, center_c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    # Gaussian-like falloff
                    dist = np.sqrt(dr**2 + dc**2)
                    prob = np.exp(-dist**2 / (2 * (cluster_radius / 2)**2))
                    if rng.random() < prob:
                        mask[nr, nc] = True

    # Add sparse background sensors
    background_density = density_level * 0.1
    background_n = int(h * w * background_density)
    for _ in range(background_n):
        r = rng.integers(0, h)
        c = rng.integers(0, w)
        mask[r, c] = True

    # Adjust density
    current_density = mask.sum() / mask.size
    if abs(current_density - density_level) > 0.05:
        # Recalculate if off by more than 5%
        if current_density < density_level:
            zero_positions = np.argwhere(~mask)
            n_to_add = int((density_level - current_density) * mask.size)
            if len(zero_positions) > 0 and n_to_add > 0:
                indices = rng.choice(len(zero_positions), size=min(n_to_add, len(zero_positions)), replace=False)
                for idx in indices:
                    r, c = zero_positions[idx]
                    mask[r, c] = True

    return mask


# ---------------------------------------------------------------------------
# Transform Functions
# ---------------------------------------------------------------------------


def downsample_to_density(
    data: np.ndarray,
    density_level: float,
    layout: str,
    *,
    seed: int = 42,
    method: str = "nearest",
    fill_value: float = 0.0,
) -> DensityTransformResult:
    """Downsample pressure map to target density.

    This is a theoretical transform - it does NOT simulate
    real hardware sensor limitations.

    Parameters
    ----------
    data : np.ndarray
        Input pressure map (H, W).
    density_level : float
        Target density as fraction (0-1).
    layout : str
        Spatial layout: "uniform", "sparse", "local_high_density".
    seed : int
        Random seed for reproducibility.
    method : str
        Interpolation method for upsampling: "nearest", "bilinear", "average".
    fill_value : float
        Value for missing (non-sensor) positions.

    Returns
    -------
    DensityTransformResult
        Transformed data and configuration.
    """
    if density_level > 1.0 or density_level <= 0:
        raise ValueError(f"density_level must be in (0, 1], got {density_level}")

    h, w = data.shape
    original_shape = (h, w)

    # Select sensor positions based on layout
    if layout == "uniform":
        position_mask = select_uniform_positions(original_shape, density_level, seed)
    elif layout == "sparse":
        position_mask = select_sparse_positions(original_shape, density_level, seed)
    elif layout == "local_high_density":
        position_mask = select_local_high_density_positions(original_shape, density_level, seed)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Create downsampled data (sparse representation)
    downsampled = np.where(position_mask, data, fill_value)

    # Build retained positions list
    retained = []
    active_count = 0
    for r in range(h):
        for c in range(w):
            is_active = position_mask[r, c]
            if is_active:
                active_count += 1
            retained.append(SensorPosition(
                row=r,
                col=c,
                is_active=is_active,
                original_value=float(data[r, c]),
            ))

    config = DensityTransformConfig(
        density_level=density_level,
        layout=layout,
        original_shape=original_shape,
        output_shape=original_shape,
        seed=seed,
        method=method,
        fill_value=fill_value,
    )

    return DensityTransformResult(
        data=downsampled,
        config=config,
        retained_positions=tuple(retained),
        n_active_sensors=active_count,
        n_total_positions=h * w,
        active_fraction=active_count / (h * w),
        interpolation_used=method,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def upsample_from_sparse(
    data: np.ndarray,
    position_mask: np.ndarray,
    method: str = "nearest",
) -> np.ndarray:
    """Upsample sparse sensor data to full resolution.

    This interpolates missing values from active sensors.
    For nearest neighbor, missing positions get the value of
    the nearest active sensor.

    Parameters
    ----------
    data : np.ndarray
        Input sparse pressure map.
    position_mask : np.ndarray
        Boolean mask of active sensor positions.
    method : str
        Interpolation method: "nearest", "average".

    Returns
    -------
    np.ndarray
        Upsampled pressure map.
    """
    if method == "nearest":
        return _nearest_neighbor_upsample(data, position_mask)
    elif method == "average":
        return _average_upsample(data, position_mask)
    else:
        raise ValueError(f"Unknown method: {method}")


def _nearest_neighbor_upsample(
    data: np.ndarray,
    position_mask: np.ndarray,
) -> np.ndarray:
    """Nearest neighbor upsampling from sparse sensors."""
    from scipy.spatial.distance import cdist

    h, w = data.shape
    result = np.zeros_like(data)

    # Get active sensor positions
    active_positions = np.argwhere(position_mask)
    if len(active_positions) == 0:
        return result

    active_values = data[position_mask]

    # For each position, find nearest active sensor
    for r in range(h):
        for c in range(w):
            if position_mask[r, c]:
                result[r, c] = data[r, c]
            else:
                # Find nearest active sensor
                dists = np.sqrt((active_positions[:, 0] - r)**2 + (active_positions[:, 1] - c)**2)
                nearest_idx = np.argmin(dists)
                result[r, c] = active_values[nearest_idx]

    return result


def _average_upsample(
    data: np.ndarray,
    position_mask: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    """Average-based upsampling from sparse sensors."""
    h, w = data.shape
    result = np.zeros_like(data)
    weights = np.zeros_like(data)

    for r in range(h):
        for c in range(w):
            # Accumulate values from nearby active sensors
            total = 0.0
            total_weight = 0.0

            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if position_mask[nr, nc]:
                            # Weight by distance
                            dist = np.sqrt(dr**2 + dc**2)
                            weight = 1.0 / (1 + dist)
                            total += data[nr, nc] * weight
                            total_weight += weight

            if total_weight > 0:
                result[r, c] = total / total_weight
            else:
                result[r, c] = 0.0

    return result


# ---------------------------------------------------------------------------
# Preset Transforms
# ---------------------------------------------------------------------------


def create_density_transform(
    density: float | str,
    layout: str = "uniform",
    seed: int = 42,
) -> tuple[float, str, dict[str, Any]]:
    """Create a density transform specification.

    Parameters
    ----------
    density : float | str
        Density level as float (0-1) or string ("100%", "50%", "25%", "12.5%").
    layout : str
        Layout strategy.
    seed : int
        Random seed.

    Returns
    -------
    tuple[float, str, dict]
        (density_level, layout, transform_params)
    """
    # Normalize density
    if isinstance(density, str):
        density_map = {
            "100%": 1.0,
            "50%": 0.5,
            "25%": 0.25,
            "12.5%": 0.125,
        }
        if density not in density_map:
            raise ValueError(f"Unknown density: {density}")
        density_level = density_map[density]
    else:
        density_level = float(density)

    if density_level not in DENSITY_LEVELS:
        raise ValueError(f"density_level must be one of {sorted(DENSITY_LEVELS)}, got {density_level}")

    params = {
        "seed": seed,
        "method": "nearest",
        "fill_value": 0.0,
    }

    return density_level, layout, params


def create_uniform_density_transforms(
    seed: int = 42,
) -> list[tuple[float, str, dict[str, Any]]]:
    """Create transforms for all standard density levels (uniform layout).

    Returns list of (density, layout, params) tuples.
    """
    base = seed
    return [
        create_density_transform(1.0, "uniform", seed=base),
        create_density_transform(0.5, "uniform", seed=base + 1),
        create_density_transform(0.25, "uniform", seed=base + 2),
        create_density_transform(0.125, "uniform", seed=base + 3),
    ]


def create_sparse_density_transforms(
    seed: int = 42,
) -> list[tuple[float, str, dict[str, Any]]]:
    """Create transforms for sparse layout at standard density levels."""
    base = seed
    return [
        create_density_transform(1.0, "sparse", seed=base),
        create_density_transform(0.5, "sparse", seed=base + 1),
        create_density_transform(0.25, "sparse", seed=base + 2),
        create_density_transform(0.125, "sparse", seed=base + 3),
    ]


def create_local_high_density_transforms(
    seed: int = 42,
) -> list[tuple[float, str, dict[str, Any]]]:
    """Create transforms for local high-density layout at standard levels."""
    base = seed
    return [
        create_density_transform(1.0, "local_high_density", seed=base),
        create_density_transform(0.5, "local_high_density", seed=base + 1),
        create_density_transform(0.25, "local_high_density", seed=base + 2),
        create_density_transform(0.125, "local_high_density", seed=base + 3),
    ]
