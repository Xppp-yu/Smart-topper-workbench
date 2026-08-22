"""Pressure perturbation module for SLP sensor robustness testing.

This module implements reproducible, composable input perturbations for
Pressure Map data to simulate real-world sensor variations and failures.

Design rules:
* Fixed seed for reproducibility (same seed → same perturbation).
* Original input is NEVER modified (copy-on-write).
* Output includes perturbation config for auditability.
* Each perturbation has unit tests.
* Perturbations are composable via factory functions.

Perturbations implemented:
- random_noise: Additive Gaussian noise
- sensor_noise: Realistic sensor noise (thermal/quantization)
- pressure_drift: Baseline pressure drift over time
- dead_sensor: Permanently stuck at zero
- missing_sensor: Entire sensor row/column missing
- local_outlier: Single anomalous sensor reading
- left_shift: Shift all values left
- right_shift: Shift all values right
- up_shift: Shift all values up
- down_shift: Shift all values down
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Constants and Types
# ---------------------------------------------------------------------------

PERTURBATION_VERSION = "slp_pressure_perturbation_v0.1"

#: Valid perturbation type names.
PERTURBATION_TYPES = frozenset({
    "random_noise",
    "sensor_noise",
    "pressure_drift",
    "dead_sensor",
    "missing_sensor",
    "local_outlier",
    "left_shift",
    "right_shift",
    "up_shift",
    "down_shift",
})


class PerturbationType(str, Enum):
    """Enumeration of available perturbation types."""
    RANDOM_NOISE = "random_noise"
    SENSOR_NOISE = "sensor_noise"
    PRESSURE_DRIFT = "pressure_drift"
    DEAD_SENSOR = "dead_sensor"
    MISSING_SENSOR = "missing_sensor"
    LOCAL_OUTLIER = "local_outlier"
    LEFT_SHIFT = "left_shift"
    RIGHT_SHIFT = "right_shift"
    UP_SHIFT = "up_shift"
    DOWN_SHIFT = "down_shift"


@dataclass(frozen=True, slots=True)
class PerturbationConfig:
    """Configuration for a single perturbation.

    This is a frozen dataclass so configs can be hashed and compared.
    """
    perturbation_type: str
    seed: int
    params: tuple[tuple[str, Any], ...]  # Sorted key-value pairs for reproducibility

    def as_dict(self) -> dict[str, Any]:
        return {
            "perturbation_type": self.perturbation_type,
            "seed": self.seed,
            "params": dict(self.params),
        }

    @classmethod
    def create(
        cls,
        perturbation_type: str,
        seed: int,
        **params: Any,
    ) -> "PerturbationConfig":
        """Create a config with sorted params for reproducibility."""
        sorted_params = tuple(sorted(params.items()))
        return cls(
            perturbation_type=perturbation_type,
            seed=seed,
            params=sorted_params,
        )


@dataclass(frozen=True, slots=True)
class PerturbedResult:
    """Result of applying perturbations.

    Contains the perturbed data and audit information.
    """
    data: np.ndarray  # Perturbed pressure map
    config: PerturbationConfig  # Configuration used
    original_shape: tuple[int, ...]
    original_dtype: str
    original_value_range: tuple[float, float]
    perturbed_value_range: tuple[float, float]
    perturbed_fraction: float  # Fraction of pixels modified
    seed: int
    timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "data": self.data.tolist(),
            "config": self.config.as_dict(),
            "original_shape": self.original_shape,
            "original_dtype": self.original_dtype,
            "original_value_range": self.original_value_range,
            "perturbed_value_range": self.perturbed_value_range,
            "perturbed_fraction": self.perturbed_fraction,
            "seed": self.seed,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class CompositePerturbation:
    """A composition of multiple perturbations.

    Each perturbation in the chain is applied sequentially.
    """
    perturbations: tuple[PerturbationConfig, ...]
    composite_seed: int  # Master seed for the composite

    def as_dict(self) -> dict[str, Any]:
        return {
            "perturbations": [p.as_dict() for p in self.perturbations],
            "composite_seed": self.composite_seed,
            "n_perturbations": len(self.perturbations),
        }


# ---------------------------------------------------------------------------
# Perturbation Functions
# ---------------------------------------------------------------------------


def add_random_noise(
    data: np.ndarray,
    *,
    seed: int,
    std: float = 0.01,
    mean: float = 0.0,
    clip: bool = True,
) -> np.ndarray:
    """Add Gaussian noise to pressure map.

    Parameters
    ----------
    data : np.ndarray
        Input pressure map (H, W).
    seed : int
        Random seed for reproducibility.
    std : float
        Standard deviation of Gaussian noise.
    mean : float
        Mean of Gaussian noise.
    clip : bool
        If True, clip values to [0, 1].

    Returns
    -------
    np.ndarray
        Noisy pressure map (copy).
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(mean, std, size=data.shape).astype(data.dtype)

    result = data + noise

    if clip:
        result = np.clip(result, 0.0, 1.0)

    return result


def add_sensor_noise(
    data: np.ndarray,
    *,
    seed: int,
    thermal_std: float = 0.005,
    quantization_bits: int = 8,
    dead_pixel_prob: float = 0.001,
) -> np.ndarray:
    """Add realistic sensor noise.

    Simulates:
    - Thermal noise (Gaussian)
    - Quantization noise
    - Random dead pixels (stuck at 0)

    Parameters
    ----------
    data : np.ndarray
        Input pressure map.
    seed : int
        Random seed.
    thermal_std : float
        Thermal noise standard deviation.
    quantization_bits : int
        Simulated ADC bits (quantization step = 1/2^bits).
    dead_pixel_prob : float
        Probability of a pixel becoming dead (stuck at 0).

    Returns
    -------
    np.ndarray
        Noisy pressure map.
    """
    rng = np.random.default_rng(seed)

    # Thermal noise
    noise = rng.normal(0, thermal_std, size=data.shape).astype(data.dtype)
    result = data + noise

    # Quantization
    step = 1.0 / (2 ** quantization_bits)
    result = np.round(result / step) * step

    # Dead pixels
    dead_mask = rng.random(data.shape) < dead_pixel_prob
    result = np.where(dead_mask, 0.0, result)

    # Clip to valid range
    result = np.clip(result, 0.0, 1.0)

    return result


def apply_pressure_drift(
    data: np.ndarray,
    *,
    seed: int,
    drift_rate: float = 0.02,
    drift_direction: str = "up",
) -> np.ndarray:
    """Apply baseline pressure drift.

    Simulates gradual baseline shift due to:
    - Sensor settling
    - Temperature changes
    - Slow pressure redistribution

    Parameters
    ----------
    data : np.ndarray
        Input pressure map.
    seed : int
        Random seed (for spatial variation).
    drift_rate : float
        Maximum drift magnitude as fraction of local value.
    drift_direction : str
        "up", "down", or "random".

    Returns
    -------
    np.ndarray
        Pressure map with drift applied.
    """
    rng = np.random.default_rng(seed)

    # Create spatial drift pattern
    h, w = data.shape
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)

    if drift_direction == "up":
        drift_sign = 1
    elif drift_direction == "down":
        drift_sign = -1
    else:  # random
        drift_sign = 1 if rng.random() > 0.5 else -1

    # Spatial pattern (higher at edges, lower in center)
    pattern = (np.abs(xx) + np.abs(yy)) / 2
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)

    # Random coefficient
    coef = rng.uniform(0.5, 1.5)

    # Apply drift
    drift = drift_sign * drift_rate * coef * pattern
    result = data + drift

    return np.clip(result, 0.0, 1.0)


def apply_dead_sensor(
    data: np.ndarray,
    *,
    seed: int,
    row: int | None = None,
    col: int | None = None,
    region_size: tuple[int, int] | None = None,
    failure_mode: str = "stuck_zero",
) -> np.ndarray:
    """Simulate dead sensors (stuck at zero or random value).

    Parameters
    ----------
    data : np.ndarray
        Input pressure map.
    seed : int
        Random seed.
    row : int | None
        Specific row to make dead. If None, select randomly.
    col : int | None
        Specific column to make dead.
    region_size : tuple[int, int] | None
        If specified, create a rectangular dead region of this size.
    failure_mode : str
        "stuck_zero" or "stuck_random".

    Returns
    -------
    np.ndarray
        Pressure map with dead sensors.
    """
    rng = np.random.default_rng(seed)
    result = data.copy()

    h, w = data.shape

    if region_size is not None:
        # Create dead region of specified size
        rh, rw = region_size
        if row is None:
            r0 = rng.integers(0, max(1, h - rh))
        else:
            r0 = min(row, h - rh)

        if col is None:
            c0 = rng.integers(0, max(1, w - rw))
        else:
            c0 = min(col, w - rw)

        if failure_mode == "stuck_zero":
            result[r0:r0 + rh, c0:c0 + rw] = 0.0
        else:
            stuck_val = rng.random()
            result[r0:r0 + rh, c0:c0 + rw] = stuck_val

    elif row is not None:
        # Single row
        if failure_mode == "stuck_zero":
            result[min(row, h - 1), :] = 0.0
        else:
            result[min(row, h - 1), :] = rng.random()

    elif col is not None:
        # Single column
        if failure_mode == "stuck_zero":
            result[:, min(col, w - 1)] = 0.0
        else:
            result[:, min(col, w - 1)] = rng.random()

    else:
        # Random single sensor
        r = rng.integers(0, h)
        c = rng.integers(0, w)
        if failure_mode == "stuck_zero":
            result[r, c] = 0.0
        else:
            result[r, c] = rng.random()

    return result


def apply_missing_sensor(
    data: np.ndarray,
    *,
    seed: int,
    rows: Sequence[int] | None = None,
    cols: Sequence[int] | None = None,
    missing_value: float = 0.0,
) -> np.ndarray:
    """Simulate missing sensor rows/columns.

    Unlike dead_sensor, this removes entire rows or columns,
    simulating wire disconnection or complete sensor failure.

    Parameters
    ----------
    data : np.ndarray
        Input pressure map.
    seed : int
        Random seed.
    rows : Sequence[int] | None
        Rows to mark as missing.
    cols : Sequence[int] | None
        Columns to mark as missing.
    missing_value : float
        Value to fill missing regions.

    Returns
    -------
    np.ndarray
        Pressure map with missing sensors.
    """
    rng = np.random.default_rng(seed)
    result = data.copy()

    h, w = data.shape

    if rows is None and cols is None:
        # Random selection
        if rng.random() > 0.5:
            # Remove row
            n_rows = rng.integers(1, min(4, h))
            row_indices = rng.choice(h, size=n_rows, replace=False)
            for r in row_indices:
                if 0 <= r < h:
                    result[r, :] = missing_value
        else:
            # Remove column
            n_cols = rng.integers(1, min(4, w))
            col_indices = rng.choice(w, size=n_cols, replace=False)
            for c in col_indices:
                if 0 <= c < w:
                    result[:, c] = missing_value
    else:
        # Specified rows/cols
        if rows is not None:
            for r in rows:
                if 0 <= r < h:
                    result[r, :] = missing_value
        if cols is not None:
            for c in cols:
                if 0 <= c < w:
                    result[:, c] = missing_value

    return result


def apply_local_outlier(
    data: np.ndarray,
    *,
    seed: int,
    n_outliers: int = 1,
    outlier_magnitude: float = 0.5,
    radius: int = 0,
) -> np.ndarray:
    """Add local outlier(s) to simulate anomalous sensor readings.

    Parameters
    ----------
    data : np.ndarray
        Input pressure map.
    seed : int
        Random seed.
    n_outliers : int
        Number of outlier locations.
    outlier_magnitude : float
        How much to deviate from local mean (as fraction of range).
    radius : int
        Radius around outlier to also modify (creates local cluster).

    Returns
    -------
    np.ndarray
        Pressure map with outliers.
    """
    rng = np.random.default_rng(seed)
    result = data.copy()

    h, w = data.shape

    for _ in range(n_outliers):
        # Random location
        r = rng.integers(0, h)
        c = rng.integers(0, w)

        # Random sign
        sign = 1 if rng.random() > 0.5 else -1

        # Apply to center
        result[r, c] = np.clip(
            result[r, c] + sign * outlier_magnitude,
            0.0, 1.0
        )

        # Apply to neighborhood if radius > 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (dr != 0 or dc != 0):
                    # Decaying influence
                    distance = abs(dr) + abs(dc)
                    factor = 1.0 / (1 + distance)
                    result[nr, nc] = np.clip(
                        result[nr, nc] + sign * outlier_magnitude * factor * 0.5,
                        0.0, 1.0
                    )

    return result


def apply_shift(
    data: np.ndarray,
    *,
    direction: str,
    pixels: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Shift all values in a direction (wrap-around or fill).

    Parameters
    ----------
    data : np.ndarray
        Input pressure map.
    direction : str
        "left", "right", "up", "down".
    pixels : int
        Number of pixels to shift.
    fill_value : float
        Value to fill in the empty space.

    Returns
    -------
    np.ndarray
        Shifted pressure map.
    """
    if pixels == 0:
        return data.copy()

    result = data.copy()
    p = abs(pixels)

    if direction == "left":
        result = np.concatenate([result[:, p:], np.full((data.shape[0], p), fill_value)], axis=1)
    elif direction == "right":
        result = np.concatenate([np.full((data.shape[0], p), fill_value), result[:, :-p]], axis=1)
    elif direction == "up":
        result = np.concatenate([result[p:, :], np.full((p, data.shape[1]), fill_value)], axis=0)
    elif direction == "down":
        result = np.concatenate([np.full((p, data.shape[1]), fill_value), result[:-p, :]], axis=0)

    return result


# ---------------------------------------------------------------------------
# Perturbation Factory
# ---------------------------------------------------------------------------


#: Registry of perturbation functions
PERTURBATION_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "random_noise": add_random_noise,
    "sensor_noise": add_sensor_noise,
    "pressure_drift": apply_pressure_drift,
    "dead_sensor": apply_dead_sensor,
    "missing_sensor": apply_missing_sensor,
    "local_outlier": apply_local_outlier,
    "left_shift": lambda d, seed, **kw: apply_shift(d, direction="left", pixels=kw.get("pixels", 1), fill_value=kw.get("fill_value", 0.0)),
    "right_shift": lambda d, seed, **kw: apply_shift(d, direction="right", pixels=kw.get("pixels", 1), fill_value=kw.get("fill_value", 0.0)),
    "up_shift": lambda d, seed, **kw: apply_shift(d, direction="up", pixels=kw.get("pixels", 1), fill_value=kw.get("fill_value", 0.0)),
    "down_shift": lambda d, seed, **kw: apply_shift(d, direction="down", pixels=kw.get("pixels", 1), fill_value=kw.get("fill_value", 0.0)),
}


def apply_perturbation(
    data: np.ndarray,
    perturbation_type: str,
    seed: int,
    **params: Any,
) -> PerturbedResult:
    """Apply a single perturbation to pressure map.

    Parameters
    ----------
    data : np.ndarray
        Input pressure map (copy is made).
    perturbation_type : str
        Type of perturbation from PERTURBATION_TYPES.
    seed : int
        Random seed.
    **params
        Perturbation-specific parameters.

    Returns
    -------
    PerturbedResult
        Result with perturbed data and config.
    """
    if perturbation_type not in PERTURBATION_FUNCTIONS:
        raise ValueError(
            f"Unknown perturbation type: {perturbation_type}. "
            f"Valid types: {sorted(PERTURBATION_FUNCTIONS.keys())}"
        )

    original_shape = data.shape
    original_dtype = str(data.dtype)
    original_range = (float(data.min()), float(data.max()))

    # Create config
    config = PerturbationConfig.create(
        perturbation_type=perturbation_type,
        seed=seed,
        **params,
    )

    # Apply perturbation
    func = PERTURBATION_FUNCTIONS[perturbation_type]
    perturbed = func(data, seed=seed, **params)

    # Compute statistics
    perturbed_range = (float(perturbed.min()), float(perturbed.max()))
    modified_mask = data != perturbed
    perturbed_fraction = float(modified_mask.sum()) / data.size

    return PerturbedResult(
        data=perturbed,
        config=config,
        original_shape=original_shape,
        original_dtype=original_dtype,
        original_value_range=original_range,
        perturbed_value_range=perturbed_range,
        perturbed_fraction=perturbed_fraction,
        seed=seed,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def apply_composite_perturbation(
    data: np.ndarray,
    perturbations: Sequence[tuple[str, int, dict[str, Any]]],
    composite_seed: int,
) -> tuple[np.ndarray, CompositePerturbation]:
    """Apply multiple perturbations in sequence.

    Each perturbation uses its own seed for reproducibility.

    Parameters
    ----------
    data : np.ndarray
        Input pressure map.
    perturbations : Sequence[tuple[str, int, dict]]
        List of (perturbation_type, seed, params) tuples.
    composite_seed : int
        Master seed (recorded for audit).

    Returns
    -------
    tuple[np.ndarray, CompositePerturbation]
        (perturbed data, composite config)
    """
    result = data.copy()
    configs: list[PerturbationConfig] = []

    for pert_type, seed, params in perturbations:
        perturbed = apply_perturbation(result, pert_type, seed, **params)
        result = perturbed.data
        configs.append(perturbed.config)

    composite = CompositePerturbation(
        perturbations=tuple(configs),
        composite_seed=composite_seed,
    )

    return result, composite


# ---------------------------------------------------------------------------
# Preset Configurations
# ---------------------------------------------------------------------------


def create_heavy_perturbation_preset(
    seed: int,
) -> list[tuple[str, int, dict[str, Any]]]:
    """Create a preset for heavy perturbation testing.

    Combines multiple sensor failure modes.
    """
    return [
        ("sensor_noise", seed, {"thermal_std": 0.02, "quantization_bits": 6, "dead_pixel_prob": 0.01}),
        ("dead_sensor", seed + 1, {"region_size": (4, 4)}),
        ("local_outlier", seed + 2, {"n_outliers": 3, "outlier_magnitude": 0.3}),
    ]


def create_light_perturbation_preset(
    seed: int,
) -> list[tuple[str, int, dict[str, Any]]]:
    """Create a preset for light perturbation testing.

    Minor sensor noise without hardware failure.
    """
    return [
        ("random_noise", seed, {"std": 0.01}),
        ("sensor_noise", seed + 1, {"thermal_std": 0.005, "quantization_bits": 10}),
    ]


def create_degradation_preset(
    seed: int,
    degradation_level: str = "medium",
) -> list[tuple[str, int, dict[str, Any]]]:
    """Create a preset for sensor degradation testing.

    Parameters
    ----------
    seed : int
        Random seed.
    degradation_level : str
        "light", "medium", or "heavy".

    Returns
    -------
    list of perturbation tuples
    """
    if degradation_level == "light":
        return [
            ("sensor_noise", seed, {"thermal_std": 0.01, "dead_pixel_prob": 0.002}),
            ("pressure_drift", seed + 1, {"drift_rate": 0.01}),
        ]
    elif degradation_level == "heavy":
        return [
            ("sensor_noise", seed, {"thermal_std": 0.05, "dead_pixel_prob": 0.02, "quantization_bits": 6}),
            ("dead_sensor", seed + 1, {"region_size": (8, 8)}),
            ("missing_sensor", seed + 2, {"rows": [10]}),
            ("pressure_drift", seed + 3, {"drift_rate": 0.05}),
            ("local_outlier", seed + 4, {"n_outliers": 5, "outlier_magnitude": 0.4}),
        ]
    else:  # medium
        return [
            ("sensor_noise", seed, {"thermal_std": 0.02, "dead_pixel_prob": 0.005, "quantization_bits": 8}),
            ("dead_sensor", seed + 1, {"region_size": (3, 3)}),
            ("pressure_drift", seed + 2, {"drift_rate": 0.02}),
        ]
