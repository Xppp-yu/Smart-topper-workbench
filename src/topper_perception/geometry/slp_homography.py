"""SLP homography math and direction-audit primitives.

This module intentionally separates mathematical validity from semantic
mapping-direction confirmation. A matrix can be invertible and yield small
round-trip error while still being applied in the wrong semantic direction.
Direction must therefore be confirmed by dataset documentation and/or fixed
visual overlays; model scores must never be used to back-infer the direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HOMOGRAPHY_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class HomographyDiagnostics:
    determinant: float
    condition_number: float
    rank: int
    invertible: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "determinant": self.determinant,
            "condition_number": self.condition_number,
            "rank": self.rank,
            "invertible": self.invertible,
        }


def validate_homography(matrix: np.ndarray) -> np.ndarray:
    """Return a finite float64 3x3 homography or raise ValueError."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError(f"homography must be 3x3, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("homography contains non-finite values")
    return value


def homography_diagnostics(matrix: np.ndarray) -> HomographyDiagnostics:
    """Describe matrix numerical validity without assigning mapping direction."""
    value = validate_homography(matrix)
    determinant = float(np.linalg.det(value))
    condition_number = float(np.linalg.cond(value))
    rank = int(np.linalg.matrix_rank(value))
    invertible = rank == 3 and abs(determinant) > HOMOGRAPHY_EPS and np.isfinite(condition_number)
    return HomographyDiagnostics(
        determinant=determinant,
        condition_number=condition_number,
        rank=rank,
        invertible=invertible,
    )


def invert_homography(matrix: np.ndarray) -> np.ndarray:
    """Invert a valid non-singular homography, failing closed otherwise."""
    value = validate_homography(matrix)
    diagnostics = homography_diagnostics(value)
    if not diagnostics.invertible:
        raise ValueError(
            "homography is singular or numerically non-invertible: "
            f"det={diagnostics.determinant}, rank={diagnostics.rank}, "
            f"cond={diagnostics.condition_number}"
        )
    inverse = np.linalg.inv(value)
    if not np.isfinite(inverse).all():
        raise ValueError("homography inverse contains non-finite values")
    return inverse


def apply_homography(
    points_xy: np.ndarray,
    matrix: np.ndarray,
    *,
    homogeneous_eps: float = HOMOGRAPHY_EPS,
) -> np.ndarray:
    """Project N x 2 Cartesian points through a 3x3 homography.

    Raises instead of silently producing infinities when the homogeneous
    denominator approaches zero.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points contain non-finite values")

    value = validate_homography(matrix)
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    projected = (value @ homogeneous.T).T
    denominator = projected[:, 2]
    if np.any(np.abs(denominator) <= homogeneous_eps):
        raise ValueError("homogeneous division denominator is zero or near zero")
    cartesian = projected[:, :2] / denominator[:, None]
    if not np.isfinite(cartesian).all():
        raise ValueError("homography projection produced non-finite coordinates")
    return cartesian


def roundtrip_errors(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Euclidean point errors after H followed by H^-1."""
    value = validate_homography(matrix)
    inverse = invert_homography(value)
    forward = apply_homography(points_xy, value)
    recovered = apply_homography(forward, inverse)
    original = np.asarray(points_xy, dtype=np.float64)
    return np.linalg.norm(recovered - original, axis=1)


def in_bounds_mask(points_xy: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Return whether projected coordinates lie inside a 0-based image grid."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0.0)
        & (points[:, 0] < float(width))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < float(height))
    )


def direction_hypothesis_metrics(
    points_xy: np.ndarray,
    matrix: np.ndarray,
    *,
    target_width: int,
    target_height: int,
) -> dict[str, object]:
    """Evaluate H and H^-1 as competing semantic direction hypotheses.

    The result is diagnostic evidence only. It deliberately does not select a
    direction, because in-bounds rate and round-trip math are insufficient to
    establish dataset semantics by themselves.
    """
    value = validate_homography(matrix)
    diagnostics = homography_diagnostics(value)
    if not diagnostics.invertible:
        return {
            **diagnostics.as_dict(),
            "direction_status": "BLOCKED_NON_INVERTIBLE",
            "direct_in_bounds_rate": None,
            "inverse_in_bounds_rate": None,
            "roundtrip_mean_error": None,
            "roundtrip_max_error": None,
        }

    inverse = invert_homography(value)
    direct = apply_homography(points_xy, value)
    inverse_projection = apply_homography(points_xy, inverse)
    direct_in_bounds = in_bounds_mask(direct, width=target_width, height=target_height)
    inverse_in_bounds = in_bounds_mask(
        inverse_projection,
        width=target_width,
        height=target_height,
    )
    errors = roundtrip_errors(points_xy, value)

    return {
        **diagnostics.as_dict(),
        "direction_status": "UNRESOLVED_REQUIRES_DOCUMENT_OR_OVERLAY_REVIEW",
        "direct_in_bounds_rate": float(direct_in_bounds.mean()) if direct_in_bounds.size else None,
        "inverse_in_bounds_rate": float(inverse_in_bounds.mean()) if inverse_in_bounds.size else None,
        "roundtrip_mean_error": float(errors.mean()) if errors.size else None,
        "roundtrip_max_error": float(errors.max()) if errors.size else None,
    }
