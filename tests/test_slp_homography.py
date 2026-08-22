"""Unit tests for SLP homography math and direction-audit primitives.

These tests cover the mathematical contract only:

* matrix shape / dtype / finiteness validation
* determinant / condition number / rank / invertible flag
* round-trip identity through H followed by H^-1
* homogeneous division protection against zero denominator
* in-bounds semantics with a 0-based image grid
* direction-hypothesis metrics refusing to choose a semantic direction

They never inspect a real SLP matrix; direction confirmation belongs to the
overlay review described in the A04 stage report.
"""

from __future__ import annotations

import numpy as np
import pytest

from topper_perception.geometry.slp_homography import (
    HOMOGRAPHY_EPS,
    HomographyDiagnostics,
    apply_homography,
    direction_hypothesis_metrics,
    homography_diagnostics,
    in_bounds_mask,
    invert_homography,
    roundtrip_errors,
    validate_homography,
)


# --- helpers ---------------------------------------------------------------

def _identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _translation(tx: float, ty: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _scaling(sx: float, sy: float) -> np.ndarray:
    return np.array(
        [
            [sx, 0.0, 0.0],
            [0.0, sy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation(theta: float) -> np.ndarray:
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


# --- validate_homography ---------------------------------------------------


def test_validate_homography_accepts_well_formed_float64_matrix() -> None:
    matrix = _translation(2.0, 3.0)
    result = validate_homography(matrix)
    assert result.shape == (3, 3)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, matrix)


def test_validate_homography_casts_int_matrix_to_float64() -> None:
    matrix = np.eye(3, dtype=np.int32)
    result = validate_homography(matrix)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, np.eye(3))


def test_validate_homography_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="homography must be 3x3"):
        validate_homography(np.eye(4))


def test_validate_homography_rejects_non_finite_values() -> None:
    bad = _translation(np.nan, 0.0)
    with pytest.raises(ValueError, match="non-finite values"):
        validate_homography(bad)

    bad_inf = _translation(0.0, np.inf)
    with pytest.raises(ValueError, match="non-finite values"):
        validate_homography(bad_inf)


# --- homography_diagnostics ------------------------------------------------


def test_homography_diagnostics_identity_is_invertible() -> None:
    diag = homography_diagnostics(_identity())
    assert isinstance(diag, HomographyDiagnostics)
    assert bool(diag.invertible) is True
    assert diag.rank == 3
    assert diag.determinant == pytest.approx(1.0)
    assert diag.condition_number == pytest.approx(1.0)


def test_homography_diagnostics_as_dict_round_trip() -> None:
    diag = homography_diagnostics(_scaling(2.0, 3.0))
    expected = {
        "determinant": diag.determinant,
        "condition_number": diag.condition_number,
        "rank": diag.rank,
        "invertible": diag.invertible,
    }
    assert diag.as_dict() == expected
    assert diag.as_dict()["determinant"] == pytest.approx(6.0)


def test_homography_diagnostics_singular_matrix_is_not_invertible() -> None:
    singular = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ],
        dtype=np.float64,
    )
    diag = homography_diagnostics(singular)
    assert diag.rank == 1
    assert diag.determinant == pytest.approx(0.0)
    assert diag.invertible is False


def test_homography_diagnostics_near_singular_below_eps() -> None:
    near_singular = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, HOMOGRAPHY_EPS / 2.0],
        ],
        dtype=np.float64,
    )
    diag = homography_diagnostics(near_singular)
    assert diag.invertible is False
    assert abs(diag.determinant) <= HOMOGRAPHY_EPS


# --- invert_homography -----------------------------------------------------


def test_invert_homography_round_trip_yields_identity() -> None:
    matrix = _translation(4.0, -5.0) @ _rotation(0.3) @ _scaling(1.5, 0.75)
    inverse = invert_homography(matrix)
    product = matrix @ inverse
    np.testing.assert_allclose(product, np.eye(3), atol=1e-9)


def test_invert_homography_rejects_singular_matrix() -> None:
    singular = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="singular or numerically non-invertible"):
        invert_homography(singular)


# --- apply_homography ------------------------------------------------------


def test_apply_homography_identity_preserves_points() -> None:
    points = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, -2.5]])
    projected = apply_homography(points, _identity())
    np.testing.assert_allclose(projected, points)


def test_apply_homography_translation_shifts_points() -> None:
    points = np.array([[0.0, 0.0], [1.0, 1.0]])
    projected = apply_homography(points, _translation(10.0, -3.0))
    np.testing.assert_allclose(projected, np.array([[10.0, -3.0], [11.0, -2.0]]))


def test_apply_homography_handles_homogeneous_scaling() -> None:
    matrix = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    points = np.array([[3.0, 4.0], [-1.0, 2.5]])
    projected = apply_homography(points, matrix)
    np.testing.assert_allclose(projected, np.array([[3.0, 4.0], [-1.0, 2.5]]))


def test_apply_homography_rejects_wrong_point_shape() -> None:
    with pytest.raises(ValueError, match="points must have shape"):
        apply_homography(np.array([1.0, 2.0, 3.0]), _identity())


def test_apply_homography_rejects_non_finite_points() -> None:
    bad = np.array([[1.0, np.nan]])
    with pytest.raises(ValueError, match="points contain non-finite values"):
        apply_homography(bad, _identity())


def test_apply_homography_raises_on_zero_homogeneous_denominator() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="denominator is zero or near zero"):
        apply_homography(np.array([[1.0, 2.0]]), matrix)


def test_apply_homography_raises_on_near_zero_denominator() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, HOMOGRAPHY_EPS / 10.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="denominator is zero or near zero"):
        apply_homography(np.array([[1.0, 2.0]]), matrix)


# --- roundtrip_errors ------------------------------------------------------


def test_roundtrip_errors_zero_for_identity_matrix() -> None:
    points = np.array([[0.0, 0.0], [10.0, 20.0], [-3.0, 4.5]])
    errors = roundtrip_errors(points, _identity())
    np.testing.assert_allclose(errors, np.zeros(len(points)), atol=1e-12)


def test_roundtrip_errors_small_for_well_conditioned_chain() -> None:
    matrix = _translation(2.0, -3.0) @ _rotation(0.1) @ _scaling(1.2, 0.8)
    points = np.array([[0.0, 0.0], [10.0, 20.0], [50.0, 100.0]])
    errors = roundtrip_errors(points, matrix)
    np.testing.assert_allclose(errors, np.zeros(len(points)), atol=1e-9)


def test_roundtrip_errors_handles_empty_input() -> None:
    errors = roundtrip_errors(np.empty((0, 2)), _identity())
    assert errors.size == 0


# --- in_bounds_mask --------------------------------------------------------


def test_in_bounds_mask_corners_and_center() -> None:
    points = np.array(
        [
            [-0.1, 5.0],
            [0.0, 5.0],
            [9.9, 5.0],
            [10.0, 5.0],
            [5.0, 5.0],
            [5.0, -0.1],
            [5.0, 7.9],
            [5.0, 8.0],
        ]
    )
    mask = in_bounds_mask(points, width=10, height=8)
    np.testing.assert_array_equal(
        mask,
        np.array([False, True, True, False, True, False, True, False]),
    )


def test_in_bounds_mask_rejects_non_positive_dimensions() -> None:
    points = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError, match="width and height must be positive"):
        in_bounds_mask(points, width=0, height=10)
    with pytest.raises(ValueError, match="width and height must be positive"):
        in_bounds_mask(points, width=10, height=-1)


def test_in_bounds_mask_handles_empty_input() -> None:
    mask = in_bounds_mask(np.empty((0, 2)), width=10, height=10)
    assert mask.size == 0


# --- direction_hypothesis_metrics -----------------------------------------


def test_direction_hypothesis_metrics_identity_into_own_grid() -> None:
    points = np.array([[0.0, 0.0], [4.0, 7.0], [9.0, 9.0]])
    metrics = direction_hypothesis_metrics(
        points, _identity(), target_width=10, target_height=10
    )
    assert bool(metrics["invertible"]) is True
    assert metrics["determinant"] == pytest.approx(1.0)
    assert metrics["direction_status"] == "UNRESOLVED_REQUIRES_DOCUMENT_OR_OVERLAY_REVIEW"
    assert metrics["direct_in_bounds_rate"] == pytest.approx(1.0)
    assert metrics["inverse_in_bounds_rate"] == pytest.approx(1.0)
    assert metrics["roundtrip_mean_error"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["roundtrip_max_error"] == pytest.approx(0.0, abs=1e-12)


def test_direction_hypothesis_metrics_scaling_in_or_out_of_target() -> None:
    points = np.array([[0.0, 0.0], [10.0, 20.0], [50.0, 50.0]])
    metrics = direction_hypothesis_metrics(
        points, _scaling(2.0, 2.0), target_width=200, target_height=200
    )
    assert bool(metrics["invertible"]) is True
    assert metrics["direct_in_bounds_rate"] == pytest.approx(1.0)
    # Inverse H maps (x, y) -> (x/2, y/2); the smaller-scale points must still
    # land in the 200x200 grid, but we deliberately do not pick a direction
    # here -- the metric is diagnostic evidence, not a decision.
    assert metrics["direction_status"] == "UNRESOLVED_REQUIRES_DOCUMENT_OR_OVERLAY_REVIEW"


def test_direction_hypothesis_metrics_singular_matrix_reports_blocked() -> None:
    singular = np.zeros((3, 3), dtype=np.float64)
    metrics = direction_hypothesis_metrics(
        np.array([[1.0, 2.0]]), singular, target_width=10, target_height=10
    )
    assert bool(metrics["invertible"]) is False
    assert metrics["direction_status"] == "BLOCKED_NON_INVERTIBLE"
    assert metrics["direct_in_bounds_rate"] is None
    assert metrics["inverse_in_bounds_rate"] is None
    assert metrics["roundtrip_mean_error"] is None
    assert metrics["roundtrip_max_error"] is None


def test_direction_hypothesis_metrics_does_not_select_direction() -> None:
    """The metric must remain agnostic even when one side is fully in-bounds."""
    matrix = _scaling(0.01, 0.01)  # direct maps every point to ~(0,0)
    points = np.array([[100.0, 200.0], [400.0, 300.0]])
    metrics = direction_hypothesis_metrics(
        points, matrix, target_width=84, target_height=192
    )
    assert metrics["direct_in_bounds_rate"] == pytest.approx(1.0)
    assert metrics["direction_status"] == "UNRESOLVED_REQUIRES_DOCUMENT_OR_OVERLAY_REVIEW"