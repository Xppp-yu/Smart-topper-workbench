"""SLP8 pressure-only non-learning region segmentation baselines (B02 v0.1).

This module implements four deterministic, CPU-only, non-learning region
segmentation baselines for the SLP_8Region_Pressure_VAL_v1.1 (8-region)
dataset.  The baselines are the lowest comparison line for the B03
(PM-only Smoke) and B04 (PM-only Mini) tasks; they are not meant to be
the best possible segmentation.

Baselines implemented
---------------------
1. ``AllBackgroundBaseline`` — every pixel is BACKGROUND (class 0).
   Used as a metric sanity floor only; it is NEVER a candidate.

2. ``TrainSpatialPriorBaseline`` — per-pixel class probability template
   fit on TRAIN labels, then ``argmax`` at predict time.  The template
   is fitted exactly once on TRAIN, then frozen; VAL/TEST are evaluated
   with the frozen template only.  No re-fitting is allowed after the
   first TRAIN fit.

3. ``PressureBodyAxisPartitionBaseline`` — body axis / longitudinal
   partition built solely from pressure-derived contact evidence:
       * contact mask via pressure > threshold (TRAIN-only fit)
       * centroid of contact mask
       * principal axis of contact mask via PCA
       * deterministic longitudinal segment allocation along the axis
       * fixed left/right split, fixed head/torso/leg ordering

4. ``PressureAxisContactIntersectionBaseline`` — intersects the
   frozen template regions with the pressure contact evidence:
       * contact evidence wins on overlap (per-class priority list)
       * empty contact → all BACKGROUND
       * degenerate PCA → fall back to ``TrainSpatialPriorBaseline``
       * out-of-bounds axis is clipped deterministically

No region label is ever used as a per-sample predictor input.  Posture
is a stratification field only and is NOT consumed by any predictor.

The B02 v0.1 contract is enforced by this module:

* Inputs are limited to ``pressure.npy`` and B01 TRAIN-only normalisation
  statistics.  Region labels, one-hot masks, ``points.csv`` field
  ``region_id``, ``class_ids_present``, ``background_pixel_count``,
  ``body_pixel_count`` and any TEST field are never consumed as
  predictor input.
* The TRAIN template is fitted once via
  :func:`fit_train_spatial_prior` and frozen.  Calling the fit
  function twice raises an error unless the caller explicitly resets
  with the documented ``reset=True`` flag (used in tests).
* The TEST access policy of B01 is forwarded: the baseline module
  MUST be supplied with TRAIN + VAL rows only, never TEST.
* The output of every ``predict`` is a label map of shape
  ``(192, 84)`` and dtype ``np.uint8`` with values in
  ``{0, 1, 2, 3, 4, 5, 6, 7, 8}`` (i.e. the 9 SLP8 label IDs).

GT provenance (DO NOT REWRITE):

* ``annotation_provenance = V221_CORRECTED_SUPPORT_AUTO_ACCEPTED``
* ``source_review_status  = NOT_REVIEWED``
* Not human pixel-level semantic masks; not medical / skin-interface
  stress / product ground truth.
* Pressure values are raw PMarray response semantics, never kPa.
* danaLab only, uncover only; do not extrapolate to cover1/cover2
  or to product / hardware / comfort claims.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

BASELINE_VERSION: str = "slp8_non_learning_v0.1"

#: Pressure matrix shape (height, width).
PRESSURE_SHAPE: tuple[int, int] = (192, 84)

#: Background label ID.
BACKGROUND_ID: int = 0

#: SLP8 region ID list (must match the A09R schema 1:1).
REGION_IDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
REGION_NAMES: tuple[str, ...] = (
    "HEAD_NECK",
    "SHOULDER",
    "THORAX_BACK",
    "LUMBAR_WAIST",
    "PELVIS_HIP",
    "ARM",
    "THIGH",
    "LOWER_LEG_FOOT",
)
REGION_ID_TO_NAME: dict[int, str] = dict(zip(REGION_IDS, REGION_NAMES))
REGION_NAME_TO_ID: dict[str, int] = {n: i for i, n in zip(REGION_IDS, REGION_NAMES)}

#: Default config version recorded in every artifact.
DEFAULT_CONFIG_VERSION: str = "slp8_non_learning_v0.1"

#: TRAIN-only fit marker used in the config.
FIT_SPLIT_TRAIN: str = "train"

#: Default deterministic seed for any RNG; baselines are fully
#: deterministic and never call a random source, but the field exists
#: so test fixtures can pin determinism.
DEFAULT_SEED: int = 20260825

#: Default contact-mask threshold fraction of the TRAIN mean pressure.
#: 5% of the mean is a conservative value that suppresses sensor noise
#: and gravity-induced body-zone bleed while still capturing the body
#: contact area on a typical SLP8 sample.
DEFAULT_CONTACT_FRACTION: float = 0.05

#: Default smoothing iterations used to denoise the contact mask.
#: Kept small (1) — additional smoothing is intentionally avoided so
#: that the contact mask remains a faithful representation of the raw
#: pressure geometry, not a pre-segmented region.
DEFAULT_CONTACT_SMOOTH_ITERS: int = 1

#: Default longitudinal segment fractions along the body axis.  These
#: are the deterministic proportions used by
#: :class:`PressureBodyAxisPartitionBaseline` to assign the 8 region
#: IDs in a head→toe order.  The values are *NOT* learned from the
#: data — they are versioned config and are exposed for future
#: B03/B04 audit, but never tuned using VAL/TEST.
DEFAULT_SEGMENT_FRACTIONS: tuple[float, ...] = (
    0.18,  # HEAD_NECK   (0.00..0.18)
    0.07,  # SHOULDER    (0.18..0.25)
    0.13,  # THORAX_BACK (0.25..0.38)
    0.12,  # LUMBAR_WAIST(0.38..0.50)
    0.10,  # PELVIS_HIP  (0.50..0.60)
    0.10,  # ARM         (0.60..0.70)  (lateral slabs)
    0.15,  # THIGH       (0.70..0.85)
    0.15,  # LOWER_LEG_FOOT (0.85..1.00)
)

#: Default left/right lateral fraction of the contact mask width.
#: The ARM region extends from x_center - lat_half .. x_center +
#: lat_half * 0.5 (a narrower lateral strip than the body trunk
#: above).  These values are versioned config, not learned.
DEFAULT_LATERAL_HALF_WIDTH: float = 0.40

#: Default per-region priority list used by the intersection
#: baseline to resolve conflicts on overlapping regions.  Higher
#: value ⇒ higher priority.  The list is fixed and versioned.
DEFAULT_REGION_PRIORITY: tuple[int, ...] = (
    2,  # SHOULDER
    5,  # PELVIS_HIP
    4,  # LUMBAR_WAIST
    3,  # THORAX_BACK
    1,  # HEAD_NECK
    7,  # THIGH
    8,  # LOWER_LEG_FOOT
    6,  # ARM
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BaselineContractError(ValueError):
    """Base exception for B02 baseline contract violations."""

    __test__ = False


class ShapeContractError(BaselineContractError):
    """Pressure / label shape does not match the SLP8 contract."""

    __test__ = False


class DtypeContractError(BaselineContractError):
    """Pressure / label dtype does not match the SLP8 contract."""

    __test__ = False


class LabelRangeError(BaselineContractError):
    """Predicted labels are outside the SLP8 0..8 range."""

    __test__ = False


class NonFinitePressureError(BaselineContractError):
    """Pressure contains NaN/Inf — the baseline refuses to predict."""

    __test__ = False


class TrainTemplateFittedError(BaselineContractError):
    """Raised when ``fit_train_spatial_prior`` is called twice on the
    same trainer without explicit ``reset=True``."""

    __test__ = False


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _validate_pressure(pressure: np.ndarray) -> np.ndarray:
    """Validate pressure shape, dtype, and finiteness.  Returns the
    input as a contiguous float64 array."""
    arr = np.asarray(pressure)
    if arr.shape != PRESSURE_SHAPE:
        raise ShapeContractError(
            f"pressure shape {arr.shape} != expected {PRESSURE_SHAPE}"
        )
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise NonFinitePressureError(
            f"pressure contains non-finite values (NaN/Inf); "
            f"non_finite_count={int(np.sum(~np.isfinite(arr)))}"
        )
    return np.ascontiguousarray(arr)


def _validate_label_map(labels: np.ndarray) -> np.ndarray:
    """Validate predicted label map shape, dtype, and value range."""
    arr = np.asarray(labels)
    if arr.shape != PRESSURE_SHAPE:
        raise ShapeContractError(
            f"label_map shape {arr.shape} != expected {PRESSURE_SHAPE}"
        )
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if int(arr.min()) < 0 or int(arr.max()) > 8:
        raise LabelRangeError(
            f"label_map values outside [0, 8]: min={int(arr.min())}, max={int(arr.max())}"
        )
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Contact mask + body axis (used by baselines 3 and 4)
# ---------------------------------------------------------------------------


def _build_contact_mask(
    pressure: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Return a boolean contact mask of shape (H, W)."""
    return pressure > float(threshold)


def _body_axis_from_contact(
    contact: np.ndarray,
) -> tuple[float, float, float, float, bool]:
    """Compute the centroid and principal axis of a boolean contact mask.

    Returns
    -------
    cx, cy : float
        Centroid of the contact mask (column, row) in (x, y) order.
    ux, uy : float
        Unit-length principal axis vector.  The direction is chosen so
        that the axis is "head-up" by convention: the component of
        the vector that is closest to "up" (smaller y in image
        coordinates) is positive.  This is the deterministic direction
        rule; no data-driven orientation is allowed.
    degenerate : bool
        True if the contact mask is empty, or has fewer than 2 active
        pixels along any row/column, in which case the principal axis
        is undefined and downstream code must fall back to the
        spatial-prior template.
    """
    if not np.any(contact):
        return 0.0, 0.0, 0.0, 1.0, True

    ys, xs = np.nonzero(contact)
    cx = float(xs.mean())
    cy = float(ys.mean())

    if xs.size < 2 or ys.size < 2:
        return cx, cy, 0.0, 1.0, True

    # Centered coordinates.
    xc = xs.astype(np.float64) - cx
    yc = ys.astype(np.float64) - cy

    # 2x2 covariance; principal axis is the eigenvector of the larger
    # eigenvalue.  This is a small symmetric problem → use a closed-form
    # eigendecomposition (no LAPACK needed).
    sxx = float(np.mean(xc * xc))
    syy = float(np.mean(yc * yc))
    sxy = float(np.mean(xc * yc))
    cov = np.array([[sxx, sxy], [sxy, syy]], dtype=np.float64)

    # Closed form eigenvalues / vectors for 2x2 symmetric matrix.
    trace = cov[0, 0] + cov[1, 1]
    det = cov[0, 0] * cov[1, 1] - cov[1, 0] * cov[0, 1]
    disc = max(trace * trace - 4.0 * det, 0.0)
    sqrt_disc = float(np.sqrt(disc))
    lam1 = 0.5 * (trace + sqrt_disc)
    # Principal eigenvector: solve (cov - lam1 * I) v = 0.
    if abs(cov[0, 0] - lam1) > abs(cov[1, 1] - lam1):
        ux = float(cov[0, 1])
        uy = float(lam1 - cov[0, 0])
    else:
        ux = float(lam1 - cov[1, 1])
        uy = float(cov[1, 0])
    norm = float(np.sqrt(ux * ux + uy * uy))
    if norm <= 0.0:
        return cx, cy, 0.0, 1.0, True
    ux /= norm
    uy /= norm

    # Deterministic head-up rule: ensure uy <= 0 (in image coords,
    # "up" is toward row 0, so a head-up axis has negative y
    # component).  If not, flip the direction.
    if uy > 0.0:
        ux = -ux
        uy = -uy

    return cx, cy, ux, uy, False


def _project_axis_lengths(
    contact: np.ndarray,
    cx: float,
    cy: float,
    ux: float,
    uy: float,
) -> tuple[float, float, float]:
    """Return (t_min, t_max, t_count) for the contact pixels along the
    principal axis.  t is the projection on (ux, uy), t_count is the
    number of contact pixels projected."""
    ys, xs = np.nonzero(contact)
    if xs.size == 0:
        return 0.0, 0.0, 0
    ts = (xs.astype(np.float64) - cx) * ux + (ys.astype(np.float64) - cy) * uy
    return float(ts.min()), float(ts.max()), int(ts.size)


# ---------------------------------------------------------------------------
# Baseline 1: all-background
# ---------------------------------------------------------------------------


class AllBackgroundBaseline:
    """All-background baseline (sanity floor only).

    Every prediction is the BACKGROUND label.  This baseline exists so
    that a metric that "scores" an empty prediction has a non-NaN
    reference value.  It is NEVER a candidate.
    """

    NAME: ClassVar[str] = "all_background"
    KIND: ClassVar[str] = "sanity_floor"

    def __init__(self) -> None:
        self._fitted: bool = True  # Always "fitted"; no state.

    # The interface signature is shared by all baselines, but this
    # baseline takes no input other than the pressure (for shape
    # validation only) — it never uses pressure values for prediction.
    def fit(self) -> None:  # noqa: D401 - inherited contract
        return None

    def predict(self, pressure: np.ndarray) -> np.ndarray:
        _validate_pressure(pressure)
        return np.zeros(PRESSURE_SHAPE, dtype=np.uint8)

    def to_state(self) -> dict[str, Any]:
        return {
            "baseline": self.NAME,
            "kind": self.KIND,
            "version": BASELINE_VERSION,
        }


# ---------------------------------------------------------------------------
# Baseline 2: train-spatial-prior template
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainSpatialPriorState:
    """Frozen state of a TRAIN-only spatial-prior template."""

    template: np.ndarray  # shape (9, H, W), float64, sums to ~1 over C
    train_count: int  # number of TRAIN samples that contributed
    train_subjects: tuple[str, ...]  # sorted subject_ids used in the fit
    class_pixel_counts: tuple[int, ...]  # per-class pixel count (9 entries)
    fit_split: str  # always "train"
    epsilon: float  # 1e-12

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": "train_spatial_prior",
            "version": BASELINE_VERSION,
            "train_count": int(self.train_count),
            "train_subjects": list(self.train_subjects),
            "class_pixel_counts": [int(x) for x in self.class_pixel_counts],
            "fit_split": self.fit_split,
            "epsilon": float(self.epsilon),
        }


class TrainSpatialPriorBaseline:
    """TRAIN-only spatial-prior template baseline.

    Fits a per-pixel class-probability template from TRAIN label maps.
    After fitting, the template is FROZEN — VAL/TEST evaluation must
    never trigger a re-fit.

    Predictions are produced by ``argmax`` of the template.  In the
    common case the template mass is concentrated on a small subset of
    classes per pixel; classes that never appear in TRAIN at a given
    pixel are guaranteed to receive 0 probability there.
    """

    NAME: ClassVar[str] = "train_spatial_prior"
    KIND: ClassVar[str] = "candidate"

    def __init__(self) -> None:
        self._state: TrainSpatialPriorState | None = None

    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    def fit(
        self,
        label_maps: Sequence[np.ndarray],
        *,
        subject_ids: Sequence[str] | None = None,
        reset: bool = False,
        epsilon: float = 1e-12,
    ) -> TrainSpatialPriorState:
        """Fit a TRAIN-only spatial prior template.

        Parameters
        ----------
        label_maps : Sequence[np.ndarray]
            One uint8 label map of shape (192, 84) per TRAIN sample.
        subject_ids : Sequence[str] | None
            Subject IDs in the same order as ``label_maps``.  Optional,
            but recommended so the fit records the subject list for
            auditability.
        reset : bool
            If True, the existing state is discarded before fitting.
            If False (default) and the trainer has already been
            fitted, a :class:`TrainTemplateFittedError` is raised.
            This protects against accidental re-fit from VAL/TEST
            code paths.
        epsilon : float
            Additive smoothing applied to the per-pixel, per-class
            accumulator.  Default 1e-12.
        """
        if self._state is not None and not reset:
            raise TrainTemplateFittedError(
                "TrainSpatialPriorBaseline.fit: trainer already fitted; "
                "use reset=True to overwrite (only allowed in tests)."
            )

        accumulator = np.zeros((9, *PRESSURE_SHAPE), dtype=np.float64)
        n = 0
        for lm in label_maps:
            arr = np.asarray(lm)
            if arr.shape != PRESSURE_SHAPE:
                raise ShapeContractError(
                    f"label_map shape {arr.shape} != expected {PRESSURE_SHAPE}"
                )
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8, copy=False)
            if int(arr.min()) < 0 or int(arr.max()) > 8:
                raise LabelRangeError(
                    f"label_map values outside [0, 8]: min={int(arr.min())}, "
                    f"max={int(arr.max())}"
                )
            for cid in range(9):
                accumulator[cid] += (arr == cid).astype(np.float64)
            n += 1
        if n == 0:
            raise BaselineContractError("fit: empty label_maps")

        # Convert to per-pixel class probability with epsilon smoothing.
        # Renormalise per-pixel so the row sums to 1.0.
        accumulator += float(epsilon)
        row_sums = accumulator.sum(axis=0, keepdims=True)
        # Avoid division by zero on rows that have zero mass (none of
        # the 9 classes ever observed) — keep them at 0/0 = 0 via
        # safe division.
        safe_sums = np.where(row_sums > 0, row_sums, 1.0)
        template = accumulator / safe_sums
        # When the row sum was 0 we set the denominator to 1.0 to avoid
        # div-by-zero; the resulting template row is also 0, which is
        # exactly what we want.

        class_pixel_counts = tuple(int(accumulator[cid].sum() - PRESSURE_SHAPE[0] * PRESSURE_SHAPE[1] * epsilon)
                                   for cid in range(9))

        if subject_ids is None:
            subjects: tuple[str, ...] = tuple()
        else:
            subjects = tuple(sorted(set(str(s) for s in subject_ids)))

        state = TrainSpatialPriorState(
            template=template,
            train_count=int(n),
            train_subjects=subjects,
            class_pixel_counts=class_pixel_counts,
            fit_split=FIT_SPLIT_TRAIN,
            epsilon=float(epsilon),
        )
        self._state = state
        return state

    def predict(self, pressure: np.ndarray) -> np.ndarray:
        _validate_pressure(pressure)
        if self._state is None:
            raise BaselineContractError(
                "TrainSpatialPriorBaseline.predict: trainer has not been fitted"
            )
        template = self._state.template
        argmax = np.argmax(template, axis=0).astype(np.uint8)
        # argmax returns 0 when a row is all zero; this is the desired
        # behaviour for the "no TRAIN evidence" case.  No further
        # masking is applied.
        return _validate_label_map(argmax)

    @property
    def state(self) -> TrainSpatialPriorState:
        if self._state is None:
            raise BaselineContractError(
                "TrainSpatialPriorBaseline.state: trainer has not been fitted"
            )
        return self._state

    def to_state(self) -> dict[str, Any]:
        if self._state is None:
            return {"baseline": self.NAME, "kind": self.KIND, "fitted": False}
        d = self._state.to_dict()
        d["kind"] = self.KIND
        d["template_sha256"] = _array_sha256(self._state.template)
        d["template_shape"] = list(self._state.template.shape)
        return d


# ---------------------------------------------------------------------------
# Body axis partition (used by baselines 3 and 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AxisPartitionConfig:
    """Versioned config for the body-axis partition baseline."""

    contact_fraction: float = DEFAULT_CONTACT_FRACTION
    contact_smooth_iters: int = DEFAULT_CONTACT_SMOOTH_ITERS
    segment_fractions: tuple[float, ...] = DEFAULT_SEGMENT_FRACTIONS
    lateral_half_width: float = DEFAULT_LATERAL_HALF_WIDTH
    region_priority: tuple[int, ...] = DEFAULT_REGION_PRIORITY
    background_id: int = BACKGROUND_ID
    region_ids: tuple[int, ...] = REGION_IDS
    fit_split: str = FIT_SPLIT_TRAIN
    version: str = BASELINE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class AxisPartitionState:
    """Frozen state of an axis partition baseline."""

    contact_threshold: float
    train_pressure_mean: float
    train_pressure_max: float
    train_sample_count: int
    fit_split: str
    config: AxisPartitionConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "contact_threshold": float(self.contact_threshold),
            "train_pressure_mean": float(self.train_pressure_mean),
            "train_pressure_max": float(self.train_pressure_max),
            "train_sample_count": int(self.train_sample_count),
            "fit_split": self.fit_split,
            "config": self.config.to_dict(),
        }


def fit_axis_partition_config(
    train_pressures: Sequence[np.ndarray],
    *,
    config: AxisPartitionConfig | None = None,
) -> AxisPartitionState:
    """Fit TRAIN-only contact-mask threshold and freeze the partition
    config for downstream baselines."""
    cfg = config or AxisPartitionConfig()
    if not train_pressures:
        raise BaselineContractError("fit_axis_partition_config: empty train_pressures")
    means: list[float] = []
    maxs: list[float] = []
    for p in train_pressures:
        arr = _validate_pressure(p)
        means.append(float(arr.mean()))
        maxs.append(float(arr.max()))
    train_pressure_mean = float(np.mean(means))
    train_pressure_max = float(np.max(maxs))
    contact_threshold = max(cfg.contact_fraction * train_pressure_mean, 1e-12)
    return AxisPartitionState(
        contact_threshold=float(contact_threshold),
        train_pressure_mean=train_pressure_mean,
        train_pressure_max=train_pressure_max,
        train_sample_count=len(train_pressures),
        fit_split=FIT_SPLIT_TRAIN,
        config=cfg,
    )


def _build_axis_partition_labels(
    pressure: np.ndarray,
    state: AxisPartitionState,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a region label map for the body axis partition.

    Returns
    -------
    labels : np.ndarray
        uint8 label map of shape (H, W).
    info : dict
        Diagnostic info (centroid, axis, length range, degenerate flag).
    """
    cfg = state.config
    contact = _build_contact_mask(pressure, threshold=state.contact_threshold)
    cx, cy, ux, uy, degenerate = _body_axis_from_contact(contact)

    info: dict[str, Any] = {
        "centroid_xy": [cx, cy],
        "axis_uv": [ux, uy],
        "degenerate": bool(degenerate),
        "contact_pixel_count": int(contact.sum()),
    }

    if degenerate or contact.sum() < 5:
        # Empty / near-empty contact.  No regions can be claimed.
        labels = np.full(PRESSURE_SHAPE, cfg.background_id, dtype=np.uint8)
        info["fallback"] = "all_background"
        return labels, info

    t_min, t_max, _ = _project_axis_lengths(contact, cx, cy, ux, uy)
    axis_length = t_max - t_min
    info["t_min"] = t_min
    info["t_max"] = t_max
    info["axis_length"] = axis_length
    if axis_length <= 0:
        labels = np.full(PRESSURE_SHAPE, cfg.background_id, dtype=np.uint8)
        info["fallback"] = "zero_axis_length"
        return labels, info

    # Build a 2D grid of (column, row) coordinates.
    H, W = PRESSURE_SHAPE
    rows = np.arange(H, dtype=np.float64)[:, None]
    cols = np.arange(W, dtype=np.float64)[None, :]
    # Project every pixel onto the principal axis (relative to centroid).
    t = (cols - cx) * ux + (rows - cy) * uy
    # Normalise to [0, 1] along the contact axis range.  Clamp to [0, 1].
    t_norm = (t - t_min) / max(axis_length, 1e-12)
    t_norm = np.clip(t_norm, 0.0, 1.0)

    # Determine the per-pixel longitudinal segment.
    seg_fracs = np.asarray(cfg.segment_fractions, dtype=np.float64)
    if seg_fracs.sum() <= 0:
        # Empty / non-positive segment fractions → all background.
        labels = np.full(PRESSURE_SHAPE, cfg.background_id, dtype=np.uint8)
        info["fallback"] = "empty_segment_fractions"
        return labels, info
    seg_fracs = seg_fracs / seg_fracs.sum()
    cum = np.cumsum(seg_fracs)
    # For each pixel, find the first segment whose cumulative fraction
    # exceeds t_norm.  ``np.searchsorted`` is the natural fit; we use
    # the 'right' side so a pixel at exactly t_norm = 0.0 lands in
    # segment 0 and t_norm = 1.0 lands in the last segment.
    seg_idx = np.searchsorted(cum, t_norm, side="right")

    # Lateral axis: orthogonal to the principal axis.  Used for the
    # left/right (ARM) split.
    nx, ny = -uy, ux  # 90° counter-clockwise rotation
    t_perp = (cols - cx) * nx + (rows - cy) * ny
    # The lateral half-width is a fraction of the contact half-width
    # along the perpendicular direction.
    t_perp_min, t_perp_max, _ = _project_axis_lengths(contact, cx, cy, nx, ny)
    perp_length = t_perp_max - t_perp_min
    if perp_length <= 0:
        perp_norm = np.zeros_like(t_norm)
    else:
        perp_norm = (t_perp - t_perp_min) / max(perp_length, 1e-12)
        perp_norm = np.clip(perp_norm, 0.0, 1.0)
    # Normalise to [-1, 1] about the centroid.
    perp_centered = (perp_norm - 0.5) * 2.0  # in [-1, 1]
    lateral_threshold = float(cfg.lateral_half_width)
    is_lateral = np.abs(perp_centered) > lateral_threshold

    # Map segment index → region ID; HEAD_NECK=1, ..., LOWER_LEG_FOOT=8.
    seg_to_region = np.asarray(cfg.region_ids, dtype=np.uint8)
    # Default: longitudinal segment.
    labels = seg_to_region[np.clip(seg_idx, 0, len(seg_to_region) - 1)]

    # Override: pixels in the lateral slabs (|perp_centered| > thr) get
    # ARM (class 6).
    ARM_ID = 6
    labels = np.where(is_lateral, np.uint8(ARM_ID), labels)

    # Background: anything outside the contact mask stays BACKGROUND.
    labels = np.where(contact, labels, np.uint8(cfg.background_id))

    return labels.astype(np.uint8), info


# ---------------------------------------------------------------------------
# Baseline 3: pressure body axis partition
# ---------------------------------------------------------------------------


class PressureBodyAxisPartitionBaseline:
    """Body-axis partition baseline.

    The predictor uses only the pressure input:

    * contact mask via pressure > threshold (TRAIN-only fit)
    * centroid + principal axis (deterministic head-up orientation rule)
    * deterministic longitudinal segments along the axis
    * lateral override for ARM

    No region label or one-hot mask is consumed at predict time.  The
    only label data that the baseline ever sees is the (TRAIN-only)
    set of pressure arrays used to fit the contact threshold.
    """

    NAME: ClassVar[str] = "pressure_body_axis_partition"
    KIND: ClassVar[str] = "candidate"

    def __init__(self) -> None:
        self._state: AxisPartitionState | None = None

    def fit(
        self,
        train_pressures: Sequence[np.ndarray],
        *,
        config: AxisPartitionConfig | None = None,
        reset: bool = False,
    ) -> AxisPartitionState:
        if self._state is not None and not reset:
            raise TrainTemplateFittedError(
                "PressureBodyAxisPartitionBaseline.fit: baseline already fitted; "
                "use reset=True to overwrite (only allowed in tests)."
            )
        state = fit_axis_partition_config(train_pressures, config=config)
        self._state = state
        return state

    def predict(self, pressure: np.ndarray) -> np.ndarray:
        _validate_pressure(pressure)
        if self._state is None:
            raise BaselineContractError(
                "PressureBodyAxisPartitionBaseline.predict: not fitted"
            )
        labels, _info = _build_axis_partition_labels(pressure, self._state)
        return _validate_label_map(labels)

    @property
    def state(self) -> AxisPartitionState:
        if self._state is None:
            raise BaselineContractError(
                "PressureBodyAxisPartitionBaseline.state: not fitted"
            )
        return self._state

    def to_state(self) -> dict[str, Any]:
        if self._state is None:
            return {"baseline": self.NAME, "kind": self.KIND, "fitted": False}
        d = self._state.to_dict()
        d["kind"] = self.KIND
        return d


# ---------------------------------------------------------------------------
# Baseline 4: pressure axis contact intersection
# ---------------------------------------------------------------------------


class PressureAxisContactIntersectionBaseline:
    """Intersection of the frozen template regions with the pressure
    contact evidence.

    The predictor combines:

    * the frozen :class:`TrainSpatialPriorBaseline` template (per-pixel
      class probability fitted on TRAIN only, NOT re-fitted at
      predict time)
    * the pressure-derived contact mask
    * the body-axis partition labels

    Conflict resolution is deterministic and versioned:

    1. pixels outside the contact mask → BACKGROUND
    2. pixels inside the contact mask where the spatial-prior template
       argmax and the body-axis partition agree → that class
    3. pixels inside the contact mask where they disagree → the higher
       priority class wins (priority list = :data:`DEFAULT_REGION_PRIORITY`)
    4. empty / degenerate contact → fall back to the spatial-prior
       template's argmax
    5. out-of-bounds axis is clipped to the contact mask
    """

    NAME: ClassVar[str] = "pressure_axis_contact_intersection"
    KIND: ClassVar[str] = "candidate"

    def __init__(self) -> None:
        self._template_state: TrainSpatialPriorState | None = None
        self._axis_state: AxisPartitionState | None = None

    def fit(
        self,
        train_label_maps: Sequence[np.ndarray],
        train_pressures: Sequence[np.ndarray],
        *,
        config: AxisPartitionConfig | None = None,
        subject_ids: Sequence[str] | None = None,
        reset: bool = False,
        epsilon: float = 1e-12,
    ) -> tuple[TrainSpatialPriorState, AxisPartitionState]:
        if (self._template_state is not None or self._axis_state is not None) and not reset:
            raise TrainTemplateFittedError(
                "PressureAxisContactIntersectionBaseline.fit: baseline already fitted; "
                "use reset=True to overwrite (only allowed in tests)."
            )
        template_trainer = TrainSpatialPriorBaseline()
        template_state = template_trainer.fit(
            train_label_maps,
            subject_ids=subject_ids,
            reset=True,
            epsilon=epsilon,
        )
        axis_state = fit_axis_partition_config(train_pressures, config=config)
        self._template_state = template_state
        self._axis_state = axis_state
        return template_state, axis_state

    def predict(self, pressure: np.ndarray) -> np.ndarray:
        _validate_pressure(pressure)
        if self._template_state is None or self._axis_state is None:
            raise BaselineContractError(
                "PressureAxisContactIntersectionBaseline.predict: not fitted"
            )
        # Spatial prior argmax (pressure never used here).
        template_argmax = np.argmax(self._template_state.template, axis=0).astype(np.uint8)
        template_argmax = _validate_label_map(template_argmax)

        # Body axis partition.
        axis_labels, axis_info = _build_axis_partition_labels(pressure, self._axis_state)
        axis_labels = _validate_label_map(axis_labels)

        # Build the contact mask used by the axis partition (must
        # match the one used in ``_build_axis_partition_labels``).
        contact = _build_contact_mask(pressure, threshold=self._axis_state.contact_threshold)

        # Priority map: higher priority number → higher precedence.
        priority = np.zeros(9, dtype=np.int32)
        for rank, cid in enumerate(self._axis_state.config.region_priority, start=1):
            priority[cid] = rank

        # Step 1: outside contact → BACKGROUND.
        labels = np.where(contact, template_argmax, np.uint8(0))
        # Step 2: inside contact, axis_labels is the partition.
        # If the partition and template agree → that class.
        # If they disagree → the higher priority class wins.
        inside = contact
        disagree = inside & (axis_labels != template_argmax)
        template_priority = priority[template_argmax]
        axis_priority = priority[axis_labels]
        # The higher-priority class wins.  Use a stable tie-break:
        # on equal priority, the axis partition wins (it carries
        # more spatial information than the prior).
        template_wins = (
            (template_priority > axis_priority)
            | ((template_priority == axis_priority) & (template_priority == 0))
        )
        chosen = np.where(
            template_wins,
            template_argmax,
            axis_labels,
        )
        labels = np.where(disagree, chosen, labels)

        # Step 4: empty / degenerate contact → fall back to template.
        if axis_info.get("fallback") is not None or int(contact.sum()) < 5:
            labels = np.where(contact, template_argmax, np.uint8(0))

        return _validate_label_map(labels.astype(np.uint8))

    def to_state(self) -> dict[str, Any]:
        if self._template_state is None or self._axis_state is None:
            return {"baseline": self.NAME, "kind": self.KIND, "fitted": False}
        d = {
            "baseline": self.NAME,
            "kind": self.KIND,
            "version": BASELINE_VERSION,
            "template": self._template_state.to_dict(),
            "axis": self._axis_state.to_dict(),
            "template_sha256": _array_sha256(self._template_state.template),
        }
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _array_sha256(arr: np.ndarray) -> str:
    """Stable SHA-256 of a numpy array via canonical bytes."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _state_sha256(state: dict[str, Any]) -> str:
    """Stable SHA-256 of a JSON-serialisable state dict."""
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def list_baselines() -> list[dict[str, str]]:
    """Return the list of available baselines as a JSON-safe list of dicts."""
    return [
        {"name": AllBackgroundBaseline.NAME, "kind": AllBackgroundBaseline.KIND, "version": BASELINE_VERSION},
        {"name": TrainSpatialPriorBaseline.NAME, "kind": TrainSpatialPriorBaseline.KIND, "version": BASELINE_VERSION},
        {"name": PressureBodyAxisPartitionBaseline.NAME, "kind": PressureBodyAxisPartitionBaseline.KIND, "version": BASELINE_VERSION},
        {"name": PressureAxisContactIntersectionBaseline.NAME, "kind": PressureAxisContactIntersectionBaseline.KIND, "version": BASELINE_VERSION},
    ]


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "BACKGROUND_ID",
    "BASELINE_VERSION",
    "DEFAULT_CONFIG_VERSION",
    "DEFAULT_CONTACT_FRACTION",
    "DEFAULT_CONTACT_SMOOTH_ITERS",
    "DEFAULT_FOREGROUND_CLASS_IDS",
    "DEFAULT_LATERAL_HALF_WIDTH",
    "DEFAULT_REGION_PRIORITY",
    "DEFAULT_SEGMENT_FRACTIONS",
    "AllBackgroundBaseline",
    "AxisPartitionConfig",
    "AxisPartitionState",
    "BaselineContractError",
    "DtypeContractError",
    "LabelRangeError",
    "NonFinitePressureError",
    "PRESSURE_SHAPE",
    "PressureAxisContactIntersectionBaseline",
    "PressureBodyAxisPartitionBaseline",
    "REGION_IDS",
    "REGION_ID_TO_NAME",
    "REGION_NAMES",
    "REGION_NAME_TO_ID",
    "ShapeContractError",
    "TrainSpatialPriorBaseline",
    "TrainSpatialPriorState",
    "TrainTemplateFittedError",
    "fit_axis_partition_config",
    "list_baselines",
]
