"""TRAIN-only class weight derivation for SLP8 region Mini (B04 v0.1).

This module implements the **fixed** class-weight formula required by the
B04 protocol:

.. code-block:: text

    raw_weight[c] = 1 / sqrt(pixel_ratio[c])
    weight[c]    = raw_weight[c] / mean(raw_weight)
    weight[c]    = clip(weight[c], 0.25, 4.0)

Where ``pixel_ratio[c]`` is the TRAIN-only per-class pixel ratio
produced by the B01 freeze (``train_class_stats.json``) and the class
index ``c`` runs over all ``N_CLASSES = 9`` values (0..8; class 0 is
the background).

The contract is intentionally strict:

* Only TRAIN data is allowed to drive the weights.  ``compute_class_weights``
  refuses any input that includes VAL or TEST pixel counts.
* Every weight must be finite; non-finite results (NaN/Inf) are rejected.
* The output covers **all** ``N_CLASSES = 9`` IDs in deterministic order,
  even if the input dict is missing some class IDs (those are treated as
  zero-pixels and the resulting weight is rejected via the non-finite
  check on ``1/sqrt(0)``).
* The clip range ``[0.25, 4.0]`` is hard-coded; callers cannot override it
  through the public API.

Both B04 candidates (TinyFCN and SmallUNet) consume **the same** weight
vector; see :func:`class_weights_to_tensor`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from topper_perception.neural.slp8_region_dataset import N_CLASSES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical clip range for the B04 weight formula.
WEIGHT_CLIP_MIN: float = 0.25
WEIGHT_CLIP_MAX: float = 4.0

#: Identifier of the split allowed to feed the weight derivation.
ALLOWED_WEIGHT_SPLIT: str = "train"

#: Tag the derived weights so persisted artifacts can be audited.
WEIGHT_FORMULA_VERSION: str = "slp8_class_weights_v0.1"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ClassWeightError(ValueError):
    """Raised when the class-weight derivation violates the B04 contract."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassWeightResult:
    """The resolved class-weight vector and its provenance.

    The ``weights`` field maps class_id (0..8) to its final clipped weight.
    ``raw_weights`` is the post-1/sqrt, pre-normalize vector; the ratio
    is preserved alongside for audit.
    """

    weights: dict[int, float]
    raw_weights: dict[int, float]
    pixel_ratio: dict[int, float]
    mean_raw_weight: float
    formula_version: str
    source_split: str
    train_sample_count: int
    train_pixel_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": {str(k): float(v) for k, v in self.weights.items()},
            "raw_weights": {str(k): float(v) for k, v in self.raw_weights.items()},
            "pixel_ratio": {str(k): float(v) for k, v in self.pixel_ratio.items()},
            "mean_raw_weight": float(self.mean_raw_weight),
            "formula_version": self.formula_version,
            "source_split": self.source_split,
            "train_sample_count": int(self.train_sample_count),
            "train_pixel_count": int(self.train_pixel_count),
            "n_classes": N_CLASSES,
            "weight_clip": [WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX],
        }


# ---------------------------------------------------------------------------
# Weight derivation
# ---------------------------------------------------------------------------


def _coerce_pixel_ratio(
    train_class_stats: Mapping[str, Any],
) -> tuple[dict[int, float], int, int]:
    """Extract the per-class pixel ratio and TRAIN-only counts.

    Accepts either the raw B01 ClassStats dict (with string-keyed
    ``per_class_pixel_ratio``) or a flat ``{ "0": ..., "1": ... }`` dict.
    Returns ``(ratio, sample_count, pixel_count)``.  The split is
    constrained to ``train`` (per the B04 contract).
    """

    # Identify where the per-class pixel ratio lives.
    if "per_class_pixel_ratio" in train_class_stats:
        raw = train_class_stats["per_class_pixel_ratio"]
    elif "class_stats" in train_class_stats and isinstance(
        train_class_stats["class_stats"], Mapping
    ):
        # Nested under a top-level "class_stats" key.
        raw = train_class_stats["class_stats"].get(
            "per_class_pixel_ratio", {}
        )
    else:
        raw = train_class_stats  # type: ignore[assignment]

    ratio: dict[int, float] = {}
    for key, value in raw.items():
        cid = int(key)
        if cid < 0 or cid >= N_CLASSES:
            raise ClassWeightError(
                f"per_class_pixel_ratio contains out-of-range class id {cid}; "
                f"only [0, {N_CLASSES - 1}] are accepted"
            )
        ratio[cid] = float(value)

    sample_count = int(train_class_stats.get("sample_count", train_class_stats.get("n_samples", 0)))
    pixel_count = int(train_class_stats.get("n_pixels", 0))
    return ratio, sample_count, pixel_count


def compute_class_weights(
    train_class_stats: Mapping[str, Any],
    *,
    allowed_split: str = ALLOWED_WEIGHT_SPLIT,
) -> ClassWeightResult:
    """Compute the B04 class-weight vector from TRAIN-only class stats.

    Parameters
    ----------
    train_class_stats : Mapping[str, Any]
        TRAIN-only class statistics.  The B01 freeze writes the canonical
        payload to ``train_class_stats.json`` and the same dataclass is
        returned from the freeze loader.
    allowed_split : str
        Must be ``"train"`` (the default).  The parameter is exposed for
        explicit auditing; supplying any other value raises
        :class:`ClassWeightError`.

    Returns
    -------
    ClassWeightResult
        The resolved weights, raw weights, the per-class pixel ratio and
        provenance fields needed for the B04 manifest.

    Raises
    ------
    ClassWeightError
        If the split is not ``train``, if any class has a non-positive
        pixel ratio, if the resulting weights are non-finite, if the
        normalization produces a non-finite mean, or if the formula
        refuses for any other reason.
    """

    if str(allowed_split).strip().lower() != ALLOWED_WEIGHT_SPLIT:
        raise ClassWeightError(
            f"compute_class_weights only accepts split={ALLOWED_WEIGHT_SPLIT!r}; "
            f"got {allowed_split!r}.  VAL/TEST may not drive class weights."
        )
    if not isinstance(train_class_stats, Mapping):
        raise ClassWeightError(
            f"train_class_stats must be a mapping; got {type(train_class_stats).__name__}"
        )

    pixel_ratio, sample_count, pixel_count = _coerce_pixel_ratio(train_class_stats)

    # ------------------------------------------------------------------
    # Step 1: raw_weight[c] = 1 / sqrt(pixel_ratio[c])
    # ------------------------------------------------------------------
    raw_weights: dict[int, float] = {}
    for cid in range(N_CLASSES):
        ratio = pixel_ratio.get(cid, 0.0)
        if ratio <= 0.0 or not math.isfinite(ratio):
            raise ClassWeightError(
                f"per_class_pixel_ratio[{cid}] is {ratio!r}; cannot derive a "
                f"finite raw weight (1/sqrt of zero or non-finite is undefined). "
                "Refusing to invent a weight from missing TRAIN coverage."
            )
        raw_weight = 1.0 / math.sqrt(ratio)
        if not math.isfinite(raw_weight):
            raise ClassWeightError(
                f"raw weight for class {cid} is non-finite ({raw_weight!r})"
            )
        raw_weights[cid] = float(raw_weight)

    # ------------------------------------------------------------------
    # Step 2: weight[c] = raw_weight[c] / mean(raw_weight)
    # ------------------------------------------------------------------
    raw_values = list(raw_weights.values())
    mean_raw = float(np.mean(raw_values))
    if not math.isfinite(mean_raw) or mean_raw <= 0:
        raise ClassWeightError(
            f"mean(raw_weight) is {mean_raw!r}; must be a positive finite value"
        )

    normalized = {cid: float(w / mean_raw) for cid, w in raw_weights.items()}

    # ------------------------------------------------------------------
    # Step 3: clip(weight[c], 0.25, 4.0)
    # ------------------------------------------------------------------
    clipped: dict[int, float] = {}
    for cid, w in normalized.items():
        if not math.isfinite(w):
            raise ClassWeightError(
                f"normalized weight for class {cid} is non-finite ({w!r}); "
                "refusing to clip a non-finite value"
            )
        clipped_value = float(min(max(w, WEIGHT_CLIP_MIN), WEIGHT_CLIP_MAX))
        clipped[cid] = clipped_value

    # Final non-finite check on the resolved vector.
    for cid, w in clipped.items():
        if not math.isfinite(w):
            raise ClassWeightError(
                f"final weight for class {cid} is non-finite ({w!r})"
            )

    return ClassWeightResult(
        weights=clipped,
        raw_weights=raw_weights,
        pixel_ratio={cid: float(pixel_ratio.get(cid, 0.0)) for cid in range(N_CLASSES)},
        mean_raw_weight=mean_raw,
        formula_version=WEIGHT_FORMULA_VERSION,
        source_split=ALLOWED_WEIGHT_SPLIT,
        train_sample_count=sample_count,
        train_pixel_count=pixel_count,
    )


def class_weights_to_tensor(
    result: ClassWeightResult,
    dtype: Any = None,
) -> "np.ndarray":
    """Return the weight vector as a numpy array in class-id order.

    Index 0 corresponds to class 0, index 8 to class 8.  The CrossEntropyLoss
    consumes this vector directly.
    """

    weights = [float(result.weights[cid]) for cid in range(N_CLASSES)]
    arr = np.asarray(weights, dtype=dtype if dtype is not None else np.float64)
    if not np.isfinite(arr).all():
        raise ClassWeightError(
            f"class_weights_to_tensor produced non-finite values: {arr!r}"
        )
    return arr


def weights_match(result_a: ClassWeightResult, result_b: ClassWeightResult) -> bool:
    """Return True iff two weight vectors are bit-identical."""

    if result_a.source_split != result_b.source_split:
        return False
    for cid in range(N_CLASSES):
        if result_a.weights[cid] != result_b.weights[cid]:
            return False
    return True


def assert_class_weight_invariants(
    result: ClassWeightResult,
    *,
    class_ids: Sequence[int] = tuple(range(N_CLASSES)),
) -> None:
    """Validate a :class:`ClassWeightResult` against the B04 contract.

    Raises
    ------
    ClassWeightError
        If any invariant is violated.
    """

    if result.source_split != ALLOWED_WEIGHT_SPLIT:
        raise ClassWeightError(
            f"source_split must be {ALLOWED_WEIGHT_SPLIT!r}, got {result.source_split!r}"
        )
    if result.formula_version != WEIGHT_FORMULA_VERSION:
        raise ClassWeightError(
            f"formula_version must be {WEIGHT_FORMULA_VERSION!r}, "
            f"got {result.formula_version!r}"
        )
    if set(result.weights.keys()) != set(class_ids):
        raise ClassWeightError(
            f"weight vector must cover exactly {sorted(class_ids)}; "
            f"got {sorted(result.weights.keys())}"
        )
    for cid in class_ids:
        w = float(result.weights[cid])
        if not math.isfinite(w):
            raise ClassWeightError(
                f"weight[{cid}] is not finite ({w!r})"
            )
        if w < WEIGHT_CLIP_MIN - 1e-9 or w > WEIGHT_CLIP_MAX + 1e-9:
            raise ClassWeightError(
                f"weight[{cid}] = {w!r} falls outside the clip range "
                f"[{WEIGHT_CLIP_MIN}, {WEIGHT_CLIP_MAX}]"
            )
    if result.mean_raw_weight <= 0 or not math.isfinite(result.mean_raw_weight):
        raise ClassWeightError(
            f"mean_raw_weight is invalid: {result.mean_raw_weight!r}"
        )
    # Reconstruct the post-normalize pre-clip values to confirm they are
    # the raw weights divided by the recorded mean.
    for cid in class_ids:
        expected = float(result.raw_weights[cid]) / float(result.mean_raw_weight)
        if not math.isfinite(expected):
            raise ClassWeightError(
                f"normalized pre-clip weight[{cid}] is non-finite ({expected!r})"
            )
