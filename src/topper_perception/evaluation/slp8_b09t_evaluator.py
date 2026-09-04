"""No-TEST B09T evaluation primitives and synthetic wiring smoke."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from topper_perception.evaluation.slp_pressure_metrics import (
    DEFAULT_FOREGROUND_CLASS_IDS,
    compute_fixed_class_macro_metrics,
)


SEED_ORDER: tuple[int, ...] = (42, 123, 2026)
N_CLASSES = 9
UNKNOWN_REGION = -1


class B09TEvaluatorError(ValueError):
    """Raised when the frozen no-TEST evaluation contract is violated."""


def hard_plurality_vote(
    predictions_by_seed: Mapping[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return primary, unanimous-secondary and three-way-disagreement mask.

    Two equal hard predictions win. If all three differ, the predeclared seed
    42 prediction wins. The secondary output is UNKNOWN_REGION unless all
    three predictions agree.
    """

    if tuple(predictions_by_seed) != SEED_ORDER:
        raise B09TEvaluatorError(f"seed order must be exactly {SEED_ORDER}")
    arrays = [np.asarray(predictions_by_seed[seed]) for seed in SEED_ORDER]
    if not arrays or any(array.shape != arrays[0].shape for array in arrays):
        raise B09TEvaluatorError("all seed predictions must have identical non-empty shape")
    if arrays[0].size == 0:
        raise B09TEvaluatorError("predictions must be non-empty")
    for seed, array in zip(SEED_ORDER, arrays):
        if not np.issubdtype(array.dtype, np.integer):
            raise B09TEvaluatorError(f"seed {seed} prediction must have integer dtype")
        if not np.all((array >= 0) & (array < N_CLASSES)):
            raise B09TEvaluatorError(f"seed {seed} prediction outside class range 0..8")

    p42, p123, p2026 = arrays
    primary = np.where((p42 == p123) | (p42 == p2026), p42,
                       np.where(p123 == p2026, p123, p42)).astype(np.int64)
    unanimous = (p42 == p123) & (p42 == p2026)
    three_way = (p42 != p123) & (p42 != p2026) & (p123 != p2026)
    secondary = np.where(unanimous, p42, UNKNOWN_REGION).astype(np.int64)
    return primary, secondary, three_way


def evaluate_hard_predictions(
    labels: np.ndarray,
    predictions_by_seed: Mapping[int, np.ndarray],
    subject_ids: Sequence[str],
) -> dict[str, Any]:
    """Evaluate already-produced hard predictions without loading any data."""

    labels = np.asarray(labels)
    if labels.ndim != 3 or labels.shape[0] != len(subject_ids):
        raise B09TEvaluatorError("labels must be [N,H,W] and match subject_ids")
    if not np.issubdtype(labels.dtype, np.integer) or not np.all((labels >= 0) & (labels < N_CLASSES)):
        raise B09TEvaluatorError("labels must be integer class IDs 0..8")
    if any(np.asarray(value).shape != labels.shape for value in predictions_by_seed.values()):
        raise B09TEvaluatorError("prediction shape must match labels")

    primary, secondary, three_way = hard_plurality_vote(predictions_by_seed)
    fixed = compute_fixed_class_macro_metrics(
        list(labels), list(primary), class_ids=DEFAULT_FOREGROUND_CLASS_IDS,
        n_classes=N_CLASSES,
    )
    all_classes = compute_fixed_class_macro_metrics(
        list(labels), list(primary), class_ids=tuple(range(N_CLASSES)),
        n_classes=N_CLASSES,
    )
    per_subject: dict[str, float] = {}
    for subject in sorted(set(subject_ids)):
        indices = [index for index, value in enumerate(subject_ids) if value == subject]
        sub = compute_fixed_class_macro_metrics(
            [labels[index] for index in indices], [primary[index] for index in indices],
            class_ids=DEFAULT_FOREGROUND_CLASS_IDS, n_classes=N_CLASSES,
        )
        per_subject[subject] = float(sub.fixed_iou)

    accepted = secondary != UNKNOWN_REGION
    raw_error = primary != labels
    rejected = ~accepted
    accepted_count = int(accepted.sum())
    raw_error_count = int(raw_error.sum())
    per_region = []
    for class_id in range(N_CLASSES):
        per_region.append({
            "class_id": class_id,
            "iou": float(all_classes.per_class_iou[class_id]),
            "dice": float(all_classes.per_class_dice[class_id]),
            "precision": float(all_classes.per_class_precision[class_id]),
            "recall": float(all_classes.per_class_recall[class_id]),
            "support": int(all_classes.per_class_tp[class_id] + all_classes.per_class_fn[class_id]),
        })
    return {
        "sample_count": int(labels.shape[0]),
        "pixel_count": int(labels.size),
        "primary": {
            "pooled_fixed_foreground_macro_iou": float(fixed.fixed_iou),
            "pooled_fixed_foreground_macro_dice": float(fixed.fixed_dice),
            "background_iou": float(all_classes.per_class_iou[0]),
            "pixel_accuracy": float(fixed.pixel_accuracy),
            "per_region": per_region,
            "per_subject_fixed_foreground_macro_iou": per_subject,
            "worst_subject_fixed_foreground_macro_iou": min(per_subject.values()),
            "three_way_disagreement_pixel_count": int(three_way.sum()),
            "three_way_disagreement_pixel_fraction": float(three_way.mean()),
        },
        "optional_unanimous_reject": {
            "unknown_value": UNKNOWN_REGION,
            "unanimous_pixel_coverage": float(accepted.mean()),
            "accepted_pixel_error_rate": float((raw_error & accepted).sum() / accepted_count) if accepted_count else 0.0,
            "rejected_error_capture_rate": float((raw_error & rejected).sum() / raw_error_count) if raw_error_count else 0.0,
        },
    }


def synthetic_smoke_payload() -> dict[str, Any]:
    """Exercise majority, all-different tie-break, metrics and reject wiring."""

    labels = np.zeros((2, 192, 84), dtype=np.int64)
    labels[0, :96] = 1
    labels[1, 96:] = 2
    p42 = labels.copy()
    p123 = labels.copy()
    p2026 = labels.copy()
    p123[0, 0, 0] = 2
    p2026[0, 0, 0] = 3  # all different -> seed 42 tie-break
    p2026[1, 0, 0] = 3  # 2/3 majority remains label
    result = evaluate_hard_predictions(
        labels, {42: p42, 123: p123, 2026: p2026}, ["SYNTH_A", "SYNTH_B"]
    )
    result["mode"] = "synthetic_no_test"
    result["test_access"] = False
    result["test_rows"] = 0
    result["gpu_run"] = False
    return result
