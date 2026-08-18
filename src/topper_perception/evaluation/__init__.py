"""Dataset-agnostic evaluation: grouped cross-validation and record aggregation."""

from topper_perception.evaluation.aggregation import (
    aggregate_record_predictions,
    record_id_from_sample_id,
    record_id_from_source_path,
)
from topper_perception.evaluation.grouped import (
    PROBA_PREFIX,
    GroupFolds,
    OofResult,
    compute_metrics,
    evaluate_grouped_oof,
    generate_group_folds,
    generate_repeated_group_folds,
    oof_summary,
    per_group_metrics,
    predict,
    reduce_repeat_metrics,
    repeated_subject_metrics,
    select_best_candidate,
    select_best_model,
    snapshot_metrics_per_repeat,
    validate_group_folds,
)

__all__ = [
    "PROBA_PREFIX",
    "GroupFolds",
    "OofResult",
    "aggregate_record_predictions",
    "compute_metrics",
    "evaluate_grouped_oof",
    "generate_group_folds",
    "generate_repeated_group_folds",
    "oof_summary",
    "per_group_metrics",
    "predict",
    "record_id_from_sample_id",
    "record_id_from_source_path",
    "reduce_repeat_metrics",
    "repeated_subject_metrics",
    "select_best_candidate",
    "select_best_model",
    "snapshot_metrics_per_repeat",
    "validate_group_folds",
]
