"""Dataset-agnostic evaluation: grouped cross-validation and record aggregation."""

from topper_perception.evaluation.aggregation import (
    aggregate_record_predictions,
    record_id_from_sample_id,
    record_id_from_source_path,
)
from topper_perception.evaluation.grouped import (
    GroupFolds,
    OofResult,
    compute_metrics,
    evaluate_grouped_oof,
    generate_group_folds,
    generate_repeated_group_folds,
    oof_summary,
    per_group_metrics,
    select_best_model,
    validate_group_folds,
)

__all__ = [
    "GroupFolds",
    "OofResult",
    "aggregate_record_predictions",
    "compute_metrics",
    "evaluate_grouped_oof",
    "generate_group_folds",
    "generate_repeated_group_folds",
    "oof_summary",
    "per_group_metrics",
    "record_id_from_sample_id",
    "record_id_from_source_path",
    "select_best_model",
    "validate_group_folds",
]
