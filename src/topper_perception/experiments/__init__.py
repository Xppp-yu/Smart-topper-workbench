"""Experiment governance: config contracts, artifacts, and the minimal runner."""

from topper_perception.experiments.artifacts import (
    atomic_write_json,
    capture_git_info,
    capture_system_info,
    compute_config_hash,
)
from topper_perception.experiments.contracts import (
    ConfigValidationError,
    DirtyWorktreeError,
    ExpIdError,
    State,
    StateTransitionError,
    transition,
    validate_exp_id,
    validate_experiment_config,
)
from topper_perception.experiments.runner import RunResult, run_experiment

__all__ = [
    "ConfigValidationError",
    "DirtyWorktreeError",
    "ExpIdError",
    "RunResult",
    "State",
    "StateTransitionError",
    "atomic_write_json",
    "capture_git_info",
    "capture_system_info",
    "compute_config_hash",
    "run_experiment",
    "transition",
    "validate_exp_id",
    "validate_experiment_config",
]
