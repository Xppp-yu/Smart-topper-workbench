"""Config-driven model construction shared across experiments."""

from topper_perception.models.registry import (
    CLASSIFIER_REGISTRY,
    PREPROCESSING_REGISTRY,
    RegisteredModel,
    build_model,
    validate_model_definition,
)

__all__ = [
    "CLASSIFIER_REGISTRY",
    "PREPROCESSING_REGISTRY",
    "RegisteredModel",
    "build_model",
    "validate_model_definition",
]
