"""Config-driven model construction shared across experiments."""

from topper_perception.models.calibrated_svm import CalibratedLinearSVM
from topper_perception.models.class_order import FrozenClassOrderClassifier
from topper_perception.models.registry import (
    CLASSIFIER_REGISTRY,
    PREPROCESSING_REGISTRY,
    RegisteredModel,
    build_model,
    validate_model_definition,
)
from topper_perception.models.template import CentroidClassifier

__all__ = [
    "CLASSIFIER_REGISTRY",
    "PREPROCESSING_REGISTRY",
    "RegisteredModel",
    "build_model",
    "validate_model_definition",
    "CalibratedLinearSVM",
    "CentroidClassifier",
    "FrozenClassOrderClassifier",
]
