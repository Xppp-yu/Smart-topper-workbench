"""Config-driven, dataset-agnostic model registry.

The registry turns declarative model definitions (estimator + params +
preprocessing in a versioned config) into real, cloneable sklearn estimators.
It is the single construction path for P5.1 and later candidate models, so
parameters live in config instead of ``if/elif`` blocks in experiment modules.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from topper_perception.models.calibrated_svm import CalibratedLinearSVM
from topper_perception.models.template import CentroidClassifier

# Estimator kinds supported by the registry.  New candidates (for example a CNN
# later) register here or in an extending module; unknown kinds fail loudly.
CLASSIFIER_REGISTRY: dict[str, type] = {
    "DummyClassifier": DummyClassifier,
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "CentroidClassifier": CentroidClassifier,
    "CalibratedLinearSVM": CalibratedLinearSVM,
}

# Preprocessing steps that may appear inside a candidate's Pipeline.  Keeping
# imputation/scaling here means they are fitted per training fold, never on the
# full table, because they always live inside the Pipeline the registry builds.
PREPROCESSING_REGISTRY: dict[str, type] = {
    "SimpleImputer": SimpleImputer,
    "StandardScaler": StandardScaler,
}

# Fixed name of the final classifier step inside every built Pipeline.
CLASSIFIER_STEP = "classifier"

REQUIRED_FIELDS = ("name", "version", "estimator")


@dataclass(frozen=True)
class RegisteredModel:
    """A named candidate model plus its frozen version string and estimator."""

    name: str
    version: str
    estimator: Any
    role: str = "candidate"


def validate_model_definition(model_cfg: Mapping[str, Any]) -> None:
    """Fail loudly when a model definition is missing fields or names unknown kinds."""
    missing = [field for field in REQUIRED_FIELDS if not model_cfg.get(field)]
    if missing:
        raise ValueError(
            f"Model config missing required fields {missing}; got keys "
            f"{sorted(str(key) for key in model_cfg)}"
        )
    estimator_name = str(model_cfg["estimator"])
    if estimator_name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown estimator {estimator_name!r}; known: {sorted(CLASSIFIER_REGISTRY)}"
        )
    for step in model_cfg.get("preprocessing", []):
        prep_name = str(step["estimator"])
        if prep_name not in PREPROCESSING_REGISTRY:
            raise ValueError(
                f"Unknown preprocessing estimator {prep_name!r}; "
                f"known: {sorted(PREPROCESSING_REGISTRY)}"
            )


def _accepts_random_state(estimator_cls: type) -> bool:
    return "random_state" in inspect.signature(estimator_cls.__init__).parameters


def _build_step(
    estimator_cls: type, params: Mapping[str, Any], *, random_state: int | None
) -> Any:
    """Instantiate one estimator, injecting the unified seed when supported.

    Parameters are passed verbatim from config; an unknown parameter surfaces as
    the sklearn ``TypeError`` instead of being silently dropped.
    """
    kwargs = dict(params)
    if (
        random_state is not None
        and _accepts_random_state(estimator_cls)
        and "random_state" not in kwargs
    ):
        kwargs["random_state"] = random_state
    return estimator_cls(**kwargs)


def build_model(
    model_cfg: Mapping[str, Any], *, random_state: int | None = None
) -> RegisteredModel:
    """Build a :class:`RegisteredModel` from a config-driven definition.

    When ``preprocessing`` is present the imputer/scaler steps are always
    packaged inside a ``Pipeline`` ahead of the classifier, so every fold's
    preprocessing is fitted on that fold's training subjects only.  With no
    preprocessing the raw estimator is returned.
    """
    validate_model_definition(model_cfg)
    name = str(model_cfg["name"])
    version = str(model_cfg["version"])
    role = str(model_cfg.get("role", "candidate"))

    estimator_cls = CLASSIFIER_REGISTRY[str(model_cfg["estimator"])]
    classifier = _build_step(estimator_cls, model_cfg.get("params", {}), random_state=random_state)

    preprocessing = list(model_cfg.get("preprocessing", []))
    if not preprocessing:
        return RegisteredModel(name, version, classifier, role=role)

    steps = [
        (
            f"prep_{index}",
            _build_step(
                PREPROCESSING_REGISTRY[str(step["estimator"])],
                step.get("params", {}),
                random_state=None,
            ),
        )
        for index, step in enumerate(preprocessing)
    ]
    steps.append((CLASSIFIER_STEP, classifier))
    return RegisteredModel(name, version, Pipeline(steps), role=role)
