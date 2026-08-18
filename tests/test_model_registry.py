"""Focused tests for the config-driven, dataset-agnostic model registry.

These tests pin the contract that P5.1's model definitions are declarative
(estimator + params + preprocessing in config) and that the registry turns them
into real, cloneable sklearn estimators whose params actually reach the
estimator.  No PoPu data is involved.
"""

from __future__ import annotations

import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from topper_perception.models.registry import (
    CLASSIFIER_REGISTRY,
    PREPROCESSING_REGISTRY,
    RegisteredModel,
    build_model,
    validate_model_definition,
)


def _logreg_cfg(**overrides) -> dict:
    cfg = {
        "name": "logistic_regression",
        "version": "logreg@multinomial",
        "estimator": "LogisticRegression",
        "params": {"solver": "lbfgs", "max_iter": 5000},
        "preprocessing": [
            {"estimator": "SimpleImputer", "params": {"strategy": "median"}},
            {"estimator": "StandardScaler", "params": {}},
        ],
        "role": "candidate",
    }
    cfg.update(overrides)
    return cfg


def test_config_params_reach_the_estimator() -> None:
    model = build_model(_logreg_cfg(), random_state=42)

    assert isinstance(model, RegisteredModel)
    assert model.name == "logistic_regression"
    assert model.version == "logreg@multinomial"
    classifier = model.estimator.named_steps["classifier"]
    assert isinstance(classifier, LogisticRegression)
    assert classifier.solver == "lbfgs"
    assert classifier.max_iter == 5000


def test_preprocessing_params_reach_pipeline_steps() -> None:
    model = build_model(_logreg_cfg(), random_state=42)

    assert isinstance(model.estimator, Pipeline)
    imputer = model.estimator.named_steps["prep_0"]
    scaler = model.estimator.named_steps["prep_1"]
    assert isinstance(imputer, SimpleImputer)
    assert imputer.strategy == "median"
    assert isinstance(scaler, StandardScaler)


def test_seed_injected_only_into_estimators_that_accept_it() -> None:
    seeded = build_model(_logreg_cfg(), random_state=7)
    assert seeded.estimator.named_steps["classifier"].random_state == 7

    # No preprocessing -> bare estimator, seed still injected.
    rf = build_model(
        {
            "name": "random_forest",
            "version": "rf@n200",
            "estimator": "RandomForestClassifier",
            "params": {"n_estimators": 200, "n_jobs": 1},
        },
        random_state=7,
    )
    assert isinstance(rf.estimator, RandomForestClassifier)
    assert rf.estimator.random_state == 7
    assert rf.estimator.n_estimators == 200

    # KNeighborsClassifier has no random_state parameter; injection must not raise.
    knn = build_model(
        {
            "name": "knn",
            "version": "knn@k5",
            "estimator": "KNeighborsClassifier",
            "params": {"n_neighbors": 5},
        },
        random_state=7,
    )
    assert isinstance(knn.estimator, KNeighborsClassifier)
    assert knn.estimator.n_neighbors == 5
    assert not hasattr(knn.estimator, "random_state")


def test_supported_kinds_are_registered() -> None:
    for name in ("DummyClassifier", "LogisticRegression", "RandomForestClassifier", "KNeighborsClassifier"):
        assert name in CLASSIFIER_REGISTRY
    assert "SimpleImputer" in PREPROCESSING_REGISTRY
    assert "StandardScaler" in PREPROCESSING_REGISTRY


def test_p5_1_b_new_candidate_kinds_are_registered() -> None:
    for name in ("CentroidClassifier", "CalibratedLinearSVM", "ExtraTreesClassifier"):
        assert name in CLASSIFIER_REGISTRY


def test_unknown_estimator_fails_with_clear_error() -> None:
    with pytest.raises(ValueError, match="NotARealClassifier"):
        build_model(_logreg_cfg(estimator="NotARealClassifier"), random_state=0)


def test_unknown_preprocessing_step_fails() -> None:
    cfg = _logreg_cfg(preprocessing=[{"estimator": "MysteryScaler", "params": {}}])
    with pytest.raises(ValueError, match="MysteryScaler"):
        build_model(cfg, random_state=0)


def test_unknown_param_reaches_sklearn_and_fails() -> None:
    cfg = _logreg_cfg(params={"definitely_not_a_param": 1})
    with pytest.raises((TypeError, ValueError)):
        build_model(cfg, random_state=0)


def test_missing_required_fields_fail() -> None:
    with pytest.raises(ValueError, match="name"):
        build_model({"estimator": "LogisticRegression", "params": {}}, random_state=0)
    with pytest.raises(ValueError, match="estimator"):
        build_model({"name": "x", "params": {}}, random_state=0)


def test_validate_model_definition_is_strict() -> None:
    validate_model_definition(_logreg_cfg())  # valid config passes
    with pytest.raises(ValueError):
        validate_model_definition({"estimator": "LogisticRegression"})


def test_registry_model_is_cloneable() -> None:
    model = build_model(_logreg_cfg(), random_state=42)
    cloned = clone(model.estimator)
    assert isinstance(cloned, Pipeline)


def test_dummy_supports_seed_injection() -> None:
    model = build_model(
        {
            "name": "dummy",
            "version": "dummy@stratified",
            "estimator": "DummyClassifier",
            "params": {"strategy": "stratified"},
        },
        random_state=3,
    )
    assert isinstance(model.estimator, DummyClassifier)
    assert model.estimator.random_state == 3


def test_model_without_preprocessing_is_a_bare_estimator() -> None:
    model = build_model(
        {
            "name": "knn",
            "version": "knn@k5",
            "estimator": "KNeighborsClassifier",
            "params": {"n_neighbors": 3},
        },
        random_state=0,
    )
    assert isinstance(model.estimator, KNeighborsClassifier)
    assert not isinstance(model.estimator, Pipeline)
