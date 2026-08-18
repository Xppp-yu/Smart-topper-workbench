"""Tests for the calibrated linear SVM candidate.

A linear SVM is often the strongest simple linear candidate, but raw LinearSVC
``decision_function`` scores are not probabilities.  ``CalibratedLinearSVM``
wraps ``LinearSVC`` in ``CalibratedClassifierCV`` so every candidate exposes a
normalized ``predict_proba`` that the grouped evaluator and record aggregation
can rely on.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone

from topper_perception.models.calibrated_svm import CalibratedLinearSVM
from topper_perception.models.registry import build_model


def _separable_two_class(n: int = 120, seed: int = 0):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, 4)
    y = np.where(x[:, 0] > 0, "A", "B").astype(str)
    x[:, 0] += np.where(y == "A", 2.0, -2.0)
    return x, y


def test_calibrated_svm_fit_predict_proba_normalized() -> None:
    x, y = _separable_two_class()
    model = CalibratedLinearSVM(C=1.0, cv=3, method="sigmoid", random_state=0)
    model.fit(x, y)
    proba = model.predict_proba(x)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(model.classes_) == {"A", "B"}
    assert (model.predict(x) == model.classes_[proba.argmax(axis=1)]).all()


def test_calibrated_svm_is_cloneable() -> None:
    clone(CalibratedLinearSVM(C=1.0, random_state=1))  # must not raise


def test_calibrated_svm_builds_through_registry_pipeline() -> None:
    model = build_model(
        {
            "name": "calibrated_linear_svm",
            "version": "svm@calibrated_linear",
            "estimator": "CalibratedLinearSVM",
            "params": {"C": 1.0, "max_iter": 5000, "cv": 3, "method": "sigmoid"},
            "preprocessing": [
                {"estimator": "SimpleImputer", "params": {"strategy": "median"}},
                {"estimator": "StandardScaler", "params": {}},
            ],
        },
        random_state=0,
    )
    x, y = _separable_two_class()
    model.estimator.fit(x, y)
    proba = model.estimator.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
