"""Tests for the centroid/template classifier used as an interpretable baseline.

The centroid classifier is the simple explainable reference candidate for
P5.1-B: it predicts the class whose training centroid is closest, and exposes
``predict_proba`` as a softmax over negative squared distances so the grouped
evaluator and record aggregation can consume calibrated-scaled scores.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone

from topper_perception.models.registry import build_model
from topper_perception.models.template import CentroidClassifier


def _separable_two_class(n: int = 120, seed: int = 0):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, 4)
    y = np.where(x[:, 0] > 0, "A", "B").astype(str)
    x[:, 0] += np.where(y == "A", 2.0, -2.0)
    return x, y


def test_centroid_predict_proba_is_normalized_and_finite() -> None:
    x, y = _separable_two_class()
    model = CentroidClassifier(metric="euclidean")
    model.fit(x, y)
    proba = model.predict_proba(x)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(model.classes_) == {"A", "B"}


def test_centroid_predict_is_argmax_of_scores() -> None:
    x, y = _separable_two_class()
    model = CentroidClassifier(metric="euclidean")
    model.fit(x, y)
    proba = model.predict_proba(x)
    expected = model.classes_[proba.argmax(axis=1)]
    assert (model.predict(x) == expected).all()


def test_centroid_learns_the_signal() -> None:
    x, y = _separable_two_class()
    model = CentroidClassifier(metric="euclidean")
    model.fit(x, y)
    assert (model.predict(x) == y).mean() > 0.95


def test_centroid_is_cloneable() -> None:
    clone(CentroidClassifier(metric="euclidean"))  # must not raise


def test_centroid_builds_through_registry_pipeline() -> None:
    model = build_model(
        {
            "name": "centroid",
            "version": "centroid@template",
            "estimator": "CentroidClassifier",
            "params": {"metric": "euclidean"},
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
