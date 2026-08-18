"""FrozenClassOrderClassifier: pins a frozen artifact's ``classes_`` order.

sklearn sorts ``classes_`` lexicographically at fit time, so a frozen PoPu model
would otherwise expose ``predict_proba`` columns in ``['empty', 'left', 'prone',
'right', 'supine']`` while the metadata claims the frozen order
``['empty', 'supine', 'prone', 'left', 'right']``.  The adapter reorders columns
so ``model.classes_`` exactly equals the frozen label order and a consumer can
trust the metadata.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from topper_perception.models.calibrated_svm import CalibratedLinearSVM
from topper_perception.models.class_order import FrozenClassOrderClassifier


def _two_class_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    x = np.vstack(
        [
            rng.normal(0.0, 0.5, size=(40, 4)) + np.array([3.0, 0.0, 0.0, 0.0]),
            rng.normal(0.0, 0.5, size=(40, 4)) + np.array([0.0, 3.0, 0.0, 0.0]),
        ]
    )
    y = np.array(["A"] * 40 + ["B"] * 40)
    return x, y


def test_fit_exposes_classes_in_frozen_order_not_sorted() -> None:
    x, y = _two_class_data()
    model = FrozenClassOrderClassifier(CalibratedLinearSVM(), class_order=["B", "A"])
    model.fit(x, y)
    assert list(model.classes_) == ["B", "A"]


def test_predict_proba_columns_follow_frozen_order() -> None:
    x, y = _two_class_data()
    inner = CalibratedLinearSVM()
    inner.fit(x, y)
    model = FrozenClassOrderClassifier(CalibratedLinearSVM(), class_order=["B", "A"])
    model.fit(x, y)

    proba = model.predict_proba(x)
    assert proba.shape == (80, 2)
    inner_proba = inner.predict_proba(x)
    position_of = {
        str(label): int(np.argwhere(np.asarray(inner.classes_) == label)[0, 0])
        for label in ["A", "B"]
    }
    # column 0 must be class "B", column 1 must be class "A" (frozen order)
    assert np.allclose(proba[:, 0], inner_proba[:, position_of["B"]])
    assert np.allclose(proba[:, 1], inner_proba[:, position_of["A"]])
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_predict_matches_argmax_of_frozen_order_proba() -> None:
    x, y = _two_class_data()
    model = FrozenClassOrderClassifier(CalibratedLinearSVM(), class_order=["B", "A"])
    model.fit(x, y)

    proba = model.predict_proba(x)
    predicted = model.predict(x)
    assert list(predicted) == [
        str(model.classes_[index]) for index in proba.argmax(axis=1)
    ]
    assert set(predicted) <= {"A", "B"}


def test_fit_rejects_class_order_that_is_not_a_permutation() -> None:
    x, y = _two_class_data()
    model = FrozenClassOrderClassifier(CalibratedLinearSVM(), class_order=["A", "Z"])
    with pytest.raises(ValueError, match="class_order"):
        model.fit(x, y)


def test_joblib_round_trip_preserves_frozen_order() -> None:
    x, y = _two_class_data()
    model = FrozenClassOrderClassifier(CalibratedLinearSVM(), class_order=["B", "A"])
    model.fit(x, y)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        joblib.dump(model, path)
        loaded = joblib.load(path)

    assert list(loaded.classes_) == ["B", "A"]
    assert np.allclose(loaded.predict_proba(x), model.predict_proba(x))


def test_adapter_inside_pipeline_exposes_frozen_classes() -> None:
    # mirrors the freeze artifact shape: Pipeline(imputer, scaler, adapter(svm))
    x, y = _two_class_data()
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        FrozenClassOrderClassifier(CalibratedLinearSVM(), class_order=["B", "A"]),
    )
    pipeline.fit(x, y)

    assert list(pipeline.classes_) == ["B", "A"]
    proba = pipeline.predict_proba(x)
    assert proba.shape == (80, 2)
    assert bool(np.isfinite(proba).all())
    assert bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-4))
