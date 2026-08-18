"""Calibrated linear SVM candidate.

A linear SVM is often the strongest simple linear candidate, but raw
``LinearSVC.decision_function`` scores are unbounded and not probabilities.
``CalibratedLinearSVM`` wraps ``LinearSVC`` in ``CalibratedClassifierCV`` so the
candidate exposes a normalized ``predict_proba`` that the grouped evaluator and
record aggregation can rely on, exactly like every other candidate.
"""

from __future__ import annotations

import inspect

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC


def _random_state_parameter(estimator_cls: type) -> bool:
    return "random_state" in inspect.signature(estimator_cls.__init__).parameters


class CalibratedLinearSVM(BaseEstimator, ClassifierMixin):
    """Linear SVM with Platt/sigmoid probability calibration.

    Parameters
    ----------
    C : float, default=1.0
        Regularization strength forwarded to :class:`sklearn.svm.LinearSVC`.
    max_iter : int, default=5000
        Maximum solver iterations forwarded to ``LinearSVC``.
    cv : int, default=3
        Number of internal calibration folds.
    method : str, default="sigmoid"
        Calibration method forwarded to :class:`sklearn.calibration.CalibratedClassifierCV`.
    random_state : int | None, default=None
        Seeded through to the internal calibrator and SVM for reproducibility.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 5000,
        cv: int = 3,
        method: str = "sigmoid",
        random_state: int | None = None,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.cv = cv
        self.method = method
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CalibratedLinearSVM":
        x = np.asarray(x, dtype=float)
        y = np.asarray(list(y))
        base = LinearSVC(
            C=self.C, max_iter=self.max_iter, random_state=self.random_state
        )
        kwargs: dict[str, object] = {"cv": self.cv, "method": self.method}
        if _random_state_parameter(CalibratedClassifierCV):
            kwargs["random_state"] = self.random_state
        calibrator = CalibratedClassifierCV(base, **kwargs)
        self.calibrated_ = calibrator.fit(x, y)
        self.classes_ = self.calibrated_.classes_
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.calibrated_.predict_proba(x), dtype=float)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.calibrated_.predict(x))
