"""Centroid/template classifier: the simple, explainable reference candidate.

The centroid classifier predicts the class whose training centroid is closest
to the query, and exposes ``predict_proba`` as a softmax over negative distances
so every candidate downstream (the grouped evaluator and record aggregation)
can consume calibrated-scaled, normalized class scores.  It is the P5.1-B
template baseline: intentionally naive, but fully explainable.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

_SUPPORTED_METRICS = ("euclidean", "manhattan")


class CentroidClassifier(BaseEstimator, ClassifierMixin):
    """Nearest-centroid classifier with softmax-distance probabilities.

    Parameters
    ----------
    metric : str, default="euclidean"
        Distance between a query and a class centroid.  One of "euclidean" or
        "manhattan".
    """

    def __init__(self, metric: str = "euclidean") -> None:
        self.metric = metric

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CentroidClassifier":
        if self.metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric {self.metric!r}; expected one of "
                f"{sorted(_SUPPORTED_METRICS)}"
            )
        x = np.asarray(x, dtype=float)
        y = np.asarray(list(y))
        self.classes_, counts = np.unique(y, return_counts=True)
        if len(self.classes_) < 2:
            raise ValueError("CentroidClassifier requires at least two classes")
        centroids: list[np.ndarray] = []
        for label in self.classes_:
            mask = y == label
            centroids.append(x[mask].mean(axis=0))
        self.centroids_ = np.asarray(centroids, dtype=float)
        self.class_priors_ = counts / len(y)
        self.n_features_in_ = x.shape[1]
        return self

    def _distances(self, x: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return np.sqrt(
                ((x[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
            )
        return np.abs(x[:, None, :] - self.centroids_[None, :, :]).sum(axis=2)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        distances = self._distances(x)
        scores = -distances
        # Stable softmax: shift by the row maximum so at least one term is exp(0).
        shifted = scores - scores.max(axis=1, keepdims=True)
        numerator = np.exp(shifted)
        return numerator / numerator.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        return np.asarray(self.classes_[proba.argmax(axis=1)])
