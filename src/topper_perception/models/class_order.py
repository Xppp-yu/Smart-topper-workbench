"""Adapter that pins a fitted classifier's ``classes_`` to an explicit order.

sklearn sorts ``classes_`` lexicographically at fit time, so a frozen artifact
whose contract is a specific label order (PoPu's ``["empty", "supine", "prone",
"left", "right"]``, the same order the OOF/record outputs use) would otherwise
expose ``predict_proba`` columns in a different, sorted order.  Wrapping the
candidate in :class:`FrozenClassOrderClassifier` makes ``model.classes_`` equal
the frozen order exactly, so metadata ``labels`` and model columns always agree.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class FrozenClassOrderClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a classifier so ``classes_`` follows ``class_order`` exactly.

    Parameters
    ----------
    estimator : estimator object
        The classifier to fit; must expose ``fit``/``predict``/``predict_proba``
        and a ``classes_`` attribute after fitting.
    class_order : sequence of str
        The exact label order the frozen artifact must expose.  Must be a
        permutation of the labels seen at fit time.

    After ``fit``, ``classes_`` equals ``class_order`` and ``predict_proba``
    columns follow that order; ``predict`` returns labels from that order.
    """

    def __init__(
        self,
        estimator: object,
        class_order: Sequence[str],
    ) -> None:
        self.estimator = estimator
        self.class_order = list(class_order)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FrozenClassOrderClassifier":
        x = np.asarray(x, dtype=float)
        y = np.asarray(list(y))
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(x, y)

        inner_labels = [str(label) for label in self.estimator_.classes_]
        order = [str(label) for label in self.class_order]
        if len(set(order)) != len(order):
            raise ValueError(
                f"class_order must not contain duplicates: {self.class_order!r}"
            )
        if set(order) != set(inner_labels):
            raise ValueError(
                f"class_order {self.class_order!r} is not a permutation of the "
                f"estimator classes {inner_labels!r}"
            )

        self._position = {
            str(label): index for index, label in enumerate(inner_labels)
        }
        self.classes_ = np.asarray(order, dtype=object)
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        inner_proba = np.asarray(self.estimator_.predict_proba(x), dtype=float)
        reordered = np.empty(
            (inner_proba.shape[0], len(self.classes_)), dtype=float
        )
        for column, label in enumerate(self.classes_):
            reordered[:, column] = inner_proba[:, self._position[str(label)]]
        return reordered

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]
