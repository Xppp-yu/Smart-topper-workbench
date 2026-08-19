"""Neural-network data contract and candidate model skeletons (P5.2).

:mod:`topper_perception.neural.data` is pure NumPy (no ``torch``) so the
traditional-ML path (P5.1) can use it without installing the optional
``neural`` dependency. :mod:`topper_perception.neural.models` imports ``torch``
and is only available once that extra is installed.
"""

from topper_perception.neural.data import (  # noqa: F401
    FROZEN_LABELS,
    INDEX_TO_LABEL,
    LABEL_TO_INDEX,
    MATRIX_CHANNELS,
    MATRIX_COLUMNS,
    MATRIX_ROWS,
    NUM_CLASSES,
    MatrixNormalizer,
    NeuralSample,
    SubjectSplit,
    build_labeled_samples,
    flip_labels,
    horizontal_flip,
    subject_split,
    to_model_input,
    validate_subject_split,
)

__all__ = [
    "FROZEN_LABELS",
    "INDEX_TO_LABEL",
    "LABEL_TO_INDEX",
    "MATRIX_CHANNELS",
    "MATRIX_COLUMNS",
    "MATRIX_ROWS",
    "NUM_CLASSES",
    "MatrixNormalizer",
    "NeuralSample",
    "SubjectSplit",
    "build_labeled_samples",
    "flip_labels",
    "horizontal_flip",
    "subject_split",
    "to_model_input",
    "validate_subject_split",
]
