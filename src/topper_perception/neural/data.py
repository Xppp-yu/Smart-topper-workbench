"""Neural-network data contract for PoPu pressure matrices (P5.2).

This module owns the supervised sample contract, subject-isolated splitting,
train-only normalization and horizontal-flip augmentation for the neural path.
It is deliberately pure NumPy and does **not** import ``torch``, so the
traditional-ML path (P5.1) can reuse the same loader/transform contract without
installing the optional ``neural`` dependency.

The matrix contract mirrors the model input ``[N, 1, 64, 27]`` float32 defined
in :mod:`topper_perception.neural.models`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from topper_perception.io.popu import POPU_POSTURES, load_tactilus_record
from topper_perception.io.popu_inventory import resolve_tactilus_root

#: Frozen label order (reused, not redefined): empty, supine, prone, left, right.
FROZEN_LABELS: tuple[str, ...] = tuple(POPU_POSTURES)

#: Frozen matrix geometry expected by every neural model (rows, columns).
MATRIX_ROWS = 64
MATRIX_COLUMNS = 27
MATRIX_CHANNELS = 1

NUM_CLASSES = len(FROZEN_LABELS)
LABEL_TO_INDEX: dict[str, int] = {
    label: index for index, label in enumerate(FROZEN_LABELS)
}
INDEX_TO_LABEL: dict[int, str] = dict(enumerate(FROZEN_LABELS))

#: Horizontal flip swaps ``left`` <-> ``right`` and leaves the other three
#: postures unchanged. Built from :data:`FROZEN_LABELS` so it cannot drift from
#: the frozen order.
_FLIP_LABEL_MAP: dict[int, int] = {index: index for index in range(NUM_CLASSES)}
_FLIP_LABEL_MAP[LABEL_TO_INDEX["left"]] = LABEL_TO_INDEX["right"]
_FLIP_LABEL_MAP[LABEL_TO_INDEX["right"]] = LABEL_TO_INDEX["left"]
_FLIP_LABEL_LUT = np.asarray(
    [_FLIP_LABEL_MAP[index] for index in range(NUM_CLASSES)], dtype=np.int64
)


@dataclass(frozen=True, slots=True)
class NeuralSample:
    """One supervised snapshot: a labeled 64x27 pressure matrix plus provenance."""

    sample_id: str
    record_id: str
    source_file: str
    subject_id: str
    posture: str
    snapshot_index: int
    matrix: NDArray[np.float32]


def build_labeled_samples(
    record_paths: Iterable[Path],
    *,
    tactilus_root: Path | None = None,
) -> list[NeuralSample]:
    """Build the deterministic supervised sample list from primary-cohort records.

    ``record_paths`` must already be restricted to the primary cohort (records
    with a reliable fixed-posture label). This function enforces the *label*
    contract: it keeps only snapshots whose ``posture`` is one of
    :data:`FROZEN_LABELS`; ``others.json`` and any other unlabeled record are
    excluded, never silently mislabeled.

    Records are visited in casefolded path order and snapshots in the loader's
    deterministic order, so the returned list is reproducible. ``tactilus_root``
    is used only to compute each sample's ``record_id`` (the source path
    relative to the dataset root, matching the P4a contract); when omitted the
    bare filename is used.
    """
    root = resolve_tactilus_root(tactilus_root) if tactilus_root is not None else None
    ordered = sorted((Path(path) for path in record_paths), key=lambda p: str(p).casefold())
    samples: list[NeuralSample] = []
    for source_file in ordered:
        source_file = source_file.expanduser()
        record_id = (
            source_file.relative_to(root).as_posix()
            if root is not None
            else source_file.name
        )
        for snapshot_index, frame in enumerate(load_tactilus_record(source_file)):
            posture = frame.posture
            if posture not in FROZEN_LABELS:
                continue
            samples.append(
                NeuralSample(
                    sample_id=f"popu-tactilus::{record_id}#frame={snapshot_index}",
                    record_id=record_id,
                    source_file=str(source_file.resolve()),
                    subject_id=frame.subject_id,
                    posture=posture,
                    snapshot_index=snapshot_index,
                    matrix=frame.values,
                )
            )
    return samples


@dataclass(frozen=True, slots=True)
class SubjectSplit:
    """A subject-disjoint partition of sample indices into train/val/test."""

    train_indices: NDArray[np.int64]
    val_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    train_subjects: tuple[str, ...]
    val_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]

    @property
    def n_train(self) -> int:
        return int(self.train_indices.size)

    @property
    def n_val(self) -> int:
        return int(self.val_indices.size)

    @property
    def n_test(self) -> int:
        return int(self.test_indices.size)


def subject_split(
    subject_ids: Sequence[str],
    *,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int | None = None,
    shuffle: bool = True,
) -> SubjectSplit:
    """Split sample indices by subject so no subject spans two splits.

    Subjects are first partitioned into disjoint train/val/test groups, then
    every sample index is assigned to the split of its subject. Because a PoPu
    record belongs to exactly one subject, this keeps a subject's snapshots and
    records together in a single split.
    """
    subjects = np.asarray([str(s) for s in subject_ids], dtype=object)
    unique = sorted({str(s) for s in subject_ids})
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum < 1.")
    if not unique:
        raise ValueError("At least one subject is required for a subject split.")
    if (val_ratio > 0 or test_ratio > 0) and len(unique) < 2:
        raise ValueError("At least two subjects are required when creating a holdout split.")

    order = list(unique)
    if shuffle:
        if seed is None:
            raise ValueError("A seed is required when shuffle=True.")
        order = [str(s) for s in np.random.RandomState(seed).permutation(order)]

    n_test = max(1, int(round(len(order) * test_ratio))) if test_ratio > 0 else 0
    n_val = max(1, int(round(len(order) * val_ratio))) if val_ratio > 0 else 0
    if n_test + n_val >= len(order):
        raise ValueError(
            "Holdout ratios leave no training subjects; reduce val_ratio/test_ratio "
            "or provide more subjects."
        )
    test_subjects = tuple(order[:n_test])
    val_subjects = tuple(order[n_test : n_test + n_val])
    train_subjects = tuple(order[n_test + n_val :])

    def _indices(chunk: tuple[str, ...]) -> NDArray[np.int64]:
        return np.flatnonzero(np.isin(subjects, np.asarray(chunk, dtype=object)))

    return SubjectSplit(
        train_indices=_indices(train_subjects),
        val_indices=_indices(val_subjects),
        test_indices=_indices(test_subjects),
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )


def validate_subject_split(split: SubjectSplit, subject_ids: Sequence[str]) -> None:
    """Assert no subject appears in two splits of ``split``."""
    subjects = np.asarray([str(s) for s in subject_ids], dtype=object)
    train = set(subjects[split.train_indices].tolist())
    val = set(subjects[split.val_indices].tolist())
    test = set(subjects[split.test_indices].tolist())
    for name, a, b in (("val", train, val), ("test", train, test), ("test", val, test)):
        overlap = a & b
        if overlap:
            raise ValueError(
                f"Subject leak between train and {name}: {sorted(overlap)}"
            )


class MatrixNormalizer:
    """Standardize pressure matrices with statistics fitted on training data only.

    The interface makes leakage impossible by construction: :meth:`fit` sees
    training matrices and nothing else, and :meth:`transform` applies those
    frozen statistics to any split. Callers must fit on train subjects and never
    refit on validation/test.
    """

    def __init__(self, *, epsilon: float = 1e-8) -> None:
        self.epsilon = float(epsilon)
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be a positive finite number.")
        self.mean_: float | None = None
        self.std_: float | None = None

    def fit(self, x: NDArray[np.floating]) -> "MatrixNormalizer":
        """Compute scalar mean/std from ``x`` (train) and freeze them."""
        array = np.asarray(x, dtype=np.float32)
        if array.size == 0:
            raise ValueError("Normalization input must not be empty.")
        if not np.isfinite(array).all():
            raise ValueError("Normalization input must be finite.")
        self.mean_ = float(np.mean(array))
        std = float(np.std(array))
        self.std_ = std if std > self.epsilon else self.epsilon
        return self

    def transform(self, x: NDArray[np.floating]) -> NDArray[np.float32]:
        """Standardize ``x`` with the statistics frozen by :meth:`fit`."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("MatrixNormalizer must be fit before transform.")
        array = np.asarray(x, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Normalization input must be finite.")
        return ((array - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, x: NDArray[np.floating]) -> NDArray[np.float32]:
        """Fit on ``x`` and return the standardized result."""
        return self.fit(x).transform(x)


def flip_labels(labels: NDArray[np.integer] | Sequence[int]) -> NDArray[np.int64]:
    """Swap ``left`` <-> ``right`` integer labels; other postures are unchanged."""
    array = np.asarray(labels, dtype=np.int64)
    if ((array < 0) | (array >= NUM_CLASSES)).any():
        raise ValueError(
            f"Label index out of range for the {NUM_CLASSES} frozen labels."
        )
    return _FLIP_LABEL_LUT[array]


def horizontal_flip(
    matrices: NDArray[np.floating],
    labels: NDArray[np.integer] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int64] | None]:
    """Flip matrices left<->right along the width axis and swap left/right labels.

    Train-only augmentation: mirrors the width (columns, the last axis), so a
    left-lying pressure map becomes a right-lying one and the label follows.
    ``empty``/``supine``/``prone`` are invariant. Validation data must call this
    with no augmentation instead.
    """
    x = np.asarray(matrices, dtype=np.float32)
    if x.ndim not in (3, 4):
        raise ValueError("Matrices must be [N, H, W] or [N, 1, H, W].")
    expected_tail = (MATRIX_ROWS, MATRIX_COLUMNS)
    if tuple(x.shape[-2:]) != expected_tail or (x.ndim == 4 and x.shape[1] != 1):
        raise ValueError(
            f"Expected matrix geometry [N, {MATRIX_ROWS}, {MATRIX_COLUMNS}] or "
            f"[N, 1, {MATRIX_ROWS}, {MATRIX_COLUMNS}], got {x.shape}."
        )
    if labels is not None and np.asarray(labels).shape != (x.shape[0],):
        raise ValueError("labels must contain exactly one label per matrix.")
    # np.flip returns a negative-stride view, which torch.from_numpy rejects.
    flipped = np.flip(x, axis=-1).copy()
    flipped_labels = flip_labels(labels) if labels is not None else None
    return flipped, flipped_labels


def to_model_input(
    samples: Sequence[NeuralSample],
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Return ``(x, y)`` in the model contract: ``x`` is ``[N, 1, 64, 27]``.

    ``x`` is float32 and ``y`` is int64 label indices in the frozen order.
    """
    matrices = []
    labels = []
    for sample in samples:
        matrix = np.asarray(sample.matrix, dtype=np.float32)
        if matrix.shape != (MATRIX_ROWS, MATRIX_COLUMNS):
            raise ValueError(
                f"Expected a {MATRIX_ROWS}x{MATRIX_COLUMNS} matrix, got {matrix.shape}."
            )
        if sample.posture not in LABEL_TO_INDEX:
            raise ValueError(f"Unknown posture label {sample.posture!r}.")
        matrices.append(matrix[np.newaxis, :, :])
        labels.append(LABEL_TO_INDEX[sample.posture])
    if not matrices:
        raise ValueError("At least one labeled sample is required.")
    x = np.stack(matrices).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return x, y
