"""PyTorch Dataset and collation for PoPu pressure matrices (P5.2-A2).

This module is torch-only and imported lazily by the neural runner, so the
experiment runner and the traditional-ML path never import ``torch``.

The :class:`PressureDataset` is a *passive* container: normalization and
augmentation are applied by the caller (train-only) *before* construction, so
the Dataset itself cannot leak statistics or samples across splits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from topper_perception.neural.data import MATRIX_CHANNELS, MATRIX_COLUMNS, MATRIX_ROWS

#: Frozen model-input geometry: ``[N, 1, 64, 27]``.
MATRIX_SHAPE = (MATRIX_CHANNELS, MATRIX_ROWS, MATRIX_COLUMNS)


class PressureDataset(Dataset):
    """An indexable view of ``[N, 1, 64, 27]`` matrices with per-sample provenance.

    Each item carries the matrix, its integer label, and its ``sample_id`` /
    ``record_id`` / ``subject_id`` so predictions can be traced back to source.
    """

    def __init__(
        self,
        matrices: np.ndarray,
        labels: np.ndarray,
        *,
        sample_ids: Sequence[str],
        record_ids: Sequence[str],
        subject_ids: Sequence[str],
    ) -> None:
        matrices = np.asarray(matrices, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        if matrices.ndim != 4 or tuple(matrices.shape[1:]) != MATRIX_SHAPE:
            raise ValueError(
                f"Expected matrices of shape [N, {MATRIX_CHANNELS}, {MATRIX_ROWS}, "
                f"{MATRIX_COLUMNS}], got {matrices.shape}."
            )
        if labels.ndim != 1 or len(labels) != len(matrices):
            raise ValueError("One integer label is required per matrix.")
        if not np.isfinite(matrices).all():
            raise ValueError("Matrices must be finite.")
        for name, seq in (
            ("sample_ids", sample_ids),
            ("record_ids", record_ids),
            ("subject_ids", subject_ids),
        ):
            if len(seq) != len(matrices):
                raise ValueError(f"{name} must have one entry per matrix.")

        self.matrices = matrices
        self.labels = labels
        self.sample_ids = [str(item) for item in sample_ids]
        self.record_ids = [str(item) for item in record_ids]
        self.subject_ids = [str(item) for item in subject_ids]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "matrix": torch.from_numpy(self.matrices[index]),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.int64),
            "sample_id": self.sample_ids[index],
            "record_id": self.record_ids[index],
            "subject_id": self.subject_ids[index],
        }


def collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors and keep provenance as parallel lists."""
    return {
        "matrix": torch.stack([item["matrix"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "sample_id": [item["sample_id"] for item in batch],
        "record_id": [item["record_id"] for item in batch],
        "subject_id": [item["subject_id"] for item in batch],
    }


def build_dataloader(
    dataset: PressureDataset,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build a deterministic, single-process DataLoader.

    ``num_workers=0`` keeps shuffling driven by the global seed set via
    :func:`topper_perception.neural.training.set_seed`, so the same seed
    reproduces the same batch order.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
