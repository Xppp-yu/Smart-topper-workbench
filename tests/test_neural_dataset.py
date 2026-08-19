"""Contract tests for the PoPu neural Dataset/collation (P5.2-A2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from topper_perception.neural.dataset import (
    PressureDataset,
    build_dataloader,
    collate_fn,
)

ROWS, COLS = 64, 27


def _dataset(n: int = 6) -> PressureDataset:
    rng = np.random.default_rng(0)
    matrices = rng.normal(size=(n, 1, ROWS, COLS)).astype(np.float32)
    labels = np.arange(n, dtype=np.int64) % 5
    return PressureDataset(
        matrices,
        labels,
        sample_ids=[f"s{i}" for i in range(n)],
        record_ids=[f"r{i}" for i in range(n)],
        subject_ids=[f"subj{i % 2}" for i in range(n)],
    )


def test_len_and_getitem_shape() -> None:
    ds = _dataset(6)
    assert len(ds) == 6
    item = ds[0]
    assert item["matrix"].shape == (1, ROWS, COLS)
    assert item["matrix"].dtype == torch.float32
    assert item["label"].dtype == torch.int64


def test_metadata_preserved() -> None:
    item = _dataset(4)[2]
    assert item["sample_id"] == "s2"
    assert item["record_id"] == "r2"
    assert item["subject_id"] == "subj0"


def test_collate_stacks_and_keeps_metadata() -> None:
    ds = _dataset(3)
    batch = collate_fn([ds[0], ds[1], ds[2]])
    assert batch["matrix"].shape == (3, 1, ROWS, COLS)
    assert batch["label"].shape == (3,)
    assert batch["sample_id"] == ["s0", "s1", "s2"]
    assert batch["subject_id"] == ["subj0", "subj1", "subj0"]


def test_dataloader_yields_batches() -> None:
    loader = build_dataloader(_dataset(5), batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3
    assert batches[0]["matrix"].shape == (2, 1, ROWS, COLS)


def test_rejects_wrong_matrix_shape() -> None:
    with pytest.raises(ValueError, match="matrices of shape"):
        PressureDataset(
            np.zeros((2, 1, 32, COLS), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
            sample_ids=["a", "b"],
            record_ids=["a", "b"],
            subject_ids=["a", "b"],
        )


def test_rejects_non_finite_matrices() -> None:
    matrices = np.full((2, 1, ROWS, COLS), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        PressureDataset(
            matrices,
            np.zeros(2, dtype=np.int64),
            sample_ids=["a", "b"],
            record_ids=["a", "b"],
            subject_ids=["a", "b"],
        )


def test_rejects_metadata_length_mismatch() -> None:
    with pytest.raises(ValueError, match="one entry per matrix"):
        PressureDataset(
            np.zeros((2, 1, ROWS, COLS), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
            sample_ids=["a", "b"],
            record_ids=["a", "b"],
            subject_ids=["a"],
        )


def test_build_dataloader_rejects_nonpositive_batch() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_dataloader(_dataset(2), batch_size=0, shuffle=False)
