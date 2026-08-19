"""Contract tests for the PoPu neural training loop (P5.2-A2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from topper_perception.neural.dataset import PressureDataset, build_dataloader
from topper_perception.neural.training import (
    evaluate,
    make_criterion,
    make_optimizer,
    predict,
    resolve_device,
    set_seed,
    train_epoch,
)

ROWS, COLS = 64, 27


class _Tiny(nn.Module):
    """Minimal linear probe so the loop is fast and deterministic on CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(ROWS * COLS, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(x, 1))


def _separable_dataset(n: int = 512, seed: int = 0) -> PressureDataset:
    # Cleanly separated 5-class clusters, scaled down so a single epoch on a
    # 1728-feature linear probe reduces the cross-entropy loss without diverging
    # (the flat input sums 1728 terms, so the raw scale must stay small).
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 5, size=n).astype(np.int64)
    base = (labels[:, None, None, None].astype(np.float32)) * 1.0
    mats = (base + rng.normal(0.0, 0.05, size=(n, 1, ROWS, COLS)).astype(np.float32)) / 100.0
    ids = [f"s{i}" for i in range(n)]
    return PressureDataset(mats, labels, sample_ids=ids, record_ids=ids, subject_ids=ids)


def _loader(ds: PressureDataset, batch_size: int, shuffle: bool):
    return build_dataloader(ds, batch_size=batch_size, shuffle=shuffle)


def test_set_seed_reproducible() -> None:
    set_seed(7)
    a = torch.rand(10).tolist()
    set_seed(7)
    b = torch.rand(10).tolist()
    assert a == b


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_unknown_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown device spec"):
        resolve_device("tpu")


def test_resolve_device_cuda_fails_without_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available; cannot exercise the no-CUDA failure path.")
    with pytest.raises(RuntimeError, match="cuda"):
        resolve_device("cuda")


def test_train_epoch_reduces_loss() -> None:
    ds = _separable_dataset()
    set_seed(0)
    model = _Tiny()
    opt = make_optimizer(model, lr=1e-3, weight_decay=0.0)
    crit = make_criterion()
    device = torch.device("cpu")

    before = evaluate(model, _loader(ds, 32, False), crit, device).loss
    info = train_epoch(model, _loader(ds, 32, True), opt, crit, device)
    after = evaluate(model, _loader(ds, 32, False), crit, device).loss

    assert info["samples"] == len(ds)
    assert np.isfinite(info["loss"])
    assert after < before


def test_train_epoch_amp_is_noop_on_cpu() -> None:
    ds = _separable_dataset(32)
    set_seed(1)
    model = _Tiny()
    opt = make_optimizer(model, lr=1e-3, weight_decay=0.0)
    crit = make_criterion()
    # amp_enabled=True on CPU must be downgraded to a no-op, not a GradScaler error.
    info = train_epoch(
        model, _loader(ds, 8, False), opt, crit, torch.device("cpu"), amp_enabled=True
    )
    assert info["amp_active"] is False
    assert np.isfinite(info["loss"])


def test_evaluate_outputs_probabilities_and_provenance() -> None:
    ds = _separable_dataset(8)
    set_seed(2)
    model = _Tiny().eval()
    result = evaluate(model, _loader(ds, 4, False), make_criterion(), torch.device("cpu"))
    assert result.probabilities.shape == (8, 5)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert result.predictions.shape == (8,)
    assert result.sample_ids == tuple(f"s{i}" for i in range(8))
    assert result.subject_ids == tuple(f"s{i}" for i in range(8))
    assert np.isfinite(result.logits).all()
    assert result.n_samples == 8


def test_predict_matches_argmax() -> None:
    ds = _separable_dataset(6)
    set_seed(3)
    model = _Tiny().eval()
    result = predict(model, _loader(ds, 6, False), torch.device("cpu"))
    assert np.array_equal(result.predictions, result.logits.argmax(axis=1))
    # predict() is forward-only: loss is NaN (no criterion) but accuracy is
    # still computed against the true labels.
    assert np.isnan(result.loss)
    assert not np.isnan(result.accuracy)
