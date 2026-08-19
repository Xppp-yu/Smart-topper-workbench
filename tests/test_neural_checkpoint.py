"""Contract tests for the PoPu neural checkpoint format (P5.2-A2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from topper_perception.neural.checkpoint import (
    build_payload,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    validate_checkpoint,
)
from topper_perception.neural.data import FROZEN_LABELS
from topper_perception.neural.models import build_model


class _Dummy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


def test_save_load_roundtrip(tmp_path) -> None:
    model = _Dummy()
    opt = _optimizer(model)
    # Advance the optimizer a step so state has non-trivial content.
    opt.zero_grad()
    model(torch.randn(2, 4)).sum().backward()
    opt.step()

    payload = build_payload(
        model=model,
        optimizer=opt,
        epoch=3,
        model_config={"name": "matrix_mlp", "params": {}},
        normalization={"mean": 1.0, "std": 2.0},
        seed=42,
        metrics={"loss": 0.5},
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, payload)
    loaded = load_checkpoint(path)
    validate_checkpoint(loaded)

    for (_, v1), (_, v2) in zip(model.state_dict().items(), loaded["model_state_dict"].items()):
        assert torch.equal(v1, v2)
    assert loaded["epoch"] == 3
    assert loaded["frozen_labels"] == list(FROZEN_LABELS)
    assert loaded["normalization"] == {"mean": 1.0, "std": 2.0}
    assert loaded["seed"] == 42
    assert loaded["metrics"] == {"loss": 0.5}


def test_load_checkpoint_is_weights_only_safe(tmp_path) -> None:
    # The checkpoint must be loadable with weights_only=True (pure tensors +
    # JSON-safe scalars), never relying on pickle deserialization.
    model = _Dummy()
    payload = build_payload(
        model=model,
        optimizer=_optimizer(model),
        epoch=0,
        model_config={"name": "matrix_mlp", "params": {}},
        normalization={"mean": 0.0, "std": 1.0},
        seed=1,
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, payload)
    # load_checkpoint itself uses weights_only=True; this asserts it succeeds.
    loaded = load_checkpoint(path)
    assert loaded["epoch"] == 0


def test_validate_rejects_missing_key() -> None:
    with pytest.raises(ValueError):
        validate_checkpoint({"version": "v0.1"})


def test_validate_rejects_label_drift(tmp_path) -> None:
    model = _Dummy()
    payload = build_payload(
        model=model,
        optimizer=_optimizer(model),
        epoch=1,
        model_config={"name": "matrix_mlp", "params": {}},
        normalization={"mean": 0.0, "std": 1.0},
        seed=1,
    )
    payload["frozen_labels"] = ["a", "b", "c", "d", "e"]
    with pytest.raises(ValueError, match="frozen_labels"):
        validate_checkpoint(payload)


def test_rng_state_roundtrip() -> None:
    torch.manual_seed(123)
    np.random.seed(123)
    state = capture_rng_state()
    assert "torch_cuda" in state
    expected_torch = torch.rand(5)
    expected_np = np.random.rand(5)

    # Rewind to the captured point and re-generate: identical sequence.
    restore_rng_state(state)
    actual_torch = torch.rand(5)
    actual_np = np.random.rand(5)

    assert torch.equal(expected_torch, actual_torch)
    assert np.array_equal(expected_np, actual_np)


def test_save_checkpoint_leaves_no_temporary_file(tmp_path) -> None:
    model = _Dummy()
    payload = build_payload(
        model=model,
        optimizer=_optimizer(model),
        epoch=0,
        model_config={"name": "matrix_mlp", "params": {}},
        normalization={"mean": 0.0, "std": 1.0},
        seed=1,
    )
    path = tmp_path / "atomic.pt"
    save_checkpoint(path, payload)
    assert path.is_file()
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_into_fresh_model_matches(tmp_path) -> None:
    model = build_model({"name": "tiny_cnn", "params": {}})
    payload = build_payload(
        model=model,
        optimizer=_optimizer(model),
        epoch=1,
        model_config={"name": "tiny_cnn", "params": {}},
        normalization={"mean": 0.0, "std": 1.0},
        seed=1,
    )
    path = tmp_path / "c.pt"
    save_checkpoint(path, payload)
    loaded = load_checkpoint(path)

    fresh = build_model({"name": "tiny_cnn", "params": {}})
    fresh.load_state_dict(loaded["model_state_dict"])
    for p1, p2 in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(p1, p2)
