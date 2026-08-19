"""Contract tests for the PoPu neural model skeletons (P5.2-A1).

These tests verify the shared model contract on synthetic random tensors (not
the full PoPu dataset): three candidates emit ``[N, 5]`` finite logits, are
config-driven, reproducible under a fixed seed, and lightweight.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from topper_perception.neural.models import (
    INPUT_SHAPE,
    MODEL_REGISTRY,
    MatrixMLP,
    SmallResNet,
    TinyCNN,
    build_model,
    count_parameters,
    validate_model_input,
)

BATCH = 4
MODEL_CLASSES = (MatrixMLP, TinyCNN, SmallResNet)


def _input(batch: int = BATCH) -> torch.Tensor:
    return torch.randn(batch, *INPUT_SHAPE)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_models_output_batch_by_five(model_cls) -> None:
    model = model_cls().eval()
    with torch.no_grad():
        out = model(_input())
    assert out.shape == (BATCH, 5)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_models_forward_is_finite(model_cls) -> None:
    model = model_cls().eval()
    with torch.no_grad():
        out = model(_input())
    assert torch.isfinite(out).all()


def test_build_model_from_config() -> None:
    for name in ("matrix_mlp", "tiny_cnn", "small_resnet"):
        model = build_model({"name": name, "params": {}})
        assert isinstance(model, MODEL_REGISTRY[name])


def test_build_model_passes_params() -> None:
    model = build_model({"name": "tiny_cnn", "params": {"channels": (8, 16)}})
    assert model.channels == (8, 16)


def test_unknown_model_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model name"):
        build_model({"name": "resnet152"})


def test_illegal_params_raise() -> None:
    with pytest.raises(ValueError):
        build_model({"name": "matrix_mlp", "params": {"bogus": 1}})
    with pytest.raises(ValueError):
        build_model({"name": "tiny_cnn", "params": {"channels": []}})
    with pytest.raises(ValueError):
        build_model({"name": "small_resnet", "params": {"num_blocks": 0}})
    with pytest.raises(ValueError, match="params must be a mapping"):
        build_model({"name": "tiny_cnn", "params": None})


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_fixed_seed_reproducible_init(model_cls) -> None:
    torch.manual_seed(123)
    first = model_cls()
    torch.manual_seed(123)
    second = model_cls()

    for p_first, p_second in zip(first.parameters(), second.parameters()):
        assert torch.equal(p_first, p_second)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_models_lightweight_and_countable(model_cls) -> None:
    count = count_parameters(model_cls())
    assert count > 0
    assert count < 1_000_000


def test_validate_model_input_errors() -> None:
    with pytest.raises(ValueError):
        validate_model_input(torch.randn(2, 1, 32, 27))
    with pytest.raises(ValueError):
        validate_model_input(torch.full((2, 1, 64, 27), float("nan")))
    with pytest.raises(TypeError):
        validate_model_input(np.zeros((2, 1, 64, 27)))
