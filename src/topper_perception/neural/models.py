"""Candidate neural model skeletons for PoPu pressure matrices (P5.2).

Three config-driven ``torch.nn.Module`` candidates share one contract:

- input  ``[N, 1, 64, 27]`` float32 pressure matrix;
- output ``[N, 5]`` logits in the frozen label order (empty, supine, prone,
  left, right);
- no pretrained weights, no external downloads.

Importing this module requires the optional ``neural`` dependency (torch). The
models are intentionally lightweight for a 64x27 pressure map; ``SmallResNet``
is a small residual stack, not an ImageNet-scale backbone.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from topper_perception.neural.data import (
    MATRIX_CHANNELS,
    MATRIX_COLUMNS,
    MATRIX_ROWS,
    NUM_CLASSES,
)

#: Frozen input geometry: one pressure channel, 64 rows, 27 columns.
INPUT_SHAPE = (MATRIX_CHANNELS, MATRIX_ROWS, MATRIX_COLUMNS)
FLAT_INPUT_DIM = MATRIX_CHANNELS * MATRIX_ROWS * MATRIX_COLUMNS  # 1728


def validate_model_input(x: torch.Tensor) -> None:
    """Fail loudly on a wrong shape or non-finite model input."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Model input must be a torch.Tensor, got {type(x).__name__}.")
    if x.ndim != 4 or tuple(x.shape[1:]) != INPUT_SHAPE:
        raise ValueError(f"Expected input shape [N, 1, 64, 27], got {tuple(x.shape)}.")
    if not torch.isfinite(x).all():
        raise ValueError("Model input contains NaN or infinity.")


class MatrixMLP(nn.Module):
    """A flatten-then-MLP baseline over the 64x27 pressure map."""

    def __init__(
        self,
        *,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.0,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        dims = [FLAT_INPUT_DIM, *hidden_dims, num_classes]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.layers = nn.Sequential(*layers)
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_model_input(x)
        return self.layers(torch.flatten(x, 1))


class TinyCNN(nn.Module):
    """A small two-block conv net with adaptive pooling to a fixed head."""

    def __init__(
        self,
        *,
        channels: tuple[int, ...] = (16, 32),
        kernel_size: int = 3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        if len(channels) < 1:
            raise ValueError("channels must contain at least one layer width.")
        padding = kernel_size // 2
        blocks: list[nn.Module] = []
        in_channels = MATRIX_CHANNELS
        for out_channels in channels:
            blocks.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            )
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool2d(2))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((4, 2))
        self.head = nn.Linear(in_channels * 4 * 2, num_classes)
        self.channels = tuple(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_model_input(x)
        feat = self.features(x)
        feat = self.pool(feat)
        return self.head(torch.flatten(feat, 1))


class _ResidualBlock(nn.Module):
    """A channel-preserving residual block for a small 64x27 input."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SmallResNet(nn.Module):
    """A lightweight residual stack (no downsampling) for a 64x27 pressure map."""

    def __init__(
        self,
        *,
        base_channels: int = 16,
        num_blocks: int = 2,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        if base_channels < 1 or num_blocks < 1:
            raise ValueError("base_channels and num_blocks must be >= 1.")
        self.stem = nn.Sequential(
            nn.Conv2d(MATRIX_CHANNELS, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock(base_channels) for _ in range(num_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 2))
        self.head = nn.Linear(base_channels * 4 * 2, num_classes)
        self.base_channels = base_channels
        self.num_blocks = num_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_model_input(x)
        feat = self.stem(x)
        feat = self.blocks(feat)
        feat = self.pool(feat)
        return self.head(torch.flatten(feat, 1))


#: Small model registry keyed by config ``name``.
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "matrix_mlp": MatrixMLP,
    "tiny_cnn": TinyCNN,
    "small_resnet": SmallResNet,
}


def validate_model_config(config: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` for a non-mapping config or an unknown model name."""
    if not isinstance(config, Mapping):
        raise ValueError("Model config must be a mapping.")
    name = config.get("name")
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name {name!r}; known names: {sorted(MODEL_REGISTRY)}"
        )


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """Build a candidate model from a config mapping.

    ``config`` has the form ``{"name": "<key>", "params": {...}}``. Unknown
    names and illegal parameters raise a clear ``ValueError``.
    """
    validate_model_config(config)
    model_cls = MODEL_REGISTRY[config["name"]]
    raw_params = config.get("params", {})
    if not isinstance(raw_params, Mapping):
        raise ValueError("Model params must be a mapping.")
    params = dict(raw_params)
    try:
        return model_cls(**params)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid parameters for {config['name']!r}: {exc}"
        ) from exc


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters())
