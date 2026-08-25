"""SLP8 Region Segmentation Model: Slp8TinyFcn (TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1).

This module provides a minimal fully-convolutional network for SLP8 pressure-only
region segmentation. The model is intentionally small and deterministic, suitable
for CPU-only smoke testing.

Model architecture:
Input [N, 1, 192, 84]
→ Conv2d(1, 8, 3, padding=1)
→ ReLU
→ Conv2d(8, 16, 3, padding=1)
→ ReLU
→ Conv2d(16, 9, 1)
→ logits [N, 9, 192, 84]

Key properties:
* No pooling - maintains spatial resolution
* No pretrained weights
* No external model downloads
* Fail-closed input validation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: SLP8 input shape (H, W).
INPUT_SHAPE = (192, 84)

#: Number of classes (0=background, 1-8=foreground).
N_CLASSES = 9

#: Background class ID.
BACKGROUND_ID = 0

#: Model version.
MODEL_VERSION = "slp8_tiny_fcn_v0.1"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Slp8TinyFcn(nn.Module):
    """Minimal fully-convolutional network for SLP8 region segmentation.

    Architecture:
    Input [N, 1, 192, 84]
    → Conv2d(1, 8, 3, padding=1)
    → ReLU
    → Conv2d(8, 16, 3, padding=1)
    → ReLU
    → Conv2d(16, 9, 1)
    → logits [N, 9, 192, 84]

    No pooling, no pretrained weights, no external downloads.
    """

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        """Initialize Slp8TinyFcn.

        Parameters
        ----------
        n_classes : int
            Number of output classes (including background). Default 9.
        """
        super().__init__()
        self.n_classes = n_classes
        self.model_version = MODEL_VERSION

        # Block 1: (1, 192, 84) → (8, 192, 84)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu1 = nn.ReLU(inplace=True)

        # Block 2: (8, 192, 84) → (16, 192, 84)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu2 = nn.ReLU(inplace=True)

        # Block 3: (16, 192, 84) → (9, 192, 84)
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=1,
            bias=True,
        )

        # Initialize weights with small values for stable training
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize conv weights with Kaiming-like initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fail-closed validation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [N, 1, 192, 84].

        Returns
        -------
        torch.Tensor
            Logits of shape [N, 9, 192, 84].

        Raises
        ------
        ValueError
            If input shape, dtype, or finiteness fails validation.
        """
        # Fail-closed shape validation
        if x.ndim != 4:
            raise ValueError(
                f"Input must be 4D [N, C, H, W], got {x.ndim}D"
            )
        if x.shape[1] != 1:
            raise ValueError(
                f"Input channel must be 1, got {x.shape[1]}"
            )
        if x.shape[2:] != torch.Size(INPUT_SHAPE):
            raise ValueError(
                f"Input spatial shape must be {INPUT_SHAPE}, got {tuple(x.shape[2:])}"
            )

        # Fail-closed dtype validation
        if x.dtype != torch.float32:
            raise ValueError(
                f"Input dtype must be torch.float32, got {x.dtype}"
            )

        # Fail-closed finiteness validation
        if not x.isfinite().all():
            raise ValueError("Input contains non-finite values (NaN or Inf)")

        # Forward pass
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        # Output shape validation
        expected_shape = torch.Size([x.shape[0], self.n_classes] + list(INPUT_SHAPE))
        if x.shape != expected_shape:
            raise ValueError(
                f"Output shape mismatch: expected {expected_shape}, got {x.shape}"
            )

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax predictions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [N, 1, 192, 84].

        Returns
        -------
        torch.Tensor
            Predicted class indices of shape [N, 192, 84].
        """
        logits = self.forward(x)
        return logits.argmax(dim=1)

    def get_config(self) -> dict[str, Any]:
        """Return model configuration as dict."""
        return {
            "model_version": self.model_version,
            "n_classes": self.n_classes,
            "input_shape": list(INPUT_SHAPE),
            "architecture": [
                {"layer": "conv1", "in": 1, "out": 8, "kernel": 3, "padding": 1},
                {"layer": "relu1", "type": "ReLU"},
                {"layer": "conv2", "in": 8, "out": 16, "kernel": 3, "padding": 1},
                {"layer": "relu2", "type": "ReLU"},
                {"layer": "conv3", "in": 16, "out": self.n_classes, "kernel": 1},
            ],
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        """Return total number of parameters (including frozen)."""
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def create_slp8_tiny_fcn(
    n_classes: int = N_CLASSES,
    device: str = "cpu",
) -> tuple[Slp8TinyFcn, dict[str, Any]]:
    """Create a Slp8TinyFcn model and move to device.

    Parameters
    ----------
    n_classes : int
        Number of output classes.
    device : str
        Device to move model to.

    Returns
    -------
    tuple[Slp8TinyFcn, dict[str, Any]]
        (model, config_dict)
    """
    model = Slp8TinyFcn(n_classes=n_classes)
    model = model.to(device)
    config = model.get_config()
    config["device"] = str(device)
    return model, config


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def create_loss_fn() -> nn.CrossEntropyLoss:
    """Create unweighted CrossEntropyLoss for region segmentation.

    B03 smoke does NOT use class weights.
    """
    return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# Parameter change tracking
# ---------------------------------------------------------------------------


def compute_param_diff(
    model: nn.Module,
    reference_state: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute L2 difference between current and reference model state.

    Parameters
    ----------
    model : nn.Module
        Model to compare.
    reference_state : dict[str, torch.Tensor]
        Reference state dict from model.state_dict().

    Returns
    -------
    dict[str, float]
        Per-layer and total L2 parameter differences.
    """
    current_state = model.state_dict()
    total_diff = 0.0
    per_layer_diff: dict[str, float] = {}

    for key in reference_state:
        if key not in current_state:
            raise KeyError(f"Key {key!r} not in current state")
        diff = (current_state[key] - reference_state[key]).float()
        layer_diff = float((diff * diff).sum().sqrt().item())
        per_layer_diff[key] = layer_diff
        total_diff += layer_diff ** 2

    total_diff = float(math.sqrt(total_diff))
    per_layer_diff["_total"] = total_diff

    return per_layer_diff


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def verify_model_output_shape(
    model: Slp8TinyFcn,
    input_tensor: torch.Tensor,
) -> bool:
    """Verify model output has expected spatial dimensions.

    Parameters
    ----------
    model : Slp8TinyFcn
        Model to test.
    input_tensor : torch.Tensor
        Input of shape [N, 1, 192, 84].

    Returns
    -------
    bool
        True if output shape is correct.
    """
    with torch.no_grad():
        output = model(input_tensor)
    expected_spatial = torch.Size(INPUT_SHAPE)
    return output.shape[2:] == expected_spatial


def verify_model_gradient_flow(
    model: Slp8TinyFcn,
    input_tensor: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
) -> bool:
    """Verify gradients flow through all layers.

    Parameters
    ----------
    model : Slp8TinyFcn
        Model to test.
    input_tensor : torch.Tensor
        Input of shape [N, 1, 192, 84].
    labels : torch.Tensor
        Ground truth of shape [N, 192, 84].
    loss_fn : nn.Module
        Loss function.

    Returns
    -------
    bool
        True if all parameters have gradients.
    """
    model.train()
    output = model(input_tensor)

    # Flatten spatial dimensions for CrossEntropyLoss
    output_flat = output.reshape(output.shape[0], -1, output.shape[2] * output.shape[3])
    labels_flat = labels.reshape(labels.shape[0], -1)

    loss = loss_fn(output_flat, labels_flat)
    loss.backward()

    # Check all parameters have gradients
    all_have_grad = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )

    # Zero gradients for clean state
    model.zero_grad()

    return all_have_grad
