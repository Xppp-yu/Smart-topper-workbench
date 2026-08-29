"""SLP8 Region Segmentation Models (TASK-SLP-B03/B04/B04A).

This module provides the SLP8 pressure-only region segmentation model
registry used by the B03 smoke, the B04 PM-only Region Mini protocol,
and the B04A controlled architecture expansion Mini (R03).

* :class:`Slp8TinyFcn` — minimal fully-convolutional network used as the
  B03 smoke architecture and as ``slp8_tiny_fcn_v0.1`` (Candidate A) of
  the B04 Mini protocol.

* :class:`Slp8SmallUnet` — small encoder/decoder with explicit bilinear
  spatial-size recovery introduced as ``slp8_small_unet_v0.1`` (Candidate B)
  in the B04 Mini protocol and reused as the B04A incumbent.

* :class:`Slp8ResUnetLite` — B04A new candidate
  ``slp8_resunet_lite_v0.1``; U-Net-style encoder-decoder with three
  residual blocks. Every residual Add uses an **explicit 1x1 Conv2d
  shortcut** (kernel=1, stride=1, padding=0, bias=true). Identity shortcuts
  for channel-mismatched blocks are forbidden by the B04A R03 contract.

* :class:`Slp8DeepLabV3PlusLite` — B04A new candidate
  ``slp8_deeplabv3plus_lite_v0.1``; lightweight DeepLabV3+ with the
  **Option A plain atrous Conv2d** variant. All Conv2d ``groups=1``;
  ASPP atrous rates ``[3, 6, 9, 12]``; 6 branches (1 pointwise + 4 atrous
  + 1 GAP) each producing 16 channels; concat 96 → 32; low-level
  projection in the decoder. Xception / depthwise-separable are forbidden
  by the B04A R03 contract.

Key properties (applies to all four):

* No pretrained weights, no external downloads.
* Fail-closed input validation (shape, dtype, finiteness).
* Parameter count is exposed via :meth:`count_parameters` for the B04
  150,000-parameter cap and the B04A 300,000-parameter cap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: SLP8 input shape (H, W).
INPUT_SHAPE = (192, 84)

#: Number of classes (0=background, 1-8=foreground).
N_CLASSES = 9

#: Background class ID.
BACKGROUND_ID = 0

#: Candidate A model version (B03 smoke architecture, reused in B04).
MODEL_VERSION = "slp8_tiny_fcn_v0.1"

#: Candidate B model version (B04 SmallUNet; B04A incumbent).
SMALL_UNET_VERSION = "slp8_small_unet_v0.1"

#: B04A new candidate — ResUNet-lite (1x1 Conv2d shortcut everywhere).
RESUNET_LITE_VERSION = "slp8_resunet_lite_v0.1"

#: B04A new candidate — DeepLabV3+-lite (Option A: plain atrous Conv2d).
DEEPLABV3PLUS_LITE_VERSION = "slp8_deeplabv3plus_lite_v0.1"

#: Hard parameter-count cap for B04 Mini candidates.
B04_MAX_PARAMETERS = 150_000

#: Hard parameter-count cap for B04A Mini new candidates.
B04A_MAX_PARAMETERS = 300_000

#: Frozen exact parameter counts for the three B04A candidates
#: (verified by :mod:`scripts.validate_b04a_protocol` and the
#: B04A implementation tests).
B04A_EXACT_PARAMETER_COUNTS: dict[str, int] = {
    SMALL_UNET_VERSION: 118_121,
    RESUNET_LITE_VERSION: 120_809,
    DEEPLABV3PLUS_LITE_VERSION: 53_449,
}

#: DeepLabV3+-lite atrous rates (frozen by B04A R03 contract).
DEEPLABV3PLUS_LITE_ATROUS_RATES: tuple[int, ...] = (3, 6, 9, 12)

#: DeepLabV3+-lite per-branch output channel count (frozen by R03).
DEEPLABV3PLUS_LITE_BRANCH_CHANNELS: int = 16

#: DeepLabV3+-lite ASPP post-concat input channels (6 branches × 16).
DEEPLABV3PLUS_LITE_POST_CONCAT_CHANNELS: int = 96

#: Registered model builders (frozen by the B04 / B04A protocols).
MODEL_REGISTRY: dict[str, "ModelBuilder"] = {}


# ---------------------------------------------------------------------------
# Internal input-validation helpers
# ---------------------------------------------------------------------------


def _validate_input_tensor(x: torch.Tensor) -> None:
    """Fail-closed validation for SLP8 region model inputs.

    Raises
    ------
    ValueError
        If the input shape, channel count, dtype, or finiteness fails.
    """

    if x.ndim != 4:
        raise ValueError(f"Input must be 4D [N, C, H, W], got {x.ndim}D")
    if x.shape[1] != 1:
        raise ValueError(f"Input channel must be 1, got {x.shape[1]}")
    if x.shape[2:] != torch.Size(INPUT_SHAPE):
        raise ValueError(
            f"Input spatial shape must be {INPUT_SHAPE}, got {tuple(x.shape[2:])}"
        )
    if x.dtype != torch.float32:
        raise ValueError(
            f"Input dtype must be torch.float32, got {x.dtype}"
        )
    if not x.isfinite().all():
        raise ValueError("Input contains non-finite values (NaN or Inf)")


def _init_conv_kaiming_zero_bias(module: nn.Module) -> None:
    """Kaiming-normal conv init + zero bias (B04A R03 contract)."""

    for sub in module.modules():
        if isinstance(sub, nn.Conv2d):
            nn.init.kaiming_normal_(
                sub.weight, mode="fan_out", nonlinearity="relu"
            )
            if sub.bias is not None:
                nn.init.zeros_(sub.bias)


@dataclass(frozen=True)
class ModelBuilder:
    """Registry entry: name -> callable producing a fresh candidate model.

    The :class:`ModelBuilder` records the canonical name (used in configs
    and metrics), the version string embedded in checkpoints, and a factory
    callable.  Builders are registered by :func:`register_model_builder` so
    that the B04 runner can resolve any candidate by name without hard-coding
    ``if/elif`` branches.
    """

    name: str
    version: str
    factory: "Any"  # Callable[..., nn.Module]


def register_model_builder(builder: ModelBuilder) -> None:
    """Register a candidate model builder in the global registry."""

    if builder.name in MODEL_REGISTRY:
        raise ValueError(
            f"Model builder {builder.name!r} already registered; "
            "B04 candidate names must be unique."
        )
    MODEL_REGISTRY[builder.name] = builder


def get_model_builder(name: str) -> ModelBuilder:
    """Return the registered builder for ``name`` or raise ``KeyError``."""

    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}; registered: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]


def list_model_builders() -> list[str]:
    """Return the sorted list of registered model names."""

    return sorted(MODEL_REGISTRY)


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
            "parameter_count": self.count_parameters(),
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


# ---------------------------------------------------------------------------
# Slp8SmallUnet (Candidate B, B04)
# ---------------------------------------------------------------------------


class Slp8SmallUnet(nn.Module):
    """Small encoder/decoder network for SLP8 region segmentation.

    Architecture (frozen by B04 v0.1):

    * Input ``[N, 1, 192, 84]``;
    * Encoder block 1: ``Conv 1→16`` + ReLU, ``Conv 16→16`` + ReLU;
    * ``MaxPool2d(2)`` → ``[N, 16, 96, 42]``;
    * Encoder block 2: ``Conv 16→32`` + ReLU, ``Conv 32→32`` + ReLU;
    * ``MaxPool2d(2)`` → ``[N, 32, 48, 21]``;
    * Bottleneck: ``Conv 32→64`` + ReLU, ``Conv 64→64`` + ReLU;
    * Decoder: bilinear upsample to the *explicit* spatial size of each
      skip connection (the protocol forbids ``scale_factor``-driven
      recovery, which is brittle for the odd ``W=84`` width after two
      downsamples);
    * Concat the corresponding encoder skip after each upsample;
    * Conv block ``96→32→32`` after the first concat;
    * Conv block ``48→16→16`` after the second concat;
    * Final ``Conv 16→9, kernel=1`` to logits;
    * Output ``[N, 9, 192, 84]``.

    No BatchNorm, no Dropout, no pretrained weights, no external downloads.
    The hard parameter-count cap is :data:`B04_MAX_PARAMETERS` (``150_000``),
    enforced by the B04 smoke contract.
    """

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.model_version = SMALL_UNET_VERSION

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.enc1_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True)
        self.enc1_relu1 = nn.ReLU(inplace=True)
        self.enc1_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.enc1_relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
        self.enc2_relu1 = nn.ReLU(inplace=True)
        self.enc2_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.enc2_relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        self.bottleneck_conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
        self.bottleneck_relu1 = nn.ReLU(inplace=True)
        self.bottleneck_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.bottleneck_relu2 = nn.ReLU(inplace=True)

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        # First upsample: (48, 21) -> (96, 42); concat with skip2 (32 ch) ->
        # 64 + 32 = 96 channels; conv 96 -> 32 -> 32.
        self.dec1_conv1 = nn.Conv2d(96, 32, kernel_size=3, padding=1, bias=True)
        self.dec1_relu1 = nn.ReLU(inplace=True)
        self.dec1_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.dec1_relu2 = nn.ReLU(inplace=True)

        # Second upsample: (96, 42) -> (192, 84); concat with skip1 (16 ch) ->
        # 32 + 16 = 48 channels; conv 48 -> 16 -> 16.
        self.dec2_conv1 = nn.Conv2d(48, 16, kernel_size=3, padding=1, bias=True)
        self.dec2_relu1 = nn.ReLU(inplace=True)
        self.dec2_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.dec2_relu2 = nn.ReLU(inplace=True)

        # ------------------------------------------------------------------
        # Final 1x1 conv to logits
        # ------------------------------------------------------------------
        self.final_conv = nn.Conv2d(16, n_classes, kernel_size=1, bias=True)

        # Recorded explicit recovery targets so tests can audit them.
        self._skip1_shape = INPUT_SHAPE             # (192, 84)
        self._skip2_shape = (INPUT_SHAPE[0] // 2, INPUT_SHAPE[1] // 2)  # (96, 42)
        self._bottleneck_shape = (INPUT_SHAPE[0] // 4, INPUT_SHAPE[1] // 4)  # (48, 21)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-normal conv init + zero bias."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fail-closed input validation.

        Raises
        ------
        ValueError
            If input shape, dtype, or finiteness fails validation.
        """
        if x.ndim != 4:
            raise ValueError(f"Input must be 4D [N, C, H, W], got {x.ndim}D")
        if x.shape[1] != 1:
            raise ValueError(f"Input channel must be 1, got {x.shape[1]}")
        if x.shape[2:] != torch.Size(INPUT_SHAPE):
            raise ValueError(
                f"Input spatial shape must be {INPUT_SHAPE}, got {tuple(x.shape[2:])}"
            )
        if x.dtype != torch.float32:
            raise ValueError(
                f"Input dtype must be torch.float32, got {x.dtype}"
            )
        if not x.isfinite().all():
            raise ValueError("Input contains non-finite values (NaN or Inf)")

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        skip1 = self.enc1_relu1(self.enc1_conv1(x))
        skip1 = self.enc1_relu2(self.enc1_conv2(skip1))
        x = self.pool1(skip1)

        skip2 = self.enc2_relu1(self.enc2_conv1(x))
        skip2 = self.enc2_relu2(self.enc2_conv2(skip2))
        x = self.pool2(skip2)

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        x = self.bottleneck_relu1(self.bottleneck_conv1(x))
        x = self.bottleneck_relu2(self.bottleneck_conv2(x))

        # ------------------------------------------------------------------
        # Decoder — explicit spatial-size recovery (no scale_factor)
        # ------------------------------------------------------------------
        target_h2, target_w2 = self._skip2_shape
        bottleneck_channels = x.shape[1]
        if x.shape[2] != target_h2 or x.shape[3] != target_w2:
            x = F.interpolate(
                x,
                size=(target_h2, target_w2),
                mode="bilinear",
                align_corners=False,
            )
        # F.interpolate must preserve channel count.
        if x.shape[1] != bottleneck_channels:
            raise RuntimeError(
                f"Decoder stage 1 channel corruption: pre-upsample had "
                f"{bottleneck_channels} channels, post-upsample has {x.shape[1]}"
            )
        if (x.shape[2], x.shape[3]) != (skip2.shape[2], skip2.shape[3]):
            raise RuntimeError(
                f"Decoder stage 1 spatial mismatch: upsample is "
                f"({x.shape[2]}, {x.shape[3]}) but skip2 is "
                f"({skip2.shape[2]}, {skip2.shape[3]})"
            )
        x = torch.cat([x, skip2], dim=1)
        x = self.dec1_relu1(self.dec1_conv1(x))
        x = self.dec1_relu2(self.dec1_conv2(x))

        target_h1, target_w1 = self._skip1_shape
        if x.shape[2] != target_h1 or x.shape[3] != target_w1:
            x = F.interpolate(
                x,
                size=(target_h1, target_w1),
                mode="bilinear",
                align_corners=False,
            )
        if (x.shape[2], x.shape[3]) != (skip1.shape[2], skip1.shape[3]):
            raise RuntimeError(
                f"Decoder stage 2 spatial mismatch: upsample is "
                f"({x.shape[2]}, {x.shape[3]}) but skip1 is "
                f"({skip1.shape[2]}, {skip1.shape[3]})"
            )
        x = torch.cat([x, skip1], dim=1)
        x = self.dec2_relu1(self.dec2_conv1(x))
        x = self.dec2_relu2(self.dec2_conv2(x))

        logits = self.final_conv(x)

        expected = torch.Size([logits.shape[0], self.n_classes] + list(INPUT_SHAPE))
        if logits.shape != expected:
            raise ValueError(
                f"Output shape mismatch: expected {expected}, got {logits.shape}"
            )
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax predictions of shape ``[N, H, W]``."""

        logits = self.forward(x)
        return logits.argmax(dim=1)

    def get_config(self) -> dict[str, Any]:
        """Return the model configuration as a JSON-safe dict."""

        return {
            "model_version": self.model_version,
            "n_classes": self.n_classes,
            "input_shape": list(INPUT_SHAPE),
            "parameter_count": self.count_parameters(),
            "architecture": [
                {"stage": "encoder1", "blocks": [
                    {"layer": "enc1_conv1", "in": 1, "out": 16, "kernel": 3, "padding": 1},
                    {"layer": "enc1_relu1", "type": "ReLU"},
                    {"layer": "enc1_conv2", "in": 16, "out": 16, "kernel": 3, "padding": 1},
                    {"layer": "enc1_relu2", "type": "ReLU"},
                    {"layer": "pool1", "type": "MaxPool2d", "kernel": 2},
                ]},
                {"stage": "encoder2", "blocks": [
                    {"layer": "enc2_conv1", "in": 16, "out": 32, "kernel": 3, "padding": 1},
                    {"layer": "enc2_relu1", "type": "ReLU"},
                    {"layer": "enc2_conv2", "in": 32, "out": 32, "kernel": 3, "padding": 1},
                    {"layer": "enc2_relu2", "type": "ReLU"},
                    {"layer": "pool2", "type": "MaxPool2d", "kernel": 2},
                ]},
                {"stage": "bottleneck", "blocks": [
                    {"layer": "bottleneck_conv1", "in": 32, "out": 64, "kernel": 3, "padding": 1},
                    {"layer": "bottleneck_relu1", "type": "ReLU"},
                    {"layer": "bottleneck_conv2", "in": 64, "out": 64, "kernel": 3, "padding": 1},
                    {"layer": "bottleneck_relu2", "type": "ReLU"},
                ]},
                {"stage": "decoder1", "blocks": [
                    {"layer": "upsample1", "type": "F.interpolate_bilinear",
                     "size": list(self._skip2_shape), "align_corners": False},
                    {"layer": "concat1", "channels": 64 + 32},
                    {"layer": "dec1_conv1", "in": 96, "out": 32, "kernel": 3, "padding": 1},
                    {"layer": "dec1_relu1", "type": "ReLU"},
                    {"layer": "dec1_conv2", "in": 32, "out": 32, "kernel": 3, "padding": 1},
                    {"layer": "dec1_relu2", "type": "ReLU"},
                ]},
                {"stage": "decoder2", "blocks": [
                    {"layer": "upsample2", "type": "F.interpolate_bilinear",
                     "size": list(self._skip1_shape), "align_corners": False},
                    {"layer": "concat2", "channels": 32 + 16},
                    {"layer": "dec2_conv1", "in": 48, "out": 16, "kernel": 3, "padding": 1},
                    {"layer": "dec2_relu1", "type": "ReLU"},
                    {"layer": "dec2_conv2", "in": 16, "out": 16, "kernel": 3, "padding": 1},
                    {"layer": "dec2_relu2", "type": "ReLU"},
                ]},
                {"stage": "final", "blocks": [
                    {"layer": "final_conv", "in": 16, "out": self.n_classes, "kernel": 1},
                ]},
            ],
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        """Return total number of parameters (including frozen)."""

        return sum(p.numel() for p in self.parameters())


def create_slp8_small_unet(
    n_classes: int = N_CLASSES,
    device: str = "cpu",
) -> tuple["Slp8SmallUnet", dict[str, Any]]:
    """Construct a :class:`Slp8SmallUnet` and move to ``device``.

    Returns
    -------
    tuple
        ``(model, config_dict)``; the config dict already includes
        ``device``.
    """

    model = Slp8SmallUnet(n_classes=n_classes)
    model = model.to(device)
    config = model.get_config()
    config["device"] = str(device)
    return model, config


# ---------------------------------------------------------------------------
# Model registry wiring
# ---------------------------------------------------------------------------


def _build_slp8_tiny_fcn(n_classes: int, device: str) -> tuple[nn.Module, dict[str, Any]]:
    model, cfg = create_slp8_tiny_fcn(n_classes=n_classes, device=device)
    return model, cfg


def _build_slp8_small_unet(n_classes: int, device: str) -> tuple[nn.Module, dict[str, Any]]:
    model, cfg = create_slp8_small_unet(n_classes=n_classes, device=device)
    return model, cfg


register_model_builder(
    ModelBuilder(
        name=MODEL_VERSION,
        version=MODEL_VERSION,
        factory=_build_slp8_tiny_fcn,
    )
)
register_model_builder(
    ModelBuilder(
        name=SMALL_UNET_VERSION,
        version=SMALL_UNET_VERSION,
        factory=_build_slp8_small_unet,
    )
)


# ---------------------------------------------------------------------------
# B04A: Slp8ResUnetLite — slp8_resunet_lite_v0.1
# ---------------------------------------------------------------------------
#
# Contract (B04A R03):
#   * Three residual blocks (enc1, enc2, bottleneck).
#   * Every residual Add uses an explicit 1x1 Conv2d shortcut with
#     kernel_size=1, stride=1, padding=0, bias=True.
#   * Identity shortcuts for channel-mismatched blocks are FORBIDDEN.
#   * main_output_channels == shortcut_output_channels and
#     main_output_shape == shortcut_output_shape for every residual block.
#   * Decoder uses frozen bilinear interpolation, concat with encoder
#     skips, and the same channel widths as the SmallUNet decoder.
#   * Output is [N, 9, 192, 84].
#   * exact_parameter_count = 120,809 (verified by validator).
#
# Implementation notes:
#   * The shortcut Conv2d is exposed as a named attribute
#     (``shortcut_conv``) so the B04A implementation tests can introspect
#     kernel/stride/padding/bias without traversing sub-modules.
#   * Each residual block runs the main path first (Conv→ReLU→Conv),
#     computes the 1x1 projection of the input, adds them, then applies a
#     post-add ReLU. The post-add ReLU is the only non-linearity applied
#     after the Add, matching the frozen forward plan.
# ---------------------------------------------------------------------------


class _ResidualBlock(nn.Module):
    """A residual block with an explicit 1x1 Conv2d projection shortcut.

    Parameters
    ----------
    in_channels : int
        Number of input channels to the main path.
    out_channels : int
        Number of output channels of the main path and the shortcut
        projection (must be equal so the residual Add is valid).
    input_shape : tuple[int, int]
        Spatial shape (H, W) of the input.  The shortcut Conv2d uses
        stride=1, padding=0, kernel=1 so the spatial shape is preserved
        exactly; this is what makes the ``Add`` well-defined.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_shape: tuple[int, int],
    ) -> None:
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"Residual block requires positive channel counts; got "
                f"in_channels={in_channels}, out_channels={out_channels}"
            )

        # ----- Main path: two 3x3 convs with ReLU between them.
        self.main_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.main_relu1 = nn.ReLU(inplace=True)
        self.main_conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # ----- Shortcut path: explicit 1x1 Conv2d channel projection.
        # kernel=1, stride=1, padding=0, bias=True (frozen by B04A R03).
        self.shortcut_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Post-add ReLU.
        self.post_add_relu = nn.ReLU(inplace=True)

        # Cache input/output shapes for tests and forward-plan auditing.
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._input_shape = tuple(input_shape)
        self._output_shape = tuple(input_shape)

    @property
    def main_output_channels(self) -> int:
        return self._out_channels

    @property
    def shortcut_output_channels(self) -> int:
        return self._out_channels

    @property
    def main_output_shape(self) -> tuple[int, int]:
        return self._output_shape

    @property
    def shortcut_output_shape(self) -> tuple[int, int]:
        return self._output_shape

    @property
    def in_channels(self) -> int:
        return self._in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: main + shortcut (1x1) → Add → ReLU."""

        main = self.main_relu1(self.main_conv1(x))
        main = self.main_conv2(main)

        shortcut = self.shortcut_conv(x)

        if shortcut.shape != main.shape:
            raise RuntimeError(
                f"Residual block shape mismatch: main={tuple(main.shape)} "
                f"shortcut={tuple(shortcut.shape)}; channels and spatial "
                f"shape must be equal"
            )
        if shortcut.shape[1] != main.shape[1]:
            raise RuntimeError(
                f"Residual block channel mismatch: main has "
                f"{main.shape[1]} channels, shortcut has "
                f"{shortcut.shape[1]} channels"
            )

        out = main + shortcut
        out = self.post_add_relu(out)
        return out


class Slp8ResUnetLite(nn.Module):
    """B04A R03 ResUNet-lite candidate (``slp8_resunet_lite_v0.1``).

    Architecture (frozen by B04A R03, 3 residual blocks):

    * Input ``[N, 1, 192, 84]``;
    * Encoder block 1: residual block with main path
      ``Conv 1→16, Conv 16→16`` and 1x1 shortcut ``Conv 1→16``;
    * ``MaxPool2d(2)`` → ``[N, 16, 96, 42]``;
    * Encoder block 2: residual block with main path
      ``Conv 16→32, Conv 32→32`` and 1x1 shortcut ``Conv 16→32``;
    * ``MaxPool2d(2)`` → ``[N, 32, 48, 21]``;
    * Bottleneck residual block: main path ``Conv 32→64, Conv 64→64`` and
      1x1 shortcut ``Conv 32→64``;
    * Decoder: bilinear upsample to the *explicit* spatial size of each
      skip connection (matching SmallUNet's explicit recovery);
    * Concat the corresponding encoder skip after each upsample;
    * Conv block ``96→32→32`` after the first concat;
    * Conv block ``48→16→16`` after the second concat;
    * Final ``Conv 16→9, kernel=1`` to logits;
    * Output ``[N, 9, 192, 84]``.

    The exact parameter count is **120,809** (verified by
    :mod:`scripts.validate_b04a_protocol` and the B04A implementation
    tests).  No BatchNorm, no Dropout, no pretrained weights, no external
    downloads.
    """

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.model_version = RESUNET_LITE_VERSION

        # ------------------------------------------------------------------
        # Encoder (3 residual blocks)
        # ------------------------------------------------------------------
        self.enc1_resblock = _ResidualBlock(
            in_channels=1, out_channels=16, input_shape=INPUT_SHAPE
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_resblock = _ResidualBlock(
            in_channels=16, out_channels=32, input_shape=(96, 42)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck_resblock = _ResidualBlock(
            in_channels=32, out_channels=64, input_shape=(48, 21)
        )

        # ------------------------------------------------------------------
        # Decoder — same shape as SmallUNet
        # ------------------------------------------------------------------
        self.dec1_conv1 = nn.Conv2d(96, 32, kernel_size=3, padding=1, bias=True)
        self.dec1_relu1 = nn.ReLU(inplace=True)
        self.dec1_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.dec1_relu2 = nn.ReLU(inplace=True)

        self.dec2_conv1 = nn.Conv2d(48, 16, kernel_size=3, padding=1, bias=True)
        self.dec2_relu1 = nn.ReLU(inplace=True)
        self.dec2_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.dec2_relu2 = nn.ReLU(inplace=True)

        # ------------------------------------------------------------------
        # Final 1x1 conv to logits
        # ------------------------------------------------------------------
        self.final_conv = nn.Conv2d(16, n_classes, kernel_size=1, bias=True)

        # Recorded explicit recovery targets (audited by tests).
        self._skip1_shape = INPUT_SHAPE             # (192, 84)
        self._skip2_shape = (INPUT_SHAPE[0] // 2, INPUT_SHAPE[1] // 2)  # (96, 42)
        self._bottleneck_shape = (INPUT_SHAPE[0] // 4, INPUT_SHAPE[1] // 4)  # (48, 21)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-normal conv init + zero bias (R03 contract)."""

        _init_conv_kaiming_zero_bias(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fail-closed input validation."""

        _validate_input_tensor(x)

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        skip1 = self.enc1_resblock(x)
        x = self.pool1(skip1)

        skip2 = self.enc2_resblock(x)
        x = self.pool2(skip2)

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        x = self.bottleneck_resblock(x)

        # ------------------------------------------------------------------
        # Decoder — explicit spatial-size recovery
        # ------------------------------------------------------------------
        target_h2, target_w2 = self._skip2_shape
        if x.shape[2] != target_h2 or x.shape[3] != target_w2:
            x = F.interpolate(
                x,
                size=(target_h2, target_w2),
                mode="bilinear",
                align_corners=False,
            )
        if (x.shape[2], x.shape[3]) != (skip2.shape[2], skip2.shape[3]):
            raise RuntimeError(
                f"Decoder stage 1 spatial mismatch: upsample is "
                f"({x.shape[2]}, {x.shape[3]}) but skip2 is "
                f"({skip2.shape[2]}, {skip2.shape[3]})"
            )
        x = torch.cat([x, skip2], dim=1)
        x = self.dec1_relu1(self.dec1_conv1(x))
        x = self.dec1_relu2(self.dec1_conv2(x))

        target_h1, target_w1 = self._skip1_shape
        if x.shape[2] != target_h1 or x.shape[3] != target_w1:
            x = F.interpolate(
                x,
                size=(target_h1, target_w1),
                mode="bilinear",
                align_corners=False,
            )
        if (x.shape[2], x.shape[3]) != (skip1.shape[2], skip1.shape[3]):
            raise RuntimeError(
                f"Decoder stage 2 spatial mismatch: upsample is "
                f"({x.shape[2]}, {x.shape[3]}) but skip1 is "
                f"({skip1.shape[2]}, {skip1.shape[3]})"
            )
        x = torch.cat([x, skip1], dim=1)
        x = self.dec2_relu1(self.dec2_conv1(x))
        x = self.dec2_relu2(self.dec2_conv2(x))

        logits = self.final_conv(x)

        expected = torch.Size([logits.shape[0], self.n_classes] + list(INPUT_SHAPE))
        if logits.shape != expected:
            raise ValueError(
                f"Output shape mismatch: expected {expected}, got {logits.shape}"
            )
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax predictions of shape ``[N, H, W]``."""

        logits = self.forward(x)
        return logits.argmax(dim=1)

    def get_config(self) -> dict[str, Any]:
        """Return the model configuration as a JSON-safe dict."""

        return {
            "model_version": self.model_version,
            "n_classes": self.n_classes,
            "input_shape": list(INPUT_SHAPE),
            "parameter_count": self.count_parameters(),
            "residual_blocks": [
                {
                    "name": "enc1_resblock",
                    "in_channels": 1,
                    "out_channels": 16,
                    "shortcut_op": "Conv2d 1x1 (k=1,s=1,p=0,bias=True)",
                },
                {
                    "name": "enc2_resblock",
                    "in_channels": 16,
                    "out_channels": 32,
                    "shortcut_op": "Conv2d 1x1 (k=1,s=1,p=0,bias=True)",
                },
                {
                    "name": "bottleneck_resblock",
                    "in_channels": 32,
                    "out_channels": 64,
                    "shortcut_op": "Conv2d 1x1 (k=1,s=1,p=0,bias=True)",
                },
            ],
            "architecture": [
                {"stage": "encoder1", "blocks": [
                    {"layer": "enc1_resblock", "in": 1, "out": 16,
                     "shortcut": "Conv2d 1x1 (k=1,s=1,p=0,bias=True)"},
                    {"layer": "pool1", "type": "MaxPool2d", "kernel": 2},
                ]},
                {"stage": "encoder2", "blocks": [
                    {"layer": "enc2_resblock", "in": 16, "out": 32,
                     "shortcut": "Conv2d 1x1 (k=1,s=1,p=0,bias=True)"},
                    {"layer": "pool2", "type": "MaxPool2d", "kernel": 2},
                ]},
                {"stage": "bottleneck", "blocks": [
                    {"layer": "bottleneck_resblock", "in": 32, "out": 64,
                     "shortcut": "Conv2d 1x1 (k=1,s=1,p=0,bias=True)"},
                ]},
                {"stage": "decoder1", "blocks": [
                    {"layer": "upsample1", "type": "F.interpolate_bilinear",
                     "size": list(self._skip2_shape), "align_corners": False},
                    {"layer": "concat1", "channels": 64 + 32},
                    {"layer": "dec1_conv1", "in": 96, "out": 32, "kernel": 3, "padding": 1},
                    {"layer": "dec1_relu1", "type": "ReLU"},
                    {"layer": "dec1_conv2", "in": 32, "out": 32, "kernel": 3, "padding": 1},
                    {"layer": "dec1_relu2", "type": "ReLU"},
                ]},
                {"stage": "decoder2", "blocks": [
                    {"layer": "upsample2", "type": "F.interpolate_bilinear",
                     "size": list(self._skip1_shape), "align_corners": False},
                    {"layer": "concat2", "channels": 32 + 16},
                    {"layer": "dec2_conv1", "in": 48, "out": 16, "kernel": 3, "padding": 1},
                    {"layer": "dec2_relu1", "type": "ReLU"},
                    {"layer": "dec2_conv2", "in": 16, "out": 16, "kernel": 3, "padding": 1},
                    {"layer": "dec2_relu2", "type": "ReLU"},
                ]},
                {"stage": "final", "blocks": [
                    {"layer": "final_conv", "in": 16, "out": self.n_classes, "kernel": 1},
                ]},
            ],
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        """Return total number of parameters (including frozen)."""

        return sum(p.numel() for p in self.parameters())

    def get_residual_block(self, name: str) -> _ResidualBlock:
        """Return the named residual block for tests and auditing.

        Supported names: ``"enc1_resblock"``, ``"enc2_resblock"``,
        ``"bottleneck_resblock"``.
        """

        if not hasattr(self, name):
            raise KeyError(
                f"Slp8ResUnetLite has no attribute {name!r}; expected one of "
                f"'enc1_resblock', 'enc2_resblock', 'bottleneck_resblock'"
            )
        block = getattr(self, name)
        if not isinstance(block, _ResidualBlock):
            raise TypeError(
                f"{name} is not a _ResidualBlock; got {type(block).__name__}"
            )
        return block


def create_slp8_resunet_lite(
    n_classes: int = N_CLASSES,
    device: str = "cpu",
) -> tuple["Slp8ResUnetLite", dict[str, Any]]:
    """Construct a :class:`Slp8ResUnetLite` and move to ``device``."""

    model = Slp8ResUnetLite(n_classes=n_classes)
    model = model.to(device)
    config = model.get_config()
    config["device"] = str(device)
    return model, config


# ---------------------------------------------------------------------------
# B04A: Slp8DeepLabV3PlusLite — slp8_deeplabv3plus_lite_v0.1 (Option A)
# ---------------------------------------------------------------------------
#
# Contract (B04A R03, Option A plain atrous Conv2d):
#   * variant = "option_A_plain_atrous_Conv2d"
#   * All Conv2d layers use groups=1.  Xception / depthwise-separable
#     are FORBIDDEN.
#   * ASPP atrous_rates = [3, 6, 9, 12]; 4 atrous branches.
#   * Six ASPP branches (1 pointwise + 4 atrous + 1 GAP), each producing
#     16 channels; concat 96 channels.
#   * post_concat: Conv2d 96 → 32 → ReLU.
#   * Decoder: low-level projection Conv2d 16 → 16, bilinear upsample,
#     concat 48 channels, two 3x3 Conv2d layers 48→32→32.
#   * Final: Conv2d 32 → 9, kernel=1.
#   * Output [N, 9, 192, 84].
#   * exact_parameter_count = 53,449.
# ---------------------------------------------------------------------------


class _AsppBranchAtrous(nn.Module):
    """A single atrous Conv2d branch of the ASPP module (Option A plain)."""

    def __init__(self, in_channels: int, out_channels: int, atrous_rate: int) -> None:
        super().__init__()
        if atrous_rate <= 0:
            raise ValueError(
                f"atrous_rate must be a positive integer; got {atrous_rate}"
            )
        self.atrous_rate = int(atrous_rate)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=self.atrous_rate,
            dilation=self.atrous_rate,
            bias=True,
            groups=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _AsppBranchPointwise(nn.Module):
    """The 1x1 pointwise branch of the ASPP module."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _AsppBranchGap(nn.Module):
    """The global-average-pool branch of the ASPP module.

    AdaptiveAvgPool2d → 1x1 Conv2d → BilinearInterpolate back to the
    ASPP feature-map size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        target_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )
        self._target_size = tuple(target_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.gap(x)
        out = self.conv(out)
        out = F.interpolate(
            out, size=self._target_size, mode="bilinear", align_corners=False
        )
        return out


class _AsppModule(nn.Module):
    """Lightweight DeepLabV3+ ASPP module (Option A, plain atrous)."""

    def __init__(
        self,
        in_channels: int,
        out_channels_per_branch: int,
        atrous_rates: tuple[int, ...],
        target_size: tuple[int, int],
    ) -> None:
        super().__init__()

        if in_channels <= 0 or out_channels_per_branch <= 0:
            raise ValueError(
                f"ASPP requires positive channel counts; got "
                f"in_channels={in_channels}, "
                f"out_channels_per_branch={out_channels_per_branch}"
            )
        if len(atrous_rates) < 1:
            raise ValueError(
                "ASPP requires at least one atrous rate; got empty tuple"
            )

        self.in_channels = in_channels
        self.out_channels_per_branch = out_channels_per_branch
        self.atrous_rates = tuple(int(r) for r in atrous_rates)
        self._target_size = tuple(target_size)

        # 1x1 pointwise branch.
        self.branch_pointwise = _AsppBranchPointwise(
            in_channels=in_channels,
            out_channels=out_channels_per_branch,
        )
        # 4 atrous 3x3 branches (Option A: plain atrous Conv2d, groups=1).
        self.branch_atrous = nn.ModuleList(
            _AsppBranchAtrous(
                in_channels=in_channels,
                out_channels=out_channels_per_branch,
                atrous_rate=rate,
            )
            for rate in self.atrous_rates
        )
        # Global-average-pool branch.
        self.branch_gap = _AsppBranchGap(
            in_channels=in_channels,
            out_channels=out_channels_per_branch,
            target_size=self._target_size,
        )

        # Post-concat 1x1 Conv 96 → 32.
        n_branches = 1 + len(self.atrous_rates) + 1
        self.concat_in_channels = n_branches * out_channels_per_branch
        self.post_concat_conv = nn.Conv2d(
            in_channels=self.concat_in_channels,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )
        self.post_concat_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.branch_pointwise(x)]
        for branch in self.branch_atrous:
            feats.append(branch(x))
        feats.append(self.branch_gap(x))

        concat = torch.cat(feats, dim=1)
        if concat.shape[1] != self.concat_in_channels:
            raise RuntimeError(
                f"ASPP concat channel mismatch: expected "
                f"{self.concat_in_channels}, got {concat.shape[1]}"
            )
        out = self.post_concat_relu(self.post_concat_conv(concat))
        return out


class Slp8DeepLabV3PlusLite(nn.Module):
    """B04A R03 DeepLabV3+-lite candidate (``slp8_deeplabv3plus_lite_v0.1``).

    Architecture (frozen by B04A R03, Option A plain atrous):

    * Input ``[N, 1, 192, 84]``;
    * Stem: ``Conv 1→16`` + ReLU, ``Conv 16→16`` + ReLU;
      (low-level features, 16 ch at full input resolution)
    * Downsample: ``Conv 16→32, stride=2`` + ReLU → ``[N, 32, 96, 42]``;
    * ASPP module: 1 pointwise + 4 atrous (rates 3/6/9/12) + 1 GAP
      branch; each branch 16 ch; concat 96 ch; post-concat 1x1 Conv
      96 → 32 + ReLU;
    * Low-level projection: ``Conv 16→16`` + ReLU (at full input
      resolution);
    * Decoder fusion: bilinear upsample to (192, 84), concat with
      low-level features (32 + 16 = 48 ch), two 3x3 Conv layers
      48→32→32, both with ReLU;
    * Final: ``Conv 32→9, kernel=1`` to logits;
    * Output ``[N, 9, 192, 84]``.

    The exact parameter count is **53,449** (verified by
    :mod:`scripts.validate_b04a_protocol` and the B04A implementation
    tests).  No BatchNorm, no Dropout, no Xception, no depthwise
    separable, no pretrained weights, no external downloads.
    """

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.model_version = DEEPLABV3PLUS_LITE_VERSION
        self.variant = "option_A_plain_atrous_Conv2d"

        # ------------------------------------------------------------------
        # Stem (low-level features, full input resolution)
        # ------------------------------------------------------------------
        self.stem_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True)
        self.stem_relu1 = nn.ReLU(inplace=True)
        self.stem_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.stem_relu2 = nn.ReLU(inplace=True)

        # ------------------------------------------------------------------
        # Downsample (stride=2)
        # ------------------------------------------------------------------
        self.down_conv = nn.Conv2d(
            16, 32, kernel_size=3, stride=2, padding=1, bias=True
        )
        self.down_relu = nn.ReLU(inplace=True)

        # ------------------------------------------------------------------
        # ASPP module
        # ------------------------------------------------------------------
        self.aspp = _AsppModule(
            in_channels=32,
            out_channels_per_branch=DEEPLABV3PLUS_LITE_BRANCH_CHANNELS,
            atrous_rates=DEEPLABV3PLUS_LITE_ATROUS_RATES,
            target_size=(96, 42),
        )

        # ------------------------------------------------------------------
        # Low-level projection (full input resolution)
        # ------------------------------------------------------------------
        self.low_level_proj_conv = nn.Conv2d(
            16, 16, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.low_level_proj_relu = nn.ReLU(inplace=True)

        # ------------------------------------------------------------------
        # Decoder fusion (48 → 32 → 32)
        # ------------------------------------------------------------------
        self.decoder_conv1 = nn.Conv2d(48, 32, kernel_size=3, padding=1, bias=True)
        self.decoder_relu1 = nn.ReLU(inplace=True)
        self.decoder_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.decoder_relu2 = nn.ReLU(inplace=True)

        # ------------------------------------------------------------------
        # Final 1x1 conv to logits
        # ------------------------------------------------------------------
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1, bias=True)

        # Recorded explicit recovery targets (audited by tests).
        self._input_shape = INPUT_SHAPE
        self._aspp_input_shape = (96, 42)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-normal conv init + zero bias (R03 contract)."""

        _init_conv_kaiming_zero_bias(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fail-closed input validation."""

        _validate_input_tensor(x)

        # ------------------------------------------------------------------
        # Stem (low-level features)
        # ------------------------------------------------------------------
        low_level = self.stem_relu1(self.stem_conv1(x))
        low_level = self.stem_relu2(self.stem_conv2(low_level))

        # ------------------------------------------------------------------
        # Downsample
        # ------------------------------------------------------------------
        x = self.down_relu(self.down_conv(low_level))
        if (x.shape[2], x.shape[3]) != self._aspp_input_shape:
            raise RuntimeError(
                f"DeepLabV3+-lite downsample shape mismatch: expected "
                f"{self._aspp_input_shape}, got ({x.shape[2]}, {x.shape[3]})"
            )

        # ------------------------------------------------------------------
        # ASPP
        # ------------------------------------------------------------------
        x = self.aspp(x)

        # ------------------------------------------------------------------
        # Low-level projection (independent branch — same input as stem).
        # ------------------------------------------------------------------
        low_level_proj = self.low_level_proj_relu(self.low_level_proj_conv(low_level))
        if (low_level_proj.shape[2], low_level_proj.shape[3]) != INPUT_SHAPE:
            raise RuntimeError(
                f"DeepLabV3+-lite low-level projection shape mismatch: "
                f"expected {INPUT_SHAPE}, got "
                f"({low_level_proj.shape[2]}, {low_level_proj.shape[3]})"
            )

        # ------------------------------------------------------------------
        # Decoder fusion
        # ------------------------------------------------------------------
        if (x.shape[2], x.shape[3]) != (96, 42):
            raise RuntimeError(
                f"DeepLabV3+-lite pre-upsample ASPP shape mismatch: "
                f"expected (96, 42), got ({x.shape[2]}, {x.shape[3]})"
            )
        x = F.interpolate(
            x, size=INPUT_SHAPE, mode="bilinear", align_corners=False
        )
        if (x.shape[2], x.shape[3]) != (low_level_proj.shape[2], low_level_proj.shape[3]):
            raise RuntimeError(
                f"DeepLabV3+-lite decoder fusion spatial mismatch: "
                f"upsample is ({x.shape[2]}, {x.shape[3]}) but low-level "
                f"projection is "
                f"({low_level_proj.shape[2]}, {low_level_proj.shape[3]})"
            )
        x = torch.cat([x, low_level_proj], dim=1)
        if x.shape[1] != 48:
            raise RuntimeError(
                f"DeepLabV3+-lite decoder concat channel mismatch: "
                f"expected 48, got {x.shape[1]}"
            )
        x = self.decoder_relu1(self.decoder_conv1(x))
        x = self.decoder_relu2(self.decoder_conv2(x))

        logits = self.final_conv(x)

        expected = torch.Size([logits.shape[0], self.n_classes] + list(INPUT_SHAPE))
        if logits.shape != expected:
            raise ValueError(
                f"Output shape mismatch: expected {expected}, got {logits.shape}"
            )
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax predictions of shape ``[N, H, W]``."""

        logits = self.forward(x)
        return logits.argmax(dim=1)

    def get_config(self) -> dict[str, Any]:
        """Return the model configuration as a JSON-safe dict."""

        return {
            "model_version": self.model_version,
            "variant": self.variant,
            "n_classes": self.n_classes,
            "input_shape": list(INPUT_SHAPE),
            "parameter_count": self.count_parameters(),
            "aspp": {
                "atrous_rates": list(self.aspp.atrous_rates),
                "n_atrous_branches": len(self.aspp.atrous_rates),
                "include_1x1_pointwise_branch": True,
                "include_gap_branch": True,
                "output_channels_per_branch": self.aspp.out_channels_per_branch,
                "concat_in_channels": self.aspp.concat_in_channels,
                "post_concat_out_channels": 32,
            },
            "architecture": [
                {"stage": "stem", "blocks": [
                    {"layer": "stem_conv1", "in": 1, "out": 16, "kernel": 3, "padding": 1},
                    {"layer": "stem_relu1", "type": "ReLU"},
                    {"layer": "stem_conv2", "in": 16, "out": 16, "kernel": 3, "padding": 1},
                    {"layer": "stem_relu2", "type": "ReLU"},
                ]},
                {"stage": "down", "blocks": [
                    {"layer": "down_conv", "in": 16, "out": 32, "kernel": 3, "stride": 2, "padding": 1},
                    {"layer": "down_relu", "type": "ReLU"},
                ]},
                {"stage": "aspp", "blocks": [
                    {"layer": "branch_pointwise", "in": 32, "out": 16, "kernel": 1, "groups": 1},
                    *[{"layer": f"branch_atrous_rate_{rate}", "in": 32, "out": 16, "kernel": 3,
                       "dilation": rate, "padding": rate, "groups": 1}
                      for rate in self.aspp.atrous_rates],
                    {"layer": "branch_gap", "ops": [
                        "AdaptiveAvgPool2d(1x1)",
                        "Conv2d 32→16 (k=1,groups=1)",
                        "BilinearInterpolate to (96, 42)",
                    ]},
                    {"layer": "concat", "result_channels": self.aspp.concat_in_channels},
                    {"layer": "post_concat_conv", "in": self.aspp.concat_in_channels, "out": 32, "kernel": 1, "groups": 1},
                    {"layer": "post_concat_relu", "type": "ReLU"},
                ]},
                {"stage": "low_level_proj", "blocks": [
                    {"layer": "low_level_proj_conv", "in": 16, "out": 16, "kernel": 1, "groups": 1},
                    {"layer": "low_level_proj_relu", "type": "ReLU"},
                ]},
                {"stage": "decoder_fusion", "blocks": [
                    {"layer": "upsample", "type": "F.interpolate_bilinear", "size": list(INPUT_SHAPE), "align_corners": False},
                    {"layer": "concat", "channels": 32 + 16},
                    {"layer": "decoder_conv1", "in": 48, "out": 32, "kernel": 3, "padding": 1, "groups": 1},
                    {"layer": "decoder_relu1", "type": "ReLU"},
                    {"layer": "decoder_conv2", "in": 32, "out": 32, "kernel": 3, "padding": 1, "groups": 1},
                    {"layer": "decoder_relu2", "type": "ReLU"},
                ]},
                {"stage": "final", "blocks": [
                    {"layer": "final_conv", "in": 32, "out": self.n_classes, "kernel": 1, "groups": 1},
                ]},
            ],
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        """Return total number of parameters (including frozen)."""

        return sum(p.numel() for p in self.parameters())


def create_slp8_deeplabv3plus_lite(
    n_classes: int = N_CLASSES,
    device: str = "cpu",
) -> tuple["Slp8DeepLabV3PlusLite", dict[str, Any]]:
    """Construct a :class:`Slp8DeepLabV3PlusLite` and move to ``device``."""

    model = Slp8DeepLabV3PlusLite(n_classes=n_classes)
    model = model.to(device)
    config = model.get_config()
    config["device"] = str(device)
    return model, config


# ---------------------------------------------------------------------------
# B04A: Builder wiring and registry
# ---------------------------------------------------------------------------


def _build_slp8_resunet_lite(
    n_classes: int, device: str
) -> tuple[nn.Module, dict[str, Any]]:
    model, cfg = create_slp8_resunet_lite(n_classes=n_classes, device=device)
    return model, cfg


def _build_slp8_deeplabv3plus_lite(
    n_classes: int, device: str
) -> tuple[nn.Module, dict[str, Any]]:
    model, cfg = create_slp8_deeplabv3plus_lite(n_classes=n_classes, device=device)
    return model, cfg


register_model_builder(
    ModelBuilder(
        name=RESUNET_LITE_VERSION,
        version=RESUNET_LITE_VERSION,
        factory=_build_slp8_resunet_lite,
    )
)
register_model_builder(
    ModelBuilder(
        name=DEEPLABV3PLUS_LITE_VERSION,
        version=DEEPLABV3PLUS_LITE_VERSION,
        factory=_build_slp8_deeplabv3plus_lite,
    )
)
