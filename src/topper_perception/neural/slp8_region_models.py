"""SLP8 Region Segmentation Models (TASK-SLP-B03/B04).

This module provides the SLP8 pressure-only region segmentation model
registry used by the B03 smoke and the B04 PM-only Region Mini protocol.

* :class:`Slp8TinyFcn` — minimal fully-convolutional network used as the
  B03 smoke architecture and as ``slp8_tiny_fcn_v0.1`` (Candidate A) of
  the B04 Mini protocol.

* :class:`Slp8SmallUnet` — small encoder/decoder with explicit bilinear
  spatial-size recovery introduced as ``slp8_small_unet_v0.1`` (Candidate B)
  in the B04 Mini protocol.

Key properties (applies to both):

* No pretrained weights, no external downloads.
* Fail-closed input validation (shape, dtype, finiteness).
* Parameter count is exposed via :meth:`count_parameters` for the B04
  150,000-parameter budget.
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

#: Candidate B model version (B04 SmallUNet).
SMALL_UNET_VERSION = "slp8_small_unet_v0.1"

#: Hard parameter-count cap for B04 Mini candidates.
B04_MAX_PARAMETERS = 150_000

#: Registered model builders (frozen by the B04 protocol).
MODEL_REGISTRY: dict[str, "ModelBuilder"] = {}


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
