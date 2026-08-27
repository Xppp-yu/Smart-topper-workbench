"""Tests for SLP8 Region Segmentation Model (TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1).

Tests cover:
1. Model output [N,9,192,84]
2. Wrong input shape failure
3. Non-finite input failure
4. Finite loss and backward
5. Parameter change
6. Model config
7. B04 model registry wiring (TinyFCN + SmallUNet, B04_MAX_PARAMETERS cap)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from topper_perception.neural.slp8_region_models import (
    B04_MAX_PARAMETERS,
    INPUT_SHAPE,
    MODEL_REGISTRY,
    MODEL_VERSION,
    N_CLASSES,
    SMALL_UNET_VERSION,
    Slp8SmallUnet,
    Slp8TinyFcn,
    compute_param_diff,
    create_loss_fn,
    create_slp8_small_unet,
    create_slp8_tiny_fcn,
    get_model_builder,
    list_model_builders,
    verify_model_gradient_flow,
    verify_model_output_shape,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> Slp8TinyFcn:
    """Create a fresh model for each test."""
    return Slp8TinyFcn(n_classes=N_CLASSES)


@pytest.fixture
def valid_input() -> torch.Tensor:
    """Create a valid input tensor."""
    return torch.randn(2, 1, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float32)


@pytest.fixture
def valid_labels() -> torch.Tensor:
    """Create valid labels."""
    labels = torch.zeros(2, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.long)
    # Add some foreground regions
    labels[:, 20:50, 10:40] = 1  # HEAD_NECK
    labels[:, 50:100, 10:74] = 2  # SHOULDER
    return labels


# ---------------------------------------------------------------------------
# Test: Model instantiation
# ---------------------------------------------------------------------------


class TestModelInstantiation:
    """Test model creation."""

    def test_model_creation(self, model):
        assert isinstance(model, nn.Module)
        assert model.n_classes == N_CLASSES
        assert model.model_version == MODEL_VERSION

    def test_model_trainable(self, model):
        # All parameters should require gradients
        for p in model.parameters():
            assert p.requires_grad

    def test_parameter_count(self, model):
        count = model.count_parameters()
        assert count > 0

        # Should be relatively small (no pretrained weights)
        assert count < 100000


# ---------------------------------------------------------------------------
# Test: Model config
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Test model configuration."""

    def test_get_config(self, model):
        config = model.get_config()

        assert config["model_version"] == MODEL_VERSION
        assert config["n_classes"] == N_CLASSES
        assert config["input_shape"] == list(INPUT_SHAPE)
        assert "architecture" in config
        assert len(config["architecture"]) == 5  # 3 conv + 2 relu


# ---------------------------------------------------------------------------
# Test: Forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    """Test forward pass with valid input."""

    def test_forward_valid_input(self, model, valid_input):
        output = model(valid_input)

        # Check output shape: [N, 9, 192, 84]
        assert output.shape == (2, N_CLASSES, INPUT_SHAPE[0], INPUT_SHAPE[1])
        assert output.dtype == torch.float32

    def test_forward_logits_not_softmax(self, model, valid_input):
        output = model(valid_input)

        # Logits can be any value (not normalized to sum=1)
        assert not torch.allclose(output.sum(dim=1), torch.ones(2, 192, 84), atol=0.1)

    def test_forward_single_sample(self, model):
        input_single = torch.randn(1, 1, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float32)
        output = model(input_single)
        assert output.shape == (1, N_CLASSES, INPUT_SHAPE[0], INPUT_SHAPE[1])


# ---------------------------------------------------------------------------
# Test: Fail-closed validation
# ---------------------------------------------------------------------------


class TestFailClosedValidation:
    """Test fail-closed input validation."""

    def test_wrong_ndim_rejected(self, model):
        # 3D input instead of 4D
        bad_input = torch.randn(2, 1, 192, dtype=torch.float32)

        with pytest.raises(ValueError, match="Input must be 4D"):
            model(bad_input)

    def test_wrong_channel_rejected(self, model):
        # Wrong number of channels
        bad_input = torch.randn(2, 3, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float32)

        with pytest.raises(ValueError, match="Input channel must be 1"):
            model(bad_input)

    def test_wrong_spatial_shape_rejected(self, model):
        # Wrong spatial dimensions
        bad_input = torch.randn(2, 1, 100, 100, dtype=torch.float32)

        with pytest.raises(ValueError, match="Input spatial shape must be"):
            model(bad_input)

    def test_wrong_dtype_rejected(self, model):
        # Wrong dtype (float64 instead of float32)
        bad_input = torch.randn(2, 1, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float64)

        with pytest.raises(ValueError, match="Input dtype must be torch.float32"):
            model(bad_input)

    def test_nan_input_rejected(self, model):
        bad_input = torch.randn(2, 1, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float32)
        bad_input[0, 0, 0, 0] = float("nan")

        with pytest.raises(ValueError, match="non-finite values"):
            model(bad_input)

    def test_inf_input_rejected(self, model):
        bad_input = torch.randn(2, 1, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float32)
        bad_input[0, 0, 0, 0] = float("inf")

        with pytest.raises(ValueError, match="non-finite values"):
            model(bad_input)


# ---------------------------------------------------------------------------
# Test: Predict function
# ---------------------------------------------------------------------------


class TestPredict:
    """Test argmax prediction."""

    def test_predict_argmax(self, model, valid_input):
        predictions = model.predict(valid_input)

        # Should have shape [N, 192, 84]
        assert predictions.shape == (2, INPUT_SHAPE[0], INPUT_SHAPE[1])
        assert predictions.dtype == torch.long

        # All values should be in [0, 8]
        assert predictions.min() >= 0
        assert predictions.max() < N_CLASSES


# ---------------------------------------------------------------------------
# Test: Training step
# ---------------------------------------------------------------------------


class TestTrainingStep:
    """Test training with loss and backward."""

    def test_loss_computation(self, model, valid_input, valid_labels):
        logits = model(valid_input)

        loss_fn = create_loss_fn()
        B, C, H, W = logits.shape
        logits_flat = logits.reshape(B, C, H * W)
        labels_flat = valid_labels.reshape(B, H * W)

        loss = loss_fn(logits_flat, labels_flat)

        assert loss.item() >= 0
        assert loss.requires_grad

    def test_backward_pass(self, model, valid_input, valid_labels):
        model.train()

        logits = model(valid_input)
        loss_fn = create_loss_fn()

        B, C, H, W = logits.shape
        logits_flat = logits.reshape(B, C, H * W)
        labels_flat = valid_labels.reshape(B, H * W)

        loss = loss_fn(logits_flat, labels_flat)
        loss.backward()

        # Check all parameters have gradients
        all_have_grad = all(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert all_have_grad

        # Clean up
        model.zero_grad()

    def test_gradient_flow_all_layers(self, model, valid_input, valid_labels):
        loss_fn = create_loss_fn()
        has_grads = verify_model_gradient_flow(
            model, valid_input, valid_labels, loss_fn
        )
        assert has_grads


# ---------------------------------------------------------------------------
# Test: Parameter change
# ---------------------------------------------------------------------------


class TestParameterChange:
    """Test that training changes parameters."""

    def test_parameters_change_after_training(self, model, valid_input, valid_labels):
        # Capture initial state
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Train for one step
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        loss_fn = create_loss_fn()

        logits = model(valid_input)
        B, C, H, W = logits.shape
        logits_flat = logits.reshape(B, C, H * W)
        labels_flat = valid_labels.reshape(B, H * W)

        loss = loss_fn(logits_flat, labels_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check parameters changed
        diff = compute_param_diff(model, initial_state)
        assert diff["_total"] > 1e-6

    def test_parameters_unchanged_without_training(self, model):
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Forward pass only
        input_tensor = torch.randn(1, 1, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float32)
        with torch.no_grad():
            model(input_tensor)

        # Parameters should be unchanged
        diff = compute_param_diff(model, initial_state)
        assert diff["_total"] < 1e-10


# ---------------------------------------------------------------------------
# Test: Model factory
# ---------------------------------------------------------------------------


class TestModelFactory:
    """Test model creation factory."""

    def test_create_model_cpu(self):
        model, config = create_slp8_tiny_fcn(n_classes=9, device="cpu")

        assert isinstance(model, Slp8TinyFcn)
        assert config["device"] == "cpu"

    def test_create_model_custom_n_classes(self):
        model, config = create_slp8_tiny_fcn(n_classes=5, device="cpu")

        assert model.n_classes == 5


# ---------------------------------------------------------------------------
# Test: Loss function
# ---------------------------------------------------------------------------


class TestLossFunction:
    """Test loss function creation."""

    def test_create_loss_fn(self):
        loss_fn = create_loss_fn()
        assert isinstance(loss_fn, nn.CrossEntropyLoss)


# ---------------------------------------------------------------------------
# Test: Output shape verification
# ---------------------------------------------------------------------------


class TestOutputShapeVerification:
    """Test output shape verification helper."""

    def test_verify_model_output_shape_valid(self, model):
        input_tensor = torch.randn(2, 1, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float32)
        assert verify_model_output_shape(model, input_tensor) is True

    def test_verify_model_output_shape_wrong(self, model):
        # Wrong input shape raises ValueError
        input_tensor = torch.randn(2, 1, 100, 100, dtype=torch.float32)
        with pytest.raises(ValueError, match="Input spatial shape"):
            verify_model_output_shape(model, input_tensor)


# ---------------------------------------------------------------------------
# Test: Model determinism
# ---------------------------------------------------------------------------


class TestModelDeterminism:
    """Test model determinism with same input."""

    def test_same_input_same_output(self, model):
        input_tensor = torch.randn(1, 1, INPUT_SHAPE[0], INPUT_SHAPE[1], dtype=torch.float32)

        # Set to eval mode
        model.eval()
        torch.manual_seed(42)

        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)

        assert torch.allclose(output1, output2)


# ---------------------------------------------------------------------------
# Test: B04 model registry (Candidate A and Candidate B)
# ---------------------------------------------------------------------------


class TestB04ModelRegistry:
    """The B04 candidate registry is registered and obeys the parameter cap."""

    def test_registry_contains_both_candidates(self):
        names = list_model_builders()
        assert MODEL_VERSION in names
        assert SMALL_UNET_VERSION in names

    def test_get_model_builder_returns_known_versions(self):
        for name, expected_version in (
            (MODEL_VERSION, MODEL_VERSION),
            (SMALL_UNET_VERSION, SMALL_UNET_VERSION),
        ):
            builder = get_model_builder(name)
            assert builder.name == name
            assert builder.version == expected_version

    def test_get_model_builder_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model_builder("slp8_nope_v0.0")

    def test_register_duplicate_raises(self):
        from topper_perception.neural.slp8_region_models import (
            ModelBuilder, register_model_builder, MODEL_REGISTRY,
        )
        # Re-registering an existing name should fail.
        with pytest.raises(ValueError, match="already registered"):
            register_model_builder(
                ModelBuilder(
                    name=MODEL_VERSION,
                    version=MODEL_VERSION,
                    factory=lambda n_classes, device: Slp8TinyFcn(n_classes=n_classes),
                )
            )

    def test_small_unet_in_registry(self):
        builder = get_model_builder(SMALL_UNET_VERSION)
        model, cfg = builder.factory(N_CLASSES, "cpu")
        assert isinstance(model, Slp8SmallUnet)
        assert cfg["model_version"] == SMALL_UNET_VERSION
        assert cfg["parameter_count"] <= B04_MAX_PARAMETERS

    def test_tiny_fcn_in_registry(self):
        builder = get_model_builder(MODEL_VERSION)
        model, cfg = builder.factory(N_CLASSES, "cpu")
        assert isinstance(model, Slp8TinyFcn)
        assert cfg["model_version"] == MODEL_VERSION
        assert cfg["parameter_count"] <= B04_MAX_PARAMETERS

    def test_small_unet_factory(self):
        model, cfg = create_slp8_small_unet(device="cpu")
        assert model.n_classes == N_CLASSES
        assert cfg["device"] == "cpu"
        assert cfg["parameter_count"] <= B04_MAX_PARAMETERS
        # Output shape sanity check
        x = torch.randn(1, 1, 192, 84, dtype=torch.float32)
        y = model(x)
        assert y.shape == (1, N_CLASSES, 192, 84)

    def test_b04_max_parameters_is_frozen(self):
        # The frozen cap is 150,000.  Changing this would break the
        # B04 contract; the test pins the value.
        assert B04_MAX_PARAMETERS == 150_000

    def test_both_candidates_pass_parameter_cap(self):
        for name in (MODEL_VERSION, SMALL_UNET_VERSION):
            builder = get_model_builder(name)
            model, _ = builder.factory(N_CLASSES, "cpu")
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert count <= B04_MAX_PARAMETERS, f"{name} has {count} parameters"

    def test_small_unet_no_batchnorm_no_dropout(self):
        model = Slp8SmallUnet()
        for module in model.modules():
            assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            assert not isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d))

    def test_small_unet_skip1_keeps_input_resolution(self):
        # Architectural check: skip1 has the same spatial size as the
        # input.  This is the "explicit spatial size recovery" the B04
        # contract requires (forbidding scale_factor-driven upsample).
        m = Slp8SmallUnet()
        assert m._skip1_shape == (192, 84)
        assert m._skip2_shape == (96, 42)
        assert m._bottleneck_shape == (48, 21)

    def test_small_unet_odd_width_recovery(self):
        # The 84 width after two MaxPool2d(2) steps (84 -> 42 -> 21)
        # must be recovered exactly via the explicit F.interpolate size,
        # not by `scale_factor=2` (which would yield 84 by luck but is
        # brittle).  Two forward passes with the same input must be
        # deterministic and produce the same shape.
        m = Slp8SmallUnet()
        m.eval()
        x = torch.randn(2, 1, 192, 84, dtype=torch.float32)
        with torch.no_grad():
            y1 = m(x)
            y2 = m(x)
        assert y1.shape == (2, N_CLASSES, 192, 84)
        assert y1.shape[3] == 84  # the odd width is preserved
        assert torch.equal(y1, y2)

    def test_small_unet_pretrained_attr_absent(self):
        # The model must not carry any field that could be interpreted
        # as a pretrained/external download.
        m = Slp8SmallUnet()
        cfg = m.get_config()
        for forbidden in ("pretrained", "checkpoint_url", "external_weights", "url"):
            assert forbidden not in cfg
        for attr in dir(m):
            assert "pretrained" not in attr.lower() or attr in {
                "model_version", "_init_weights",
            }
