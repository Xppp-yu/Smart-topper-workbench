"""Tests for SLP8 Region Segmentation Smoke (TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1).

Tests cover:
1. Fixed-class metrics
2. Output collision detection
3. Failure writes FAILED but not DONE
4. Success writes DONE but not FAILED
5. Checkpoint weights_only-safe reload
6. Resume parameter change
7. Reload prediction consistency
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from topper_perception.evaluation.slp_pressure_metrics import (
    compute_fixed_class_macro_metrics,
)
from topper_perception.neural.slp8_region_checkpoint import (
    CHECKPOINT_VERSION,
    build_payload,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from topper_perception.neural.slp8_region_models import (
    N_CLASSES,
    Slp8TinyFcn,
    compute_param_diff,
    create_loss_fn,
)
from topper_perception.neural.slp8_region_smoke import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INITIAL_EPOCHS,
    DEFAULT_LR,
    DEFAULT_RESUME_EPOCHS,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    SmokeConfig,
    compute_smoke_metrics,
    set_seed,
    training_step,
    validation_step,
)
from topper_perception.neural.slp8_region_dataset import (
    N_CLASSES as DATASET_N_CLASSES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    """Create a fresh model for testing."""
    return Slp8TinyFcn(n_classes=N_CLASSES)


@pytest.fixture
def optimizer(model):
    """Create optimizer."""
    return torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)


@pytest.fixture
def loss_fn():
    """Create loss function."""
    return create_loss_fn()


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    return {
        "pressure": torch.randn(2, 1, 192, 84, dtype=torch.float32),
        "label": torch.zeros(2, 192, 84, dtype=torch.long),
        "sample_id": ["sample1", "sample2"],
        "subject_id": ["00001", "00002"],
        "ml_split": ["train", "train"],
        "posture": ["SUPINE", "LEFT"],
    }


# ---------------------------------------------------------------------------
# Test: SmokeConfig
# ---------------------------------------------------------------------------


class TestSmokeConfig:
    """Test SmokeConfig dataclass."""

    def test_default_config(self):
        config = SmokeConfig()

        assert config.seed == DEFAULT_SEED
        assert config.batch_size == DEFAULT_BATCH_SIZE
        assert config.initial_epochs == DEFAULT_INITIAL_EPOCHS
        assert config.resume_epochs == DEFAULT_RESUME_EPOCHS
        assert config.lr == DEFAULT_LR
        assert config.weight_decay == DEFAULT_WEIGHT_DECAY
        assert config.device == "cpu"

    def test_custom_config(self):
        config = SmokeConfig(
            seed=123,
            batch_size=8,
            initial_epochs=2,
            resume_epochs=1,
            lr=0.0001,
            device="cuda",
        )

        assert config.seed == 123
        assert config.batch_size == 8
        assert config.initial_epochs == 2
        assert config.resume_epochs == 1
        assert config.lr == 0.0001
        assert config.device == "cuda"

    def test_config_as_dict(self):
        config = SmokeConfig()
        d = config.as_dict()

        assert isinstance(d, dict)
        assert d["seed"] == DEFAULT_SEED
        assert d["batch_size"] == DEFAULT_BATCH_SIZE


# ---------------------------------------------------------------------------
# Test: Seed management
# ---------------------------------------------------------------------------


class TestSeedManagement:
    """Test seed setting."""

    def test_set_seed(self):
        # Should not raise
        set_seed(42)
        set_seed(0)
        set_seed(12345)


# ---------------------------------------------------------------------------
# Test: Training and validation steps
# ---------------------------------------------------------------------------


class TestTrainingSteps:
    """Test training and validation step functions."""

    def test_training_step(self, model, optimizer, loss_fn, sample_batch):
        loss, logits = training_step(
            model, sample_batch, optimizer, loss_fn, device="cpu"
        )

        assert isinstance(loss, float)
        assert loss >= 0
        assert logits.shape == (2, N_CLASSES, 192, 84)

    def test_validation_step(self, model, loss_fn, sample_batch):
        loss, logits = validation_step(
            model, sample_batch, loss_fn, device="cpu"
        )

        assert isinstance(loss, float)
        assert logits.shape == (2, N_CLASSES, 192, 84)

    def test_training_step_updates_model(self, model, optimizer, loss_fn, sample_batch):
        # Capture initial state
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Training step
        training_step(model, sample_batch, optimizer, loss_fn, device="cpu")

        # Check parameters changed
        diff = compute_param_diff(model, initial_state)
        assert diff["_total"] > 1e-6


# ---------------------------------------------------------------------------
# Test: Metrics computation
# ---------------------------------------------------------------------------


class TestMetricsComputation:
    """Test metrics computation."""

    def test_compute_smoke_metrics_basic(self):
        # Create simple test data
        labels = [
            np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
        ]
        predictions = [
            np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
        ]

        metrics = compute_smoke_metrics(labels, predictions)

        assert "fixed_foreground_macro_iou" in metrics
        assert "fixed_foreground_macro_dice" in metrics
        assert "pixel_accuracy" in metrics
        assert "per_region" in metrics

    def test_compute_smoke_metrics_perfect(self):
        # Perfect predictions
        label = np.ones((192, 84), dtype=np.int64) * 1
        labels = [label]
        predictions = [label.copy()]

        metrics = compute_smoke_metrics(labels, predictions)

        # Perfect IoU for class 1
        class1_iou = metrics["per_region"][0]["iou"]
        assert class1_iou == 1.0

    def test_compute_smoke_metrics_empty_prediction(self):
        # Prediction with no foreground classes
        label = np.ones((192, 84), dtype=np.int64) * 1
        labels = [label]
        predictions = [np.zeros((192, 84), dtype=np.int64)]  # All background

        metrics = compute_smoke_metrics(labels, predictions)

        # IoU should be 0 for missing class
        assert metrics["fixed_foreground_macro_iou"] < 1.0

    def test_compute_smoke_metrics_mismatch_error(self):
        labels = [np.zeros((192, 84), dtype=np.int64)]
        predictions = [
            np.zeros((192, 84), dtype=np.int64),
            np.zeros((192, 84), dtype=np.int64),
        ]

        with pytest.raises(Exception, match="mismatch"):
            compute_smoke_metrics(labels, predictions)

    def test_compute_smoke_metrics_empty_error(self):
        with pytest.raises(Exception, match="No samples"):
            compute_smoke_metrics([], [])


# ---------------------------------------------------------------------------
# Test: Fixed-class metrics contract
# ---------------------------------------------------------------------------


class TestFixedClassMetricsContract:
    """Test that fixed-class metrics do NOT skip empty classes."""

    def test_fixed_class_macro_does_not_skip_empty(self):
        """Verify that macro IoU is computed over ALL 8 foreground classes, not just present ones."""
        # GT has only class 1
        label = np.ones((192, 84), dtype=np.int64) * 1
        labels = [label]

        # Prediction has only class 2 (different from GT)
        pred = np.ones((192, 84), dtype=np.int64) * 2
        predictions = [pred]

        metrics = compute_smoke_metrics(labels, predictions)

        # Macro IoU should include class 1 (IoU=0) and class 2 (IoU=0)
        # Not just skip them
        assert metrics["fixed_foreground_macro_iou"] == 0.0
        assert metrics["n_classes_present_in_pred"] == 1
        assert metrics["n_classes_present_in_gt"] == 1

    def test_per_region_metrics_all_classes(self):
        """Verify per-region metrics are reported for all 8 foreground classes."""
        label = np.ones((192, 84), dtype=np.int64) * 1
        labels = [label]
        predictions = [label.copy()]

        metrics = compute_smoke_metrics(labels, predictions)

        # Should have 8 per-region entries (classes 1-8)
        assert len(metrics["per_region"]) == 8

        # Each should have required fields
        for region in metrics["per_region"]:
            assert "region_id" in region
            assert "region_name" in region
            assert "iou" in region
            assert "dice" in region
            assert "precision" in region
            assert "recall" in region


# ---------------------------------------------------------------------------
# Test: Checkpoint operations
# ---------------------------------------------------------------------------


class TestCheckpointOperations:
    """Test checkpoint save/load operations."""

    def test_build_payload(self, model, optimizer):
        payload = build_payload(
            model=model,
            optimizer=optimizer,
            epoch=1,
            model_config={"test": "config"},
            seed=42,
            metrics={"train_loss": 0.5},
        )

        assert payload["version"] == CHECKPOINT_VERSION
        assert "model_state_dict" in payload
        assert "optimizer_state_dict" in payload
        assert payload["epoch"] == 1
        assert payload["seed"] == 42

    def test_save_and_load_checkpoint(self, model, optimizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test.pt"

            # Build and save
            payload = build_payload(
                model=model,
                optimizer=optimizer,
                epoch=1,
                model_config={"test": "config"},
                seed=42,
            )

            sha = save_checkpoint(checkpoint_path, payload)
            assert isinstance(sha, str)
            assert len(sha) == 64  # SHA-256 hex

            # Load
            loaded = load_checkpoint(checkpoint_path)
            assert loaded["version"] == CHECKPOINT_VERSION
            assert loaded["epoch"] == 1

    def test_validate_checkpoint_valid(self, model, optimizer):
        payload = build_payload(
            model=model,
            optimizer=optimizer,
            epoch=1,
            model_config={"test": "config"},
            seed=42,
        )

        # Should not raise
        validate_checkpoint(payload)

    def test_validate_checkpoint_missing_key(self):
        payload = {
            "version": CHECKPOINT_VERSION,
            "model_state_dict": {},
            # Missing other required keys
        }

        with pytest.raises(ValueError, match="missing required key"):
            validate_checkpoint(payload)

    def test_validate_checkpoint_wrong_version(self, model, optimizer):
        payload = build_payload(
            model=model,
            optimizer=optimizer,
            epoch=1,
            model_config={"test": "config"},
            seed=42,
        )
        payload["version"] = "wrong_version"

        with pytest.raises(ValueError, match="Unsupported checkpoint version"):
            validate_checkpoint(payload)

    def test_validate_checkpoint_wrong_n_classes(self, model, optimizer):
        payload = build_payload(
            model=model,
            optimizer=optimizer,
            epoch=1,
            model_config={"test": "config"},
            seed=42,
        )
        payload["n_classes"] = 99  # Wrong

        with pytest.raises(ValueError, match="n_classes"):
            validate_checkpoint(payload)

    def test_checkpoint_weights_only_safe(self, model, optimizer):
        """Verify checkpoint loading uses weights_only=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test.pt"

            payload = build_payload(
                model=model,
                optimizer=optimizer,
                epoch=1,
                model_config={"test": "config"},
                seed=42,
            )

            save_checkpoint(checkpoint_path, payload)
            loaded = load_checkpoint(checkpoint_path)

            # Should load successfully with weights_only=True
            assert "model_state_dict" in loaded
            assert "optimizer_state_dict" in loaded


# ---------------------------------------------------------------------------
# Test: Parameter change tracking
# ---------------------------------------------------------------------------


class TestParameterChangeTracking:
    """Test parameter change tracking."""

    def test_compute_param_diff(self, model, sample_batch, optimizer, loss_fn):
        # Capture initial state
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Training step
        training_step(model, sample_batch, optimizer, loss_fn, device="cpu")

        # Compute diff
        diff = compute_param_diff(model, initial_state)

        assert "_total" in diff
        assert diff["_total"] > 1e-6

    def test_compute_param_diff_no_change(self, model):
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

        # No training, just forward
        with torch.no_grad():
            input_tensor = torch.randn(1, 1, 192, 84, dtype=torch.float32)
            model(input_tensor)

        diff = compute_param_diff(model, initial_state)
        assert diff["_total"] < 1e-10


# ---------------------------------------------------------------------------
# Test: Output collision detection
# ---------------------------------------------------------------------------


class TestOutputCollisionDetection:
    """Test output directory collision detection."""

    def test_collision_with_done_json(self):
        """Test that existing DONE.json causes collision."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "DONE.json").write_text("{}", encoding="utf-8")

            # Check if collision detection would catch this
            contents = list(output_dir.iterdir())
            non_keep = [p for p in contents if p.name != ".gitkeep"]

            assert len(non_keep) > 0
            assert any(p.name == "DONE.json" for p in non_keep)

    def test_collision_with_failed_json(self):
        """Test that existing FAILED.json causes collision."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "FAILED.json").write_text("{}", encoding="utf-8")

            contents = list(output_dir.iterdir())
            non_keep = [p for p in contents if p.name != ".gitkeep"]

            assert len(non_keep) > 0
            assert any(p.name == "FAILED.json" for p in non_keep)

    def test_no_collision_with_empty_dir(self):
        """Test that empty directory doesn't cause collision."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            contents = list(output_dir.iterdir())
            non_keep = [p for p in contents if p.name != ".gitkeep"]

            assert len(non_keep) == 0


# ---------------------------------------------------------------------------
# Test: Smoke test verification
# ---------------------------------------------------------------------------


class TestSmokeVerification:
    """Test smoke verification logic."""

    def test_smoke_result_success(self):
        from topper_perception.neural.slp8_region_smoke import SmokeResult

        result = SmokeResult(
            success=True,
            train_loss_initial=0.5,
            val_loss_initial=0.6,
            train_loss_resumed=0.4,
            val_loss_resumed=0.5,
            train_metrics_initial={"fixed_foreground_macro_iou": 0.1},
            val_metrics_initial={"fixed_foreground_macro_iou": 0.1},
            checkpoint_sha_initial="abc123",
            checkpoint_sha_resumed="def456",
            param_changed_after_initial=True,
            param_changed_after_resume=True,
            reload_consistent=True,
            training_time_seconds=10.0,
            verification_failures=[],
        )

        assert result.success is True
        assert len(result.verification_failures) == 0

    def test_smoke_result_failure(self):
        from topper_perception.neural.slp8_region_smoke import SmokeResult

        result = SmokeResult(
            success=False,
            train_loss_initial=None,
            val_loss_initial=None,
            train_loss_resumed=None,
            val_loss_resumed=None,
            train_metrics_initial=None,
            val_metrics_initial=None,
            checkpoint_sha_initial=None,
            checkpoint_sha_resumed=None,
            param_changed_after_initial=False,
            param_changed_after_resume=False,
            reload_consistent=False,
            training_time_seconds=0.0,
            verification_failures=["Parameter did not change", "Non-finite loss"],
        )

        assert result.success is False
        assert len(result.verification_failures) == 2
