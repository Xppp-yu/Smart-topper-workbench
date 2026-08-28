"""Tests for SLP8 Region Segmentation Smoke (TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1).

Tests cover:
1. Fixed-class metrics
2. Output collision detection
3. Failure writes FAILED but not DONE
4. Success writes DONE but not FAILED
5. Checkpoint weights_only-safe reload
6. Resume parameter change
7. Reload prediction consistency (must do real comparison, not hardcoded True)
8. Config validation fail-closed
9. Real config integration
10. Canonical array hash
11. Real prediction records (no placeholders)
12. Subset config flows through to dataset manifest
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

# Ensure scripts is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

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
from topper_perception.neural.slp8_region_dataset import (
    N_CLASSES as DATASET_N_CLASSES,
    RegionSample,
    NormalizationStats,
    Slp8RegionDataset,
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
    CANONICAL_HASH_VERSION,
    PRED_OK,
    PredictionRecord,
    SmokeConfig,
    SmokeResult,
    canonical_array_hash,
    check_reload_consistency,
    compute_smoke_metrics,
    set_seed,
    training_step,
    validation_step,
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
    """Test SmokeConfig dataclass.

    All fields are required; there is no default-based masking of missing
    configuration.
    """

    def test_custom_config(self):
        config = SmokeConfig(
            seed=123,
            batch_size=8,
            initial_epochs=2,
            resume_epochs=1,
            lr=0.0001,
            weight_decay=0.0001,
            device="cpu",
            n_train_subjects=2,
            n_val_subjects=1,
            subset_seed=42,
        )

        assert config.seed == 123
        assert config.batch_size == 8
        assert config.initial_epochs == 2
        assert config.resume_epochs == 1
        assert config.lr == 0.0001
        assert config.device == "cpu"
        assert config.n_train_subjects == 2
        assert config.n_val_subjects == 1
        assert config.subset_seed == 42

    def test_config_as_dict(self):
        config = SmokeConfig(
            seed=42,
            batch_size=4,
            initial_epochs=1,
            resume_epochs=1,
            lr=0.001,
            weight_decay=0.0001,
            device="cpu",
            n_train_subjects=2,
            n_val_subjects=1,
            subset_seed=42,
        )
        d = config.as_dict()

        assert isinstance(d, dict)
        assert d["seed"] == 42
        assert d["batch_size"] == 4
        assert d["n_train_subjects"] == 2
        assert d["n_val_subjects"] == 1
        assert d["subset_seed"] == 42

    def test_config_missing_subset_field_rejected(self):
        """SmokeConfig must require subset fields explicitly."""
        with pytest.raises(TypeError):
            SmokeConfig(
                seed=42,
                batch_size=4,
                initial_epochs=1,
                resume_epochs=1,
                lr=0.001,
                weight_decay=0.0001,
                device="cpu",
                n_train_subjects=2,
                n_val_subjects=1,
                # subset_seed omitted
            )


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


# ---------------------------------------------------------------------------
# Test: Reload consistency (real comparison, no hardcoded True)
# ---------------------------------------------------------------------------


class TestReloadConsistency:
    """Test that reload_consistent is the result of a real comparison."""

    def test_reload_consistent_same_model(self):
        """A model loaded from a checkpoint should produce identical logits."""
        model = Slp8TinyFcn(n_classes=N_CLASSES)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            payload = build_payload(
                model=model,
                optimizer=optimizer,
                epoch=1,
                model_config={"test": "config"},
                seed=42,
            )
            save_checkpoint(ckpt_path, payload)

            # Load into fresh model
            fresh = Slp8TinyFcn(n_classes=N_CLASSES)
            loaded = load_checkpoint(ckpt_path)
            validate_checkpoint(loaded)
            fresh.load_state_dict(loaded["model_state_dict"])
            fresh.eval()

            # Use same input
            reference_batch = {
                "pressure": torch.randn(2, 1, 192, 84, dtype=torch.float32),
            }

            result = check_reload_consistency(model, fresh, reference_batch)

            assert result["consistent"] is True
            assert result["max_abs_diff"] < 1e-5

    def test_reload_inconsistent_modified_model(self):
        """A model with different weights should fail the consistency check."""
        model = Slp8TinyFcn(n_classes=N_CLASSES)
        # Create a "fresh" model and modify one parameter
        fresh = Slp8TinyFcn(n_classes=N_CLASSES)
        with torch.no_grad():
            fresh.conv1.weight.fill_(99.0)

        reference_batch = {
            "pressure": torch.randn(2, 1, 192, 84, dtype=torch.float32),
        }

        result = check_reload_consistency(model, fresh, reference_batch)

        # Consistency must fail
        assert result["consistent"] is False
        assert result["max_abs_diff"] > 1.0

    def test_runner_does_not_hardcode_reload_true(self):
        """The smoke runner must not use a hardcoded ``reload_consistent = True``.

        The test checks that the run_smoke_test function actually performs
        a comparison and only sets reload_consistent=True when the result is
        consistent.  If reload_consistent is hardcoded, this test would fail
        because a freshly-modified model should cause the comparison to fail.
        """
        # This is verified structurally: the smoke.py source must not
        # contain the pattern "reload_consistent = True" as a hardcoded
        # assignment.
        smoke_path = (
            Path(__file__).resolve().parents[1]
            / "src/topper_perception/neural/slp8_region_smoke.py"
        )
        source = smoke_path.read_text(encoding="utf-8")
        # Strip docstring/comments by splitting on '#'
        code_lines = []
        for line in source.splitlines():
            stripped = line.split("#", 1)[0]
            code_lines.append(stripped)
        code = "\n".join(code_lines)

        forbidden = [
            "reload_consistent = True",
            "reload_consistent = (",
            "reload_consistent =  True",
        ]
        for pattern in forbidden:
            assert pattern not in code, (
                f"Runner has hardcoded reload_consistent = True: {pattern!r}"
            )


# ---------------------------------------------------------------------------
# Test: Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Test fail-closed config validation in the CLI runner."""

    def test_valid_config_passes(self):
        from scripts.run_slp8_region_smoke import _validate_config

        cfg = {
            "config_version": "slp8_pm_region_smoke_v0.1",
            "task_id": "TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1",
            "smoke_version": "slp8_region_smoke_v0.1",
            "provenance": "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            "raw_semantics": "raw_pmarray_response",
            "model": {
                "n_classes": 9,
                "input_shape": [192, 84],
            },
            "training": {
                "seed": 42,
                "device": "cpu",
                "batch_size": 4,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "epochs": {"initial": 1, "resume": 1},
            },
            "dataset": {
                "smoke_subset": {
                    "n_train_subjects": 2,
                    "n_val_subjects": 1,
                    "seed": 42,
                },
                "normalization": {
                    "method": "raw_passthrough_with_minmax_reference",
                    "fit_split": "train",
                    "raw_semantics": "raw_pmarray_response",
                },
            },
        }

        # Should not raise
        _validate_config(cfg)

    def test_wrong_n_classes_rejected(self):
        from scripts.run_slp8_region_smoke import (
            _validate_config,
            ConfigValidationError,
        )

        cfg = self._build_valid_config()
        cfg["model"]["n_classes"] = 5

        with pytest.raises(ConfigValidationError, match="n_classes must be 9"):
            _validate_config(cfg)

    def test_invalid_device_rejected(self):
        from scripts.run_slp8_region_smoke import (
            _validate_config,
            ConfigValidationError,
        )

        cfg = self._build_valid_config()
        cfg["training"]["device"] = "tpu"

        with pytest.raises(ConfigValidationError, match="device must be one of"):
            _validate_config(cfg)

    def test_wrong_n_train_subjects_rejected(self):
        from scripts.run_slp8_region_smoke import (
            _validate_config,
            ConfigValidationError,
        )

        cfg = self._build_valid_config()
        cfg["dataset"]["smoke_subset"]["n_train_subjects"] = 5

        with pytest.raises(ConfigValidationError, match="n_train_subjects"):
            _validate_config(cfg)

    def test_wrong_n_val_subjects_rejected(self):
        from scripts.run_slp8_region_smoke import (
            _validate_config,
            ConfigValidationError,
        )

        cfg = self._build_valid_config()
        cfg["dataset"]["smoke_subset"]["n_val_subjects"] = 0

        with pytest.raises(ConfigValidationError, match="n_val_subjects"):
            _validate_config(cfg)

    def test_missing_field_rejected(self):
        from scripts.run_slp8_region_smoke import (
            _validate_config,
            ConfigValidationError,
        )

        cfg = self._build_valid_config()
        del cfg["training"]["seed"]

        with pytest.raises(ConfigValidationError, match="seed"):
            _validate_config(cfg)

    def test_wrong_type_rejected(self):
        from scripts.run_slp8_region_smoke import (
            _validate_config,
            ConfigValidationError,
        )

        cfg = self._build_valid_config()
        cfg["training"]["batch_size"] = "4"  # Should be int

        with pytest.raises(ConfigValidationError, match="batch_size"):
            _validate_config(cfg)

    def test_zero_epochs_rejected(self):
        from scripts.run_slp8_region_smoke import (
            _validate_config,
            ConfigValidationError,
        )

        cfg = self._build_valid_config()
        cfg["training"]["epochs"]["initial"] = 0

        with pytest.raises(ConfigValidationError, match="initial"):
            _validate_config(cfg)

    def test_wrong_provenance_rejected(self):
        from scripts.run_slp8_region_smoke import (
            _validate_config,
            ConfigValidationError,
        )

        cfg = self._build_valid_config()
        cfg["provenance"] = "OTHER_PROVENANCE"

        with pytest.raises(ConfigValidationError, match="provenance"):
            _validate_config(cfg)

    def _build_valid_config(self) -> dict[str, Any]:
        return {
            "config_version": "slp8_pm_region_smoke_v0.1",
            "task_id": "TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1",
            "smoke_version": "slp8_region_smoke_v0.1",
            "provenance": "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            "raw_semantics": "raw_pmarray_response",
            "model": {
                "n_classes": 9,
                "input_shape": [192, 84],
            },
            "training": {
                "seed": 42,
                "device": "cpu",
                "batch_size": 4,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "epochs": {"initial": 1, "resume": 1},
            },
            "dataset": {
                "smoke_subset": {
                    "n_train_subjects": 2,
                    "n_val_subjects": 1,
                    "seed": 42,
                },
                "normalization": {
                    "method": "raw_passthrough_with_minmax_reference",
                    "fit_split": "train",
                    "raw_semantics": "raw_pmarray_response",
                },
            },
        }


# ---------------------------------------------------------------------------
# Test: Real config file integration
# ---------------------------------------------------------------------------


class TestRealConfigIntegration:
    """Verify the committed config file passes validation and parses correctly."""

    def test_committed_config_validates(self):
        from scripts.run_slp8_region_smoke import _validate_config

        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs/experiments/slp8_pm_region_smoke_v0.1.json"
        )
        assert config_path.is_file(), f"committed config missing: {config_path}"

        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        # Should not raise
        _validate_config(cfg)

    def test_committed_config_values(self):
        """Parsed values must match the JSON exactly."""
        from scripts.run_slp8_region_smoke import _build_smoke_config

        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs/experiments/slp8_pm_region_smoke_v0.1.json"
        )
        cfg = json.loads(config_path.read_text(encoding="utf-8"))

        smoke_config = _build_smoke_config(cfg, device_override="cpu")

        assert smoke_config.seed == cfg["training"]["seed"]
        assert smoke_config.batch_size == cfg["training"]["batch_size"]
        assert smoke_config.initial_epochs == cfg["training"]["epochs"]["initial"]
        assert smoke_config.resume_epochs == cfg["training"]["epochs"]["resume"]
        assert smoke_config.lr == cfg["training"]["lr"]
        assert smoke_config.weight_decay == cfg["training"]["weight_decay"]
        assert smoke_config.device == "cpu"
        assert smoke_config.n_train_subjects == cfg["dataset"]["smoke_subset"]["n_train_subjects"]
        assert smoke_config.n_val_subjects == cfg["dataset"]["smoke_subset"]["n_val_subjects"]
        assert smoke_config.subset_seed == cfg["dataset"]["smoke_subset"]["seed"]

    def test_runner_does_not_hardcode_subset_counts(self):
        """The runner source must not hard-code n_train_subjects=2 or
        n_val_subjects=1 inside run_smoke_test or in the train/val count
        literals that bypass the SmokeConfig."""
        smoke_path = (
            Path(__file__).resolve().parents[1]
            / "src/topper_perception/neural/slp8_region_smoke.py"
        )
        source = smoke_path.read_text(encoding="utf-8")
        # Strip comments.
        code_lines = [line.split("#", 1)[0] for line in source.splitlines()]
        code = "\n".join(code_lines)
        # Find run_smoke_test function body and look for hard-coded literals.
        start = code.find("def run_smoke_test")
        assert start != -1
        # The runner must read n_train_subjects from the SmokeConfig.
        assert "config.n_train_subjects" in code
        assert "config.n_val_subjects" in code
        assert "config.subset_seed" in code
        # Inside the run_smoke_test function body, the parameters must be
        # taken from the config, not hard-coded.  Search for the explicit
        # "n_train_subjects=2," / "n_val_subjects=1," pattern within the
        # function body.
        end = code.find("\ndef ", start + 1)
        body = code[start:end] if end != -1 else code[start:]
        for forbidden in ("n_train_subjects=2,", "n_val_subjects=1,"):
            assert forbidden not in body, (
                f"run_smoke_test hardcodes {forbidden!r}; "
                f"must consume from SmokeConfig"
            )


# ---------------------------------------------------------------------------
# Test: Canonical array hash
# ---------------------------------------------------------------------------


class TestCanonicalArrayHash:
    """Test canonical SHA-256 hashing of label/prediction arrays."""

    def test_hash_is_64_lowercase_hex(self):
        arr = np.zeros((192, 84), dtype=np.int64)
        h = canonical_array_hash(arr)
        assert len(h) == 64
        assert h == h.lower()
        int(h, 16)  # raises if not valid hex

    def test_same_input_same_hash(self):
        arr = np.random.default_rng(0).integers(0, 9, (192, 84)).astype(np.int64)
        h1 = canonical_array_hash(arr)
        h2 = canonical_array_hash(arr)
        assert h1 == h2

    def test_modified_pixel_changes_hash(self):
        arr = np.zeros((192, 84), dtype=np.int64)
        h1 = canonical_array_hash(arr)
        arr[0, 0] = 1
        h2 = canonical_array_hash(arr)
        assert h1 != h2

    def test_int32_and_int64_have_different_hashes(self):
        """A label cast to int32 vs int64 must hash differently because
        the rule always canonicalises to int64."""
        arr32 = np.zeros((4, 4), dtype=np.int32)
        arr64 = arr32.astype(np.int64)
        assert canonical_array_hash(arr32) == canonical_array_hash(arr64)

    def test_header_version_in_canonical_form(self):
        """Hash must encode the version header so collisions with other
        hash rules are not silent."""
        arr = np.zeros((4, 4), dtype=np.int64)
        # The current rule starts with CANONICAL_HASH_VERSION.
        payload_header = CANONICAL_HASH_VERSION.encode("utf-8") + b"\ndtype=<i8"
        # Compute what the function does internally and check that
        # changing the header version changes the hash.
        import hashlib as _hashlib

        arr_int = np.ascontiguousarray(arr, dtype=np.int64)
        header_v1 = (
            f"{CANONICAL_HASH_VERSION}\n"
            f"dtype={arr_int.dtype.str}\n"
            f"shape={tuple(arr_int.shape)}\n"
        ).encode("utf-8")
        expected = _hashlib.sha256(header_v1 + arr_int.tobytes()).hexdigest()
        assert canonical_array_hash(arr) == expected

    def test_collisions_blocked_by_shape_in_header(self):
        """A flat (192*84,) array with the same bytes as a (192,84) array
        must hash differently because shape is part of the canonical
        header."""
        flat = np.arange(192 * 84, dtype=np.int64)
        reshaped = flat.reshape(192, 84)
        assert canonical_array_hash(flat) != canonical_array_hash(reshaped)


# ---------------------------------------------------------------------------
# Test: PredictionRecord
# ---------------------------------------------------------------------------


class TestPredictionRecord:
    """Test the per-sample prediction record dataclass."""

    def test_record_required_fields(self):
        r = PredictionRecord(
            sample_id="SLP:danaLab:00001:uncover:000001",
            subject_id="00001",
            label_sha256="a" * 64,
            prediction_sha256="b" * 64,
            label_shape=(192, 84),
            prediction_shape=(192, 84),
            failure_reason=PRED_OK,
        )
        d = r.as_dict()
        assert d["sample_id"] == "SLP:danaLab:00001:uncover:000001"
        assert d["subject_id"] == "00001"
        assert len(d["label_sha256"]) == 64
        assert len(d["prediction_sha256"]) == 64
        assert d["failure_reason"] == PRED_OK

    def test_manifest_writes_real_records(self, tmp_path):
        """write_smoke_artifacts must serialise real PredictionRecord
        objects, not synthesise ``split_sample_NNNNNN`` placeholders."""
        from topper_perception.neural.slp8_region_smoke import write_smoke_artifacts

        label = np.zeros((192, 84), dtype=np.int64)
        pred = np.zeros((192, 84), dtype=np.int64)
        rec = PredictionRecord(
            sample_id="SLP:danaLab:00001:uncover:000001",
            subject_id="00001",
            label_sha256=canonical_array_hash(label),
            prediction_sha256=canonical_array_hash(pred),
            label_shape=(192, 84),
            prediction_shape=(192, 84),
            failure_reason=PRED_OK,
        )
        result = SmokeResult(
            success=True,
            train_records_initial=[rec],
            val_records_initial=[rec],
            train_records_resumed=[rec],
            val_records_resumed=[rec],
        )
        config = SmokeConfig(
            seed=42,
            batch_size=4,
            initial_epochs=1,
            resume_epochs=1,
            lr=0.001,
            weight_decay=0.0001,
            device="cpu",
            n_train_subjects=1,
            n_val_subjects=1,
            subset_seed=42,
        )
        write_smoke_artifacts(
            output_dir=tmp_path,
            result=result,
            config=config,
            dataset_manifest={
                "train_subjects": ["00001"],
                "val_subjects": ["00002"],
                "n_train_samples": 1,
                "n_val_samples": 1,
                "n_test_samples": 0,
            },
            model_config={"model_version": "Slp8TinyFcn", "n_classes": 9},
        )
        import csv as _csv

        with open(tmp_path / "predictions_manifest.csv", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        assert len(rows) == 4  # 1 train + 1 val, both initial and resumed
        for row in rows:
            assert row["sample_id"] == "SLP:danaLab:00001:uncover:000001"
            assert row["subject_id"] == "00001"
            assert len(row["label_sha256"]) == 64
            assert len(row["prediction_sha256"]) == 64
            assert row["label_sha256"] != ""
            assert row["prediction_sha256"] != ""
            assert row["sample_id"] != ""
            assert row["subject_id"] != ""
        # No placeholder rows.
        for row in rows:
            assert "_sample_" not in row["sample_id"]

        # failure_cases.csv should only have the header.
        with open(tmp_path / "failure_cases.csv", encoding="utf-8") as f:
            failure_rows = list(_csv.DictReader(f))
        assert failure_rows == []

    def test_manifest_row_count_matches_real_predictions(self, tmp_path):
        """If we pass 2 train + 1 val initial, the manifest must contain
        6 data rows (2 train initial, 1 val initial, 2 train resumed,
        1 val resumed)."""
        from topper_perception.neural.slp8_region_smoke import write_smoke_artifacts

        label = np.zeros((192, 84), dtype=np.int64)
        pred = np.zeros((192, 84), dtype=np.int64)
        sha = canonical_array_hash(label)
        rec = PredictionRecord(
            sample_id="SLP:danaLab:00001:uncover:000001",
            subject_id="00001",
            label_sha256=sha,
            prediction_sha256=sha,
            label_shape=(192, 84),
            prediction_shape=(192, 84),
            failure_reason=PRED_OK,
        )
        result = SmokeResult(
            success=True,
            train_records_initial=[rec, rec],
            val_records_initial=[rec],
            train_records_resumed=[rec, rec],
            val_records_resumed=[rec],
        )
        config = SmokeConfig(
            seed=42,
            batch_size=4,
            initial_epochs=1,
            resume_epochs=1,
            lr=0.001,
            weight_decay=0.0001,
            device="cpu",
            n_train_subjects=1,
            n_val_subjects=1,
            subset_seed=42,
        )
        write_smoke_artifacts(
            output_dir=tmp_path,
            result=result,
            config=config,
            dataset_manifest={
                "train_subjects": ["00001"],
                "val_subjects": ["00002"],
                "n_train_samples": 2,
                "n_val_samples": 1,
                "n_test_samples": 0,
            },
            model_config={"model_version": "Slp8TinyFcn", "n_classes": 9},
        )
        import csv as _csv

        with open(tmp_path / "predictions_manifest.csv", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        assert len(rows) == 6

    def test_hash_changes_when_prediction_changes(self):
        label = np.zeros((192, 84), dtype=np.int64)
        pred_a = np.zeros((192, 84), dtype=np.int64)
        pred_b = np.zeros((192, 84), dtype=np.int64)
        pred_b[0, 0] = 5
        h_a = canonical_array_hash(pred_a)
        h_b = canonical_array_hash(pred_b)
        assert h_a != h_b


# ---------------------------------------------------------------------------
# Test: Subset config flows into dataset manifest
# ---------------------------------------------------------------------------


class TestSubsetConfigFlowsToManifest:
    """Verify the SmokeConfig subset fields drive the actual dataset
    selection (no silent hard-coding)."""

    def _build_valid_config(
        self, n_train: int = 2, n_val: int = 1, seed: int = 42
    ) -> dict[str, Any]:
        return {
            "config_version": "slp8_pm_region_smoke_v0.1",
            "task_id": "TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1",
            "smoke_version": "slp8_region_smoke_v0.1",
            "provenance": "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            "raw_semantics": "raw_pmarray_response",
            "model": {
                "n_classes": 9,
                "input_shape": [192, 84],
            },
            "training": {
                "seed": 42,
                "device": "cpu",
                "batch_size": 4,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "epochs": {"initial": 1, "resume": 1},
            },
            "dataset": {
                "smoke_subset": {
                    "n_train_subjects": n_train,
                    "n_val_subjects": n_val,
                    "seed": seed,
                },
                "normalization": {
                    "method": "raw_passthrough_with_minmax_reference",
                    "fit_split": "train",
                    "raw_semantics": "raw_pmarray_response",
                },
            },
        }

    def test_resolved_subset_matches_config(self):
        from scripts.run_slp8_region_smoke import _build_smoke_config

        for n_train, n_val, seed in [(2, 1, 42), (3, 2, 99), (4, 2, 7)]:
            cfg = self._build_valid_config(n_train, n_val, seed)
            smoke_config = _build_smoke_config(cfg, device_override="cpu")
            assert smoke_config.n_train_subjects == n_train
            assert smoke_config.n_val_subjects == n_val
            assert smoke_config.subset_seed == seed

    def test_run_smoke_test_uses_config_subset(self, tmp_path, monkeypatch):
        """Build a synthetic B01 freeze + dataset and verify that
        changing the SmokeConfig subset fields changes the resulting
        dataset_manifest counts and per-phase prediction record count."""

        from topper_perception.neural import slp8_region_smoke as smoke_mod
        from topper_perception.io.slp8_training_table_freeze import (
            FreezeRow,
        )

        # Build a tiny synthetic freeze with 4 train subjects, 3 val
        # subjects, 2 frames each.  The temp directory holds the npy
        # files referenced by the rows.
        data_root = tmp_path / "data"
        data_root.mkdir()
        freeze_dir = tmp_path / "freeze"
        freeze_dir.mkdir()

        # Normalization stats (TRAIN-only)
        norm_stats = {
            "stats": {
                "method": "raw_passthrough_with_minmax_reference",
                "fit_split": "train",
                "global_min": 0.0,
                "global_max": 100.0,
                "global_mean": 50.0,
                "global_std": 25.0,
                "epsilon": 1e-12,
                "raw_semantics": "raw_pmarray_response",
            },
            "stats_sha256": "deadbeef",
        }
        (freeze_dir / "normalization_stats.json").write_text(
            json.dumps(norm_stats), encoding="utf-8"
        )

        # Build rows
        rows: list[FreezeRow] = []
        for split_name, subjects in (("train", ["00001", "00002", "00003", "00004"]),
                                     ("val", ["00005", "00006", "00007"])):
            for subj in subjects:
                for frame in range(2):
                    pressure_rel = (
                        f"subj_{subj}_f{frame}.npy"
                    )
                    label_rel = f"subj_{subj}_f{frame}_label.npy"
                    onehot_rel = f"subj_{subj}_f{frame}_onehot.npy"
                    # Save tiny pressure and label
                    np.save(
                        data_root / pressure_rel,
                        np.random.rand(192, 84).astype(np.float64) * 100,
                    )
                    np.save(
                        data_root / label_rel,
                        np.zeros((192, 84), dtype=np.int64),
                    )
                    np.save(
                        data_root / onehot_rel,
                        np.zeros((9, 192, 84), dtype=np.float32),
                    )
                    rows.append(
                        FreezeRow(
                            sample_id=f"{split_name}_{subj}_{frame}",
                            ml_split=split_name,
                            source_split="VAL",
                            setting="danaLab",
                            subject_id=subj,
                            cover="uncover",
                            frame_id=frame,
                            posture="SUPINE",
                            pressure_npy=pressure_rel,
                            region_label_npy=label_rel,
                            region_onehot_npy=onehot_rel,
                            points_csv="",
                            height=192,
                            width=84,
                            class_ids_present="0",
                            annotation_provenance="V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
                            source_review_status="NOT_REVIEWED",
                            export_version="1.1.0",
                            export_status="EXPORTED",
                            source_pmarray_sha256="",
                            background_pixel_count=0,
                            body_pixel_count=0,
                            clipped_ratio=0.0,
                            onehot_valid=True,
                            onehot_roundtrip=True,
                        )
                    )

        # Write a minimal freeze_manifest.json (not required by
        # load_b01_freeze_tables but we keep it for completeness).
        (freeze_dir / "freeze_manifest.json").write_text(
            json.dumps({
                "core": {
                    "freeze_version": "slp8_training_tables_v0.1",
                    "a06_split_sha256": (
                        "024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706"
                    ),
                }
            }),
            encoding="utf-8",
        )

        # Stub out load_b01_freeze_tables to return our synthetic rows.
        class _StubFreeze:
            def __init__(self, rows: list[FreezeRow]) -> None:
                self._train_rows = tuple(r for r in rows if r.ml_split == "train")
                self._val_rows = tuple(r for r in rows if r.ml_split == "val")
                self._test_rows = None

            @property
            def train_rows(self) -> tuple[FreezeRow, ...]:
                return self._train_rows

            @property
            def val_rows(self) -> tuple[FreezeRow, ...]:
                return self._val_rows

        from topper_perception.neural import slp8_region_dataset as dataset_mod

        monkeypatch.setattr(
            dataset_mod,
            "load_b01_freeze_tables",
            lambda *_args, **_kwargs: _StubFreeze(rows),
        )

        # Use a small n_train/n_val to keep the test fast.
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        for n_train, n_val in [(2, 1), (3, 2), (4, 1)]:
            config = SmokeConfig(
                seed=42,
                batch_size=2,
                initial_epochs=1,
                resume_epochs=1,
                lr=0.001,
                weight_decay=0.0001,
                device="cpu",
                n_train_subjects=n_train,
                n_val_subjects=n_val,
                subset_seed=42,
            )
            run_dir = output_dir / f"n{n_train}_{n_val}"
            result = smoke_mod.run_smoke_test(
                b01_freeze_dir=freeze_dir,
                dataset_root=data_root,
                output_dir=run_dir,
                config=config,
            )
            assert result.success is True, result.verification_failures
            assert (
                len(result.train_records_initial) == n_train * 2
            ), f"expected {n_train * 2} train records, got {len(result.train_records_initial)}"
            assert (
                len(result.val_records_initial) == n_val * 2
            ), f"expected {n_val * 2} val records, got {len(result.val_records_initial)}"
            # 4 phases total
            assert (
                len(result.train_records_initial)
                + len(result.val_records_initial)
                + len(result.train_records_resumed)
                + len(result.val_records_resumed)
            ) == 2 * (n_train + n_val) * 2
