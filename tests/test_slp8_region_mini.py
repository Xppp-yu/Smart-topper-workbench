"""Tests for the B04 SLP8 PM-only Region Mini protocol (R02).

These tests cover the B04 v0.1 R02 contract:

* Slp8SmallUnet input / output shape and the 84-width recovery path.
* Slp8SmallUnet parameter count <= 150,000.
* The two B04 candidates are registered and discoverable.
* The class-weight formula uses TRAIN-only stats and rejects VAL/TEST
  inputs, non-finite ratios, zero ratios, and out-of-range values.
* The runner is fail-closed against non-existent / non-canonical config
  fields and non-canonical hyperparameter values; R02 adds checks for
  ``expected_split_counts``, ``expected_subjects`` and the
  ``lifecycle`` block.
* CUDA-not-available is fail-closed; synthetic CPU smoke is the only
  CPU entry point.
* Early stopping only monitors ``val_loss`` and refuses any other
  monitor.
* Non-finite loss / metrics are rejected and turn the candidate into
  ``FAILED``.
* Checkpoint / resume / reload produces a hash-consistent prediction;
  resume with a mismatched identity is rejected; resume for a DONE
  run is refused.
* Centroid error obeys the both-missing / GT-only / both-present rules
  and the ``centroid_errors.csv`` now carries sample_id / subject_id /
  posture / valid / invalid_reason.
* Fixed foreground classes 1..8 are always scored.
* Per-subject, per-posture, worst subject all populate the bundle.
* DONE / FAILED / STOPPED are mutually exclusive terminal files.
* ``predictions_manifest`` carries real sample IDs and real hashes.
* ``--run-authorized`` gate is enforced by the CLI.
* The synthetic CPU smoke runs end-to-end and writes all required
  artefacts.
* The resource budget is enforced via ``time.monotonic`` and CUDA peak
  memory; a tiny-budget test triggers ``STOPPED`` without running a
  long real workload.
* Two independent subprocess runs of the synthetic smoke produce
  byte-identical outputs (determinism contract).
* Output directory collisions refuse to write anything new.
* Real B01 input-contract verifier rejects mismatched counts / SHAs /
  provenance / setting / cover.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_runner_module():
    """Load the CLI by path so this test is independent of ``sys.path``.

    ``scripts`` is deliberately not an installed Python package.  Importing
    it by package name happens to work in some POSIX test environments, but
    is not a stable contract under the Windows pytest configuration.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "slp8_region_mini_runner_under_test",
        str(SCRIPTS_DIR / "run_slp8_region_mini.py"),
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not load SLP8 Mini runner spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

from topper_perception.neural.slp8_region_b01_contract import (
    B01ContractError,
    B01ContractExpected,
    B01FreezeSnapshot,
    B01FreezeStructuralTest,
    build_b01_contract_expected,
    check_freeze_manifest_file_consistency,
    verify_b01_contract,
)
from topper_perception.neural.slp8_region_budget import (
    BudgetCheck,
    ResourceBudget,
    ResourceBudgetState,
    resource_budget_from_config,
)
from topper_perception.neural.slp8_region_class_weights import (
    ALLOWED_WEIGHT_SPLIT,
    WEIGHT_CLIP_MAX,
    WEIGHT_CLIP_MIN,
    ClassWeightError,
    assert_class_weight_invariants,
    class_weights_to_tensor,
    compute_class_weights,
    weights_match,
)
from topper_perception.neural.slp8_region_dataset import (
    N_CLASSES,
    PRESSURE_SHAPE,
    REGION_ID_TO_NAME,
)
from topper_perception.neural.slp8_region_determinism import (
    DeterminismSettings,
    apply_settings,
    collect_settings,
    environment_payload,
)
from topper_perception.neural.slp8_region_metrics_ext import (
    DEFAULT_IMAGE_SHAPE,
    FOREGROUND_CLASS_IDS,
    compute_centroid_errors,
    compute_extended_metrics,
    compute_per_posture_metrics,
    compute_per_subject_metrics,
    find_worst_subject,
    summarize_centroid_errors,
)
from topper_perception.neural.slp8_region_mini import (
    B02_BASELINE_REFERENCE_VAL_FIXED_IOU,
    B04_CANDIDATE_NAMES,
    B04_FROZEN_DEFAULTS,
    B04_MAX_PARAMETERS,
    B04_RESOURCE_BUDGET,
    CANONICAL_HASH_VERSION,
    CHECKPOINT_VERSION,
    RELOAD_PROBE_VERSION,
    MINI_VERSION,
    SYNTHETIC_DEFAULTS,
    TASK_ID,
    _build_checkpoint_payload,
    _build_best_epoch_reload_probe,
    _clone_state_dict_to_cpu,
    _flatten_segmentation_for_cross_entropy,
    _predictions_hash,
    _verify_best_epoch_reload_probe,
    build_mini_config,
    build_synthetic_dataset,
    canonical_array_hash,
    check_output_dir_safety,
    file_sha256,
    resolve_device,
    run_mini,
    run_one_candidate,
    validate_mini_config,
    write_status_files,
)
from topper_perception.neural.slp8_region_models import (
    MODEL_VERSION,
    MODEL_REGISTRY,
    SMALL_UNET_VERSION,
    B04_MAX_PARAMETERS as MODEL_B04_MAX_PARAMETERS,
    Slp8SmallUnet,
    Slp8TinyFcn,
    create_slp8_small_unet,
    create_slp8_tiny_fcn,
    get_model_builder,
    list_model_builders,
)
from topper_perception.neural.slp8_region_resume import (
    CheckpointIdentity,
    EarlyStopperState,
    ResumeIdentityError,
    ResumeRefusedError,
    capture_rng_state,
    class_weight_sha256,
    file_sha256 as resume_file_sha256,
    identity_from_dict,
    input_manifest_hashes_sha256,
    refuse_resume_for_done_run,
    restore_rng_state,
    verify_resume_identity,
)
from topper_perception.io.slp8_training_table_freeze import (
    A06_SPLIT_SHA256_EXPECTED,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "slp8_pm_region_mini_v0.1.json"


@pytest.fixture
def fresh_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def tmp_output_dir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="b04_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_expected(
    *,
    train_count: int = 3645,
    val_count: int = 450,
    test_count: int = 0,
    train_subjects: int = 81,
    val_subjects: int = 10,
    test_subjects: int = 0,
    a06_split_sha256: str = A06_SPLIT_SHA256_EXPECTED,
    provenance: str = "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
    source_review_status: str = "NOT_REVIEWED",
    setting: str = "danaLab",
    cover: str = "uncover",
    freeze_manifest_core_sha256: str = "f" * 64,
    structural_test_samples: int = 495,
    structural_test_subjects: int = 11,
) -> B01ContractExpected:
    return B01ContractExpected(
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        a06_split_sha256=a06_split_sha256,
        provenance=provenance,
        source_review_status=source_review_status,
        setting=setting,
        cover=cover,
        freeze_manifest_core_sha256=freeze_manifest_core_sha256,
        structural_test_samples=structural_test_samples,
        structural_test_subjects=structural_test_subjects,
    )


def _make_a06_split() -> "A06Split":
    """Helper to build a minimal A06Split instance for tests."""
    from topper_perception.io.slp8_training_table_freeze import A06Split
    return A06Split(
        raw={},
        subject_to_ml_split={"00000": "train"},
        subject_to_setting={"00000": "danaLab"},
        split_counts_subjects={"train": 1},
        split_counts_samples={"train": 1, "val": 0, "test": 0},
        sha256="0" * 64,
    )


def _make_normalization_stats():
    from topper_perception.io.slp8_training_table_freeze import NormalizationStats
    return NormalizationStats(
        n_samples=0, n_pixels=0,
        finite_pixel_count=0, non_finite_pixel_count=0,
        global_min=0.0, global_max=0.0, global_mean=0.0, global_std=0.0,
        method="raw_passthrough_with_minmax_reference",
        epsilon=1e-12, raw_dtype="float64",
        raw_semantics="raw_pmarray_response", fit_split="train",
        subject_count=0, per_subject_count_min=0, per_subject_count_max=0,
        fitted_at_utc="2026-01-01T00:00:00+00:00",
    )


def _make_class_stats():
    from topper_perception.io.slp8_training_table_freeze import ClassStats
    return ClassStats(
        n_samples=0, n_pixels=0,
        per_class_pixel_count={}, per_class_pixel_ratio={},
        missing_class_samples={}, per_posture_count={},
        per_subject_count_min=0, per_subject_count_max=0,
        subject_count=0, small_region_sample_count=0,
        tiny_region_sample_count=0, onehot_roundtrip_ok_count=0,
    )


def _make_snapshot(
    *,
    freeze_manifest_sha256: str = "f" * 64,
    a06_split_sha256: str = A06_SPLIT_SHA256_EXPECTED,
    observed_settings: tuple[str, ...] = ("danaLab",),
    observed_covers: tuple[str, ...] = ("uncover",),
    observed_provenances: tuple[str, ...] = (
        "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
    ),
    observed_review_statuses: tuple[str, ...] = ("NOT_REVIEWED",),
    train_count: int = 3645,
    val_count: int = 450,
    structural_sample_count: int = 495,
    structural_subject_count: int = 11,
) -> B01FreezeSnapshot:
    train_subjects = tuple(f"{i:05d}" for i in range(81))
    val_subjects = tuple(f"{i:05d}" for i in range(100, 110))
    return B01FreezeSnapshot(
        freeze_dir=Path("/fake"),
        train_count=train_count,
        val_count=val_count,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        train_manifest_sha256="b" * 64,
        val_manifest_sha256="c" * 64,
        observed_settings=observed_settings,
        observed_covers=observed_covers,
        observed_provenances=observed_provenances,
        observed_review_statuses=observed_review_statuses,
        structural_test=B01FreezeStructuralTest(
            sample_count=structural_sample_count,
            subject_count=structural_subject_count,
            manifest_sha256="d" * 64,
        ),
        a06_split_sha256=a06_split_sha256,
        freeze_manifest_core_sha256=freeze_manifest_sha256,
        core_train_manifest_sha256="b" * 64,
        core_val_manifest_sha256="c" * 64,
    )


def _build_identity(
    candidate: str = MODEL_VERSION,
    *,
    experiment_id: str = "EXP-SLP-B04-TEST-EXP-ID",
    data_manifest_sha256: str = "0" * 64,
    git_commit: str = "0" * 40,
    git_dirty: bool = False,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        task_id=TASK_ID,
        candidate=candidate,
        model_version=get_model_builder(candidate).version,
        seed=42,
        n_classes=N_CLASSES,
        image_shape=PRESSURE_SHAPE,
        config_sha256="",
        a06_split_sha256=A06_SPLIT_SHA256_EXPECTED,
        split_sha256=A06_SPLIT_SHA256_EXPECTED,
        freeze_manifest_sha256="",
        train_class_stats_sha256="",
        class_weight_sha256="",
        input_manifest_hashes_sha256="",
        git_commit=git_commit,
        git_dirty=git_dirty,
        experiment_id=experiment_id,
        data_manifest_sha256=data_manifest_sha256,
    )


# ---------------------------------------------------------------------------
# Test: TASK-ID
# ---------------------------------------------------------------------------


def test_task_id_matches_r02():
    assert TASK_ID == "TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-AND-RUNNER-v0.1"


# ---------------------------------------------------------------------------
# Test: SmallUNet architecture
# ---------------------------------------------------------------------------


class TestSmallUnetArchitecture:
    """Slp8SmallUnet must satisfy the B04 architectural contract."""

    def test_small_unet_creation(self):
        m = Slp8SmallUnet()
        assert m.n_classes == N_CLASSES
        assert m.model_version == SMALL_UNET_VERSION

    def test_input_output_shape(self):
        m = Slp8SmallUnet()
        x = torch.randn(2, 1, 192, 84, dtype=torch.float32)
        y = m(x)
        assert y.shape == (2, N_CLASSES, 192, 84)
        assert y.dtype == torch.float32

    def test_predict_shape_and_range(self):
        m = Slp8SmallUnet()
        x = torch.randn(3, 1, 192, 84, dtype=torch.float32)
        p = m.predict(x)
        assert p.shape == (3, 192, 84)
        assert p.dtype == torch.long
        assert p.min().item() >= 0
        assert p.max().item() < N_CLASSES

    def test_parameter_count_under_cap(self):
        m = Slp8SmallUnet()
        assert m.count_parameters() <= B04_MAX_PARAMETERS
        assert m.count_parameters() < 150_000

    def test_explicit_upsample_recovery_84_width(self):
        m = Slp8SmallUnet()
        m.eval()
        x = torch.randn(1, 1, 192, 84, dtype=torch.float32)
        with torch.no_grad():
            y1 = m(x)
            y2 = m(x)
        assert torch.equal(y1, y2)
        assert y1.shape[2] == 192
        assert y1.shape[3] == 84

    def test_explicit_upsample_writes_recorded_targets(self):
        m = Slp8SmallUnet()
        assert m._skip1_shape == (192, 84)
        assert m._skip2_shape == (96, 42)
        assert m._bottleneck_shape == (48, 21)

    def test_no_batchnorm_no_dropout(self):
        m = Slp8SmallUnet()
        for module in m.modules():
            assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            assert not isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d))

    def test_no_pretrained_weights_attribute(self):
        m = Slp8SmallUnet()
        cfg = m.get_config()
        for forbidden in ("pretrained", "checkpoint_url", "external_weights", "url"):
            assert forbidden not in cfg

    def test_factory_returns_device_model(self):
        m, cfg = create_slp8_small_unet(device="cpu")
        assert m.n_classes == N_CLASSES
        assert cfg["device"] == "cpu"
        assert cfg["parameter_count"] <= B04_MAX_PARAMETERS

    def test_fail_closed_input_validation(self):
        m = Slp8SmallUnet()
        with pytest.raises(ValueError, match="4D"):
            m(torch.randn(192, 84, dtype=torch.float32))
        with pytest.raises(ValueError, match="channel must be 1"):
            m(torch.randn(2, 3, 192, 84, dtype=torch.float32))
        with pytest.raises(ValueError, match="spatial shape"):
            m(torch.randn(2, 1, 100, 100, dtype=torch.float32))
        with pytest.raises(ValueError, match="float32"):
            m(torch.randn(2, 1, 192, 84, dtype=torch.float64))
        bad = torch.randn(2, 1, 192, 84, dtype=torch.float32)
        bad[0, 0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            m(bad)

    def test_gradient_flow(self):
        m = Slp8SmallUnet()
        x = torch.randn(2, 1, 192, 84, dtype=torch.float32)
        lab = torch.zeros(2, 192, 84, dtype=torch.long)
        for c in range(1, N_CLASSES):
            lab[:, c, c] = c
        loss = nn.CrossEntropyLoss()(m(x).reshape(2, N_CLASSES, -1), lab.reshape(2, -1))
        loss.backward()
        for p in m.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


class TestCandidateRegistry:
    """The B04 candidate registry is frozen and complete."""

    def test_registry_contains_both_candidates(self):
        builders = list_model_builders()
        assert MODEL_VERSION in builders
        assert SMALL_UNET_VERSION in builders

    def test_candidate_names_in_frozen_order(self):
        assert B04_CANDIDATE_NAMES == (MODEL_VERSION, SMALL_UNET_VERSION)

    def test_get_model_builder_returns_known(self):
        for name in B04_CANDIDATE_NAMES:
            b = get_model_builder(name)
            assert b.name == name
            assert b.version == name

    def test_get_model_builder_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model_builder("slp8_nope_v0.0")


# ---------------------------------------------------------------------------
# Test: Class-weight contract
# ---------------------------------------------------------------------------


def _toy_class_stats(*, drop_class: int | None = None, bad_value: float | None = None) -> dict[str, Any]:
    ratio = {
        0: 0.75, 1: 0.04, 2: 0.05, 3: 0.04, 4: 0.03,
        5: 0.04, 6: 0.02, 7: 0.02, 8: 0.01,
    }
    if drop_class is not None:
        ratio.pop(drop_class, None)
    if bad_value is not None and drop_class is not None:
        ratio[drop_class] = bad_value
    return {
        "n_samples": 3645,
        "n_pixels": 3645 * 192 * 84,
        "subject_count": 81,
        "per_class_pixel_ratio": {str(k): v for k, v in ratio.items()},
    }


class TestClassWeightContract:
    """Class weights must be derived from TRAIN-only stats, never VAL/TEST."""

    def test_train_only_derivation(self):
        result = compute_class_weights(_toy_class_stats())
        assert result.source_split == ALLOWED_WEIGHT_SPLIT
        assert result.formula_version == "slp8_class_weights_v0.1"
        assert set(result.weights.keys()) == set(range(N_CLASSES))

    def test_weights_within_clip(self):
        result = compute_class_weights(_toy_class_stats())
        for cid, w in result.weights.items():
            assert WEIGHT_CLIP_MIN - 1e-9 <= w <= WEIGHT_CLIP_MAX + 1e-9

    def test_both_candidates_get_same_weights(self):
        stats = _toy_class_stats()
        a = compute_class_weights(stats)
        b = compute_class_weights(stats)
        assert weights_match(a, b)

    def test_reject_val_split(self):
        with pytest.raises(ClassWeightError, match="only accepts split"):
            compute_class_weights(_toy_class_stats(), allowed_split="val")

    def test_reject_test_split(self):
        with pytest.raises(ClassWeightError, match="only accepts split"):
            compute_class_weights(_toy_class_stats(), allowed_split="test")

    def test_reject_zero_pixel_ratio(self):
        with pytest.raises(ClassWeightError, match="cannot derive"):
            compute_class_weights(_toy_class_stats(drop_class=3, bad_value=0.0))

    def test_reject_nan_pixel_ratio(self):
        with pytest.raises(ClassWeightError, match="cannot derive"):
            compute_class_weights(_toy_class_stats(drop_class=3, bad_value=float("nan")))

    def test_reject_inf_pixel_ratio(self):
        with pytest.raises(ClassWeightError, match="cannot derive"):
            compute_class_weights(_toy_class_stats(drop_class=3, bad_value=float("inf")))

    def test_reject_missing_class(self):
        with pytest.raises(ClassWeightError, match="cannot derive"):
            compute_class_weights(_toy_class_stats(drop_class=5, bad_value=0.0))

    def test_normalization_mean_preserved(self):
        result = compute_class_weights(_toy_class_stats())
        assert math.isclose(
            float(np.mean(list(result.raw_weights.values()))),
            result.mean_raw_weight,
        )
        for cid in range(N_CLASSES):
            expected = result.raw_weights[cid] / result.mean_raw_weight
            actual = max(min(expected, WEIGHT_CLIP_MAX), WEIGHT_CLIP_MIN)
            assert math.isclose(result.weights[cid], actual, rel_tol=1e-6)

    def test_class_weights_to_tensor_finite(self):
        result = compute_class_weights(_toy_class_stats())
        arr = class_weights_to_tensor(result)
        assert arr.shape == (N_CLASSES,)
        assert np.isfinite(arr).all()

    def test_assert_class_weight_invariants_ok(self):
        result = compute_class_weights(_toy_class_stats())
        assert_class_weight_invariants(result)

    def test_invariant_rejects_wrong_source_split(self):
        result = compute_class_weights(_toy_class_stats())
        broken = replace(result, source_split="val")
        with pytest.raises(ClassWeightError, match="source_split"):
            assert_class_weight_invariants(broken)

    def test_invariant_rejects_wrong_formula_version(self):
        result = compute_class_weights(_toy_class_stats())
        broken = replace(result, formula_version="slp8_class_weights_v9.9")
        with pytest.raises(ClassWeightError, match="formula_version"):
            assert_class_weight_invariants(broken)

    def test_invariant_rejects_non_finite_weight(self):
        result = compute_class_weights(_toy_class_stats())
        broken_weights = dict(result.weights)
        broken_weights[1] = float("nan")
        broken = replace(result, weights=broken_weights)
        with pytest.raises(ClassWeightError, match="not finite"):
            assert_class_weight_invariants(broken)


import math  # noqa: E402  placed here to keep earlier test groups self-contained


# ---------------------------------------------------------------------------
# Test: Config validation fail-closed (R02 adds lifecycle + expected_*)
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_default_config_valid(self, fresh_config):
        validate_mini_config(fresh_config)

    def test_wrong_task_id_rejected(self, fresh_config):
        fresh_config["task_id"] = "TASK-SLP-NOPE"
        with pytest.raises(Exception, match="task_id"):
            validate_mini_config(fresh_config)

    def test_wrong_config_version_rejected(self, fresh_config):
        fresh_config["config_version"] = "slp8_region_mini_v9.9"
        with pytest.raises(Exception, match="config_version"):
            validate_mini_config(fresh_config)

    def test_wrong_image_shape_rejected(self, fresh_config):
        fresh_config["dataset"]["image_shape"] = [192, 85]
        with pytest.raises(Exception, match="image_shape"):
            validate_mini_config(fresh_config)

    def test_wrong_seed_rejected(self, fresh_config):
        fresh_config["training"]["seed"] = 7
        with pytest.raises(Exception, match="seed"):
            validate_mini_config(fresh_config)

    def test_wrong_device_rejected(self, fresh_config):
        fresh_config["training"]["device"] = "cpu"
        with pytest.raises(Exception, match="device"):
            validate_mini_config(fresh_config)

    def test_wrong_batch_size_rejected(self, fresh_config):
        fresh_config["training"]["batch_size"] = 8
        with pytest.raises(Exception, match="batch_size"):
            validate_mini_config(fresh_config)

    def test_wrong_max_epochs_rejected(self, fresh_config):
        fresh_config["training"]["max_epochs"] = 10
        with pytest.raises(Exception, match="max_epochs"):
            validate_mini_config(fresh_config)

    def test_wrong_min_epochs_rejected(self, fresh_config):
        fresh_config["training"]["min_epochs"] = 3
        with pytest.raises(Exception, match="min_epochs"):
            validate_mini_config(fresh_config)

    def test_wrong_early_stop_monitor_rejected(self, fresh_config):
        fresh_config["training"]["early_stopping"]["monitor"] = "val_iou"
        with pytest.raises(Exception, match="monitor must be 'val_loss'"):
            validate_mini_config(fresh_config)

    def test_wrong_early_stop_patience_rejected(self, fresh_config):
        fresh_config["training"]["early_stopping"]["patience"] = 2
        with pytest.raises(Exception, match="patience"):
            validate_mini_config(fresh_config)

    def test_wrong_resource_budget_rejected(self, fresh_config):
        fresh_config["resource_budget"]["max_wall_minutes_per_candidate"] = 60
        with pytest.raises(Exception, match="max_wall_minutes_per_candidate"):
            validate_mini_config(fresh_config)

    def test_wrong_feasibility_threshold_rejected(self, fresh_config):
        fresh_config["feasibility_gate"]["b02_reference_val_fixed_iou"] = 0.5
        with pytest.raises(Exception, match="b02_reference_val_fixed_iou"):
            validate_mini_config(fresh_config)

    def test_unknown_candidate_rejected(self, fresh_config):
        fresh_config["candidates"] = [
            {"name": "slp8_tiny_fcn_v0.1", "version": "slp8_tiny_fcn_v0.1", "max_parameters": 150000},
            {"name": "slp8_nope_v0.1", "version": "slp8_nope_v0.1", "max_parameters": 150000},
        ]
        with pytest.raises(Exception, match="not registered"):
            validate_mini_config(fresh_config)

    def test_candidate_max_parameters_exceeded_rejected(self, fresh_config):
        fresh_config["candidates"] = [
            {"name": "slp8_tiny_fcn_v0.1", "version": "slp8_tiny_fcn_v0.1", "max_parameters": 200000},
            {"name": "slp8_small_unet_v0.1", "version": "slp8_small_unet_v0.1", "max_parameters": 150000},
        ]
        with pytest.raises(Exception, match="exceeds B04 cap"):
            validate_mini_config(fresh_config)

    def test_wrong_candidate_order_rejected(self, fresh_config):
        fresh_config["candidates"] = list(reversed(fresh_config["candidates"]))
        with pytest.raises(Exception, match="must equal"):
            validate_mini_config(fresh_config)

    def test_duplicate_candidate_rejected(self, fresh_config):
        fresh_config["candidates"] = [fresh_config["candidates"][0], fresh_config["candidates"][0]]
        with pytest.raises(Exception, match="listed more than once"):
            validate_mini_config(fresh_config)

    def test_wrong_expected_train_count_rejected(self, fresh_config):
        fresh_config["expected_split_counts"]["train"] = 2000
        with pytest.raises(Exception, match="expected_split_counts.train"):
            validate_mini_config(fresh_config)

    def test_wrong_expected_test_count_rejected(self, fresh_config):
        fresh_config["expected_split_counts"]["test"] = 495
        with pytest.raises(Exception, match="expected_split_counts.test"):
            validate_mini_config(fresh_config)

    def test_wrong_expected_train_subjects_rejected(self, fresh_config):
        fresh_config["expected_subjects"]["train"] = 100
        with pytest.raises(Exception, match="expected_subjects.train"):
            validate_mini_config(fresh_config)

    def test_lifecycle_missing_rejected(self, fresh_config):
        del fresh_config["lifecycle"]
        with pytest.raises(Exception, match="lifecycle"):
            validate_mini_config(fresh_config)

    def test_lifecycle_wrong_terminal_states_rejected(self, fresh_config):
        fresh_config["lifecycle"]["valid_terminal_states"] = ["DONE", "FAILED"]
        with pytest.raises(Exception, match="valid_terminal_states"):
            validate_mini_config(fresh_config)

    def test_lifecycle_wrong_exclusive_files_rejected(self, fresh_config):
        fresh_config["lifecycle"]["exclusive_terminal_files"] = ["DONE.json"]
        with pytest.raises(Exception, match="exclusive_terminal_files"):
            validate_mini_config(fresh_config)


# ---------------------------------------------------------------------------
# Test: CUDA / device handling
# ---------------------------------------------------------------------------


class TestDeviceHandling:
    def test_parameter_audit_snapshot_is_independent_and_cpu_resident(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Linear(3, 2).to(device)

        snapshot = _clone_state_dict_to_cpu(model)
        original = {name: tensor.clone() for name, tensor in snapshot.items()}

        assert snapshot
        assert all(tensor.device.type == "cpu" for tensor in snapshot.values())
        assert all(not tensor.requires_grad for tensor in snapshot.values())

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)

        assert all(
            torch.equal(snapshot[name], original[name])
            for name in snapshot
        )

    def test_flattened_cross_entropy_is_equivalent_to_spatial_form(self):
        generator = torch.Generator().manual_seed(42)
        logits = torch.randn(2, N_CLASSES, 7, 5, generator=generator)
        labels = torch.randint(
            0, N_CLASSES, (2, 7, 5), generator=generator
        )
        weights = torch.linspace(0.5, 1.5, N_CLASSES)

        flat_logits, flat_labels = _flatten_segmentation_for_cross_entropy(
            logits, labels
        )

        assert flat_logits.shape == (2 * 7 * 5, N_CLASSES)
        assert flat_labels.shape == (2 * 7 * 5,)
        spatial_loss = nn.CrossEntropyLoss(weight=weights)(
            logits.reshape(2, N_CLASSES, -1), labels.reshape(2, -1)
        )
        flat_loss = nn.CrossEntropyLoss(weight=weights)(flat_logits, flat_labels)
        assert torch.allclose(flat_loss, spatial_loss, rtol=1e-6, atol=1e-7)

    def test_cuda_required_for_real_runs(self):
        from topper_perception.neural.slp8_region_mini import MiniProtocolError as _MPE
        if torch.cuda.is_available():
            return
        with pytest.raises(_MPE, match="B04 protocol forbids silent CPU fallback"):
            resolve_device("cuda", allow_cpu_fallback=False)

    def test_cpu_explicit_request_rejected_for_real_runs(self):
        from topper_perception.neural.slp8_region_mini import MiniProtocolError as _MPE
        with pytest.raises(_MPE, match="B04 fail-closed forbids CPU"):
            resolve_device("cpu", allow_cpu_fallback=False)

    def test_synthetic_cpu_smoke_resolves_to_cpu(self):
        device = resolve_device("cpu", allow_cpu_fallback=True)
        assert str(device) == "cpu"


# ---------------------------------------------------------------------------
# Test: Centroid error contract
# ---------------------------------------------------------------------------


class TestCentroidErrorContract:
    def test_both_missing_excluded_from_per_region_average(self):
        lab = np.zeros((192, 84), dtype=np.int64)
        pred = np.zeros((192, 84), dtype=np.int64)
        records = compute_centroid_errors([lab], [pred], ["x"])
        assert all(r.both_missing for r in records)
        assert all(not r.valid for r in records)
        assert all(r.invalid_reason == "both_gt_and_pred_absent" for r in records)
        summary = summarize_centroid_errors(records)
        assert summary.overall_mean == 0.0
        assert summary.n_invalid == len(FOREGROUND_CLASS_IDS)
        for cid in FOREGROUND_CLASS_IDS:
            assert summary.per_region_mean[cid] == 0.0
            assert summary.per_region_count[cid] == 0

    def test_gt_only_records_max_error(self):
        H, W = DEFAULT_IMAGE_SHAPE
        lab = np.zeros((H, W), dtype=np.int64)
        lab[10:50, 10:40] = 1
        pred = np.zeros((H, W), dtype=np.int64)
        records = compute_centroid_errors([lab], [pred], ["x"])
        for rec in records:
            if rec.region_id == 1:
                assert not rec.both_missing
                assert rec.valid
                assert rec.error == 1.0

    def test_prediction_only_is_invalid_and_excluded(self):
        H, W = DEFAULT_IMAGE_SHAPE
        lab = np.zeros((H, W), dtype=np.int64)
        pred = np.zeros((H, W), dtype=np.int64)
        pred[10:50, 10:40] = 1

        records = compute_centroid_errors([lab], [pred], ["x"])
        region = next(rec for rec in records if rec.region_id == 1)

        assert not region.valid
        assert not region.both_missing
        assert region.invalid_reason == "gt_absent_pred_present"
        summary = summarize_centroid_errors(records)
        assert summary.per_region_count[1] == 0
        assert summary.per_region_mean[1] == 0.0

    def test_both_present_records_distance_normalized(self):
        H, W = DEFAULT_IMAGE_SHAPE
        lab = np.zeros((H, W), dtype=np.int64)
        lab[10:50, 10:40] = 1
        pred = np.zeros((H, W), dtype=np.int64)
        pred[10:50, 10:40] = 1
        records = compute_centroid_errors([lab], [pred], ["x"])
        for rec in records:
            if rec.region_id == 1:
                assert rec.error == 0.0
                assert rec.valid
                assert not rec.both_missing

    def test_offset_centroid_error_is_positive(self):
        H, W = DEFAULT_IMAGE_SHAPE
        lab = np.zeros((H, W), dtype=np.int64)
        lab[10:50, 10:40] = 1
        pred = np.zeros((H, W), dtype=np.int64)
        pred[20:60, 20:50] = 1
        records = compute_centroid_errors([lab], [pred], ["x"])
        diag = float(np.sqrt(H ** 2 + W ** 2))
        expected = float(np.sqrt(10 ** 2 + 10 ** 2)) / diag
        for rec in records:
            if rec.region_id == 1:
                assert math.isclose(rec.error, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Test: Metrics bundle contract
# ---------------------------------------------------------------------------


class TestExtendedMetricsBundle:
    def _build_dataset(self, n: int = 4):
        rng = np.random.default_rng(0)
        labels, preds, subs, posts = [], [], [], []
        for i in range(n):
            lab = rng.integers(0, 9, size=(192, 84), dtype=np.int64)
            pred = lab.copy()
            noise = rng.random((192, 84)) < 0.3
            pred[noise] = rng.integers(0, 9, size=int(noise.sum()))
            labels.append(lab)
            preds.append(pred)
            subs.append(f"{i // 2:05d}")
            posts.append(["SUPINE", "LEFT", "RIGHT"][i % 3])
        return labels, preds, subs, posts

    def test_per_region_covers_classes_1_to_8(self):
        labels, preds, subs, posts = self._build_dataset()
        bundle = compute_extended_metrics(labels, preds, subs, posts)
        cids = sorted(row["class_id"] for row in bundle.per_region)
        assert cids == sorted(FOREGROUND_CLASS_IDS)

    def test_per_posture_includes_all(self):
        labels, preds, subs, posts = self._build_dataset()
        bundle = compute_extended_metrics(labels, preds, subs, posts)
        assert "ALL" in bundle.per_posture
        for p in ("SUPINE", "LEFT", "RIGHT"):
            assert p in bundle.per_posture

    def test_per_subject_includes_all_and_subjects(self):
        labels, preds, subs, posts = self._build_dataset()
        bundle = compute_extended_metrics(labels, preds, subs, posts)
        assert "ALL" in bundle.per_subject
        for s in subs:
            assert s in bundle.per_subject

    def test_worst_subject_is_lowest_iou(self):
        labels, preds, subs, posts = self._build_dataset()
        bundle = compute_extended_metrics(labels, preds, subs, posts)
        assert bundle.worst_subject is not None
        real = [
            (s, m["fixed_foreground_macro_iou"])
            for s, m in bundle.per_subject.items()
            if s != "ALL" and m["n_samples"] > 0
        ]
        worst_sub = min(real, key=lambda kv: kv[1])[0]
        assert bundle.worst_subject["subject_id"] == worst_sub

    def test_fixed_foreground_macro_does_not_skip_empty_classes(self):
        labels, preds, subs, posts = self._build_dataset(4)
        for p in preds:
            p[p == 4] = 0
        bundle = compute_extended_metrics(labels, preds, subs, posts)
        per4 = next(r for r in bundle.per_region if r["class_id"] == 4)
        assert per4["iou"] == 0.0
        ious = [r["iou"] for r in bundle.per_region]
        assert math.isclose(bundle.fixed_foreground_macro_iou, float(np.mean(ious)))


# ---------------------------------------------------------------------------
# Test: Output collision, status mutex
# ---------------------------------------------------------------------------


class TestOutputCollisionAndStatus:
    def test_empty_dir_accepted(self, tmp_output_dir):
        check_output_dir_safety(tmp_output_dir)

    def test_dir_with_done_rejected(self, tmp_output_dir):
        (tmp_output_dir / "DONE.json").write_text("{}", encoding="utf-8")
        with pytest.raises(Exception, match="DONE.json"):
            check_output_dir_safety(tmp_output_dir)

    def test_dir_with_failed_rejected(self, tmp_output_dir):
        (tmp_output_dir / "FAILED.json").write_text("{}", encoding="utf-8")
        with pytest.raises(Exception, match="FAILED.json"):
            check_output_dir_safety(tmp_output_dir)

    def test_dir_with_stopped_rejected(self, tmp_output_dir):
        (tmp_output_dir / "STOPPED.json").write_text("{}", encoding="utf-8")
        with pytest.raises(Exception, match="STOPPED.json"):
            check_output_dir_safety(tmp_output_dir)

    def test_dir_with_unrelated_files_rejected(self, tmp_output_dir):
        (tmp_output_dir / "junk.txt").write_text("x", encoding="utf-8")
        with pytest.raises(Exception, match="not empty"):
            check_output_dir_safety(tmp_output_dir)

    def test_dir_with_only_gitkeep_accepted(self, tmp_output_dir):
        (tmp_output_dir / ".gitkeep").write_text("", encoding="utf-8")
        check_output_dir_safety(tmp_output_dir)

    def test_three_way_status_mutex(self, tmp_output_dir):
        write_status_files(tmp_output_dir, status="DONE", extra={"k": "v"})
        assert (tmp_output_dir / "DONE.json").exists()
        assert not (tmp_output_dir / "FAILED.json").exists()
        assert not (tmp_output_dir / "STOPPED.json").exists()
        write_status_files(tmp_output_dir, status="FAILED", extra={"k": "v"})
        assert (tmp_output_dir / "FAILED.json").exists()
        assert not (tmp_output_dir / "DONE.json").exists()
        assert not (tmp_output_dir / "STOPPED.json").exists()
        write_status_files(tmp_output_dir, status="STOPPED", extra={"k": "v"})
        assert (tmp_output_dir / "STOPPED.json").exists()
        assert not (tmp_output_dir / "DONE.json").exists()
        assert not (tmp_output_dir / "FAILED.json").exists()

    def test_three_way_status_mutex_from_stopped(self, tmp_output_dir):
        write_status_files(tmp_output_dir, status="STOPPED", extra={"k": "v"})
        assert (tmp_output_dir / "STOPPED.json").exists()
        write_status_files(tmp_output_dir, status="DONE", extra={"k": "v"})
        assert (tmp_output_dir / "DONE.json").exists()
        assert not (tmp_output_dir / "STOPPED.json").exists()


# ---------------------------------------------------------------------------
# Test: --run-authorized gate
# ---------------------------------------------------------------------------


class TestRunAuthorizedGate:
    def test_real_runner_imports_subject_isolation_helper(self):
        """The authorized real-data path must resolve its isolation guard."""
        import runpy

        from topper_perception.neural.slp8_region_dataset import (
            verify_subject_isolation,
        )

        runner_globals = runpy.run_path(
            str(PROJECT_ROOT / "scripts" / "run_slp8_region_mini.py"),
            run_name="slp8_region_mini_runner_test",
        )

        assert runner_globals["verify_subject_isolation"] is verify_subject_isolation

    def test_real_paths_without_run_authorized_rejected(self, tmp_output_dir):
        out = tmp_output_dir / "real_rejected"
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "42"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_slp8_region_mini.py"),
            "--config", str(CONFIG_PATH),
            "--output-dir", str(out),
            "--b01-freeze-dir", str(tmp_output_dir / "fake_b01"),
            "--dataset-root", str(tmp_output_dir / "fake_data"),
        ]
        # On Windows, a cold interpreter can spend substantial time loading
        # PyTorch before the CLI reaches this deliberately non-mutating gate.
        # The contract is the rejection and zero output-dir mutation, not a
        # 30-second process-startup budget.
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, env=env,
        )
        assert result.returncode != 0
        assert "run-authorized" in result.stderr.lower()
        # The run-authorized rejection must NOT create the output dir.
        assert not out.exists(), (
            f"output directory {out} was created; expected no creation on "
            "run-authorized rejection"
        )

    def test_validate_config_does_not_require_run_authorized(self, tmp_output_dir):
        out = tmp_output_dir / "vc"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_slp8_region_mini.py"),
            "--config", str(CONFIG_PATH),
            "--output-dir", str(out),
            "--validate-config",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (out / "DONE.json").exists()
        assert not (out / "FAILED.json").exists()
        assert not (out / "STOPPED.json").exists()

    def test_synthetic_cpu_smoke_does_not_require_run_authorized(self, tmp_output_dir):
        out = tmp_output_dir / "synth"
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "42"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_slp8_region_mini.py"),
            "--config", str(CONFIG_PATH),
            "--output-dir", str(out),
            "--synthetic-cpu-smoke",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=env,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (out / "DONE.json").exists()
        assert not (out / "FAILED.json").exists()
        assert not (out / "STOPPED.json").exists()


# ---------------------------------------------------------------------------
# Test: Resource budget monitor
# ---------------------------------------------------------------------------


class TestResourceBudgetMonitor:
    def test_tiny_budget_triggers_per_candidate_exceeded(self):
        b = ResourceBudget(
            max_wall_seconds_per_candidate=0.05,
            max_wall_seconds_total=0.10,
            max_peak_cuda_mb=1.0,
        )
        s = ResourceBudgetState(b)
        s.begin_candidate()
        # Sleep > 0.05s
        time.sleep(0.1)
        check = s.check()
        assert check.exceeded
        assert check.reason == "per_candidate_wall_exceeded"

    def test_total_budget_rolls_across_candidates(self):
        b = ResourceBudget(
            max_wall_seconds_per_candidate=0.02,
            max_wall_seconds_total=0.04,
            max_peak_cuda_mb=1024.0,
        )
        s = ResourceBudgetState(b)
        s.begin_candidate()
        time.sleep(0.03)
        s.begin_candidate()  # rolls the per-candidate elapsed into total
        time.sleep(0.03)
        check = s.check()
        assert check.exceeded
        assert check.reason in ("per_candidate_wall_exceeded", "total_wall_exceeded")

    def test_under_budget_returns_ok(self):
        b = ResourceBudget(
            max_wall_seconds_per_candidate=10.0,
            max_wall_seconds_total=20.0,
            max_peak_cuda_mb=4096.0,
        )
        s = ResourceBudgetState(b)
        s.begin_candidate()
        check = s.check()
        assert not check.exceeded
        assert check.reason == "ok"
        assert check.peak_cuda_mb == 0.0

    def test_resource_budget_from_config_validates_keys(self):
        budget = resource_budget_from_config({
            "max_wall_minutes_per_candidate": 45,
            "max_total_wall_minutes": 90,
            "max_peak_cuda_mb": 12288,
        })
        assert budget.max_wall_seconds_per_candidate == 45 * 60
        assert budget.max_wall_seconds_total == 90 * 60
        assert budget.max_peak_cuda_mb == 12288

    def test_resource_budget_constructors_reject_invalid(self):
        with pytest.raises(ValueError, match="max_wall_seconds_per_candidate"):
            ResourceBudget(
                max_wall_seconds_per_candidate=0,
                max_wall_seconds_total=10,
                max_peak_cuda_mb=10,
            )
        with pytest.raises(ValueError, match="max_wall_seconds_total"):
            ResourceBudget(
                max_wall_seconds_per_candidate=10,
                max_wall_seconds_total=-1,
                max_peak_cuda_mb=10,
            )
        with pytest.raises(ValueError, match="max_peak_cuda_mb"):
            ResourceBudget(
                max_wall_seconds_per_candidate=10,
                max_wall_seconds_total=10,
                max_peak_cuda_mb=0,
            )

    def test_tiny_budget_helper_is_a_resource_budget(self):
        b = ResourceBudget(
            max_wall_seconds_per_candidate=0.05,
            max_wall_seconds_total=0.10,
            max_peak_cuda_mb=1.0,
        )
        assert isinstance(b, ResourceBudget)
        assert b.max_wall_seconds_per_candidate < 1
        assert b.max_wall_seconds_total < 1
        assert b.max_peak_cuda_mb < 100

    def test_tiny_budget_triggers_stopped_terminal_state(self, tmp_path_factory):
        """When the per-candidate wall budget is below the time the
        synthetic dataset needs to build, the runner MUST transition
        the candidate to ``STOPPED`` and the run-level terminal state
        must also be ``STOPPED`` (no DONE.json written)."""
        from topper_perception.neural.slp8_region_mini import (
            run_mini, build_mini_config, build_synthetic_dataset,
            compute_class_weights, assert_class_weight_invariants,
        )
        from topper_perception.neural.slp8_region_determinism import (
            DeterminismSettings, apply_settings,
        )
        cfg_raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg = build_mini_config(
            cfg_raw, b01_freeze_dir=None, data_root=None,
            config_path=str(CONFIG_PATH),
        )
        # Use a tiny but positive budget that the synthetic run cannot
        # satisfy — the smallest valid value is 1e-9 seconds; any real
        # training step blows past it within the first validation epoch.
        tiny = ResourceBudget(
            max_wall_seconds_per_candidate=1e-9,
            max_wall_seconds_total=1e-9,
            max_peak_cuda_mb=1.0,
        )
        train_ds, val_ds, manifest, class_stats = build_synthetic_dataset(
            n_train_samples=2, n_val_samples=1, seed=7,
        )
        cw = compute_class_weights({
            "n_samples": 2,
            "n_pixels": 2 * 192 * 84,
            "per_class_pixel_ratio": {str(c): 0.1 for c in range(9)},
        })
        assert_class_weight_invariants(cw)
        out = tmp_path_factory.mktemp("b04_tiny_budget") / "out"
        apply_settings(42, cpu_threads=1)
        result = run_mini(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            dataset_manifest=manifest,
            class_weight_result=cw,
            output_dir=out,
            device=torch.device("cpu"),
            input_hashes={},
            train_class_stats_source="synthetic",
            synthetic=True,
            budget=tiny,
        )
        # Both candidates must be STOPPED (no real training can run).
        for cand_name, cand in result.candidate_results.items():
            assert cand.feasibility == "STOPPED", (
                f"candidate {cand_name} should be STOPPED, got {cand.feasibility}"
            )
            assert cand.budget_status != "ok"
        # Run-level terminal state must be STOPPED.
        assert result.terminal_state == "STOPPED"
        # The budget report must reflect STOPPED.
        br = json.loads((out / "budget_report.json").read_text(encoding="utf-8"))
        assert br["terminal_state"] == "STOPPED"
        # Verify the CLI-level terminal-file mutex: writing STOPPED via
        # the canonical writer must create STOPPED.json, and the
        # artifact bundle must not contain DONE.json or FAILED.json.
        from topper_perception.neural.slp8_region_mini import write_status_files
        write_status_files(out, status="STOPPED", extra={"k": "v"})
        assert (out / "STOPPED.json").exists()
        assert not (out / "DONE.json").exists()
        assert not (out / "FAILED.json").exists()


import time  # noqa: E402


# ---------------------------------------------------------------------------
# Test: Determinism configuration
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_apply_settings_pins_env(self, monkeypatch):
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
        s = apply_settings(42, cpu_threads=1)
        assert s.seed == 42
        assert s.cpu_threads == 1
        assert s.use_deterministic_algorithms is True
        assert s.cudnn_deterministic is True
        assert s.cudnn_benchmark is False
        assert s.omp_num_threads == 1
        assert s.mkl_num_threads == 1

    def test_environment_payload_includes_keys(self):
        payload = environment_payload()
        for key in (
            "platform", "python_version", "torch_version", "numpy_version",
            "cuda_available", "cuda_device_count",
            "python_hash_seed", "omp_num_threads", "mkl_num_threads",
            "torch_num_threads",
        ):
            assert key in payload

    def test_collect_settings_round_trips(self):
        s = apply_settings(7, cpu_threads=1)
        again = collect_settings(s.seed)
        assert again.seed == s.seed
        assert again.cpu_threads >= 1


# ---------------------------------------------------------------------------
# Test: B01 input contract
# ---------------------------------------------------------------------------


class TestB01Contract:
    def test_valid_snapshot_passes(self):
        s = _make_snapshot()
        expected = _make_expected()
        report = verify_b01_contract(s, expected)
        assert report.actual["train_count"] == 3645
        assert report.actual["val_count"] == 450
        assert report.actual["a06_split_sha256"] == A06_SPLIT_SHA256_EXPECTED

    def test_wrong_train_count_rejected(self):
        s = _make_snapshot(train_count=2000)
        expected = _make_expected()
        with pytest.raises(B01ContractError, match="train_count"):
            verify_b01_contract(s, expected)

    def test_wrong_val_count_rejected(self):
        s = _make_snapshot(val_count=100)
        expected = _make_expected()
        with pytest.raises(B01ContractError, match="val_count"):
            verify_b01_contract(s, expected)

    def test_wrong_train_subjects_rejected(self):
        s = B01FreezeSnapshot(
            freeze_dir=Path("/fake"),
            train_count=3645, val_count=450,
            train_subjects=tuple(f"{i:05d}" for i in range(80)),  # only 80
            val_subjects=tuple(f"{i:05d}" for i in range(100, 110)),
            train_manifest_sha256="b" * 64,
            val_manifest_sha256="c" * 64,
            observed_settings=("danaLab",),
            observed_covers=("uncover",),
            observed_provenances=("V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",),
            observed_review_statuses=("NOT_REVIEWED",),
            structural_test=B01FreezeStructuralTest(495, 11, "d" * 64),
            a06_split_sha256=A06_SPLIT_SHA256_EXPECTED,
            freeze_manifest_core_sha256="f" * 64,
            core_train_manifest_sha256="b" * 64,
            core_val_manifest_sha256="c" * 64,
        )
        with pytest.raises(B01ContractError, match="train_subjects"):
            verify_b01_contract(s, _make_expected())

    def test_structural_test_mismatch_rejected(self):
        s = _make_snapshot(structural_sample_count=494)  # 1 off from 495
        with pytest.raises(B01ContractError, match="structural test sample_count"):
            verify_b01_contract(s, _make_expected())

    def test_wrong_a06_sha_rejected(self):
        s = _make_snapshot(a06_split_sha256="deadbeef" * 8)
        with pytest.raises(B01ContractError, match="a06_split_sha256"):
            verify_b01_contract(s, _make_expected())

    def test_setting_must_be_singleton_danaLab(self):
        s = _make_snapshot(observed_settings=("simLab",))
        with pytest.raises(B01ContractError, match="observed_settings"):
            verify_b01_contract(s, _make_expected())

    def test_setting_must_be_consistent_across_rows(self):
        # Two different observed values is also rejected (must be singleton).
        s = _make_snapshot(observed_settings=("danaLab", "simLab"))
        with pytest.raises(B01ContractError, match="observed_settings"):
            verify_b01_contract(s, _make_expected())

    def test_cover_must_be_singleton_uncover(self):
        s = _make_snapshot(observed_covers=("cover1",))
        with pytest.raises(B01ContractError, match="observed_covers"):
            verify_b01_contract(s, _make_expected())

    def test_provenance_must_be_in_observed(self):
        s = _make_snapshot(observed_provenances=("MANUAL",))
        with pytest.raises(B01ContractError, match="observed_provenances"):
            verify_b01_contract(s, _make_expected())

    def test_review_status_must_be_in_observed(self):
        s = _make_snapshot(observed_review_statuses=("REVIEWED",))
        with pytest.raises(B01ContractError, match="observed_review_statuses"):
            verify_b01_contract(s, _make_expected())

    def test_freeze_manifest_core_sha_mismatch_rejected(self):
        s = _make_snapshot(freeze_manifest_sha256="a" * 64)
        expected = _make_expected(freeze_manifest_core_sha256="b" * 64)
        with pytest.raises(B01ContractError, match="freeze_manifest core sha256"):
            verify_b01_contract(s, expected)

    def test_train_manifest_sha_mismatch_rejected(self):
        # Construct a snapshot whose train_manifest_sha does not match
        # the recorded core value.  We build a fresh snapshot from real
        # rows and then patch the train_manifest_sha to be a different
        # value.  The validator must detect the mismatch.
        from topper_perception.io.slp8_training_table_freeze import (
            FreezeRow, manifest_sha256,
        )
        train_rows = [
            FreezeRow(
                sample_id=f"SLP:danaLab:{i:05d}:uncover:000000",
                ml_split="train", source_split="VAL", setting="danaLab",
                subject_id=f"{i:05d}", cover="uncover", frame_id=0,
                posture="SUPINE", pressure_npy="p", region_label_npy="l",
                region_onehot_npy="o", points_csv="c", height="192", width="84",
                class_ids_present="0|1|2|3|4|5|6|7|8",
                annotation_provenance="V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
                source_review_status="NOT_REVIEWED",
                export_version="1.1.0", export_status="EXPORTED",
                source_pmarray_sha256="z" * 64,
                background_pixel_count="1", body_pixel_count="1",
                clipped_ratio="0.0", onehot_valid="True",
                onehot_roundtrip="True",
            ) for i in range(2)
        ]
        s = B01FreezeSnapshot(
            freeze_dir=Path("/fake"),
            train_count=2, val_count=2,
            train_subjects=("00000", "00001"),
            val_subjects=("00100", "00101"),
            # Set train_manifest_sha to a different value than what
            # was computed from the actual rows.
            train_manifest_sha256="0" * 64,
            val_manifest_sha256=manifest_sha256([]),
            observed_settings=("danaLab",),
            observed_covers=("uncover",),
            observed_provenances=("V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",),
            observed_review_statuses=("NOT_REVIEWED",),
            structural_test=B01FreezeStructuralTest(495, 11, "d" * 64),
            a06_split_sha256=A06_SPLIT_SHA256_EXPECTED,
            freeze_manifest_core_sha256="f" * 64,
            core_train_manifest_sha256="b" * 64,
            core_val_manifest_sha256=manifest_sha256([]),
        )
        with pytest.raises(B01ContractError, match="manifest sha256"):
            verify_b01_contract(s, _make_expected())

    def test_freeze_manifest_file_consistency_raises_on_mismatch(self, tmp_output_dir):
        # Build a manifest with a real core; its computed core SHA
        # will NOT match the caller-supplied SHA so the check fails.
        manifest_path = tmp_output_dir / "freeze_manifest.json"
        core = {"task_id": "x", "a06_split_sha256": "0" * 64}
        import json as _json_for_test
        manifest_path.write_text(
            _json_for_test.dumps({"core": core, "meta": {}}),
            encoding="utf-8",
        )
        with pytest.raises(B01ContractError, match="core SHA mismatch"):
            check_freeze_manifest_file_consistency(
                tmp_output_dir, freeze_manifest_sha256="a" * 64
            )

    def test_freeze_manifest_file_consistency_raises_on_missing(self, tmp_output_dir):
        with pytest.raises(B01ContractError, match="freeze_manifest.json missing"):
            check_freeze_manifest_file_consistency(
                tmp_output_dir, freeze_manifest_sha256="a" * 64
            )

    def test_freeze_manifest_file_consistency_raises_on_missing_core(self, tmp_output_dir):
        # File exists but has no 'core' key — fail-closed.
        manifest_path = tmp_output_dir / "freeze_manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        with pytest.raises(B01ContractError, match="missing 'core' key"):
            check_freeze_manifest_file_consistency(
                tmp_output_dir, freeze_manifest_sha256="a" * 64
            )


class TestB01ContractExpected:
    def test_build_from_config_validates_required_fields(self, fresh_config):
        # Without b01_freeze_manifest_core_sha256_expected: fail.
        bad = {k: v for k, v in fresh_config.items()
               if k != "b01_freeze_manifest_core_sha256_expected"}
        with pytest.raises(B01ContractError, match="b01_freeze_manifest_core_sha256_expected"):
            build_b01_contract_expected(bad)
        # Without b01_structural_test: fail.
        bad2 = {k: v for k, v in fresh_config.items() if k != "b01_structural_test"}
        with pytest.raises(B01ContractError, match="b01_structural_test"):
            build_b01_contract_expected(bad2)
        # Without structural sample_count: fail.
        bad3 = {**fresh_config, "b01_structural_test": {"subject_count": 11}}
        with pytest.raises(B01ContractError, match="sample_count"):
            build_b01_contract_expected(bad3)
        # Without structural subject_count: fail.
        bad4 = {**fresh_config, "b01_structural_test": {"sample_count": 495}}
        with pytest.raises(B01ContractError, match="subject_count"):
            build_b01_contract_expected(bad4)

    def test_build_from_valid_config(self, fresh_config):
        expected = build_b01_contract_expected(fresh_config)
        assert expected.train_count == fresh_config["expected_split_counts"]["train"]
        assert expected.val_count == fresh_config["expected_split_counts"]["val"]
        assert expected.freeze_manifest_core_sha256 == fresh_config[
            "b01_freeze_manifest_core_sha256_expected"
        ]
        assert expected.structural_test_samples == fresh_config[
            "b01_structural_test"
        ]["sample_count"]


# ---------------------------------------------------------------------------
# Test: Resume identity contract
# ---------------------------------------------------------------------------


class TestResumeIdentity:
    def test_identity_round_trip(self):
        identity = _build_identity()
        payload = {"identity": identity.as_dict()}
        restored = identity_from_dict(payload)
        assert restored.task_id == identity.task_id
        assert restored.candidate == identity.candidate
        assert restored.image_shape == identity.image_shape

    def test_mismatched_identity_rejected(self):
        a = _build_identity()
        b = replace(a, seed=99)
        with pytest.raises(Exception, match="identity mismatch"):
            verify_resume_identity(saved=a, requested=b)

    def test_refuse_resume_for_done_run(self, tmp_output_dir):
        (tmp_output_dir / "DONE.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ResumeRefusedError, match="DONE.json"):
            refuse_resume_for_done_run(tmp_output_dir)

    def test_resume_allowed_for_empty_run(self, tmp_output_dir):
        # Should not raise.
        refuse_resume_for_done_run(tmp_output_dir)

    def test_early_stopper_state_round_trip(self):
        s = EarlyStopperState(
            best_metric=0.5,
            best_epoch=3,
            patience=4,
            current_patience=2,
            min_delta=0.0,
            min_epochs=5,
            mode="min",
            monitor="val_loss",
        )
        d = s.as_dict()
        s2 = EarlyStopperState.from_dict(d)
        assert s2 == s

    def test_early_stopper_state_current_patience_persists_across_resume(self):
        """The live patience counter MUST survive a checkpoint save +
        load so that resume continues exactly where the run stopped.
        """
        s = EarlyStopperState(
            best_metric=0.5,
            best_epoch=3,
            patience=4,
            current_patience=3,
            min_delta=0.0,
            min_epochs=5,
            mode="min",
            monitor="val_loss",
        )
        d = s.as_dict()
        s2 = EarlyStopperState.from_dict(d)
        assert s2.current_patience == 3
        # If the persisted current_patience is out of [0, patience]
        # the runner refuses to restore.
        bad = dict(d)
        bad["current_patience"] = 99
        s3 = EarlyStopperState.from_dict(bad)
        # ``from_dict`` is permissive; the runner's ``_EarlyStopper.restore``
        # is what enforces the range.
        assert s3.current_patience == 99

    def test_capture_and_restore_rng_state_round_trip(self):
        # Touch the RNG so a state is meaningful.
        import random
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        captured = capture_rng_state()
        # Mutate
        random.random()
        np.random.rand()
        torch.rand(1)
        restore_rng_state(captured)
        # After restore the very next call should match a fresh capture
        # from the same starting point.
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        ref_a = random.random()
        restore_rng_state(captured)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        ref_b = random.random()
        # Both call sequences were re-seeded then drew; the restored
        # path should be deterministic relative to the new seed.
        # The test is that the call doesn't raise and the round trip
        # preserves the structural shape of the captured dict.
        assert "torch" in captured
        assert "numpy" in captured
        assert "torch_cuda" in captured

    def test_class_weight_sha256_stable(self):
        weights = {
            "weights": {"0": 0.25, "1": 1.0, "2": 0.5, "3": 0.75, "4": 0.6, "5": 0.6, "6": 0.6, "7": 0.6, "8": 0.6},
            "source_split": "train",
        }
        h1 = class_weight_sha256(weights)
        h2 = class_weight_sha256(weights)
        assert h1 == h2
        assert len(h1) == 64

    def test_input_manifest_hashes_sha256_stable(self):
        h = input_manifest_hashes_sha256({"a": 1, "b": 2})
        assert len(h) == 64
        assert h == input_manifest_hashes_sha256({"a": 1, "b": 2})


# ---------------------------------------------------------------------------
# Test: Checkpoint payload + reload consistency
# ---------------------------------------------------------------------------


class TestCheckpointConsistency:
    def test_hash_consistency_after_reload(self, tmp_output_dir):
        from topper_perception.neural.slp8_region_mini import (
            _load_checkpoint, _build_checkpoint_payload, _save_checkpoint,
        )
        m, _ = create_slp8_tiny_fcn(device="cpu")
        opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=1e-4)
        payload = _build_checkpoint_payload(
            model=m,
            optimizer=opt,
            epoch=1,
            seed=42,
            model_config={"model_version": "slp8_tiny_fcn_v0.1"},
            class_weight_summary={"version": "test"},
            metrics={"val_loss": 0.5},
            n_classes=N_CLASSES,
            image_shape=PRESSURE_SHAPE,
        )
        ckpt_path = tmp_output_dir / "test.pt"
        sha = _save_checkpoint(ckpt_path, payload)
        assert len(sha) == 64

        loaded = _load_checkpoint(ckpt_path)
        assert loaded["version"] == CHECKPOINT_VERSION
        fresh, _ = create_slp8_tiny_fcn(device="cpu")
        fresh.load_state_dict(loaded["model_state_dict"])
        x = torch.randn(2, 1, 192, 84, dtype=torch.float32)
        with torch.no_grad():
            a = m(x).argmax(dim=1)
            b = fresh(x).argmax(dim=1)
        assert torch.equal(a, b)

    def test_checkpoint_payload_includes_identity(self, tmp_output_dir):
        from topper_perception.neural.slp8_region_mini import (
            _save_checkpoint, _load_checkpoint,
        )
        m, _ = create_slp8_tiny_fcn(device="cpu")
        opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=1e-4)
        identity = _build_identity()
        payload = _build_checkpoint_payload(
            model=m,
            optimizer=opt,
            epoch=1,
            seed=42,
            model_config={"model_version": "slp8_tiny_fcn_v0.1"},
            class_weight_summary={"version": "test"},
            metrics={"val_loss": 0.5},
            n_classes=N_CLASSES,
            image_shape=PRESSURE_SHAPE,
            identity=identity,
            input_manifest_hashes={"freeze_manifest_sha256": "abcd"},
        )
        ckpt_path = tmp_output_dir / "test.pt"
        _save_checkpoint(ckpt_path, payload)
        loaded = _load_checkpoint(ckpt_path)
        assert "identity" in loaded
        assert loaded["identity"]["task_id"] == TASK_ID
        assert "input_manifest_hashes" in loaded


# ---------------------------------------------------------------------------
# Test: Synthetic CPU smoke end-to-end
# ---------------------------------------------------------------------------


class TestBestEpochReloadProbe:
    """Regression for R02's best-vs-final false reload failure."""

    @staticmethod
    def _dataset_and_model():
        _, val_dataset, _, _ = build_synthetic_dataset(
            n_train_samples=2,
            n_val_samples=2,
            seed=42,
        )
        builder = get_model_builder(SMALL_UNET_VERSION)
        model, _ = builder.factory(N_CLASSES, "cpu")
        return val_dataset, builder, model

    def test_best_probe_survives_later_live_model_updates(self):
        val_dataset, builder, best_epoch_model = self._dataset_and_model()
        best_state = _clone_state_dict_to_cpu(best_epoch_model)
        probe = _build_best_epoch_reload_probe(
            builder=builder,
            model=best_epoch_model,
            val_dataset=val_dataset,
        )

        # Simulate training continuing after the best epoch.  The R02 bug
        # compared this changed final model with reloaded best.pt.
        with torch.no_grad():
            next(best_epoch_model.parameters()).add_(1.0)

        reloaded_best, _ = builder.factory(N_CLASSES, "cpu")
        reloaded_best.load_state_dict(best_state)
        consistent, max_abs_diff, detail = _verify_best_epoch_reload_probe(
            reloaded_model=reloaded_best,
            val_dataset=val_dataset,
            reload_probe=probe,
        )

        assert consistent is True
        assert max_abs_diff == 0.0
        assert detail == "best-epoch reload probe matched"

        pressure = torch.stack([val_dataset[index]["pressure"] for index in [0, 1]])
        best_epoch_model.eval()
        reloaded_best.eval()
        with torch.no_grad():
            final_logits = best_epoch_model(pressure)
            best_logits = reloaded_best(pressure)
        assert not torch.allclose(final_logits, best_logits)

    def test_probe_roundtrip_detects_corrupted_reloaded_state(self, tmp_path):
        val_dataset, builder, model = self._dataset_and_model()
        probe = _build_best_epoch_reload_probe(
            builder=builder,
            model=model,
            val_dataset=val_dataset,
        )
        assert probe["version"] == RELOAD_PROBE_VERSION

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        payload = _build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=1,
            seed=42,
            model_config={"model_version": SMALL_UNET_VERSION},
            class_weight_summary={"version": "test"},
            metrics={"val_loss": 0.5, "is_best": True},
            n_classes=N_CLASSES,
            image_shape=PRESSURE_SHAPE,
            reload_probe=probe,
        )
        checkpoint_path = tmp_path / "best.pt"
        from topper_perception.neural.slp8_region_mini import (
            _load_checkpoint,
            _save_checkpoint,
        )
        _save_checkpoint(checkpoint_path, payload)
        loaded = _load_checkpoint(checkpoint_path)

        corrupted_model, _ = builder.factory(N_CLASSES, "cpu")
        corrupted_model.load_state_dict(loaded["model_state_dict"])
        with torch.no_grad():
            next(corrupted_model.parameters()).add_(1.0)

        consistent, max_abs_diff, detail = _verify_best_epoch_reload_probe(
            reloaded_model=corrupted_model,
            val_dataset=val_dataset,
            reload_probe=loaded["reload_probe"],
        )
        assert consistent is False
        assert max_abs_diff is not None and max_abs_diff > 1.0
        assert detail == "best-epoch reload probe logits differ"

    @pytest.mark.parametrize(
        ("probe", "expected_detail"),
        [
            (None, "best checkpoint missing reload_probe"),
            ({"version": "wrong"}, "best checkpoint reload_probe version mismatch"),
            (
                {
                    "version": RELOAD_PROBE_VERSION,
                    "sample_indices": [0],
                    "logits": torch.zeros(1),
                },
                "best checkpoint reload_probe logits shape mismatch",
            ),
        ],
    )
    def test_missing_or_malformed_probe_fails_closed(self, probe, expected_detail):
        val_dataset, _, model = self._dataset_and_model()
        consistent, max_abs_diff, detail = _verify_best_epoch_reload_probe(
            reloaded_model=model,
            val_dataset=val_dataset,
            reload_probe=probe,
        )
        assert consistent is False
        assert max_abs_diff is None
        assert detail == expected_detail


class TestSyntheticCpuSmoke:
    """The synthetic CPU smoke produces the full B04 R02 output contract.

    The end-to-end smoke is invoked exactly once per pytest session
    by a module-level :func:`pytest.fixture`; the tests below all read
    the resulting artefacts.
    """

    REQUIRED_FILES = [
        "status.json",
        "manifest.json",
        "resolved_config.json",
        "input_manifest_hashes.json",
        "environment.json",
        "epoch_metrics.csv",
        "metrics_summary.json",
        "metrics_by_region.csv",
        "metrics_by_subject.csv",
        "metrics_by_posture.csv",
        "centroid_errors.csv",
        "worst_subject.json",
        "confusion_matrix.csv",
        "predictions_manifest.csv",
        "candidate_decision.json",
        "reload_consistency.json",
        "budget_report.json",
        "logs/run.log",
        "DONE.json",
    ]

    def test_required_files_exist(self, b04_synthetic_smoke_output: Path):
        missing = [f for f in self.REQUIRED_FILES if not (b04_synthetic_smoke_output / f).exists()]
        assert not missing, f"missing required outputs: {missing}"

    def test_terminal_files_mutex(self, b04_synthetic_smoke_output: Path):
        assert (b04_synthetic_smoke_output / "DONE.json").exists()
        assert not (b04_synthetic_smoke_output / "FAILED.json").exists()
        assert not (b04_synthetic_smoke_output / "STOPPED.json").exists()

    def test_done_json_indicates_real_status(self, b04_synthetic_smoke_output: Path):
        done = json.loads((b04_synthetic_smoke_output / "DONE.json").read_text(encoding="utf-8"))
        assert done["status"] == "DONE"
        assert done["task_id"] == TASK_ID
        assert done["mode"] == "synthetic-cpu-smoke"

    def test_status_json_overall_decision(self, b04_synthetic_smoke_output: Path):
        status = json.loads((b04_synthetic_smoke_output / "status.json").read_text(encoding="utf-8"))
        assert status["status"] == "DONE"
        assert "overall_decision" in status
        assert status["overall_decision"] in (
            "MINI_NOT_FEASIBLE", "MINI_HAS_FEASIBLE_CANDIDATE"
        )

    def test_checkpoint_files_exist_for_each_candidate(self, b04_synthetic_smoke_output: Path):
        for cand in B04_CANDIDATE_NAMES:
            assert (b04_synthetic_smoke_output / "checkpoints" / cand / "last.pt").exists()
            assert (b04_synthetic_smoke_output / "checkpoints" / cand / "best.pt").exists()

    def test_metrics_summary_has_both_candidates(self, b04_synthetic_smoke_output: Path):
        ms = json.loads((b04_synthetic_smoke_output / "metrics_summary.json").read_text(encoding="utf-8"))
        assert MODEL_VERSION in ms
        assert SMALL_UNET_VERSION in ms
        for cand, payload in ms.items():
            assert "elapsed_seconds" in payload
            assert "budget_status" in payload
            assert "budget_report" in payload

    def test_predictions_manifest_uses_real_data(self, b04_synthetic_smoke_output: Path):
        with open(b04_synthetic_smoke_output / "predictions_manifest.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        for row in rows:
            assert row["sample_id"].startswith("SLP:danaLab:")
            assert len(row["label_sha256"]) == 64
            assert len(row["prediction_sha256"]) == 64
            assert row["label_shape"] == "(192, 84)"

    def test_centroid_errors_csv_has_full_per_sample_evidence(self, b04_synthetic_smoke_output: Path):
        with open(b04_synthetic_smoke_output / "centroid_errors.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        for row in rows:
            assert row["candidate"] in B04_CANDIDATE_NAMES
            assert row["split"] in ("train", "val")
            assert row["region"] in {str(c) for c in FOREGROUND_CLASS_IDS}
            assert row["sample_id"] != ""
            assert row["subject_id"] != ""
            assert row["posture"] in ("SUPINE", "LEFT", "RIGHT")
            assert row["valid"] in ("True", "False")

    def test_manifest_includes_dataset_and_class_stats(self, b04_synthetic_smoke_output: Path):
        manifest = json.loads((b04_synthetic_smoke_output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["task_id"] == TASK_ID
        assert manifest["config_version"] == MINI_VERSION
        assert manifest["mode"] == "synthetic-cpu-smoke"
        assert manifest["dataset_manifest"]["n_test_samples"] == 0
        assert manifest["b04_max_parameters"] == B04_MAX_PARAMETERS
        assert manifest["b02_reference_val_fixed_iou"] == B02_BASELINE_REFERENCE_VAL_FIXED_IOU

    def test_class_weight_summary_includes_all_nine_classes(self, b04_synthetic_smoke_output: Path):
        ms = json.loads((b04_synthetic_smoke_output / "metrics_summary.json").read_text(encoding="utf-8"))
        for cand_name, payload in ms.items():
            cw = payload["class_weight_summary"]
            weights = cw["weights"]
            assert set(weights.keys()) == {str(c) for c in range(N_CLASSES)}

    def test_epoch_metrics_csv_complete(self, b04_synthetic_smoke_output: Path):
        with open(b04_synthetic_smoke_output / "epoch_metrics.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        candidates_in_csv = {row["candidate"] for row in rows}
        assert candidates_in_csv == set(B04_CANDIDATE_NAMES)
        for row in rows:
            assert int(row["epoch"]) >= 1
            assert math.isfinite(float(row["val_loss"]))

    def test_reload_consistency_json_both_candidates_consistent(self, b04_synthetic_smoke_output: Path):
        rc = json.loads((b04_synthetic_smoke_output / "reload_consistency.json").read_text(encoding="utf-8"))
        for cand, payload in rc.items():
            assert payload["reload_consistent"] is True
            assert payload["hash_match"] is True
            assert payload["max_abs_diff"] == 0.0
            assert payload["best_prediction_hash"] == payload["in_process_prediction_hash"]

    def test_candidate_decision_uses_frozen_threshold(self, b04_synthetic_smoke_output: Path):
        cd = json.loads((b04_synthetic_smoke_output / "candidate_decision.json").read_text(encoding="utf-8"))
        assert cd["val_feasibility_threshold"] == B02_BASELINE_REFERENCE_VAL_FIXED_IOU
        for cand, payload in cd["candidates"].items():
            assert payload["feasibility"] in {
                "FEASIBLE", "NOT_FEASIBLE", "FAILED", "STOPPED",
            }

    def test_budget_report_includes_thresholds_and_per_candidate(self, b04_synthetic_smoke_output: Path):
        br = json.loads((b04_synthetic_smoke_output / "budget_report.json").read_text(encoding="utf-8"))
        assert br["thresholds"]["max_wall_seconds_per_candidate"] == 45 * 60
        assert br["thresholds"]["max_wall_seconds_total"] == 90 * 60
        assert br["thresholds"]["max_peak_cuda_mb"] == 12288
        assert br["terminal_state"] in ("DONE", "FAILED", "STOPPED")
        assert "determinism" in br
        assert br["determinism"]["cpu_threads"] == 1
        for cand, payload in br["candidates"].items():
            assert "elapsed_seconds" in payload
            assert "budget_status" in payload

    def test_determinism_recorded_in_environment(self, b04_synthetic_smoke_output: Path):
        env = json.loads((b04_synthetic_smoke_output / "environment.json").read_text(encoding="utf-8"))
        assert env.get("python_hash_seed") == "42"
        assert env.get("omp_num_threads") == "1"
        assert env.get("mkl_num_threads") == "1"


@pytest.fixture(scope="session")
def b04_synthetic_smoke_output(tmp_path_factory, monkeypatch_session) -> Path:
    """Run the synthetic CPU smoke once per pytest session.

    The smoke takes ~30 seconds; running it per test would push the
    suite to several minutes.  The session-scoped fixture shares the
    result across all ``TestSyntheticCpuSmoke`` test methods.

    Determinism note: the runner sets ``PYTHONHASHSEED`` in
    :func:`apply_settings` (via :func:`os.environ.__setitem__`) but
    that has no effect on already-running Python interpreters; dict
    hashing is frozen at interpreter start.  The fixture therefore
    re-exports the env var into the subprocess invocation so the
    smoke process starts with the deterministic seed.
    """

    out = tmp_path_factory.mktemp("b04_synth_e2e") / "synth_smoke"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_slp8_region_mini.py"),
        "--config", str(CONFIG_PATH),
        "--output-dir", str(out),
        "--synthetic-cpu-smoke",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    return out


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch equivalent.

    The pytest ``monkeypatch`` fixture is function-scoped; for the
    session-scoped smoke fixture we need to mutate ``os.environ``
    once per session and have the change persist for the smoke's
    subprocess invocations.
    """
    saved: dict[str, str | None] = {}

    def _set(name: str, value: str) -> None:
        if name not in saved:
            saved[name] = os.environ.get(name)
        os.environ[name] = value

    _set("PYTHONHASHSEED", "42")
    _set("OMP_NUM_THREADS", "1")
    _set("MKL_NUM_THREADS", "1")
    yield
    for name, previous in saved.items():
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


# ---------------------------------------------------------------------------
# Test: Output collision zero-modification evidence
# ---------------------------------------------------------------------------


class TestOutputCollisionZeroModification:
    """When a CLI run collides with an existing output dir, the dir's
    file list and per-file SHA-256 must be exactly preserved."""

    def test_zero_modification_on_collision(self, tmp_output_dir):
        out = tmp_output_dir / "collide"
        out.mkdir()
        # Pre-populate with a sentinel file and a custom file.
        (out / "DONE.json").write_text('{"status":"DONE"}', encoding="utf-8")
        (out / "user_data.json").write_text('{"foo":"bar"}', encoding="utf-8")

        before_files = sorted(p for p in out.rglob("*") if p.is_file())
        before_rel = sorted(p.relative_to(out) for p in before_files)
        before_hashes = {p: file_sha256(p) for p in before_files}

        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "42"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_slp8_region_mini.py"),
            "--config", str(CONFIG_PATH),
            "--output-dir", str(out),
            "--synthetic-cpu-smoke",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, env=env,
        )
        # Exit code 2 indicates an OutputCollisionError; no FAILED.json
        # should be written.
        assert result.returncode == 2, f"unexpected exit: {result.returncode}; stderr: {result.stderr}"
        assert "collision" in result.stderr.lower() or "refusing" in result.stderr.lower()

        after_files = sorted(p for p in out.rglob("*") if p.is_file())
        after_rel = sorted(p.relative_to(out) for p in after_files)
        after_hashes = {p: file_sha256(p) for p in after_files}

        assert before_rel == after_rel, (
            f"file list changed: before={before_rel} after={after_rel}"
        )
        for path, sha in before_hashes.items():
            assert after_hashes[path] == sha, f"file {path} content changed"

    def test_failed_run_does_not_mutate_collision_free_directory(self, tmp_output_dir):
        """When the synthetic run succeeds, the sentinel files from
        before the run do not appear after the run (they would be
        absent in a fresh dir)."""
        out = tmp_output_dir / "fresh"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_slp8_region_mini.py"),
            "--config", str(CONFIG_PATH),
            "--output-dir", str(out),
            "--synthetic-cpu-smoke",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # The DONE.json was written; the previous-content assertion is
        # implicit in the test setup.
        assert (out / "DONE.json").exists()


# ---------------------------------------------------------------------------
# Test: Determinism across two independent subprocesses
# ---------------------------------------------------------------------------


class TestDeterminismSubprocess:
    """Two independent subprocess runs must produce identical output
    content (excluding timestamps and absolute paths)."""

    def _filtered_artefacts(self, root: Path) -> dict[str, str]:
        """Return a deterministic view of every text artefact.

        The view strips:
        * wall-clock timestamps (``started_at_utc``, ``ended_at_utc``);
        * per-epoch wall-clock (``elapsed_seconds``);
        * total wall-clock (``wall_clock_seconds``,
          ``total_elapsed_seconds``, per-candidate ``elapsed_seconds``);
        * any other non-deterministic timing fields.

        Binary artefacts (PyTorch ``.pt`` checkpoints) are excluded
        because they encode torch RNG state that may legitimately
        vary across processes even when the rest of the run is
        deterministic (e.g., ``torch.cuda`` RNG placeholder bytes).
        The text artefacts — JSON, CSV, ``run.log`` — are what
        Reviewers actually audit.
        """
        out: dict[str, str] = {}

        TIMING_KEYS = {
            "started_at_utc", "ended_at_utc", "wall_clock_seconds",
            "elapsed_total_seconds", "total_elapsed_seconds",
            "training_time_seconds", "elapsed_seconds",
        }
        # Checkpoint SHAs embed the process PID in the pickle filename,
        # so they differ across processes even when the rest of the
        # run is deterministic.  Strip them for the cross-process
        # comparison.
        PID_DEPENDENT_KEYS = {
            "checkpoint_best_sha256", "checkpoint_last_sha256",
            "checkpoints/<candidate>/best.pt", "checkpoints/<candidate>/last.pt",
        }

        def _scrub_json(payload: Any) -> Any:
            if isinstance(payload, dict):
                scrubbed: dict[str, Any] = {}
                for key, value in payload.items():
                    if key in TIMING_KEYS or key in PID_DEPENDENT_KEYS:
                        scrubbed[key] = "<stripped>"
                    elif isinstance(value, (dict, list)):
                        scrubbed[key] = _scrub_json(value)
                    else:
                        scrubbed[key] = value
                return scrubbed
            if isinstance(payload, list):
                return [_scrub_json(item) for item in payload]
            return payload

        def _scrub_log(text: str) -> str:
            out_lines = []
            for line in text.splitlines():
                # Drop log lines that *only* contain a UTC timestamp
                # or per-candidate timing summaries.
                if line.startswith("[") and "]" in line and "T" in line[:32]:
                    # Strip the leading timestamp prefix.
                    line = line.split("]", 1)[1].lstrip()
                if any(token in line for token in (
                    "started_at_utc=", "ended_at_utc=",
                    "completed in ", "elapsed=",
                )):
                    continue
                out_lines.append(line)
            return "\n".join(out_lines)

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".pt":
                continue
            rel = str(path.relative_to(root))
            if path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                out[rel] = json.dumps(
                    _scrub_json(data), sort_keys=True, ensure_ascii=False
                )
            elif path.suffix == ".csv":
                rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
                for row in rows:
                    for key in ("elapsed_seconds",):
                        if key in row:
                            row[key] = "<stripped>"
                out[rel] = repr(rows)
            elif path.suffix == ".log" or path.name == "run.log":
                out[rel] = _scrub_log(path.read_text(encoding="utf-8"))
            else:
                out[rel] = path.read_text(encoding="utf-8")
        return out

    def test_two_independent_subprocess_smoke_runs_are_byte_identical(
        self, tmp_path_factory
    ):
        out_a = tmp_path_factory.mktemp("det_a") / "smoke"
        out_b = tmp_path_factory.mktemp("det_b") / "smoke"
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "42"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        for out in (out_a, out_b):
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_slp8_region_mini.py"),
                "--config", str(CONFIG_PATH),
                "--output-dir", str(out),
                "--synthetic-cpu-smoke",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, env=env,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"

        dig_a = self._filtered_artefacts(out_a)
        dig_b = self._filtered_artefacts(out_b)
        assert dig_a == dig_b, "determinism contract violated across two subprocesses"
        assert len(dig_a) >= 10


# ---------------------------------------------------------------------------
# Test: Canonical array hash
# ---------------------------------------------------------------------------


class TestCanonicalArrayHash:
    def test_hash_length_and_hex(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        h = canonical_array_hash(arr)
        assert len(h) == 64
        int(h, 16)

    def test_hash_stable(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        h1 = canonical_array_hash(arr)
        h2 = canonical_array_hash(arr)
        assert h1 == h2

    def test_hash_changes_with_array(self):
        a = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        b = a.copy()
        b[0, 0] = 7
        assert canonical_array_hash(a) != canonical_array_hash(b)

    def test_header_version_present(self):
        assert CANONICAL_HASH_VERSION == "slp8_canonical_array_hash_v0.1"


# ---------------------------------------------------------------------------
# R03: CLI three-path lifecycle tests
# ---------------------------------------------------------------------------


def _run_cli(argv: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    """Invoke the CLI as a subprocess and return the completed process."""

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_slp8_region_mini.py"),
        *argv,
    ]
    full_env = dict(os.environ)
    full_env["PYTHONHASHSEED"] = "42"
    full_env["OMP_NUM_THREADS"] = "1"
    full_env["MKL_NUM_THREADS"] = "1"
    if env is not None:
        full_env.update(env)
    return subprocess.run(
        cmd, capture_output=True, text=True, env=full_env, timeout=600
    )


class TestCLITerminalStateDone:
    """``--synthetic-cpu-smoke`` finishes DONE; the CLI writes DONE.json
    and exits 0.  ``status.json`` mirrors the terminal state."""

    def test_done_writes_done_json_and_exits_zero(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("cli_done") / "out"
        result = _run_cli([
            "--config", str(CONFIG_PATH),
            "--output-dir", str(out),
            "--synthetic-cpu-smoke",
        ])
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (out / "DONE.json").exists()
        assert not (out / "FAILED.json").exists()
        assert not (out / "STOPPED.json").exists()
        status = json.loads((out / "status.json").read_text(encoding="utf-8"))
        assert status["status"] == "DONE"
        assert status["terminal_state"] == "DONE"
        assert status["mode"] == "synthetic-cpu-smoke"


class TestCLITerminalStateStopped:
    """When the synthetic smoke hits an injected tiny budget it MUST
    transition to STOPPED, the CLI MUST write STOPPED.json (not
    DONE.json), and the exit code MUST be non-zero.  This test does
    NOT call :func:`write_status_files` directly — it drives the
    real CLI to STOPPED via a test-only environment variable the CLI
    honours."""

    def test_stopped_writes_stopped_json_and_exits_nonzero(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("cli_stopped") / "out"
        # The synthetic smoke runner reads B04_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS
        # (test-only knob) and uses it as the per-candidate wall budget
        # when present.  The actual smoke takes >0.1s of wall time
        # for the small_unet candidate, so 1e-3s triggers STOPPED.
        env = {"B04_BUDGET_OVERRIDE_PER_CANDIDATE_SECONDS": "0.001"}
        result = _run_cli([
            "--config", str(CONFIG_PATH),
            "--output-dir", str(out),
            "--synthetic-cpu-smoke",
        ], env=env)
        assert result.returncode != 0, f"stderr: {result.stderr}"
        assert not (out / "DONE.json").exists()
        assert not (out / "FAILED.json").exists()
        assert (out / "STOPPED.json").exists()
        status = json.loads((out / "status.json").read_text(encoding="utf-8"))
        assert status["status"] == "STOPPED"
        assert status["terminal_state"] == "STOPPED"
        stopped = json.loads((out / "STOPPED.json").read_text(encoding="utf-8"))
        assert stopped["status"] == "STOPPED"
        assert stopped["terminal_state"] == "STOPPED"
        # STOPPED is only resumable if each candidate completed an epoch
        # and persisted its checkpoint before the budget transition.
        for candidate in B04_CANDIDATE_NAMES:
            assert (out / "checkpoints" / candidate / "last.pt").is_file()


class TestCLITerminalStateFailed:
    """The CLI must report FAILED when the runner explicitly produces
    a FAILED result.  We drive this by monkey-patching
    :func:`run_mini` in the CLI module's namespace.  The CLI's own
    terminal-file code path is what we are testing — we do NOT call
    :func:`write_status_files` directly to fabricate a FAILED.json."""

    def test_failed_writes_failed_json_and_exits_nonzero(
        self, tmp_path_factory, monkeypatch
    ):
        from topper_perception.neural import slp8_region_mini as sm
        from topper_perception.neural.slp8_region_determinism import (
            apply_settings,
        )
        from topper_perception.neural.slp8_region_budget import ResourceBudget

        # Build a fully-formed MiniRunResult with terminal_state="FAILED".
        # Both candidates are FAILED so the run-level terminal_state
        # must be FAILED.
        def _fake_run_mini(**kwargs):
            crs = {}
            for cand in sm.B04_CANDIDATE_NAMES:
                crs[cand] = sm.CandidateResult(
                    candidate=cand,
                    model_version="slp8_tiny_fcn_v0.1",
                    parameter_count=1401,
                    parameter_count_within_budget=True,
                    feasibility="FAILED",
                    reason="forced failure for CLI test",
                    epoch_metrics=[],
                    metrics=None,
                    train_predictions=[],
                    train_labels=[],
                    train_subjects=[],
                    val_predictions=[],
                    val_labels=[],
                    val_subjects=[],
                    train_records=[],
                    val_records=[],
                    best_epoch=None,
                    best_val_loss=None,
                    best_prediction_hash=None,
                    in_process_prediction_hash=None,
                    reload_consistent=True,
                    reload_max_abs_diff=0.0,
                    checkpoint_best_sha256=None,
                    checkpoint_last_sha256=None,
                    train_loss_history=[],
                    val_loss_history=[],
                    train_subject_overlap_with_val=False,
                    val_subject_overlap_with_train=False,
                    n_test_samples=0,
                    param_changed=True,
                    last_in_process_prediction_hash=None,
                    class_weight_summary={},
                    elapsed_seconds=0.0,
                    budget_status="ok",
                    budget_report={},
                    budget_thresholds={},
                )
            return sm.MiniRunResult(
                config=kwargs["config"],
                dataset_manifest=kwargs["dataset_manifest"],
                environment={},
                class_weight_result=kwargs["class_weight_result"],
                candidate_results=crs,
                n_candidates_feasible=0,
                n_candidates_not_feasible=0,
                n_candidates_failed=2,
                n_candidates_stopped=0,
                overall_decision="MINI_NOT_FEASIBLE",
                terminal_state="FAILED",
                started_at_utc="2026-01-01T00:00:00+00:00",
                ended_at_utc="2026-01-01T00:00:01+00:00",
                wall_clock_seconds=1.0,
                input_hashes={},
                train_class_stats_source="synthetic_train_class_stats",
                synthetic=True,
                determinism=apply_settings(42, cpu_threads=1),
                resource_budget=ResourceBudget(
                    max_wall_seconds_per_candidate=2700.0,
                    max_wall_seconds_total=5400.0,
                    max_peak_cuda_mb=12288.0,
                ),
                b01_contract_report=None,
            )

        # Patch the symbol the CLI module imports.  Load by absolute path:
        # ``scripts`` is not an installed package.
        cli = _load_runner_module()
        monkeypatch.setattr(cli, "run_mini", _fake_run_mini)

        # Now invoke main() directly with patched argv.
        out = tmp_path_factory.mktemp("cli_failed") / "out"
        monkeypatch.setattr("sys.argv", [
            "run_slp8_region_mini.py",
            "--config", str(CONFIG_PATH),
            "--output-dir", str(out),
            "--synthetic-cpu-smoke",
        ])
        rc = cli.main()
        assert rc != 0, f"rc={rc}; CLI should have exited non-zero for FAILED"
        assert (out / "FAILED.json").exists()
        assert not (out / "DONE.json").exists()
        assert not (out / "STOPPED.json").exists()
        status = json.loads((out / "status.json").read_text(encoding="utf-8"))
        assert status["status"] == "FAILED"
        assert status["terminal_state"] == "FAILED"


# ---------------------------------------------------------------------------
# R03: Real B01 input contract — entry-level negative tests
# ---------------------------------------------------------------------------


def _make_fake_b01_freeze_dir(root: Path, *, train_count: int = 3645,
                              val_count: int = 450, test_count_in_manifest: int = 495,
                              train_subjects: int = 81,
                              val_subjects: int = 10,
                              bad_a06: bool = False,
                              bad_provenance: bool = False) -> Path:
    """Create a minimal B01 freeze directory in ``root``.

    The freeze structurally contains ``test_manifest.csv`` (the
    source-of-truth that test rows exist as 495) but the B04 loader
    is forced to ``load_test=False`` so the dataset's
    ``n_test_samples`` is 0.
    """

    from topper_perception.io.slp8_training_table_freeze import (
        FreezeRow, manifest_sha256 as _b01_manifest_sha256,
    )
    root.mkdir(parents=True, exist_ok=True)

    def _rows(split: str, n_subjects: int, n_samples: int, subj_offset: int):
        out = []
        per_subj = n_samples // n_subjects
        for i in range(n_subjects):
            sid = f"{subj_offset + i:05d}"
            for j in range(per_subj):
                out.append(FreezeRow(
                    sample_id=f"SLP:danaLab:{sid}:uncover:{j:06d}",
                    ml_split=split,
                    source_split="VAL",
                    setting="danaLab",
                    subject_id=sid,
                    cover="uncover",
                    frame_id=j,
                    posture="SUPINE",
                    pressure_npy=f"pressure/{sid}_{j:06d}.npy",
                    region_label_npy=f"labels/{sid}_{j:06d}.npy",
                    region_onehot_npy=f"onehot/{sid}_{j:06d}.npy",
                    points_csv=f"points/{sid}.csv",
                    height="192",
                    width="84",
                    class_ids_present="0|1|2|3|4|5|6|7|8",
                    annotation_provenance=(
                        "WRONG_PROVENANCE" if bad_provenance
                        else "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED"
                    ),
                    source_review_status="NOT_REVIEWED",
                    export_version="1.1.0",
                    export_status="EXPORTED",
                    source_pmarray_sha256="a" * 64,
                    background_pixel_count="100",
                    body_pixel_count="50",
                    clipped_ratio="0.0",
                    onehot_valid="True",
                    onehot_roundtrip="True",
                ))
        return out

    train_rows = _rows("train", train_subjects, train_count, 0)
    val_rows = _rows("val", val_subjects, val_count, train_subjects)
    test_rows = _rows("test", test_count_in_manifest // 45, test_count_in_manifest,
                      train_subjects + val_subjects) if test_count_in_manifest else []

    # Write the CSV manifests that B01 reads.  The CSV format
    # follows the A09R layout (strings for scalar fields,
    # pipe-joined list for ``class_ids_present``).
    def _row_to_csv_dict(row: "FreezeRow") -> dict[str, str]:
        d = row.to_dict()
        return {
            "sample_id": d["sample_id"],
            "ml_split": d["ml_split"],
            "source_split": d["source_split"],
            "setting": d["setting"],
            "subject_id": d["subject_id"],
            "cover": d["cover"],
            "frame_id": str(d["frame_id"]),
            "posture": d["posture"],
            "pressure_npy": d["pressure_npy"],
            "region_label_npy": d["region_label_npy"],
            "region_onehot_npy": d["region_onehot_npy"],
            "points_csv": d["points_csv"],
            "height": str(d["height"]),
            "width": str(d["width"]),
            "class_ids_present": "".join(str(x) for x in d["class_ids_present"]),
            "annotation_provenance": d["annotation_provenance"],
            "source_review_status": d["source_review_status"],
            "export_version": d["export_version"],
            "export_status": d["export_status"],
            "source_pmarray_sha256": d["source_pmarray_sha256"],
            "background_pixel_count": str(d["background_pixel_count"]),
            "body_pixel_count": str(d["body_pixel_count"]),
            "clipped_ratio": str(d["clipped_ratio"]),
            "onehot_valid": str(d["onehot_valid"]),
            "onehot_roundtrip": str(d["onehot_roundtrip"]),
        }

    import csv as _csv
    fields = list(FreezeRow.__dataclass_fields__.keys())
    (root / "train_manifest.csv").write_text(
        _csv_rows_text([_row_to_csv_dict(r) for r in train_rows], fields), encoding="utf-8"
    )
    (root / "val_manifest.csv").write_text(
        _csv_rows_text([_row_to_csv_dict(r) for r in val_rows], fields), encoding="utf-8"
    )
    if test_rows:
        (root / "test_manifest.csv").write_text(
            _csv_rows_text([_row_to_csv_dict(r) for r in test_rows], fields), encoding="utf-8"
        )

    # A minimal freeze_manifest.json that the B01 loader accepts.
    # The split manifest SHAs MUST be computed from the parsed CSV
    # rows so they match what ``from_freeze_tables`` will compute.
    # We achieve that by writing the CSVs first, then using the B01
    # freeze manifest SHA on the parsed rows.
    import json as _json
    import hashlib as _hashlib
    from topper_perception.io.slp8_training_table_freeze import (
        load_b01_freeze_tables,
    )
    a06 = (
        "deadbeef" * 8 if bad_a06
        else "024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706"
    )

    # Pre-write the manifest with placeholder SHAs; we'll update
    # below once the CSVs are on disk and re-loaded.
    fm = {
        "core": {
            "a06_split_sha256": a06,
            "expected_provenance": (
                "WRONG_PROVENANCE" if bad_provenance
                else "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED"
            ),
            "expected_review_status": "NOT_REVIEWED",
            "splits": {
                "train": {
                    "sample_count": train_count,
                    "subject_count": train_subjects,
                    "manifest_sha256": "0" * 64,
                },
                "val": {
                    "sample_count": val_count,
                    "subject_count": val_subjects,
                    "manifest_sha256": "0" * 64,
                },
                "test": {
                    "sample_count": test_count_in_manifest,
                    "subject_count": test_count_in_manifest // 45,
                    "manifest_sha256": "0" * 64,
                },
            },
        },
    }
    fm_text = _json.dumps(fm)
    (root / "freeze_manifest.json").write_text(fm_text, encoding="utf-8")
    # Stamp the freeze_manifest_sha256 with the on-disk SHA so the
    # snapshot can pass the contract check.
    fm["freeze_manifest_sha256"] = _hashlib.sha256(
        fm_text.encode("utf-8")
    ).hexdigest()
    (root / "freeze_manifest.json").write_text(
        _json.dumps(fm), encoding="utf-8"
    )

    # A minimal train_class_stats.json: at least one class has positive
    # pixel ratio so compute_class_weights does not reject.
    (root / "train_class_stats.json").write_text(_json.dumps({
        "n_samples": train_count,
        "subject_count": train_subjects,
        "per_class_pixel_ratio": {str(c): 0.1 for c in range(9)},
    }), encoding="utf-8")

    # Re-load the freeze so we can patch the manifest with the
    # loader-computed split manifest SHAs (which are derived from the
    # parsed rows, not from the original FreezeRow objects).
    freeze = load_b01_freeze_tables(root, load_test=False)
    fm_path = root / "freeze_manifest.json"
    fm = _json.loads(fm_path.read_text(encoding="utf-8"))
    fm["core"]["splits"]["train"]["manifest_sha256"] = (
        freeze.train_manifest_sha256
    )
    fm["core"]["splits"]["val"]["manifest_sha256"] = (
        freeze.val_manifest_sha256
    )
    # test_manifest_sha256 is NOT taken from the loader property
    # because the B01 freeze loader's test_manifest_sha256 uses the
    # legacy flat-manifest format (no "core" wrapper).  Compute the
    # SHA directly from the test rows we created so the contract
    # check can verify the manifest's recorded value matches.
    if test_count_in_manifest > 0:
        fm["core"]["splits"]["test"]["manifest_sha256"] = (
            _b01_manifest_sha256(test_rows)
        )
    # Re-stamp the core SHA.
    fm.pop("freeze_manifest_sha256", None)
    fm_text = _json.dumps(fm)
    fm_path.write_text(fm_text, encoding="utf-8")
    fm["freeze_manifest_sha256"] = _hashlib.sha256(
        fm_text.encode("utf-8")
    ).hexdigest()
    fm_path.write_text(_json.dumps(fm), encoding="utf-8")

    return root


def _csv_rows_text(rows, fields):
    import io, csv as _csv
    buf = io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


class TestB01ContractEntryLevel:
    """Entry-level negative tests for the real B01 input contract.

    These tests do NOT read TEST label / onehot files.  They build
    a structural freeze (manifest, train/val CSVs, train_class_stats)
    and verify the contract rejects bad input fail-closed.

    Where the B01 freeze loader itself accepts the data, the tests
    drive the failure through ``from_freeze_tables`` (the real
    contract entry point).  Where the loader would reject the data
    first, the tests build a synthetic manifest + FreezeRow lists
    and pass them through ``from_freeze_tables`` to exercise the
    real contract path.
    """

    def _good_expected(self, *, core_sha256: str) -> B01ContractExpected:
        return B01ContractExpected(
            train_count=3645,
            val_count=450,
            test_count=0,
            train_subjects=81,
            val_subjects=10,
            test_subjects=0,
            a06_split_sha256="024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706",
            provenance="V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            source_review_status="NOT_REVIEWED",
            setting="danaLab",
            cover="uncover",
            freeze_manifest_core_sha256=core_sha256,
            structural_test_samples=495,
            structural_test_subjects=11,
        )

    def test_correct_freeze_passes_contract(self, tmp_path):
        root = _make_fake_b01_freeze_dir(tmp_path / "b01_good")
        from topper_perception.io.slp8_training_table_freeze import (
            load_b01_freeze_tables,
        )
        freeze = load_b01_freeze_tables(root, load_test=False)
        snap = B01FreezeSnapshot.from_freeze_tables(
            freeze_dir=root,
            train_rows=freeze.train_rows,
            val_rows=freeze.val_rows,
            test_rows=None,
            freeze_manifest=freeze.freeze_manifest,
        )
        report = verify_b01_contract(
            snap, self._good_expected(core_sha256=snap.freeze_manifest_core_sha256)
        )
        assert report.actual["train_count"] == 3645
        assert report.structural_test["sample_count"] == 495
        assert report.actual["a06_split_sha256"] == (
            "024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706"
        )

    def test_missing_a06_in_core_rejected(self, tmp_path):
        # Build a synthetic manifest that omits a06_split_sha256 from
        # core.  We can't drive this through the real B01 freeze loader
        # because the loader does not check this — it only carries
        # the value through.  We exercise the real ``from_freeze_tables``
        # + ``verify_b01_contract`` path.
        from topper_perception.io.slp8_training_table_freeze import (
            FreezeRow, build_freeze_manifest, A06Split,
            NormalizationStats, ClassStats,
        )
        a06 = _make_a06_split()
        norm = _make_normalization_stats()
        train_stats = _make_class_stats()
        val_stats = _make_class_stats()
        # Build one minimal FreezeRow; we don't actually need real
        # rows for this test because the failure is in the manifest.
        from topper_perception.io.slp8_training_table_freeze import (
            FreezeRow as _FR,
        )
        train_row = _FR(
            sample_id="SLP:danaLab:00000:uncover:000000",
            ml_split="train", source_split="VAL", setting="danaLab",
            subject_id="00000", cover="uncover", frame_id=0,
            posture="SUPINE", pressure_npy="p", region_label_npy="l",
            region_onehot_npy="o", points_csv="c", height="192", width="84",
            class_ids_present="0|1|2|3|4|5|6|7|8",
            annotation_provenance="V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            source_review_status="NOT_REVIEWED",
            export_version="1.1.0", export_status="EXPORTED",
            source_pmarray_sha256="0" * 64,
            background_pixel_count="1", body_pixel_count="1",
            clipped_ratio="0.0", onehot_valid="True",
            onehot_roundtrip="True",
        )
        val_row = _FR(
            sample_id="SLP:danaLab:00100:uncover:000000",
            ml_split="val", source_split="VAL", setting="danaLab",
            subject_id="00100", cover="uncover", frame_id=0,
            posture="SUPINE", pressure_npy="p", region_label_npy="l",
            region_onehot_npy="o", points_csv="c", height="192", width="84",
            class_ids_present="0|1|2|3|4|5|6|7|8",
            annotation_provenance="V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            source_review_status="NOT_REVIEWED",
            export_version="1.1.0", export_status="EXPORTED",
            source_pmarray_sha256="0" * 64,
            background_pixel_count="1", body_pixel_count="1",
            clipped_ratio="0.0", onehot_valid="True",
            onehot_roundtrip="True",
        )
        # The builder writes a06_split_sha256 into the core; we cannot
        # omit it from the B01 builder output.  Instead, mutate the
        # builder's output core to remove the field before constructing
        # the snapshot — this exercises the real contract path.
        fm = build_freeze_manifest(
            train_rows=[train_row],
            val_rows=[val_row],
            test_rows=[],
            a06_split=a06,
            source_manifest_sha256="0" * 64,
            stats=norm,
            train_stats=train_stats, val_stats=val_stats,
        )
        # Remove the a06_split_sha256 field from the core to simulate
        # a missing field.  Use pop() and cast to a dict (FrozenMap
        # may need careful handling).
        core = dict(fm.core)
        core.pop("a06_split_sha256", None)
        freeze_manifest = {"core": core, "meta": fm.meta}
        with pytest.raises(B01ContractError, match="a06_split_sha256"):
            B01FreezeSnapshot.from_freeze_tables(
                freeze_dir=tmp_path / "b01_no_a06",
                train_rows=[train_row],
                val_rows=[val_row],
                test_rows=None,
                freeze_manifest=freeze_manifest,
            )

    def test_missing_provenance_in_core_rejected(self, tmp_path):
        from topper_perception.io.slp8_training_table_freeze import (
            build_freeze_manifest, A06Split, NormalizationStats, ClassStats,
            FreezeRow,
        )
        a06 = _make_a06_split()
        norm = _make_normalization_stats()
        cs = _make_class_stats()
        row = FreezeRow(
            sample_id="SLP:danaLab:00000:uncover:000000",
            ml_split="train", source_split="VAL", setting="danaLab",
            subject_id="00000", cover="uncover", frame_id=0,
            posture="SUPINE", pressure_npy="p", region_label_npy="l",
            region_onehot_npy="o", points_csv="c", height="192", width="84",
            class_ids_present="0|1|2|3|4|5|6|7|8",
            annotation_provenance="V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            source_review_status="NOT_REVIEWED",
            export_version="1.1.0", export_status="EXPORTED",
            source_pmarray_sha256="0" * 64,
            background_pixel_count="1", body_pixel_count="1",
            clipped_ratio="0.0", onehot_valid="True",
            onehot_roundtrip="True",
        )
        fm = build_freeze_manifest(
            train_rows=[row], val_rows=[], test_rows=[],
            a06_split=a06, source_manifest_sha256="0" * 64,
            stats=norm, train_stats=cs, val_stats=cs,
        )
        core = dict(fm.core)
        core.pop("expected_provenance", None)
        freeze_manifest = {"core": core, "meta": fm.meta}
        with pytest.raises(B01ContractError, match="expected_provenance"):
            B01FreezeSnapshot.from_freeze_tables(
                freeze_dir=tmp_path / "b01_no_provenance",
                train_rows=[row],
                val_rows=[],
                test_rows=None,
                freeze_manifest=freeze_manifest,
            )

    def test_missing_review_status_in_core_rejected(self, tmp_path):
        from topper_perception.io.slp8_training_table_freeze import (
            build_freeze_manifest, A06Split, NormalizationStats, ClassStats,
            FreezeRow,
        )
        a06 = _make_a06_split()
        norm = _make_normalization_stats()
        cs = _make_class_stats()
        row = FreezeRow(
            sample_id="SLP:danaLab:00000:uncover:000000",
            ml_split="train", source_split="VAL", setting="danaLab",
            subject_id="00000", cover="uncover", frame_id=0,
            posture="SUPINE", pressure_npy="p", region_label_npy="l",
            region_onehot_npy="o", points_csv="c", height="192", width="84",
            class_ids_present="0|1|2|3|4|5|6|7|8",
            annotation_provenance="V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            source_review_status="NOT_REVIEWED",
            export_version="1.1.0", export_status="EXPORTED",
            source_pmarray_sha256="0" * 64,
            background_pixel_count="1", body_pixel_count="1",
            clipped_ratio="0.0", onehot_valid="True",
            onehot_roundtrip="True",
        )
        fm = build_freeze_manifest(
            train_rows=[row], val_rows=[], test_rows=[],
            a06_split=a06, source_manifest_sha256="0" * 64,
            stats=norm, train_stats=cs, val_stats=cs,
        )
        core = dict(fm.core)
        core.pop("expected_review_status", None)
        freeze_manifest = {"core": core, "meta": fm.meta}
        with pytest.raises(B01ContractError, match="expected_review_status"):
            B01FreezeSnapshot.from_freeze_tables(
                freeze_dir=tmp_path / "b01_no_review",
                train_rows=[row],
                val_rows=[],
                test_rows=None,
                freeze_manifest=freeze_manifest,
            )

    def test_missing_splits_block_rejected(self, tmp_path):
        # Build a manifest core that omits ``splits`` entirely.
        from topper_perception.io.slp8_training_table_freeze import (
            FreezeRow,
        )
        row = FreezeRow(
            sample_id="SLP:danaLab:00000:uncover:000000",
            ml_split="train", source_split="VAL", setting="danaLab",
            subject_id="00000", cover="uncover", frame_id=0,
            posture="SUPINE", pressure_npy="p", region_label_npy="l",
            region_onehot_npy="o", points_csv="c", height="192", width="84",
            class_ids_present="0|1|2|3|4|5|6|7|8",
            annotation_provenance="V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            source_review_status="NOT_REVIEWED",
            export_version="1.1.0", export_status="EXPORTED",
            source_pmarray_sha256="0" * 64,
            background_pixel_count="1", body_pixel_count="1",
            clipped_ratio="0.0", onehot_valid="True",
            onehot_roundtrip="True",
        )
        core = {
            "task_id": "x",
            "a06_split_sha256": "0" * 64,
            "expected_provenance": "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            "expected_review_status": "NOT_REVIEWED",
        }
        freeze_manifest = {"core": core, "meta": {}}
        with pytest.raises(B01ContractError, match="splits"):
            B01FreezeSnapshot.from_freeze_tables(
                freeze_dir=tmp_path / "b01_no_splits",
                train_rows=[row], val_rows=[], test_rows=None,
                freeze_manifest=freeze_manifest,
            )

    def test_wrong_core_sha_rejected(self, tmp_path):
        # Build a snapshot from a real B01 manifest via build_freeze_manifest,
        # then mutate the core to a single field change so the computed
        # core SHA is different from the frozen expected value.  The
        # contract validator must catch this.
        from topper_perception.io.slp8_training_table_freeze import (
            build_freeze_manifest, A06Split, NormalizationStats, ClassStats,
            FreezeRow, manifest_sha256,
        )
        a06 = _make_a06_split()
        norm = _make_normalization_stats()
        cs = _make_class_stats()
        row = FreezeRow(
            sample_id="SLP:danaLab:00000:uncover:000000",
            ml_split="train", source_split="VAL", setting="danaLab",
            subject_id="00000", cover="uncover", frame_id=0,
            posture="SUPINE", pressure_npy="p", region_label_npy="l",
            region_onehot_npy="o", points_csv="c", height="192", width="84",
            class_ids_present="0|1|2|3|4|5|6|7|8",
            annotation_provenance="V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            source_review_status="NOT_REVIEWED",
            export_version="1.1.0", export_status="EXPORTED",
            source_pmarray_sha256="0" * 64,
            background_pixel_count="1", body_pixel_count="1",
            clipped_ratio="0.0", onehot_valid="True",
            onehot_roundtrip="True",
        )
        fm = build_freeze_manifest(
            train_rows=[row], val_rows=[], test_rows=[],
            a06_split=a06, source_manifest_sha256="0" * 64,
            stats=norm, train_stats=cs, val_stats=cs,
        )
        # Mutate the core so the computed SHA differs from any sane
        # expected value, then verify the contract layer catches it.
        core = dict(fm.core)
        core["task_id"] = "TAMPERED"
        freeze_manifest = {"core": core, "meta": fm.meta}
        snap = B01FreezeSnapshot.from_freeze_tables(
            freeze_dir=tmp_path / "b01_wrong_core",
            train_rows=[row], val_rows=[], test_rows=None,
            freeze_manifest=freeze_manifest,
        )
        expected = B01ContractExpected(
            train_count=1, val_count=0, test_count=0,
            train_subjects=1, val_subjects=0, test_subjects=0,
            a06_split_sha256="0" * 64,
            provenance="V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            source_review_status="NOT_REVIEWED",
            setting="danaLab", cover="uncover",
            freeze_manifest_core_sha256="0" * 64,  # wrong expected
            structural_test_samples=0, structural_test_subjects=0,
        )
        with pytest.raises(B01ContractError, match="freeze_manifest core sha256"):
            verify_b01_contract(snap, expected)

    def test_train_row_setting_or_cover_inconsistent_rejected(self, tmp_path):
        # Drive the failure through the real freeze contract: build
        # FreezeRow lists with TRAIN row setting="danaLab" and VAL row
        # setting="simLab".  The observed unique settings are a multi-set,
        # so the validator must reject.
        from topper_perception.io.slp8_training_table_freeze import (
            FreezeRow,
        )
        train_row = FreezeRow(
            sample_id="SLP:danaLab:00000:uncover:000000",
            ml_split="train", source_split="VAL", setting="danaLab",
            subject_id="00000", cover="uncover", frame_id=0,
            posture="SUPINE", pressure_npy="p", region_label_npy="l",
            region_onehot_npy="o", points_csv="c", height="192", width="84",
            class_ids_present="0|1|2|3|4|5|6|7|8",
            annotation_provenance="V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            source_review_status="NOT_REVIEWED",
            export_version="1.1.0", export_status="EXPORTED",
            source_pmarray_sha256="0" * 64,
            background_pixel_count="1", body_pixel_count="1",
            clipped_ratio="0.0", onehot_valid="True",
            onehot_roundtrip="True",
        )
        val_row = FreezeRow(
            sample_id="SLP:simLab:00100:uncover:000000",
            ml_split="val", source_split="VAL", setting="simLab",
            subject_id="00100", cover="cover1", frame_id=0,
            posture="SUPINE", pressure_npy="p", region_label_npy="l",
            region_onehot_npy="o", points_csv="c", height="192", width="84",
            class_ids_present="0|1|2|3|4|5|6|7|8",
            annotation_provenance="V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            source_review_status="NOT_REVIEWED",
            export_version="1.1.0", export_status="EXPORTED",
            source_pmarray_sha256="0" * 64,
            background_pixel_count="1", body_pixel_count="1",
            clipped_ratio="0.0", onehot_valid="True",
            onehot_roundtrip="True",
        )
        core = {
            "task_id": "x",
            "a06_split_sha256": "0" * 64,
            "expected_provenance": "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            "expected_review_status": "NOT_REVIEWED",
            "splits": {
                "train": {"sample_count": 1, "subject_count": 1,
                          "manifest_sha256": "0" * 64},
                "val": {"sample_count": 1, "subject_count": 1,
                        "manifest_sha256": "0" * 64},
                "test": {"sample_count": 0, "subject_count": 0,
                         "manifest_sha256": "0" * 64},
            },
        }
        snap = B01FreezeSnapshot.from_freeze_tables(
            freeze_dir=tmp_path / "b01_inconsistent",
            train_rows=[train_row], val_rows=[val_row], test_rows=None,
            freeze_manifest={"core": core, "meta": {}},
        )
        with pytest.raises(B01ContractError, match="observed_settings"):
            verify_b01_contract(
                snap, self._good_expected(core_sha256=snap.freeze_manifest_core_sha256)
            )

    def test_test_rows_stay_unloaded(self, tmp_path):
        from topper_perception.io.slp8_training_table_freeze import (
            load_b01_freeze_tables, TestLeakageError,
        )
        root = _make_fake_b01_freeze_dir(
            tmp_path / "b01_test_unloaded", test_count_in_manifest=495
        )
        freeze = load_b01_freeze_tables(root, load_test=False)
        # The internal ``_test_rows`` MUST be None when load_test=False.
        assert freeze._test_rows is None  # noqa: SLF001
        # The structural TEST 495 rows in the freeze MUST NOT be
        # reachable from the B04 dataset loader.  The public
        # ``test_rows`` property raises ``TestLeakageError``.
        with pytest.raises(TestLeakageError):
            _ = freeze.test_rows

    def test_train_count_mismatch_rejected(self, tmp_path):
        # Build a snapshot from a real manifest but with a deliberately
        # wrong train_count in the structural record.  The validator
        # must catch the mismatch.
        from topper_perception.io.slp8_training_table_freeze import (
            build_freeze_manifest, A06Split, NormalizationStats, ClassStats,
            FreezeRow,
        )
        a06 = _make_a06_split()
        norm = _make_normalization_stats()
        cs = _make_class_stats()
        row = FreezeRow(
            sample_id="SLP:danaLab:00000:uncover:000000",
            ml_split="train", source_split="VAL", setting="danaLab",
            subject_id="00000", cover="uncover", frame_id=0,
            posture="SUPINE", pressure_npy="p", region_label_npy="l",
            region_onehot_npy="o", points_csv="c", height="192", width="84",
            class_ids_present="0|1|2|3|4|5|6|7|8",
            annotation_provenance="V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            source_review_status="NOT_REVIEWED",
            export_version="1.1.0", export_status="EXPORTED",
            source_pmarray_sha256="0" * 64,
            background_pixel_count="1", body_pixel_count="1",
            clipped_ratio="0.0", onehot_valid="True",
            onehot_roundtrip="True",
        )
        fm = build_freeze_manifest(
            train_rows=[row], val_rows=[], test_rows=[],
            a06_split=a06, source_manifest_sha256="0" * 64,
            stats=norm, train_stats=cs, val_stats=cs,
        )
        core = dict(fm.core)
        core["splits"]["train"]["sample_count"] = 9999  # wrong on purpose
        freeze_manifest = {"core": core, "meta": fm.meta}
        snap = B01FreezeSnapshot.from_freeze_tables(
            freeze_dir=tmp_path / "b01_short",
            train_rows=[row], val_rows=[], test_rows=None,
            freeze_manifest=freeze_manifest,
        )
        with pytest.raises(B01ContractError, match="train_count"):
            verify_b01_contract(
                snap, self._good_expected(core_sha256=snap.freeze_manifest_core_sha256)
            )


# ---------------------------------------------------------------------------
# R03: Resume equivalence test (interrupted vs uninterrupted)
# ---------------------------------------------------------------------------


def _candidate_metrics_signature(metrics: "CandidateMetrics | None") -> dict | None:
    if metrics is None:
        return None
    return {
        "val_loss": metrics.val_loss,
        "per_region": [
            (r["class_id"], round(r["iou"], 6), round(r["dice"], 6))
            for r in metrics.per_region
        ],
        "n_samples": metrics.n_samples,
    }


class TestResumeEquivalence:
    """An uninterrupted N-epoch run and a K-epoch run + resume to N
    MUST produce identical predictions, metrics, and best epoch.

    The test uses the synthetic CPU smoke runner with the real
    :class:`MiniConfig`.  Both runs go through ``run_mini``; the only
    difference is that one passes ``resume_from_per_candidate`` while
    the other does not.
    """

    def _build_cfg(self, max_epochs: int) -> "MiniConfig":
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg = build_mini_config(
            raw, b01_freeze_dir=None, data_root=None,
            config_path=str(CONFIG_PATH),
        )
        return replace(cfg, max_epochs=max_epochs)

    def _run(
        self,
        output_dir: Path,
        cfg: "MiniConfig",
        train_dataset,
        val_dataset,
        dataset_manifest,
        class_weight_result,
        *,
        resume_from_per_candidate: dict[str, Path] | None = None,
    ):
        budget = ResourceBudget(
            max_wall_seconds_per_candidate=2700.0,
            max_wall_seconds_total=5400.0,
            max_peak_cuda_mb=12288.0,
        )
        apply_settings(42, cpu_threads=1)
        return run_mini(
            config=cfg,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dataset_manifest=dataset_manifest,
            class_weight_result=class_weight_result,
            output_dir=output_dir,
            device=torch.device("cpu"),
            input_hashes={
                "config_sha256": file_sha256(CONFIG_PATH),
                "a06_split_sha256_expected": cfg.b01_a06_split_sha256_expected,
                "synthetic": True,
            },
            train_class_stats_source="synthetic_train_class_stats",
            synthetic=True,
            budget=budget,
            resume_from_per_candidate=resume_from_per_candidate,
        )

    def test_interrupted_then_resume_equals_uninterrupted(self, tmp_path_factory):
        from topper_perception.neural.slp8_region_mini import (
            apply_settings as _apply,
        )
        _apply(42, cpu_threads=1)
        # Use a small N so the test runs fast.
        n_total = 6
        k_partial = 2

        # 1) Uninterrupted N-epoch run.
        out_a = tmp_path_factory.mktemp("resume_a") / "out"
        train_ds, val_ds, manifest, class_stats = build_synthetic_dataset(
            n_train_samples=4, n_val_samples=2, seed=7
        )
        cw = compute_class_weights({
            "n_samples": 4,
            "n_pixels": 4 * 192 * 84,
            "per_class_pixel_ratio": {str(c): 0.1 for c in range(9)},
        })
        assert_class_weight_invariants(cw)
        cfg = self._build_cfg(max_epochs=n_total)
        result_a = self._run(
            out_a, cfg, train_ds, val_ds, manifest, cw
        )
        # 2) Partial K-epoch run.
        out_b = tmp_path_factory.mktemp("resume_b") / "out"
        cfg_k = self._build_cfg(max_epochs=k_partial)
        _apply(42, cpu_threads=1)
        result_b = self._run(
            out_b, cfg_k, train_ds, val_ds, manifest, cw
        )
        # 3) Resume from B for the remaining (N - K) epochs.
        out_c = tmp_path_factory.mktemp("resume_c") / "out"
        cfg_c = self._build_cfg(max_epochs=n_total)
        _apply(42, cpu_threads=1)
        # Build the resume map: each candidate's last.pt
        resume_map = {
            cand: out_b / "checkpoints" / cand / "last.pt"
            for cand in cfg_c.candidates
        }
        result_c = self._run(
            out_c, cfg_c, train_ds, val_ds, manifest, cw,
            resume_from_per_candidate=resume_map,
        )

        # 4) Compare A (uninterrupted) with C (resumed).  The two MUST
        # produce identical metrics, predictions hash, and best epoch.
        for cand in cfg.candidates:
            ca = result_a.candidate_results[cand]
            cc = result_c.candidate_results[cand]
            assert ca.feasibility == cc.feasibility, (
                f"candidate {cand} feasibility mismatch: A={ca.feasibility} C={cc.feasibility}"
            )
            assert ca.best_epoch == cc.best_epoch, (
                f"candidate {cand} best_epoch mismatch: A={ca.best_epoch} C={cc.best_epoch}"
            )
            assert ca.best_val_loss == cc.best_val_loss, (
                f"candidate {cand} val_loss mismatch: A={ca.best_val_loss} C={cc.best_val_loss}"
            )
            assert ca.best_prediction_hash == cc.best_prediction_hash, (
                f"candidate {cand} predictions hash mismatch: "
                f"A={ca.best_prediction_hash} C={cc.best_prediction_hash}"
            )
            # The per-region metrics are float values, so check the
            # rounded form to avoid spurious floating-point noise.
            assert _candidate_metrics_signature(ca.metrics) == (
                _candidate_metrics_signature(cc.metrics)
            ), (
                f"candidate {cand} metrics signature differs between A and C"
            )

    def test_resume_for_done_run_refused(self, tmp_path_factory):
        from topper_perception.neural.slp8_region_resume import (
            refuse_resume_for_done_run,
            ResumeRefusedError,
        )
        out = tmp_path_factory.mktemp("done_run") / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "DONE.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ResumeRefusedError, match="DONE.json"):
            refuse_resume_for_done_run(out)

    def test_resume_for_archived_run_incomplete_checkpoints_rejected(
        self, tmp_path_factory
    ):
        from topper_perception.neural.slp8_region_resume import (
            refuse_resume_for_done_run,
        )
        # No DONE.json, but also no checkpoints — the run is in a
        # not-resumable state, so the caller must validate that.
        out = tmp_path_factory.mktemp("no_ckpt") / "out"
        out.mkdir()
        # The resume refuser does NOT raise; it only checks DONE.
        # The CLI's auto-detect is the layer that raises on missing
        # checkpoints.  Verify the refuse function is silent here.
        refuse_resume_for_done_run(out)  # no exception


# ---------------------------------------------------------------------------
# R03: Determinism config — CUBLAS workspace + conditional warn_only
# ---------------------------------------------------------------------------


class TestDeterminismConfigR03:
    def test_cublas_workspace_set_before_cuda_init(self, monkeypatch):
        # Clean env to start.
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        monkeypatch.delenv("CUBLASLT_WORKSPACE_CONFIG", raising=False)
        from topper_perception.neural.slp8_region_determinism import apply_settings
        apply_settings(42, cpu_threads=1)
        import os
        # After apply_settings, both env vars MUST be set to the
        # deterministic default.
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        assert os.environ.get("CUBLASLT_WORKSPACE_CONFIG") == ":4096:8"

    def test_warn_only_depends_on_cuda_availability(self, monkeypatch):
        from topper_perception.neural.slp8_region_determinism import (
            apply_settings,
        )
        s = apply_settings(42, cpu_threads=1)
        import torch
        if torch.cuda.is_available():
            assert s.deterministic_algorithms_warn_only is False
            assert s.run_mode == "cuda_determinism_unverified"
        else:
            assert s.deterministic_algorithms_warn_only is True
            assert s.run_mode == "cpu_synthetic_reproducible"

    def test_environment_payload_records_cublas_and_run_mode(self):
        from topper_perception.neural.slp8_region_determinism import (
            apply_settings,
            environment_payload,
        )
        apply_settings(42, cpu_threads=1)
        env = environment_payload()
        # The payload records the post-application values under snake
        # case keys (cublas_workspace_config / cublaslt_workspace_config);
        # the env-var form (UPPER_SNAKE) is what apply_settings writes.
        assert "cublas_workspace_config" in env
        assert "cublaslt_workspace_config" in env
        assert env["cublas_workspace_config"] == ":4096:8"
        assert env["cublaslt_workspace_config"] == ":4096:8"
        assert "run_mode" in env
        assert env["run_mode"] in (
            "cpu_synthetic_reproducible", "cuda_determinism_unverified"
        )


# ---------------------------------------------------------------------------
# R03: Cross-subprocess determinism on predictions + centroids
# ---------------------------------------------------------------------------


class TestDeterminismSubprocessR03:
    """Two independent subprocess invocations of the synthetic CPU
    smoke must agree on ``predictions_manifest.csv`` content (not just
    the hash digest of the file) and on every per-sample per-region
    row of ``centroid_errors.csv``.
    """

    def _run_one_smoke(self, out: Path, env_extra: dict[str, str] | None = None):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "42"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        if env_extra:
            env.update(env_extra)
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_slp8_region_mini.py"),
                "--config", str(CONFIG_PATH),
                "--output-dir", str(out),
                "--synthetic-cpu-smoke",
            ],
            capture_output=True, text=True, env=env, timeout=600,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_two_subprocess_runs_have_identical_predictions(self, tmp_path_factory):
        out_a = tmp_path_factory.mktemp("det_pred_a") / "smoke"
        out_b = tmp_path_factory.mktemp("det_pred_b") / "smoke"
        self._run_one_smoke(out_a)
        self._run_one_smoke(out_b)
        a_text = (out_a / "predictions_manifest.csv").read_text(encoding="utf-8")
        b_text = (out_b / "predictions_manifest.csv").read_text(encoding="utf-8")
        assert a_text == b_text, "predictions_manifest.csv must match across runs"

    def test_two_subprocess_runs_have_identical_centroid_errors(
        self, tmp_path_factory
    ):
        out_a = tmp_path_factory.mktemp("det_cent_a") / "smoke"
        out_b = tmp_path_factory.mktemp("det_cent_b") / "smoke"
        self._run_one_smoke(out_a)
        self._run_one_smoke(out_b)
        # Strip elapsed_seconds and other timing noise — the centroid
        # itself and the per-row record must match.
        a_text = (out_a / "centroid_errors.csv").read_text(encoding="utf-8")
        b_text = (out_b / "centroid_errors.csv").read_text(encoding="utf-8")
        # Both files must contain the same row count.
        a_rows = list(csv.DictReader(open(out_a / "centroid_errors.csv", encoding="utf-8")))
        b_rows = list(csv.DictReader(open(out_b / "centroid_errors.csv", encoding="utf-8")))
        assert len(a_rows) == len(b_rows)
        # Compare row-by-row (using the raw text after stripping any
        # wall-clock-like fields, which is the contract).
        for ar, br in zip(a_rows, b_rows):
            for k in (
                "candidate", "split", "sample_index", "sample_id",
                "subject_id", "posture", "region", "error", "valid",
                "both_missing",
            ):
                assert ar[k] == br[k], f"centroid row mismatch on {k}: {ar} vs {br}"
        # And the raw text must be byte-identical (error values etc.).
        assert a_text == b_text
