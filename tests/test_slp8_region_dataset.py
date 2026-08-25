"""Tests for SLP8 Region Dataset (TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1).

Tests cover:
1. Dataset tensor shape/dtype
2. Label range
3. Lazy per-sample loading
4. TRAIN/VAL subject isolation
5. Deterministic subject subset
6. TEST default deny
7. Normalization application
"""

from __future__ import annotations

import tempfile
import json
from pathlib import Path
from typing import Any
import os

import numpy as np
import pytest
import torch

from topper_perception.neural.slp8_region_dataset import (
    BACKGROUND_ID,
    FOREGROUND_IDS,
    N_CLASSES,
    NormalizationStats,
    PRESSURE_SHAPE,
    REGION_NAMES,
    REGION_ID_TO_NAME,
    RegionSample,
    Slp8RegionDataset,
    SubjectOverlapError,
    Slp8TestDataAccessError,
    build_dataloader,
    build_smoke_dataset,
    collate_fn,
    select_smoke_subjects,
    verify_label_range,
    verify_subject_isolation,
)

from topper_perception.neural.slp8_region_models import INPUT_SHAPE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_freeze_rows() -> list[dict[str, Any]]:
    """Create mock freeze rows for testing."""
    rows = []

    # TRAIN subjects: 00001, 00002
    for subj in ["00001", "00002"]:
        for frame in range(45):
            rows.append({
                "sample_id": f"SLP:danaLab:{subj}:uncover:{frame:06d}",
                "ml_split": "train",
                "subject_id": subj,
                "posture": "SUPINE",
                "pressure_npy": f"SLP/danaLab/{subj}/uncover/pressure_{frame:06d}.npy",
                "region_label_npy": f"SLP/danaLab/{subj}/uncover/region_label_{frame:06d}.npy",
                "region_onehot_npy": f"SLP/danaLab/{subj}/uncover/region_onehot_{frame:06d}.npy",
            })

    # VAL subjects: 00003
    for frame in range(45):
        rows.append({
            "sample_id": f"SLP:danaLab:00003:uncover:{frame:06d}",
            "ml_split": "val",
            "subject_id": "00003",
            "posture": "LEFT",
            "pressure_npy": f"SLP/danaLab/00003/uncover/pressure_{frame:06d}.npy",
            "region_label_npy": f"SLP/danaLab/00003/uncover/region_label_{frame:06d}.npy",
            "region_onehot_npy": f"SLP/danaLab/00003/uncover/region_onehot_{frame:06d}.npy",
        })

    return rows


@pytest.fixture
def temp_dataset_dir(mock_freeze_rows) -> Path:
    """Create a temporary dataset directory with mock pressure/label files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir)

        for row in mock_freeze_rows:
            # Create pressure file (192, 84) float64
            pressure_dir = dataset_root / row["pressure_npy"]
            pressure_dir.parent.mkdir(parents=True, exist_ok=True)

            # Create dummy pressure array with some finite values
            pressure = np.random.rand(192, 84).astype(np.float64) * 100
            np.save(pressure_dir, pressure, allow_pickle=False)

            # Create label file (192, 84) int64 with values in [0, 8]
            label_dir = dataset_root / row["region_label_npy"]
            label_dir.parent.mkdir(parents=True, exist_ok=True)

            # Create label with some regions
            label = np.zeros((192, 84), dtype=np.int64)
            # Add some foreground regions
            label[20:50, 10:40] = 1  # HEAD_NECK
            label[50:100, 10:74] = 2  # SHOULDER
            label[100:140, 20:64] = 3  # THORAX_BACK
            np.save(label_dir, label, allow_pickle=False)

        yield dataset_root


@pytest.fixture
def temp_b01_freeze_dir(temp_dataset_dir, mock_freeze_rows) -> Path:
    """Create a temporary B01 freeze directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        freeze_dir = Path(tmpdir)

        # Create normalization_stats.json
        norm_stats = {
            "method": "raw_passthrough_with_minmax_reference",
            "raw_semantics": "raw_pmarray_response",
            "fit_split": "train",
            "global_min": 0.0,
            "global_max": 100.0,
            "global_mean": 50.0,
            "global_std": 25.0,
            "epsilon": 1e-12,
            "stats_sha256": "abc123",
        }
        (freeze_dir / "normalization_stats.json").write_text(
            json.dumps(norm_stats), encoding="utf-8"
        )

        # Create freeze_manifest.json
        freeze_manifest = {
            "core": {
                "freeze_version": "slp8_training_tables_v0.1",
                "a06_split_sha256": "024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706",
            }
        }
        (freeze_dir / "freeze_manifest.json").write_text(
            json.dumps(freeze_manifest), encoding="utf-8"
        )

        # Create train/val manifests
        train_rows = [r for r in mock_freeze_rows if r["ml_split"] == "train"]
        val_rows = [r for r in mock_freeze_rows if r["ml_split"] == "val"]

        train_manifest = freeze_dir / "train_manifest.csv"
        val_manifest = freeze_dir / "val_manifest.csv"

        cols = ["sample_id", "ml_split", "subject_id", "posture", "pressure_npy",
                "region_label_npy", "region_onehot_npy"]

        train_manifest.write_text(
            "\n".join([",".join(r[c] for c in cols) for r in train_rows]),
            encoding="utf-8",
        )
        val_manifest.write_text(
            "\n".join([",".join(r[c] for c in cols) for r in val_rows]),
            encoding="utf-8",
        )

        yield freeze_dir


# ---------------------------------------------------------------------------
# Test: Constants and configurations
# ---------------------------------------------------------------------------


class TestConstants:
    """Test module constants."""

    def test_n_classes(self):
        assert N_CLASSES == 9

    def test_background_id(self):
        assert BACKGROUND_ID == 0

    def test_foreground_ids(self):
        assert FOREGROUND_IDS == (1, 2, 3, 4, 5, 6, 7, 8)

    def test_pressure_shape(self):
        assert PRESSURE_SHAPE == (192, 84)

    def test_region_names(self):
        assert len(REGION_NAMES) == 9
        assert REGION_NAMES[0] == "BACKGROUND"
        assert REGION_NAMES[1] == "HEAD_NECK"

    def test_region_id_to_name(self):
        assert REGION_ID_TO_NAME[0] == "BACKGROUND"
        assert REGION_ID_TO_NAME[1] == "HEAD_NECK"


# ---------------------------------------------------------------------------
# Test: NormalizationStats
# ---------------------------------------------------------------------------


class TestNormalizationStats:
    """Test NormalizationStats."""

    def test_from_b01_stats(self):
        stats_dict = {
            "method": "raw_passthrough_with_minmax_reference",
            "raw_semantics": "raw_pmarray_response",
            "fit_split": "train",
            "global_min": 0.0,
            "global_max": 100.0,
            "global_mean": 50.0,
            "global_std": 25.0,
            "epsilon": 1e-12,
        }
        stats = NormalizationStats.from_b01_stats(stats_dict)

        assert stats.method == "raw_passthrough_with_minmax_reference"
        assert stats.fit_split == "train"
        assert stats.global_min == 0.0
        assert stats.global_max == 100.0

    def test_apply_normalization(self):
        stats = NormalizationStats(
            global_min=0.0,
            global_max=100.0,
            global_mean=50.0,
            global_std=25.0,
            method="raw_passthrough_with_minmax_reference",
            raw_semantics="raw_pmarray_response",
            fit_split="train",
            epsilon=1e-12,
        )

        # Raw pressure (192, 84) float64
        pressure = np.full((192, 84), 50.0, dtype=np.float64)

        # Normalized output (1, 192, 84) float32
        normalized = stats.apply(pressure)

        assert normalized.shape == (1, 192, 84)
        assert normalized.dtype == np.float32
        assert np.allclose(normalized, 0.5)

    def test_apply_normalization_edge_cases(self):
        stats = NormalizationStats(
            global_min=0.0,
            global_max=100.0,
            global_mean=50.0,
            global_std=25.0,
            method="raw_passthrough_with_minmax_reference",
            raw_semantics="raw_pmarray_response",
            fit_split="train",
            epsilon=1e-12,
        )

        # Test min value
        pressure_min = np.zeros((192, 84), dtype=np.float64)
        normalized_min = stats.apply(pressure_min)
        assert np.allclose(normalized_min, 0.0)

        # Test max value
        pressure_max = np.full((192, 84), 100.0, dtype=np.float64)
        normalized_max = stats.apply(pressure_max)
        assert np.allclose(normalized_max, 1.0)


# ---------------------------------------------------------------------------
# Test: RegionSample
# ---------------------------------------------------------------------------


class TestRegionSample:
    """Test RegionSample dataclass."""

    def test_create_region_sample(self):
        sample = RegionSample(
            sample_id="SLP:danaLab:00001:uncover:000001",
            subject_id="00001",
            ml_split="train",
            posture="SUPINE",
            pressure_path="SLP/danaLab/00001/uncover/pressure_000001.npy",
            label_path="SLP/danaLab/00001/uncover/region_label_000001.npy",
            onehot_path="SLP/danaLab/00001/uncover/region_onehot_000001.npy",
        )

        assert sample.sample_id == "SLP:danaLab:00001:uncover:000001"
        assert sample.subject_id == "00001"
        assert sample.ml_split == "train"
        assert sample.posture == "SUPINE"


# ---------------------------------------------------------------------------
# Test: Subject selection
# ---------------------------------------------------------------------------


class TestSelectSmokeSubjects:
    """Test deterministic smoke subject selection."""

    def test_select_smoke_subjects_no_overlap(self, mock_freeze_rows):
        # Create FreezeRow-like objects
        class MockRow:
            def __init__(self, d):
                self.__dict__.update(d)

        rows = [MockRow(r) for r in mock_freeze_rows]

        selected_train, selected_val = select_smoke_subjects(
            rows,
            seed=42,
            n_train_subjects=2,
            n_val_subjects=1,
        )

        # Check no overlap
        assert len(set(selected_train) & set(selected_val)) == 0
        assert len(selected_train) == 2
        assert len(selected_val) == 1

    def test_select_smoke_subjects_deterministic(self, mock_freeze_rows):
        class MockRow:
            def __init__(self, d):
                self.__dict__.update(d)

        rows = [MockRow(r) for r in mock_freeze_rows]

        # Run twice with same seed
        train1, val1 = select_smoke_subjects(rows, seed=42, n_train_subjects=2, n_val_subjects=1)
        train2, val2 = select_smoke_subjects(rows, seed=42, n_train_subjects=2, n_val_subjects=1)

        assert train1 == train2
        assert val1 == val2

    def test_select_smoke_subjects_same_seed_same_result(self, mock_freeze_rows):
        """Same seed always produces same result."""
        class MockRow:
            def __init__(self, d):
                self.__dict__.update(d)

        rows = [MockRow(r) for r in mock_freeze_rows]

        # Run twice with same seed
        train1, val1 = select_smoke_subjects(rows, seed=42, n_train_subjects=2, n_val_subjects=1)
        train2, val2 = select_smoke_subjects(rows, seed=42, n_train_subjects=2, n_val_subjects=1)

        # Should be identical
        assert train1 == train2
        assert val1 == val2


# ---------------------------------------------------------------------------
# Test: Subject isolation
# ---------------------------------------------------------------------------


class TestSubjectIsolation:
    """Test TRAIN/VAL subject isolation verification."""

    def test_verify_subject_isolation_no_overlap(self):
        train_subjects = ["00001", "00002"]
        val_subjects = ["00003"]

        assert verify_subject_isolation(train_subjects, val_subjects) is True

    def test_verify_subject_isolation_with_overlap(self):
        train_subjects = ["00001", "00002", "00003"]
        val_subjects = ["00003", "00004"]

        assert verify_subject_isolation(train_subjects, val_subjects) is False


# ---------------------------------------------------------------------------
# Test: Label verification
# ---------------------------------------------------------------------------


class TestLabelVerification:
    """Test label range verification."""

    def test_verify_label_range_valid(self):
        labels = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        assert verify_label_range(labels, n_classes=9) is True

    def test_verify_label_range_with_invalid(self):
        labels = np.array([[0, 1, 9], [3, 4, 5], [6, 7, 8]])  # 9 is invalid
        assert verify_label_range(labels, n_classes=9) is False

    def test_verify_label_range_negative(self):
        labels = np.array([[-1, 1, 2], [3, 4, 5], [6, 7, 8]])  # -1 is invalid
        assert verify_label_range(labels, n_classes=9) is False


# ---------------------------------------------------------------------------
# Test: Collate function
# ---------------------------------------------------------------------------


class TestCollateFn:
    """Test collate function for region segmentation."""

    def test_collate_fn(self):
        batch = [
            {
                "pressure": torch.randn(1, 192, 84),
                "label": torch.zeros(192, 84, dtype=torch.long),
                "sample_id": "sample1",
                "subject_id": "00001",
                "ml_split": "train",
                "posture": "SUPINE",
            },
            {
                "pressure": torch.randn(1, 192, 84),
                "label": torch.ones(192, 84, dtype=torch.long),
                "sample_id": "sample2",
                "subject_id": "00002",
                "ml_split": "train",
                "posture": "LEFT",
            },
        ]

        collated = collate_fn(batch)

        assert collated["pressure"].shape == (2, 1, 192, 84)
        assert collated["label"].shape == (2, 192, 84)
        assert collated["sample_id"] == ["sample1", "sample2"]
        assert collated["subject_id"] == ["00001", "00002"]


# ---------------------------------------------------------------------------
# Test: Dataset construction with real files
# ---------------------------------------------------------------------------


class TestSlp8RegionDatasetWithFiles:
    """Test Slp8RegionDataset with actual temp files."""

    def test_dataset_construction(self, temp_dataset_dir, mock_freeze_rows):
        # Create samples
        samples = []
        for row in mock_freeze_rows[:5]:
            samples.append(RegionSample(
                sample_id=row["sample_id"],
                subject_id=row["subject_id"],
                ml_split=row["ml_split"],
                posture=row["posture"],
                pressure_path=row["pressure_npy"],
                label_path=row["region_label_npy"],
                onehot_path=row["region_onehot_npy"],
            ))

        # Create normalization stats
        norm_stats = NormalizationStats(
            global_min=0.0,
            global_max=100.0,
            global_mean=50.0,
            global_std=25.0,
            method="raw_passthrough_with_minmax_reference",
            raw_semantics="raw_pmarray_response",
            fit_split="train",
            epsilon=1e-12,
        )

        # Create dataset
        dataset = Slp8RegionDataset(
            samples=samples,
            dataset_root=temp_dataset_dir,
            normalization=norm_stats,
        )

        assert len(dataset) == 5

        # Get first item
        item = dataset[0]

        assert "pressure" in item
        assert "label" in item
        assert item["pressure"].shape == (1, 192, 84)
        assert item["label"].shape == (192, 84)
        assert item["pressure"].dtype == torch.float32
        assert item["label"].dtype == torch.int64

    def test_dataset_empty_samples_error(self, temp_dataset_dir):
        norm_stats = NormalizationStats(
            global_min=0.0,
            global_max=100.0,
            global_mean=50.0,
            global_std=25.0,
            method="raw_passthrough_with_minmax_reference",
            raw_semantics="raw_pmarray_response",
            fit_split="train",
            epsilon=1e-12,
        )

        with pytest.raises(ValueError, match="samples must be non-empty"):
            Slp8RegionDataset(
                samples=[],
                dataset_root=temp_dataset_dir,
                normalization=norm_stats,
            )

    def test_dataset_missing_file_error(self, temp_dataset_dir):
        samples = [
            RegionSample(
                sample_id="missing",
                subject_id="00001",
                ml_split="train",
                posture="SUPINE",
                pressure_path="missing/file.npy",
                label_path="missing/label.npy",
                onehot_path="missing/onehot.npy",
            )
        ]

        norm_stats = NormalizationStats(
            global_min=0.0,
            global_max=100.0,
            global_mean=50.0,
            global_std=25.0,
            method="raw_passthrough_with_minmax_reference",
            raw_semantics="raw_pmarray_response",
            fit_split="train",
            epsilon=1e-12,
        )

        with pytest.raises(FileNotFoundError):
            Slp8RegionDataset(
                samples=samples,
                dataset_root=temp_dataset_dir,
                normalization=norm_stats,
            )


# ---------------------------------------------------------------------------
# Test: DataLoader
# ---------------------------------------------------------------------------


class TestBuildDataloader:
    """Test DataLoader builder."""

    def test_build_dataloader(self, temp_dataset_dir, mock_freeze_rows):
        samples = []
        for row in mock_freeze_rows[:10]:
            samples.append(RegionSample(
                sample_id=row["sample_id"],
                subject_id=row["subject_id"],
                ml_split=row["ml_split"],
                posture=row["posture"],
                pressure_path=row["pressure_npy"],
                label_path=row["region_label_npy"],
                onehot_path=row["region_onehot_npy"],
            ))

        norm_stats = NormalizationStats(
            global_min=0.0,
            global_max=100.0,
            global_mean=50.0,
            global_std=25.0,
            method="raw_passthrough_with_minmax_reference",
            raw_semantics="raw_pmarray_response",
            fit_split="train",
            epsilon=1e-12,
        )

        dataset = Slp8RegionDataset(
            samples=samples,
            dataset_root=temp_dataset_dir,
            normalization=norm_stats,
        )

        dataloader = build_dataloader(
            dataset,
            batch_size=4,
            shuffle=False,
        )

        assert dataloader.batch_size == 4
        assert dataloader.num_workers == 0

        # Get one batch
        batch = next(iter(dataloader))
        assert batch["pressure"].shape == (4, 1, 192, 84)
        assert batch["label"].shape == (4, 192, 84)
