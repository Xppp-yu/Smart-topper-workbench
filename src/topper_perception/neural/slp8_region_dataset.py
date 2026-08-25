"""SLP8 Region Segmentation Dataset (TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1).

This module provides a PyTorch Dataset for SLP8 pressure-only region segmentation,
using the B01 frozen training tables as input.

Key contracts:
* Subject-level splits: loaded from B01 freeze tables
* Pressure normalization: TRAIN-only (from B01 normalization_stats.json)
* TEST access: MUST use load_b01_freeze_tables(..., load_test=False)
* No augmentation in B03 smoke
* Lazy per-sample loading: pressure arrays are loaded on demand
* Deterministic subject subset selection for smoke
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from topper_perception.io.slp8_training_table_freeze import (
    FreezeRow,
    load_b01_freeze_tables,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: SLP8 pressure array shape.
PRESSURE_SHAPE = (192, 84)

#: Number of region classes (0=background, 1-8=foreground regions).
N_CLASSES = 9

#: Background class ID.
BACKGROUND_ID = 0

#: Foreground region IDs.
FOREGROUND_IDS = tuple(range(1, 9))

#: Region names in display order (matches label values).
REGION_NAMES = (
    "BACKGROUND",
    "HEAD_NECK",
    "SHOULDER",
    "THORAX_BACK",
    "LUMBAR_WAIST",
    "PELVIS_HIP",
    "ARM",
    "THIGH",
    "LOWER_LEG_FOOT",
)

#: Region ID to name mapping.
REGION_ID_TO_NAME = {i: name for i, name in enumerate(REGION_NAMES)}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class Slp8RegionDatasetError(Exception):
    """Base exception for SLP8 region dataset errors."""
    pass


class SubjectOverlapError(Slp8RegionDatasetError):
    """Raised when TRAIN and VAL subject sets overlap."""


class Slp8TestDataAccessError(Slp8RegionDatasetError):
    """Raised when TEST data is accidentally accessed."""


class NormalizationError(Slp8RegionDatasetError):
    """Raised when normalization stats are invalid."""


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    """TRAIN-only normalization statistics from B01 freeze."""

    global_min: float
    global_max: float
    global_mean: float
    global_std: float
    method: str  # e.g. "raw_passthrough_with_minmax_reference"
    raw_semantics: str  # e.g. "raw_pmarray_response"
    fit_split: str  # e.g. "train"
    epsilon: float  # e.g. 1e-12

    @classmethod
    def from_b01_stats(cls, stats: dict[str, Any]) -> "NormalizationStats":
        """Create NormalizationStats from B01 normalization_stats.json dict."""
        return cls(
            global_min=float(stats["global_min"]),
            global_max=float(stats["global_max"]),
            global_mean=float(stats["global_mean"]),
            global_std=float(stats["global_std"]),
            method=str(stats["method"]),
            raw_semantics=str(stats.get("raw_semantics", "raw_pmarray_response")),
            fit_split=str(stats["fit_split"]),
            epsilon=float(stats.get("epsilon", 1e-12)),
        )

    def apply(self, pressure: np.ndarray) -> np.ndarray:
        """Apply normalization to raw pressure array.

        B03 uses raw_passthrough_with_minmax_reference:
        output = (input - min) / (max - min + epsilon)

        Parameters
        ----------
        pressure : np.ndarray
            Raw pressure array of shape (192, 84), dtype float64.

        Returns
        -------
        np.ndarray
            Normalized pressure of shape (1, 192, 84), dtype float32.
        """
        if self.method != "raw_passthrough_with_minmax_reference":
            raise NormalizationError(
                f"Unsupported normalization method: {self.method!r}. "
                f"Expected 'raw_passthrough_with_minmax_reference'."
            )

        # Compute range with epsilon to avoid division by zero
        pmin = float(self.global_min)
        pmax = float(self.global_max)
        epsilon = self.epsilon
        pmin_val = min(pmin, pmax)  # Handle case where min == max
        pmax_val = max(pmin, pmax)
        value_range = max(pmax_val - pmin_val, epsilon)

        # Normalize: (x - min) / (max - min + epsilon)
        normalized = (pressure - pmin_val) / value_range

        # Convert to float32 and add channel dimension
        normalized = normalized.astype(np.float32)

        # Add channel dimension: (192, 84) -> (1, 192, 84)
        normalized = np.expand_dims(normalized, axis=0)

        return normalized


# ---------------------------------------------------------------------------
# Sample metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegionSample:
    """Metadata for one SLP8 region segmentation sample."""

    sample_id: str
    subject_id: str
    ml_split: str  # "train" or "val"
    posture: str  # "SUPINE", "LEFT", or "RIGHT"
    pressure_path: str  # Relative path from dataset root
    label_path: str  # Relative path from dataset root
    onehot_path: str  # Relative path from dataset root


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Slp8RegionDataset(Dataset):
    """PyTorch Dataset for SLP8 pressure-only region segmentation.

    This dataset implements lazy per-sample loading:
    * Pressure arrays are loaded on demand during __getitem__
    * Labels are loaded on demand during __getitem__
    * Normalization is applied per-sample

    Parameters
    ----------
    samples : Sequence[RegionSample]
        Sequence of sample metadata records.
    dataset_root : Path
        Root directory of the SLP8 dataset.
    normalization : NormalizationStats
        TRAIN-only normalization statistics.
    """

    def __init__(
        self,
        samples: Sequence[RegionSample],
        dataset_root: Path,
        normalization: NormalizationStats,
    ) -> None:
        if not samples:
            raise ValueError("samples must be non-empty")

        self._samples = list(samples)
        self._dataset_root = Path(dataset_root)
        self._normalization = normalization

        # Pre-validate paths exist (but don't load data yet)
        for sample in self._samples[:3]:  # Check first 3 samples
            pressure_path = self._dataset_root / sample.pressure_path
            if not pressure_path.exists():
                raise FileNotFoundError(
                    f"Pressure file not found: {pressure_path}"
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index]

        # Load pressure: (192, 84), float64
        pressure_path = self._dataset_root / sample.pressure_path
        pressure = np.load(pressure_path, mmap_mode=None, allow_pickle=False)
        if pressure.shape != PRESSURE_SHAPE:
            raise ValueError(
                f"Pressure shape mismatch for {sample.sample_id}: "
                f"expected {PRESSURE_SHAPE}, got {pressure.shape}"
            )
        if not np.isfinite(pressure).all():
            raise ValueError(
                f"Non-finite pressure values in {sample.sample_id}"
            )

        # Apply normalization: (192, 84) float64 -> (1, 192, 84) float32
        pressure_input = self._normalization.apply(pressure)

        # Load label: (192, 84), int64
        label_path = self._dataset_root / sample.label_path
        label = np.load(label_path, mmap_mode=None, allow_pickle=False)
        if label.shape != PRESSURE_SHAPE:
            raise ValueError(
                f"Label shape mismatch for {sample.sample_id}: "
                f"expected {PRESSURE_SHAPE}, got {label.shape}"
            )
        if label.dtype != np.int64:
            label = label.astype(np.int64)
        # Validate label range
        if not ((label >= 0) & (label < N_CLASSES)).all():
            invalid_mask = ~((label >= 0) & (label < N_CLASSES))
            invalid_vals = label[invalid_mask]
            raise ValueError(
                f"Label values out of range [0, {N_CLASSES - 1}] "
                f"in {sample.sample_id}: {invalid_vals[:10].tolist()}"
            )

        return {
            "pressure": torch.from_numpy(pressure_input),
            "label": torch.from_numpy(label),
            "sample_id": sample.sample_id,
            "subject_id": sample.subject_id,
            "ml_split": sample.ml_split,
            "posture": sample.posture,
        }


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate batch with spatial labels (not just classification).

    Unlike PoPu collate_fn, SLP8 region segmentation has 2D labels.
    """
    # Stack pressure inputs: (B, 1, 192, 84)
    pressure = torch.stack([item["pressure"] for item in batch])

    # Stack labels: (B, 192, 84)
    label = torch.stack([item["label"] for item in batch])

    return {
        "pressure": pressure,
        "label": label,
        "sample_id": [item["sample_id"] for item in batch],
        "subject_id": [item["subject_id"] for item in batch],
        "ml_split": [item["ml_split"] for item in batch],
        "posture": [item["posture"] for item in batch],
    }


def build_dataloader(
    dataset: Slp8RegionDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader for SLP8 region segmentation.

    Parameters
    ----------
    dataset : Slp8RegionDataset
        The dataset to load from.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle (default False for deterministic smoke).
    drop_last : bool
        Whether to drop the last incomplete batch.

    Returns
    -------
    DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,  # Single-threaded for deterministic smoke
        collate_fn=collate_fn,
    )


# ---------------------------------------------------------------------------
# Dataset builder from B01 freeze
# ---------------------------------------------------------------------------


def select_smoke_subjects(
    rows: Sequence[FreezeRow],
    *,
    seed: int = 42,
    n_train_subjects: int = 2,
    n_val_subjects: int = 1,
) -> tuple[list[str], list[str]]:
    """Select deterministic subject subset for smoke test.

    Parameters
    ----------
    rows : Sequence[FreezeRow]
        All freeze rows from B01.
    seed : int
        Random seed for deterministic selection.
    n_train_subjects : int
        Number of TRAIN subjects to select.
    n_val_subjects : int
        Number of VAL subjects to select.

    Returns
    -------
    tuple[list[str], list[str]]
        (selected_train_subjects, selected_val_subjects)
    """
    import random

    rng = random.Random(seed)

    # Extract unique subjects by split
    train_subjects = sorted(
        set(r.subject_id for r in rows if r.ml_split == "train")
    )
    val_subjects = sorted(
        set(r.subject_id for r in rows if r.ml_split == "val")
    )

    # Check for overlap
    overlap = set(train_subjects) & set(val_subjects)
    if overlap:
        raise SubjectOverlapError(
            f"TRAIN/VAL subject overlap detected: {sorted(overlap)}"
        )

    # Deterministic shuffle and select
    rng.shuffle(train_subjects)
    rng.shuffle(val_subjects)

    selected_train = train_subjects[:n_train_subjects]
    selected_val = val_subjects[:n_val_subjects]

    return selected_train, selected_val


def build_smoke_dataset(
    b01_freeze_dir: Path,
    dataset_root: Path,
    *,
    seed: int = 42,
    n_train_subjects: int = 2,
    n_val_subjects: int = 1,
) -> tuple[Slp8RegionDataset, Slp8RegionDataset, dict[str, Any]]:
    """Build TRAIN and VAL datasets for SLP8 region segmentation smoke.

    Parameters
    ----------
    b01_freeze_dir : Path
        B01 freeze directory containing training tables.
    dataset_root : Path
        SLP8 dataset root directory.
    seed : int
        Random seed for deterministic subject selection.
    n_train_subjects : int
        Number of TRAIN subjects to include in smoke.
    n_val_subjects : int
        Number of VAL subjects to include in smoke.

    Returns
    -------
    tuple[Slp8RegionDataset, Slp8RegionDataset, dict[str, Any]]
        (train_dataset, val_dataset, manifest_dict)

    Raises
    ------
    Slp8TestDataAccessError
        If TEST data is accidentally accessed.
    """
    # Load B01 freeze tables (TEST access is blocked by default)
    freeze = load_b01_freeze_tables(b01_freeze_dir, load_test=False)

    if freeze.test_rows is not None:
        raise Slp8TestDataAccessError(
            "TEST rows should not be present when load_test=False"
        )

    # Load normalization stats
    norm_stats_path = b01_freeze_dir / "normalization_stats.json"
    if not norm_stats_path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {norm_stats_path}"
        )
    with open(norm_stats_path, encoding="utf-8") as f:
        norm_data = json.load(f)
    normalization = NormalizationStats.from_b01_stats(norm_data)

    if normalization.fit_split != "train":
        raise NormalizationError(
            f"Normalization must be fit on TRAIN split, "
            f"got fit_split={normalization.fit_split!r}"
        )

    # Select smoke subjects
    all_rows = list(freeze.train_rows) + list(freeze.val_rows)
    selected_train_subjects, selected_val_subjects = select_smoke_subjects(
        all_rows,
        seed=seed,
        n_train_subjects=n_train_subjects,
        n_val_subjects=n_val_subjects,
    )

    # Build sample lists
    train_samples: list[RegionSample] = []
    val_samples: list[RegionSample] = []

    for row in all_rows:
        sample = RegionSample(
            sample_id=row.sample_id,
            subject_id=row.subject_id,
            ml_split=row.ml_split,
            posture=row.posture,
            pressure_path=row.pressure_npy,
            label_path=row.region_label_npy,
            onehot_path=row.region_onehot_npy,
        )
        if row.ml_split == "train" and row.subject_id in selected_train_subjects:
            train_samples.append(sample)
        elif row.ml_split == "val" and row.subject_id in selected_val_subjects:
            val_samples.append(sample)

    # Create datasets
    train_dataset = Slp8RegionDataset(
        samples=train_samples,
        dataset_root=dataset_root,
        normalization=normalization,
    )
    val_dataset = Slp8RegionDataset(
        samples=val_samples,
        dataset_root=dataset_root,
        normalization=normalization,
    )

    # Build manifest
    manifest = {
        "train_subjects": sorted(selected_train_subjects),
        "val_subjects": sorted(selected_val_subjects),
        "n_train_samples": len(train_samples),
        "n_val_samples": len(val_samples),
        "n_test_samples": 0,  # TEST not loaded
        "seed": seed,
        "normalization_method": normalization.method,
        "normalization_fit_split": normalization.fit_split,
        "normalization_stats_sha256": norm_data.get("stats_sha256", "unknown"),
    }

    return train_dataset, val_dataset, manifest


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def verify_subject_isolation(
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
) -> bool:
    """Verify TRAIN and VAL subjects are completely disjoint.

    Returns
    -------
    bool
        True if subjects are isolated, False if overlap exists.
    """
    train_set = set(train_subjects)
    val_set = set(val_subjects)
    return len(train_set & val_set) == 0


def verify_label_range(labels: np.ndarray, n_classes: int = N_CLASSES) -> bool:
    """Verify all label values are within valid range.

    Parameters
    ----------
    labels : np.ndarray
        Label array to verify.
    n_classes : int
        Number of classes (including background).

    Returns
    -------
    bool
        True if all labels are in [0, n_classes).
    """
    return bool(((labels >= 0) & (labels < n_classes)).all())
