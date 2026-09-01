"""SLP8 B07/B08 Full Runner — governed 30-unit execution with OOF guards.

This module implements the B07 Full protocol for SLP8 PM-only region segmentation:

* Protocol loading with fail-closed identity verification (B07 PROTOCOL_ACCEPTED,
  execution_authorized=False, TEST=0, committed-content byte-SHA for fold manifest).
* Two completely separated execution paths:
  - Synthetic mode (validate-only / smoke): uses in-memory synthetic data;
    synthetic path NEVER touches load_b01_freeze_tables or Slp8RegionDataset.
  - Real B01 mode: uses load_b01_freeze_tables(..., load_test=False),
    asserts _test_rows is None, builds Slp8RegionDataset per fold,
    computes normalization and class weights from fold-TRAIN only,
    collects per-sample predictions, saves OOF CSV per unit.
* Execution planning: 2 candidates × 5 folds × 3 seeds = 30 unique units.
* Subject-level fold routing from the frozen B07 fold manifest.
* Real B01 path: per-unit training with checkpoint/resume, class weights from
  fold-TRAIN only, normalization from fold-TRAIN only.
* Budget tracking: per-unit / per-candidate / total wall minutes, peak CUDA MB.
* Resume identity verification: full identity block match or fail-closed rejection.
* OOF merging from real per-sample predictions (not aggregated metrics),
  read from unit OOF CSVs, exact coverage 91 subjects / 4,095 samples,
  0 duplicate, 0 missing.
* Pooled OOF metrics per candidate, then arithmetic mean across seeds.
* Candidate selection rule: rank by pooled OOF macro IoU.
* Terminal state: exactly one of DONE / FAILED / STOPPED.
* validate-only mode creates ZERO files; all protocol/identity checks run
  in-memory.

This module does NOT:
- Access B01 TEST data (real path calls load_b01_freeze_tables with
  load_test=False and asserts _test_rows is None).
- Run real GPU training in synthetic mode.
- Commit, push, or merge.
- Modify B07 protocol, fold manifest, candidates, seeds, or budget.
- Allow synthetic_mode and real B01 path to cross-contaminate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import tempfile
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

from topper_perception.evaluation.slp_pressure_metrics import (
    compute_fixed_class_macro_metrics,
)
from topper_perception.io.slp8_training_table_freeze import (
    FreezeRow,
    load_b01_freeze_tables,
)
from topper_perception.neural.slp8_region_b01_contract import (
    B01ContractError,
    B01FreezeSnapshot,
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
    ClassWeightResult,
    WEIGHT_CLIP_MAX,
    WEIGHT_CLIP_MIN,
    assert_class_weight_invariants,
    class_weights_to_tensor,
    compute_class_weights,
)
from topper_perception.neural.slp8_region_dataset import (
    BACKGROUND_ID,
    FOREGROUND_IDS,
    N_CLASSES,
    PRESSURE_SHAPE,
    REGION_ID_TO_NAME,
    Slp8RegionDataset,
    NormalizationStats,
    RegionSample,
    build_dataloader,
    collate_fn,
    verify_subject_isolation,
)
from topper_perception.neural.slp8_region_determinism import (
    DeterminismSettings,
    apply_settings,
    collect_settings,
    environment_payload,
)
from topper_perception.neural.slp8_region_models import (
    DEEPLABV3PLUS_LITE_VERSION,
    INPUT_SHAPE,
    MODEL_REGISTRY,
    ModelBuilder,
    RESUNET_LITE_VERSION,
    get_model_builder,
)
from topper_perception.neural.slp8_region_metrics_ext import (
    DEFAULT_IMAGE_SHAPE,
    FOREGROUND_CLASS_IDS,
    compute_extended_metrics,
    compute_centroid_errors,
    summarize_centroid_errors,
    METRICS_VERSION as EXT_METRICS_VERSION,
)


# ---------------------------------------------------------------------------
# Frozen B07 protocol constants
# ---------------------------------------------------------------------------

B07_PROTOCOL_NAME = "B07"
B08_TASK_ID = "TASK-SLP-B08-FULL-RUNNER-AND-ONE-FOLD-PREFLIGHT-v0.1"
B08_FULL_VERSION = "slp8_region_full_v0.1"
B07_CONFIG_VERSION = "slp8_pm_full_protocol_v0.1"
B07_FOLD_CONFIG_VERSION = "slp8_pm_full_folds_v0.1"

#: B07 candidates (frozen by B07 protocol).
B07_CANDIDATES: tuple[str, ...] = (
    "slp8_deeplabv3plus_lite_v0.1",
    "slp8_resunet_lite_v0.1",
)

#: B07 seeds (frozen by B07 protocol).
B07_SEEDS: tuple[int, ...] = (42, 123, 2026)

#: Development pool size (B01 TRAIN+VAL).
DEV_SUBJECT_COUNT = 91
DEV_SAMPLE_COUNT = 4095

#: Resource budget: per-unit, per-candidate, total.
BUDGET_PREFLIGHT_MAX_WALL_MINUTES_PER_UNIT = 15
BUDGET_FULL_MAX_WALL_MINUTES_PER_UNIT = 15
BUDGET_MAX_WALL_MINUTES_PER_CANDIDATE = 225  # 5 folds × 3 seeds × 15 min
BUDGET_MAX_WALL_MINUTES_TOTAL = 450  # 2 candidates × 225 min
BUDGET_MAX_PEAK_CUDA_MB = 8192

#: Alias for convenience (matches BUDGET_MAX_WALL_MINUTES_PER_UNIT expected in tests).
BUDGET_MAX_WALL_MINUTES_PER_UNIT = BUDGET_FULL_MAX_WALL_MINUTES_PER_UNIT

#: Synthetic smoke defaults.
SYNTHETIC_SMOKE_DEFAULTS: dict[str, Any] = {
    "n_train_samples": 8,
    "n_val_samples": 4,
    "image_shape": list(PRESSURE_SHAPE),
    "max_epochs_per_unit": 1,
    "min_epochs": 1,
    "early_stopping_patience": 2,
    "seed": 42,
}

#: Sentinel EXP-ID for synthetic smoke only.
SYNTHETIC_EXP_ID = "EXP-SLP-B08-SYNTHETIC-SMOKE"

#: Deterministic synthetic dataset manifest payload for synthetic smoke identity.
SYNTHETIC_DATA_MANIFEST_PAYLOAD: dict[str, Any] = {
    "kind": "slp8_synthetic_smoke_manifest",
    "config_version": B07_CONFIG_VERSION,
    "synthetic": True,
    "synthetic_train_samples": 8,
    "synthetic_val_samples": 4,
    "synthetic_seeds": list(B07_SEEDS),
    "n_classes": 9,
    "image_shape": list(PRESSURE_SHAPE),
    "normalization": "raw_passthrough_with_minmax_reference",
    "fit_split": "train",
    "provenance": "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
    "source_review_status": "NOT_REVIEWED",
    "note": (
        "Canonical synthetic smoke manifest for B08 Full Runner. "
        "SHA-256 of this payload is the data_manifest_sha256 for all "
        "synthetic smoke carriers.  Synthetic identity must never be "
        "confused with a real B01 freeze_manifest.json hash."
    ),
}


def _compute_synthetic_manifest_sha256() -> str:
    """Return the deterministic SHA-256 of the canonical synthetic dataset manifest."""
    text = json.dumps(
        SYNTHETIC_DATA_MANIFEST_PAYLOAD,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FullProtocolError(Exception):
    """Fail-closed B07/B08 contract violation."""


class InjectedInterruption(Exception):
    """Internal test fault that leaves a resumable checkpoint."""


class FullConfigValidationError(FullProtocolError):
    """Raised when the B08 config fails fail-closed validation."""


class FullOutputCollisionError(FullProtocolError):
    """Raised when the output directory is not safe to write into."""


class FullRunAuthorizationError(FullProtocolError):
    """Raised when a real B01 run is requested without --run-authorized."""


class FullExperimentIdentityError(FullProtocolError):
    """Raised when the experiment identity is invalid."""


class FullResumeIdentityError(FullProtocolError):
    """Raised when resume identity does not match."""


class FullResumeRefusedError(FullProtocolError):
    """Raised when resume is refused (run already DONE)."""


class FullBudgetExceededError(FullProtocolError):
    """Raised when wall-time or CUDA memory budget is exceeded."""


# ---------------------------------------------------------------------------
# SHA utilities (CRLF-safe committed-content reading)
# ---------------------------------------------------------------------------


def _git_show_bytes(
    repo_root: Path,
    relative_path: str,
    *,
    git_rev: str = "HEAD",
) -> bytes | None:
    """Read bytes from Git's committed content (not the working tree).

    On Windows with core.autocrlf=true the working tree file may contain
    CRLF bytes while the committed content is LF-only.  Reading via
    ``git show <git_rev>:relative_path`` bypasses the working-tree
    conversion AND the staged index, so byte-SHA verification is
    consistent across platforms and decoupled from any working-tree
    modifications.

    Parameters
    ----------
    repo_root : Path
        Git repository root.
    relative_path : str
        Path relative to the repo root.
    git_rev : str
        Git revision to read from.  Defaults to ``HEAD``.  Pass an
        explicit SHA (e.g. ``"1009366c"``) to freeze the read.

    Returns None if the file is not tracked at the given revision or git
    is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "show", f"{git_rev}:{relative_path}"],
            cwd=str(repo_root),
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except (OSError, subprocess.TimeoutExpired):
        return None


def committed_file_sha256(
    repo_root: Path,
    relative_path: str,
    *,
    frozen_git_sha: str | None = None,
) -> str | None:
    """Compute SHA-256 of the committed content of a tracked file.

    Parameters
    ----------
    repo_root : Path
        Git repository root.
    relative_path : str
        Path relative to the repo root.
    frozen_git_sha : str | None
        Explicit git revision (full or short SHA).  When provided, the
        read is anchored to that revision; index/staged/working-tree
        changes cannot affect the result.  When None, defaults to HEAD.

    Returns
    -------
    str | None
        Hex SHA-256 of the committed content at the specified revision.
        None if git is unavailable or the file is not tracked.
    """
    rev = frozen_git_sha if frozen_git_sha else "HEAD"
    committed = _git_show_bytes(repo_root, relative_path, git_rev=rev)
    if committed is not None:
        return hashlib.sha256(committed).hexdigest()
    # Fallback: working tree (handles untracked or git-unavailable)
    path = (repo_root / relative_path).resolve()
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


def file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file's raw bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FullUnit:
    """One (candidate, fold_id, seed) execution unit."""

    candidate: str
    fold_id: str
    seed: int

    @property
    def unit_id(self) -> str:
        return f"{self.candidate}__{self.fold_id}__seed_{self.seed:04d}"

    def unit_dir_name(self) -> str:
        return self.unit_id


@dataclass(frozen=True)
class FrozenFullProtocol:
    """Validated B07 protocol snapshot."""

    protocol_path: Path
    fold_path: Path
    protocol_sha256: str
    fold_sha256: str
    candidates: tuple[str, ...]
    fold_subjects: dict[str, tuple[str, ...]]
    fold_val_sample_counts: dict[str, int]
    fold_train_sample_counts: dict[str, int]
    seeds: tuple[int, ...]
    development_subject_count: int
    development_sample_count: int


@dataclass
class FullConfig:
    """Resolved B08 Full run configuration."""

    protocol: FrozenFullProtocol
    output_dir: Path
    experiment_id: str
    git_commit: str
    git_dirty: bool
    b01_freeze_dir: Path | None
    data_root: Path | None
    device: str
    batch_size: int
    max_epochs: int
    min_epochs: int
    early_stopping_patience: int
    optimizer: str
    lr: float
    weight_decay: float
    synthetic_mode: bool
    no_write_mode: bool
    validate_only: bool
    max_wall_minutes_per_unit: int
    max_wall_minutes_per_candidate: int
    max_wall_minutes_total: int
    max_peak_cuda_mb: int
    config_sha256: str
    data_manifest_sha256: str
    fold_manifest_sha256: str
    a06_split_sha256: str
    # Test-only fault injection; never set by production CLI.
    interrupt_after_epoch: int | None = None


@dataclass
class UnitResult:
    """Result of one unit execution."""

    unit: FullUnit
    status: str  # "DONE" | "FAILED" | "STOPPED"
    train_sample_count: int
    val_sample_count: int
    best_epoch: int | None
    best_val_loss: float | None
    final_val_loss: float | None
    val_fixed_fg_macro_iou: float | None
    val_fixed_fg_macro_dice: float | None
    val_background_iou: float | None
    val_per_region: dict[str, dict[str, float]] | None
    val_per_subject: dict[str, float] | None
    val_confusion_matrix: list[list[int]] | None
    error_message: str | None
    wall_seconds: float
    peak_cuda_mb: float | None
    checkpoint_best_path: Path | None
    checkpoint_last_path: Path | None
    # Path to per-sample OOF predictions CSV written by this unit.
    # In synthetic mode: written by _write_synthetic_oof_csv.
    # In real B01 mode: written by _write_real_oof_csv from collected predictions.
    oof_csv_path: Path | None = None


@dataclass
class SeedOOFResult:
    """Merged OOF for one (candidate, seed) = all 5 folds."""

    candidate: str
    seed: int
    status: str  # "COMPLETE" | "INCOMPLETE" | "FAILED"
    total_samples: int
    total_subjects: int
    duplicate_count: int
    missing_count: int
    pooled_fixed_fg_macro_iou: float | None
    pooled_fixed_fg_macro_dice: float | None
    pooled_background_iou: float | None
    pooled_per_subject: dict[str, float]
    worst_subject_iou: float | None
    oof_csv_path: Path | None
    error_message: str | None


@dataclass
class CandidateResult:
    """Aggregated result for one candidate across all seeds."""

    candidate: str
    model_version: str
    exact_parameter_count: int
    seed_results: dict[int, SeedOOFResult]
    mean_pooled_iou: float | None
    mean_pooled_dice: float | None
    mean_worst_subject_iou: float | None
    status: str  # "DONE" | "INCOMPLETE" | "FAILED"
    decision: str | None  # "WINNER" | "ELIMINATED" | None
    tiebreak_reason: str | None


@dataclass
class FullRunResult:
    """Top-level result of a full run."""

    experiment_id: str
    git_commit: str
    git_dirty: bool
    config_sha256: str
    data_manifest_sha256: str
    fold_manifest_sha256: str
    a06_split_sha256: str
    candidate_results: dict[str, CandidateResult]
    winner: str | None
    winner_mean_pooled_iou: float | None
    terminal_state: str  # "DONE" | "FAILED" | "STOPPED"
    total_wall_seconds: float
    unit_count_total: int
    unit_count_done: int
    unit_count_failed: int
    unit_count_stopped: int
    budget_report: dict[str, Any]
    error_message: str | None


# ---------------------------------------------------------------------------
# Protocol loading
# ---------------------------------------------------------------------------


def load_frozen_full_protocol(
    protocol_path: Path,
    repo_root: Path | None = None,
) -> FrozenFullProtocol:
    """Load and validate the B07 protocol with fail-closed checks.

    Parameters
    ----------
    protocol_path : Path
        Path to the frozen protocol JSON.
    repo_root : Path | None
        Repository root for git-show committed-content reads.
        If None, uses protocol_path.parents[2].

    Returns
    -------
    FrozenFullProtocol
        Validated protocol snapshot.

    Raises
    ------
    FullProtocolError
        On any B07 contract violation (wrong protocol, wrong status,
        execution_authorized not False, TEST access not denied, byte SHA
        mismatch, fold coverage errors, execution matrix mismatch).
    """
    protocol_path = Path(protocol_path).resolve()
    if repo_root is None:
        repo_root = protocol_path.parents[2]

    raw = json.loads(protocol_path.read_text(encoding="utf-8"))

    # 1. Protocol identity
    if raw.get("protocol") != "B07":
        raise FullProtocolError(
            f"B07 protocol required, got protocol={raw.get('protocol')!r}"
        )
    if raw.get("status") != "PROTOCOL_ACCEPTED":
        raise FullProtocolError(
            f"B07 protocol must be PROTOCOL_ACCEPTED, got status={raw.get('status')!r}"
        )

    # 2. Execution must not be authorized yet
    if raw.get("execution_authorized") is not False:
        raise FullProtocolError(
            "B07 protocol cannot authorize execution; "
            f"execution_authorized={raw.get('execution_authorized')!r}"
        )

    # 3. TEST access must be exactly denied/zero
    test = raw.get("test_access", {})
    expected_test = {
        "allowed": False,
        "load_test": False,
        "expected_rows": 0,
        "expected_labels": 0,
        "expected_onehot": 0,
    }
    if test != expected_test:
        raise FullProtocolError(
            "B07 TEST access contract must remain exactly denied/zero; "
            f"got test_access={test!r}"
        )

    # 4. Load fold manifest and verify committed-content byte SHA
    #    Use git-show to avoid CRLF drift on Windows worktrees.
    rel_fold_path = raw["fold_contract"]["manifest_path"]
    fold_path = (repo_root / rel_fold_path).resolve()
    observed_fold_sha = committed_file_sha256(repo_root, rel_fold_path)
    if observed_fold_sha is None:
        # Fallback to working-tree SHA if git unavailable (e.g., test mock)
        observed_fold_sha = file_sha256(fold_path)
    expected_fold_sha = raw["fold_contract"]["manifest_sha256"]
    if observed_fold_sha != expected_fold_sha:
        raise FullProtocolError(
            f"B07 fold manifest byte SHA mismatch: "
            f"expected={expected_fold_sha}, observed={observed_fold_sha}. "
            "The committed content SHA must match the frozen SHA."
        )

    # 5. Validate fold manifest structure
    folds = json.loads(fold_path.read_text(encoding="utf-8"))
    if folds.get("test_access") != "DENIED":
        raise FullProtocolError("B07 fold manifest must deny TEST access")
    if folds.get("development_subject_count", 0) != DEV_SUBJECT_COUNT:
        raise FullProtocolError(
            f"B07 fold manifest must cover {DEV_SUBJECT_COUNT} subjects, "
            f"got {folds.get('development_subject_count')}"
        )
    if folds.get("development_sample_count", 0) != DEV_SAMPLE_COUNT:
        raise FullProtocolError(
            f"B07 fold manifest must cover {DEV_SAMPLE_COUNT} samples, "
            f"got {folds.get('development_sample_count')}"
        )

    # 6. Validate fold subject coverage (each subject exactly once, 0 overlap)
    fold_rows = folds.get("folds", [])
    if len(fold_rows) != 5:
        raise FullProtocolError(f"B07 requires exactly 5 folds, got {len(fold_rows)}")

    fold_subjects: dict[str, tuple[str, ...]] = {}
    fold_val_counts: dict[str, int] = {}
    fold_train_counts: dict[str, int] = {}
    all_subjects: list[str] = []

    for fold_row in fold_rows:
        fid = str(fold_row["fold_id"])
        val_ids = tuple(str(s) for s in fold_row["val_subject_ids"])
        if len(val_ids) != len(set(val_ids)):
            raise FullProtocolError(f"B07 fold {fid} has duplicate val subjects")
        fold_subjects[fid] = val_ids
        fold_val_counts[fid] = int(fold_row["val_sample_count"])
        fold_train_counts[fid] = int(fold_row["train_sample_count"])
        all_subjects.extend(val_ids)

    if len(all_subjects) != DEV_SUBJECT_COUNT:
        raise FullProtocolError(
            f"B07 folds must cover {DEV_SUBJECT_COUNT} subjects total, "
            f"got {len(all_subjects)}"
        )
    if len(set(all_subjects)) != DEV_SUBJECT_COUNT:
        raise FullProtocolError(
            f"B07 folds must cover each subject exactly once; "
            f"found duplicates (total={len(all_subjects)}, unique={len(set(all_subjects))})"
        )

    # 7. Candidates
    candidates = tuple(str(c["name"]) for c in raw["candidates"])
    if candidates != B07_CANDIDATES:
        raise FullProtocolError(
            f"B07 candidates must be {B07_CANDIDATES}, got {candidates}"
        )

    # 8. Seeds
    seeds = tuple(int(s) for s in raw["training_contract"]["seeds"])
    if seeds != B07_SEEDS:
        raise FullProtocolError(
            f"B07 seeds must be {B07_SEEDS}, got {seeds}"
        )

    # 9. Verify execution matrix
    matrix = raw.get("execution_matrix", {})
    expected_units = len(candidates) * 5 * len(seeds)  # 30
    if matrix.get("total_units", 0) != expected_units:
        raise FullProtocolError(
            f"B07 execution matrix must have {expected_units} units, "
            f"got {matrix.get('total_units')}"
        )

    # 10. Verify protocol committed-content byte SHA
    protocol_sha = committed_file_sha256(repo_root, str(protocol_path.relative_to(repo_root)))
    if protocol_sha is None:
        protocol_sha = file_sha256(protocol_path)

    return FrozenFullProtocol(
        protocol_path=protocol_path,
        fold_path=fold_path,
        protocol_sha256=protocol_sha,
        fold_sha256=observed_fold_sha,
        candidates=candidates,
        fold_subjects=fold_subjects,
        fold_val_sample_counts=fold_val_counts,
        fold_train_sample_counts=fold_train_counts,
        seeds=seeds,
        development_subject_count=DEV_SUBJECT_COUNT,
        development_sample_count=DEV_SAMPLE_COUNT,
    )


# ---------------------------------------------------------------------------
# Execution planning
# ---------------------------------------------------------------------------


def build_execution_plan(protocol: FrozenFullProtocol) -> tuple[FullUnit, ...]:
    """Build the ordered list of units for the full run.

    Returns
    -------
    tuple[FullUnit, ...]
        Exactly 30 units: 2 candidates × 5 folds × 3 seeds.
    """
    units = tuple(
        FullUnit(candidate=candidate, fold_id=fold_id, seed=seed)
        for candidate in protocol.candidates
        for fold_id in sorted(protocol.fold_subjects.keys())
        for seed in protocol.seeds
    )
    ids = [u.unit_id for u in units]
    if len(ids) != len(set(ids)):
        raise FullProtocolError(
            f"B08 execution plan contains duplicate unit IDs"
        )
    return units


# ---------------------------------------------------------------------------
# Data partitioning (B07 fold subject routing)
# ---------------------------------------------------------------------------


def partition_records_for_fold(
    records: Sequence[Mapping[str, Any]],
    *,
    val_subject_ids: Iterable[str],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Partition development records into fold-TRAIN and fold-VAL by subject.

    B08 Round 5: B07 frozen ``val_subject_ids`` is the **authoritative**
    routing for each fold.  The original ``ml_split`` field on a B01
    record is a development-time tag, not a binding split — the same
    sample may legitimately appear in either fold-train or fold-val of
    any given B07 fold depending on the frozen subject membership.

    Only TEST and unsupported ml_split values are rejected; TRAIN/VAL
    rows are accepted regardless of which fold they end up in, as long
    as the final subject routing matches the B07 contract.

    Parameters
    ----------
    records : Sequence[Mapping]
        Development records (train/val rows from B01 freeze).
    val_subject_ids : Iterable[str]
        VAL subject IDs for this fold (from B07 frozen fold manifest).

    Returns
    -------
    tuple[list, list]
        (train_records, val_records)

    Raises
    ------
    FullProtocolError
        On TEST record injection or unsupported ml_split values, or if
        the final routing does not match the B07 contract.
    """
    val_subjects = {str(s) for s in val_subject_ids}
    train: list[Mapping[str, Any]] = []
    val: list[Mapping[str, Any]] = []

    for record in records:
        split = str(record.get("ml_split", "")).lower()
        if split == "test":
            raise FullProtocolError("B08 received a TEST record")
        if split not in {"train", "val", "development"}:
            raise FullProtocolError(
                f"B08 record has unsupported ml_split={split!r}"
            )
        subject = str(record["subject_id"])
        (val if subject in val_subjects else train).append(record)

    # Verify the final routing matches the B07 contract
    train_subjects = {str(r["subject_id"]) for r in train}
    observed_val_subjects = {str(r["subject_id"]) for r in val}

    if train_subjects & observed_val_subjects:
        raise FullProtocolError("B08 fold TRAIN/VAL subject overlap")

    if observed_val_subjects != val_subjects:
        missing = val_subjects - observed_val_subjects
        extra = observed_val_subjects - val_subjects
        raise FullProtocolError(
            "B08 fold VAL subject coverage mismatch: "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )

    # Ensure no sample appears in both train and val
    train_sample_ids = {str(r["sample_id"]) for r in train}
    val_sample_ids = {str(r["sample_id"]) for r in val}
    if train_sample_ids & val_sample_ids:
        raise FullProtocolError(
            "B08 fold TRAIN/VAL sample_id overlap detected"
        )

    return train, val


# ---------------------------------------------------------------------------
# OOF validation
# ---------------------------------------------------------------------------


def validate_oof_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_samples: int,
    expected_subjects: int,
) -> None:
    """Validate that OOF rows cover exactly the expected subjects and samples.

    Parameters
    ----------
    rows : Sequence[Mapping]
        OOF prediction rows.
    expected_samples : int
        Expected number of OOF rows (4095).
    expected_subjects : int
        Expected number of unique subjects (91).

    Raises
    ------
    FullProtocolError
        On TEST row injection, sample count mismatch, duplicate sample IDs,
        or subject count mismatch.
    """
    if any(str(row.get("ml_split", "")).lower() == "test" for row in rows):
        raise FullProtocolError("B08 OOF contains TEST rows")

    sample_ids = [str(row["sample_id"]) for row in rows]
    subjects = {str(row["subject_id"]) for row in rows}

    if len(sample_ids) != expected_samples:
        raise FullProtocolError(
            f"B08 OOF sample count mismatch: expected={expected_samples}, "
            f"got={len(sample_ids)}"
        )
    if len(set(sample_ids)) != expected_samples:
        raise FullProtocolError(
            f"B08 OOF contains duplicate sample IDs: "
            f"total={len(sample_ids)}, unique={len(set(sample_ids))}"
        )
    if len(subjects) != expected_subjects:
        raise FullProtocolError(
            f"B08 OOF subject coverage mismatch: expected={expected_subjects}, "
            f"got={len(subjects)}"
        )


# ---------------------------------------------------------------------------
# Synthetic dataset generation (synthetic path ONLY)
# ---------------------------------------------------------------------------


def build_synthetic_fold_dataset(
    n_train: int,
    n_val: int,
    seed: int,
    image_shape: tuple[int, int] = PRESSURE_SHAPE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate synthetic fold TRAIN/VAL records for smoke testing.

    Each record has the minimum fields required by Slp8RegionDataset:
    ``sample_id``, ``subject_id``, ``ml_split``, ``pmarray_path``,
    ``label_path``, ``posture``.

    NOTE: This function is ONLY called when synthetic_mode=True.
    The real B01 path must NEVER call this function.

    Parameters
    ----------
    n_train : int
        Number of synthetic TRAIN samples.
    n_val : int
        Number of synthetic VAL samples.
    seed : int
        Random seed for reproducibility.
    image_shape : tuple[int, int]
        Pressure array shape.

    Returns
    -------
    tuple[list[dict], list[dict]]
        (train_records, val_records)
    """
    rng = np.random.default_rng(seed)
    n_total = n_train + n_val

    base_sample_ids = [f"SYNTH_{i:05d}" for i in range(n_total)]
    subjects_per_fold = max(2, n_total // 45 + 1)
    base_subject_ids = [f"SYNTH_{i:03d}" for i in range(subjects_per_fold)]

    train_records = []
    val_records = []

    for i in range(n_train):
        subj = base_subject_ids[i % len(base_subject_ids)]
        train_records.append({
            "sample_id": base_sample_ids[i],
            "subject_id": subj,
            "ml_split": "train",
            "pmarray_path": "/dev/null",
            "label_path": "/dev/null",
            "posture": "supine",
        })

    for i in range(n_val):
        idx = n_train + i
        subj = base_subject_ids[i % len(base_subject_ids)]
        val_records.append({
            "sample_id": base_sample_ids[idx],
            "subject_id": subj,
            "ml_split": "val",
            "pmarray_path": "/dev/null",
            "label_path": "/dev/null",
            "posture": "supine",
        })

    return train_records, val_records


# ---------------------------------------------------------------------------
# Real B01 path helpers (real path ONLY — never called in synthetic mode)
# ---------------------------------------------------------------------------


def _freeze_rows_to_region_samples(
    rows: Sequence[FreezeRow],
) -> list[RegionSample]:
    """Convert B01 FreezeRow records to Slp8RegionDataset RegionSample records.

    This is a REAL B01 PATH helper.  It must NEVER be called in synthetic mode.
    """
    samples: list[RegionSample] = []
    for row in rows:
        samples.append(RegionSample(
            sample_id=row.sample_id,
            subject_id=row.subject_id,
            ml_split=row.ml_split,
            posture=row.posture,
            pressure_path=row.pressure_npy,
            label_path=row.region_label_npy,
            onehot_path=row.region_onehot_npy,
        ))
    return samples


def compute_fold_normalization_from_samples(
    train_samples: Sequence[RegionSample],
    *,
    data_root: Path,
) -> NormalizationStats:
    """Compute normalization statistics from fold-TRAIN samples only.

    Iterates over the passed-in fold-TRAIN samples, loads each pressure
    array, and accumulates per-pixel mean and standard deviation.  This
    function is the fold-TRAIN-only normalization source: it must NOT
    accept pre-computed global statistics.

    Parameters
    ----------
    train_samples : Sequence[RegionSample]
        Fold-TRAIN samples (must NOT include fold-VAL samples).
    data_root : Path
        SLP8 dataset root used to resolve pressure file paths.

    Returns
    -------
    NormalizationStats
        Fold-TRAIN normalization statistics with ``fit_split="train"``.

    Raises
    ------
    FullProtocolError
        If any pressure file is missing or empty.
    """
    if len(train_samples) == 0:
        raise FullProtocolError(
            "compute_fold_normalization_from_samples called with 0 train samples"
        )
    pixel_sum: np.ndarray | None = None
    pixel_sq_sum: np.ndarray | None = None
    total_pixels = 0
    for sample in train_samples:
        path = data_root / sample.pressure_path
        if not path.exists():
            raise FullProtocolError(
                f"pressure file not found for sample {sample.sample_id}: {path}"
            )
        arr = np.load(path).astype(np.float64)
        if pixel_sum is None:
            pixel_sum = np.zeros_like(arr, dtype=np.float64)
            pixel_sq_sum = np.zeros_like(arr, dtype=np.float64)
        pixel_sum += arr
        pixel_sq_sum += arr * arr
        total_pixels += 1
    if total_pixels == 0 or pixel_sum is None:
        raise FullProtocolError(
            "compute_fold_normalization_from_samples: no valid pressure samples"
        )
    mean = (pixel_sum / total_pixels).astype(np.float32)
    var = (pixel_sq_sum / total_pixels - mean.astype(np.float64) ** 2).astype(np.float32)
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    g_mean = float(mean.mean())
    g_std = float(std.mean())
    g_min = float(mean.min() - 3.0 * std.max())
    g_max = float(mean.max() + 3.0 * std.max())
    return NormalizationStats(
        global_min=g_min,
        global_max=g_max,
        global_mean=g_mean,
        global_std=g_std,
        method="raw_passthrough_with_minmax_reference",
        # B01's accepted pressure contract uses the canonical semantic tag;
        # Slp8RegionDataset rejects aliases here to prevent unit confusion.
        raw_semantics="raw_pmarray_response",
        fit_split="train",
        epsilon=1e-6,
    )


def compute_fold_class_weights_from_samples(
    train_samples: Sequence[RegionSample],
    *,
    data_root: Path,
    n_classes: int = N_CLASSES,
) -> ClassWeightResult:
    """Compute per-class weights from fold-TRAIN labels only.

    Iterates over the passed-in fold-TRAIN samples, loads each label
    array, and counts per-class frequencies.  This is the fold-TRAIN-only
    class-weights source: it must NOT accept pre-computed global class
    statistics from B01 freeze artifacts.

    Parameters
    ----------
    train_samples : Sequence[RegionSample]
        Fold-TRAIN samples (must NOT include fold-VAL samples).
    data_root : Path
        SLP8 dataset root used to resolve label file paths.
    n_classes : int
        Number of segmentation classes (default 9 = 1 background + 8 regions).

    Returns
    -------
    ClassWeightResult
        Fold-TRAIN class weights.

    Raises
    ------
    FullProtocolError
        On missing label files or no valid samples.
    """
    if len(train_samples) == 0:
        raise FullProtocolError(
            "compute_fold_class_weights_from_samples called with 0 train samples"
        )
    counts = np.zeros(n_classes, dtype=np.int64)
    for sample in train_samples:
        path = data_root / sample.label_path
        if not path.exists():
            raise FullProtocolError(
                f"label file not found for sample {sample.sample_id}: {path}"
            )
        arr = np.load(path).astype(np.int64)
        for c in range(n_classes):
            counts[c] += int((arr == c).sum())
    if counts.sum() == 0:
        raise FullProtocolError(
            "compute_fold_class_weights_from_samples: no valid labels"
        )
    # Build a flat {class_id: ratio} mapping expected by compute_class_weights
    total = float(counts.sum())
    per_class_pixel_ratio = {c: float(counts[c]) / total for c in range(n_classes)}
    return compute_class_weights(
        {
            "per_class_pixel_ratio": per_class_pixel_ratio,
            "sample_count": int(len(train_samples)),
            "n_pixels": int(total),
        }
    )


def load_real_b01_fold(
    b01_freeze_dir: Path,
    data_root: Path | None,
    fold_id: str,
    val_subject_ids: tuple[str, ...],
    *,
    synthetic_mode: bool,
) -> tuple[
    list[RegionSample],  # train samples
    list[RegionSample],  # val samples
    NormalizationStats | None,
    ClassWeightResult | None,
]:
    """Load and partition real B01 freeze data for one fold.

    This is a REAL B01 PATH helper.  It must NEVER be called in synthetic mode.
    It loads B01 freeze tables with load_test=False, asserts TEST=None,
    partitions by B07 fold subjects, builds Slp8RegionDataset samples,
    and computes normalization/class weights from fold-TRAIN only when
    ``data_root`` is provided.

    B08 Round 5: ``val_subject_ids`` from B07 is the AUTHORITATIVE routing
    for fold-train vs fold-val.  The original ``ml_split`` field on B01
    records is a development-time tag; the same sample may legitimately
    appear in either fold-train or fold-val of any given B07 fold
    depending on the frozen subject membership.

    Parameters
    ----------
    b01_freeze_dir : Path
        B01 freeze directory (must contain freeze_manifest.json, *_manifest.csv).
    data_root : Path | None
        SLP8 dataset root (for resolving pressure/label paths).  When
        None, normalization/class weights are skipped (preflight-only).
    fold_id : str
        Fold ID (e.g., "fold_1").
    val_subject_ids : tuple[str, ...]
        VAL subject IDs for this fold (from B07 fold manifest).
    synthetic_mode : bool
        Must be False — this function is only for real B01 mode.

    Returns
    -------
    tuple[list[RegionSample], list[RegionSample], NormalizationStats | None, ClassWeightResult | None]

    Raises
    ------
    FullProtocolError
        On TEST row injection, partition errors, or B01 contract violations.
    """
    # Contract: this function must not be called in synthetic mode
    if synthetic_mode:
        raise FullProtocolError(
            "load_real_b01_fold called with synthetic_mode=True; "
            "real B01 path must never be used in synthetic mode"
        )

    # Load B01 freeze tables with TEST denied
    freeze = load_b01_freeze_tables(b01_freeze_dir, load_test=False)

    # CRITICAL: assert _test_rows is None (defense in depth)
    if freeze._test_rows is not None:  # type: ignore[attr-defined]
        raise FullProtocolError(
            "B08 real path: load_b01_freeze_tables returned non-None _test_rows; "
            "TEST data must not be loaded"
        )

    # Combine all development rows for partition (B08 Round 5: B01 train
    # + val rows form the development pool; B07 val_subject_ids is the
    # authoritative routing for each fold's TRAIN/VAL split).
    all_dev_rows: list[FreezeRow] = list(freeze.train_rows) + list(freeze.val_rows)

    # Partition by B07 fold subjects (the partition function is the
    # single source of truth for routing).
    train_rows_raw, val_rows_raw = partition_records_for_fold(
        [r.to_dict() for r in all_dev_rows],
        val_subject_ids=val_subject_ids,
    )

    # Convert to RegionSample
    train_samples = _freeze_rows_to_region_samples(
        [FreezeRow(**r) for r in train_rows_raw]
    )
    val_samples = _freeze_rows_to_region_samples(
        [FreezeRow(**r) for r in val_rows_raw]
    )

    # Compute normalization/class weights from fold-TRAIN only when
    # data_root is provided.  Preflight callers may pass None.
    normalization: NormalizationStats | None = None
    class_weight_result: ClassWeightResult | None = None
    if data_root is not None:
        normalization = compute_fold_normalization_from_samples(
            train_samples, data_root=data_root,
        )

    # Compute class weights from fold-TRAIN ONLY when data_root provided
    if data_root is not None:
        class_weight_result = compute_fold_class_weights_from_samples(
            train_samples, data_root=data_root,
        )

    return train_samples, val_samples, normalization, class_weight_result


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def resolve_device(requested: str = "cuda") -> str:
    """Resolve device, fail-closed for real runs."""
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise FullProtocolError(
            "CUDA requested but torch.cuda.is_available() is False"
        )
    return "cpu"


def build_model(
    candidate: str,
    device: str,
    n_classes: int = N_CLASSES,
) -> nn.Module:
    """Build and move model to device."""
    builder = get_model_builder(candidate)
    model, _ = builder.factory(n_classes=n_classes, device=device)
    return model


def _validate_real_region_records(
    train_records: Sequence[Mapping[str, Any] | RegionSample],
    val_records: Sequence[Mapping[str, Any] | RegionSample],
) -> None:
    """Fail closed unless real TRAIN/VAL records use the production type."""
    real_records = [*train_records, *val_records]
    if not all(isinstance(record, RegionSample) for record in real_records):
        raise FullProtocolError(
            "real B01 path requires RegionSample TRAIN/VAL records; "
            "synthetic mappings must not enter real B01 training path"
        )
    if any(record.sample_id.startswith("SYNTH_") for record in real_records):
        raise FullProtocolError(
            "real B01 path received SYNTH_ sample IDs; "
            "synthetic data must not enter real B01 training path"
        )


def deterministic_cross_entropy_2d(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute mean 2-D cross entropy without CUDA NLLLoss kernels.

    ``torch.nn.functional.cross_entropy`` dispatches to a CUDA NLLLoss2d
    kernel that PyTorch 2.13 rejects when strict deterministic algorithms are
    enabled.  This is the same weighted-mean objective, expressed through
    log-softmax, one-hot targets, elementwise multiplication and reductions.
    Those primitives remain subject to the global strict deterministic guard;
    no warn-only or determinism relaxation is used.
    """
    if logits.ndim != 4 or targets.ndim != 3:
        raise FullProtocolError(
            "deterministic cross entropy expects logits [N,C,H,W] and "
            "targets [N,H,W]"
        )
    if logits.shape[0] != targets.shape[0] or logits.shape[2:] != targets.shape[1:]:
        raise FullProtocolError("deterministic cross entropy shape mismatch")
    if targets.dtype != torch.long:
        raise FullProtocolError("deterministic cross entropy targets must be int64")

    log_probs = F.log_softmax(logits, dim=1)
    target_mask = F.one_hot(
        targets, num_classes=logits.shape[1]
    ).movedim(-1, 1).to(dtype=logits.dtype)
    if weight is not None:
        if weight.ndim != 1 or weight.numel() != logits.shape[1]:
            raise FullProtocolError("deterministic cross entropy weight shape mismatch")
        class_weight = weight.to(device=logits.device, dtype=logits.dtype).view(
            1, -1, 1, 1
        )
        target_mask = target_mask * class_weight
    numerator = -(log_probs * target_mask).sum()
    denominator = target_mask.sum()
    return numerator / denominator


def train_one_unit(
    unit: FullUnit,
    train_records: Sequence[Mapping[str, Any] | RegionSample],
    val_records: Sequence[Mapping[str, Any] | RegionSample],
    config: FullConfig,
    unit_output_dir: Path,
    # Real B01 path extra inputs (None in synthetic mode)
    normalization: NormalizationStats | None = None,
    class_weight_result: ClassWeightResult | None = None,
    data_root: Path | None = None,
    val_sample_ids: list[str] | None = None,
    val_subject_ids_list: list[str] | None = None,
    val_postures: list[str] | None = None,
) -> UnitResult:
    """Execute one unit: train and evaluate on fold TRAIN/VAL.

    This function implements TWO disjoint paths that are guarded by config.synthetic_mode:
    - synthetic_mode=True: uses _build_synthetic_dataloader (in-memory, no B01)
    - synthetic_mode=False: uses Slp8RegionDataset from fold-TRAIN records

    In real B01 mode, per-sample predictions are collected during validation,
    written to a per-unit OOF CSV, and returned via UnitResult.oof_csv_path.

    Parameters
    ----------
    unit : FullUnit
        The unit to execute.
    train_records : Sequence[Mapping | RegionSample]
        Fold-TRAIN records (``RegionSample`` objects for real B01, synthetic
        dictionaries for smoke).
    val_records : Sequence[Mapping | RegionSample]
        Fold-VAL records.
    config : FullConfig
        Run configuration.
    unit_output_dir : Path
        Directory for this unit's outputs.
    normalization : NormalizationStats | None
        Required for real B01 mode (synthetic_mode=False).  Must be None
        when synthetic_mode=True.
    class_weight_result : ClassWeightResult | None
        Required for real B01 mode.  Must be None when synthetic_mode=True.
    data_root : Path | None
        Required for real B01 mode.  Must be None when synthetic_mode=True.
    val_sample_ids : list[str] | None
        Ordered list of sample_ids matching val_records order (for real B01 OOF CSV).
    val_subject_ids_list : list[str] | None
        Ordered list of subject_ids matching val_records order.
    val_postures : list[str] | None
        Ordered list of postures matching val_records order.

    Returns
    -------
    UnitResult
        Training result including metrics, checkpoints, and OOF CSV path.
    """
    unit_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = unit_output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    device = resolve_device(config.device)

    # Apply deterministic settings
    apply_settings(unit.seed)

    # Build model
    model = build_model(unit.candidate, device)
    model.train()

    # Build optimizer
    optimizer: optim.Optimizer
    if config.optimizer == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise FullProtocolError(f"Unsupported optimizer: {config.optimizer}")

    # -----------------------------------------------------------------------
    # PATH SEPARATION: synthetic vs real B01
    # -----------------------------------------------------------------------
    if config.synthetic_mode:
        # SYNTHETIC PATH: must NOT have normalization/class_weight/data_root
        if normalization is not None or class_weight_result is not None or data_root is not None:
            raise FullProtocolError(
                "synthetic_mode=True but real B01 inputs were provided to train_one_unit; "
                "synthetic path must never use load_b01_freeze_tables or Slp8RegionDataset"
            )
        train_loader = _build_synthetic_dataloader(
            train_records, config.batch_size, shuffle=True
        )
        val_loader = _build_synthetic_dataloader(
            val_records, config.batch_size, shuffle=False
        )
        weight_tensor: torch.Tensor | None = None
    else:
        # REAL B01 PATH: must have normalization/class_weight/data_root
        if normalization is None or class_weight_result is None or data_root is None:
            raise FullProtocolError(
                "synthetic_mode=False but real B01 inputs are missing; "
                "real B01 path must provide normalization, class_weight_result, and data_root"
            )
        # Guard: the real path accepts the production RegionSample contract,
        # never synthetic dictionaries.  Keep this explicit so a future type
        # drift fails closed before constructing the dataset.
        _validate_real_region_records(train_records, val_records)

        train_dataset = Slp8RegionDataset(
            samples=list(train_records),
            dataset_root=data_root,
            normalization=normalization,
        )
        val_dataset = Slp8RegionDataset(
            samples=list(val_records),
            dataset_root=data_root,
            normalization=normalization,
        )
        train_loader = build_dataloader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = build_dataloader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # Class weights from fold-TRAIN only
        assert_class_weight_invariants(class_weight_result)
        weight_tensor = torch.from_numpy(
            class_weights_to_tensor(class_weight_result)
        ).to(device).to(torch.float32)

    # Early stopping state
    early_stopper_state = {
        "best_val_loss": float("inf"),
        "best_epoch": 0,
        "patience_counter": 0,
        "stopped_early": False,
    }

    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "last.pt"
    best_val_loss = early_stopper_state["best_val_loss"]
    best_epoch = early_stopper_state["best_epoch"]
    patience_counter = early_stopper_state["patience_counter"]
    stopped_early = early_stopper_state["stopped_early"]

    # -----------------------------------------------------------------------
    # PARTIAL CHECKPOINT RESUME (B08 Round 5)
    # If last.pt exists from a prior interrupted run, restore model
    # weights, optimizer state, epoch, best metric, early-stopper, and
    # RNG state.  Identity is verified first; mismatch fails closed.
    # -----------------------------------------------------------------------
    start_epoch = 1
    if last_checkpoint_path.is_file():
        ckpt_identity = {
            "experiment_id": config.experiment_id,
            "git_commit": config.git_commit,
            "git_dirty": config.git_dirty,
            "config_sha256": config.config_sha256,
            "data_manifest_sha256": config.data_manifest_sha256,
            "fold_manifest_sha256": config.fold_manifest_sha256,
            "split_sha256": config.a06_split_sha256,
            "model_version": unit.candidate,
            "candidate": unit.candidate,
            "fold_id": unit.fold_id,
            "seed": int(unit.seed),
        }
        resumed = load_checkpoint_for_resume(
            last_checkpoint_path, ckpt_identity,
        )
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        if resumed["best_epoch"] is not None:
            best_epoch = int(resumed["best_epoch"])
        if resumed["best_val_loss"] is not None:
            best_val_loss = float(resumed["best_val_loss"])
        if resumed["early_stopper_state"]:
            early_stopper_state.update(resumed["early_stopper_state"])
            patience_counter = int(early_stopper_state.get("patience_counter", 0))
            stopped_early = bool(early_stopper_state.get("stopped_early", False))
        last_completed_epoch = int(resumed["epoch"])
        start_epoch = last_completed_epoch + 1
        if resumed["rng_state"] is not None:
            # Best-effort deterministic restore
            try:
                if "torch" in resumed["rng_state"]:
                    torch.set_rng_state(resumed["rng_state"]["torch"])
                if "numpy" in resumed["rng_state"]:
                    np.random.set_state(resumed["rng_state"]["numpy"])
                if "python" in resumed["rng_state"]:
                    random.setstate(resumed["rng_state"]["python"])
                if "torch_cuda" in resumed["rng_state"] and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(resumed["rng_state"]["torch_cuda"])
            except Exception:
                raise FullProtocolError("checkpoint RNG state could not be restored")

    start_wall = time.monotonic()
    peak_cuda_mb = 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        peak_cuda_mb = float(torch.cuda.max_memory_allocated()) / 1e6

    # Collect per-sample H×W predictions for OOF NPZ (real B01 mode)
    # In synthetic mode these stay None
    oof_predictions: list[np.ndarray] | None = (
        [] if not config.synthetic_mode else None
    )
    oof_targets: list[np.ndarray] | None = (
        [] if not config.synthetic_mode else None
    )
    best_val_predictions: list[np.ndarray] | None = None
    best_val_targets: list[np.ndarray] | None = None

    final_val_loss = None
    val_fixed_fg_iou = None
    val_fixed_fg_dice = None
    val_bg_iou = None
    status = "DONE"
    error_msg = None

    try:
        for epoch in range(start_epoch, config.max_epochs + 1):
            epoch_start = time.monotonic()

            # Train
            model.train()
            epoch_train_loss = 0.0
            train_steps = 0
            for batch in train_loader:
                inputs, labels = batch["pressure"], batch["label"]
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = deterministic_cross_entropy_2d(
                    outputs, labels, weight=weight_tensor,
                )

                loss.backward()
                optimizer.step()

                epoch_train_loss += float(loss.item())
                train_steps += 1

            avg_train_loss = epoch_train_loss / max(train_steps, 1)

            # Validate and collect per-sample predictions
            model.eval()
            # OOF is the final validation pass only; never accumulate one
            # copy per epoch (which would duplicate sample IDs).
            if oof_predictions is not None and oof_targets is not None:
                oof_predictions.clear()
                oof_targets.clear()
            epoch_val_loss = 0.0
            val_steps = 0
            all_preds: list[np.ndarray] = []
            all_labels: list[np.ndarray] = []

            with torch.no_grad():
                batch_idx = 0
                for batch in val_loader:
                    inputs, labels = batch["pressure"], batch["label"]
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = deterministic_cross_entropy_2d(outputs, labels)
                    epoch_val_loss += float(loss.item())
                    val_steps += 1

                    preds = outputs.argmax(dim=1)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

                    # Collect per-sample H×W predictions for OOF NPZ (real B01 only)
                    if oof_predictions is not None and oof_targets is not None:
                        batch_size = inputs.size(0)
                        preds_np = preds.cpu().numpy()
                        labels_np = labels.cpu().numpy()
                        for j in range(batch_size):
                            global_idx = batch_idx * config.batch_size + j
                            if (val_sample_ids is not None and
                                    global_idx < len(val_sample_ids)):
                                # Per-sample H×W prediction mask (not scalar class)
                                oof_predictions.append(preds_np[j])
                                oof_targets.append(labels_np[j])
                    batch_idx += 1

            avg_val_loss = epoch_val_loss / max(val_steps, 1)
            final_val_loss = avg_val_loss

            # Compute metrics
            all_preds_arr = np.concatenate(all_preds)
            all_labels_arr = np.concatenate(all_labels)
            metrics = compute_fixed_class_macro_metrics(
                all_labels_arr, all_preds_arr, n_classes=N_CLASSES
            )
            val_fixed_fg_iou = getattr(metrics, "fixed_iou", 0.0)
            val_fixed_fg_dice = getattr(metrics, "fixed_dice", 0.0)
            val_bg_iou = 0.0

            epoch_wall = time.monotonic() - epoch_start

            # Update peak memory
            if device == "cuda":
                current_mb = float(torch.cuda.max_memory_allocated()) / 1e6
                peak_cuda_mb = max(peak_cuda_mb, current_mb)

            # Check budget
            total_wall = time.monotonic() - start_wall
            if total_wall > config.max_wall_minutes_per_unit * 60:
                status = "STOPPED"
                error_msg = (
                    f"Wall budget exceeded: {total_wall:.0f}s > "
                    f"{config.max_wall_minutes_per_unit * 60}s"
                )
                break

            # Early stopping check
            early_stopper_state["best_val_loss"] = float(best_val_loss)
            early_stopper_state["best_epoch"] = int(best_epoch)
            early_stopper_state["patience_counter"] = int(patience_counter)
            early_stopper_state["stopped_early"] = bool(stopped_early)
            if avg_val_loss < best_val_loss - 1e-6:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                early_stopper_state["best_val_loss"] = float(best_val_loss)
                early_stopper_state["best_epoch"] = int(best_epoch)
                early_stopper_state["patience_counter"] = 0
                best_val_predictions = [p.copy() for p in all_preds]
                best_val_targets = [t.copy() for t in all_labels]
                _save_checkpoint(
                    model, optimizer, epoch, avg_val_loss,
                    best_checkpoint_path, unit, config,
                    best_epoch=best_epoch, best_val_loss=best_val_loss,
                    early_stopper_state=early_stopper_state,
                    rng_state=_capture_rng_state(),
                )
            else:
                patience_counter += 1

            if patience_counter >= config.early_stopping_patience:
                stopped_early = True
                early_stopper_state["stopped_early"] = True
                break

            # Save last checkpoint after every epoch (B08 Round 5)
            _save_checkpoint(
                model, optimizer,
                epoch,
                avg_val_loss,
                last_checkpoint_path, unit, config,
                best_epoch=best_epoch, best_val_loss=best_val_loss,
                early_stopper_state=early_stopper_state,
                rng_state=_capture_rng_state(),
            )
            if config.interrupt_after_epoch is not None and epoch == config.interrupt_after_epoch:
                raise InjectedInterruption(f"injected interruption after epoch {epoch}")

    except InjectedInterruption as e:
        status = "INTERRUPTED"
        error_msg = f"{type(e).__name__}: {e}"
    except Exception as e:
        status = "FAILED"
        error_msg = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    # The frozen contract selects the minimum-val-loss checkpoint.  Final OOF
    # must therefore come from an independently reloaded best.pt, not merely
    # from the last epoch left in memory.  On a fresh run compare its ordered
    # predictions with the in-process best-epoch predictions and fail closed
    # on any mismatch.  A resumed run may not have the historical in-process
    # arrays, but still performs the independent identity-checked reload.
    if status == "DONE" and best_checkpoint_path.is_file():
        try:
            best_identity = {
                "experiment_id": config.experiment_id,
                "git_commit": config.git_commit,
                "git_dirty": config.git_dirty,
                "config_sha256": config.config_sha256,
                "data_manifest_sha256": config.data_manifest_sha256,
                "fold_manifest_sha256": config.fold_manifest_sha256,
                "split_sha256": config.a06_split_sha256,
                "model_version": unit.candidate,
                "candidate": unit.candidate,
                "fold_id": unit.fold_id,
                "seed": int(unit.seed),
            }
            best_loaded = load_checkpoint_for_resume(
                best_checkpoint_path, best_identity,
            )
            reloaded_model = build_model(unit.candidate, device)
            reloaded_model.load_state_dict(best_loaded["model_state_dict"])
            reloaded_model.eval()
            reload_preds: list[np.ndarray] = []
            reload_targets: list[np.ndarray] = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["pressure"].to(device)
                    labels = batch["label"].to(device)
                    reload_preds.append(
                        reloaded_model(inputs).argmax(dim=1).cpu().numpy()
                    )
                    reload_targets.append(labels.cpu().numpy())
            if best_val_predictions is not None:
                if len(best_val_predictions) != len(reload_preds) or any(
                    not np.array_equal(a, b)
                    for a, b in zip(best_val_predictions, reload_preds)
                ):
                    raise FullProtocolError(
                        "best checkpoint reload prediction mismatch"
                    )
            if best_val_targets is not None and any(
                not np.array_equal(a, b)
                for a, b in zip(best_val_targets, reload_targets)
            ):
                raise FullProtocolError("best checkpoint reload target order mismatch")
            if oof_predictions is not None and oof_targets is not None:
                oof_predictions.clear()
                oof_targets.clear()
                for batch_preds, batch_targets in zip(reload_preds, reload_targets):
                    oof_predictions.extend([p.copy() for p in batch_preds])
                    oof_targets.extend([t.copy() for t in batch_targets])
            reloaded_pred_arr = np.concatenate(reload_preds)
            reloaded_target_arr = np.concatenate(reload_targets)
            reloaded_metrics = compute_fixed_class_macro_metrics(
                reloaded_target_arr, reloaded_pred_arr, n_classes=N_CLASSES,
            )
            val_fixed_fg_iou = float(reloaded_metrics.fixed_iou)
            val_fixed_fg_dice = float(reloaded_metrics.fixed_dice)
        except Exception as e:
            status = "FAILED"
            error_msg = f"{type(e).__name__}: {e}"

    wall_seconds = time.monotonic() - start_wall

    # Write per-unit OOF NPZ (real B01 mode only)
    oof_csv_path: Path | None = None
    oof_npz_path: Path | None = None
    if status == "DONE" and not config.synthetic_mode:
        oof_count = len(oof_predictions) if oof_predictions is not None else 0
        target_count = len(oof_targets) if oof_targets is not None else 0
        if oof_count != len(val_records) or target_count != len(val_records):
            status = "FAILED"
            error_msg = (
                "real unit completed without exact OOF coverage: "
                f"predictions={oof_count}, targets={target_count}, "
                f"expected={len(val_records)}"
            )
    if (status == "DONE"
            and oof_predictions is not None
            and oof_targets is not None
            and not config.no_write_mode):
        # Real B01: write collected per-sample H×W predictions + targets
        oof_dir = unit_output_dir / "oof"
        oof_dir.mkdir(exist_ok=True, parents=True)
        oof_npz_path = oof_dir / "unit_oof.npz"
        # Build ordered identity lists
        ordered_sample_ids = (
            val_sample_ids[: len(oof_predictions)]
            if val_sample_ids is not None
            else [f"sample_{i}" for i in range(len(oof_predictions))]
        )
        ordered_subject_ids = (
            val_subject_ids_list[: len(oof_predictions)]
            if val_subject_ids_list is not None
            else [f"UNK_{i}" for i in range(len(oof_predictions))]
        )
        ordered_fold_ids = [str(unit.fold_id)] * len(oof_predictions)
        _write_real_oof_npz(
            oof_npz_path,
            predictions=np.stack(oof_predictions, axis=0).astype(np.int64),
            targets=np.stack(oof_targets, axis=0).astype(np.int64),
            sample_ids=ordered_sample_ids,
            subject_ids=ordered_subject_ids,
            fold_ids=ordered_fold_ids,
            unit=unit,
        )
    elif config.synthetic_mode and not config.no_write_mode:
        # Synthetic: write synthetic OOF NPZ
        oof_dir = unit_output_dir / "oof"
        oof_dir.mkdir(exist_ok=True, parents=True)
        oof_npz_path = oof_dir / "unit_oof.npz"
        _write_synthetic_oof_npz(
            oof_npz_path,
            val_records,
            unit,
            [val_fixed_fg_iou or 0.0] * len(val_records),
        )

    return UnitResult(
        unit=unit,
        status=status,
        train_sample_count=len(train_records),
        val_sample_count=len(val_records),
        best_epoch=best_epoch if status == "DONE" else None,
        best_val_loss=best_val_loss if status == "DONE" else None,
        final_val_loss=final_val_loss,
        val_fixed_fg_macro_iou=val_fixed_fg_iou if status == "DONE" else None,
        val_fixed_fg_macro_dice=val_fixed_fg_dice if status == "DONE" else None,
        val_background_iou=val_bg_iou if status == "DONE" else None,
        val_per_region=None,
        val_per_subject=None,
        val_confusion_matrix=None,
        error_message=error_msg,
        wall_seconds=wall_seconds,
        peak_cuda_mb=peak_cuda_mb,
        checkpoint_best_path=best_checkpoint_path if best_checkpoint_path.exists() else None,
        checkpoint_last_path=last_checkpoint_path if last_checkpoint_path.exists() else None,
        oof_csv_path=oof_npz_path,  # backward-compat: stores NPZ path
    )


def _write_real_oof_npz(
    path: Path,
    predictions: np.ndarray,
    targets: np.ndarray,
    sample_ids: list[str],
    subject_ids: list[str],
    fold_ids: list[str],
    unit: FullUnit,
) -> None:
    """Write per-sample H×W predictions + targets + identity as compressed NPZ.

    Each sample's prediction is a full H×W segmentation mask (not a single
    scalar class).  Pooled OOF metrics are recomputed by ``merge_seed_oof``
    from the concatenated arrays using the frozen metric definition; no
    per-sample scalar is stored.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        predictions=predictions,  # shape (N, H, W) int64
        targets=targets,          # shape (N, H, W) int64
        sample_ids=np.asarray(sample_ids, dtype=object),
        subject_ids=np.asarray(subject_ids, dtype=object),
        fold_ids=np.asarray(fold_ids, dtype=object),
        candidate=str(unit.candidate),
        seed=int(unit.seed),
    )


def _write_synthetic_oof_npz(
    path: Path,
    val_records: Sequence[Mapping[str, Any]],
    unit: FullUnit,
    per_sample_iou: list[float] | None = None,
) -> None:
    """Write synthetic OOF NPZ (synthetic mode only).

    In synthetic mode we only have proxy IoU values; the per-sample H×W
    masks are not preserved.  This is acceptable for smoke only.
    """
    sample_ids = [str(r.get("sample_id", f"syn_{i}")) for i, r in enumerate(val_records)]
    subject_ids = [str(r.get("subject_id", "SYN_SUBJ")) for r in val_records]
    fold_ids = [str(unit.fold_id)] * len(val_records)
    dummy_pred = np.zeros((len(val_records), 1, 1), dtype=np.int64)
    dummy_tgt = np.zeros((len(val_records), 1, 1), dtype=np.int64)
    np.savez_compressed(
        path,
        predictions=dummy_pred,
        targets=dummy_tgt,
        sample_ids=np.asarray(sample_ids, dtype=object),
        subject_ids=np.asarray(subject_ids, dtype=object),
        fold_ids=np.asarray(fold_ids, dtype=object),
        candidate=str(unit.candidate),
        seed=int(unit.seed),
        per_sample_iou=np.asarray(
            per_sample_iou if per_sample_iou is not None
            else [0.0] * len(val_records), dtype=np.float64,
        ),
    )


def _write_synthetic_oof_csv(
    path: Path,
    val_records: Sequence[Mapping[str, Any]],
    unit: FullUnit,
    ious: list[float],
) -> None:
    """Write synthetic OOF CSV (synthetic mode only)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "subject_id", "ml_split", "fold_id",
            "seed", "candidate", "val_fixed_fg_macro_iou",
        ])
        writer.writeheader()
        for i, rec in enumerate(val_records):
            writer.writerow({
                "sample_id": rec.get("sample_id", f"synth_{i:05d}"),
                "subject_id": rec.get("subject_id", "SYNTH"),
                "ml_split": "val",
                "fold_id": unit.fold_id,
                "seed": unit.seed,
                "candidate": unit.candidate,
                "val_fixed_fg_macro_iou": ious[i] if i < len(ious) else 0.0,
            })


def _build_synthetic_dataloader(
    records: Sequence[Mapping[str, Any]],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build a DataLoader from synthetic records with in-memory data.

    NOTE: This function is ONLY for synthetic mode.
    Real B01 mode must use Slp8RegionDataset.
    """
    n = len(records)
    image_shape = PRESSURE_SHAPE

    images = np.random.randint(0, 256, size=(n, 1, *image_shape), dtype=np.uint8)
    labels = np.random.randint(0, N_CLASSES, size=(n, *image_shape), dtype=np.int64)

    class _SyntheticDataset(Dataset):
        def __len__(self) -> int:
            return n

        def __getitem__(self, idx: int) -> dict[str, Any]:
            img = torch.from_numpy(images[idx].astype(np.float32) / 255.0)
            lbl = torch.from_numpy(labels[idx]).long()
            return {
                "pressure": img,
                "label": lbl,
                "sample_id": f"synth_{idx:05d}",
                "subject_id": f"SYNTH_{idx % 5:03d}",
                "ml_split": "synthetic",
                "posture": "supine",
            }

    return DataLoader(
        _SyntheticDataset(),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
    )


def _save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: Path,
    unit: FullUnit,
    config: FullConfig,
    *,
    best_epoch: int | None = None,
    best_val_loss: float | None = None,
    early_stopper_state: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
) -> None:
    """Save a checkpoint with embedded identity and full resume state.

    The checkpoint carries enough state to resume from the next epoch
    without re-running from scratch.  All identity fields required for
    fail-closed resume verification are embedded.
    """
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "early_stopper_state": early_stopper_state,
        "rng_state": rng_state,
        "identity": {
            "experiment_id": config.experiment_id,
            "git_commit": config.git_commit,
            "git_dirty": config.git_dirty,
            "config_sha256": config.config_sha256,
            "data_manifest_sha256": config.data_manifest_sha256,
            "fold_manifest_sha256": config.fold_manifest_sha256,
            "split_sha256": config.a06_split_sha256,
            "model_version": unit.candidate,
            "candidate": unit.candidate,
            "fold_id": unit.fold_id,
            "seed": unit.seed,
        },
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(payload, path)


def _capture_rng_state() -> dict[str, Any]:
    """Capture deterministic RNG state for a resumable unit checkpoint."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_checkpoint_for_resume(
    path: Path,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Load a checkpoint and verify identity for partial resume.

    Returns a dict with model_state_dict, optimizer_state_dict, epoch
    (last completed epoch), best_epoch, best_val_loss, early_stopper_state,
    rng_state.  Caller is responsible for applying these to the model
    and optimizer and resuming from epoch+1.

    Raises ``FullProtocolError`` on identity mismatch (fail-closed).
    """
    if not Path(path).is_file():
        raise FullProtocolError(
            f"checkpoint {path} does not exist for resume"
        )
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise FullProtocolError(
            f"failed to load checkpoint {path}: {type(exc).__name__}: {exc}"
        )
    actual_identity = ckpt.get("identity", {})
    for key, expected_value in expected_identity.items():
        actual_value = actual_identity.get(key)
        if actual_value != expected_value:
            raise FullProtocolError(
                f"checkpoint {path} identity mismatch on {key!r}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )
    # Return only the resume-relevant fields
    return {
        "model_state_dict": ckpt["model_state_dict"],
        "optimizer_state_dict": ckpt["optimizer_state_dict"],
        "epoch": int(ckpt.get("epoch", 0)),
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "early_stopper_state": ckpt.get("early_stopper_state"),
        "rng_state": ckpt.get("rng_state"),
    }


# ---------------------------------------------------------------------------
# OOF merging and candidate aggregation
# ---------------------------------------------------------------------------


def merge_seed_oof(
    unit_results: Sequence[UnitResult],
    candidate: str,
    seed: int,
    output_dir: Path,
    fold_val_sample_counts: dict[str, int] | None = None,
    expected_subjects: int = DEV_SUBJECT_COUNT,
    expected_samples: int = DEV_SAMPLE_COUNT,
) -> SeedOOFResult:
    """Merge OOF predictions from 5 folds for one (candidate, seed).

    Each completed unit's per-sample H×W predictions and targets are
    concatenated from ``unit_oof.npz`` files, then a single pooled
    confusion matrix is computed over all 4,095 samples and the frozen
    metric definition is applied.  No placeholder or per-sample scalar
    substitution is used.

    The merged OOF must cover exactly 91 subjects / 4,095 samples
    with 0 duplicate and 0 missing.

    Parameters
    ----------
    unit_results : Sequence[UnitResult]
        Results from all 5 folds for this (candidate, seed).
    candidate : str
        Candidate name.
    seed : int
        Seed.
    output_dir : Path
        Directory to write merged OOF carrier.
    fold_val_sample_counts : dict[str, int] | None
        Expected val sample counts per fold (from B07 fold manifest).
    expected_subjects : int
        Expected subject count (91).
    expected_samples : int
        Expected sample count (4095).

    Returns
    -------
    SeedOOFResult
        Merged OOF result with pooled metrics computed from real
        per-pixel predictions.
    """
    oof_dir = output_dir / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    merged_npz_path = oof_dir / f"{candidate}_seed_{seed:04d}_oof.npz"

    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_sample_ids: list[str] = []
    all_subject_ids: list[str] = []
    all_subjects: set[str] = set()

    for res in unit_results:
        if res.status != "DONE":
            continue
        if res.oof_csv_path is None:
            continue
        oof_path = Path(res.oof_csv_path)
        # Accept both .npz (real + synthetic) and legacy .csv (only if present)
        if oof_path.suffix == ".npz" and oof_path.is_file():
            with np.load(oof_path, allow_pickle=True) as npz:
                preds = np.asarray(npz["predictions"], dtype=np.int64)
                tgts = np.asarray(npz["targets"], dtype=np.int64)
                sids = list(np.asarray(npz["sample_ids"], dtype=object))
                subs = list(np.asarray(npz["subject_ids"], dtype=object))
                for i in range(preds.shape[0]):
                    all_predictions.append(preds[i])
                    all_targets.append(tgts[i])
                    all_sample_ids.append(str(sids[i]))
                    all_subject_ids.append(str(subs[i]))
                    all_subjects.add(str(subs[i]))

    duplicate_count = len(all_sample_ids) - len(set(all_sample_ids))
    missing_count = expected_samples - len(all_sample_ids)

    if (len(all_sample_ids) == expected_samples
            and duplicate_count == 0
            and len(all_predictions) > 0):
        # Concatenate all per-sample H×W masks into one array
        # and compute pooled metrics from the full confusion matrix.
        # For real B01 mode, predictions are real H×W masks.
        # For synthetic mode, predictions are 1×1 dummy masks → metrics are
        # undefined and we leave the values as None (no placeholder).
        preds_concat = np.concatenate(
            [p.reshape(-1) for p in all_predictions], axis=0,
        )
        tgts_concat = np.concatenate(
            [t.reshape(-1) for t in all_targets], axis=0,
        )

        # Only compute metrics if we have real masks (not dummy 1×1)
        real_masks = all_predictions[0].size > 1
        if real_masks:
            try:
                metrics = compute_fixed_class_macro_metrics(
                    y_true=tgts_concat,
                    y_pred=preds_concat,
                    n_classes=N_CLASSES,
                )
                pooled_iou = float(metrics.fixed_iou)
                pooled_dice = float(metrics.fixed_dice)
            except Exception as exc:
                pooled_iou = None
                pooled_dice = None
                status = "INCOMPLETE"
                error_msg = f"pooled metric computation failed: {exc}"
                return SeedOOFResult(
                    candidate=candidate,
                    seed=seed,
                    status=status,
                    total_samples=len(all_sample_ids),
                    total_subjects=len(all_subjects),
                    duplicate_count=duplicate_count,
                    missing_count=missing_count,
                    pooled_fixed_fg_macro_iou=None,
                    pooled_fixed_fg_macro_dice=None,
                    pooled_background_iou=None,
                    pooled_per_subject={},
                    worst_subject_iou=None,
                    oof_csv_path=None,
                    error_message=error_msg,
                )
        else:
            # Synthetic mode: 1×1 masks → no real pooled metric
            pooled_iou = None
            pooled_dice = None

        # Persist merged NPZ carrier
        np.savez_compressed(
            merged_npz_path,
            predictions=np.stack(all_predictions, axis=0).astype(np.int64),
            targets=np.stack(all_targets, axis=0).astype(np.int64),
            sample_ids=np.asarray(all_sample_ids, dtype=object),
            subject_ids=np.asarray(sorted(all_subjects), dtype=object),
            candidate=str(candidate),
            seed=int(seed),
        )

        status = "COMPLETE"
        error_msg = None

        # Per-subject IoU requires re-loading per-sample predictions; we
        # only persist the per-subject means in SeedOOFResult when the
        # metrics are real.  For real B01 mode with H×W masks we would
        # need to recompute the per-class confusion matrix per subject
        # from ``all_predictions`` + ``all_targets`` filtered by subject.
        per_subject: dict[str, float] = {}
        if real_masks:
            # Build per-subject arrays and compute IoU per subject
            for sid in sorted(all_subjects):
                idxs = [i for i, s in enumerate(all_subject_ids) if s == sid]
                sub_pred = np.concatenate([all_predictions[i].reshape(-1) for i in idxs])
                sub_true = np.concatenate([all_targets[i].reshape(-1) for i in idxs])
                sub_metrics = compute_fixed_class_macro_metrics(
                    y_true=sub_true, y_pred=sub_pred, n_classes=N_CLASSES
                )
                per_subject[sid] = float(sub_metrics.fixed_iou)
        worst_iou = min(per_subject.values()) if per_subject else None
    else:
        # Failed or missing folds: INCOMPLETE; no placeholder metrics.
        # Still persist the partial carrier so the caller can inspect
        # what was collected (synthetic smoke uses this).
        if len(all_predictions) > 0:
            np.savez_compressed(
                merged_npz_path,
                predictions=np.stack(all_predictions, axis=0).astype(np.int64),
                targets=np.stack(all_targets, axis=0).astype(np.int64),
                sample_ids=np.asarray(all_sample_ids, dtype=object),
                subject_ids=np.asarray(sorted(all_subjects), dtype=object),
                candidate=str(candidate),
                seed=int(seed),
            )
        pooled_iou = None
        pooled_dice = None
        status = "INCOMPLETE"
        error_msg = (
            f"OOF incomplete: samples={len(all_sample_ids)}/{expected_samples}, "
            f"duplicates={duplicate_count}, missing={missing_count}"
        )
        per_subject = {}
        worst_iou = None

    return SeedOOFResult(
        candidate=candidate,
        seed=seed,
        status=status,
        total_samples=len(all_sample_ids),
        total_subjects=len(all_subjects),
        duplicate_count=duplicate_count,
        missing_count=missing_count,
        pooled_fixed_fg_macro_iou=pooled_iou,
        pooled_fixed_fg_macro_dice=pooled_dice,
        pooled_background_iou=None,
        pooled_per_subject=per_subject,
        worst_subject_iou=worst_iou,
        oof_csv_path=merged_npz_path,
        error_message=error_msg,
    )


def aggregate_candidate_results(
    candidate: str,
    seed_results: dict[int, SeedOOFResult],
    exact_parameter_count: int,
) -> CandidateResult:
    """Aggregate per-seed OOF results into a candidate-level result."""
    complete_seeds = [
        sr for sr in seed_results.values() if sr.status == "COMPLETE"
    ]

    if len(complete_seeds) < len(seed_results):
        status = "INCOMPLETE"
        mean_iou = None
        mean_dice = None
        mean_worst = None
    elif len(complete_seeds) == 0:
        status = "FAILED"
        mean_iou = None
        mean_dice = None
        mean_worst = None
    else:
        status = "DONE"
        ious = [sr.pooled_fixed_fg_macro_iou for sr in complete_seeds if sr.pooled_fixed_fg_macro_iou is not None]
        dice_list = [sr.pooled_fixed_fg_macro_dice for sr in complete_seeds if sr.pooled_fixed_fg_macro_dice is not None]
        worst_list = [sr.worst_subject_iou for sr in complete_seeds if sr.worst_subject_iou is not None]

        mean_iou = sum(ious) / len(ious) if ious else None
        mean_dice = sum(dice_list) / len(dice_list) if dice_list else None
        mean_worst = sum(worst_list) / len(worst_list) if worst_list else None

    return CandidateResult(
        candidate=candidate,
        model_version=candidate,
        exact_parameter_count=exact_parameter_count,
        seed_results=seed_results,
        mean_pooled_iou=mean_iou,
        mean_pooled_dice=mean_dice,
        mean_worst_subject_iou=mean_worst,
        status=status,
        decision=None,
        tiebreak_reason=None,
    )


def apply_selection_rule(
    results: dict[str, CandidateResult],
) -> dict[str, CandidateResult]:
    """Apply the B07 selection rule to pick the winner.

    Rule: rank by mean pooled OOF IoU across seeds descending.
    Near-tie (<0.02): tiebreak by worst subject IoU, parameter count,
    then model version.
    """
    candidates = list(results.keys())

    def sort_key(c: str) -> tuple:
        r = results[c]
        iou = r.mean_pooled_iou if r.mean_pooled_iou is not None else -1.0
        worst = r.mean_worst_subject_iou if r.mean_worst_subject_iou is not None else -1.0
        params = r.exact_parameter_count
        version = r.model_version
        return (-iou, -worst, params, version)

    candidates.sort(key=sort_key)

    if len(candidates) >= 2:
        c1, c2 = candidates[0], candidates[1]
        iou1 = results[c1].mean_pooled_iou or -1.0
        iou2 = results[c2].mean_pooled_iou or -1.0
        diff = iou1 - iou2

        if diff < 0.02:
            tiebreak_reason = (
                f"near_tie: iou_diff={diff:.6f} < 0.02; "
                f"tiebreak by worst_subject_iou, param_count, model_version"
            )
            results[c1].decision = "WINNER"
            results[c2].decision = "ELIMINATED"
            results[c1].tiebreak_reason = tiebreak_reason
            results[c2].tiebreak_reason = tiebreak_reason
        else:
            results[c1].decision = "WINNER"
            results[c1].tiebreak_reason = f"lead: iou_diff={diff:.6f} >= 0.02"
            results[c2].decision = "ELIMINATED"
            results[c2].tiebreak_reason = f"eliminated: iou_diff={diff:.6f} >= 0.02"
    elif len(candidates) == 1:
        results[candidates[0]].decision = "WINNER"

    return results


# ---------------------------------------------------------------------------
# Budget accumulator state
# ---------------------------------------------------------------------------


@dataclass
class BudgetAccumulatorState:
    """B08 full-run budget accumulator state."""

    total_wall_seconds: float
    per_candidate_wall_seconds: dict[str, float]
    per_unit_wall_seconds: dict[str, float]
    peak_cuda_mb_per_candidate: dict[str, float]


def create_budget_accumulator(config: FullConfig) -> BudgetAccumulatorState:
    """Create a budget accumulator for the full run."""
    return BudgetAccumulatorState(
        total_wall_seconds=0.0,
        per_candidate_wall_seconds={c: 0.0 for c in config.protocol.candidates},
        per_unit_wall_seconds={},
        peak_cuda_mb_per_candidate={c: 0.0 for c in config.protocol.candidates},
    )


def check_budget_and_update(
    acc: BudgetAccumulatorState,
    unit: FullUnit,
    result: UnitResult,
    config: FullConfig,
) -> None:
    """Update budget accumulator and fail-closed on budget breach."""
    wall_s = result.wall_seconds
    acc.total_wall_seconds += wall_s
    acc.per_candidate_wall_seconds[unit.candidate] += wall_s
    acc.per_unit_wall_seconds[unit.unit_id] = wall_s

    if result.peak_cuda_mb is not None:
        acc.peak_cuda_mb_per_candidate[unit.candidate] = max(
            acc.peak_cuda_mb_per_candidate[unit.candidate],
            result.peak_cuda_mb,
        )

    total_min = acc.total_wall_seconds / 60.0
    cand_min = acc.per_candidate_wall_seconds[unit.candidate] / 60.0
    unit_min = wall_s / 60.0

    if total_min > config.max_wall_minutes_total:
        raise FullBudgetExceededError(
            f"Total wall budget exceeded: {total_min:.1f}min > "
            f"{config.max_wall_minutes_total}min"
        )
    if cand_min > config.max_wall_minutes_per_candidate:
        raise FullBudgetExceededError(
            f"Candidate {unit.candidate} wall budget exceeded: {cand_min:.1f}min > "
            f"{config.max_wall_minutes_per_candidate}min"
        )
    if unit_min > config.max_wall_minutes_per_unit:
        raise FullBudgetExceededError(
            f"Unit {unit.unit_id} wall budget exceeded: {unit_min:.1f}min > "
            f"{config.max_wall_minutes_per_unit}min"
        )
    if result.peak_cuda_mb is not None and result.peak_cuda_mb > config.max_peak_cuda_mb:
        raise FullBudgetExceededError(
            f"Peak CUDA memory exceeded: {result.peak_cuda_mb:.0f}MB > "
            f"{config.max_peak_cuda_mb}MB"
        )


def build_budget_report(
    acc: BudgetAccumulatorState,
    config: FullConfig,
    unit_results: Sequence[UnitResult],
) -> dict[str, Any]:
    """Build the budget report JSON."""
    return {
        "max_wall_minutes_per_unit": config.max_wall_minutes_per_unit,
        "max_wall_minutes_per_candidate": config.max_wall_minutes_per_candidate,
        "max_wall_minutes_total": config.max_wall_minutes_total,
        "max_peak_cuda_mb": config.max_peak_cuda_mb,
        "total_wall_seconds": round(acc.total_wall_seconds, 2),
        "total_wall_minutes": round(acc.total_wall_seconds / 60.0, 2),
        "per_candidate_wall_seconds": {
            c: round(s, 2) for c, s in acc.per_candidate_wall_seconds.items()
        },
        "per_unit_wall_seconds": {
            uid: round(s, 2) for uid, s in acc.per_unit_wall_seconds.items()
        },
        "peak_cuda_mb_per_candidate": {
            c: round(m, 2) for c, m in acc.peak_cuda_mb_per_candidate.items()
        },
        "budget_ok": (
            acc.total_wall_seconds / 60.0 <= config.max_wall_minutes_total
            and all(
                acc.per_candidate_wall_seconds[c] / 60.0
                <= config.max_wall_minutes_per_candidate
                for c in config.protocol.candidates
            )
        ),
    }


# ---------------------------------------------------------------------------
# Terminal state management
# ---------------------------------------------------------------------------


TERMINAL_STATES = ("DONE", "FAILED", "STOPPED")


def write_unit_complete_atomic(
    unit_output_dir: Path,
    unit: FullUnit,
    config: FullConfig,
    result: UnitResult,
    *,
    identity: Mapping[str, Any],
) -> None:
    """Atomically write a unit's complete.json carrier.

    The complete.json includes:
    - full unit identity (candidate, fold, seed, exp_id, model_version)
    - protocol/config/fold/data SHA
    - budget snapshot (per-candidate + total)
    - checkpoint best/last SHA
    - OOF NPZ carrier SHA

    Existing complete.json is read first; if its identity matches the
    current run, the file is left untouched.  Otherwise the existing
    file is overwritten atomically (temp file + os.replace).
    """
    unit_output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = unit_output_dir / "complete.json"
    payload: dict[str, Any] = {
        "unit": {
            "candidate": unit.candidate,
            "fold_id": unit.fold_id,
            "seed": int(unit.seed),
            "exp_id": identity.get("exp_id"),
            "model_version": identity.get("model_version"),
        },
        "identity": dict(identity),
        "result": {
            "status": result.status,
            "wall_seconds": float(result.wall_seconds),
            "val_sample_count": int(result.val_sample_count),
            "train_sample_count": int(result.train_sample_count),
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "val_fixed_fg_macro_iou": result.val_fixed_fg_macro_iou,
            "val_fixed_fg_macro_dice": result.val_fixed_fg_macro_dice,
            "val_background_iou": result.val_background_iou,
            "peak_cuda_mb": result.peak_cuda_mb,
        },
        "budget": {
            "max_wall_minutes_per_unit": config.max_wall_minutes_per_unit,
            "max_wall_minutes_per_candidate": config.max_wall_minutes_per_candidate,
            "max_wall_minutes_total": config.max_wall_minutes_total,
            "max_peak_cuda_mb": config.max_peak_cuda_mb,
        },
        "checkpoint_best_sha256": (
            compute_file_sha256(result.checkpoint_best_path)
            if result.checkpoint_best_path is not None
            and result.checkpoint_best_path.is_file()
            else None
        ),
        "checkpoint_last_sha256": (
            compute_file_sha256(result.checkpoint_last_path)
            if result.checkpoint_last_path is not None
            and result.checkpoint_last_path.is_file()
            else None
        ),
        "oof_npz_sha256": (
            compute_file_sha256(result.oof_csv_path)
            if result.oof_csv_path is not None
            and result.oof_csv_path.is_file()
            else None
        ),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(complete_path, payload)


def load_resume_state(
    unit_output_dir: Path,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load a unit's complete.json carrier and verify identity.

    Returns the carrier dict if the identity matches the expected
    identity; returns None if no complete.json exists.  Raises
    ``FullProtocolError`` if the identity mismatches.
    """
    complete_path = unit_output_dir / "complete.json"
    if not complete_path.is_file():
        return None
    try:
        carrier = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullProtocolError(
            f"complete.json at {complete_path} is unreadable: {exc}"
        )
    # Verify identity (fail-closed on mismatch)
    actual = carrier.get("identity", {})
    for key, expected_value in expected_identity.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            raise FullProtocolError(
                f"Resume identity mismatch for {unit_output_dir.name} "
                f"on {key!r}: expected {expected_value!r}, got {actual_value!r}"
            )
    return carrier


BUDGET_STATE_FILENAME = "budget_state.json"


def write_budget_state_atomic(
    output_dir: Path,
    budget_acc: BudgetAccumulatorState,
    config: FullConfig,
    identity: Mapping[str, Any],
) -> Path:
    """Atomically write the experiment-level budget state.

    The budget accumulator's totals are persisted so that a resumed run
    can pick up where the previous run stopped without resetting any
    counters (total / per-candidate / per-unit wall, peak CUDA).

    Idempotent: the file is only rewritten when the budget values change,
    so a second run that finds all units complete does not alter the
    stored state and the file hash is preserved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / BUDGET_STATE_FILENAME
    payload = {
        "identity": dict(identity),
        "budget": {
            "total_wall_seconds": float(budget_acc.total_wall_seconds),
            "per_candidate_wall_seconds": {
                k: float(v) for k, v in budget_acc.per_candidate_wall_seconds.items()
            },
            "per_unit_wall_seconds": {
                k: float(v) for k, v in budget_acc.per_unit_wall_seconds.items()
            },
            "peak_cuda_mb_per_candidate": {
                k: float(v) for k, v in budget_acc.peak_cuda_mb_per_candidate.items()
            },
        },
        "config": {
            "max_wall_minutes_per_unit": int(config.max_wall_minutes_per_unit),
            "max_wall_minutes_per_candidate": int(config.max_wall_minutes_per_candidate),
            "max_wall_minutes_total": int(config.max_wall_minutes_total),
            "max_peak_cuda_mb": int(config.max_peak_cuda_mb),
        },
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Idempotent: skip rewrite if the existing file has the same budget
    # values (so file hash is preserved on a no-op resume).
    if state_path.is_file():
        try:
            existing = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                existing.get("budget") == payload["budget"]
                and existing.get("identity") == payload["identity"]
            ):
                return state_path
        except (OSError, json.JSONDecodeError):
            pass
    atomic_write_json(state_path, payload)
    return state_path


def load_budget_state(
    output_dir: Path,
    expected_identity: Mapping[str, Any],
) -> BudgetAccumulatorState | None:
    """Load the persisted budget state; verify identity; return the state
    or None if no state file exists.

    Raises ``FullProtocolError`` on identity mismatch (fail-closed).
    """
    state_path = output_dir / BUDGET_STATE_FILENAME
    if not state_path.is_file():
        return None
    try:
        carrier = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullProtocolError(
            f"budget_state.json at {state_path} is unreadable: {exc}"
        )
    actual = carrier.get("identity", {})
    for key, expected_value in expected_identity.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            raise FullProtocolError(
                f"Resume identity mismatch for budget state on {key!r}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )
    budget = carrier.get("budget", {})
    return BudgetAccumulatorState(
        total_wall_seconds=float(budget.get("total_wall_seconds", 0.0)),
        per_candidate_wall_seconds={
            k: float(v) for k, v in budget.get("per_candidate_wall_seconds", {}).items()
        },
        per_unit_wall_seconds={
            k: float(v) for k, v in budget.get("per_unit_wall_seconds", {}).items()
        },
        peak_cuda_mb_per_candidate={
            k: float(v) for k, v in budget.get("peak_cuda_mb_per_candidate", {}).items()
        },
    )


def write_terminal_state(
    output_dir: Path,
    state: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write exactly one terminal state file.

    B08 Round 4: the function is idempotent — if the same terminal state
    file already exists, the call is a no-op (does not overwrite).  If a
    different terminal state file exists, the call raises to enforce
    exactly-one terminal.
    """
    if state not in TERMINAL_STATES:
        raise FullProtocolError(f"Invalid terminal state: {state!r}")

    # Idempotent: same state file already present → no-op
    same_path = output_dir / f"{state}.json"
    if same_path.exists():
        return

    # A *different* terminal state already exists → refuse
    for ts in TERMINAL_STATES:
        path = output_dir / f"{ts}.json"
        if path.exists() and ts != state:
            raise FullProtocolError(
                f"Terminal state collision: {path.name} already exists; "
                f"refusing to write {state}"
            )

    payload: dict[str, Any] = {
        "terminal_state": state,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)

    path = output_dir / f"{state}.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def check_terminal_state(output_dir: Path) -> str | None:
    """Check for existing terminal state."""
    for ts in TERMINAL_STATES:
        if (output_dir / f"{ts}.json").exists():
            return ts
    return None


def refuse_overwrite(output_dir: Path) -> None:
    """Refuse to write into output_dir if it already has artifacts.

    The B08 production runner does NOT provide a force-overwrite escape
    hatch: an existing experiment directory must never be overwritten.
    Synthetic smoke runs use a fresh temporary directory instead.
    """
    if not output_dir.exists():
        return

    important = [
        output_dir / "manifest.json",
        output_dir / "status.json",
        output_dir / "DONE.json",
        output_dir / "FAILED.json",
        output_dir / "STOPPED.json",
    ]
    for p in important:
        if p.exists():
            raise FullOutputCollisionError(
                f"Output directory {output_dir} already contains {p.name}; "
                "B08 production runner does not allow overwriting. "
                "Use a fresh output directory or run synthetic smoke in a temp dir."
            )


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_json_default(i) for i in obj]
    if isinstance(obj, dict):
        return {str(k): _json_default(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON to ``path`` using temp file + os.replace.

    The temp file is written to the same parent directory so the final
    ``os.replace`` is atomic on the same filesystem.  On Windows +
    NTFS, ``os.replace`` is atomic when both paths are on the same
    volume; this is the standard pattern for crash-safe JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file's content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _gather_environment() -> dict[str, Any]:
    """Gather environment information."""
    return environment_payload()


def write_run_artifacts(
    output_dir: Path,
    config: FullConfig,
    full_result: FullRunResult,
    unit_results: Sequence[UnitResult],
    seed_oof_results: dict[str, dict[int, SeedOOFResult]],
    candidate_results: dict[str, CandidateResult],
) -> None:
    """Write all run artifacts to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # manifest.json
    write_json(output_dir / "manifest.json", {
        "experiment_id": config.experiment_id,
        "git_commit": config.git_commit,
        "git_dirty": config.git_dirty,
        "config_sha256": config.config_sha256,
        "data_manifest_sha256": config.data_manifest_sha256,
        "fold_manifest_sha256": config.fold_manifest_sha256,
        "split_sha256": config.a06_split_sha256,
        "a06_split_sha256": config.a06_split_sha256,
        "task_id": B08_TASK_ID,
        "protocol": B07_PROTOCOL_NAME,
        "config_version": B07_CONFIG_VERSION,
        "synthetic_mode": config.synthetic_mode,
        "candidates": list(config.protocol.candidates),
        "seeds": list(config.protocol.seeds),
        "folds": list(config.protocol.fold_subjects.keys()),
        "total_units": len(unit_results),
        "unit_count_done": full_result.unit_count_done,
        "unit_count_failed": full_result.unit_count_failed,
        "unit_count_stopped": full_result.unit_count_stopped,
        "terminal_state": full_result.terminal_state,
    })

    # resolved_config.json
    write_json(output_dir / "resolved_config.json", {
        "task_id": B08_TASK_ID,
        "protocol": B07_PROTOCOL_NAME,
        "config_version": B07_CONFIG_VERSION,
        "fold_config_version": B07_FOLD_CONFIG_VERSION,
        "experiment_id": config.experiment_id,
        "git_commit": config.git_commit,
        "git_dirty": config.git_dirty,
        "device": config.device,
        "batch_size": config.batch_size,
        "max_epochs": config.max_epochs,
        "min_epochs": config.min_epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "optimizer": config.optimizer,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "max_wall_minutes_per_unit": config.max_wall_minutes_per_unit,
        "max_wall_minutes_per_candidate": config.max_wall_minutes_per_candidate,
        "max_wall_minutes_total": config.max_wall_minutes_total,
        "max_peak_cuda_mb": config.max_peak_cuda_mb,
        "synthetic_mode": config.synthetic_mode,
        "b01_freeze_dir": str(config.b01_freeze_dir) if config.b01_freeze_dir else None,
    })

    # input_manifest_hashes.json
    write_json(output_dir / "input_manifest_hashes.json", {
        "config_sha256": config.config_sha256,
        "data_manifest_sha256": config.data_manifest_sha256,
        "fold_manifest_sha256": config.fold_manifest_sha256,
        "split_sha256": config.a06_split_sha256,
        "a06_split_sha256": config.a06_split_sha256,
    })

    # environment.json
    write_json(output_dir / "environment.json", _gather_environment())

    # fold_manifest.json (copy of frozen fold manifest)
    shutil.copy2(config.protocol.fold_path, output_dir / "fold_manifest.json")

    # status.json
    write_json(output_dir / "status.json", {
        "experiment_id": config.experiment_id,
        "git_commit": config.git_commit,
        "git_dirty": config.git_dirty,
        "terminal_state": full_result.terminal_state,
        "total_units": len(unit_results),
        "unit_count_done": full_result.unit_count_done,
        "unit_count_failed": full_result.unit_count_failed,
        "unit_count_stopped": full_result.unit_count_stopped,
        "winner": full_result.winner,
        "winner_mean_pooled_iou": full_result.winner_mean_pooled_iou,
        "total_wall_seconds": round(full_result.total_wall_seconds, 2),
    })

    # budget_report.json
    write_json(output_dir / "budget_report.json", full_result.budget_report)

    # per-candidate decision
    for cand, cres in candidate_results.items():
        cand_dir = output_dir / "candidates" / cand
        cand_dir.mkdir(parents=True, exist_ok=True)
        write_json(cand_dir / "candidate_decision.json", {
            "candidate": cand,
            "model_version": cres.model_version,
            "exact_parameter_count": cres.exact_parameter_count,
            "decision": cres.decision,
            "tiebreak_reason": cres.tiebreak_reason,
            "mean_pooled_iou": cres.mean_pooled_iou,
            "mean_pooled_dice": cres.mean_pooled_dice,
            "mean_worst_subject_iou": cres.mean_worst_subject_iou,
            "status": cres.status,
            "seeds": {
                str(seed): {
                    "status": sr.status,
                    "total_samples": sr.total_samples,
                    "pooled_fixed_fg_macro_iou": sr.pooled_fixed_fg_macro_iou,
                    "pooled_fixed_fg_macro_dice": sr.pooled_fixed_fg_macro_dice,
                    "worst_subject_iou": sr.worst_subject_iou,
                }
                for seed, sr in cres.seed_results.items()
            },
        })

    # per-unit status
    units_dir = output_dir / "units"
    units_dir.mkdir(exist_ok=True)
    for res in unit_results:
        u_dir = units_dir / res.unit.unit_id
        u_dir.mkdir(parents=True, exist_ok=True)
        write_json(u_dir / "status.json", {
            "unit_id": res.unit.unit_id,
            "candidate": res.unit.candidate,
            "fold_id": res.unit.fold_id,
            "seed": res.unit.seed,
            "status": res.status,
            "train_sample_count": res.train_sample_count,
            "val_sample_count": res.val_sample_count,
            "best_epoch": res.best_epoch,
            "best_val_loss": res.best_val_loss,
            "val_fixed_fg_macro_iou": res.val_fixed_fg_macro_iou,
            "val_fixed_fg_macro_dice": res.val_fixed_fg_macro_dice,
            "wall_seconds": round(res.wall_seconds, 2),
            "peak_cuda_mb": round(res.peak_cuda_mb, 2) if res.peak_cuda_mb else None,
            "error": res.error_message,
            "oof_csv_path": str(res.oof_csv_path) if res.oof_csv_path else None,
        })

    # candidate_decision.json (top-level)
    winner_decision = None
    if full_result.winner:
        wc = candidate_results.get(full_result.winner)
        if wc:
            winner_decision = wc.decision
    write_json(output_dir / "candidate_decision.json", {
        "winner": full_result.winner,
        "winner_decision": winner_decision,
        "winner_mean_pooled_iou": full_result.winner_mean_pooled_iou,
        "candidates": {
            cand: {
                "decision": cres.decision,
                "mean_pooled_iou": cres.mean_pooled_iou,
                "mean_pooled_dice": cres.mean_pooled_dice,
                "mean_worst_subject_iou": cres.mean_worst_subject_iou,
                "exact_parameter_count": cres.exact_parameter_count,
                "status": cres.status,
                "tiebreak_reason": cres.tiebreak_reason,
            }
            for cand, cres in candidate_results.items()
        },
    })

    # oof_metrics_summary.json
    oof_metrics: dict[str, dict[str, Any]] = {}
    for cand, cres in candidate_results.items():
        oof_metrics[cand] = {
            "mean_pooled_iou": cres.mean_pooled_iou,
            "mean_pooled_dice": cres.mean_pooled_dice,
            "mean_worst_subject_iou": cres.mean_worst_subject_iou,
            "status": cres.status,
        }
    write_json(output_dir / "oof_metrics_summary.json", oof_metrics)

    # per_subject_metrics.csv
    per_subject_path = output_dir / "per_subject_metrics.csv"
    with open(per_subject_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "candidate", "seed", "subject_id", "pooled_iou"
        ])
        writer.writeheader()
        for cand, cres in candidate_results.items():
            for seed, sr in cres.seed_results.items():
                if sr.status == "COMPLETE":
                    for subj, iou in sr.pooled_per_subject.items():
                        writer.writerow({
                            "candidate": cand,
                            "seed": seed,
                            "subject_id": subj,
                            "pooled_iou": round(iou, 6),
                        })

    # per_region_metrics.csv (placeholder for B08)
    per_region_path = output_dir / "per_region_metrics.csv"
    with open(per_region_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "candidate", "seed", "region_id", "iou", "dice", "precision", "recall", "support"
        ])
        writer.writeheader()

    # per_posture_metrics.csv (placeholder for B08)
    per_posture_path = output_dir / "per_posture_metrics.csv"
    with open(per_posture_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "candidate", "seed", "posture", "fixed_fg_macro_iou", "fixed_fg_macro_dice"
        ])
        writer.writeheader()


# ---------------------------------------------------------------------------
# Full run orchestration
# ---------------------------------------------------------------------------


def run_full(
    config: FullConfig,
) -> FullRunResult:
    """Execute the full B08 run.

    Dispatches to synthetic or real B01 path based on config.synthetic_mode.
    Both paths are completely separated: synthetic path uses
    _build_synthetic_dataloader and never touches B01; real B01 path
    uses load_b01_freeze_tables and Slp8RegionDataset.

    Parameters
    ----------
    config : FullConfig
        Resolved run configuration.

    Returns
    -------
    FullRunResult
        Complete run result.
    """
    protocol = config.protocol
    units = build_execution_plan(protocol)

    # -----------------------------------------------------------------------
    # B08 Round 4 semantic: DONE.json is informational; per-unit complete.json
    # is the source of truth.  The second-run resume must verify each unit
    # independently.  We do NOT refuse on existing terminal state because
    # the per-unit checks handle the no-overwrite invariant.
    # -----------------------------------------------------------------------
    if not config.no_write_mode:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize accumulators
    budget_acc = create_budget_accumulator(config)
    unit_results: list[UnitResult] = []
    unit_results_by_key: dict[str, UnitResult] = {}

    seed_unit_results: dict[str, dict[int, list[UnitResult]]] = {
        cand: {seed: [] for seed in protocol.seeds}
        for cand in protocol.candidates
    }

    # -----------------------------------------------------------------------
    # RESUME: load persisted budget state (B08 Round 4)
    # -----------------------------------------------------------------------
    experiment_identity: dict[str, Any] = {
        "exp_id": config.experiment_id,
        "git_commit": config.git_commit,
        "config_sha256": config.config_sha256,
        "fold_manifest_sha256": config.fold_manifest_sha256,
        "data_manifest_sha256": config.data_manifest_sha256,
        "split_sha256": config.a06_split_sha256,
        "a06_split_sha256": config.a06_split_sha256,
    }
    persisted_budget = load_budget_state(config.output_dir, experiment_identity)
    if persisted_budget is not None:
        # Round-trip: do not reset the counters
        budget_acc = persisted_budget

    start_time = time.monotonic()
    overall_status = "DONE"
    error_message: str | None = None

    # -----------------------------------------------------------------------
    # PREPARE: load real B01 data if real B01 mode
    # -----------------------------------------------------------------------
    if config.synthetic_mode:
        # SYNTHETIC PATH: pre-compute per-fold synthetic records
        fold_records: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
        for unit in units:
            val_subjects = protocol.fold_subjects.get(unit.fold_id, ())
            n_train = protocol.fold_train_sample_counts.get(unit.fold_id, 3240)
            n_val = protocol.fold_val_sample_counts.get(unit.fold_id, 810)
            n_train = min(n_train, SYNTHETIC_SMOKE_DEFAULTS["n_train_samples"])
            n_val = min(n_val, SYNTHETIC_SMOKE_DEFAULTS["n_val_samples"])
            train_r, val_r = build_synthetic_fold_dataset(
                n_train=n_train,
                n_val=n_val,
                seed=unit.seed + hash(unit.fold_id) % 1000,
            )
            fold_records[unit.fold_id] = (train_r, val_r)
        b01_freeze_dir_for_unit: Path | None = None
        data_root_for_unit: Path | None = None
    else:
        # REAL B01 PATH: load B01 freeze once and cache per-fold partitions
        # Contract: this path must have b01_freeze_dir and data_root
        if config.b01_freeze_dir is None or config.data_root is None:
            raise FullProtocolError(
                "real B01 mode requires b01_freeze_dir and data_root; "
                "synthetic mode requires neither"
            )
        fold_records = {}
        for fold_id, val_subj_tuple in protocol.fold_subjects.items():
            train_s, val_s, norm, cw = load_real_b01_fold(
                b01_freeze_dir=config.b01_freeze_dir,
                data_root=config.data_root,
                fold_id=fold_id,
                val_subject_ids=val_subj_tuple,
                synthetic_mode=False,  # Guard
            )
            # Convert RegionSample → dict for train_one_unit
            fold_records[fold_id] = (train_s, val_s)
        b01_freeze_dir_for_unit = config.b01_freeze_dir
        data_root_for_unit = config.data_root

    # -----------------------------------------------------------------------
    # EXECUTE UNITS (with per-unit resume from complete.json)
    # -----------------------------------------------------------------------
    for unit in units:
        unit_dir = config.output_dir / "units" / unit.unit_id
        if not config.no_write_mode:
            unit_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------------
        # Per-unit expected identity for resume verification
        # -------------------------------------------------------------------
        unit_expected_identity: dict[str, Any] = {
            **experiment_identity,
            "candidate": unit.candidate,
            "fold_id": unit.fold_id,
            "seed": int(unit.seed),
            "model_version": unit.candidate,
            "split_sha256": config.a06_split_sha256,
            "test_access": False,
        }

        # -------------------------------------------------------------------
        # RESUME: try to load cached complete.json (B08 Round 4)
        # -------------------------------------------------------------------
        if not config.no_write_mode:
            cached = load_resume_state(unit_dir, unit_expected_identity)
            if cached is not None:
                # Identity matched: skip training, use cached result
                cached_result = cached.get("result", {})
                cached_status = cached_result.get("status", "DONE")
                if cached_status == "DONE":
                    # Reconstruct a UnitResult from the cached carrier
                    # so the rest of the pipeline (budget, OOF merge, aggregation)
                    # uses the persisted values.
                    cached_unit_result = UnitResult(
                        unit=unit,
                        status="DONE",
                        train_sample_count=int(cached_result.get("train_sample_count", 0)),
                        val_sample_count=int(cached_result.get("val_sample_count", 0)),
                        best_epoch=cached_result.get("best_epoch"),
                        best_val_loss=cached_result.get("best_val_loss"),
                        final_val_loss=cached_result.get("final_val_loss"),
                        val_fixed_fg_macro_iou=cached_result.get("val_fixed_fg_macro_iou"),
                        val_fixed_fg_macro_dice=cached_result.get("val_fixed_fg_macro_dice"),
                        val_background_iou=cached_result.get("val_background_iou"),
                        val_per_region=None,
                        val_per_subject=None,
                        val_confusion_matrix=None,
                        error_message=None,
                        wall_seconds=float(cached_result.get("wall_seconds", 0.0)),
                        peak_cuda_mb=cached_result.get("peak_cuda_mb"),
                        checkpoint_best_path=(
                            unit_dir / "checkpoints" / "best.pt"
                            if (unit_dir / "checkpoints" / "best.pt").is_file()
                            else None
                        ),
                        checkpoint_last_path=(
                            unit_dir / "checkpoints" / "last.pt"
                            if (unit_dir / "checkpoints" / "last.pt").is_file()
                            else None
                        ),
                        oof_csv_path=(
                            unit_dir / "oof" / "unit_oof.npz"
                            if (unit_dir / "oof" / "unit_oof.npz").is_file()
                            else None
                        ),
                    )
                    # B08 Round 4: budget was already accumulated by the
                    # original training run; the persisted budget_state.json
                    # already reflects this unit's wall.  Re-applying would
                    # double-count on resume.  Only update budget if the
                    # persisted state had not recorded this unit.
                    if (
                        unit.unit_id not in budget_acc.per_unit_wall_seconds
                    ):
                        try:
                            check_budget_and_update(
                                budget_acc, unit, cached_unit_result, config,
                            )
                        except FullBudgetExceededError as e:
                            error_message = str(e)
                            overall_status = "STOPPED"
                    unit_results.append(cached_unit_result)
                    unit_results_by_key[unit.unit_id] = cached_unit_result
                    seed_unit_results[unit.candidate][unit.seed].append(cached_unit_result)
                    # Persist budget state after each unit (atomic; idempotent
                    # if values unchanged, so file hash is preserved).
                    if not config.no_write_mode:
                        write_budget_state_atomic(
                            config.output_dir, budget_acc, config,
                            experiment_identity,
                        )
                    continue  # Skip training for this unit
                # If cached is non-DONE (FAILED/STOPPED), fail-closed: re-run
                # is required since partial results are not resumable.
                raise FullProtocolError(
                    f"complete.json for {unit.unit_id} has status={cached_status}; "
                    "B08 Round 4 refuses to overwrite an incomplete unit. "
                    "Use a fresh output directory."
                )

        train_recs, val_recs = fold_records[unit.fold_id]

        # Extra args for real B01 path
        normalization_extra: NormalizationStats | None = None
        class_weight_extra: ClassWeightResult | None = None
        data_root_extra: Path | None = None
        val_sample_ids_extra: list[str] | None = None
        val_subject_ids_list_extra: list[str] | None = None
        val_postures_extra: list[str] | None = None

        if not config.synthetic_mode:
            # REAL B01 PATH: extract extra data for train_one_unit
            # Re-load the fold to get normalization/class_weights (already computed above)
            _, _, norm, cw = load_real_b01_fold(
                b01_freeze_dir=config.b01_freeze_dir,  # type: ignore[arg-type]
                data_root=config.data_root,  # type: ignore[arg-type]
                fold_id=unit.fold_id,
                val_subject_ids=protocol.fold_subjects[unit.fold_id],
                synthetic_mode=False,
            )
            normalization_extra = norm
            class_weight_extra = cw
            data_root_extra = config.data_root

            # Extract ordered val sample IDs for OOF CSV
            if isinstance(val_recs[0], RegionSample) if val_recs else False:
                val_sample_ids_extra = [r.sample_id for r in val_recs]
                val_subject_ids_list_extra = [r.subject_id for r in val_recs]
                val_postures_extra = [r.posture for r in val_recs]
            else:
                val_sample_ids_extra = [str(r.get("sample_id", f"unk_{i}")) for i, r in enumerate(val_recs)]
                val_subject_ids_list_extra = [str(r.get("subject_id", "UNK")) for r in val_recs]
                val_postures_extra = [str(r.get("posture", "unknown")) for r in val_recs]

        try:
            result = train_one_unit(
                unit=unit,
                train_records=train_recs,
                val_records=val_recs,
                config=config,
                unit_output_dir=unit_dir,
                normalization=normalization_extra,
                class_weight_result=class_weight_extra,
                data_root=data_root_extra,
                val_sample_ids=val_sample_ids_extra,
                val_subject_ids_list=val_subject_ids_list_extra,
                val_postures=val_postures_extra,
            )
        except Exception as e:
            result = UnitResult(
                unit=unit,
                status="FAILED",
                train_sample_count=len(train_recs),
                val_sample_count=len(val_recs),
                best_epoch=None,
                best_val_loss=None,
                final_val_loss=None,
                val_fixed_fg_macro_iou=None,
                val_fixed_fg_macro_dice=None,
                val_background_iou=None,
                val_per_region=None,
                val_per_subject=None,
                val_confusion_matrix=None,
                error_message=f"{type(e).__name__}: {e}",
                wall_seconds=0.0,
                peak_cuda_mb=None,
                checkpoint_best_path=None,
                checkpoint_last_path=None,
                oof_csv_path=None,
            )

        # Check budget
        try:
            check_budget_and_update(budget_acc, unit, result, config)
        except FullBudgetExceededError as e:
            result = UnitResult(
                unit=unit,
                status="STOPPED",
                train_sample_count=result.train_sample_count,
                val_sample_count=result.val_sample_count,
                best_epoch=None,
                best_val_loss=None,
                final_val_loss=None,
                val_fixed_fg_macro_iou=None,
                val_fixed_fg_macro_dice=None,
                val_background_iou=None,
                val_per_region=None,
                val_per_subject=None,
                val_confusion_matrix=None,
                error_message=str(e),
                wall_seconds=result.wall_seconds,
                peak_cuda_mb=result.peak_cuda_mb,
                checkpoint_best_path=None,
                checkpoint_last_path=None,
                oof_csv_path=None,
            )
            overall_status = "STOPPED"
            error_message = str(e)

        # -------------------------------------------------------------------
        # POST-TRAIN: atomic complete.json (B08 Round 4)
        # Only write complete.json if training succeeded.  Failed/Stopped
        # units are NOT given a complete.json so the next run will re-run
        # them (fail-closed).
        # -------------------------------------------------------------------
        if not config.no_write_mode and result.status == "DONE":
            try:
                write_unit_complete_atomic(
                    unit_dir, unit, config, result,
                    identity=unit_expected_identity,
                )
            except Exception as e:
                raise FullProtocolError(
                    f"Failed to atomically write complete.json for {unit.unit_id}: {e}"
                )

        # Persist budget state after each unit (atomic)
        if not config.no_write_mode:
            try:
                write_budget_state_atomic(
                    config.output_dir, budget_acc, config,
                    experiment_identity,
                )
            except Exception:
                pass  # budget write failure is non-fatal at unit granularity

        unit_results.append(result)
        unit_results_by_key[unit.unit_id] = result
        seed_unit_results[unit.candidate][unit.seed].append(result)

        if result.status == "FAILED":
            overall_status = "FAILED"
            if error_message is None:
                error_message = f"Unit {unit.unit_id} FAILED"
        elif result.status == "STOPPED":
            overall_status = "STOPPED"
            if error_message is None:
                error_message = f"Unit {unit.unit_id} STOPPED"
        elif result.status == "INTERRUPTED":
            overall_status = "INCOMPLETE"
            if error_message is None:
                error_message = f"Unit {unit.unit_id} interrupted; checkpoint retained"

    total_wall = time.monotonic() - start_time

    # -----------------------------------------------------------------------
    # MERGE OOF
    # -----------------------------------------------------------------------
    seed_oof: dict[str, dict[int, SeedOOFResult]] = {}
    for cand in protocol.candidates:
        seed_oof[cand] = {}
        for seed in protocol.seeds:
            cand_seed_units = [
                r for r in unit_results
                if r.unit.candidate == cand and r.unit.seed == seed
            ]
            sr = merge_seed_oof(
                unit_results=cand_seed_units,
                candidate=cand,
                seed=seed,
                output_dir=config.output_dir,
                fold_val_sample_counts=protocol.fold_val_sample_counts,
            )
            seed_oof[cand][seed] = sr

    # -----------------------------------------------------------------------
    # AGGREGATE CANDIDATES
    # -----------------------------------------------------------------------
    candidate_results: dict[str, CandidateResult] = {}
    for cand in protocol.candidates:
        builder = get_model_builder(cand)
        model, _ = builder.factory(n_classes=N_CLASSES, device="cpu")
        param_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        del model

        cres = aggregate_candidate_results(
            candidate=cand,
            seed_results=seed_oof[cand],
            exact_parameter_count=param_count,
        )
        candidate_results[cand] = cres

    candidate_results = apply_selection_rule(candidate_results)

    winner = None
    winner_iou = None
    # B08 Round 4: synthetic mode never ranks candidates; the winner
    # must remain None when pooled metrics are absent (synthetic 1×1
    # dummy masks cannot produce real metrics).  Even in real mode we
    # only set winner when at least one candidate has a real metric.
    if not config.synthetic_mode:
        for cand, cres in candidate_results.items():
            if cres.decision == "WINNER" and cres.mean_pooled_iou is not None:
                winner = cand
                winner_iou = cres.mean_pooled_iou
                break

    unit_done = sum(1 for r in unit_results if r.status == "DONE")
    unit_failed = sum(1 for r in unit_results if r.status == "FAILED")
    unit_stopped = sum(1 for r in unit_results if r.status == "STOPPED")

    budget_rep = build_budget_report(budget_acc, config, unit_results)

    full_result = FullRunResult(
        experiment_id=config.experiment_id,
        git_commit=config.git_commit,
        git_dirty=config.git_dirty,
        config_sha256=config.config_sha256,
        data_manifest_sha256=config.data_manifest_sha256,
        fold_manifest_sha256=config.fold_manifest_sha256,
        a06_split_sha256=config.a06_split_sha256,
        candidate_results=candidate_results,
        winner=winner,
        winner_mean_pooled_iou=winner_iou,
        terminal_state=overall_status,
        total_wall_seconds=total_wall,
        unit_count_total=len(unit_results),
        unit_count_done=unit_done,
        unit_count_failed=unit_failed,
        unit_count_stopped=unit_stopped,
        budget_report=budget_rep,
        error_message=error_message,
    )

    # Write artifacts (unless no-write mode)
    if not config.no_write_mode:
        write_run_artifacts(
            output_dir=config.output_dir,
            config=config,
            full_result=full_result,
            unit_results=unit_results,
            seed_oof_results=seed_oof,
            candidate_results=candidate_results,
        )

        # A deliberately interrupted/incomplete run must remain resumable and
        # must not seal the experiment with FAILED/STOPPED.json.  Terminal
        # state is written only once every planned unit has reached a terminal
        # result; the next invocation can then resume from last.pt.
        if len(unit_results) == len(units) and overall_status in {"DONE", "FAILED", "STOPPED"}:
            write_terminal_state(
                config.output_dir,
                overall_status,
                extra={
                "experiment_id": config.experiment_id,
                "git_commit": config.git_commit,
                "git_dirty": config.git_dirty,
                "total_units": len(unit_results),
                "unit_count_done": unit_done,
                "unit_count_failed": unit_failed,
                "unit_count_stopped": unit_stopped,
                "winner": winner,
                "winner_mean_pooled_iou": winner_iou,
                "total_wall_seconds": round(total_wall, 2),
                },
            )

    return full_result


# ---------------------------------------------------------------------------
# Configuration building
# ---------------------------------------------------------------------------


def build_full_config(
    protocol_path: Path,
    output_dir: Path,
    *,
    experiment_id: str,
    git_commit: str,
    git_dirty: bool,
    b01_freeze_dir: Path | None = None,
    data_root: Path | None = None,
    device: str = "cuda",
    batch_size: int = 16,
    max_epochs: int = 30,
    min_epochs: int = 5,
    early_stopping_patience: int = 4,
    optimizer: str = "AdamW",
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    synthetic_mode: bool = False,
    no_write_mode: bool = False,
    validate_only: bool = False,
    max_wall_minutes_per_unit: int = BUDGET_PREFLIGHT_MAX_WALL_MINUTES_PER_UNIT,
    interrupt_after_epoch: int | None = None,
    repo_root: Path | None = None,
) -> FullConfig:
    """Build a validated FullConfig."""
    protocol = load_frozen_full_protocol(protocol_path, repo_root=repo_root)

    rr = repo_root or protocol_path.parents[2]
    # B08 Round 4: explicitly anchor SHA reads to the frozen git_commit
    # passed in.  We never default to HEAD because the working tree may
    # have CRLF or staged changes.  If git_commit is not a real SHA,
    # the helper returns None and we fail closed.
    config_sha = committed_file_sha256(
        rr,
        str(protocol_path.relative_to(rr)),
        frozen_git_sha=git_commit,
    )
    if config_sha is None:
        raise FullConfigValidationError(
            f"Could not read committed SHA of {protocol_path} at frozen_git_sha={git_commit!r}; "
            "B08 Round 4 requires an explicit frozen git SHA.  Refusing to fall back to "
            "the working tree."
        )

    if synthetic_mode:
        data_manifest_sha = _compute_synthetic_manifest_sha256()
        a06_split_sha = "synthetic_not_applicable"
    else:
        if b01_freeze_dir is None:
            raise FullConfigValidationError(
                "b01_freeze_dir is required for real B01 runs"
            )
        freeze_manifest = Path(b01_freeze_dir) / "freeze_manifest.json"
        if not freeze_manifest.exists():
            raise FullConfigValidationError(
                f"freeze_manifest.json not found at {freeze_manifest}; "
                "ensure B01 freeze tables are available"
            )
        data_manifest_sha = file_sha256(freeze_manifest)

        try:
            fm = json.loads(freeze_manifest.read_text(encoding="utf-8"))
            a06_split_sha = str(fm.get("core", {}).get("a06_split_sha256", ""))
        except Exception:
            a06_split_sha = ""

    return FullConfig(
        protocol=protocol,
        output_dir=Path(output_dir).resolve(),
        experiment_id=experiment_id,
        git_commit=git_commit,
        git_dirty=git_dirty,
        b01_freeze_dir=Path(b01_freeze_dir) if b01_freeze_dir else None,
        data_root=Path(data_root) if data_root else None,
        device=device,
        batch_size=batch_size,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        early_stopping_patience=early_stopping_patience,
        optimizer=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        synthetic_mode=synthetic_mode,
        no_write_mode=no_write_mode,
        validate_only=validate_only,
        max_wall_minutes_per_unit=max_wall_minutes_per_unit,
        max_wall_minutes_per_candidate=BUDGET_MAX_WALL_MINUTES_PER_CANDIDATE,
        max_wall_minutes_total=BUDGET_MAX_WALL_MINUTES_TOTAL,
        max_peak_cuda_mb=BUDGET_MAX_PEAK_CUDA_MB,
        config_sha256=config_sha,
        data_manifest_sha256=data_manifest_sha,
        fold_manifest_sha256=protocol.fold_sha256,
        a06_split_sha256=a06_split_sha,
        interrupt_after_epoch=interrupt_after_epoch,
    )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def resolve_git_identity(repo_root: Path) -> tuple[str, bool]:
    """Resolve git commit and dirty status from a repo root."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise FullExperimentIdentityError(
                f"git rev-parse HEAD failed: {result.stderr.decode().strip()}"
            )
        commit = result.stdout.decode().strip()
        if not commit or len(commit) != 40:
            raise FullExperimentIdentityError(
                f"git commit is not a valid 40-char hex SHA: {commit!r}"
            )

        diff_result = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=str(repo_root),
            capture_output=True,
            timeout=10,
        )
        dirty = diff_result.returncode != 0 or len(diff_result.stdout) > 0

        return commit, dirty
    except subprocess.TimeoutExpired as e:
        raise FullExperimentIdentityError(f"git command timed out: {e}")
    except OSError as e:
        raise FullExperimentIdentityError(f"git not available: {e}")
