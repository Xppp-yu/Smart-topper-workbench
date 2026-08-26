"""SLP8 PM-only Region Mini Runner (TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-AND-RUNNER-v0.1).

This module is the **B04 Mini core**: it freezes the protocol that the
Experiment Runner would later execute on real B01 data.  B04 ships the
runner, the model registry, the class-weight contract, the extended
metrics, the fail-closed configuration validation, the resource
budget monitor, the checkpoint / resume contract, and the determinism
configuration.  The actual real run is gated behind an explicit
``--run-authorized`` flag (see ``scripts/run_slp8_region_mini.py``) and
is **not** performed by this task.

Design contract (R02)
=====================

* **Data contract** — Only B01 freeze tables are read; the runner
  never constructs a TEST loader and refuses any caller-supplied path
  that does not look like a real B01 freeze directory.  TEST rows
  remain inaccessible because :func:`load_b01_freeze_tables` is called
  with ``load_test=False``.  The real B01 path performs a
  fail-closed contract check (counts / subject counts / A06 SHA /
  provenance / setting / cover) — see
  :mod:`topper_perception.neural.slp8_region_b01_contract`.

* **Class-weight contract** — The class weight vector is derived from
  ``train_class_stats.json`` only; VAL/TEST are rejected.

* **Training contract** — AdamW, lr=0.001, weight_decay=1e-4, no
  scheduler, no augmentation, ``num_workers=0``, single seed (42),
  ``max_epochs=20`` and ``min_epochs=5`` with early stopping on
  ``val_loss`` (patience=4, min_delta=0.0, mode="min").  CUDA is the
  requested device; when it is unavailable the runner fail-closes
  (refuses to fall back to CPU) **unless** the run is a synthetic CPU
  smoke, in which case CPU is the only allowed device.

* **Resource budget** — The runner is monitored end-to-end:
  ``time.monotonic`` is sampled after every validation epoch, the
  per-candidate and total wall budgets are compared, and the CUDA peak
  memory (via :func:`torch.cuda.max_memory_allocated` after a
  :func:`torch.cuda.reset_peak_memory_stats` at candidate start) is
  compared against the frozen threshold.  Any exceedance transitions
  the candidate to ``STOPPED`` (not ``FAILED``) and the run never
  writes ``DONE.json``.

* **Checkpoint / resume contract** — One ``last.pt`` and one
  ``best.pt`` per candidate.  ``best`` is selected by the lowest
  val_loss (mode="min", earliest-epoch tie-break).  Independent
  reload (a freshly-built model loaded from ``best.pt``) must produce
  predictions whose canonical hash matches the in-process predictions.
  Every checkpoint embeds a :class:`CheckpointIdentity` block; resume
  with a mismatched identity is rejected fail-closed.  Resume for a
  run that already produced ``DONE.json`` is refused.

* **Determinism contract** — :func:`apply_settings` configures Python,
  NumPy and torch RNGs plus CPU thread count and deterministic-algorithm
  flags; two independent processes with the same seed / commit /
  config must produce identical ``predictions_manifest`` /
  ``centroid_errors`` / ``candidate_decision`` outputs.

* **Output contract** — Every output file in the B04 specification is
  written (see :func:`write_mini_artifacts`).  Output directory
  collisions are rejected **before any file is written**.  The
  terminal ``DONE.json`` / ``FAILED.json`` / ``STOPPED.json`` files
  are mutually exclusive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

from topper_perception.evaluation.slp_pressure_metrics import (
    compute_fixed_class_macro_metrics,
)
from topper_perception.neural.slp8_region_b01_contract import (
    B01ContractError,
    B01FreezeSnapshot,
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
    B04_MAX_PARAMETERS,
    INPUT_SHAPE,
    MODEL_REGISTRY,
    ModelBuilder,
    SMALL_UNET_VERSION,
    MODEL_VERSION,
    get_model_builder,
    list_model_builders,
)
from topper_perception.neural.slp8_region_metrics_ext import (
    DEFAULT_IMAGE_SHAPE,
    FOREGROUND_CLASS_IDS,
    compute_extended_metrics,
    compute_centroid_errors,
    summarize_centroid_errors,
    METRICS_VERSION as EXT_METRICS_VERSION,
)
from topper_perception.neural.slp8_region_resume import (
    CheckpointIdentity,
    EarlyStopperState,
    ResumeIdentityError,
    ResumeRefusedError,
    capture_rng_state,
    class_weight_sha256,
    file_sha256 as _resume_file_sha256,
    identity_from_dict,
    input_manifest_hashes_sha256,
    refuse_resume_for_done_run,
    restore_rng_state,
    verify_resume_identity,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_ID = "TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-AND-RUNNER-v0.1"
MINI_VERSION = "slp8_region_mini_v0.1"

#: Default seed.
DEFAULT_SEED = 42

#: Allowed candidates for B04 Mini (frozen).
B04_CANDIDATE_NAMES: tuple[str, ...] = (
    MODEL_VERSION,         # slp8_tiny_fcn_v0.1
    SMALL_UNET_VERSION,    # slp8_small_unet_v0.1
)

#: Frozen training defaults from the B04 protocol.
B04_FROZEN_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "device": "cuda",
    "batch_size": 16,
    "max_epochs": 20,
    "min_epochs": 5,
    "early_stopping": {
        "monitor": "val_loss",
        "mode": "min",
        "patience": 4,
        "min_delta": 0.0,
    },
    "optimizer": "AdamW",
    "lr": 0.001,
    "weight_decay": 1e-4,
    "scheduler": "none",
    "augmentation": "none",
    "num_workers": 0,
}

#: Frozen resource budget.
B04_RESOURCE_BUDGET: dict[str, Any] = {
    "max_wall_minutes_per_candidate": 45,
    "max_total_wall_minutes": 90,
    "max_peak_cuda_mb": 12288,
    "candidates_serial": True,
}

#: Frozen B02 baseline reference (Train Spatial Prior).  Used as the
#: FEASIBLE gate threshold for B04 candidates (must be >= 0.205644).
B02_BASELINE_REFERENCE_VAL_FIXED_IOU = 0.205644

#: Synthetic CPU smoke defaults.
SYNTHETIC_DEFAULTS: dict[str, Any] = {
    "n_train_samples": 8,
    "n_val_samples": 4,
    "image_shape": list(PRESSURE_SHAPE),
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MiniProtocolError(Exception):
    """Base error for B04 Mini protocol violations."""


class ConfigValidationError(MiniProtocolError):
    """Raised when the B04 Mini config fails the fail-closed validation."""


class OutputCollisionError(MiniProtocolError):
    """Raised when the output directory is not safe to write into."""


class RunAuthorizationError(MiniProtocolError):
    """Raised when a real B01 run is requested without --run-authorized."""


class NonFiniteMetricsError(MiniProtocolError):
    """Raised when a loss/metric is non-finite after a candidate epoch."""


# ---------------------------------------------------------------------------
# Mini configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MiniConfig:
    """The B04 Mini frozen configuration (built from the JSON config).

    Every field is explicit; defaults are not used to mask missing
    config.  Validation lives in :func:`validate_mini_config`.
    """

    task_id: str
    config_version: str
    config_path: str
    candidates: tuple[str, ...]
    seed: int
    device: str
    batch_size: int
    max_epochs: int
    min_epochs: int
    early_stopping_monitor: str
    early_stopping_mode: str
    early_stopping_patience: int
    early_stopping_min_delta: float
    optimizer: str
    lr: float
    weight_decay: float
    num_workers: int
    resource_budget: dict[str, Any]
    data_root: str
    b01_freeze_dir: str
    b01_a06_split_sha256_expected: str
    provenance: str
    raw_semantics: str
    source_review_status: str
    n_classes: int
    image_shape: tuple[int, int]
    max_parameters: int
    val_feasibility_threshold: float
    expected_train_count: int
    expected_val_count: int
    expected_test_count: int
    expected_train_subjects: int
    expected_val_subjects: int
    expected_test_subjects: int
    expected_provenance: str
    expected_source_review_status: str
    expected_setting: str
    expected_cover: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "config_version": self.config_version,
            "config_path": self.config_path,
            "candidates": list(self.candidates),
            "seed": self.seed,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "early_stopping_monitor": self.early_stopping_monitor,
            "early_stopping_mode": self.early_stopping_mode,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "num_workers": self.num_workers,
            "resource_budget": dict(self.resource_budget),
            "data_root": "REDACTED_LOCAL_PATH",
            "b01_freeze_dir": "REDACTED_LOCAL_PATH",
            "b01_a06_split_sha256_expected": self.b01_a06_split_sha256_expected,
            "provenance": self.provenance,
            "raw_semantics": self.raw_raw_semantics_safe(),
            "source_review_status": self.source_review_status,
            "n_classes": self.n_classes,
            "image_shape": list(self.image_shape),
            "max_parameters": self.max_parameters,
            "val_feasibility_threshold": self.val_feasibility_threshold,
            "expected_split_counts": {
                "train": int(self.expected_train_count),
                "val": int(self.expected_val_count),
                "test": int(self.expected_test_count),
            },
            "expected_subjects": {
                "train": int(self.expected_train_subjects),
                "val": int(self.expected_val_subjects),
                "test": int(self.expected_test_subjects),
            },
            "expected_provenance": self.expected_provenance,
            "expected_source_review_status": self.expected_source_review_status,
            "expected_setting": self.expected_setting,
            "expected_cover": self.expected_cover,
        }

    def raw_raw_semantics_safe(self) -> str:
        return self.raw_semantics


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


_REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "config_version",
    "task_id",
    "provenance",
    "raw_semantics",
    "source_review_status",
    "b01_a06_split_sha256_expected",
    "candidates",
    "training",
    "dataset",
    "metrics",
    "resource_budget",
    "feasibility_gate",
    "expected_split_counts",
    "expected_subjects",
    "lifecycle",
)

_REQUIRED_LIFECYCLE_KEYS: tuple[str, ...] = (
    "valid_terminal_states",
    "exclusive_terminal_files",
)

_REQUIRED_CANDIDATE_KEYS: tuple[str, ...] = (
    "name",
    "version",
    "max_parameters",
)

_REQUIRED_TRAINING_KEYS: tuple[str, ...] = (
    "seed",
    "device",
    "batch_size",
    "max_epochs",
    "min_epochs",
    "early_stopping",
    "optimizer",
    "lr",
    "weight_decay",
    "num_workers",
)

_REQUIRED_EARLY_STOPPING_KEYS: tuple[str, ...] = (
    "monitor",
    "mode",
    "patience",
    "min_delta",
)

_REQUIRED_DATASET_KEYS: tuple[str, ...] = (
    "image_shape",
    "n_classes",
)

_REQUIRED_METRICS_KEYS: tuple[str, ...] = (
    "fixed_foreground_class_ids",
    "compute_per_region",
    "compute_per_posture",
    "compute_per_subject",
    "compute_centroid_error",
)

_REQUIRED_FEASIBILITY_KEYS: tuple[str, ...] = (
    "b02_reference_val_fixed_iou",
)


def _expect_keys(parent: str, payload: Mapping[str, Any], required: Sequence[str]) -> None:
    missing = [k for k in required if k not in payload]
    if missing:
        raise ConfigValidationError(
            f"{parent}: missing required keys {missing!r}"
        )


def validate_mini_config(cfg: Mapping[str, Any]) -> None:
    """Validate the B04 Mini JSON config (fail-closed)."""

    _expect_keys("config", cfg, _REQUIRED_TOP_LEVEL_KEYS)

    if cfg["task_id"] != TASK_ID:
        raise ConfigValidationError(
            f"config.task_id {cfg['task_id']!r} != expected {TASK_ID!r}"
        )
    if cfg["config_version"] != MINI_VERSION:
        raise ConfigValidationError(
            f"config.config_version {cfg['config_version']!r} != "
            f"expected {MINI_VERSION!r}"
        )
    if cfg["provenance"] != "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED":
        raise ConfigValidationError(
            "config.provenance must be 'V221_CORRECTED_SUPPORT_AUTO_ACCEPTED'"
        )
    if cfg["raw_semantics"] != "raw_pmarray_response":
        raise ConfigValidationError(
            "config.raw_semantics must be 'raw_pmarray_response' (NOT kPa)"
        )
    if cfg["source_review_status"] != "NOT_REVIEWED":
        raise ConfigValidationError(
            "config.source_review_status must be 'NOT_REVIEWED'"
        )

    candidates = cfg["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ConfigValidationError(
            "config.candidates must be a non-empty list"
        )
    candidate_names: list[str] = []
    for c in candidates:
        _expect_keys("config.candidates[]", c, _REQUIRED_CANDIDATE_KEYS)
        name = c["name"]
        if name not in MODEL_REGISTRY:
            raise ConfigValidationError(
                f"candidate {name!r} is not registered; known: "
                f"{list_model_builders()}"
            )
        if c["version"] != get_model_builder(name).version:
            raise ConfigValidationError(
                f"candidate {name!r} version mismatch: config says "
                f"{c['version']!r}, registry has {get_model_builder(name).version!r}"
            )
        max_params = int(c["max_parameters"])
        if max_params > B04_MAX_PARAMETERS:
            raise ConfigValidationError(
                f"candidate {name!r} max_parameters={max_params} exceeds "
                f"B04 cap of {B04_MAX_PARAMETERS}"
            )
        if name in candidate_names:
            raise ConfigValidationError(
                f"candidate {name!r} listed more than once in config.candidates"
            )
        candidate_names.append(name)
    if tuple(candidate_names) != B04_CANDIDATE_NAMES:
        # B04 v0.1: the candidate set is frozen and the order is part of
        # the contract.
        raise ConfigValidationError(
            f"config.candidates must equal {list(B04_CANDIDATE_NAMES)!r} in order; "
            f"got {candidate_names!r}"
        )

    training = cfg["training"]
    _expect_keys("config.training", training, _REQUIRED_TRAINING_KEYS)
    if int(training["seed"]) != DEFAULT_SEED:
        raise ConfigValidationError(
            f"config.training.seed must be {DEFAULT_SEED} for the frozen protocol"
        )
    if str(training["device"]) != "cuda":
        raise ConfigValidationError(
            f"config.training.device must be 'cuda' for the frozen protocol"
        )
    if int(training["batch_size"]) != 16:
        raise ConfigValidationError(
            f"config.training.batch_size must be 16; got {training['batch_size']}"
        )
    if int(training["max_epochs"]) != 20:
        raise ConfigValidationError(
            f"config.training.max_epochs must be 20; got {training['max_epochs']}"
        )
    if int(training["min_epochs"]) != 5:
        raise ConfigValidationError(
            f"config.training.min_epochs must be 5; got {training['min_epochs']}"
        )
    early = training["early_stopping"]
    _expect_keys("config.training.early_stopping", early, _REQUIRED_EARLY_STOPPING_KEYS)
    if str(early["monitor"]) != "val_loss":
        raise ConfigValidationError(
            f"config.training.early_stopping.monitor must be 'val_loss'; got "
            f"{early['monitor']!r}"
        )
    if str(early["mode"]) != "min":
        raise ConfigValidationError(
            f"config.training.early_stopping.mode must be 'min'; got "
            f"{early['mode']!r}"
        )
    if int(early["patience"]) != 4:
        raise ConfigValidationError(
            f"config.training.early_stopping.patience must be 4; got "
            f"{early['patience']}"
        )
    if float(early["min_delta"]) != 0.0:
        raise ConfigValidationError(
            f"config.training.early_stopping.min_delta must be 0.0; got "
            f"{early['min_delta']}"
        )
    if str(training["optimizer"]) != "AdamW":
        raise ConfigValidationError(
            f"config.training.optimizer must be 'AdamW'; got {training['optimizer']!r}"
        )
    if float(training["lr"]) != 0.001:
        raise ConfigValidationError(
            f"config.training.lr must be 0.001; got {training['lr']}"
        )
    if float(training["weight_decay"]) != 1e-4:
        raise ConfigValidationError(
            f"config.training.weight_decay must be 1e-4; got {training['weight_decay']}"
        )
    if int(training["num_workers"]) != 0:
        raise ConfigValidationError(
            f"config.training.num_workers must be 0; got {training['num_workers']}"
        )

    dataset = cfg["dataset"]
    _expect_keys("config.dataset", dataset, _REQUIRED_DATASET_KEYS)
    if tuple(dataset["image_shape"]) != INPUT_SHAPE:
        raise ConfigValidationError(
            f"config.dataset.image_shape must be {list(INPUT_SHAPE)!r}; "
            f"got {dataset['image_shape']!r}"
        )
    if int(dataset["n_classes"]) != N_CLASSES:
        raise ConfigValidationError(
            f"config.dataset.n_classes must be {N_CLASSES}; got {dataset['n_classes']}"
        )

    metrics_cfg = cfg["metrics"]
    _expect_keys("config.metrics", metrics_cfg, _REQUIRED_METRICS_KEYS)
    if list(metrics_cfg["fixed_foreground_class_ids"]) != list(FOREGROUND_CLASS_IDS):
        raise ConfigValidationError(
            f"config.metrics.fixed_foreground_class_ids must be "
            f"{list(FOREGROUND_CLASS_IDS)!r}; got {metrics_cfg['fixed_foreground_class_ids']!r}"
        )

    rb = cfg["resource_budget"]
    if int(rb["max_wall_minutes_per_candidate"]) != 45:
        raise ConfigValidationError(
            f"config.resource_budget.max_wall_minutes_per_candidate must be 45; "
            f"got {rb['max_wall_minutes_per_candidate']}"
        )
    if int(rb["max_total_wall_minutes"]) != 90:
        raise ConfigValidationError(
            f"config.resource_budget.max_total_wall_minutes must be 90; "
            f"got {rb['max_total_wall_minutes']}"
        )
    if int(rb["max_peak_cuda_mb"]) != 12288:
        raise ConfigValidationError(
            f"config.resource_budget.max_peak_cuda_mb must be 12288; "
            f"got {rb['max_peak_cuda_mb']}"
        )

    fg = cfg["feasibility_gate"]
    _expect_keys("config.feasibility_gate", fg, _REQUIRED_FEASIBILITY_KEYS)
    threshold = float(fg["b02_reference_val_fixed_iou"])
    if not math.isclose(threshold, B02_BASELINE_REFERENCE_VAL_FIXED_IOU, rel_tol=0, abs_tol=0):
        raise ConfigValidationError(
            f"config.feasibility_gate.b02_reference_val_fixed_iou must be "
            f"{B02_BASELINE_REFERENCE_VAL_FIXED_IOU}; got {threshold}"
        )

    expected_split = cfg.get("expected_split_counts", {})
    if int(expected_split.get("train", -1)) != 3645:
        raise ConfigValidationError(
            "config.expected_split_counts.train must be 3645; got "
            f"{expected_split.get('train')!r}"
        )
    if int(expected_split.get("val", -1)) != 450:
        raise ConfigValidationError(
            "config.expected_split_counts.val must be 450; got "
            f"{expected_split.get('val')!r}"
        )
    if int(expected_split.get("test", -1)) != 0:
        raise ConfigValidationError(
            "config.expected_split_counts.test must be 0; got "
            f"{expected_split.get('test')!r}"
        )

    expected_subjects = cfg.get("expected_subjects", {})
    if int(expected_subjects.get("train", -1)) != 81:
        raise ConfigValidationError(
            "config.expected_subjects.train must be 81; got "
            f"{expected_subjects.get('train')!r}"
        )
    if int(expected_subjects.get("val", -1)) != 10:
        raise ConfigValidationError(
            "config.expected_subjects.val must be 10; got "
            f"{expected_subjects.get('val')!r}"
        )
    if int(expected_subjects.get("test", -1)) != 0:
        raise ConfigValidationError(
            "config.expected_subjects.test must be 0; got "
            f"{expected_subjects.get('test')!r}"
        )

    lifecycle = cfg.get("lifecycle", {})
    _expect_keys("config.lifecycle", lifecycle, _REQUIRED_LIFECYCLE_KEYS)
    if sorted(lifecycle.get("valid_terminal_states", [])) != sorted(
        ["DONE", "FAILED", "STOPPED"]
    ):
        raise ConfigValidationError(
            "config.lifecycle.valid_terminal_states must be "
            "['DONE', 'FAILED', 'STOPPED']"
        )
    if sorted(lifecycle.get("exclusive_terminal_files", [])) != sorted(
        ["DONE.json", "FAILED.json", "STOPPED.json"]
    ):
        raise ConfigValidationError(
            "config.lifecycle.exclusive_terminal_files must be "
            "['DONE.json', 'FAILED.json', 'STOPPED.json']"
        )


def build_mini_config(
    cfg: Mapping[str, Any],
    *,
    b01_freeze_dir: str | None,
    data_root: str | None,
    config_path: str | None = None,
) -> MiniConfig:
    """Build a :class:`MiniConfig` from a validated JSON config."""

    training = cfg["training"]
    early = training["early_stopping"]
    rb = cfg["resource_budget"]
    expected_split = cfg.get("expected_split_counts", {})
    expected_subjects = cfg.get("expected_subjects", {})
    return MiniConfig(
        task_id=str(cfg["task_id"]),
        config_version=str(cfg["config_version"]),
        config_path=str(config_path) if config_path else "",
        candidates=tuple(str(c["name"]) for c in cfg["candidates"]),
        seed=int(training["seed"]),
        device=str(training["device"]),
        batch_size=int(training["batch_size"]),
        max_epochs=int(training["max_epochs"]),
        min_epochs=int(training["min_epochs"]),
        early_stopping_monitor=str(early["monitor"]),
        early_stopping_mode=str(early["mode"]),
        early_stopping_patience=int(early["patience"]),
        early_stopping_min_delta=float(early["min_delta"]),
        optimizer=str(training["optimizer"]),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
        num_workers=int(training["num_workers"]),
        resource_budget={
            "max_wall_minutes_per_candidate": int(rb["max_wall_minutes_per_candidate"]),
            "max_total_wall_minutes": int(rb["max_total_wall_minutes"]),
            "max_peak_cuda_mb": int(rb["max_peak_cuda_mb"]),
            "candidates_serial": bool(rb.get("candidates_serial", True)),
        },
        data_root=str(data_root) if data_root else "",
        b01_freeze_dir=str(b01_freeze_dir) if b01_freeze_dir else "",
        b01_a06_split_sha256_expected=str(cfg["b01_a06_split_sha256_expected"]),
        provenance=str(cfg["provenance"]),
        raw_semantics=str(cfg["raw_semantics"]),
        source_review_status=str(cfg["source_review_status"]),
        n_classes=int(cfg["dataset"]["n_classes"]),
        image_shape=tuple(int(v) for v in cfg["dataset"]["image_shape"]),
        max_parameters=int(B04_MAX_PARAMETERS),
        val_feasibility_threshold=float(cfg["feasibility_gate"]["b02_reference_val_fixed_iou"]),
        expected_train_count=int(expected_split.get("train", 3645)),
        expected_val_count=int(expected_split.get("val", 450)),
        expected_test_count=int(expected_split.get("test", 0)),
        expected_train_subjects=int(expected_subjects.get("train", 81)),
        expected_val_subjects=int(expected_subjects.get("val", 10)),
        expected_test_subjects=int(expected_subjects.get("test", 0)),
        expected_provenance=str(
            cfg.get("expected_provenance", "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED")
        ),
        expected_source_review_status=str(
            cfg.get("expected_source_review_status", "NOT_REVIEWED")
        ),
        expected_setting=str(cfg.get("expected_setting", "danaLab")),
        expected_cover=str(cfg.get("expected_cover", "uncover")),
    )


# ---------------------------------------------------------------------------
# Output directory safety
# ---------------------------------------------------------------------------


def check_output_dir_safety(output_dir: Path) -> None:
    """Refuse to write into an occupied output directory.

    The rule: if the output directory already exists and contains
    anything other than a ``.gitkeep`` marker, the runner raises
    :class:`OutputCollisionError`.  Sentinel files
    (``DONE.json`` / ``FAILED.json`` / ``STOPPED.json``) are detected
    explicitly so the error message points the operator at the right
    file rather than a generic "not empty" complaint.
    """

    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise OutputCollisionError(
            f"output path exists but is not a directory: {output_dir}"
        )
    sentinels = ("DONE.json", "FAILED.json", "STOPPED.json")
    for sentinel in sentinels:
        if (output_dir / sentinel).is_file():
            raise OutputCollisionError(
                f"output directory already contains {sentinel}; refusing to "
                f"overwrite.  Choose a fresh output_dir.  ({output_dir})"
            )
    contents = list(output_dir.iterdir())
    non_keep = [p for p in contents if p.name != ".gitkeep"]
    if non_keep:
        raise OutputCollisionError(
            f"output directory is not empty ({len(non_keep)} entries); "
            f"refusing to overwrite.  ({output_dir})"
        )


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Canonical array hash
# ---------------------------------------------------------------------------


CANONICAL_HASH_VERSION = "slp8_canonical_array_hash_v0.1"


def canonical_array_hash(arr: np.ndarray) -> str:
    """Stable SHA-256 of a label/prediction array (int64, shape-tagged)."""

    arr_int = np.ascontiguousarray(arr, dtype=np.int64)
    header = (
        f"{CANONICAL_HASH_VERSION}\n"
        f"dtype={arr_int.dtype.str}\n"
        f"shape={tuple(arr_int.shape)}\n"
    ).encode("utf-8")
    return hashlib.sha256(header + arr_int.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Seed / device helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/torch RNGs for determinism."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str, *, allow_cpu_fallback: bool) -> torch.device:
    """Resolve the requested device with optional CPU fallback.

    The B04 protocol requires ``device='cuda'`` for real runs and
    fail-closed if CUDA is not available.  The synthetic CPU smoke
    path is the only context where ``allow_cpu_fallback=True`` is
    allowed.
    """

    spec = str(requested).strip().lower()
    if spec in ("auto", ""):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu_fallback:
            return torch.device("cpu")
        raise MiniProtocolError(
            "device='auto' was requested without allow_cpu_fallback; "
            "B04 forbids silent CPU fallback for real runs."
        )
    if spec == "cpu":
        if allow_cpu_fallback:
            return torch.device("cpu")
        raise MiniProtocolError(
            "device='cpu' was requested for a real B01 run; "
            "B04 fail-closed forbids CPU for real Mini."
        )
    if spec == "cuda":
        if not torch.cuda.is_available():
            if allow_cpu_fallback:
                raise MiniProtocolError(
                    "device='cuda' was requested but CUDA is unavailable; "
                    "synthetic CPU smoke must explicitly request device='cpu'."
                )
            raise MiniProtocolError(
                "device='cuda' was requested but torch.cuda.is_available() is False; "
                "B04 protocol forbids silent CPU fallback."
            )
        return torch.device("cuda")
    raise MiniProtocolError(f"unknown device spec {requested!r}")


# ---------------------------------------------------------------------------
# Synthetic data path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticDataset:
    """A toy SLP8-style pressure+label dataset for synthetic CPU smoke."""

    samples: list[RegionSample]
    pressure_arrays: list[np.ndarray]
    label_arrays: list[np.ndarray]
    dataset_root: Path
    normalization: NormalizationStats
    train_class_stats: dict[str, Any]
    n_train_subjects: int
    n_val_subjects: int

    def __len__(self) -> int:  # pragma: no cover - cosmetic
        return len(self.samples)


def build_synthetic_dataset(
    *,
    n_train_samples: int,
    n_val_samples: int,
    seed: int,
    image_shape: tuple[int, int] = PRESSURE_SHAPE,
    n_train_subjects: int = 2,
    n_val_subjects: int = 1,
) -> tuple[
    "Slp8SyntheticDataset",
    "Slp8SyntheticDataset",
    dict[str, Any],
    dict[str, Any],
]:
    """Build deterministic synthetic (pressure, label) pairs.

    The synthetic pressure is a tiny float32 in the raw-pmarray-response
    semantics (no normalization applied beyond dtype/channel).  The
    label map is constructed so that every class 0..8 has at least one
    pixel in TRAIN; the per-class pixel ratio is recorded in the
    returned train_class_stats.
    """

    rng = np.random.default_rng(seed)
    # Use a per-run tempdir for the synthetic NPY cache so the repo
    # never carries these build artifacts.  ``tempfile.mkdtemp``
    # returns a unique directory; the OS will clean it up eventually.
    import tempfile
    cache_root = Path(tempfile.mkdtemp(prefix="b04_synth_cache_"))
    pressure_dir = cache_root / "pressure"
    label_dir = cache_root / "labels"
    pressure_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    train_subjects = [f"{idx:05d}" for idx in range(1, n_train_subjects + 1)]
    val_subjects = [f"{idx:05d}" for idx in range(n_train_subjects + 1, n_train_subjects + n_val_subjects + 1)]

    train_samples: list[RegionSample] = []
    val_samples: list[RegionSample] = []
    train_pressures: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    val_pressures: list[np.ndarray] = []
    val_labels: list[np.ndarray] = []

    per_class_pixel_count: Counter[int] = Counter()
    n_train_pixels = 0

    for split, subjects, count, pressure_buf, label_buf, sample_list in (
        ("train", train_subjects, n_train_samples, train_pressures, train_labels, train_samples),
        ("val", val_subjects, n_val_samples, val_pressures, val_labels, val_samples),
    ):
        for i in range(count):
            subject = subjects[i % len(subjects)]
            sample_id = f"SLP:danaLab:{subject}:uncover:{i:06d}"
            # Pressure: small positive float (raw PMarray response).  Use
            # float64 to match the B01 contract and rely on the
            # raw_passthrough normalization to cast to float32.
            pressure = rng.uniform(0.0, 1.0, size=image_shape).astype(np.float64)
            # Label: start with mostly background, sprinkle each foreground
            # class as a small rectangular block.  Deterministic per (i, subject).
            label = np.zeros(image_shape, dtype=np.int64)
            for cid in range(1, N_CLASSES):
                seed_jitter = (hash((split, subject, i, cid)) % 13) + 4
                h0 = (cid * 11 + i * 3) % (image_shape[0] - seed_jitter - 1)
                w0 = (cid * 17 + i * 5) % (image_shape[1] - seed_jitter - 1)
                label[h0:h0 + seed_jitter, w0:w0 + seed_jitter] = cid
            # Ensure every class 0..8 is present at least once in TRAIN
            # so the class weight derivation has positive coverage.
            if split == "train":
                label[0, 0] = 0
                for cid in range(1, N_CLASSES):
                    label[cid, 1] = cid
                # Recompute per-class counts for TRAIN-only stats.
                unique, counts = np.unique(label, return_counts=True)
                for cid, cnt in zip(unique.tolist(), counts.tolist()):
                    per_class_pixel_count[cid] += int(cnt)
                n_train_pixels += int(label.size)

            # Persist to disk so the dataset follows the same lazy-load
            # path as the real B01 dataset.  The paths stored on the
            # ``RegionSample`` are RELATIVE to ``cache_root`` so the
            # synthetic dataset can resolve them via its
            # ``dataset_root / sample.label_path`` contract.
            pressure_relpath = f"pressure/{split}_{i:06d}.npy"
            label_relpath = f"labels/{split}_{i:06d}.npy"
            pressure_path = cache_root / pressure_relpath
            label_path = cache_root / label_relpath
            np.save(pressure_path, pressure)
            np.save(label_path, label)
            sample = RegionSample(
                sample_id=sample_id,
                subject_id=subject,
                ml_split=split,
                posture=["SUPINE", "LEFT", "RIGHT"][i % 3],
                pressure_path=pressure_relpath,
                label_path=label_relpath,
                onehot_path=label_relpath,
            )
            sample_list.append(sample)
            pressure_buf.append(pressure)
            label_buf.append(label)

    n_train_pixels_total = n_train_pixels if n_train_pixels > 0 else 1
    per_class_pixel_ratio: dict[int, float] = {}
    for cid in range(N_CLASSES):
        per_class_pixel_ratio[cid] = float(per_class_pixel_count.get(cid, 0)) / float(n_train_pixels_total)

    normalization = NormalizationStats(
        global_min=0.0,
        global_max=1.0,
        global_mean=0.5,
        global_std=0.28867513,
        method="raw_passthrough_with_minmax_reference",
        raw_semantics="raw_pmarray_response",
        fit_split="train",
        epsilon=1e-12,
    )

    train_class_stats = {
        "n_samples": int(n_train_samples),
        "n_pixels": int(n_train_pixels),
        "subject_count": int(n_train_subjects),
        "per_class_pixel_ratio": {str(k): float(v) for k, v in per_class_pixel_ratio.items()},
    }

    train_dataset = Slp8SyntheticDataset(
        samples=train_samples,
        pressure_arrays=train_pressures,
        dataset_root=cache_root,
        normalization=normalization,
    )
    val_dataset = Slp8SyntheticDataset(
        samples=val_samples,
        pressure_arrays=val_pressures,
        dataset_root=cache_root,
        normalization=normalization,
    )

    manifest = {
        "train_subjects": list(train_subjects),
        "val_subjects": list(val_subjects),
        "n_train_samples": int(n_train_samples),
        "n_val_samples": int(n_val_samples),
        "n_test_samples": 0,
        "seed": int(seed),
        "normalization_method": normalization.method,
        "normalization_fit_split": normalization.fit_split,
        "synthetic": True,
    }
    return train_dataset, val_dataset, manifest, train_class_stats


class Slp8SyntheticDataset(Dataset):
    """In-memory synthetic pressure dataset that mirrors Slp8RegionDataset."""

    def __init__(
        self,
        *,
        samples: Sequence[RegionSample],
        pressure_arrays: Sequence[np.ndarray],
        dataset_root: Path,
        normalization: NormalizationStats,
    ) -> None:
        if len(samples) != len(pressure_arrays):
            raise MiniProtocolError(
                f"samples and pressure_arrays length mismatch: "
                f"{len(samples)} vs {len(pressure_arrays)}"
            )
        self._samples = list(samples)
        self._pressure_arrays = list(pressure_arrays)
        self._dataset_root = Path(dataset_root)
        self._normalization = normalization

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index]
        pressure = np.asarray(self._pressure_arrays[index], dtype=np.float64)
        if pressure.shape != PRESSURE_SHAPE:
            raise MiniProtocolError(
                f"Synthetic pressure shape mismatch: expected {PRESSURE_SHAPE}, "
                f"got {pressure.shape}"
            )
        pressure_input = self._normalization.apply(pressure)
        label_path = self._dataset_root / sample.label_path
        label = np.load(label_path, mmap_mode=None, allow_pickle=False)
        if label.shape != PRESSURE_SHAPE:
            raise MiniProtocolError(
                f"Synthetic label shape mismatch: expected {PRESSURE_SHAPE}, "
                f"got {label.shape}"
            )
        if label.dtype != np.int64:
            label = label.astype(np.int64)
        if not ((label >= 0) & (label < N_CLASSES)).all():
            raise MiniProtocolError(
                f"Synthetic label values out of range [0, {N_CLASSES - 1}]"
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
# Per-candidate runner
# ---------------------------------------------------------------------------


@dataclass
class EpochMetricsRow:
    """One row of the ``epoch_metrics.csv`` table."""

    candidate: str
    epoch: int
    train_loss: float
    val_loss: float
    is_best: bool
    elapsed_seconds: float


@dataclass
class PredictionRecord:
    """One per-sample prediction record for the predictions_manifest."""

    candidate: str
    split: str
    sample_id: str
    subject_id: str
    posture: str
    label_sha256: str
    prediction_sha256: str
    label_shape: tuple[int, int]
    prediction_shape: tuple[int, int]
    failure_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "split": self.split,
            "sample_id": self.sample_id,
            "subject_id": self.subject_id,
            "posture": self.posture,
            "label_sha256": self.label_sha256,
            "prediction_sha256": self.prediction_sha256,
            "label_shape": str(self.label_shape),
            "prediction_shape": str(self.prediction_shape),
            "failure_reason": self.failure_reason,
        }


@dataclass
class CandidateMetrics:
    """Per-candidate aggregate metrics (the 'final' snapshot)."""

    fixed_foreground_macro_iou: float
    fixed_foreground_macro_dice: float
    pixel_accuracy: float
    background_iou: float
    val_loss: float
    per_region: list[dict[str, Any]]
    per_posture: dict[str, dict[str, Any]]
    per_subject: dict[str, dict[str, Any]]
    worst_subject: dict[str, Any] | None
    confusion_matrix: np.ndarray
    centroid_error_summary: dict[str, Any]
    n_samples: int
    n_test_samples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fixed_foreground_macro_iou": float(self.fixed_foreground_macro_iou),
            "fixed_foreground_macro_dice": float(self.fixed_foreground_macro_dice),
            "pixel_accuracy": float(self.pixel_accuracy),
            "background_iou": float(self.background_iou),
            "val_loss": float(self.val_loss),
            "per_region": list(self.per_region),
            "per_posture": {k: v for k, v in self.per_posture.items()},
            "per_subject": {k: v for k, v in self.per_subject.items()},
            "worst_subject": dict(self.worst_subject) if self.worst_subject is not None else None,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "centroid_error_summary": dict(self.centroid_error_summary),
            "n_samples": int(self.n_samples),
            "n_test_samples": int(self.n_test_samples),
        }


@dataclass
class CandidateResult:
    """Aggregated result for one B04 candidate."""

    candidate: str
    model_version: str
    parameter_count: int
    parameter_count_within_budget: bool
    feasibility: str  # "FEASIBLE" | "NOT_FEASIBLE" | "STOPPED" | "FAILED"
    reason: str
    epoch_metrics: list[EpochMetricsRow]
    metrics: CandidateMetrics | None
    train_predictions: list[np.ndarray]
    train_labels: list[np.ndarray]
    train_subjects: list[str]
    val_predictions: list[np.ndarray]
    val_labels: list[np.ndarray]
    val_subjects: list[str]
    train_records: list[PredictionRecord]
    val_records: list[PredictionRecord]
    best_epoch: int | None
    best_val_loss: float | None
    best_prediction_hash: str | None
    in_process_prediction_hash: str | None
    reload_consistent: bool
    reload_max_abs_diff: float | None
    checkpoint_best_sha256: str | None
    checkpoint_last_sha256: str | None
    train_loss_history: list[float]
    val_loss_history: list[float]
    train_subject_overlap_with_val: bool
    val_subject_overlap_with_train: bool
    n_test_samples: int
    param_changed: bool
    last_in_process_prediction_hash: str | None
    class_weight_summary: dict[str, Any]
    elapsed_seconds: float
    budget_status: str  # "ok" | "per_candidate_wall_exceeded" | "total_wall_exceeded" | "cuda_peak_exceeded"
    budget_report: dict[str, Any]
    budget_thresholds: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "model_version": self.model_version,
            "parameter_count": int(self.parameter_count),
            "parameter_count_within_budget": bool(self.parameter_count_within_budget),
            "feasibility": self.feasibility,
            "reason": self.reason,
            "epoch_metrics": [asdict(row) for row in self.epoch_metrics],
            "metrics": self.metrics.as_dict() if self.metrics is not None else None,
            "best_epoch": int(self.best_epoch) if self.best_epoch is not None else None,
            "best_val_loss": float(self.best_val_loss) if self.best_val_loss is not None else None,
            "best_prediction_hash": self.best_prediction_hash,
            "in_process_prediction_hash": self.in_process_prediction_hash,
            "last_in_process_prediction_hash": self.last_in_process_prediction_hash,
            "reload_consistent": bool(self.reload_consistent),
            "reload_max_abs_diff": float(self.reload_max_abs_diff) if self.reload_max_abs_diff is not None else None,
            "checkpoint_best_sha256": self.checkpoint_best_sha256,
            "checkpoint_last_sha256": self.checkpoint_last_sha256,
            "train_loss_history": [float(v) for v in self.train_loss_history],
            "val_loss_history": [float(v) for v in self.val_loss_history],
            "train_subject_overlap_with_val": bool(self.train_subject_overlap_with_val),
            "val_subject_overlap_with_train": bool(self.val_subject_overlap_with_train),
            "n_test_samples": int(self.n_test_samples),
            "param_changed": bool(self.param_changed),
            "class_weight_summary": dict(self.class_weight_summary),
            "elapsed_seconds": float(self.elapsed_seconds),
            "budget_status": str(self.budget_status),
            "budget_report": dict(self.budget_report),
            "budget_thresholds": dict(self.budget_thresholds),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


# ---------------------------------------------------------------------------
# Checkpoint payload
# ---------------------------------------------------------------------------


CHECKPOINT_VERSION = "slp8_region_mini_v0.1"


def _build_checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    seed: int,
    model_config: Mapping[str, Any],
    class_weight_summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
    n_classes: int,
    image_shape: tuple[int, int],
    stopper: "_EarlyStopper | None" = None,
    train_loss_history: Sequence[float] | None = None,
    val_loss_history: Sequence[float] | None = None,
    epoch_metrics: Sequence[EpochMetricsRow] | None = None,
    identity: CheckpointIdentity | None = None,
    input_manifest_hashes: Mapping[str, Any] | None = None,
    rng_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a versioned B04 checkpoint payload.

    R02 expands the payload with the full B04 identity block,
    the early-stopper state, the metric history, the input-hash
    snapshot, and the captured RNG state.  The payload is
    ``weights_only=True`` safe because every value is either a
    primitive, a ``numpy`` / ``torch`` tensor, or a JSON-safe dict /
    list.
    """

    payload: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": {k: v for k, v in optimizer.state_dict().items()},
        "epoch": int(epoch),
        "seed": int(seed),
        "model_config": dict(model_config),
        "class_weight_summary": dict(class_weight_summary),
        "metrics": dict(metrics),
        "n_classes": int(n_classes),
        "image_shape": list(image_shape),
    }
    if identity is not None:
        payload["identity"] = identity.as_dict()
    if stopper is not None:
        payload["early_stopper"] = stopper.snapshot().as_dict()
    if train_loss_history is not None:
        payload["train_loss_history"] = [float(v) for v in train_loss_history]
    if val_loss_history is not None:
        payload["val_loss_history"] = [float(v) for v in val_loss_history]
    if epoch_metrics is not None:
        payload["epoch_metrics"] = [asdict(row) for row in epoch_metrics]
    if input_manifest_hashes is not None:
        payload["input_manifest_hashes"] = dict(input_manifest_hashes)
    if rng_state is not None:
        payload["rng_state"] = dict(rng_state)
    return payload


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return file_sha256(path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "pressure": batch["pressure"].to(device),
        "label": batch["label"].to(device),
        "sample_id": batch["sample_id"],
        "subject_id": batch["subject_id"],
        "posture": batch["posture"],
        "ml_split": batch["ml_split"],
    }


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = _to_device(batch, device)
        logits = model(batch["pressure"])
        B, C, H, W = logits.shape
        logits_flat = logits.reshape(B, C, H * W)
        label_flat = batch["label"].reshape(B, H * W)
        loss = loss_fn(logits_flat, label_flat)
        if not torch.isfinite(loss):
            raise NonFiniteMetricsError(
                f"non-finite training loss: {loss.detach().item()!r}"
            )
        optimizer.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise NonFiniteMetricsError("non-finite gradient detected")
        optimizer.step()
        total += float(loss.detach().cpu().item()) * int(B)
        n += int(B)
    if n == 0:
        raise NonFiniteMetricsError("train loader produced 0 samples")
    return float(total / n)


def _validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, list[np.ndarray], list[np.ndarray], list[str], list[str], list[str]]:
    model.eval()
    total = 0.0
    n = 0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    sample_ids: list[str] = []
    subject_ids: list[str] = []
    postures: list[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            logits = model(batch["pressure"])
            B, C, H, W = logits.shape
            logits_flat = logits.reshape(B, C, H * W)
            label_flat = batch["label"].reshape(B, H * W)
            loss = loss_fn(logits_flat, label_flat)
            if not torch.isfinite(loss):
                raise NonFiniteMetricsError(
                    f"non-finite validation loss: {loss.detach().item()!r}"
                )
            total += float(loss.detach().cpu().item()) * int(B)
            n += int(B)
            preds = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
            labels = batch["label"].cpu().numpy().astype(np.int64)
            for i in range(B):
                all_preds.append(preds[i])
                all_labels.append(labels[i])
                sample_ids.append(str(batch["sample_id"][i]))
                subject_ids.append(str(batch["subject_id"][i]))
                postures.append(str(batch["posture"][i]))
    if n == 0:
        raise NonFiniteMetricsError("validation loader produced 0 samples")
    return float(total / n), all_labels, all_preds, sample_ids, subject_ids, postures


def _predictions_hash(preds: Sequence[np.ndarray]) -> str:
    """Aggregate canonical hash over an ordered list of (H, W) predictions."""

    h = hashlib.sha256()
    h.update(f"{CANONICAL_HASH_VERSION}\n".encode("utf-8"))
    h.update(f"n={len(preds)}\n".encode("utf-8"))
    for i, arr in enumerate(preds):
        arr_int = np.ascontiguousarray(arr, dtype=np.int64)
        header = f"sample={i}\nshape={tuple(arr_int.shape)}\n".encode("utf-8")
        h.update(header + arr_int.tobytes())
    return h.hexdigest()


def _build_prediction_records(
    *,
    candidate: str,
    split: str,
    labels: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    sample_ids: Sequence[str],
    subject_ids: Sequence[str],
    postures: Sequence[str],
) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for lab, pred, sid, sub, pos in zip(labels, predictions, sample_ids, subject_ids, postures):
        if not np.isfinite(lab.astype(np.float64)).all():
            failure = "non_finite_label"
        elif not ((lab >= 0) & (lab < N_CLASSES)).all():
            failure = "label_out_of_range"
        else:
            failure = "ok"
        records.append(PredictionRecord(
            candidate=candidate,
            split=split,
            sample_id=str(sid),
            subject_id=str(sub),
            posture=str(pos),
            label_sha256=canonical_array_hash(lab),
            prediction_sha256=canonical_array_hash(pred),
            label_shape=tuple(lab.shape),
            prediction_shape=tuple(pred.shape),
            failure_reason=failure,
        ))
    return records


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class _EarlyStopper:
    """Minimal early-stopping bookkeeping with mode='min' / monitor='val_loss'."""

    def __init__(
        self,
        *,
        monitor: str,
        mode: str,
        patience: int,
        min_delta: float,
        min_epochs: int,
    ) -> None:
        if monitor != "val_loss":
            raise MiniProtocolError(
                f"early stopping monitor must be 'val_loss'; got {monitor!r}"
            )
        if mode != "min":
            raise MiniProtocolError(
                f"early stopping mode must be 'min'; got {mode!r}"
            )
        if patience < 0:
            raise MiniProtocolError("patience must be >= 0")
        if min_epochs < 1:
            raise MiniProtocolError("min_epochs must be >= 1")
        if min_delta < 0 or not math.isfinite(min_delta):
            raise MiniProtocolError("min_delta must be a finite non-negative value")
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_epochs = int(min_epochs)
        self.best: float | None = None
        self.best_epoch: int | None = None
        self._patience: int = 0

    def step(self, epoch: int, value: float) -> tuple[bool, bool]:
        if not math.isfinite(value):
            raise NonFiniteMetricsError(
                f"monitor value is non-finite at epoch {epoch}: {value!r}"
            )
        if self.best is None:
            is_best = True
        else:
            is_best = value < self.best - self.min_delta
        if is_best:
            self.best = float(value)
            self.best_epoch = int(epoch)
            self._patience = 0
        else:
            self._patience += 1
        should_stop = (
            epoch >= self.min_epochs
            and not is_best
            and self._patience >= self.patience
        )
        return is_best, should_stop

    def snapshot(self) -> EarlyStopperState:
        return EarlyStopperState(
            best_metric=self.best,
            best_epoch=self.best_epoch,
            patience=self.patience,
            min_delta=self.min_delta,
            min_epochs=self.min_epochs,
            mode=self.mode,
            monitor=self.monitor,
        )

    def restore(self, state: EarlyStopperState) -> None:
        if state.monitor != "val_loss" or state.mode != "min":
            raise ResumeIdentityError(
                "early-stopper identity mismatch on resume: "
                f"monitor={state.monitor}, mode={state.mode}"
            )
        if (
            state.patience != self.patience
            or state.min_epochs != self.min_epochs
            or state.min_delta != self.min_delta
        ):
            raise ResumeIdentityError(
                "early-stopper hyperparameter mismatch on resume"
            )
        self.best = state.best_metric
        self.best_epoch = state.best_epoch
        # _patience is implicit from history (no need to restore).
        self._patience = 0


# ---------------------------------------------------------------------------
# Single-candidate runner
# ---------------------------------------------------------------------------


def run_one_candidate(
    *,
    candidate_name: str,
    config: MiniConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    class_weight_result: ClassWeightResult,
    output_dir: Path,
    device: torch.device,
    budget_state: ResourceBudgetState,
    identity: CheckpointIdentity,
    input_manifest_hashes: Mapping[str, Any],
    deterministic: DeterminismSettings,
    resume_from: Path | None = None,
) -> CandidateResult:
    """Train one B04 candidate end-to-end and write its outputs.

    The runner always writes ``checkpoints/<candidate>/last.pt`` after
    each epoch and overwrites ``checkpoints/<candidate>/best.pt`` only
    when ``val_loss`` improves (mode='min', earliest-epoch tie-break).
    Independent reload of ``best.pt`` is verified and a canonical hash
    of its predictions is compared to the in-process predictions.

    The runner checks the resource budget (wall-clock and CUDA peak
    memory) at every validation epoch; any exceedance transitions the
    candidate to ``STOPPED`` and the run never writes ``DONE.json``.

    When ``resume_from`` points at a previous ``last.pt``, the runner
    restores the model / optimizer / RNG / early-stopper state and
    history, then resumes from the next epoch.  The saved
    :class:`CheckpointIdentity` is compared against the requested
    identity; any mismatch raises :class:`ResumeIdentityError`.
    """

    apply_settings(config.seed, cpu_threads=deterministic.cpu_threads)

    # ------------------------------------------------------------------
    # Refuse to resume a closed (DONE) experiment.
    # ------------------------------------------------------------------
    refuse_resume_for_done_run(output_dir)

    builder = get_model_builder(candidate_name)
    model, model_config = builder.factory(N_CLASSES, str(device))
    if model_config.get("device") != str(device):
        model_config["device"] = str(device)
    if model_config.get("n_classes") != N_CLASSES:
        raise MiniProtocolError(
            f"candidate {candidate_name} reported n_classes={model_config.get('n_classes')}; "
            f"expected {N_CLASSES}"
        )
    parameter_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if parameter_count > B04_MAX_PARAMETERS:
        raise MiniProtocolError(
            f"candidate {candidate_name} has {parameter_count} parameters; "
            f"exceeds B04 cap of {B04_MAX_PARAMETERS}"
        )

    weight_tensor = torch.from_numpy(
        class_weights_to_tensor(class_weight_result)
    ).to(device).to(torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    train_loader = _make_loader(
        train_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
    )
    val_loader = _make_loader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
    )

    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    checkpoint_dir = output_dir / "checkpoints" / candidate_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"

    stopper = _EarlyStopper(
        monitor=config.early_stopping_monitor,
        mode=config.early_stopping_mode,
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        min_epochs=config.min_epochs,
    )

    # ------------------------------------------------------------------
    # Budget + identity setup
    # ------------------------------------------------------------------
    budget_state.begin_candidate()
    budget_state.reset_cuda_peak()

    epoch_metrics: list[EpochMetricsRow] = []
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_train_preds: list[np.ndarray] | None = None
    best_val_preds: list[np.ndarray] | None = None
    best_train_labels: list[np.ndarray] | None = None
    best_val_labels: list[np.ndarray] | None = None
    best_train_subjects: list[str] | None = None
    best_train_postures: list[str] | None = None
    best_val_subjects: list[str] | None = None
    best_val_postures: list[str] | None = None
    best_train_sample_ids: list[str] | None = None
    best_val_sample_ids: list[str] | None = None
    best_epoch: int | None = None
    best_val_loss: float | None = None
    best_sha: str | None = None
    last_sha: str | None = None
    feasibility: str = "FAILED"
    reason: str = "not_run"
    stopped_early: bool = False
    budget_status: str = "ok"
    budget_report: dict[str, Any] = {}
    budget_thresholds: dict[str, Any] = budget_state.budget.as_dict()

    start_epoch: int = 1
    resume_checkpoint_payload: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Optional resume
    # ------------------------------------------------------------------
    if resume_from is not None:
        if not Path(resume_from).is_file():
            raise MiniProtocolError(
                f"resume_from path does not exist: {resume_from}"
            )
        payload = _load_checkpoint(Path(resume_from))
        saved_identity = identity_from_dict(payload)
        verify_resume_identity(saved_identity, identity)
        try:
            model.load_state_dict(payload["model_state_dict"])
        except Exception as exc:
            raise ResumeIdentityError(
                f"resume model state_dict load failed: {exc}"
            )
        try:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        except Exception as exc:
            raise ResumeIdentityError(
                f"resume optimizer state_dict load failed: {exc}"
            )
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            try:
                restore_rng_state(rng_state)
            except Exception as exc:
                raise ResumeIdentityError(
                    f"resume RNG state restore failed: {exc}"
                )
        if "early_stopper" in payload:
            stopper.restore(EarlyStopperState.from_dict(payload["early_stopper"]))
        train_loss_history = [float(x) for x in payload.get("train_loss_history", [])]
        val_loss_history = [float(x) for x in payload.get("val_loss_history", [])]
        epoch_metrics = [
            EpochMetricsRow(**row) for row in payload.get("epoch_metrics", [])
        ]
        resume_checkpoint_payload = payload
        start_epoch = int(payload.get("epoch", 0)) + 1
        # best.pt may have been written previously; re-load its SHA so
        # the manifest remains accurate.
        if payload.get("is_best"):
            best_sha = file_sha256(best_path) if best_path.is_file() else None
            best_epoch = int(payload.get("epoch"))
            best_val_loss = float(payload.get("metrics", {}).get("val_loss")) if payload.get("metrics", {}).get("val_loss") is not None else None
        last_sha = file_sha256(last_path) if last_path.is_file() else None

    budget_exceeded: bool = False
    budget_check_history: list[dict[str, Any]] = []

    if start_epoch > config.max_epochs:
        # Resumed past the last epoch; treat as completion and proceed
        # to the metrics / reload / decision stage.
        stopped_early = True

    for epoch in range(start_epoch, config.max_epochs + 1):
        t_epoch = time.perf_counter()
        try:
            train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_labels, val_preds, val_sids, val_subjs, val_pos = _validate(
                model, val_loader, loss_fn, device
            )
        except NonFiniteMetricsError as exc:
            feasibility = "FAILED"
            reason = f"non-finite metric: {exc}"
            return _build_candidate_result(
                candidate_name=candidate_name,
                model_version=builder.version,
                parameter_count=parameter_count,
                feasibility=feasibility,
                reason=reason,
                epoch_metrics=epoch_metrics,
                train_loss_history=train_loss_history,
                val_loss_history=val_loss_history,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                best_sha=best_sha,
                last_sha=last_sha,
                param_changed=False,
                train_class_weight_result=class_weight_result,
                reload_consistent=False,
                reload_max_abs_diff=None,
                elapsed_seconds=budget_state.elapsed_seconds,
                budget_status=budget_status,
                budget_report=budget_report,
                budget_thresholds=budget_thresholds,
            )

        # ------------------------------------------------------------------
        # Resource budget check (after every validation epoch)
        # ------------------------------------------------------------------
        budget_state.update_cuda_peak()
        budget_check = budget_state.check()
        budget_check_history.append(budget_check.as_dict())
        if budget_check.exceeded:
            budget_status = budget_check.reason
            budget_report = budget_check.as_dict()
            feasibility = "STOPPED"
            reason = f"resource budget exceeded: {budget_check.reason}"
            budget_exceeded = True
            break

        train_loss_history.append(float(train_loss))
        val_loss_history.append(float(val_loss))

        is_best, should_stop = stopper.step(epoch, val_loss)
        elapsed = float(time.perf_counter() - t_epoch)

        epoch_metrics.append(EpochMetricsRow(
            candidate=candidate_name,
            epoch=epoch,
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            is_best=bool(is_best),
            elapsed_seconds=elapsed,
        ))

        payload = _build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            seed=config.seed,
            model_config=model_config,
            class_weight_summary=class_weight_result.as_dict(),
            metrics={
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "is_best": bool(is_best),
            },
            n_classes=N_CLASSES,
            image_shape=PRESSURE_SHAPE,
            stopper=stopper,
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
            epoch_metrics=epoch_metrics,
            identity=identity,
            input_manifest_hashes=dict(input_manifest_hashes),
            rng_state=capture_rng_state(),
        )
        last_sha = _save_checkpoint(last_path, payload)

        if is_best:
            best_sha = _save_checkpoint(best_path, payload)
            best_epoch = epoch
            best_val_loss = float(val_loss)
            best_val_preds = list(val_preds)
            best_val_labels = list(val_labels)
            best_val_subjects = list(val_subjs)
            best_val_postures = list(val_pos)
            best_val_sample_ids = list(val_sids)
            # Re-run training inference so the predictions_manifest
            # carries real per-sample evidence for the best epoch.
            (
                _t_loss,
                train_labels_epoch,
                train_preds_epoch,
                train_sids,
                train_subjs,
                train_pos,
            ) = _validate(model, train_loader, loss_fn, device)
            best_train_preds = list(train_preds_epoch)
            best_train_labels = list(train_labels_epoch)
            best_train_subjects = list(train_subjs)
            best_train_postures = list(train_pos)
            best_train_sample_ids = list(train_sids)

        if should_stop:
            stopped_early = True
            break

    if not budget_exceeded:
        budget_state.update_cuda_peak()
        final_check = budget_state.check()
        budget_check_history.append(final_check.as_dict())
        if final_check.exceeded:
            budget_status = final_check.reason
            budget_report = final_check.as_dict()
            budget_exceeded = True

    if budget_exceeded:
        return _build_candidate_result(
            candidate_name=candidate_name,
            model_version=builder.version,
            parameter_count=parameter_count,
            feasibility="STOPPED",
            reason=reason,
            epoch_metrics=epoch_metrics,
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_sha=best_sha,
            last_sha=last_sha,
            param_changed=True,
            train_class_weight_result=class_weight_result,
            reload_consistent=False,
            reload_max_abs_diff=None,
            elapsed_seconds=budget_state.elapsed_seconds,
            budget_status=budget_status,
            budget_report=budget_report,
            budget_thresholds=budget_thresholds,
        )

    if not budget_report:
        budget_state.update_cuda_peak()
        budget_report = budget_state.check().as_dict()
        budget_status = budget_report["reason"]

    # ------------------------------------------------------------------
    # Independent reload of best.pt
    # ------------------------------------------------------------------
    best_loaded = _load_checkpoint(best_path)
    if str(best_loaded.get("version", "")) != CHECKPOINT_VERSION:
        raise MiniProtocolError(
            f"best checkpoint version mismatch: {best_loaded.get('version')!r}"
        )
    fresh_model, _ = builder.factory(N_CLASSES, "cpu")
    fresh_model.load_state_dict(
        {k: v.float() if hasattr(v, "float") else v for k, v in best_loaded["model_state_dict"].items()}
    )
    fresh_model = fresh_model.to("cpu")
    fresh_model.eval()

    val_subset_indices = list(range(min(2, len(val_dataset))))
    if val_subset_indices:
        ref_pressure = torch.stack(
            [val_dataset[i]["pressure"] for i in val_subset_indices]
        )
        with torch.no_grad():
            current_logits = model.to("cpu")(ref_pressure)
            fresh_logits = fresh_model(ref_pressure)
        max_abs_diff = float((current_logits - fresh_logits).abs().max().item())
        reload_consistent = bool(
            torch.allclose(current_logits, fresh_logits, rtol=1e-5, atol=1e-6)
        )
    else:
        max_abs_diff = 0.0
        reload_consistent = True

    fresh_model = fresh_model.to(device)
    reloaded_val_loader = _make_loader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0,
    )
    with torch.no_grad():
        reloaded_preds: list[np.ndarray] = []
        for batch in reloaded_val_loader:
            pressure = batch["pressure"].to(device)
            logits = fresh_model(pressure)
            reloaded_preds.extend(
                [p.cpu().numpy().astype(np.int64) for p in logits.argmax(dim=1)]
            )
    reloaded_hash = _predictions_hash(reloaded_preds)
    in_process_hash = _predictions_hash(best_val_preds) if best_val_preds else None
    hash_consistent = bool(in_process_hash is not None and reloaded_hash == in_process_hash)

    final_state_diff = float(sum(
        (current.float() - initial_state[k].float()).pow(2).sum().sqrt().item()
        for k, current in model.state_dict().items()
    ))
    param_changed = final_state_diff > 1e-6

    if (
        best_val_preds is None
        or best_val_labels is None
        or best_val_subjects is None
        or best_val_postures is None
    ):
        feasibility = "FAILED"
        reason = "best checkpoint missing predictions"
        metrics: CandidateMetrics | None = None
    else:
        bundle = compute_extended_metrics(
            labels=best_val_labels,
            predictions=best_val_preds,
            subject_ids=best_val_subjects,
            postures=best_val_postures,
        )
        metrics = CandidateMetrics(
            fixed_foreground_macro_iou=float(bundle.fixed_foreground_macro_iou),
            fixed_foreground_macro_dice=float(bundle.fixed_foreground_macro_dice),
            pixel_accuracy=float(bundle.pixel_accuracy),
            background_iou=float(bundle.background_iou),
            val_loss=float(best_val_loss) if best_val_loss is not None else 0.0,
            per_region=bundle.per_region,
            per_posture=bundle.per_posture,
            per_subject=bundle.per_subject,
            worst_subject=bundle.worst_subject,
            confusion_matrix=bundle.confusion_matrix,
            centroid_error_summary=bundle.centroid_error_summary.as_dict(),
            n_samples=int(bundle.n_samples),
            n_test_samples=0,
        )

    if metrics is None:
        feasibility = "FAILED"
        reason = "no best metrics"
    else:
        if not reload_consistent or not hash_consistent:
            feasibility = "FAILED"
            reason = (
                f"reload consistency failed (max_abs_diff={max_abs_diff:.3e}, "
                f"hash_match={hash_consistent})"
            )
        elif not param_changed:
            feasibility = "FAILED"
            reason = "parameters did not change during training"
        elif not math.isfinite(metrics.fixed_foreground_macro_iou):
            feasibility = "FAILED"
            reason = "non-finite fixed foreground macro IoU"
        elif not stopped_early and len(val_loss_history) < config.min_epochs:
            feasibility = "FAILED"
            reason = f"fewer than min_epochs={config.min_epochs} completed"
        else:
            threshold = config.val_feasibility_threshold
            if metrics.fixed_foreground_macro_iou >= threshold:
                feasibility = "FEASIBLE"
                reason = (
                    f"VAL fixed foreground macro IoU="
                    f"{metrics.fixed_foreground_macro_iou:.6f} >= "
                    f"{threshold:.6f}"
                )
            else:
                feasibility = "NOT_FEASIBLE"
                reason = (
                    f"VAL fixed foreground macro IoU="
                    f"{metrics.fixed_foreground_macro_iou:.6f} < "
                    f"{threshold:.6f}"
                )

    train_records: list[PredictionRecord] = []
    val_records: list[PredictionRecord] = []
    if best_train_preds is not None and best_train_labels is not None:
        train_records = _build_prediction_records(
            candidate=candidate_name,
            split="train",
            labels=best_train_labels,
            predictions=best_train_preds,
            sample_ids=best_train_sample_ids or [],
            subject_ids=best_train_subjects or [],
            postures=best_train_postures or [],
        )
    if best_val_preds is not None and best_val_labels is not None:
        val_records = _build_prediction_records(
            candidate=candidate_name,
            split="val",
            labels=best_val_labels,
            predictions=best_val_preds,
            sample_ids=best_val_sample_ids or [],
            subject_ids=best_val_subjects or [],
            postures=best_val_postures or [],
        )

    train_subjs_set = set(best_train_subjects or [])
    val_subjs_set = set(best_val_subjects or [])
    train_overlap = bool(train_subjs_set & val_subjs_set)
    val_overlap = bool(val_subjs_set & train_subjs_set)

    return CandidateResult(
        candidate=candidate_name,
        model_version=builder.version,
        parameter_count=parameter_count,
        parameter_count_within_budget=bool(parameter_count <= B04_MAX_PARAMETERS),
        feasibility=feasibility,
        reason=reason,
        epoch_metrics=epoch_metrics,
        metrics=metrics,
        train_predictions=list(best_train_preds) if best_train_preds else [],
        train_labels=list(best_train_labels) if best_train_labels else [],
        train_subjects=list(best_train_subjects) if best_train_subjects else [],
        val_predictions=list(best_val_preds) if best_val_preds else [],
        val_labels=list(best_val_labels) if best_val_labels else [],
        val_subjects=list(best_val_subjects) if best_val_subjects else [],
        train_records=train_records,
        val_records=val_records,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_prediction_hash=reloaded_hash,
        in_process_prediction_hash=in_process_hash,
        reload_consistent=bool(reload_consistent and hash_consistent),
        reload_max_abs_diff=max_abs_diff,
        checkpoint_best_sha256=best_sha,
        checkpoint_last_sha256=last_sha,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        train_subject_overlap_with_val=train_overlap,
        val_subject_overlap_with_train=val_overlap,
        n_test_samples=0,
        param_changed=param_changed,
        last_in_process_prediction_hash=_predictions_hash(best_train_preds) if best_train_preds else None,
        class_weight_summary=class_weight_result.as_dict(),
        elapsed_seconds=budget_state.elapsed_seconds,
        budget_status=budget_status,
        budget_report=budget_report,
        budget_thresholds=budget_thresholds,
    )


def _build_candidate_result(
    *,
    candidate_name: str,
    model_version: str,
    parameter_count: int,
    feasibility: str,
    reason: str,
    epoch_metrics: list[EpochMetricsRow],
    train_loss_history: list[float],
    val_loss_history: list[float],
    best_epoch: int | None,
    best_val_loss: float | None,
    best_sha: str | None,
    last_sha: str | None,
    param_changed: bool,
    train_class_weight_result: ClassWeightResult,
    reload_consistent: bool,
    reload_max_abs_diff: float | None,
    elapsed_seconds: float,
    budget_status: str,
    budget_report: dict[str, Any],
    budget_thresholds: dict[str, Any],
) -> CandidateResult:
    """Construct a :class:`CandidateResult` for early-exit branches."""

    return CandidateResult(
        candidate=candidate_name,
        model_version=model_version,
        parameter_count=parameter_count,
        parameter_count_within_budget=bool(parameter_count <= B04_MAX_PARAMETERS),
        feasibility=feasibility,
        reason=reason,
        epoch_metrics=epoch_metrics,
        metrics=None,
        train_predictions=[],
        train_labels=[],
        train_subjects=[],
        val_predictions=[],
        val_labels=[],
        val_subjects=[],
        train_records=[],
        val_records=[],
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_prediction_hash=None,
        in_process_prediction_hash=None,
        reload_consistent=reload_consistent,
        reload_max_abs_diff=reload_max_abs_diff,
        checkpoint_best_sha256=best_sha,
        checkpoint_last_sha256=last_sha,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        train_subject_overlap_with_val=False,
        val_subject_overlap_with_train=False,
        n_test_samples=0,
        param_changed=param_changed,
        last_in_process_prediction_hash=None,
        class_weight_summary=train_class_weight_result.as_dict(),
        elapsed_seconds=float(elapsed_seconds),
        budget_status=str(budget_status),
        budget_report=dict(budget_report),
        budget_thresholds=dict(budget_thresholds),
    )


# ---------------------------------------------------------------------------
# Run the Mini (orchestrator)
# ---------------------------------------------------------------------------


@dataclass
class MiniRunResult:
    """Result of the orchestrator :func:`run_mini`."""

    config: MiniConfig
    dataset_manifest: dict[str, Any]
    environment: dict[str, Any]
    class_weight_result: ClassWeightResult
    candidate_results: dict[str, CandidateResult]
    n_candidates_feasible: int
    n_candidates_not_feasible: int
    n_candidates_failed: int
    n_candidates_stopped: int
    overall_decision: str  # "MINI_NOT_FEASIBLE" | "MINI_HAS_FEASIBLE_CANDIDATE"
    terminal_state: str  # "DONE" | "FAILED" | "STOPPED"
    started_at_utc: str
    ended_at_utc: str
    wall_clock_seconds: float
    input_hashes: dict[str, Any]
    train_class_stats_source: str
    synthetic: bool
    determinism: DeterminismSettings
    resource_budget: ResourceBudget
    b01_contract_report: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.config.task_id,
            "config_version": self.config.config_version,
            "n_candidates_feasible": int(self.n_candidates_feasible),
            "n_candidates_not_feasible": int(self.n_candidates_not_feasible),
            "n_candidates_failed": int(self.n_candidates_failed),
            "n_candidates_stopped": int(self.n_candidates_stopped),
            "overall_decision": self.overall_decision,
            "terminal_state": self.terminal_state,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "wall_clock_seconds": float(self.wall_clock_seconds),
            "environment": self.environment,
            "input_hashes": self.input_hashes,
            "train_class_stats_source": self.train_class_stats_source,
            "synthetic": bool(self.synthetic),
            "determinism": self.determinism.as_dict(),
            "resource_budget": self.resource_budget.as_dict(),
            "b01_contract_report": self.b01_contract_report,
        }


def _gather_environment() -> dict[str, Any]:
    return environment_payload()


def run_mini(
    *,
    config: MiniConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    dataset_manifest: dict[str, Any],
    class_weight_result: ClassWeightResult,
    output_dir: Path,
    device: torch.device,
    input_hashes: dict[str, Any],
    train_class_stats_source: str,
    synthetic: bool,
    budget: ResourceBudget | None = None,
    b01_contract_report: dict[str, Any] | None = None,
    resume_from_per_candidate: dict[str, Path] | None = None,
) -> MiniRunResult:
    """Run the B04 Mini end-to-end across all frozen candidates.

    The orchestrator wires the resource budget, the checkpoint
    identity, the determinism settings, and the (optional) per-
    candidate resume path into the per-candidate runner.  After every
    candidate it inspects the feasibility to compute the terminal
    state:

    * any FAILED → terminal_state="FAILED"
    * any STOPPED (no FAILED) → terminal_state="STOPPED"
    * all candidates FEASIBLE / NOT_FEASIBLE → terminal_state="DONE"
    """

    started_at = datetime.now(timezone.utc).isoformat()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert_class_weight_invariants(class_weight_result)

    # Apply the determinism configuration once for the whole run.
    determinism = apply_settings(config.seed, cpu_threads=1)

    if budget is None:
        budget = resource_budget_from_config(
            {
                "max_wall_minutes_per_candidate": 45,
                "max_total_wall_minutes": 90,
                "max_peak_cuda_mb": 12288,
            }
        )
    budget_state = ResourceBudgetState(budget)

    candidate_results: dict[str, CandidateResult] = {}
    n_feasible = 0
    n_not_feasible = 0
    n_failed = 0
    n_stopped = 0

    input_manifest_hashes_for_payload: dict[str, Any] = dict(input_hashes)

    for candidate_name in config.candidates:
        cand_dir = output_dir / "checkpoints" / candidate_name
        for name in ("last.pt", "best.pt"):
            if (cand_dir / name).exists() and (
                resume_from_per_candidate is None
                or candidate_name not in resume_from_per_candidate
            ):
                raise OutputCollisionError(
                    f"checkpoint {cand_dir / name} already exists; refusing to overwrite"
                )
        identity = CheckpointIdentity(
            task_id=config.task_id,
            candidate=candidate_name,
            model_version=get_model_builder(candidate_name).version,
            seed=int(config.seed),
            n_classes=int(N_CLASSES),
            image_shape=tuple(PRESSURE_SHAPE),
            config_sha256=file_sha256(Path(config.config_path)) if hasattr(config, "config_path") else "",
            a06_split_sha256=str(config.b01_a06_split_sha256_expected),
            freeze_manifest_sha256=str(input_hashes.get("freeze_manifest_sha256", "")),
            train_class_stats_sha256=str(input_hashes.get("train_class_stats_sha256", "")),
            class_weight_sha256=class_weight_sha256(class_weight_result.as_dict()),
            input_manifest_hashes_sha256=input_manifest_hashes_sha256(
                input_manifest_hashes_for_payload
            ),
        )
        resume_path = (
            Path(resume_from_per_candidate[candidate_name])
            if (resume_from_per_candidate and candidate_name in resume_from_per_candidate)
            else None
        )
        result = run_one_candidate(
            candidate_name=candidate_name,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            class_weight_result=class_weight_result,
            output_dir=output_dir,
            device=device,
            budget_state=budget_state,
            identity=identity,
            input_manifest_hashes=input_manifest_hashes_for_payload,
            deterministic=determinism,
            resume_from=resume_path,
        )
        candidate_results[candidate_name] = result
        if result.feasibility == "FEASIBLE":
            n_feasible += 1
        elif result.feasibility == "NOT_FEASIBLE":
            n_not_feasible += 1
        elif result.feasibility == "STOPPED":
            n_stopped += 1
        else:
            n_failed += 1

    ended_at = datetime.now(timezone.utc).isoformat()
    if n_feasible == 0:
        decision = "MINI_NOT_FEASIBLE"
    else:
        decision = "MINI_HAS_FEASIBLE_CANDIDATE"

    # ------------------------------------------------------------------
    # Terminal state machine
    # ------------------------------------------------------------------
    if n_failed > 0:
        terminal_state = "FAILED"
    elif n_stopped > 0:
        terminal_state = "STOPPED"
    else:
        terminal_state = "DONE"

    # ------------------------------------------------------------------
    # Write artifacts (epoch_metrics, metrics_summary, metrics_by_*,
    # confusion_matrix, centroid_errors, worst_subject, predictions_manifest,
    # candidate_decision, reload_consistency, budget_report).
    # ------------------------------------------------------------------
    write_mini_artifacts(
        output_dir=output_dir,
        config=config,
        dataset_manifest=dataset_manifest,
        class_weight_result=class_weight_result,
        candidate_results=candidate_results,
        input_hashes=input_hashes,
        train_class_stats_source=train_class_stats_source,
        budget=budget,
        budget_state=budget_state,
        determinism=determinism,
        b01_contract_report=b01_contract_report,
    )

    return MiniRunResult(
        config=config,
        dataset_manifest=dataset_manifest,
        environment=_gather_environment(),
        class_weight_result=class_weight_result,
        candidate_results=candidate_results,
        n_candidates_feasible=n_feasible,
        n_candidates_not_feasible=n_not_feasible,
        n_candidates_failed=n_failed,
        n_candidates_stopped=n_stopped,
        overall_decision=decision,
        terminal_state=terminal_state,
        started_at_utc=started_at,
        ended_at_utc=ended_at,
        wall_clock_seconds=0.0,  # populated by the CLI
        input_hashes=input_hashes,
        train_class_stats_source=train_class_stats_source,
        synthetic=synthetic,
        determinism=determinism,
        resource_budget=budget,
        b01_contract_report=b01_contract_report,
    )


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def write_mini_artifacts(
    *,
    output_dir: Path,
    config: MiniConfig,
    dataset_manifest: dict[str, Any],
    class_weight_result: ClassWeightResult,
    candidate_results: dict[str, CandidateResult],
    input_hashes: dict[str, Any],
    train_class_stats_source: str,
    budget: ResourceBudget,
    budget_state: ResourceBudgetState,
    determinism: DeterminismSettings,
    b01_contract_report: dict[str, Any] | None,
) -> None:
    output_dir = Path(output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- epoch_metrics.csv ----
    epoch_csv = output_dir / "epoch_metrics.csv"
    with open(epoch_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate", "epoch", "train_loss", "val_loss", "is_best", "elapsed_seconds"]
        )
        writer.writeheader()
        for cand_name, cand in candidate_results.items():
            for row in cand.epoch_metrics:
                writer.writerow(asdict(row))

    # ---- metrics_summary.json (per candidate) ----
    metrics_summary = {
        cand: {
            "feasibility": result.feasibility,
            "reason": result.reason,
            "metrics": result.metrics.as_dict() if result.metrics is not None else None,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "reload_consistent": result.reload_consistent,
            "reload_max_abs_diff": result.reload_max_abs_diff,
            "checkpoint_best_sha256": result.checkpoint_best_sha256,
            "checkpoint_last_sha256": result.checkpoint_last_sha256,
            "n_test_samples": result.n_test_samples,
            "parameter_count": result.parameter_count,
            "parameter_count_within_budget": result.parameter_count_within_budget,
            "model_version": result.model_version,
            "train_subject_overlap_with_val": result.train_subject_overlap_with_val,
            "val_subject_overlap_with_train": result.val_subject_overlap_with_train,
            "param_changed": result.param_changed,
            "best_prediction_hash": result.best_prediction_hash,
            "in_process_prediction_hash": result.in_process_prediction_hash,
            "last_in_process_prediction_hash": result.last_in_process_prediction_hash,
            "class_weight_summary": dict(result.class_weight_summary),
            "elapsed_seconds": float(result.elapsed_seconds),
            "budget_status": str(result.budget_status),
            "budget_report": dict(result.budget_report),
            "budget_thresholds": dict(result.budget_thresholds),
        }
        for cand, result in candidate_results.items()
    }
    write_json(output_dir / "metrics_summary.json", metrics_summary)

    # ---- metrics_by_region.csv ----
    region_csv = output_dir / "metrics_by_region.csv"
    with open(region_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "candidate", "class_id", "iou", "dice", "precision", "recall",
                "tp", "fp", "fn",
            ]
        )
        writer.writeheader()
        for cand_name, cand in candidate_results.items():
            if cand.metrics is None:
                continue
            for row in cand.metrics.per_region:
                writer.writerow({
                    "candidate": cand_name,
                    "class_id": row["class_id"],
                    "iou": row["iou"],
                    "dice": row["dice"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                })

    # ---- metrics_by_posture.csv ----
    posture_csv = output_dir / "metrics_by_posture.csv"
    with open(posture_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "candidate", "posture", "n_samples",
                "fixed_foreground_macro_iou", "fixed_foreground_macro_dice", "pixel_accuracy",
            ]
        )
        writer.writeheader()
        for cand_name, cand in candidate_results.items():
            if cand.metrics is None:
                continue
            for posture, payload in cand.metrics.per_posture.items():
                writer.writerow({
                    "candidate": cand_name,
                    "posture": posture,
                    "n_samples": payload.get("n_samples", 0),
                    "fixed_foreground_macro_iou": payload.get("fixed_foreground_macro_iou", 0.0),
                    "fixed_foreground_macro_dice": payload.get("fixed_foreground_macro_dice", 0.0),
                    "pixel_accuracy": payload.get("pixel_accuracy", 0.0),
                })

    # ---- metrics_by_subject.csv ----
    subject_csv = output_dir / "metrics_by_subject.csv"
    with open(subject_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "candidate", "subject_id", "n_samples",
                "fixed_foreground_macro_iou", "fixed_foreground_macro_dice", "pixel_accuracy",
            ]
        )
        writer.writeheader()
        for cand_name, cand in candidate_results.items():
            if cand.metrics is None:
                continue
            for subject, payload in cand.metrics.per_subject.items():
                writer.writerow({
                    "candidate": cand_name,
                    "subject_id": subject,
                    "n_samples": payload.get("n_samples", 0),
                    "fixed_foreground_macro_iou": payload.get("fixed_foreground_macro_iou", 0.0),
                    "fixed_foreground_macro_dice": payload.get("fixed_foreground_macro_dice", 0.0),
                    "pixel_accuracy": payload.get("pixel_accuracy", 0.0),
                })

    # ---- confusion_matrix.csv (per candidate) ----
    cm_csv = output_dir / "confusion_matrix.csv"
    with open(cm_csv, "w", newline="", encoding="utf-8") as f:
        header = ["candidate", "true_class", "pred_class_0", "pred_class_1", "pred_class_2",
                  "pred_class_3", "pred_class_4", "pred_class_5", "pred_class_6", "pred_class_7",
                  "pred_class_8"]
        writer = csv.writer(f)
        writer.writerow(header)
        for cand_name, cand in candidate_results.items():
            if cand.metrics is None:
                continue
            cm = cand.metrics.confusion_matrix
            for i in range(cm.shape[0]):
                row = [cand_name, i] + [int(v) for v in cm[i].tolist()]
                writer.writerow(row)

    # ---- centroid_errors.csv (per candidate, per sample, per region) ----
    centroid_csv = output_dir / "centroid_errors.csv"
    with open(centroid_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "candidate", "split", "sample_index", "sample_id",
                "subject_id", "posture", "region", "error",
                "valid", "invalid_reason", "both_missing",
            ]
        )
        writer.writeheader()
        for cand_name, cand in candidate_results.items():
            if cand.metrics is None:
                continue
            for split_name, preds, labels, subjects, sample_ids, postures in (
                (
                    "train",
                    cand.train_predictions,
                    _gather_train_labels_for_centroid(cand),
                    _gather_train_subjects_for_centroid(cand),
                    [r.sample_id for r in cand.train_records],
                    [r.posture for r in cand.train_records],
                ),
                (
                    "val",
                    cand.val_predictions,
                    _gather_val_labels_for_centroid(cand),
                    _gather_val_subjects_for_centroid(cand),
                    [r.sample_id for r in cand.val_records],
                    [r.posture for r in cand.val_records],
                ),
            ):
                if not preds or not labels or not subjects:
                    continue
                records = compute_centroid_errors(
                    labels=labels,
                    predictions=preds,
                    subject_ids=subjects,
                )
                for rec in records:
                    sample_id = (
                        sample_ids[rec.sample_index]
                        if rec.sample_index < len(sample_ids)
                        else ""
                    )
                    posture = (
                        postures[rec.sample_index]
                        if rec.sample_index < len(postures)
                        else ""
                    )
                    if rec.both_missing:
                        valid = False
                        invalid_reason = "both_gt_and_pred_absent"
                    else:
                        valid = True
                        invalid_reason = ""
                    writer.writerow({
                        "candidate": cand_name,
                        "split": split_name,
                        "sample_index": rec.sample_index,
                        "sample_id": sample_id,
                        "subject_id": rec.subject_id,
                        "posture": posture,
                        "region": rec.region_id,
                        "error": rec.error,
                        "valid": valid,
                        "invalid_reason": invalid_reason,
                        "both_missing": rec.both_missing,
                    })

    # ---- worst_subject.json ----
    worst_subject_payload: dict[str, Any] = {}
    for cand_name, cand in candidate_results.items():
        worst_subject_payload[cand_name] = cand.metrics.worst_subject if cand.metrics else None
    write_json(output_dir / "worst_subject.json", worst_subject_payload)

    # ---- predictions_manifest.csv ----
    pred_csv = output_dir / "predictions_manifest.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "candidate", "split", "sample_id", "subject_id", "posture",
                "label_sha256", "prediction_sha256", "label_shape", "prediction_shape",
                "failure_reason",
            ]
        )
        writer.writeheader()
        for cand_name, cand in candidate_results.items():
            for record in cand.train_records + cand.val_records:
                writer.writerow(record.as_dict())

    # ---- candidate_decision.json ----
    n_feasible = sum(1 for r in candidate_results.values() if r.feasibility == "FEASIBLE")
    n_not_feasible = sum(1 for r in candidate_results.values() if r.feasibility == "NOT_FEASIBLE")
    n_failed = sum(1 for r in candidate_results.values() if r.feasibility == "FAILED")
    n_stopped = sum(1 for r in candidate_results.values() if r.feasibility == "STOPPED")
    if n_failed > 0:
        terminal_state = "FAILED"
    elif n_stopped > 0:
        terminal_state = "STOPPED"
    else:
        terminal_state = "DONE"
    decision_payload = {
        "task_id": config.task_id,
        "config_version": config.config_version,
        "val_feasibility_threshold": config.val_feasibility_threshold,
        "terminal_state": terminal_state,
        "candidates": {
            cand_name: {
                "feasibility": result.feasibility,
                "reason": result.reason,
                "fixed_foreground_macro_iou": (
                    result.metrics.fixed_foreground_macro_iou
                    if result.metrics is not None else None
                ),
                "val_loss": result.best_val_loss,
                "best_epoch": result.best_epoch,
                "elapsed_seconds": float(result.elapsed_seconds),
                "budget_status": str(result.budget_status),
            }
            for cand_name, result in candidate_results.items()
        },
        "n_feasible": n_feasible,
        "n_not_feasible": n_not_feasible,
        "n_failed": n_failed,
        "n_stopped": n_stopped,
        "overall_decision": (
            "MINI_NOT_FEASIBLE" if n_feasible == 0 else "MINI_HAS_FEASIBLE_CANDIDATE"
        ),
    }
    write_json(output_dir / "candidate_decision.json", decision_payload)

    # ---- reload_consistency.json ----
    reload_payload = {
        cand_name: {
            "reload_consistent": result.reload_consistent,
            "max_abs_diff": result.reload_max_abs_diff,
            "best_prediction_hash": result.best_prediction_hash,
            "in_process_prediction_hash": result.in_process_prediction_hash,
            "hash_match": bool(
                result.best_prediction_hash is not None
                and result.in_process_prediction_hash is not None
                and result.best_prediction_hash == result.in_process_prediction_hash
            ),
        }
        for cand_name, result in candidate_results.items()
    }
    write_json(output_dir / "reload_consistency.json", reload_payload)

    # ---- budget_report.json ----
    budget_report = {
        "thresholds": budget.as_dict(),
        "elapsed_total_seconds": float(budget_state.total_elapsed_seconds),
        "peak_cuda_mb": float(budget_state.peak_cuda_mb),
        "candidates": {
            cand_name: {
                "elapsed_seconds": float(result.elapsed_seconds),
                "budget_status": str(result.budget_status),
                "budget_report": dict(result.budget_report),
            }
            for cand_name, result in candidate_results.items()
        },
        "terminal_state": terminal_state,
        "determinism": determinism.as_dict(),
    }
    if b01_contract_report is not None:
        budget_report["b01_contract_report"] = b01_contract_report
    write_json(output_dir / "budget_report.json", budget_report)

    # ---- run.log ----
    log_lines = [
        f"task_id={config.task_id}",
        f"config_version={config.config_version}",
        f"started_at_utc={datetime.now(timezone.utc).isoformat()}",
        f"device={config.device}",
        f"candidates={list(config.candidates)}",
        f"n_train_samples={dataset_manifest.get('n_train_samples')}",
        f"n_val_samples={dataset_manifest.get('n_val_samples')}",
        f"n_test_samples={dataset_manifest.get('n_test_samples', 0)}",
        f"train_class_stats_source={train_class_stats_source}",
        f"determinism={json.dumps(determinism.as_dict(), sort_keys=True)}",
        f"terminal_state={terminal_state}",
    ]
    for cand_name, result in candidate_results.items():
        log_lines.append(
            f"candidate={cand_name} feasibility={result.feasibility} "
            f"budget_status={result.budget_status} elapsed={result.elapsed_seconds:.2f}s "
            f"reason={result.reason}"
        )
    (log_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Top-level helper: write a single "DONE.json" / "FAILED.json" file
# ---------------------------------------------------------------------------


def write_status_files(
    output_dir: Path,
    *,
    status: str,
    extra: Mapping[str, Any],
) -> None:
    """Emit exactly one of ``DONE.json`` / ``FAILED.json`` / ``STOPPED.json``.

    The three terminal files are mutually exclusive: writing one of
    them deletes the other two so a downstream reader can rely on
    ``ls output_dir | grep .json`` to see at most one terminal file.
    """

    output_dir = Path(output_dir)
    if status not in {"DONE", "FAILED", "STOPPED"}:
        raise MiniProtocolError(f"unknown terminal status {status!r}")
    targets = {
        "DONE": output_dir / "DONE.json",
        "FAILED": output_dir / "FAILED.json",
        "STOPPED": output_dir / "STOPPED.json",
    }
    target = targets[status]
    for other_status, other_path in targets.items():
        if other_status != status and other_path.exists():
            other_path.unlink()
    payload = {
        "status": status,
        "task_id": TASK_ID,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(dict(extra))
    write_json(target, payload)


# ---------------------------------------------------------------------------
# Centroid-error helpers for the CSV emission
# ---------------------------------------------------------------------------


def _gather_train_labels_for_centroid(cand: CandidateResult) -> list[np.ndarray]:
    return list(cand.train_labels)


def _gather_train_subjects_for_centroid(cand: CandidateResult) -> list[str]:
    return list(cand.train_subjects)


def _gather_val_labels_for_centroid(cand: CandidateResult) -> list[np.ndarray]:
    return list(cand.val_labels)


def _gather_val_subjects_for_centroid(cand: CandidateResult) -> list[str]:
    return list(cand.val_subjects)
