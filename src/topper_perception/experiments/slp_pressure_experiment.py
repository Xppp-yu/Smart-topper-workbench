"""Pressure-only experiment configuration and manifest.

This module defines the configuration structure for SLP pressure-only model
experiments (TASK-SLP-B01 and beyond).

It provides:
- ExperimentConfig: Validated experiment configuration
- ExperimentManifest: Frozen experiment record with results
- Config validation: Stdlib-only checks mirroring the JSON schema

Design rules:
* experiment_id, input_contract_version, split_manifest, region_label_manifest
  are required at config time.
* region_label_manifest may be empty/null before A17 freeze, but the
  training entry point must reject empty labels.
* Preprocessing, model_name, random_seed, perturbation_config, density_config,
  metrics, runtime device are all specified here.
* Config changes are immutable after QUEUED.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


def _compute_file_sha256(path: Path) -> str:
    """Compute SHA256 of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_SCHEMA_VERSION = "slp_pressure_experiment_v0.1"

#: EXP-ID pattern: EXP- followed by letters/digits/dot/underscore/dash.
EXP_ID_PATTERN = re.compile(r"^EXP-[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Required config fields.
REQUIRED_FIELDS = (
    "experiment_id",
    "task_id",
    "scope",
    "seed",
    "input_contract_version",
    "split_manifest",
    "region_label_manifest",
)

#: Optional fields with defaults.
OPTIONAL_FIELDS = (
    "preprocessing",
    "model_name",
    "perturbation_config",
    "density_config",
    "metrics",
    "runtime_device",
    "label_manifest_required",
)

#: Valid scopes.
VALID_SCOPES = ("smoke", "mini", "full")

#: Valid runtime devices.
VALID_DEVICES = ("cpu", "cuda", "mps")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ExperimentConfigError(Exception):
    """Base exception for experiment config errors."""
    pass


class ConfigValidationError(ExperimentConfigError):
    """Configuration validation failed."""
    pass


class ExpIdError(ExperimentConfigError):
    """Invalid experiment ID."""
    pass


class LabelManifestError(ExperimentConfigError):
    """Label manifest validation failed."""
    pass


class SplitManifestError(ExperimentConfigError):
    """Split manifest validation failed."""
    pass


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing pipeline configuration."""
    normalize: bool = True
    normalize_range: tuple[float, float] = (0.0, 1.0)
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    resize: tuple[int, int] | None = None  # (H, W) or None to keep original
    to_tensor: bool = True
    transpose_channels: bool = False  # HWC -> CHW for PyTorch

    def as_dict(self) -> dict[str, Any]:
        return {
            "normalize": self.normalize,
            "normalize_range": list(self.normalize_range),
            "mean": list(self.mean) if self.mean else None,
            "std": list(self.std) if self.std else None,
            "resize": list(self.resize) if self.resize else None,
            "to_tensor": self.to_tensor,
            "transpose_channels": self.transpose_channels,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PreprocessingConfig":
        return cls(
            normalize=data.get("normalize", True),
            normalize_range=tuple(data.get("normalize_range", [0.0, 1.0])),
            mean=tuple(data["mean"]) if data.get("mean") else None,
            std=tuple(data["std"]) if data.get("std") else None,
            resize=tuple(data["resize"]) if data.get("resize") else None,
            to_tensor=data.get("to_tensor", True),
            transpose_channels=data.get("transpose_channels", False),
        )


@dataclass(frozen=True)
class PerturbationConfig:
    """Perturbation configuration for robustness testing."""
    enabled: bool = False
    perturbations: tuple[dict[str, Any], ...] = ()
    preset: str | None = None  # "light", "medium", "heavy", or None
    composite_seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "perturbations": list(self.perturbations),
            "preset": self.preset,
            "composite_seed": self.composite_seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PerturbationConfig":
        perturbations = data.get("perturbations", [])
        if isinstance(perturbations, list):
            perturbations = tuple(perturbations)
        return cls(
            enabled=data.get("enabled", False),
            perturbations=perturbations,
            preset=data.get("preset"),
            composite_seed=data.get("composite_seed", 42),
        )


@dataclass(frozen=True)
class DensityConfig:
    """Density transform configuration for sensor density testing."""
    enabled: bool = False
    density_level: float = 1.0  # 1.0, 0.5, 0.25, 0.125
    layout: str = "uniform"  # "uniform", "sparse", "local_high_density"
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "density_level": self.density_level,
            "layout": self.layout,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DensityConfig":
        return cls(
            enabled=data.get("enabled", False),
            density_level=data.get("density_level", 1.0),
            layout=data.get("layout", "uniform"),
            seed=data.get("seed", 42),
        )


@dataclass(frozen=True)
class MetricsConfig:
    """Metrics to compute."""
    mIoU: bool = True
    macro_f1: bool = True
    per_region_iou: bool = True
    accuracy: bool = True
    precision: bool = True
    recall: bool = True
    confusion_matrix: bool = True
    ignore_label: int = -1
    uncertain_label: int = -2

    def as_dict(self) -> dict[str, Any]:
        return {
            "mIoU": self.mIoU,
            "macro_f1": self.macro_f1,
            "per_region_iou": self.per_region_iou,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "confusion_matrix": self.confusion_matrix,
            "ignore_label": self.ignore_label,
            "uncertain_label": self.uncertain_label,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricsConfig":
        return cls(
            mIoU=data.get("mIoU", True),
            macro_f1=data.get("macro_f1", True),
            per_region_iou=data.get("per_region_iou", True),
            accuracy=data.get("accuracy", True),
            precision=data.get("precision", True),
            recall=data.get("recall", True),
            confusion_matrix=data.get("confusion_matrix", True),
            ignore_label=data.get("ignore_label", -1),
            uncertain_label=data.get("uncertain_label", -2),
        )


@dataclass(frozen=True)
class SplitManifest:
    """Split manifest reference."""
    path: str
    sha256: str | None = None  # Optional content hash
    version: str = "slp_subject_split_v0.1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SplitManifest":
        return cls(
            path=data["path"],
            sha256=data.get("sha256"),
            version=data.get("version", "slp_subject_split_v0.1"),
        )


@dataclass(frozen=True)
class RegionLabelManifest:
    """Region label manifest reference."""
    path: str | None = None  # None means labels not yet frozen
    sha256: str | None = None
    version: str | None = None
    is_frozen: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "version": self.version,
            "is_frozen": self.is_frozen,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RegionLabelManifest":
        if data is None:
            return cls()
        return cls(
            path=data.get("path"),
            sha256=data.get("sha256"),
            version=data.get("version"),
            is_frozen=data.get("is_frozen", False),
        )

    def requires_labels(self) -> bool:
        """Check if labels are required but not available."""
        return self.path is None or not self.is_frozen


@dataclass(frozen=True)
class PressureExperimentConfig:
    """Complete pressure-only experiment configuration.

    This is a frozen dataclass after validation.
    """
    experiment_id: str
    task_id: str
    scope: str
    seed: int
    input_contract_version: str
    split_manifest: SplitManifest
    region_label_manifest: RegionLabelManifest
    preprocessing: PreprocessingConfig
    model_name: str | None
    perturbation_config: PerturbationConfig
    density_config: DensityConfig
    metrics: MetricsConfig
    runtime_device: str
    label_manifest_required: bool = True  # If True, reject empty labels

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "task_id": self.task_id,
            "scope": self.scope,
            "seed": self.seed,
            "input_contract_version": self.input_contract_version,
            "split_manifest": self.split_manifest.as_dict(),
            "region_label_manifest": self.region_label_manifest.as_dict(),
            "preprocessing": self.preprocessing.as_dict(),
            "model_name": self.model_name,
            "perturbation_config": self.perturbation_config.as_dict(),
            "density_config": self.density_config.as_dict(),
            "metrics": self.metrics.as_dict(),
            "runtime_device": self.runtime_device,
            "label_manifest_required": self.label_manifest_required,
        }

    def to_json(self, path: Path) -> None:
        """Write config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PressureExperimentConfig":
        """Create config from dict (after validation)."""
        return cls(
            experiment_id=data["experiment_id"],
            task_id=data["task_id"],
            scope=data["scope"],
            seed=data["seed"],
            input_contract_version=data["input_contract_version"],
            split_manifest=SplitManifest.from_dict(data["split_manifest"]),
            region_label_manifest=RegionLabelManifest.from_dict(
                data.get("region_label_manifest")
            ),
            preprocessing=PreprocessingConfig.from_dict(
                data.get("preprocessing", {})
            ),
            model_name=data.get("model_name"),
            perturbation_config=PerturbationConfig.from_dict(
                data.get("perturbation_config", {})
            ),
            density_config=DensityConfig.from_dict(
                data.get("density_config", {})
            ),
            metrics=MetricsConfig.from_dict(
                data.get("metrics", {})
            ),
            runtime_device=data.get("runtime_device", "cpu"),
            label_manifest_required=data.get("label_manifest_required", True),
        )


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------


def validate_exp_id(exp_id: str) -> None:
    """Validate experiment ID format."""
    if not isinstance(exp_id, str) or not exp_id:
        raise ExpIdError("experiment_id must be a non-empty string.")
    if EXP_ID_PATTERN.fullmatch(exp_id) is None:
        raise ExpIdError(
            f"experiment_id {exp_id!r} must match pattern: "
            "EXP-<ALNUM>... using only letters, digits, '.', '_', '-'"
        )


def validate_experiment_config(
    data: Mapping[str, Any],
    strict_label_check: bool = True,
) -> PressureExperimentConfig:
    """Validate and create a pressure-only experiment config.

    Parameters
    ----------
    data : Mapping[str, Any]
        Raw config dict.
    strict_label_check : bool
        If True (default), require valid region_label_manifest for Mini/Full.

    Returns
    -------
    PressureExperimentConfig
        Validated frozen config.

    Raises
    ------
    ConfigValidationError
        If validation fails.
    SplitManifestError
        If split_manifest.path is empty or invalid.
    LabelManifestError
        If labels are required but not available.
    """
    if not isinstance(data, Mapping):
        raise ConfigValidationError("Config must be a JSON object.")

    # Check required fields
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise ConfigValidationError(f"Missing required fields: {missing}")

    # Validate experiment_id
    validate_exp_id(data["experiment_id"])

    # Validate scope
    scope = data["scope"]
    if scope not in VALID_SCOPES:
        raise ConfigValidationError(
            f"scope must be one of {VALID_SCOPES}, got {scope!r}"
        )

    # Validate seed
    seed = data["seed"]
    if not isinstance(seed, int):
        raise ConfigValidationError("seed must be an integer.")

    # Validate input_contract_version
    icv = data["input_contract_version"]
    if not isinstance(icv, str) or not icv:
        raise ConfigValidationError("input_contract_version must be a non-empty string.")

    # Validate split_manifest
    sm = data["split_manifest"]
    if not isinstance(sm, Mapping):
        raise ConfigValidationError("split_manifest must be an object.")
    if "path" not in sm:
        raise SplitManifestError(
            "split_manifest must have 'path' field."
        )
    if not sm["path"] or not isinstance(sm["path"], str):
        raise SplitManifestError(
            "split_manifest.path must be a non-empty string."
        )

    # Check split_manifest file existence (uniform for absolute and relative paths)
    split_path = Path(sm["path"])
    if not split_path.is_file():
        raise SplitManifestError(
            f"split_manifest.path does not exist or is not a file: {split_path}. "
            "Please verify the path is correct."
        )
    # Validate split_manifest version if provided
    if sm.get("version") is not None and not sm.get("version"):
        raise SplitManifestError(
            "split_manifest.version must be non-empty if provided."
        )

    # Validate region_label_manifest requirements for Mini/Full
    # Mini/Full ALWAYS require frozen R2/R3 manifest unconditionally.
    # The caller cannot bypass this requirement.
    rlm = data.get("region_label_manifest")

    if scope in ("mini", "full"):
        if rlm is None:
            raise LabelManifestError(
                f"scope={scope} requires region_label_manifest. "
                "Ground truth has not been frozen yet (A17 pending). "
                "Use scope=smoke instead."
            )
        if isinstance(rlm, Mapping):
            if not rlm.get("path"):
                raise LabelManifestError(
                    f"scope={scope} requires a non-empty region_label_manifest.path. "
                    "Ground truth has not been frozen yet (A17 pending)."
                )
            # Check label manifest file existence (uniform for absolute and relative paths)
            label_path = Path(rlm["path"])
            if not label_path.is_file():
                raise LabelManifestError(
                    f"region_label_manifest.path does not exist or is not a file: {label_path}. "
                    "Please verify the path is correct."
                )
            # Require is_frozen=true for Mini/Full
            if not rlm.get("is_frozen", False):
                raise LabelManifestError(
                    f"scope={scope} requires is_frozen=True for region_label_manifest. "
                    "R2/R3 labels must be frozen before Mini/Full (A17 Gate)."
                )
            # Require version and sha256 for frozen manifest
            if not rlm.get("version"):
                raise LabelManifestError(
                    f"scope={scope} requires version in region_label_manifest. "
                    "A17 manifest version must be specified."
                )
            if not rlm.get("sha256"):
                raise LabelManifestError(
                    f"scope={scope} requires sha256 in region_label_manifest. "
                    "A17 manifest content hash must be specified."
                )
            # Verify SHA256 if both declared and file exists
            declared_sha = rlm.get("sha256", "")
            if declared_sha and label_path.is_file():
                actual_sha = _compute_file_sha256(label_path)
                if actual_sha != declared_sha:
                    raise LabelManifestError(
                        f"SHA256 mismatch for region_label_manifest: "
                        f"declared={declared_sha}, actual={actual_sha}. "
                        "The manifest file may have been modified."
                    )
    elif scope == "smoke":
        # Smoke can run without labels, but must explicitly set scope=smoke
        if rlm is not None and isinstance(rlm, Mapping):
            if rlm.get("label_manifest_required", False) and not rlm.get("path"):
                raise LabelManifestError(
                    "Smoke config has label_manifest_required=True but no path. "
                    "Use scope=smoke with label_manifest_required=False, "
                    "or provide a valid label manifest path."
                )

    # Validate runtime_device
    device = data.get("runtime_device", "cpu")
    if device not in VALID_DEVICES:
        raise ConfigValidationError(
            f"runtime_device must be one of {VALID_DEVICES}, got {device!r}"
        )

    # Create config through proper validation
    return PressureExperimentConfig.from_dict(data)


# ---------------------------------------------------------------------------
# Manifest Functions
# ---------------------------------------------------------------------------


def compute_config_hash(config: PressureExperimentConfig) -> str:
    """Compute deterministic hash of config for reproducibility."""
    config_json = json.dumps(
        config.as_dict(),
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()


def create_default_config(
    experiment_id: str,
    task_id: str,
    scope: str = "smoke",
    *,
    split_manifest_path: str,
    region_label_manifest_path: str | None = None,
    seed: int = 42,
    model_name: str | None = None,
    device: str = "cpu",
) -> PressureExperimentConfig:
    """Create a default pressure-only experiment config.

    This is a convenience function for common configurations.
    The returned config has already been validated.

    Parameters
    ----------
    experiment_id : str
        EXP-ID (e.g., "EXP-SLP-B01-SMOKE-001").
    task_id : str
        TASK-ID for provenance.
    scope : str
        "smoke", "mini", or "full".
    split_manifest_path : str
        Path to A06 split manifest.
    region_label_manifest_path : str | None
        Path to A17 label manifest. REQUIRED for mini/full scope.
        Smoke can run without labels.
    seed : int
        Random seed.
    model_name : str | None
        Model name/architecture.
    device : str
        Runtime device.

    Returns
    -------
    PressureExperimentConfig
        Validated frozen config.

    Raises
    ------
    SplitManifestError
        If split_manifest_path is empty.
    LabelManifestError
        If mini/full scope is used without region_label_manifest_path.
    """
    if not split_manifest_path or not isinstance(split_manifest_path, str):
        raise SplitManifestError(
            "split_manifest_path must be a non-empty string."
        )

    # Mini/Full require labels unconditionally
    if scope in ("mini", "full") and not region_label_manifest_path:
        raise LabelManifestError(
            f"scope={scope} requires region_label_manifest_path. "
            "Ground truth has not been frozen yet (A17 pending). "
            "Use scope=smoke instead."
        )

    config_dict = {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "scope": scope,
        "seed": seed,
        "input_contract_version": "slp_pressure_only_input_contract_v0.1",
        "split_manifest": {
            "path": split_manifest_path,
            "version": "slp_subject_split_v0.1",
        },
        "region_label_manifest": {
            "path": region_label_manifest_path,
            "is_frozen": False,  # Must be explicitly set to True after A17 freeze
            "version": None,
            "sha256": None,
        } if region_label_manifest_path else None,
        "preprocessing": {
            "normalize": True,
            "normalize_range": [0.0, 1.0],
            "to_tensor": True,
        },
        "model_name": model_name,
        "perturbation_config": {
            "enabled": False,
        },
        "density_config": {
            "enabled": False,
            "density_level": 1.0,
            "layout": "uniform",
        },
        "metrics": {
            "mIoU": True,
            "macro_f1": True,
            "per_region_iou": True,
        },
        "runtime_device": device,
    }

    # Go through proper validation (not bypassed)
    return validate_experiment_config(config_dict, strict_label_check=True)


# ---------------------------------------------------------------------------
# Serialization Helpers
# ---------------------------------------------------------------------------


def load_experiment_config(path: Path | str) -> PressureExperimentConfig:
    """Load experiment config from JSON file.

    The loaded config is validated through validate_experiment_config.

    Parameters
    ----------
    path : Path | str
        Path to JSON config file.

    Returns
    -------
    PressureExperimentConfig
        Validated frozen config.

    Raises
    ------
    ConfigValidationError
        If the config does not pass validation.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_experiment_config(data)


def save_experiment_config(
    config: PressureExperimentConfig,
    path: Path | str,
) -> None:
    """Save experiment config to JSON file."""
    config.to_json(Path(path))
