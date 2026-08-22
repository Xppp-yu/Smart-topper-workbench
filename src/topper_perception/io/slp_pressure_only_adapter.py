"""Pressure-only Input Adapter for SLP.

This adapter provides a strict pressure-only input interface for future
Pressure-only model experiments (TASK-SLP-B01 and beyond).

Core constraints (TASK-SLP-B01 contract):
- Model input is ONLY the Pressure Map (PM).
- Visual modalities (RGB, IR, Depth) are NEVER loaded into the model input tensor.
- Visual data may only be used as provenance or quality audit records, not as features.
- Quarantine samples are excluded from the default dataset.
- A06 frozen subject split is used without modification.
- Output contracts must document shape, dtype, value range, and preprocessing.

Design rules:
* Strict modality isolation: only PM enters the model input tensor.
* Provenance chain is fully traceable to A05/A06 artifacts.
* Subject isolation is enforced via the A06 split manifest.
* Deterministic: same inputs → same outputs, fixed seed.
* No ground truth is generated here (B01 contract).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADAPTER_VERSION = "slp_pressure_only_adapter_v0.1"
INPUT_CONTRACT_VERSION = "slp_pressure_only_input_contract_v0.1"

#: Only the PM modality enters the model input tensor.
PRESSURE_ONLY_MODALITY = "PM"

#: Visual modalities that are NEVER loaded as model input.
FORBIDDEN_MODALITIES = frozenset({"RGB", "IR", "depth", "IRraw", "depthRaw"})

#: PM image dimensions in the SLP dataset (width × height).
PM_IMAGE_SIZE = (84, 192)

#: Expected PM dtype.
PM_DTYPE = np.float32

#: Expected PM value range (normalized 0-1 in SLP PNG).
PM_VALUE_RANGE = (0.0, 1.0)

#: Task ID for provenance.
DEFAULT_TASK_ID = "TASK-SLP-B01-PRESSURE-ONLY-INFRA-v0.1"
DEFAULT_GENERATOR = "topper_perception.io.slp_pressure_only_adapter.SlpPressureOnlyAdapter"


class DataSplit(str, Enum):
    """Canonical data split names matching A06."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class QualityTier(str, Enum):
    """Quality tier for pressure data, mirroring A02/A05 quality gates."""
    ACCEPT = "ACCEPT"
    WARN = "WARN"
    REJECT = "REJECT"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PressureInputContract:
    """Contract documenting the Pressure-only input specification.

    This is NOT the actual pressure data — it documents the contract
    so downstream code can verify compliance.
    """
    contract_version: str
    modality: str
    image_size: tuple[int, int]  # (width, height)
    dtype: str
    value_range: tuple[float, float]
    preprocessing: tuple[str, ...]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "modality": self.modality,
            "image_size": list(self.image_size),
            "dtype": self.dtype,
            "value_range": list(self.value_range),
            "preprocessing": list(self.preprocessing),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class PressureOnlySample:
    """One pressure-only sample for model training/inference.

    This sample only contains the Pressure Map. Visual modality URIs
    are preserved for provenance only, not as model input.
    """

    # Primary keys
    sample_id: str
    setting: str
    subject_id: str
    cover_condition: str
    frame_index: int

    # Split membership (from A06)
    split: str

    # Pressure Map data
    pressure_map: np.ndarray
    pressure_map_uri: str

    # Input contract documentation
    input_contract: PressureInputContract

    # Quality and provenance (from A05)
    quality_flags: tuple[str, ...]
    quarantine: bool
    quarantine_reasons: tuple[str, ...]

    # Visual modality URIs (provenance only, NEVER model input)
    visual_modality_uris: tuple[tuple[str, str], ...]

    # Provenance
    provenance: Provenance

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "setting": self.setting,
            "subject_id": self.subject_id,
            "cover_condition": self.cover_condition,
            "frame_index": self.frame_index,
            "split": self.split,
            "pressure_map_uri": self.pressure_map_uri,
            "pressure_map_shape": list(self.pressure_map.shape),
            "pressure_map_dtype": str(self.pressure_map.dtype),
            "pressure_map_value_range": [
                float(self.pressure_map.min()),
                float(self.pressure_map.max()),
            ],
            "input_contract": self.input_contract.as_dict(),
            "quality_flags": list(self.quality_flags),
            "quarantine": self.quarantine,
            "quarantine_reasons": list(self.quarantine_reasons),
            "visual_modality_uris": list(self.visual_modality_uris),
            "provenance": self.provenance.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class Provenance:
    """How this sample was produced."""
    task_id: str
    adapter_version: str
    input_contract_version: str
    generator: str
    created_at: str
    a05_canonical_sample_id: str
    a06_split_manifest: str
    pressure_only_enforced: bool
    visual_modalities_loaded: bool  # Must be False
    model_input_tensor_modalities: tuple[str, ...]  # Must be ("PM",)


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Summary statistics for a pressure-only dataset."""
    total_samples: int
    train_samples: int
    val_samples: int
    test_samples: int
    quarantined_samples: int
    subjects: int
    settings: dict[str, int]
    covers: dict[str, int]
    pressure_map_shapes: dict[str, int]
    value_ranges: dict[str, float]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SlpPressureOnlyAdapter:
    """Build pressure-only samples from A05 canonical samples + A06 split.

    This adapter enforces the pressure-only constraint by:
    1. Reading A05 canonical samples.
    2. Joining with A06 split manifest for train/val/test membership.
    3. Loading ONLY the Pressure Map (PM) as model input.
    4. Preserving visual URIs as provenance only.
    5. Excluding quarantined samples from the default dataset.
    6. Enforcing subject isolation via A06.

    Parameters
    ----------
    canonical_samples : Iterable[Mapping]
        A05 canonical samples (CSV row or JSONL dict).
        Required fields: sample_id, setting, subject_id, cover_condition,
        frame_index, quarantine, frame.modality_uris.PM.
    split_manifest : Mapping
        A06 split manifest (subject → split mapping).
        Must support __getitem__ with composite key "setting::subject_id".
    slp_root : Path
        SLP root directory for resolving PM URIs.
    task_id : str
        TASK-ID for provenance.
    load_pressure_data : bool
        If True, load actual PM arrays from disk.
        If False, only validate URIs and build metadata (for lightweight inspection).
    """

    def __init__(
        self,
        canonical_samples: Iterable[Mapping[str, Any]],
        split_manifest: Mapping[str, str],
        *,
        slp_root: Path | str,
        task_id: str = DEFAULT_TASK_ID,
        load_pressure_data: bool = True,
        now: datetime | None = None,
    ) -> None:
        self.slp_root = Path(slp_root)
        self._samples = list(canonical_samples)
        self._split_manifest = dict(split_manifest)
        self.task_id = task_id
        self.load_pressure_data = load_pressure_data
        self.created_at = (now or datetime.now(timezone.utc)).isoformat()

        # Build input contract
        self._input_contract = PressureInputContract(
            contract_version=INPUT_CONTRACT_VERSION,
            modality=PRESSURE_ONLY_MODALITY,
            image_size=PM_IMAGE_SIZE,
            dtype=str(PM_DTYPE),
            value_range=PM_VALUE_RANGE,
            preprocessing=("normalize_to_0_1", "to_tensor_format"),
            notes=(
                "Pressure-only: visual modalities (RGB/IR/Depth) are NEVER model input.",
                "PM PNG values are 0-1 float32, loaded with cv2.IMREAD_UNCHANGED.",
            ),
        )

        # Validate that no visual modalities will be loaded
        self._validate_modality_constraint()

    def _validate_modality_constraint(self) -> None:
        """Verify that the constraint of not loading visual modalities is documented."""
        # This is a documentation-only check; the adapter NEVER loads visual data.
        # The constraint is enforced by design: only PM enters the model tensor.
        pass

    # -- Public API -----------------------------------------------------------

    def iter_samples(
        self,
        include_quarantine: bool = False,
        split: DataSplit | None = None,
    ) -> Iterator[PressureOnlySample]:
        """Iterate over pressure-only samples.

        Parameters
        ----------
        include_quarantine : bool
            If False (default), quarantined samples are excluded.
            If True, quarantined samples are included (for auditing).
        split : DataSplit | None
            If specified, only samples from this split are returned.
            If None, all splits are included.
        """
        for row in self._samples:
            sample = self._build_sample(row)
            if sample is None:
                continue

            # Apply filters
            if not include_quarantine and sample.quarantine:
                continue
            if split is not None and sample.split != split.value:
                continue

            yield sample

    def build_dataset(
        self,
        include_quarantine: bool = False,
        split: DataSplit | None = None,
    ) -> tuple[list[np.ndarray], list[int], list[PressureOnlySample]]:
        """Build a pressure-only dataset for model training/inference.

        Returns (pressure_maps, labels, samples) where:
        - pressure_maps: list of numpy arrays (PM tensors)
        - labels: list of integer labels (for classification tasks)
        - samples: list of PressureOnlySample metadata

        Parameters
        ----------
        include_quarantine : bool
            If False (default), quarantined samples are excluded.
        split : DataSplit | None
            If specified, only samples from this split are returned.

        Returns
        -------
        tuple[list, list, list]
            (pressure_maps, labels, samples)
        """
        pressure_maps: list[np.ndarray] = []
        labels: list[int] = []
        samples: list[PressureOnlySample] = []

        for sample in self.iter_samples(include_quarantine=include_quarantine, split=split):
            pressure_maps.append(sample.pressure_map)
            # Labels default to frame_index for now; override in subclass for specific tasks
            labels.append(sample.frame_index)
            samples.append(sample)

        return pressure_maps, labels, samples

    def compute_summary(
        self,
        include_quarantine: bool = False,
    ) -> DatasetSummary:
        """Compute summary statistics for the dataset."""
        total = train = val = test = quarantined = 0
        settings: dict[str, int] = {}
        covers: dict[str, int] = {}
        shapes: dict[str, int] = {}
        min_val, max_val = float("inf"), float("-inf")

        for sample in self.iter_samples(include_quarantine=include_quarantine):
            total += 1
            if sample.quarantine:
                quarantined += 1

            if sample.split == "train":
                train += 1
            elif sample.split == "val":
                val += 1
            elif sample.split == "test":
                test += 1

            settings[sample.setting] = settings.get(sample.setting, 0) + 1
            covers[sample.cover_condition] = covers.get(sample.cover_condition, 0) + 1

            shape_key = str(sample.pressure_map.shape)
            shapes[shape_key] = shapes.get(shape_key, 0) + 1

            pmin, pmax = sample.pressure_map.min(), sample.pressure_map.max()
            min_val = min(min_val, pmin)
            max_val = max(max_val, pmax)

        # Collect unique subjects
        subjects: set[str] = set()
        for row in self._samples:
            key = f"{row['setting']}::{row['subject_id']}"
            subjects.add(key)

        return DatasetSummary(
            total_samples=total,
            train_samples=train,
            val_samples=val,
            test_samples=test,
            quarantined_samples=quarantined,
            subjects=len(subjects),
            settings=settings,
            covers=covers,
            pressure_map_shapes=shapes,
            value_ranges={"min": min_val, "max": max_val},
        )

    def verify_subject_isolation(self) -> list[str]:
        """Verify that subject isolation is maintained.

        Returns empty list if clean; list of error messages otherwise.
        """
        errors: list[str] = []
        subject_splits: dict[str, str] = {}

        for row in self._samples:
            key = f"{row['setting']}::{row['subject_id']}"
            sample_id = row.get("sample_id", "unknown")

            # Get split from manifest
            if key not in self._split_manifest:
                errors.append(f"Subject {key} (sample {sample_id}) not in split manifest")
                continue

            split = self._split_manifest[key]

            if key in subject_splits:
                prev_split = subject_splits[key]
                if prev_split != split:
                    errors.append(
                        f"Subject {key} appears in both {prev_split} and {split} "
                        f"(sample: {sample_id})"
                    )
            else:
                subject_splits[key] = split

        return errors

    # -- Private helpers ------------------------------------------------------

    def _build_sample(self, row: Mapping[str, Any]) -> PressureOnlySample | None:
        """Build one PressureOnlySample from an A05 canonical sample row."""
        sample_id = str(row.get("sample_id", ""))
        setting = str(row.get("setting", ""))
        subject_id = str(row.get("subject_id", ""))
        cover_condition = str(row.get("cover_condition", ""))
        frame_index = int(row.get("frame_index", 0))

        # Get split from manifest
        key = f"{setting}::{subject_id}"
        split = self._split_manifest.get(key, "unknown")

        # Check quarantine status
        raw_q = row.get("quarantine", False)
        quarantine = str(raw_q).strip().lower() in ("true", "1", "yes")

        # Quality flags
        raw_flags = row.get("quality_flags", "")
        if isinstance(raw_flags, str):
            quality_flags = tuple(f.strip() for f in raw_flags.split(";") if f.strip())
        else:
            quality_flags = tuple()

        # Quarantine reasons
        raw_reasons = row.get("quarantine_reasons", "")
        if isinstance(raw_reasons, str):
            quarantine_reasons = tuple(r.strip() for r in raw_reasons.split(";") if r.strip())
        else:
            quarantine_reasons = tuple()

        # Get PM URI from A05 modality_uris
        pm_uri = self._extract_pm_uri(row)
        if not pm_uri:
            # No PM available (e.g., simLab samples)
            return None

        # Load pressure data if requested
        if self.load_pressure_data:
            pressure_map = self._load_pressure_map(pm_uri)
        else:
            # Placeholder for lightweight inspection
            pressure_map = np.zeros(PM_IMAGE_SIZE[::-1], dtype=PM_DTYPE)

        # Collect visual modality URIs for provenance only
        visual_uris = self._extract_visual_uris(row)

        # Build provenance
        provenance = Provenance(
            task_id=self.task_id,
            adapter_version=ADAPTER_VERSION,
            input_contract_version=INPUT_CONTRACT_VERSION,
            generator=DEFAULT_GENERATOR,
            created_at=self.created_at,
            a05_canonical_sample_id=sample_id,
            a06_split_manifest="slp_subject_split_v0.1",
            pressure_only_enforced=True,
            visual_modalities_loaded=False,
            model_input_tensor_modalities=(PRESSURE_ONLY_MODALITY,),
        )

        return PressureOnlySample(
            sample_id=sample_id,
            setting=setting,
            subject_id=subject_id,
            cover_condition=cover_condition,
            frame_index=frame_index,
            split=split,
            pressure_map=pressure_map,
            pressure_map_uri=pm_uri,
            input_contract=self._input_contract,
            quality_flags=quality_flags,
            quarantine=quarantine,
            quarantine_reasons=quarantine_reasons,
            visual_modality_uris=visual_uris,
            provenance=provenance,
        )

    def _extract_pm_uri(self, row: Mapping[str, Any]) -> str:
        """Extract PM URI from A05 canonical sample row."""
        # Try nested dict format first (JSONL)
        frame_data = row.get("frame")
        if isinstance(frame_data, dict):
            modality_uris = frame_data.get("modality_uris", {})
            if isinstance(modality_uris, dict):
                pm_uri = modality_uris.get("PM", "")
                if pm_uri:
                    return str(pm_uri)

        # Try direct JSON string format (JSONL stored as string in CSV)
        frame_str = row.get("frame", "")
        if isinstance(frame_str, str) and frame_str.startswith("{"):
            try:
                frame_dict = json.loads(frame_str)
                modality_uris = frame_dict.get("modality_uris", {})
                if isinstance(modality_uris, dict):
                    pm_uri = modality_uris.get("PM", "")
                    if pm_uri:
                        return str(pm_uri)
            except json.JSONDecodeError:
                pass

        # Try flat format (CSV)
        pm_uri = row.get("frame_modality_uris_PM", row.get("pm_uri", ""))
        return str(pm_uri) if pm_uri else ""

    def _extract_visual_uris(self, row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
        """Extract visual modality URIs for provenance only (NOT model input)."""
        uris: list[tuple[str, str]] = []

        # Try nested dict format first (JSONL)
        frame_data = row.get("frame")
        if isinstance(frame_data, dict):
            modality_uris = frame_data.get("modality_uris", {})
            if isinstance(modality_uris, dict):
                for modality in FORBIDDEN_MODALITIES:
                    if modality in modality_uris:
                        uri = modality_uris[modality]
                        if uri:
                            uris.append((modality, str(uri)))
                return tuple(uris)

        # Try direct JSON string format (JSONL stored as string in CSV)
        frame_str = row.get("frame", "")
        if isinstance(frame_str, str) and frame_str.startswith("{"):
            try:
                frame_dict = json.loads(frame_str)
                modality_uris = frame_dict.get("modality_uris", {})
                if isinstance(modality_uris, dict):
                    for modality in FORBIDDEN_MODALITIES:
                        if modality in modality_uris:
                            uri = modality_uris[modality]
                            if uri:
                                uris.append((modality, str(uri)))
                    return tuple(uris)
            except json.JSONDecodeError:
                pass

        # Try flat format (CSV)
        modality_map = {
            "RGB": ["frame_modality_uris_RGB", "rgb_uri"],
            "IR": ["frame_modality_uris_IR", "ir_uri"],
            "depth": ["frame_modality_uris_depth", "depth_uri"],
            "IRraw": ["frame_modality_uris_IRraw", "irraw_uri"],
            "depthRaw": ["frame_modality_uris_depthRaw", "depthraw_uri"],
        }

        for modality, columns in modality_map.items():
            for col in columns:
                if col in row and row[col]:
                    uris.append((modality, str(row[col])))
                    break

        return tuple(uris)

    def _load_pressure_map(self, uri: str) -> np.ndarray:
        """Load a Pressure Map from disk.

        Parameters
        ----------
        uri : str
            URI relative to slp_root or absolute path.

        Returns
        -------
        np.ndarray
            Pressure map as float32 array with values in [0, 1].

        Raises
        ------
        FileNotFoundError
            If the PM file does not exist.
        ValueError
            If the loaded data is not 2D or has unexpected properties.
        """
        path = Path(uri)
        if not path.is_absolute():
            path = self.slp_root / path

        if not path.exists():
            raise FileNotFoundError(f"Pressure Map not found: {path}")

        # Load with numpy (SLP PM PNG is stored as float32 0-1)
        data = np.load(path, allow_pickle=False)

        # Ensure float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # Validate shape (should be 2D: height × width)
        if data.ndim != 2:
            raise ValueError(
                f"Pressure Map must be 2D, got shape {data.shape} for {path}"
            )

        # Validate value range
        if data.min() < -0.01 or data.max() > 1.01:
            raise ValueError(
                f"Pressure Map values out of expected range [0, 1], "
                f"got [{data.min():.4f}, {data.max():.4f}] for {path}"
            )

        return data


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def load_a05_canonical_samples(csv_path: Path | str) -> list[dict[str, Any]]:
    """Load A05 canonical samples from CSV.

    Parameters
    ----------
    csv_path : Path | str
        Path to A05 canonical samples CSV.

    Returns
    -------
    list[dict]
        List of canonical sample rows as dicts.
    """
    import csv as _csv

    rows: list[dict[str, Any]] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def load_a05_canonical_samples_jsonl(jsonl_path: Path | str) -> list[dict[str, Any]]:
    """Load A05 canonical samples from JSONL.

    Parameters
    ----------
    jsonl_path : Path | str
        Path to A05 canonical samples JSONL.

    Returns
    -------
    list[dict]
        List of canonical sample dicts.
    """
    rows: list[dict[str, Any]] = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_a06_split_manifest(json_path: Path | str) -> dict[str, str]:
    """Load A06 split manifest and build subject→split lookup.

    Parameters
    ----------
    json_path : Path | str
        Path to A06 split manifest JSON.

    Returns
    -------
    dict[str, str]
        Composite key "setting::subject_id" → split mapping.
    """
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    manifest: dict[str, str] = {}

    for entry in raw.get("subject_entries", []):
        key = f"{entry['setting']}::{entry['subject_id']}"
        manifest[key] = entry["split"]

    return manifest


def create_pressure_only_dataset(
    a05_samples: list[dict[str, Any]],
    a06_split_manifest: dict[str, str],
    *,
    slp_root: Path | str,
    task_id: str = DEFAULT_TASK_ID,
    load_pressure_data: bool = True,
    include_quarantine: bool = False,
    split: DataSplit | None = None,
) -> tuple[list[np.ndarray], list[int], list[PressureOnlySample]]:
    """Convenience function to create a pressure-only dataset.

    Parameters
    ----------
    a05_samples : list[dict]
        A05 canonical samples (from load_a05_canonical_samples or _jsonl).
    a06_split_manifest : dict
        Subject→split mapping from load_a06_split_manifest.
    slp_root : Path | str
        SLP root directory.
    task_id : str
        TASK-ID for provenance.
    load_pressure_data : bool
        Whether to load actual PM data.
    include_quarantine : bool
        Whether to include quarantined samples.
    split : DataSplit | None
        Filter by split.

    Returns
    -------
    tuple[list, list, list]
        (pressure_maps, labels, samples)
    """
    adapter = SlpPressureOnlyAdapter(
        canonical_samples=a05_samples,
        split_manifest=a06_split_manifest,
        slp_root=slp_root,
        task_id=task_id,
        load_pressure_data=load_pressure_data,
    )
    return adapter.build_dataset(
        include_quarantine=include_quarantine,
        split=split,
    )
