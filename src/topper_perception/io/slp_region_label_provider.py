"""Region Label Provider Interface for SLP Pressure-only experiments.

This module provides the interface contract for future ground truth integration.
NO ground truth is generated in this task (B01 contract).

The Region Label Provider reads region labels from external sources (A17 freeze)
and provides them to the training pipeline in a standardized format.

Interface requirements (TASK-SLP-B01 contract):
- region label URI
- label schema version
- label quality tier
- review status
- ignore / uncertain mask
- split provenance
- subject isolation validation

Region categories are read from external schema/config (not designed here).
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

PROVIDER_VERSION = "slp_region_label_provider_v0.1"

#: Default path to region schema (relative to repo root).
DEFAULT_REGION_SCHEMA_PATH = "configs/annotations/slp_region_annotation_v0.1.schema.json"

#: Label tiers that can be used for training (from A09).
TRAINABLE_TIERS: frozenset[str] = frozenset({"R2", "R3"})

#: Label tiers that require review (from A09).
REVIEW_REQUIRED_TIERS: frozenset[str] = frozenset({"R0", "R1", "R2", "R3"})

#: Review statuses that indicate the label is ready for training.
ACCEPTED_REVIEW_STATUSES: frozenset[str] = frozenset({
    "accepted",
    "edited",
    "adjudicated",
})

#: Review statuses that indicate the label requires review or is uncertain.
PENDING_REVIEW_STATUSES: frozenset[str] = frozenset({
    "pending",
    "uncertain",
})

#: Review statuses that indicate the label should not be used.
REJECTED_REVIEW_STATUSES: frozenset[str] = frozenset({
    "rejected",
})


class LabelTier(str, Enum):
    """Label tier from A09/A17."""
    R0 = "R0"  # Geometric proposal (not training-ready)
    R1 = "R1"  # Refined proposal (not training-ready)
    R2 = "R2"  # Human reviewed (training candidate)
    R3 = "R3"  # Human consensus (training primary)


class ReviewStatus(str, Enum):
    """Review status for a region label."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    EDITED = "edited"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"
    ADJUDICATED = "adjudicated"


class QualityTier(str, Enum):
    """Quality tier for region labels."""
    HIGH = "high"      # R3 + accepted
    MEDIUM = "medium"  # R2 + accepted
    LOW = "low"        # R0/R1 or pending/uncertain
    REJECTED = "rejected"  # rejected


@dataclass(frozen=True, slots=True)
class RegionLabel:
    """One region label for a sample.

    This is a read-only view of a label from the A17 frozen manifest.
    It does not generate or modify labels.
    """

    # Identification
    annotation_id: str
    sample_id: str
    setting: str
    subject_id: str
    cover_condition: str
    frame_index: int

    # Label metadata
    region_id: str
    label_tier: str
    label_source: str
    review_status: str

    # Polygon data
    polygon: np.ndarray  # Shape: (N, 2) for N vertices

    # Quality
    alignment_confidence: float
    anatomical_confidence: float
    quality_flags: tuple[str, ...]

    # Provenance
    provenance: LabelProvenance

    # Masks
    is_ignore: bool = False  # Label marked as ignore (poor quality, OOD, etc.)
    is_uncertain: bool = False  # Label has uncertain status

    def as_dict(self) -> dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "sample_id": self.sample_id,
            "setting": self.setting,
            "subject_id": self.subject_id,
            "cover_condition": self.cover_condition,
            "frame_index": self.frame_index,
            "region_id": self.region_id,
            "label_tier": self.label_tier,
            "label_source": self.label_source,
            "review_status": self.review_status,
            "polygon": self.polygon.tolist() if self.polygon is not None else None,
            "alignment_confidence": self.alignment_confidence,
            "anatomical_confidence": self.anatomical_confidence,
            "quality_flags": list(self.quality_flags),
            "provenance": self.provenance.as_dict(),
            "is_ignore": self.is_ignore,
            "is_uncertain": self.is_uncertain,
        }


@dataclass(frozen=True, slots=True)
class LabelProvenance:
    """Provenance information for a region label."""
    source_artifacts: tuple[str, ...]
    generator: str
    created_at: str
    algorithm_version: str | None = None
    parameter_hash: str | None = None


@dataclass(frozen=True, slots=True)
class SampleLabels:
    """All region labels for one sample.

    This groups labels by sample_id for efficient lookup during training.
    """
    sample_id: str
    setting: str
    subject_id: str
    cover_condition: str
    frame_index: int
    split: str  # From A06 split

    # All labels for this sample
    labels: tuple[RegionLabel, ...]

    # Masks for training
    ignore_mask: np.ndarray | None = None  # Binary mask: 1 = ignore
    uncertain_mask: np.ndarray | None = None  # Binary mask: 1 = uncertain

    # Aggregated quality
    quality_tier: str = QualityTier.LOW.value

    # Subject isolation check
    subject_isolation_verified: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "setting": self.setting,
            "subject_id": self.subject_id,
            "cover_condition": self.cover_condition,
            "frame_index": self.frame_index,
            "split": self.split,
            "labels": [l.as_dict() for l in self.labels],
            "label_count": len(self.labels),
            "quality_tier": self.quality_tier,
            "subject_isolation_verified": self.subject_isolation_verified,
        }


@dataclass(frozen=True, slots=True)
class LabelManifest:
    """Manifest of available region labels.

    This is the index for the A17 frozen label set.
    """
    manifest_version: str
    schema_version: str
    created_at: str
    total_annotations: int
    trainable_annotations: int  # R2/R3 + accepted
    by_tier: dict[str, int]
    by_review_status: dict[str, int]
    by_quality_tier: dict[str, int]
    by_split: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "total_annotations": self.total_annotations,
            "trainable_annotations": self.trainable_annotations,
            "by_tier": self.by_tier,
            "by_review_status": self.by_review_status,
            "by_quality_tier": self.by_quality_tier,
            "by_split": self.by_split,
        }


# ---------------------------------------------------------------------------
# Schema Loader
# ---------------------------------------------------------------------------


class RegionSchema:
    """Loader for region category definitions from external schema.

    This reads the region schema from configs/annotations/ without modifying it.
    """

    def __init__(
        self,
        schema_path: Path | str | None = None,
    ) -> None:
        if schema_path is None:
            # Try multiple possible locations
            possible_paths = [
                Path(__file__).parent.parent.parent.parent / DEFAULT_REGION_SCHEMA_PATH,  # repo root
                Path(__file__).parent.parent / DEFAULT_REGION_SCHEMA_PATH,  # src root
            ]
            for p in possible_paths:
                if p.exists():
                    schema_path = p
                    break
            else:
                schema_path = possible_paths[0]  # Use the first one (will fail with clear error)

        self.path = Path(schema_path)
        self._schema: dict[str, Any] | None = None
        self._region_ids: tuple[str, ...] | None = None

    @property
    def schema(self) -> dict[str, Any]:
        """Load and cache the schema."""
        if self._schema is None:
            if not self.path.exists():
                raise FileNotFoundError(f"Schema not found: {self.path}")
            self._schema = json.loads(self.path.read_text(encoding="utf-8"))
        return self._schema

    @property
    def region_ids(self) -> tuple[str, ...]:
        """Get the list of valid region IDs from the schema."""
        if self._region_ids is None:
            region_enum = self.schema.get("properties", {}).get("region_id", {})
            self._region_ids = tuple(region_enum.get("enum", []))
        return self._region_ids

    def is_valid_region_id(self, region_id: str) -> bool:
        """Check if a region_id is valid according to the schema."""
        return region_id in self.region_ids

    def get_region_metadata(self, region_id: str) -> dict[str, Any]:
        """Get metadata for a region from the schema."""
        # Schema doesn't have per-region metadata, return basic info
        return {
            "region_id": region_id,
            "schema_path": str(self.path),
            "schema_version": self.schema.get("$id", ""),
        }


# ---------------------------------------------------------------------------
# Label Provider
# ---------------------------------------------------------------------------


class RegionLabelProviderError(Exception):
    """Base exception for region label provider errors."""
    pass


class LabelNotFoundError(RegionLabelProviderError):
    """Label for a sample was not found."""
    pass


class LabelManifestEmptyError(RegionLabelProviderError):
    """Label manifest is empty (ground truth not yet frozen)."""
    pass


class LabelValidationError(RegionLabelProviderError):
    """Label validation failed."""
    pass


class RegionIdValidationError(RegionLabelProviderError):
    """Unknown region_id in label manifest."""
    pass


class PolygonValidationError(RegionLabelProviderError):
    """Malformed polygon in label manifest."""
    pass


class RegionLabelProvider:
    """Provider for region labels from A17 frozen manifest.

    This provider reads region labels from an external manifest and
    provides them to the training pipeline. It does NOT generate
    labels or modify the manifest.

    Design rules:
    * Labels are read-only from the provider's perspective.
    * The manifest path is validated at construction.
    * Samples without labels fail explicitly, not silently.
    * Subject isolation is verified before providing labels.
    * Training-ready labels (R2/R3 + accepted) are distinguished.

    Parameters
    ----------
    label_manifest_path : Path | str
        Path to the A17 frozen label manifest JSON.
    a06_split_manifest : Mapping[str, str]
        Subject→split mapping from A06.
    region_schema : RegionSchema | None
        Schema for validating region IDs. If None, uses default.
    require_training_ready : bool
        If True, only provide training-ready labels (R2/R3 + accepted).
        If False, provide all labels including R0/R1 for analysis.
    task_id : str
        TASK-ID for provenance.
    """

    def __init__(
        self,
        label_manifest_path: Path | str,
        a06_split_manifest: Mapping[str, str],
        *,
        region_schema: RegionSchema | None = None,
        require_training_ready: bool = True,
        task_id: str = "TASK-SLP-B01-PRESSURE-ONLY-INFRA-v0.1",
        now: datetime | None = None,
    ) -> None:
        self.manifest_path = Path(label_manifest_path)
        self._a06_split = dict(a06_split_manifest)
        self.region_schema = region_schema or RegionSchema()
        self.require_training_ready = require_training_ready
        self.task_id = task_id
        self.created_at = (now or datetime.now(timezone.utc)).isoformat()

        # Validate manifest exists
        if not self.manifest_path.exists():
            raise LabelManifestEmptyError(
                f"Label manifest not found: {self.manifest_path}. "
                "Ground truth has not been frozen yet (A17 pending)."
            )

        # Load manifest
        self._manifest: dict[str, Any] = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )

        # Build index by sample_id for fast lookup
        self._sample_index: dict[str, list[dict[str, Any]]] = {}
        self._build_sample_index()

        # Compute manifest summary
        self._summary = self._compute_summary()

    def _build_sample_index(self) -> None:
        """Build an index of labels by sample_id."""
        for entry in self._manifest.get("annotations", []):
            sample_id = str(entry.get("sample_id", ""))
            if sample_id not in self._sample_index:
                self._sample_index[sample_id] = []
            self._sample_index[sample_id].append(entry)

    def _compute_summary(self) -> LabelManifest:
        """Compute summary statistics from the manifest."""
        annotations = self._manifest.get("annotations", [])

        by_tier: dict[str, int] = {}
        by_review: dict[str, int] = {}
        by_quality: dict[str, int] = {}
        by_split: dict[str, int] = {}
        trainable = 0

        for entry in annotations:
            tier = str(entry.get("label_tier", "unknown"))
            review = str(entry.get("review_status", "unknown"))
            quality = self._compute_quality_tier(entry)
            split = self._get_split_for_sample(entry)

            by_tier[tier] = by_tier.get(tier, 0) + 1
            by_review[review] = by_review.get(review, 0) + 1
            by_quality[quality] = by_quality.get(quality, 0) + 1
            by_split[split] = by_split.get(split, 0) + 1

            # Training-ready: R2/R3 + accepted/edited/adjudicated
            if tier in TRAINABLE_TIERS and review in ACCEPTED_REVIEW_STATUSES:
                trainable += 1

        return LabelManifest(
            manifest_version=self._manifest.get("manifest_version", "unknown"),
            schema_version=self._manifest.get("schema_version", "unknown"),
            created_at=self._manifest.get("created_at", self.created_at),
            total_annotations=len(annotations),
            trainable_annotations=trainable,
            by_tier=by_tier,
            by_review_status=by_review,
            by_quality_tier=by_quality,
            by_split=by_split,
        )

    def _compute_quality_tier(self, entry: dict[str, Any]) -> str:
        """Compute quality tier for a label entry."""
        tier = str(entry.get("label_tier", "R0"))
        review = str(entry.get("review_status", "pending"))

        if review == "rejected":
            return QualityTier.REJECTED.value

        if tier == "R3" and review in ACCEPTED_REVIEW_STATUSES:
            return QualityTier.HIGH.value
        if tier == "R2" and review in ACCEPTED_REVIEW_STATUSES:
            return QualityTier.MEDIUM.value

        return QualityTier.LOW.value

    def _get_split_for_sample(self, entry: dict[str, Any]) -> str:
        """Get split for a sample from A06 manifest."""
        setting = str(entry.get("setting", ""))
        subject_id = str(entry.get("subject_id", ""))
        key = f"{setting}::{subject_id}"
        return self._a06_split.get(key, "unknown")

    # -- Public API -----------------------------------------------------------

    @property
    def summary(self) -> LabelManifest:
        """Get summary statistics of the label manifest."""
        return self._summary

    def get_sample_labels(
        self,
        sample_id: str,
        require_training_ready: bool | None = None,
    ) -> SampleLabels:
        """Get all labels for one sample.

        Parameters
        ----------
        sample_id : str
            The sample_id to look up.
        require_training_ready : bool | None
            Override the instance-level require_training_ready setting.
            - True: only return R2/R3 + accepted/edited/adjudicated labels.
            - False: return all labels including R0/R1 (for analysis).
            - None: use the instance-level setting.

        Returns
        -------
        SampleLabels
            All labels for the sample.

        Raises
        ------
        LabelNotFoundError
            If no labels exist for the sample.
        LabelValidationError
            If validation fails or no training-ready labels are found.
        """
        # Resolve per-call override or use instance default
        require_trainable = (
            require_training_ready
            if require_training_ready is not None
            else self.require_training_ready
        )

        entries = self._sample_index.get(sample_id, [])

        if not entries:
            raise LabelNotFoundError(
                f"No labels found for sample_id: {sample_id}. "
                "Ground truth may not exist for this sample."
            )

        # Parse entries into RegionLabel objects
        labels: list[RegionLabel] = []
        for entry in entries:
            try:
                label = self._parse_label_entry(entry)
            except (RegionIdValidationError, PolygonValidationError, LabelValidationError) as e:
                # Fail-closed: malformed labels must not be silently skipped
                raise LabelValidationError(
                    f"Invalid label entry for sample_id={sample_id}: {e}"
                ) from e

            # Filter if training-ready is required
            if require_trainable:
                is_trainable = (
                    label.label_tier in TRAINABLE_TIERS
                    and label.review_status in ACCEPTED_REVIEW_STATUSES
                )
                if not is_trainable:
                    continue

            labels.append(label)

        if not labels:
            if require_trainable:
                raise LabelNotFoundError(
                    f"No training-ready labels found for sample_id: {sample_id}. "
                    f"Found {len(entries)} entries but none meet training criteria "
                    f"(tier in {sorted(TRAINABLE_TIERS)} + review in "
                    f"{sorted(ACCEPTED_REVIEW_STATUSES)})."
                )
            else:
                raise LabelNotFoundError(
                    f"No labels found for sample_id: {sample_id}."
                )

        # Get sample metadata from first entry
        first = entries[0]
        setting = str(first.get("setting", ""))
        subject_id = str(first.get("subject_id", ""))
        cover = str(first.get("cover_condition", ""))
        frame_index = int(first.get("frame_index", 0))
        split = self._get_split_for_sample(first)

        # Verify subject isolation
        isolation_verified = self._verify_subject_isolation(subject_id, setting)

        # Build masks
        ignore_mask, uncertain_mask = self._build_masks(labels)

        # Compute quality tier
        quality_tier = self._aggregate_quality_tier(labels)

        return SampleLabels(
            sample_id=sample_id,
            setting=setting,
            subject_id=subject_id,
            cover_condition=cover,
            frame_index=frame_index,
            split=split,
            labels=tuple(labels),
            ignore_mask=ignore_mask,
            uncertain_mask=uncertain_mask,
            quality_tier=quality_tier,
            subject_isolation_verified=isolation_verified,
        )

    def iter_sample_labels(
        self,
        split: str | None = None,
        quality_tier: str | None = None,
        require_training_ready: bool | None = None,
    ) -> Iterator[SampleLabels]:
        """Iterate over all sample labels.

        Parameters
        ----------
        split : str | None
            If specified, only return samples from this split.
        quality_tier : str | None
            If specified, only return samples with this quality tier.
        require_training_ready : bool | None
            Override the instance-level require_training_ready setting.
        """
        require_trainable = (
            require_training_ready
            if require_training_ready is not None
            else self.require_training_ready
        )

        for sample_id in self._sample_index:
            try:
                sample_labels = self.get_sample_labels(
                    sample_id, require_training_ready=require_training_ready
                )
            except LabelNotFoundError:
                continue

            # Apply filters
            if split is not None and sample_labels.split != split:
                continue
            if quality_tier is not None and sample_labels.quality_tier != quality_tier:
                continue
            if require_trainable and sample_labels.quality_tier == QualityTier.REJECTED.value:
                continue

            yield sample_labels

    def has_labels(self, sample_id: str) -> bool:
        """Check if labels exist for a sample."""
        entries = self._sample_index.get(sample_id, [])
        if not entries:
            return False

        if self.require_training_ready:
            for entry in entries:
                tier = str(entry.get("label_tier", ""))
                review = str(entry.get("review_status", ""))
                if tier in TRAINABLE_TIERS and review in ACCEPTED_REVIEW_STATUSES:
                    return True
            return False

        return True

    # -- Private helpers ------------------------------------------------------

    def _parse_label_entry(self, entry: dict[str, Any]) -> RegionLabel | None:
        """Parse a manifest entry into a RegionLabel.

        Raises
        ------
        RegionIdValidationError
            If region_id is unknown according to schema.
        PolygonValidationError
            If polygon data is malformed.
        """
        # Validate region_id against schema (fail-closed)
        region_id = str(entry.get("region_id", ""))
        if not region_id:
            raise LabelValidationError(
                f"Missing region_id in annotation {entry.get('annotation_id', 'unknown')}"
            )
        if not self.region_schema.is_valid_region_id(region_id):
            raise RegionIdValidationError(
                f"Unknown region_id '{region_id}' in annotation {entry.get('annotation_id', 'unknown')}. "
                f"Valid region_ids: {self.region_schema.region_ids}"
            )

        # Validate sample_id
        sample_id = str(entry.get("sample_id", ""))
        if not sample_id:
            raise LabelValidationError(
                f"Missing sample_id in annotation for region_id={region_id}"
            )

        # Validate annotation_id
        annotation_id = str(entry.get("annotation_id", ""))
        if not annotation_id:
            raise LabelValidationError(
                f"Missing annotation_id for sample_id={sample_id}, region_id={region_id}"
            )

        # Parse polygon (fail-closed for malformed)
        polygon_data = entry.get("final_polygon") or entry.get("proposal_polygon")
        if polygon_data is None:
            raise PolygonValidationError(
                f"Missing polygon (final_polygon/proposal_polygon) for "
                f"annotation_id={annotation_id}"
            )

        try:
            polygon = np.array(polygon_data, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise PolygonValidationError(
                f"Invalid polygon data for annotation_id={annotation_id}: {e}"
            )

        if polygon.ndim != 2:
            raise PolygonValidationError(
                f"Polygon must be 2D array, got {polygon.ndim}D for "
                f"annotation_id={annotation_id}"
            )
        if polygon.shape[1] != 2:
            raise PolygonValidationError(
                f"Polygon must have shape (N, 2), got {polygon.shape} for "
                f"annotation_id={annotation_id}"
            )
        if polygon.shape[0] < 3:
            raise PolygonValidationError(
                f"Polygon must have at least 3 vertices, got {polygon.shape[0]} for "
                f"annotation_id={annotation_id}"
            )

        # Validate split (fail-closed for unknown split)
        setting = str(entry.get("setting", ""))
        subject_id = str(entry.get("subject_id", ""))
        key = f"{setting}::{subject_id}"
        if key and key not in self._a06_split:
            # Unknown split in A06 - fail-closed
            raise LabelValidationError(
                f"Unknown split for {key} (sample_id={sample_id}). "
                "Subject must be in A06 split manifest."
            )

        # Parse provenance
        prov_data = entry.get("provenance", {})
        provenance = LabelProvenance(
            source_artifacts=tuple(prov_data.get("source_artifacts", [])),
            generator=str(prov_data.get("generator", "")),
            created_at=str(prov_data.get("created_at", "")),
            algorithm_version=prov_data.get("algorithm_version"),
            parameter_hash=prov_data.get("parameter_hash"),
        )

        # Quality flags
        flags_raw = entry.get("quality_flags", [])
        if isinstance(flags_raw, str):
            flags = tuple(f.strip() for f in flags_raw.split(";") if f.strip())
        else:
            flags = tuple(flags_raw)

        # Determine ignore/uncertain status
        review_status = str(entry.get("review_status", "pending"))
        is_ignore = (
            review_status in REJECTED_REVIEW_STATUSES
            or "rejected" in flags
        )
        is_uncertain = (
            review_status in PENDING_REVIEW_STATUSES
            or review_status == "uncertain"
        )

        return RegionLabel(
            annotation_id=annotation_id,
            sample_id=sample_id,
            setting=setting,
            subject_id=subject_id,
            cover_condition=str(entry.get("cover_condition", "")),
            frame_index=int(entry.get("frame_index", 0)),
            region_id=region_id,
            label_tier=str(entry.get("label_tier", "R0")),
            label_source=str(entry.get("label_source", "")),
            review_status=review_status,
            polygon=polygon,
            alignment_confidence=float(entry.get("alignment_confidence", 0.0)),
            anatomical_confidence=float(entry.get("anatomical_confidence", 0.0)),
            quality_flags=flags,
            provenance=provenance,
            is_ignore=is_ignore,
            is_uncertain=is_uncertain,
        )

    def _build_masks(
        self,
        labels: Sequence[RegionLabel],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Build ignore and uncertain masks from labels.

        Returns (ignore_mask, uncertain_mask) as binary masks.
        These are spatial masks matching the PM dimensions.
        """
        if not labels:
            return None, None

        # For now, return None since we don't have rasterization yet
        # This will be implemented when B01 actually loads labels
        return None, None

    def _aggregate_quality_tier(self, labels: Sequence[RegionLabel]) -> str:
        """Aggregate quality tier across all labels for a sample."""
        tiers = [self._compute_quality_tier_entry(l) for l in labels]

        if QualityTier.HIGH.value in tiers:
            return QualityTier.HIGH.value
        if QualityTier.MEDIUM.value in tiers:
            return QualityTier.MEDIUM.value
        if QualityTier.LOW.value in tiers:
            return QualityTier.LOW.value
        return QualityTier.REJECTED.value

    def _compute_quality_tier_entry(self, label: RegionLabel) -> str:
        """Compute quality tier for a single label."""
        if label.review_status in REJECTED_REVIEW_STATUSES:
            return QualityTier.REJECTED.value

        if label.label_tier == "R3" and label.review_status in ACCEPTED_REVIEW_STATUSES:
            return QualityTier.HIGH.value
        if label.label_tier == "R2" and label.review_status in ACCEPTED_REVIEW_STATUSES:
            return QualityTier.MEDIUM.value

        return QualityTier.LOW.value

    def _verify_subject_isolation(self, subject_id: str, setting: str) -> bool:
        """Verify that subject isolation is maintained for this label."""
        key = f"{setting}::{subject_id}"
        return key in self._a06_split


# ---------------------------------------------------------------------------
# Mock Provider for Testing
# ---------------------------------------------------------------------------


class MockRegionLabelProvider(RegionLabelProvider):
    """Mock provider that returns synthetic labels for testing.

    This is ONLY for unit testing and infrastructure development.
    NO synthetic labels should be used for actual model training.
    """

    def __init__(
        self,
        sample_ids: Sequence[str],
        region_schema: RegionSchema | None = None,
        task_id: str = "MOCK",
    ) -> None:
        # Set created_at first since it's used in _build_mock_manifest
        self.created_at = datetime.now(timezone.utc).isoformat()

        # Build mock manifest
        self.region_schema = region_schema or RegionSchema()
        mock_manifest = self._build_mock_manifest(sample_ids)

        # Initialize with a non-existent manifest path (will be overridden)
        self._mock_mode = True
        self._mock_samples = {s["sample_id"]: s for s in mock_manifest["annotations"]}

        # Call parent init with dummy path (will be overridden)
        self.manifest_path = Path("MOCK://in_memory")
        self._a06_split: dict[str, str] = {}
        self.require_training_ready = False
        self.task_id = task_id
        self._manifest = mock_manifest
        self._sample_index: dict[str, list[dict[str, Any]]] = {
            k: [v] for k, v in self._mock_samples.items()
        }
        self._summary = self._compute_summary()

    def _build_mock_manifest(
        self,
        sample_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Build a mock manifest with synthetic labels."""
        annotations = []
        region_ids = self.region_schema.region_ids

        for sample_id in sample_ids:
            # Parse sample_id to extract metadata
            # Format: slp::setting::subject_id::cover::frame_index
            parts = sample_id.split("::")
            if len(parts) >= 5:
                setting, subject_id, cover, frame_idx = (
                    parts[1], parts[2], parts[3], int(parts[4])
                )
            else:
                setting, subject_id, cover, frame_idx = "danaLab", "00001", "uncover", 1

            for region_id in region_ids:
                annotations.append({
                    "annotation_id": f"mock::{sample_id}::{region_id}",
                    "sample_id": sample_id,
                    "setting": setting,
                    "subject_id": subject_id,
                    "cover_condition": cover,
                    "frame_index": frame_idx,
                    "region_id": region_id,
                    "label_tier": "R2",
                    "label_source": "mock",
                    "review_status": "accepted",
                    "final_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "alignment_confidence": 0.9,
                    "anatomical_confidence": 0.8,
                    "quality_flags": [],
                    "provenance": {
                        "source_artifacts": ["mock"],
                        "generator": "MockRegionLabelProvider",
                        "created_at": self.created_at,
                    },
                })

        return {
            "manifest_version": "mock_v0.1",
            "schema_version": "slp_region_annotation_v0.1",
            "created_at": self.created_at,
            "annotations": annotations,
        }
