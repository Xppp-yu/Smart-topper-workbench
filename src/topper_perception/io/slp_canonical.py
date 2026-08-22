"""SLP Canonical Sample and Frame/Joint/Region adapter.

This module sits on top of the A03 Frame Master Index and the A04 Homography
audit and produces a stable per-frame Canonical Sample that the rest of the SLP
pipeline can consume without re-deriving modality URIs, joint provenance, or
the A04 geometry contract.

Design rules (mirroring the A05 task contract):

* The Frame layer is the only layer that knows about per-frame raw modality
  files. It is sourced from the A03 index without re-pairing files by sort
  order.
* The Joint layer records where the original 14-joint GT (J0) lives and
  carries the A04 homography contract for every modality. It never silently
  picks a semantic direction; the A04 direction status (``UNRESOLVED_*`` or
  ``BLOCKED_*``) is preserved verbatim.
* The Region layer is a placeholder. It only references the frozen
  ``slp_region_annotation_v0.1`` schema and never carries training
  annotations in A05; downstream tasks (A10--A17) must not write region
  truth into the canonical sample.
* Quality flags and quarantine decisions are explicit. Missing modalities,
  ambiguous frame matches, and ``BLOCKED_*`` homography rows all surface
  here. The adapter never silently imputes a missing modality or a
  homography direction.
* Provenance is required on every sample. A05 does not write split
  membership, review status, or model predictions back to the raw samples.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .slp_frame_index import (
    CANONICAL_MODALITIES,
    FRAME_INDEX_COLUMNS,
    SlpFrameIndexRow,
)
from .slp_homography_audit import (
    HOMOGRAPHY_AUDIT_COLUMNS,
    SlpHomographyAuditRow,
)
from .slp_inventory import (
    COVER_CONDITIONS,
    JOINT_COUNT,
    SETTINGS,
    iter_subject_directories,
    resolve_slp_root,
)


ADAPTER_VERSION = "slp_canonical_adapter_v0.1"
CANONICAL_SCHEMA_VERSION = "slp_canonical_sample_v0.1"
REGION_SCHEMA_VERSION = "slp_region_annotation_v0.1"
DEFAULT_TASK_ID = "TASK-SLP-A05-CANONICAL-ADAPTER-v0.1"
DEFAULT_GENERATOR = "topper_perception.io.slp_canonical.SlpCanonicalAdapter"

# Coordinate frame label for A05. A04 has not confirmed a semantic direction
# or an origin offset, so the canonical sample stays explicit about that.
RAW_COORDINATE_FRAME = "raw_dataset_pixel_coordinates_no_offset"
RAW_COORDINATE_ORIGIN_STATUS = "UNRESOLVED_RAW_DATASET_COORDINATES_NO_OFFSET_APPLIED"

# Joint layer status markers. These are explicit so downstream tasks cannot
# accidentally treat A05 canonical samples as containing J1 truth.
J0_GT_SOURCE = "manual_original"
J1_STATUS_NOT_GENERATED = (
    "not_generated_A04_direction_unresolved_see_homography_contract"
)
JOINT_PROVENANCE_J0 = "j0_only_j1_pending_a04_direction_resolution"

REGION_PLACEHOLDER_STATUS = (
    "A10_to_A17_pending_no_training_truth_generated_by_a05"
)
REGION_CAN_BE_USED_AS_TRAINING_TRUTH = False

HOMOGRAPHY_MODALITIES = ("RGB", "IR", "depth")
J0_MODALITIES = ("RGB", "IR")


# ---------------------------------------------------------------------------
# Layer data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HomographyContract:
    """Per-subject A04 geometry contract for one modality.

    All fields are surfaced verbatim from the A04 audit row so that A05 never
    silently reinterprets ``UNRESOLVED_*`` or ``BLOCKED_*`` direction states.
    """

    modality: str
    matrix_uri: str
    matrix_present: bool
    invertible: bool
    direction_status: str
    coordinate_origin_status: str
    probe_roundtrip_mean_error: float | None
    probe_roundtrip_max_error: float | None
    direct_joint_in_bounds_rate: float | None
    inverse_joint_in_bounds_rate: float | None
    error_codes: tuple[str, ...]
    blocked: bool
    unresolved_direction: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "modality": self.modality,
            "matrix_uri": self.matrix_uri,
            "matrix_present": self.matrix_present,
            "invertible": self.invertible,
            "direction_status": self.direction_status,
            "coordinate_origin_status": self.coordinate_origin_status,
            "probe_roundtrip_mean_error": self.probe_roundtrip_mean_error,
            "probe_roundtrip_max_error": self.probe_roundtrip_max_error,
            "direct_joint_in_bounds_rate": self.direct_joint_in_bounds_rate,
            "inverse_joint_in_bounds_rate": self.inverse_joint_in_bounds_rate,
            "error_codes": list(self.error_codes),
            "blocked": self.blocked,
            "unresolved_direction": self.unresolved_direction,
        }


@dataclass(frozen=True, slots=True)
class FrameLayer:
    """Per-frame raw modality URIs sourced from the A03 Frame Master Index."""

    sample_id: str
    setting: str
    subject_id: str
    cover_condition: str
    frame_index: int
    modality_uris: dict[str, str]
    missing_modalities: tuple[str, ...]
    expected_missing_modalities: tuple[str, ...]
    ambiguous_modalities: tuple[str, ...]
    uri_existence_flags: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "setting": self.setting,
            "subject_id": self.subject_id,
            "cover_condition": self.cover_condition,
            "frame_index": self.frame_index,
            "modality_uris": dict(self.modality_uris),
            "missing_modalities": list(self.missing_modalities),
            "expected_missing_modalities": list(self.expected_missing_modalities),
            "ambiguous_modalities": list(self.ambiguous_modalities),
            "uri_existence_flags": dict(self.uri_existence_flags),
        }


@dataclass(frozen=True, slots=True)
class JointLayer:
    """Joint layer: original 14-joint GT (J0) provenance + A04 contract.

    The A05 canonical sample does not generate J1 joints; the
    homography contract is carried so that later tasks (A07 / A18) can decide
    how (or whether) to derive J1 once A04 direction is resolved.
    """

    j0_source_uris: dict[str, str]
    j0_present: dict[str, bool]
    j0_artifact_count: int
    joint_provenance_status: str
    j1_status: str
    homography_contracts: dict[str, HomographyContract]

    def as_dict(self) -> dict[str, object]:
        return {
            "j0_source_uris": dict(self.j0_source_uris),
            "j0_present": dict(self.j0_present),
            "j0_artifact_count": self.j0_artifact_count,
            "joint_provenance_status": self.joint_provenance_status,
            "j1_status": self.j1_status,
            "homography_contracts": {
                modality: contract.as_dict()
                for modality, contract in self.homography_contracts.items()
            },
        }


@dataclass(frozen=True, slots=True)
class RegionLayer:
    """Region layer placeholder.

    A05 only carries the schema version and an explicit placeholder status.
    Real region annotations belong to A10--A17 and must not be back-filled
    into the canonical sample.
    """

    schema_version: str
    placeholder_status: str
    annotation_count: int
    annotations: tuple[dict[str, object], ...]
    can_be_used_as_training_truth: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "placeholder_status": self.placeholder_status,
            "annotation_count": self.annotation_count,
            "annotations": [dict(item) for item in self.annotations],
            "can_be_used_as_training_truth": self.can_be_used_as_training_truth,
        }


@dataclass(frozen=True, slots=True)
class Provenance:
    """How this canonical sample was produced."""

    task_id: str
    adapter_version: str
    canonical_schema_version: str
    generator: str
    created_at: str
    a03_frame_index_source: str
    a04_homography_audit_source: str
    slp_root: str
    pairing_method: str
    semantic_direction_auto_selected: bool
    coordinate_origin_auto_shifted: bool
    silent_imputation: bool
    subject_split_applied: bool
    review_status_applied: bool
    model_prediction_applied: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CanonicalSample:
    """A stable per-frame SLP sample with Frame/Joint/Region layers."""

    sample_id: str
    setting: str
    subject_id: str
    cover_condition: str
    frame_index: int
    coordinate_frame: str
    coordinate_origin_status: str
    frame: FrameLayer
    joint: JointLayer
    region: RegionLayer
    provenance: Provenance
    quality_flags: tuple[str, ...]
    quarantine: bool
    quarantine_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "setting": self.setting,
            "subject_id": self.subject_id,
            "cover_condition": self.cover_condition,
            "frame_index": self.frame_index,
            "coordinate_frame": self.coordinate_frame,
            "coordinate_origin_status": self.coordinate_origin_status,
            "frame": self.frame.as_dict(),
            "joint": self.joint.as_dict(),
            "region": self.region.as_dict(),
            "provenance": self.provenance.as_dict(),
            "quality_flags": list(self.quality_flags),
            "quarantine": self.quarantine,
            "quarantine_reasons": list(self.quarantine_reasons),
        }


# ---------------------------------------------------------------------------
# URI / value helpers
# ---------------------------------------------------------------------------


def _split_csv(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value)
    if not text:
        return []
    return [item for item in text.split(";") if item]


def _relative_or_absolute(path: Path, slp_root: Path) -> str:
    """Return a POSIX URI, relative to ``slp_root`` if possible."""
    try:
        return path.relative_to(slp_root).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_modality_uri(row_values: Mapping[str, object], modality: str) -> str:
    column = {
        "RGB": "rgb_uri",
        "IR": "ir_uri",
        "IRraw": "irraw_uri",
        "depth": "depth_uri",
        "depthRaw": "depthraw_uri",
        "PM": "pm_uri",
    }[modality]
    value = row_values.get(column, "")
    return str(value) if value is not None else ""


def _check_uri_existence(slp_root: Path, uri: str) -> str:
    if not uri:
        return "absent"
    candidate = Path(uri)
    if not candidate.is_absolute():
        candidate = slp_root / uri
    return "present" if candidate.is_file() else "missing_on_disk"


# ---------------------------------------------------------------------------
# A04 contract parsing
# ---------------------------------------------------------------------------


def _parse_homography_contract(row: SlpHomographyAuditRow) -> HomographyContract:
    values = row.as_dict()
    modality = str(values.get("modality", ""))
    direction_status = str(values.get("direction_status", ""))
    error_codes = tuple(_split_csv(values.get("error_codes")))
    blocked = direction_status.startswith("BLOCKED_") or not bool(values.get("invertible"))
    unresolved = direction_status.startswith("UNRESOLVED_")
    return HomographyContract(
        modality=modality,
        matrix_uri=str(values.get("matrix_uri", "")),
        matrix_present=bool(values.get("matrix_present")),
        invertible=bool(values.get("invertible")),
        direction_status=direction_status,
        coordinate_origin_status=str(values.get("coordinate_origin_status", "")),
        probe_roundtrip_mean_error=_to_optional_float(values.get("probe_roundtrip_mean_error")),
        probe_roundtrip_max_error=_to_optional_float(values.get("probe_roundtrip_max_error")),
        direct_joint_in_bounds_rate=_to_optional_float(values.get("direct_joint_in_bounds_rate")),
        inverse_joint_in_bounds_rate=_to_optional_float(values.get("inverse_joint_in_bounds_rate")),
        error_codes=error_codes,
        blocked=blocked,
        unresolved_direction=unresolved,
    )


def _to_optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _index_homography_contracts(
    rows: Iterable[SlpHomographyAuditRow],
) -> dict[tuple[str, str, str], HomographyContract]:
    """Build a (setting, subject_id, modality) -> contract lookup."""
    contracts: dict[tuple[str, str, str], HomographyContract] = {}
    for row in rows:
        values = row.as_dict()
        key = (
            str(values.get("setting", "")),
            str(values.get("subject_id", "")),
            str(values.get("modality", "")),
        )
        contracts[key] = _parse_homography_contract(row)
    return contracts


def _placeholder_homography_contract(modality: str) -> HomographyContract:
    return HomographyContract(
        modality=modality,
        matrix_uri="",
        matrix_present=False,
        invertible=False,
        direction_status="UNRESOLVED_NO_A04_AUDIT_ROW",
        coordinate_origin_status=RAW_COORDINATE_ORIGIN_STATUS,
        probe_roundtrip_mean_error=None,
        probe_roundtrip_max_error=None,
        direct_joint_in_bounds_rate=None,
        inverse_joint_in_bounds_rate=None,
        error_codes=("missing_a04_audit_row",),
        blocked=True,
        unresolved_direction=True,
    )


# ---------------------------------------------------------------------------
# J0 source URI helpers
# ---------------------------------------------------------------------------


def _j0_source_uri(slp_root: Path, subject_dir: Path, modality: str) -> str:
    if modality not in J0_MODALITIES:
        raise ValueError(f"J0 is not defined for modality: {modality}")
    path = subject_dir / f"joints_gt_{modality}.mat"
    return _relative_or_absolute(path, slp_root)


def _subject_dir_for(slp_root: Path, setting: str, subject_id: str) -> Path:
    return slp_root / setting / subject_id


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SlpCanonicalAdapter:
    """Build Canonical Sample objects from A03 rows and an A04 audit."""

    def __init__(
        self,
        *,
        slp_root: Path,
        a03_frame_rows: Iterable[SlpFrameIndexRow],
        a04_audit_rows: Iterable[SlpHomographyAuditRow] | None = None,
        task_id: str = DEFAULT_TASK_ID,
        generator: str = DEFAULT_GENERATOR,
        a03_frame_index_source: str = "in_memory",
        a04_audit_source: str = "in_memory",
        now: datetime | None = None,
    ) -> None:
        self.slp_root = slp_root
        self._a03_rows = [row for row in a03_frame_rows]
        self._homography_contracts = _index_homography_contracts(a04_audit_rows or [])
        self.task_id = task_id
        self.generator = generator
        self.a03_frame_index_source = a03_frame_index_source
        self.a04_audit_source = a04_audit_source
        self.created_at = (now or datetime.now(timezone.utc)).isoformat()
        self.subject_dir_cache: dict[tuple[str, str], Path] = {}

    # -- public API --------------------------------------------------------

    def iter_canonical_samples(self) -> Iterator[CanonicalSample]:
        provenance_template = Provenance(
            task_id=self.task_id,
            adapter_version=ADAPTER_VERSION,
            canonical_schema_version=CANONICAL_SCHEMA_VERSION,
            generator=self.generator,
            created_at=self.created_at,
            a03_frame_index_source=self.a03_frame_index_source,
            a04_homography_audit_source=self.a04_audit_source,
            slp_root=str(self.slp_root.resolve()),
            pairing_method="explicit_frame_index_join_via_a03",
            semantic_direction_auto_selected=False,
            coordinate_origin_auto_shifted=False,
            silent_imputation=False,
            subject_split_applied=False,
            review_status_applied=False,
            model_prediction_applied=False,
        )
        for row in self._a03_rows:
            yield self.build_canonical_sample(row, provenance_template=provenance_template)

    def build_canonical_sample(
        self,
        row: SlpFrameIndexRow,
        *,
        provenance_template: Provenance | None = None,
    ) -> CanonicalSample:
        values = row.as_dict()
        sample_id = str(values["sample_id"])
        setting = str(values["setting"])
        subject_id = str(values["subject_id"])
        cover_condition = str(values["cover_condition"])
        frame_index = int(values["frame_index"])

        # Frame layer ------------------------------------------------------------
        modality_uris: dict[str, str] = {}
        uri_existence_flags: dict[str, str] = {}
        for modality in CANONICAL_MODALITIES:
            uri = _resolve_modality_uri(values, modality)
            modality_uris[modality] = uri
            uri_existence_flags[modality] = _check_uri_existence(self.slp_root, uri)
        frame_layer = FrameLayer(
            sample_id=sample_id,
            setting=setting,
            subject_id=subject_id,
            cover_condition=cover_condition,
            frame_index=frame_index,
            modality_uris=modality_uris,
            missing_modalities=tuple(_split_csv(values["missing_modalities"])),
            expected_missing_modalities=tuple(_split_csv(values["expected_missing_modalities"])),
            ambiguous_modalities=tuple(_split_csv(values["ambiguous_modalities"])),
            uri_existence_flags=uri_existence_flags,
        )

        # Joint layer ------------------------------------------------------------
        subject_dir = self._get_subject_dir(setting, subject_id)
        j0_source_uris = {
            modality: _j0_source_uri(self.slp_root, subject_dir, modality)
            for modality in J0_MODALITIES
        }
        j0_present = {
            modality: (self.slp_root / uri).is_file()
            for modality, uri in j0_source_uris.items()
        }
        homography_contracts = {
            modality: self._homography_contract_for(setting, subject_id, modality)
            for modality in HOMOGRAPHY_MODALITIES
        }
        joint_layer = JointLayer(
            j0_source_uris=j0_source_uris,
            j0_present=j0_present,
            j0_artifact_count=JOINT_COUNT,
            joint_provenance_status=JOINT_PROVENANCE_J0,
            j1_status=J1_STATUS_NOT_GENERATED,
            homography_contracts=homography_contracts,
        )

        # Region layer -----------------------------------------------------------
        region_layer = RegionLayer(
            schema_version=REGION_SCHEMA_VERSION,
            placeholder_status=REGION_PLACEHOLDER_STATUS,
            annotation_count=0,
            annotations=(),
            can_be_used_as_training_truth=REGION_CAN_BE_USED_AS_TRAINING_TRUTH,
        )

        # Quality flags + quarantine -------------------------------------------
        quality_flags, quarantine, quarantine_reasons = self._compute_quality(
            frame_layer=frame_layer,
            joint_layer=joint_layer,
        )

        # Provenance -------------------------------------------------------------
        provenance = provenance_template or Provenance(
            task_id=self.task_id,
            adapter_version=ADAPTER_VERSION,
            canonical_schema_version=CANONICAL_SCHEMA_VERSION,
            generator=self.generator,
            created_at=self.created_at,
            a03_frame_index_source=self.a03_frame_index_source,
            a04_homography_audit_source=self.a04_audit_source,
            slp_root=str(self.slp_root.resolve()),
            pairing_method="explicit_frame_index_join_via_a03",
            semantic_direction_auto_selected=False,
            coordinate_origin_auto_shifted=False,
            silent_imputation=False,
            subject_split_applied=False,
            review_status_applied=False,
            model_prediction_applied=False,
        )

        return CanonicalSample(
            sample_id=sample_id,
            setting=setting,
            subject_id=subject_id,
            cover_condition=cover_condition,
            frame_index=frame_index,
            coordinate_frame=RAW_COORDINATE_FRAME,
            coordinate_origin_status=RAW_COORDINATE_ORIGIN_STATUS,
            frame=frame_layer,
            joint=joint_layer,
            region=region_layer,
            provenance=provenance,
            quality_flags=tuple(sorted(set(quality_flags))),
            quarantine=quarantine,
            quarantine_reasons=tuple(quarantine_reasons),
        )

    # -- internals ---------------------------------------------------------

    def _get_subject_dir(self, setting: str, subject_id: str) -> Path:
        key = (setting, subject_id)
        if key not in self.subject_dir_cache:
            self.subject_dir_cache[key] = _subject_dir_for(self.slp_root, setting, subject_id)
        return self.subject_dir_cache[key]

    def _homography_contract_for(
        self, setting: str, subject_id: str, modality: str
    ) -> HomographyContract:
        contract = self._homography_contracts.get((setting, subject_id, modality))
        if contract is None:
            return _placeholder_homography_contract(modality)
        return contract

    def _compute_quality(
        self,
        *,
        frame_layer: FrameLayer,
        joint_layer: JointLayer,
    ) -> tuple[list[str], bool, list[str]]:
        """Return (quality_flags, quarantine, quarantine_reasons).

        ``quality_flags`` covers every observation A05 cares about, including
        soft warnings such as ``homography_unresolved_<modality>`` and
        ``coordinate_origin_unresolved``. ``quarantine_reasons`` is the
        narrower list of *hard* reasons that actually trigger quarantine
        (missing modality, ambiguous modality, blocked homography, missing
        J0). The two are kept distinct on purpose: soft warnings are surfaced
        on every sample and should not crowd the quarantine reason counts.
        """
        flags: list[str] = []
        hard_reasons: list[str] = []

        # A03 frame-layer signals ------------------------------------------------
        for modality in frame_layer.missing_modalities:
            flags.append(f"missing_{modality}")
            hard_reasons.append(f"missing_modality:{modality}")
        for modality in frame_layer.ambiguous_modalities:
            flags.append(f"ambiguous_{modality}")
            hard_reasons.append(f"ambiguous_modality:{modality}")
        for modality, status in frame_layer.uri_existence_flags.items():
            if status == "missing_on_disk":
                flags.append(f"uri_missing_on_disk:{modality}")
                hard_reasons.append(f"uri_missing_on_disk:{modality}")

        # Joint layer signals ----------------------------------------------------
        for modality, present in joint_layer.j0_present.items():
            if not present:
                flags.append(f"j0_missing_{modality}")
                hard_reasons.append(f"j0_missing:{modality}")
        for modality, contract in joint_layer.homography_contracts.items():
            if contract.blocked:
                flags.append(f"homography_blocked_{modality}")
                hard_reasons.append(f"homography_blocked:{modality}")
            elif contract.unresolved_direction:
                # Soft warning: direction is unresolved but the matrix is
                # mathematically valid. We do NOT put this into the hard
                # reasons so that downstream quarantine logic does not
                # silently drop every sample on the A04 evidence alone.
                flags.append(f"homography_unresolved_{modality}")

        # Coordinate origin is a global soft warning ---------------------------
        flags.append("coordinate_origin_unresolved")

        # Region layer is always a placeholder in A05 --------------------------
        flags.append("region_placeholder_only")

        quarantine = bool(
            frame_layer.missing_modalities
            or frame_layer.ambiguous_modalities
            or any(contract.blocked for contract in joint_layer.homography_contracts.values())
            or any(not present for present in joint_layer.j0_present.values())
        )
        return flags, quarantine, hard_reasons


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_a03_frame_index_csv(path: Path) -> list[SlpFrameIndexRow]:
    """Load an A03 frame index CSV as :class:`SlpFrameIndexRow` values."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[SlpFrameIndexRow] = []
        for raw in reader:
            if not raw:
                continue
            values = {column: raw.get(column, "") for column in FRAME_INDEX_COLUMNS}
            values["frame_index"] = int(values["frame_index"]) if values["frame_index"] != "" else 0
            values["quarantine"] = _parse_bool(values.get("quarantine", ""))
            rows.append(SlpFrameIndexRow(values))
    return rows


def load_a04_homography_audit_csv(path: Path) -> list[SlpHomographyAuditRow]:
    """Load an A04 homography audit CSV as :class:`SlpHomographyAuditRow` values."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[SlpHomographyAuditRow] = []
        for raw in reader:
            if not raw:
                continue
            values = {column: raw.get(column, "") for column in HOMOGRAPHY_AUDIT_COLUMNS}
            for boolean_column in ("matrix_present", "invertible"):
                values[boolean_column] = _parse_bool(values.get(boolean_column, ""))
            for float_column in (
                "determinant",
                "condition_number",
                "rank",
                "source_width",
                "source_height",
                "pm_width",
                "pm_height",
                "probe_roundtrip_mean_error",
                "probe_roundtrip_max_error",
                "joint_points",
                "direct_joint_in_bounds_rate",
                "inverse_joint_in_bounds_rate",
            ):
                values[float_column] = _parse_optional_int_or_float(values.get(float_column, ""))
            rows.append(SlpHomographyAuditRow(values))
    return rows


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parse_optional_int_or_float(value: object) -> object:
    if value is None or value == "":
        return ""
    text = str(value).strip()
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


CANONICAL_CSV_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "setting",
    "subject_id",
    "cover_condition",
    "frame_index",
    "coordinate_frame",
    "coordinate_origin_status",
    # Frame layer
    "rgb_uri",
    "ir_uri",
    "irraw_uri",
    "depth_uri",
    "depthraw_uri",
    "pm_uri",
    "rgb_uri_status",
    "ir_uri_status",
    "irraw_uri_status",
    "depth_uri_status",
    "depthraw_uri_status",
    "pm_uri_status",
    "missing_modalities",
    "expected_missing_modalities",
    "ambiguous_modalities",
    # Joint layer
    "j0_rgb_uri",
    "j0_ir_uri",
    "j0_rgb_present",
    "j0_ir_present",
    "j0_artifact_count",
    "joint_provenance_status",
    "j1_status",
    # Homography contracts
    "rgb_homography_matrix_uri",
    "rgb_homography_matrix_present",
    "rgb_homography_invertible",
    "rgb_homography_direction_status",
    "rgb_homography_coordinate_origin_status",
    "rgb_homography_probe_roundtrip_max_error",
    "rgb_homography_direct_joint_in_bounds_rate",
    "rgb_homography_inverse_joint_in_bounds_rate",
    "rgb_homography_error_codes",
    "ir_homography_matrix_uri",
    "ir_homography_matrix_present",
    "ir_homography_invertible",
    "ir_homography_direction_status",
    "ir_homography_coordinate_origin_status",
    "ir_homography_probe_roundtrip_max_error",
    "ir_homography_direct_joint_in_bounds_rate",
    "ir_homography_inverse_joint_in_bounds_rate",
    "ir_homography_error_codes",
    "depth_homography_matrix_uri",
    "depth_homography_matrix_present",
    "depth_homography_invertible",
    "depth_homography_direction_status",
    "depth_homography_coordinate_origin_status",
    "depth_homography_probe_roundtrip_max_error",
    "depth_homography_direct_joint_in_bounds_rate",
    "depth_homography_inverse_joint_in_bounds_rate",
    "depth_homography_error_codes",
    # Region layer
    "region_schema_version",
    "region_placeholder_status",
    "region_annotation_count",
    "region_can_be_used_as_training_truth",
    # Provenance
    "provenance_task_id",
    "provenance_adapter_version",
    "provenance_canonical_schema_version",
    "provenance_generator",
    "provenance_created_at",
    "provenance_a03_frame_index_source",
    "provenance_a04_homography_audit_source",
    "provenance_slp_root",
    "provenance_pairing_method",
    "provenance_semantic_direction_auto_selected",
    "provenance_coordinate_origin_auto_shifted",
    "provenance_silent_imputation",
    "provenance_subject_split_applied",
    "provenance_review_status_applied",
    "provenance_model_prediction_applied",
    # Quality
    "quality_flags",
    "quarantine",
    "quarantine_reasons",
)


def canonical_sample_to_csv_row(sample: CanonicalSample) -> dict[str, object]:
    """Flatten a :class:`CanonicalSample` to a CSV-friendly dict."""
    frame = sample.frame
    joint = sample.joint
    region = sample.region
    provenance = sample.provenance
    rgb_contract = joint.homography_contracts.get("RGB", _placeholder_homography_contract("RGB"))
    ir_contract = joint.homography_contracts.get("IR", _placeholder_homography_contract("IR"))
    depth_contract = joint.homography_contracts.get("depth", _placeholder_homography_contract("depth"))

    def _homography_fields(contract: HomographyContract, prefix: str) -> dict[str, object]:
        return {
            f"{prefix}_homography_matrix_uri": contract.matrix_uri,
            f"{prefix}_homography_matrix_present": contract.matrix_present,
            f"{prefix}_homography_invertible": contract.invertible,
            f"{prefix}_homography_direction_status": contract.direction_status,
            f"{prefix}_homography_coordinate_origin_status": contract.coordinate_origin_status,
            f"{prefix}_homography_probe_roundtrip_max_error": contract.probe_roundtrip_max_error,
            f"{prefix}_homography_direct_joint_in_bounds_rate": contract.direct_joint_in_bounds_rate,
            f"{prefix}_homography_inverse_joint_in_bounds_rate": contract.inverse_joint_in_bounds_rate,
            f"{prefix}_homography_error_codes": ";".join(contract.error_codes),
        }

    row: dict[str, object] = {
        "sample_id": sample.sample_id,
        "setting": sample.setting,
        "subject_id": sample.subject_id,
        "cover_condition": sample.cover_condition,
        "frame_index": sample.frame_index,
        "coordinate_frame": sample.coordinate_frame,
        "coordinate_origin_status": sample.coordinate_origin_status,
        # Frame
        "rgb_uri": frame.modality_uris.get("RGB", ""),
        "ir_uri": frame.modality_uris.get("IR", ""),
        "irraw_uri": frame.modality_uris.get("IRraw", ""),
        "depth_uri": frame.modality_uris.get("depth", ""),
        "depthraw_uri": frame.modality_uris.get("depthRaw", ""),
        "pm_uri": frame.modality_uris.get("PM", ""),
        "rgb_uri_status": frame.uri_existence_flags.get("RGB", ""),
        "ir_uri_status": frame.uri_existence_flags.get("IR", ""),
        "irraw_uri_status": frame.uri_existence_flags.get("IRraw", ""),
        "depth_uri_status": frame.uri_existence_flags.get("depth", ""),
        "depthraw_uri_status": frame.uri_existence_flags.get("depthRaw", ""),
        "pm_uri_status": frame.uri_existence_flags.get("PM", ""),
        "missing_modalities": ";".join(frame.missing_modalities),
        "expected_missing_modalities": ";".join(frame.expected_missing_modalities),
        "ambiguous_modalities": ";".join(frame.ambiguous_modalities),
        # Joint
        "j0_rgb_uri": joint.j0_source_uris.get("RGB", ""),
        "j0_ir_uri": joint.j0_source_uris.get("IR", ""),
        "j0_rgb_present": joint.j0_present.get("RGB", False),
        "j0_ir_present": joint.j0_present.get("IR", False),
        "j0_artifact_count": joint.j0_artifact_count,
        "joint_provenance_status": joint.joint_provenance_status,
        "j1_status": joint.j1_status,
    }
    row.update(_homography_fields(rgb_contract, "rgb"))
    row.update(_homography_fields(ir_contract, "ir"))
    row.update(_homography_fields(depth_contract, "depth"))
    row.update(
        {
            "region_schema_version": region.schema_version,
            "region_placeholder_status": region.placeholder_status,
            "region_annotation_count": region.annotation_count,
            "region_can_be_used_as_training_truth": region.can_be_used_as_training_truth,
            "provenance_task_id": provenance.task_id,
            "provenance_adapter_version": provenance.adapter_version,
            "provenance_canonical_schema_version": provenance.canonical_schema_version,
            "provenance_generator": provenance.generator,
            "provenance_created_at": provenance.created_at,
            "provenance_a03_frame_index_source": provenance.a03_frame_index_source,
            "provenance_a04_homography_audit_source": provenance.a04_homography_audit_source,
            "provenance_slp_root": provenance.slp_root,
            "provenance_pairing_method": provenance.pairing_method,
            "provenance_semantic_direction_auto_selected": provenance.semantic_direction_auto_selected,
            "provenance_coordinate_origin_auto_shifted": provenance.coordinate_origin_auto_shifted,
            "provenance_silent_imputation": provenance.silent_imputation,
            "provenance_subject_split_applied": provenance.subject_split_applied,
            "provenance_review_status_applied": provenance.review_status_applied,
            "provenance_model_prediction_applied": provenance.model_prediction_applied,
            "quality_flags": ";".join(sample.quality_flags),
            "quarantine": sample.quarantine,
            "quarantine_reasons": ";".join(sample.quarantine_reasons),
        }
    )
    return row


def write_canonical_csv(samples: Sequence[CanonicalSample], path: Path) -> None:
    """Write a wide CSV of canonical samples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CANONICAL_CSV_COLUMNS))
        writer.writeheader()
        for sample in samples:
            writer.writerow(canonical_sample_to_csv_row(sample))


def write_canonical_jsonl(samples: Sequence[CanonicalSample], path: Path) -> None:
    """Write one JSON object per line for full structured provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.as_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarise_canonical_samples(samples: Iterable[CanonicalSample]) -> dict[str, object]:
    """Compute a real-data summary across all canonical samples.

    Notes
    -----
    * ``quarantine_reason_counts`` and ``quarantine_rows`` describe samples
      that the adapter actually quarantined (missing modality, ambiguous
      modality, blocked homography, or missing J0). ``quality_flag_counts``
      is the broader set of soft warnings (e.g. ``coordinate_origin_unresolved``
      or ``homography_unresolved_<modality>``); these are surfaced on every
      sample but do not by themselves trigger quarantine.
    * ``homography_contracts_attached`` is the total number of contracts
      carried across all samples (one contract per frame per homography
      modality). ``homography_audit_rows_seen`` is the unique A04 audit
      rows underlying those contracts.
    """
    materialized = list(samples)
    by_setting = Counter()
    by_cover = Counter()
    by_setting_subject: set[tuple[str, str]] = set()

    missing_modality_counter: Counter[str] = Counter()
    expected_missing_counter: Counter[str] = Counter()
    ambiguous_modality_counter: Counter[str] = Counter()
    quality_flag_counter: Counter[str] = Counter()
    quarantine_reason_counter: Counter[str] = Counter()
    quarantine_rows = 0
    j0_missing_rows = 0
    j0_missing_by_modality: Counter[str] = Counter()
    homography_blocked_rows = 0
    homography_unresolved_rows = 0
    uri_missing_on_disk_rows = 0
    uri_missing_on_disk_by_modality: Counter[str] = Counter()
    coordinate_origin_unresolved_rows = 0
    region_placeholder_rows = 0
    total_expected_uris = 0
    traceable_uris = 0
    absent_uris = 0
    a04_audit_rows_seen: set[tuple[str, str, str]] = set()
    homography_contracts_attached: Counter[str] = Counter()
    homography_blocked_by_modality: Counter[str] = Counter()
    homography_unresolved_by_modality: Counter[str] = Counter()

    for sample in materialized:
        by_setting[sample.setting] += 1
        by_cover[sample.cover_condition] += 1
        by_setting_subject.add((sample.setting, sample.subject_id))
        for modality in sample.frame.missing_modalities:
            missing_modality_counter[modality] += 1
        for modality in sample.frame.expected_missing_modalities:
            expected_missing_counter[modality] += 1
        for modality in sample.frame.ambiguous_modalities:
            ambiguous_modality_counter[modality] += 1
        for flag in sample.quality_flags:
            quality_flag_counter[flag] += 1
        for modality, present in sample.joint.j0_present.items():
            if not present:
                j0_missing_rows += 1
                j0_missing_by_modality[modality] += 1
        for modality, contract in sample.joint.homography_contracts.items():
            a04_audit_rows_seen.add((sample.setting, sample.subject_id, modality))
            homography_contracts_attached[modality] += 1
            if contract.blocked:
                homography_blocked_rows += 1
                homography_blocked_by_modality[modality] += 1
            if contract.unresolved_direction:
                homography_unresolved_rows += 1
                homography_unresolved_by_modality[modality] += 1
        for modality, status in sample.frame.uri_existence_flags.items():
            total_expected_uris += 1
            if status == "present":
                traceable_uris += 1
            elif status == "missing_on_disk":
                uri_missing_on_disk_rows += 1
                uri_missing_on_disk_by_modality[modality] += 1
            else:
                absent_uris += 1
        if "coordinate_origin_unresolved" in sample.quality_flags:
            coordinate_origin_unresolved_rows += 1
        if "region_placeholder_only" in sample.quality_flags:
            region_placeholder_rows += 1
        if sample.quarantine:
            quarantine_rows += 1
            for reason in sample.quarantine_reasons:
                quarantine_reason_counter[reason] += 1

    traceable_rate = (traceable_uris / total_expected_uris) if total_expected_uris else 0.0
    return {
        "dataset": "SLP",
        "task_id": "TASK-SLP-A05-CANONICAL-ADAPTER-v0.1",
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(materialized),
        "subjects": len(by_setting_subject),
        "subjects_per_setting": {
            setting: sum(1 for s, _ in by_setting_subject if s == setting)
            for setting in sorted({s for s, _ in by_setting_subject})
        },
        "frames_by_setting": dict(sorted(by_setting.items())),
        "frames_by_cover": dict(sorted(by_cover.items())),
        "missing_modality_frame_counts": dict(sorted(missing_modality_counter.items())),
        "expected_missing_modality_frame_counts": dict(sorted(expected_missing_counter.items())),
        "ambiguous_modality_frame_counts": dict(sorted(ambiguous_modality_counter.items())),
        "quarantine_rows": quarantine_rows,
        "quarantine_reason_counts": dict(sorted(quarantine_reason_counter.items())),
        "quality_flag_counts": dict(sorted(quality_flag_counter.items())),
        "j0_missing_rows": j0_missing_rows,
        "j0_missing_by_modality": dict(sorted(j0_missing_by_modality.items())),
        "homography_audit_rows_seen": len(a04_audit_rows_seen),
        "homography_contracts_attached": dict(sorted(homography_contracts_attached.items())),
        "homography_blocked_rows": homography_blocked_rows,
        "homography_blocked_by_modality": dict(sorted(homography_blocked_by_modality.items())),
        "homography_unresolved_rows": homography_unresolved_rows,
        "homography_unresolved_by_modality": dict(sorted(homography_unresolved_by_modality.items())),
        "coordinate_origin_unresolved_rows": coordinate_origin_unresolved_rows,
        "region_placeholder_rows": region_placeholder_rows,
        "uri_traceability": {
            "total_expected_uris": total_expected_uris,
            "traceable_uris": traceable_uris,
            "absent_uris": absent_uris,
            "uri_missing_on_disk_rows": uri_missing_on_disk_rows,
            "uri_missing_on_disk_by_modality": dict(sorted(uri_missing_on_disk_by_modality.items())),
            "traceable_rate": traceable_rate,
        },
        "semantic_direction_auto_selected": False,
        "coordinate_origin_auto_shifted": False,
        "subject_split_applied": False,
        "review_status_applied": False,
        "model_prediction_applied": False,
        "silent_imputation": False,
    }


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------


def build_adapter_from_artifacts(
    *,
    slp_root: Path,
    a03_frame_index_csv: Path | None = None,
    a04_homography_audit_csv: Path | None = None,
    task_id: str = DEFAULT_TASK_ID,
) -> tuple[SlpCanonicalAdapter, list[SlpFrameIndexRow], list[SlpHomographyAuditRow]]:
    """Build an adapter from artifact files; at least ``a03_frame_index_csv`` is required."""
    if a03_frame_index_csv is None:
        raise ValueError("a03_frame_index_csv is required to build the adapter")
    a03_rows = load_a03_frame_index_csv(a03_frame_index_csv)
    a04_rows = (
        load_a04_homography_audit_csv(a04_homography_audit_csv)
        if a04_homography_audit_csv is not None
        else []
    )
    adapter = SlpCanonicalAdapter(
        slp_root=slp_root,
        a03_frame_rows=a03_rows,
        a04_audit_rows=a04_rows,
        task_id=task_id,
        a03_frame_index_source=str(a03_frame_index_csv),
        a04_audit_source=str(a04_homography_audit_csv) if a04_homography_audit_csv else "none",
    )
    return adapter, a03_rows, a04_rows


def resolve_subject_id_set(slp_root: Path) -> set[tuple[str, str]]:
    """Return the (setting, subject_id) pairs the adapter is expected to know about."""
    return {
        (setting, subject_dir.name)
        for setting, subject_dir in iter_subject_directories(slp_root)
    }


__all__ = [
    "ADAPTER_VERSION",
    "CANONICAL_CSV_COLUMNS",
    "CANONICAL_SCHEMA_VERSION",
    "CanonicalSample",
    "DEFAULT_GENERATOR",
    "DEFAULT_TASK_ID",
    "FrameLayer",
    "HomographyContract",
    "J0_GT_SOURCE",
    "J1_STATUS_NOT_GENERATED",
    "JOINT_PROVENANCE_J0",
    "JointLayer",
    "Provenance",
    "RAW_COORDINATE_FRAME",
    "RAW_COORDINATE_ORIGIN_STATUS",
    "REGION_PLACEHOLDER_STATUS",
    "REGION_SCHEMA_VERSION",
    "RegionLayer",
    "SlpCanonicalAdapter",
    "build_adapter_from_artifacts",
    "canonical_sample_to_csv_row",
    "load_a03_frame_index_csv",
    "load_a04_homography_audit_csv",
    "resolve_subject_id_set",
    "summarise_canonical_samples",
    "write_canonical_csv",
    "write_canonical_jsonl",
]


def _ensure_module_constants_visible() -> None:
    # Marker so the module is recognized as fully initialized.
    return None
