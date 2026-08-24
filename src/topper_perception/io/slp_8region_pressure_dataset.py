"""SLP 8-Region Pressure-Only GT Dataset Adapter.

Owner decision (2026-08-24): SLP_8Region_Pressure_VAL_v1.1 is the project
accepted reference GT for SLP8 pressure-only region segmentation, despite its
annotation_provenance being V221_CORRECTED_SUPPORT_AUTO_ACCEPTED and
source_review_status being NOT_REVIEWED.

This adapter:
- Reads val_manifest.csv (NOT os.listdir guessing) to get sample records.
- Uses relative paths inside the dataset root; D:\\ absolute source paths are
  only used for provenance tracing and are never used for file loading.
- Validates pressure/label arrays fail-closed: shape, dtype, finite, range,
  onehot roundtrip, and path containment.
- Pressure is kept as raw PMarray response semantics; it is NOT converted to kPa.
- Normalization is NOT fitted here; callers must fit on TRAIN subjects only.
- Raw data is never modified; no files are written back to the dataset directory.

GT provenance contract:
  - annotation_provenance = V221_CORRECTED_SUPPORT_AUTO_ACCEPTED
  - source_review_status = NOT_REVIEWED
  - NOT human pixel-level semantic masks
  - NOT medical, skin-interface stress, or product ground truth
  - NOT suitable for claim of anatomical accuracy

The 8-region class schema (slp8-v2.2.1) is DIFFERENT from the 10-region
polygon-based slp_region_annotation_v0.1 ontology (R0-R3 OpenCV/human-review
route). These two schemas are NOT equivalent and MUST NOT be conflated.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

ADAPTER_VERSION = "slp_8region_pressure_dataset_adapter_v0.1"
DATASET_ID = "SLP_8Region_Pressure_VAL_v1.1"
CLASS_SCHEMA_VERSION = "slp8-v2.2.1-canonical-export-v1.1"

#: 9 channels: 1 background + 8 semantic regions.
NUM_CLASSES = 9
BACKGROUND_ID = 0

#: Pressure matrix shape (height, width).
PRESSURE_SHAPE = (192, 84)
PRESSURE_DTYPE = np.float64  # as exported: float64

#: Region label shape = pressure shape.
LABEL_SHAPE = (192, 84)
LABEL_DTYPE = np.uint8

#: One-hot shape (C, H, W).
ONEHOT_SHAPE = (9, 192, 84)
ONEHOT_DTYPE = np.uint8

#: Class name → ID mapping (canonical).
CLASS_NAME_TO_ID: dict[str, int] = {
    "BACKGROUND": 0,
    "HEAD_NECK": 1,
    "SHOULDER": 2,
    "THORAX_BACK": 3,
    "LUMBAR_WAIST": 4,
    "PELVIS_HIP": 5,
    "ARM": 6,
    "THIGH": 7,
    "LOWER_LEG_FOOT": 8,
}
CLASS_ID_TO_NAME: dict[int, str] = {v: k for k, v in CLASS_NAME_TO_ID.items()}

#: Expected postures.
VALID_POSTURES = frozenset({"SUPINE", "LEFT", "RIGHT"})

#: Expected settings (this dataset only has danaLab).
VALID_SETTINGS = frozenset({"danaLab"})

#: Expected covers (this dataset only has uncover).
VALID_COVERS = frozenset({"uncover"})

#: Expected annotation provenances.
VALID_ANNOTATION_PROVENANCES = frozenset({
    "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
})

#: Expected export statuses.
VALID_EXPORT_STATUSES = frozenset({"EXPORTED"})

#: Expected split values (VAL is the only split in this dataset).
VALID_SPLITS = frozenset({"VAL"})


# ---------------------------------------------------------------------------
# Error / warning definitions
# ---------------------------------------------------------------------------

class Slp8RegionDatasetError(Exception):
    """Base exception for SLP 8-region dataset errors."""
    pass


class PathContainmentViolation(Slp8RegionDatasetError):
    """A file path escapes the declared dataset root."""
    pass


class SampleNotFoundError(Slp8RegionDatasetError):
    """Requested sample_id does not exist in manifest."""
    pass


class ValidationError(Slp8RegionDatasetError):
    """A sample failed a fail-closed validation check."""
    pass


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Slp8RegionSample:
    """One sample from SLP_8Region_Pressure_VAL_v1.1."""

    sample_id: str
    split: str
    setting: str
    subject_id: str
    cover: str
    frame_id: int
    posture: str

    #: Relative paths from dataset root.
    pressure_npy: Path   # samples/.../pressure.npy
    region_label_npy: Path
    #: region_onehot.npy path. Required in SLP_8Region_Pressure_VAL_v1.1
    #: (all 4,590 samples include onehot). Empty CSV value raises
    #: Slp8RegionDatasetError during _parse_row.
    region_onehot_npy: Path
    #: points.csv path. Optional in this dataset.
    points_csv: Path | None

    height: int
    width: int
    class_ids_present: tuple[int, ...]
    background_pixel_count: int
    body_pixel_count: int
    clipped_ratio: float
    onehot_valid: bool
    onehot_roundtrip: bool

    #: Provenance — MUST NOT be rewritten.
    annotation_provenance: str
    source_review_status: str
    export_version: str
    export_status: str
    source_flags: tuple[str, ...]
    source_status: str

    #: SHA-256 of pressure.npy as declared in manifest.
    pressure_sha256: str

    #: Provenance chain.
    adapter_version: str = ADAPTER_VERSION
    loaded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def _resolve_contained_path(
        self,
        rel_path: Path,
        root: Path,
        field_name: str,
    ) -> Path:
        """Resolve a relative path and verify it stays inside root.

        Uses pathlib.Path.relative_to() which raises ValueError when the
        resolved path is not relative to root (i.e., escapes via ../ or
        absolute path). We also explicitly reject absolute-path strings.

        Parameters
        ----------
        rel_path : Path
            Relative path as stored in the manifest.
        root : Path
            The declared dataset root.
        field_name : str
            Field name for error messages.

        Returns
        -------
        Path
            The resolved absolute path, confirmed to be inside root.

        Raises
        ------
        PathContainmentViolation
            When rel_path resolves outside root or is absolute.
        """
        # Reject absolute-path strings (Windows C:\ or POSIX /)
        s = str(rel_path)
        if s and (s[0] == "/" or (len(s) > 1 and s[1] == ":")):
            raise PathContainmentViolation(
                f"{field_name}: absolute path not allowed: {rel_path}"
            )

        resolved = (root / rel_path).resolve()
        root_resolved = root.resolve()

        # relative_to() raises ValueError when resolved is not under root
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            raise PathContainmentViolation(
                f"{field_name}: path escapes dataset root: "
                f"{rel_path!r} → {resolved} (root={root_resolved})"
            )

        return resolved

    def pressure_path(self, root: Path) -> Path:
        """Resolve pressure.npy relative path to absolute path; fail if outside root."""
        return self._resolve_contained_path(
            self.pressure_npy, root, "pressure_npy"
        )

    def label_path(self, root: Path) -> Path:
        """Resolve region_label.npy; fail if outside root."""
        return self._resolve_contained_path(
            self.region_label_npy, root, "region_label_npy"
        )

    def onehot_path(self, root: Path) -> Path:
        """Resolve region_onehot.npy; always returns a Path (required field)."""
        return self._resolve_contained_path(
            self.region_onehot_npy, root, "region_onehot_npy"
        )

    def points_csv_path(self, root: Path) -> Path | None:
        """Resolve points.csv relative path; returns None if not present."""
        if self.points_csv is None:
            return None
        return self._resolve_contained_path(
            self.points_csv, root, "points_csv"
        )


@dataclass
class Slp8RegionLoadResult:
    """Result of loading one sample's arrays."""

    sample: Slp8RegionSample
    pressure: np.ndarray   # shape (192, 84), dtype float64
    region_label: np.ndarray  # shape (192, 84), dtype uint8
    region_onehot: np.ndarray  # shape (9, 192, 84), dtype uint8; required field

    pressure_sha256: str
    onehot_roundtrip_ok: bool


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class Slp8RegionDatasetAdapter:
    """Read-only adapter for SLP_8Region_Pressure_VAL_v1.1.

    Parameters
    ----------
    dataset_root : Path
        Absolute path to SLP_8Region_Pressure_VAL_v1.1 root.
        Must contain manifest/val_manifest.csv.
    validate_on_load : bool
        If True (default), run fail-closed checks on every load_sample() call.
        Set to False only for high-throughput inference where the caller manages
        their own validation loop.
    """

    ADAPTER_VERSION: ClassVar[str] = ADAPTER_VERSION
    DATASET_ID: ClassVar[str] = DATASET_ID

    def __init__(
        self,
        dataset_root: Path,
        *,
        validate_on_load: bool = True,
    ) -> None:
        self._root = Path(dataset_root).resolve()
        self._validate_on_load = validate_on_load
        self._manifest_path = self._root / "manifest" / "val_manifest.csv"
        self._samples: dict[str, dict[str, Any]] | None = None
        self._loaded_at = datetime.now(timezone.utc).isoformat()

        if not self._manifest_path.exists():
            raise Slp8RegionDatasetError(
                f"val_manifest.csv not found at {self._manifest_path}"
            )

    # ── Indexing ────────────────────────────────────────────────────────────

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        """Lazily load and parse val_manifest.csv into a dict keyed by sample_id."""
        if self._samples is not None:
            return self._samples

        rows: dict[str, dict[str, Any]] = {}
        # Try utf-8-sig first (handles BOM), fall back to utf-8
        for enc in ("utf-8-sig", "utf-8"):
            try:
                with self._manifest_path.open(newline="", encoding=enc) as f:
                    reader = csv.DictReader(f)
                    first_row = next(reader)
                    if "sample_id" not in first_row:
                        raise Slp8RegionDatasetError(
                            f"CSV at {self._manifest_path} missing 'sample_id' column "
                            f"(encoding={enc!r}, fieldnames={reader.fieldnames})"
                        )
                    # Rewind by re-opening (simple approach for small manifest)
                break
            except UnicodeDecodeError:
                continue

        with self._manifest_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                sid = raw["sample_id"].strip()
                if not sid or sid in rows:
                    raise Slp8RegionDatasetError(
                        f"Duplicate or empty sample_id in manifest: {sid!r}"
                    )
                rows[sid] = raw

        self._samples = rows
        return rows

    @property
    def sample_count(self) -> int:
        return len(self._load_manifest())

    @property
    def subject_ids(self) -> list[str]:
        """All 102 subject IDs in dataset order (sorted for determinism)."""
        rows = self._load_manifest()
        return sorted({r["subject_id"] for r in rows.values()})

    @property
    def postures(self) -> set[str]:
        rows = self._load_manifest()
        return {r["posture"] for r in rows.values()}

    def iter_samples(
        self,
        *,
        subject_ids: list[str] | None = None,
        postures: list[str] | None = None,
    ) -> list[Slp8RegionSample]:
        """Iterate all samples, optionally filtered by subject or posture.

        All parameters are filters (AND-combined); None means no filter.
        """
        rows = self._load_manifest()
        result: list[Slp8RegionSample] = []

        for sid, raw in rows.items():
            if subject_ids and raw["subject_id"] not in subject_ids:
                continue
            if postures and raw["posture"] not in postures:
                continue
            result.append(self._parse_row(sid, raw))

        return result

    def get_sample(self, sample_id: str) -> Slp8RegionSample:
        """Get one sample by sample_id; raises SampleNotFoundError."""
        rows = self._load_manifest()
        raw = rows.get(sample_id.strip())
        if raw is None:
            raise SampleNotFoundError(f"sample_id not in manifest: {sample_id!r}")
        return self._parse_row(sample_id, raw)

    # ── Loading ─────────────────────────────────────────────────────────────

    def load_sample(
        self,
        sample_id: str,
        *,
        root: Path | None = None,
        validate: bool | None = None,
    ) -> Slp8RegionLoadResult:
        """Load one sample's pressure + region arrays with optional validation.

        Parameters
        ----------
        sample_id : str
            Primary key.
        root : Path | None
            Override dataset root (mainly for testing). Defaults to self._root.
        validate : bool | None
            Override validate_on_load. None → use instance default.

        Returns
        -------
        Slp8RegionLoadResult

        Raises
        ------
        SampleNotFoundError
        PathContainmentViolation
        ValidationError
        Slp8RegionDatasetError
        """
        sample = self.get_sample(sample_id)
        dataset_root = Path(root) if root else self._root
        should_validate = (
            validate if validate is not None else self._validate_on_load
        )

        p_path = sample.pressure_path(dataset_root)
        l_path = sample.label_path(dataset_root)
        o_path = sample.onehot_path(dataset_root)

        # ── Load arrays ────────────────────────────────────────────────────

        pressure = np.load(p_path, allow_pickle=False)  # type: ignore[arg-type]
        region_label = np.load(l_path, allow_pickle=False)  # type: ignore[arg-type]

        region_onehot: np.ndarray | None = None
        if o_path.exists():
            region_onehot = np.load(o_path, allow_pickle=False)  # type: ignore[arg-type]
        else:
            # region_onehot_npy is required; absence is a validation failure
            raise Slp8RegionDatasetError(
                f"region_onehot.npy not found (required field): {o_path}"
            )

        # ── Compute pressure SHA256 ─────────────────────────────────────────

        pressure_bytes = p_path.read_bytes()
        actual_sha256 = hashlib.sha256(pressure_bytes).hexdigest()

        # ── Fail-closed validation ───────────────────────────────────────────

        if should_validate:
            self._validate(
                sample=sample,
                pressure=pressure,
                region_label=region_label,
                region_onehot=region_onehot,
                pressure_sha256=actual_sha256,
                root=dataset_root,
            )

        # ── Onehot roundtrip check ───────────────────────────────────────────

        # region_onehot is always non-None here (required field; absence raises above)
        reconstructed = np.argmax(region_onehot, axis=0).astype(np.uint8)
        onehot_roundtrip_ok = bool(np.array_equal(reconstructed, region_label))

        return Slp8RegionLoadResult(
            sample=sample,
            pressure=pressure,
            region_label=region_label,
            region_onehot=region_onehot,
            pressure_sha256=actual_sha256,
            onehot_roundtrip_ok=onehot_roundtrip_ok,
        )

    # ── Validation ──────────────────────────────────────────────────────────

    def _validate(
        self,
        sample: Slp8RegionSample,
        pressure: np.ndarray,
        region_label: np.ndarray,
        region_onehot: np.ndarray,
        pressure_sha256: str,
        root: Path,
    ) -> None:
        """Fail-closed validation of loaded arrays."""
        errors: list[str] = []

        # ── pressure shape/dtype/finite ──────────────────────────────────

        if pressure.shape != PRESSURE_SHAPE:
            errors.append(
                f"pressure shape {pressure.shape} != expected {PRESSURE_SHAPE}"
            )
        if pressure.dtype != PRESSURE_DTYPE:
            errors.append(
                f"pressure dtype {pressure.dtype} != expected {PRESSURE_DTYPE}"
            )
        if not np.isfinite(pressure).all():
            errors.append("pressure contains non-finite values (NaN/Inf)")

        # ── label shape/dtype/range ───────────────────────────────────────

        if region_label.shape != LABEL_SHAPE:
            errors.append(
                f"region_label shape {region_label.shape} != expected {LABEL_SHAPE}"
            )
        if region_label.dtype != LABEL_DTYPE:
            errors.append(
                f"region_label dtype {region_label.dtype} != expected {LABEL_DTYPE}"
            )
        label_min = int(region_label.min())
        label_max = int(region_label.max())
        if label_min < 0 or label_max > 8:
            errors.append(
                f"region_label values [{label_min}, {label_max}] out of range [0, 8]"
            )

        # ── onehot shape/dtype/roundtrip ─────────────────────────────────

        # region_onehot is always non-None (required field)
        # Shape must be correct before any pixel-level checks
            if region_onehot.shape != ONEHOT_SHAPE:
                errors.append(
                    f"region_onehot shape {region_onehot.shape} != "
                    f"expected {ONEHOT_SHAPE}"
                )
            if region_onehot.dtype != ONEHOT_DTYPE:
                errors.append(
                    f"region_onehot dtype {region_onehot.dtype} != "
                    f"expected {ONEHOT_DTYPE}"
                )
            # Only continue semantic checks if shape is already correct
            if region_onehot.shape == ONEHOT_SHAPE:
                # 1. Values must be 0/1
                unique_vals = set(np.unique(region_onehot).tolist())
                if unique_vals - {0, 1}:
                    errors.append(
                        f"region_onehot contains non-binary values: {unique_vals}"
                    )
                # 2. Per-pixel channel sum must be exactly 1 (mutually exclusive)
                if 0 not in unique_vals - {0, 1}:  # Only check if values are already binary
                    pixel_sums = region_onehot.sum(axis=0)  # shape (192, 84)
                    bad_pixels = np.sum(pixel_sums != 1)
                    if bad_pixels > 0:
                        errors.append(
                            f"region_onehot: {int(bad_pixels)} pixels have "
                            f"channel sum != 1 (two or zero active channels)"
                        )
                # 3. argmax(onehot) must equal label (full roundtrip)
                reconstructed = np.argmax(region_onehot, axis=0).astype(np.uint8)
                if not np.array_equal(reconstructed, region_label):
                    diff_count = int(np.sum(reconstructed != region_label))
                    errors.append(
                        f"region_onehot roundtrip: argmax(onehot) != label "
                        f"({diff_count}/{region_label.size} pixels differ)"
                    )

        # ── SHA256 match ───────────────────────────────────────────────────

        if pressure_sha256 != sample.pressure_sha256:
            errors.append(
                f"pressure SHA256 mismatch: manifest={sample.pressure_sha256}, "
                f"actual={pressure_sha256}"
            )

        if errors:
            raise ValidationError(
                f"Sample {sample.sample_id} failed validation:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _parse_row(self, sample_id: str, raw: dict[str, str]) -> Slp8RegionSample:
        """Parse one CSV row into a Slp8RegionSample dataclass."""
        frame_id_val = int(raw["frame_id"])
        class_ids = tuple(int(x) for x in raw["class_ids_present"].split("|"))

        # Validate split (only VAL in this dataset)
        split = raw["split"].strip()
        if split not in VALID_SPLITS:
            raise Slp8RegionDatasetError(
                f"Unknown split {split!r} in {sample_id}; "
                f"expected one of {sorted(VALID_SPLITS)}"
            )

        # Validate posture
        posture = raw["posture"].strip()
        if posture not in VALID_POSTURES:
            raise Slp8RegionDatasetError(
                f"Unknown posture {posture!r} in {sample_id}"
            )

        # Validate setting (only danaLab in this dataset)
        setting = raw["setting"].strip()
        if setting not in VALID_SETTINGS:
            raise Slp8RegionDatasetError(
                f"Unknown setting {setting!r} in {sample_id}; "
                f"expected one of {sorted(VALID_SETTINGS)}"
            )

        # Validate cover (only uncover in this dataset)
        cover = raw["cover"].strip()
        if cover not in VALID_COVERS:
            raise Slp8RegionDatasetError(
                f"Unknown cover {cover!r} in {sample_id}; "
                f"expected one of {sorted(VALID_COVERS)}"
            )

        # Validate annotation_provenance
        provenance = raw["annotation_provenance"].strip()
        if provenance not in VALID_ANNOTATION_PROVENANCES:
            raise Slp8RegionDatasetError(
                f"Unknown annotation_provenance {provenance!r} in {sample_id}; "
                f"expected one of {sorted(VALID_ANNOTATION_PROVENANCES)}"
            )

        # Validate export_status
        export_status = raw["export_status"].strip()
        if export_status not in VALID_EXPORT_STATUSES:
            raise Slp8RegionDatasetError(
                f"Unknown export_status {export_status!r} in {sample_id}; "
                f"expected one of {sorted(VALID_EXPORT_STATUSES)}"
            )

        # region_onehot_npy is required (schema: required, dataset: all 4590 present)
        # points_csv is optional
        onehot_rel = raw.get("region_onehot_npy", "").strip()
        if not onehot_rel:
            raise Slp8RegionDatasetError(
                f"region_onehot_npy is required but empty in {sample_id}"
            )
        points_rel = raw.get("points_csv", "").strip()
        source_flags_raw = raw.get("source_flags", "").strip()

        return Slp8RegionSample(
            sample_id=sample_id,
            split=raw["split"].strip(),
            setting=raw["setting"].strip(),
            subject_id=raw["subject_id"].strip(),
            cover=raw["cover"].strip(),
            frame_id=frame_id_val,
            posture=posture,
            pressure_npy=Path(raw["pressure_npy"].strip()),
            region_label_npy=Path(raw["region_label_npy"].strip()),
            region_onehot_npy=Path(onehot_rel) if onehot_rel else None,
            points_csv=Path(points_rel) if points_rel else None,
            height=int(raw["height"]),
            width=int(raw["width"]),
            class_ids_present=class_ids,
            background_pixel_count=int(raw["background_count"]),
            body_pixel_count=int(raw["body_pixel_count"]),
            clipped_ratio=float(raw["clipped_ratio"]),
            onehot_valid=raw.get("onehot_valid", "True").strip() == "True",
            onehot_roundtrip=raw.get("onehot_roundtrip", "True").strip() == "True",
            annotation_provenance=raw["annotation_provenance"].strip(),
            source_review_status=raw["source_review_status"].strip(),
            export_version=raw["export_version"].strip(),
            export_status=raw["export_status"].strip(),
            source_flags=tuple(
                f.strip() for f in source_flags_raw.split("|") if f.strip()
            ),
            source_status=raw.get("source_status", "").strip(),
            pressure_sha256=raw["source_pmarray_sha256"].strip(),
        )

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary without loading any arrays."""
        rows = self._load_manifest()
        postures: dict[str, int] = {}
        subjects: dict[str, int] = {}
        for raw in rows.values():
            p = raw["posture"].strip()
            postures[p] = postures.get(p, 0) + 1
            s = raw["subject_id"].strip()
            subjects[s] = subjects.get(s, 0) + 1

        return {
            "dataset_id": DATASET_ID,
            "adapter_version": ADAPTER_VERSION,
            "manifest_path": str(self._manifest_path),
            "dataset_root": str(self._root),
            "total_samples": len(rows),
            "unique_subjects": len(subjects),
            "per_posture": postures,
            "per_subject_count_min": min(subjects.values()),
            "per_subject_count_max": max(subjects.values()),
            "loaded_at": self._loaded_at,
        }


# ---------------------------------------------------------------------------
# Class schema access
# ---------------------------------------------------------------------------

def get_class_name(region_id: int) -> str:
    """Map integer class ID → canonical class name."""
    return CLASS_ID_TO_NAME[region_id]


def get_class_id(class_name: str) -> int:
    """Map canonical class name → integer class ID."""
    return CLASS_NAME_TO_ID[class_name]
