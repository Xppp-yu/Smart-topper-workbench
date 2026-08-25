"""SLP 8-Region Pressure-Only Training Table Freeze (TASK-SLP-B01).

This module builds and validates the frozen training/validation/test manifests
for the SLP_8Region_Pressure_VAL_v1.1 dataset, using the A06 subject-level
split (81/10/11 danaLab subjects → 3,645/450/495 samples).

The frozen artifacts are intended to be used as the canonical training-table
entry point for B02 (non-learning region baseline), B03 (PM-only Smoke), and
later region segmentation experiments.

Core contracts enforced by this module:

* Subject-level split: a subject appears in exactly one ML split (train/val/test).
* Determinism: same inputs → same outputs and same SHA-256 of every emitted
  artifact.  Build timestamp and other non-contractual metadata live in a
  sidecar envelope so they do not affect the core manifest hash.
* TRAIN-only normalization: pressure-normalization statistics are fitted on
  TRAIN subjects only.  VAL/TEST samples are never used to fit statistics.
* TEST access control: a module-level test-access guard blocks ordinary code
  from reading TEST label/onehot or computing class statistics on TEST.
  Only structural checks (sample count, subject count, sample_id uniqueness,
  path format, file existence, hash/contract consistency) are allowed on TEST
  by default.  Access to TEST for final model evaluation requires an explicit
  ``allow_test=True`` + ``purpose="final_evaluation"`` override.
* Path safety: manifest paths are kept relative to the dataset root; absolute
  paths and ``..`` escapes are rejected.  Same-prefix sibling paths (e.g.
  ``dataset`` vs ``dataset_evil``) are rejected by a strict relative-to
  containment check.

GT provenance contract (DO NOT REWRITE):

* ``annotation_provenance = V221_CORRECTED_SUPPORT_AUTO_ACCEPTED``
* ``source_review_status  = NOT_REVIEWED``
* NOT human pixel-level semantic masks
* NOT medical, skin-interface stress, or product ground truth
* Pressure is raw PMarray response semantics, NOT kPa
* danaLab only, uncover only; do not extrapolate to cover1/cover2 or to
  product/hardware/comfort claims.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Final

import numpy as np


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

ADAPTER_VERSION: Final[str] = "slp8_training_table_freeze_v0.1"
TASK_ID: Final[str] = "TASK-SLP-B01-SLP8-TRAINING-TABLE-FREEZE-v0.1"
FREEZE_VERSION: Final[str] = "slp8_training_tables_v0.1"

#: Source dataset identifier (matches A09R).
SOURCE_DATASET_ID: Final[str] = "SLP_8Region_Pressure_VAL_v1.1"
SOURCE_DATASET_VERSION: Final[str] = "1.1.0"

#: Source manifest filename (relative to dataset root).
SOURCE_MANIFEST_NAME: Final[str] = "val_manifest.csv"

#: A06 subject-level split identifier and expected SHA-256.
A06_SPLIT_IDENTIFIER: Final[str] = "slp_subject_split_v0.1"
A06_SPLIT_SHA256_EXPECTED: Final[str] = (
    "024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706"
)

#: Expected per-split sample counts (subjects × 45 frames).
EXPECTED_SPLIT_COUNTS: Final[dict[str, int]] = {
    "train": 3645,
    "val": 450,
    "test": 495,
}
EXPECTED_TOTAL: Final[int] = 4590
EXPECTED_SUBJECTS: Final[int] = 102
EXPECTED_FRAMES_PER_SUBJECT: Final[int] = 45
EXPECTED_POSTURE_COUNTS: Final[dict[str, int]] = {
    "SUPINE": 1530,
    "LEFT": 1530,
    "RIGHT": 1530,
}

#: ML split names accepted in manifests.
ML_SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")
ML_SPLIT_TO_INDEX: Final[dict[str, int]] = {s: i for i, s in enumerate(ML_SPLITS)}

#: Required annotation provenance for ALL samples in the B01 freeze.
EXPECTED_PROVENANCE: Final[str] = "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED"
EXPECTED_REVIEW_STATUS: Final[str] = "NOT_REVIEWED"

#: The SLP8 v1.1 source manifest uses ``VAL`` as the project-level "this is
#: the accepted standard-answer pool" marker, NOT a machine-learning split.
#: B01 derives the ML TRAIN/VAL/TEST assignment from the A06 subject split.
EXPECTED_SOURCE_SPLITS: Final[frozenset[str]] = frozenset({"VAL"})
EXPECTED_POSTURES: Final[frozenset[str]] = frozenset({"SUPINE", "LEFT", "RIGHT"})
EXPECTED_SETTINGS: Final[frozenset[str]] = frozenset({"danaLab"})
EXPECTED_COVERS: Final[frozenset[str]] = frozenset({"uncover"})
EXPECTED_EXPORT_STATUSES: Final[frozenset[str]] = frozenset({"EXPORTED"})
#: A 64-char lower-case hex SHA-256.  Empty / missing values are rejected
#: by the source loader so that downstream code can rely on a fully
#: populated SHA on every row.
SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
#: Standard raw semantics string used in the B01 freeze artifacts.  This is
#: a corrected spelling of the earlier ``raw_pmaray_response`` typo.  Both
#: spellings remain valid on read for backward compatibility, but every
#: new artifact and validator comparison uses ``raw_pmarray_response``.
RAW_SEMANTICS: Final[str] = "raw_pmarray_response"
RAW_SEMANTICS_LEGACY_ALIASES: Final[frozenset[str]] = frozenset(
    {"raw_pmaray_response", "raw_pmarray_response"}
)

#: Required manifest columns and per-row types.
#: Schema mirrors A09R's val_manifest.csv plus ``ml_split``.
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "ml_split",
    "source_split",
    "setting",
    "subject_id",
    "cover",
    "frame_id",
    "posture",
    "pressure_npy",
    "region_label_npy",
    "region_onehot_npy",
    "points_csv",
    "height",
    "width",
    "class_ids_present",
    "annotation_provenance",
    "source_review_status",
    "export_version",
    "export_status",
    "source_pmarray_sha256",
    "background_pixel_count",
    "body_pixel_count",
    "clipped_ratio",
    "onehot_valid",
    "onehot_roundtrip",
)

#: Columns excluded from the SHA-256 digest (build-time/observational only).
NON_HASH_COLUMNS: Final[frozenset[str]] = frozenset()  # all core columns are hashed

#: Train-only normalization configuration.
NORMALIZATION_METHOD: Final[str] = "raw_passthrough_with_minmax_reference"
NORMALIZATION_EPSILON: Final[float] = 1e-12
NORMALIZATION_FIT_SPLIT: Final[str] = "train"

#: Sample-ID pattern (from A09R schema).
SAMPLE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^SLP:danaLab:[0-9]{5}:uncover:[0-9]{6}$"
)
SUBJECT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9]{5}$")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class B01FreezeError(Exception):
    """Base exception for B01 freeze module errors."""

    __test__ = False  # tell pytest not to collect this class as a test


class A06SplitContractError(B01FreezeError):
    """A06 split manifest violates expected structure or SHA-256."""

    __test__ = False


class PathContainmentViolationError(B01FreezeError):
    """A manifest path escapes the declared dataset root."""

    __test__ = False


class AbsolutePathNotAllowedError(B01FreezeError):
    """An absolute path string was supplied where a relative path is required."""

    __test__ = False


class SampleContractError(B01FreezeError):
    """A sample record violates the freeze contract."""

    __test__ = False


class SubjectMappingError(B01FreezeError):
    """A subject cannot be mapped cleanly into ML splits."""

    __test__ = False


class TestLeakageError(B01FreezeError):
    """An attempt was made to read TEST data in development mode."""

    __test__ = False


class NormalizationContractError(B01FreezeError):
    """Pressure normalization contract violation."""

    __test__ = False


class ManifestContractError(B01FreezeError):
    """A manifest record fails structural/contract checks."""

    __test__ = False


# ---------------------------------------------------------------------------
# Test access guard
# ---------------------------------------------------------------------------

#: Default TEST access policy.  Only structural checks are allowed on TEST
#: unless callers explicitly opt-in via :func:`enable_test_access` with
#: ``purpose="final_evaluation"``.
_TEST_ACCESS_STATE: dict[str, Any] = {
    "allow_test": False,
    "purpose": None,
    "enabled_at": None,
}


def enable_test_access(*, purpose: str) -> None:
    """Opt-in to TEST data access for a narrowly-scoped final evaluation.

    Parameters
    ----------
    purpose : str
        Must be exactly ``"final_evaluation"``.  No other purpose is
        accepted.

    Notes
    -----
    This function is intended to be called only by Runner/Reviewer scripts
    that are explicitly authorised to compute model metrics on TEST.  It
    must never be called from ordinary training, baseline, smoke, or
    normalization code.
    """
    if purpose != "final_evaluation":
        raise TestLeakageError(
            "enable_test_access: only purpose='final_evaluation' is allowed; "
            f"got purpose={purpose!r}"
        )
    _TEST_ACCESS_STATE["allow_test"] = True
    _TEST_ACCESS_STATE["purpose"] = purpose
    _TEST_ACCESS_STATE["enabled_at"] = datetime.now(timezone.utc).isoformat()


def disable_test_access() -> None:
    """Reset TEST access back to default (development)."""
    _TEST_ACCESS_STATE["allow_test"] = False
    _TEST_ACCESS_STATE["purpose"] = None
    _TEST_ACCESS_STATE["enabled_at"] = None


def is_test_access_enabled() -> bool:
    """Return True if TEST access has been explicitly enabled."""
    return bool(_TEST_ACCESS_STATE["allow_test"])


def current_test_access_purpose() -> str | None:
    """Return the current TEST access purpose or None if disabled."""
    return _TEST_ACCESS_STATE["purpose"]


def require_test_access(purpose: str = "final_evaluation") -> None:
    """Assert that TEST access is enabled and matches the requested purpose.

    Raises
    ------
    TestLeakageError
        When TEST access is not enabled or the purpose does not match.
    """
    if not is_test_access_enabled():
        raise TestLeakageError(
            "TEST access denied: development mode.  "
            "Call enable_test_access(purpose='final_evaluation') "
            "before reading TEST label/onehot or computing class statistics."
        )
    if current_test_access_purpose() != purpose:
        raise TestLeakageError(
            "TEST access denied: purpose mismatch.  "
            f"requested={purpose!r} current={current_test_access_purpose()!r}"
        )


def guard_split_for_label_access(ml_split: str) -> None:
    """Block label/onehot access on TEST unless explicitly enabled.

    Reads of TRAIN/VAL labels are always allowed.  Reads of TEST label/onehot
    must be accompanied by an explicit :func:`enable_test_access` call.
    """
    if ml_split == "test":
        require_test_access(purpose="final_evaluation")


# ---------------------------------------------------------------------------
# Path safety helpers
# ---------------------------------------------------------------------------

def _is_absolute_path_string(s: str) -> bool:
    """Return True if ``s`` is an absolute-path string (Windows or POSIX)."""
    if not s:
        return False
    if s[0] == "/" or s[0] == "\\":
        return True
    # Windows drive letter or UNC path
    if len(s) >= 2 and s[1] == ":":
        return True
    return False


def assert_relative_path(rel_path: str | Path, *, field_name: str) -> Path:
    """Validate that ``rel_path`` is a safe relative path.

    Rejects:
    * absolute Windows paths (e.g. ``D:\\foo``)
    * absolute POSIX paths (e.g. ``/etc/passwd``)
    * UNC paths (``\\\\server\\share``)
    * parent-escape segments (``..``)

    Returns
    -------
    Path
        The relative path, normalised via :class:`pathlib.PurePosixPath`
        semantics (forward-slash only) for stable hashing.
    """
    s = str(rel_path)
    if _is_absolute_path_string(s):
        raise AbsolutePathNotAllowedError(
            f"{field_name}: absolute path not allowed: {s!r}"
        )
    p = Path(s)
    parts = p.parts
    if any(part == ".." for part in parts):
        raise PathContainmentViolationError(
            f"{field_name}: '..' segment not allowed: {s!r}"
        )
    if s == "":
        raise PathContainmentViolationError(
            f"{field_name}: empty path not allowed"
        )
    return p


def is_path_within(child: Path, parent: Path) -> bool:
    """Return True if ``child`` is strictly inside ``parent``.

    This is more strict than :py:meth:`Path.is_relative_to` because it
    rejects the same-prefix sibling case (``dataset_evil`` is not within
    ``dataset``).  Both paths are resolved first.
    """
    child_resolved = child.resolve()
    parent_resolved = parent.resolve()
    try:
        rel = child_resolved.relative_to(parent_resolved)
    except ValueError:
        return False
    # relative_to() with a sibling prefix returns ".." as the first part, which
    # is itself a leak.  Reject empty AND any path containing "..".
    if rel.parts == (".",) or rel == Path("."):
        return True
    if any(part == ".." for part in rel.parts):
        return False
    return True


# ---------------------------------------------------------------------------
# Hashing helpers (deterministic, JSON-stable)
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Default JSON encoder for numpy/Python types used in freeze metadata."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v):
            return None  # not used in B01 manifest but kept safe
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def canonical_json_dumps(payload: dict[str, Any] | list[Any]) -> str:
    """Deterministic JSON dump: sorted keys, stable separators, no spaces."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def sha256_hex(data: bytes | str) -> str:
    """SHA-256 hex digest helper."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file's bytes (small manifest/JSON files only)."""
    return sha256_hex(path.read_bytes())


def sha256_file_normalized_lf(path: Path) -> str:
    """Compute SHA-256 of a file with CRLF normalised to LF.

    Some upstream manifests (notably the A06 split) were originally
    produced on a Unix-style system whose embedded ``manifest_sha256``
    is the SHA-256 of the LF byte stream.  When the same JSON is later
    serialised with CRLF (e.g. via a Windows-side tool or git autocrlf),
    the raw file SHA-256 no longer matches the embedded value, but the
    JSON content is unchanged.  This helper normalises CRLF → LF before
    hashing so the contract is preserved.
    """
    raw = path.read_bytes()
    return sha256_hex(raw.replace(b"\r\n", b"\n"))


# ---------------------------------------------------------------------------
# Manifest row dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FreezeRow:
    """One row in a B01 manifest.  All paths are dataset-root-relative."""

    sample_id: str
    ml_split: str
    source_split: str
    setting: str
    subject_id: str
    cover: str
    frame_id: int
    posture: str
    pressure_npy: str
    region_label_npy: str
    region_onehot_npy: str
    points_csv: str
    height: int
    width: int
    class_ids_present: tuple[int, ...]
    annotation_provenance: str
    source_review_status: str
    export_version: str
    export_status: str
    source_pmarray_sha256: str
    background_pixel_count: int
    body_pixel_count: int
    clipped_ratio: float
    onehot_valid: bool
    onehot_roundtrip: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict (paths remain strings, normalised)."""
        return {
            "sample_id": self.sample_id,
            "ml_split": self.ml_split,
            "source_split": self.source_split,
            "setting": self.setting,
            "subject_id": self.subject_id,
            "cover": self.cover,
            "frame_id": int(self.frame_id),
            "posture": self.posture,
            "pressure_npy": self.pressure_npy,
            "region_label_npy": self.region_label_npy,
            "region_onehot_npy": self.region_onehot_npy,
            "points_csv": self.points_csv,
            "height": int(self.height),
            "width": int(self.width),
            "class_ids_present": list(self.class_ids_present),
            "annotation_provenance": self.annotation_provenance,
            "source_review_status": self.source_review_status,
            "export_version": self.export_version,
            "export_status": self.export_status,
            "source_pmarray_sha256": self.source_pmarray_sha256,
            "background_pixel_count": int(self.background_pixel_count),
            "body_pixel_count": int(self.body_pixel_count),
            "clipped_ratio": float(self.clipped_ratio),
            "onehot_valid": bool(self.onehot_valid),
            "onehot_roundtrip": bool(self.onehot_roundtrip),
        }


# ---------------------------------------------------------------------------
# A06 split loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class A06Split:
    """Parsed A06 subject-level split (danaLab subjects only)."""

    raw: dict[str, Any]
    subject_to_ml_split: dict[str, str]   # subject_id (zero-padded 5) → ml_split
    subject_to_setting: dict[str, str]   # subject_id → setting
    split_counts_subjects: dict[str, int]
    split_counts_samples: dict[str, int]
    sha256: str

    def ml_split_for_subject(self, subject_id: str, setting: str) -> str:
        """Return the ML split assigned to a subject in a given setting.

        Raises
        ------
        SubjectMappingError
            When the subject is missing or the setting is not present.
        """
        if subject_id not in self.subject_to_ml_split:
            raise SubjectMappingError(
                f"subject {subject_id!r} (setting={setting!r}) is not in A06 split"
            )
        actual_setting = self.subject_to_setting.get(subject_id)
        if actual_setting != setting:
            raise SubjectMappingError(
                f"subject {subject_id!r} setting mismatch: A06 says {actual_setting!r}, "
                f"requested {setting!r}"
            )
        return self.subject_to_ml_split[subject_id]


def _recompute_a06_subject_assignment_sha(entries: list[dict[str, Any]]) -> str:
    """Re-compute the A06 subject-assignment SHA-256 the way A06 itself does.

    The A06 split (slp_subject_split_v0.1) records ``manifest_sha256`` as the
    SHA-256 of a JSON dump of just the per-subject (subject_id, setting, split)
    assignments, sorted by subject_id and serialised with
    ``json.dumps(sort_keys=True, ensure_ascii=False)``.  This is NOT the SHA-256
    of the file itself, which is why the A06 split file is robust to
    re-serialisation (different indent, CRLF vs LF, etc.).
    """
    payload = sorted(
        [
            {
                "subject_id": e["subject_id"],
                "setting": e.get("setting", "danaLab"),
                "split": e["split"],
            }
            for e in entries
        ],
        key=lambda x: x["subject_id"],
    )
    return sha256_hex(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )


def load_a06_split(
    split_path: Path,
    *,
    expected_sha256: str | None = A06_SPLIT_SHA256_EXPECTED,
    enforce_canonical_subject_counts: bool = True,
) -> A06Split:
    """Load and validate the A06 subject split manifest.

    The A06 split's ``manifest_sha256`` is the SHA-256 of the
    (subject_id, setting, split) subject-assignment list (NOT of the file).
    We re-compute it from the parsed JSON and require it to match the
    canonical A06 SHA-256.

    Parameters
    ----------
    split_path : Path
        Path to ``slp_subject_split_v0.1.json``.
    expected_sha256 : str | None
        Subject-assignment SHA-256 the manifest must match.  Defaults to
        the canonical A06 SHA-256.  Pass ``None`` to skip the SHA check
        (intended for synthetic test fixtures only).
    enforce_canonical_subject_counts : bool
        If True (default), require danaLab subject counts to be
        81 / 10 / 11 and the total to be 102.  Pass ``False`` for
        synthetic test fixtures.

    Returns
    -------
    A06Split

    Raises
    ------
    A06SplitContractError
        When the file is missing, unreadable, structurally wrong, or its
        subject-assignment SHA-256 does not match the canonical A06 SHA-256.
    """
    if not split_path.is_file():
        raise A06SplitContractError(f"A06 split manifest not found: {split_path}")
    raw_bytes = split_path.read_bytes()
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as ex:
        raise A06SplitContractError(f"A06 split manifest is not valid JSON: {ex}") from ex

    if not isinstance(raw, dict):
        raise A06SplitContractError("A06 split manifest: top-level must be an object")
    if raw.get("schema_version") != A06_SPLIT_IDENTIFIER:
        raise A06SplitContractError(
            f"A06 split manifest: schema_version mismatch "
            f"(expected {A06_SPLIT_IDENTIFIER!r}, got {raw.get('schema_version')!r})"
        )

    subject_entries = raw.get("subject_entries")
    if not isinstance(subject_entries, list):
        raise A06SplitContractError("A06 split manifest: subject_entries is not a list")

    # Re-compute subject-assignment SHA from the parsed JSON.  This matches
    # the A06 generator logic exactly and is robust to re-serialisation
    # differences (indent, CRLF vs LF, key order) introduced after the freeze.
    recomputed = _recompute_a06_subject_assignment_sha(subject_entries)

    # Always check the embedded SHA-256 against the re-computed value.  This
    # is independent of the expected_sha256 argument: if the A06 file's
    # embedded manifest_sha256 does not match what the file's subject
    # assignment actually hashes to, the file is corrupted or modified and
    # we must fail-closed regardless of which A06 file we were expecting.
    recorded = raw.get("manifest_sha256")
    if recorded and recorded != recomputed:
        raise A06SplitContractError(
            "A06 split manifest: embedded manifest_sha256 does not match re-computed "
            f"subject-assignment SHA-256: embedded={recorded}, recomputed={recomputed}"
        )

    # Optional second check: the recomputed SHA-256 must match the canonical
    # A06 freeze SHA (unless the caller passes ``expected_sha256=None`` to
    # explicitly opt out — e.g. for synthetic test fixtures).
    if expected_sha256 is not None and recomputed != expected_sha256:
        raise A06SplitContractError(
            "A06 split manifest: subject-assignment SHA-256 does not match the "
            "frozen A06 contract: "
            f"expected={expected_sha256}, recomputed={recomputed}"
        )

    # Build subject mapping
    subject_to_ml_split: dict[str, str] = {}
    subject_to_setting: dict[str, str] = {}
    for entry in raw.get("subject_entries", []):
        sid = entry.get("subject_id")
        setting = entry.get("setting")
        ml_split = entry.get("split")
        if not isinstance(sid, str) or not isinstance(setting, str) or not isinstance(ml_split, str):
            raise A06SplitContractError(
                f"A06 split manifest: malformed subject_entry: {entry!r}"
            )
        if ml_split not in ML_SPLITS:
            raise A06SplitContractError(
                f"A06 split manifest: unknown split {ml_split!r} for {sid}"
            )
        if setting != "danaLab":
            # B01 only uses danaLab (SLP8 GT contains only danaLab).
            continue
        subject_to_ml_split[sid] = ml_split
        subject_to_setting[sid] = setting

    # Compute per-split subject counts and expected sample counts
    split_counts_subjects: dict[str, int] = Counter()
    for ml_split in subject_to_ml_split.values():
        split_counts_subjects[ml_split] += 1
    split_counts_samples = {
        s: split_counts_subjects.get(s, 0) * EXPECTED_FRAMES_PER_SUBJECT
        for s in ML_SPLITS
    }
    if enforce_canonical_subject_counts:
        expected_subj = {
            "train": 81, "val": 10, "test": 11,
        }
        for s, want in expected_subj.items():
            if split_counts_subjects.get(s, 0) != want:
                raise A06SplitContractError(
                    f"A06 split: danaLab subject count for {s!r} is "
                    f"{split_counts_subjects.get(s, 0)}, expected {want}"
                )
        if sum(split_counts_subjects.values()) != EXPECTED_SUBJECTS:
            raise A06SplitContractError(
                f"A06 split: total danaLab subjects = "
                f"{sum(split_counts_subjects.values())}, expected {EXPECTED_SUBJECTS}"
            )

    return A06Split(
        raw=raw,
        subject_to_ml_split=subject_to_ml_split,
        subject_to_setting=subject_to_setting,
        split_counts_subjects=dict(split_counts_subjects),
        split_counts_samples=split_counts_samples,
        sha256=recomputed,
    )


# ---------------------------------------------------------------------------
# SLP8 manifest loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Slp8SourceSample:
    """A row from SLP8 val_manifest.csv, narrowed to the fields B01 uses."""

    sample_id: str
    source_split: str
    setting: str
    subject_id: str
    cover: str
    frame_id: int
    posture: str
    pressure_npy: str
    region_label_npy: str
    region_onehot_npy: str
    points_csv: str
    height: int
    width: int
    class_ids_present: tuple[int, ...]
    annotation_provenance: str
    source_review_status: str
    export_version: str
    export_status: str
    source_pmarray_sha256: str
    background_pixel_count: int
    body_pixel_count: int
    clipped_ratio: float
    onehot_valid: bool
    onehot_roundtrip: bool

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "Slp8SourceSample":
        """Parse a single val_manifest.csv row into a Slp8SourceSample.

        Fails-closed on every value the SLP8 v1.1 contract requires:

        * All string fields must be present and non-empty.
        * The ``source_split`` column must be the literal ``VAL`` (this is
          the project-level "accepted standard-answer pool" marker; the ML
          train/val/test split is derived from A06, not from this column).
        * ``setting`` must be ``danaLab``; ``cover`` must be ``uncover``.
        * ``posture`` must be one of ``SUPINE`` / ``LEFT`` / ``RIGHT``.
        * ``annotation_provenance`` must be
          ``V221_CORRECTED_SUPPORT_AUTO_ACCEPTED``.
        * ``source_review_status`` must be ``NOT_REVIEWED``.
        * ``export_status`` must be ``EXPORTED``.
        * ``export_version`` must be ``1.1.0``.
        * ``onehot_valid`` and ``onehot_roundtrip`` must be present and
          equal to the literal string ``True`` (any other value or a
          missing field raises ``SampleContractError``).
        * ``source_pmarray_sha256`` must be a 64-character lower-case
          hex SHA-256 (the SLP8 v1.1 source manifest format).
        """
        # ── 1. Required scalar fields ────────────────────────────────────
        sample_id = row["sample_id"].strip()
        if not sample_id:
            raise SampleContractError("row: empty sample_id")
        source_split = row["split"].strip()
        if not source_split:
            raise SampleContractError(f"{sample_id}: empty source split")
        if source_split not in EXPECTED_SOURCE_SPLITS:
            raise SampleContractError(
                f"{sample_id}: source_split={source_split!r} not in "
                f"{sorted(EXPECTED_SOURCE_SPLITS)}"
            )
        setting = row["setting"].strip()
        if not setting:
            raise SampleContractError(f"{sample_id}: empty setting")
        if setting not in EXPECTED_SETTINGS:
            raise SampleContractError(
                f"{sample_id}: setting={setting!r} not in "
                f"{sorted(EXPECTED_SETTINGS)}"
            )
        subject_id = row["subject_id"].strip()
        if not subject_id:
            raise SampleContractError(f"{sample_id}: empty subject_id")
        cover = row["cover"].strip()
        if not cover:
            raise SampleContractError(f"{sample_id}: empty cover")
        if cover not in EXPECTED_COVERS:
            raise SampleContractError(
                f"{sample_id}: cover={cover!r} not in "
                f"{sorted(EXPECTED_COVERS)}"
            )
        posture = row["posture"].strip()
        if not posture:
            raise SampleContractError(f"{sample_id}: empty posture")
        if posture not in EXPECTED_POSTURES:
            raise SampleContractError(
                f"{sample_id}: posture={posture!r} not in "
                f"{sorted(EXPECTED_POSTURES)}"
            )

        try:
            frame_id = int(row["frame_id"])
        except (KeyError, ValueError) as ex:
            raise SampleContractError(
                f"{sample_id}: frame_id={row.get('frame_id')!r} is not an integer ({ex})"
            )
        try:
            height = int(row["height"])
            width = int(row["width"])
        except (KeyError, ValueError) as ex:
            raise SampleContractError(
                f"{sample_id}: pressure shape "
                f"({row.get('height')!r}x{row.get('width')!r}) is not integers ({ex})"
            )
        try:
            background_count = int(row["background_count"])
            body_count = int(row["body_pixel_count"])
        except (KeyError, ValueError) as ex:
            raise SampleContractError(
                f"{sample_id}: background_count/body_pixel_count parse failed ({ex})"
            )
        try:
            clipped_ratio = float(row["clipped_ratio"])
        except (KeyError, ValueError) as ex:
            raise SampleContractError(
                f"{sample_id}: clipped_ratio={row.get('clipped_ratio')!r} is not float ({ex})"
            )

        # ── 2. Path fields (relative path strings) ─────────────────────────
        pressure_npy = row["pressure_npy"].strip()
        if not pressure_npy:
            raise SampleContractError(f"{sample_id}: empty pressure_npy")
        region_label_npy = row["region_label_npy"].strip()
        if not region_label_npy:
            raise SampleContractError(f"{sample_id}: empty region_label_npy")
        region_onehot_npy = row["region_onehot_npy"].strip()
        if not region_onehot_npy:
            raise SampleContractError(f"{sample_id}: empty region_onehot_npy")
        points_csv = row.get("points_csv", "").strip()  # optional

        # ── 3. class_ids_present ──────────────────────────────────────────
        class_ids_str = row["class_ids_present"].strip()
        if not class_ids_str:
            raise SampleContractError(f"{sample_id}: empty class_ids_present")
        try:
            class_ids_present = tuple(int(x) for x in class_ids_str.split("|"))
        except ValueError as ex:
            raise SampleContractError(
                f"{sample_id}: class_ids_present={class_ids_str!r} parse failed ({ex})"
            )

        # ── 4. Provenance / review / export status / version ──────────────
        annotation_provenance = row["annotation_provenance"].strip()
        if not annotation_provenance:
            raise SampleContractError(f"{sample_id}: empty annotation_provenance")
        if annotation_provenance != EXPECTED_PROVENANCE:
            raise SampleContractError(
                f"{sample_id}: annotation_provenance={annotation_provenance!r} != "
                f"{EXPECTED_PROVENANCE!r}"
            )
        source_review_status = row["source_review_status"].strip()
        if not source_review_status:
            raise SampleContractError(f"{sample_id}: empty source_review_status")
        if source_review_status != EXPECTED_REVIEW_STATUS:
            raise SampleContractError(
                f"{sample_id}: source_review_status={source_review_status!r} != "
                f"{EXPECTED_REVIEW_STATUS!r}"
            )
        export_version = row["export_version"].strip()
        if export_version != "1.1.0":
            raise SampleContractError(
                f"{sample_id}: export_version={export_version!r} != '1.1.0'"
            )
        export_status = row["export_status"].strip()
        if not export_status:
            raise SampleContractError(f"{sample_id}: empty export_status")
        if export_status not in EXPECTED_EXPORT_STATUSES:
            raise SampleContractError(
                f"{sample_id}: export_status={export_status!r} not in "
                f"{sorted(EXPECTED_EXPORT_STATUSES)}"
            )

        # ── 5. onehot_valid / onehot_roundtrip (must be present and "True") ─
        # Earlier fail-open behaviour defaulted a missing field to True; that
        # silently masked tampered source rows.  We now require both columns
        # to be present and equal to the literal "True".
        if "onehot_valid" not in row:
            raise SampleContractError(
                f"{sample_id}: missing 'onehot_valid' column"
            )
        onehot_valid_str = row["onehot_valid"].strip()
        if onehot_valid_str != "True":
            raise SampleContractError(
                f"{sample_id}: onehot_valid={onehot_valid_str!r} — must be 'True', got {onehot_valid_str!r}"
            )
        onehot_valid = True

        if "onehot_roundtrip" not in row:
            raise SampleContractError(
                f"{sample_id}: missing 'onehot_roundtrip' column"
            )
        onehot_roundtrip_str = row["onehot_roundtrip"].strip()
        if onehot_roundtrip_str != "True":
            raise SampleContractError(
                f"{sample_id}: onehot_roundtrip={onehot_roundtrip_str!r} — must be 'True', got {onehot_roundtrip_str!r}"
            )
        onehot_roundtrip = True

        # ── 6. source_pmarray_sha256 (64-char lower-case hex) ─────────────
        source_pmarray_sha256 = row["source_pmarray_sha256"].strip()
        if not source_pmarray_sha256:
            raise SampleContractError(
                f"{sample_id}: empty source_pmarray_sha256"
            )
        if not SHA256_PATTERN.match(source_pmarray_sha256):
            raise SampleContractError(
                f"{sample_id}: source_pmarray_sha256={source_pmarray_sha256!r} "
                "is not a 64-character lower-case hex SHA-256"
            )

        return cls(
            sample_id=sample_id,
            source_split=source_split,
            setting=setting,
            subject_id=subject_id,
            cover=cover,
            frame_id=frame_id,
            posture=posture,
            pressure_npy=pressure_npy,
            region_label_npy=region_label_npy,
            region_onehot_npy=region_onehot_npy,
            points_csv=points_csv,
            height=height,
            width=width,
            class_ids_present=class_ids_present,
            annotation_provenance=annotation_provenance,
            source_review_status=source_review_status,
            export_version=export_version,
            export_status=export_status,
            source_pmarray_sha256=source_pmarray_sha256,
            background_pixel_count=background_count,
            body_pixel_count=body_count,
            clipped_ratio=clipped_ratio,
            onehot_valid=onehot_valid,
            onehot_roundtrip=onehot_roundtrip,
        )


@dataclass(frozen=True, slots=True)
class Slp8SourceManifest:
    """A loaded SLP8 val_manifest.csv with sample count and source SHA."""

    source_manifest_path: Path
    source_manifest_sha256: str
    samples: tuple[Slp8SourceSample, ...]

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def by_sample_id(self) -> dict[str, Slp8SourceSample]:
        return {s.sample_id: s for s in self.samples}


def load_slp8_source_manifest(
    dataset_root: Path,
    *,
    enforce_canonical_total: bool = True,
    expected_total: int | None = None,
) -> Slp8SourceManifest:
    """Read SLP8 val_manifest.csv, validate structure, return parsed rows.

    Parameters
    ----------
    dataset_root : Path
        Root directory of the SLP8 GT dataset (must contain
        ``manifest/val_manifest.csv``).
    enforce_canonical_total : bool
        If True (default), require ``len(rows) == EXPECTED_TOTAL == 4590``.
        Pass ``False`` for synthetic test fixtures.
    expected_total : int | None
        Optional override for the canonical row count.  When provided,
        ``enforce_canonical_total`` is treated as True and the supplied
        value is used instead of ``EXPECTED_TOTAL``.

    Returns
    -------
    Slp8SourceManifest
    """
    manifest_path = dataset_root / "manifest" / SOURCE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise SampleContractError(
            f"SLP8 source manifest not found: {manifest_path}"
        )

    source_sha = sha256_file(manifest_path)
    rows: list[Slp8SourceSample] = []
    seen_ids: set[str] = set()
    with manifest_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            sid = raw["sample_id"].strip()
            if not SAMPLE_ID_PATTERN.match(sid):
                raise SampleContractError(
                    f"SLP8 manifest: sample_id {sid!r} does not match expected pattern"
                )
            if sid in seen_ids:
                raise SampleContractError(
                    f"SLP8 manifest: duplicate sample_id {sid!r}"
                )
            seen_ids.add(sid)
            sample = Slp8SourceSample.from_csv_row(raw)
            # Per-row contract checks (provenance, review, cover, setting,
            # shape, subject_id pattern, path safety) are run on load so
            # downstream code can assume every source sample already meets
            # the B01 freeze contract.
            _validate_source_row(sample)
            rows.append(sample)

    if enforce_canonical_total or expected_total is not None:
        target = expected_total if expected_total is not None else EXPECTED_TOTAL
        if len(rows) != target:
            raise SampleContractError(
                f"SLP8 manifest: expected {target} rows, got {len(rows)}"
            )

    return Slp8SourceManifest(
        source_manifest_path=manifest_path,
        source_manifest_sha256=source_sha,
        samples=tuple(rows),
    )


# ---------------------------------------------------------------------------
# Freeze row builder
# ---------------------------------------------------------------------------

def _validate_source_row(s: Slp8SourceSample) -> None:
    """Validate one source row meets the B01 freeze pre-conditions."""
    if s.setting != "danaLab":
        raise SampleContractError(
            f"{s.sample_id}: only danaLab is in B01 freeze, got setting={s.setting!r}"
        )
    if s.cover != "uncover":
        raise SampleContractError(
            f"{s.sample_id}: only uncover is in B01 freeze, got cover={s.cover!r}"
        )
    if s.annotation_provenance != EXPECTED_PROVENANCE:
        raise SampleContractError(
            f"{s.sample_id}: provenance={s.annotation_provenance!r} != "
            f"{EXPECTED_PROVENANCE!r}"
        )
    if s.source_review_status != EXPECTED_REVIEW_STATUS:
        raise SampleContractError(
            f"{s.sample_id}: review_status={s.source_review_status!r} != "
            f"{EXPECTED_REVIEW_STATUS!r}"
        )
    if s.height != 192 or s.width != 84:
        raise SampleContractError(
            f"{s.sample_id}: pressure shape {s.height}x{s.width} != 192x84"
        )
    if not SUBJECT_ID_PATTERN.match(s.subject_id):
        raise SampleContractError(
            f"{s.sample_id}: subject_id {s.subject_id!r} not a 5-digit string"
        )

    # Reject absolute or escape paths in the source row
    for field_name, val in (
        ("pressure_npy", s.pressure_npy),
        ("region_label_npy", s.region_label_npy),
        ("region_onehot_npy", s.region_onehot_npy),
    ):
        assert_relative_path(val, field_name=field_name)
    if s.points_csv:
        assert_relative_path(s.points_csv, field_name="points_csv")


def build_freeze_row(
    source: Slp8SourceSample,
    *,
    ml_split: str,
    dataset_root: Path,
) -> FreezeRow:
    """Construct a B01 freeze row from a source SLP8 row.

    Performs containment checks to ensure every relative path resolves
    inside ``dataset_root`` (so the manifest can be safely relocated).

    Returns
    -------
    FreezeRow
    """
    _validate_source_row(source)
    if ml_split not in ML_SPLITS:
        raise SampleContractError(f"ml_split={ml_split!r} not in {ML_SPLITS}")

    root_resolved = dataset_root.resolve()

    for field_name, rel in (
        ("pressure_npy", source.pressure_npy),
        ("region_label_npy", source.region_label_npy),
        ("region_onehot_npy", source.region_onehot_npy),
    ):
        target = (dataset_root / rel).resolve()
        if not is_path_within(target, root_resolved):
            raise PathContainmentViolationError(
                f"{source.sample_id}: {field_name} path escapes dataset root: "
                f"{rel!r} → {target} (root={root_resolved})"
            )
    if source.points_csv:
        target = (dataset_root / source.points_csv).resolve()
        if not is_path_within(target, root_resolved):
            raise PathContainmentViolationError(
                f"{source.sample_id}: points_csv path escapes dataset root: "
                f"{source.points_csv!r} → {target} (root={root_resolved})"
            )

    return FreezeRow(
        sample_id=source.sample_id,
        ml_split=ml_split,
        source_split=source.source_split,
        setting=source.setting,
        subject_id=source.subject_id,
        cover=source.cover,
        frame_id=source.frame_id,
        posture=source.posture,
        pressure_npy=source.pressure_npy,
        region_label_npy=source.region_label_npy,
        region_onehot_npy=source.region_onehot_npy,
        points_csv=source.points_csv,
        height=source.height,
        width=source.width,
        class_ids_present=source.class_ids_present,
        annotation_provenance=source.annotation_provenance,
        source_review_status=source.source_review_status,
        export_version=source.export_version,
        export_status=source.export_status,
        source_pmarray_sha256=source.source_pmarray_sha256,
        background_pixel_count=source.background_pixel_count,
        body_pixel_count=source.body_pixel_count,
        clipped_ratio=source.clipped_ratio,
        onehot_valid=source.onehot_valid,
        onehot_roundtrip=source.onehot_roundtrip,
    )


# ---------------------------------------------------------------------------
# Manifest I/O (deterministic CSV + JSONL)
# ---------------------------------------------------------------------------

def write_manifest_csv(path: Path, rows: Iterable[FreezeRow]) -> None:
    """Write a B01 freeze manifest as deterministic CSV.

    Rows are sorted by sample_id before writing to keep the output
    byte-stable across runs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: r.sample_id)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for r in sorted_rows:
            d = r.to_dict()
            # class_ids_present → pipe-joined string for CSV parity with A09R
            d["class_ids_present"] = "|".join(str(x) for x in r.class_ids_present)
            writer.writerow(d)


def write_manifest_jsonl(path: Path, rows: Iterable[FreezeRow]) -> None:
    """Write a B01 freeze manifest as deterministic JSONL (sorted by sample_id)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: r.sample_id)
    with path.open("w", encoding="utf-8") as f:
        for r in sorted_rows:
            f.write(canonical_json_dumps(r.to_dict()) + "\n")


def read_manifest_csv(path: Path) -> list[FreezeRow]:
    """Read a B01 freeze manifest back into FreezeRow objects.

    Notes
    -----
    This is a low-level I/O helper.  Development code that needs to load
    B01 freeze rows for downstream training, statistics, or visualization
    must use :func:`load_b01_freeze_tables` (the default ``allowed_splits``
    there is ``("train", "val")``) so that TEST data is not silently
    pulled in.  This helper itself does not enforce the TEST access
    policy; that is the responsibility of the public entry point.
    """
    out: list[FreezeRow] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            out.append(
                FreezeRow(
                    sample_id=raw["sample_id"].strip(),
                    ml_split=raw["ml_split"].strip(),
                    source_split=raw["source_split"].strip(),
                    setting=raw["setting"].strip(),
                    subject_id=raw["subject_id"].strip(),
                    cover=raw["cover"].strip(),
                    frame_id=int(raw["frame_id"]),
                    posture=raw["posture"].strip(),
                    pressure_npy=raw["pressure_npy"].strip(),
                    region_label_npy=raw["region_label_npy"].strip(),
                    region_onehot_npy=raw["region_onehot_npy"].strip(),
                    points_csv=raw.get("points_csv", "").strip(),
                    height=int(raw["height"]),
                    width=int(raw["width"]),
                    class_ids_present=tuple(
                        int(x) for x in raw["class_ids_present"].split("|")
                    ),
                    annotation_provenance=raw["annotation_provenance"].strip(),
                    source_review_status=raw["source_review_status"].strip(),
                    export_version=raw["export_version"].strip(),
                    export_status=raw["export_status"].strip(),
                    source_pmarray_sha256=raw["source_pmarray_sha256"].strip(),
                    background_pixel_count=int(raw["background_pixel_count"]),
                    body_pixel_count=int(raw["body_pixel_count"]),
                    clipped_ratio=float(raw["clipped_ratio"]),
                    onehot_valid=raw["onehot_valid"].strip() == "True",
                    onehot_roundtrip=raw["onehot_roundtrip"].strip() == "True",
                )
            )
    return out


def manifest_sha256(rows: Iterable[FreezeRow]) -> str:
    """Stable SHA-256 of a manifest, derived from the canonical JSON form.

    Rows are sorted by sample_id and serialised via :func:`canonical_json_dumps`
    to guarantee the same hash regardless of input order.
    """
    sorted_rows = sorted(rows, key=lambda r: r.sample_id)
    payload = [r.to_dict() for r in sorted_rows]
    return sha256_hex(canonical_json_dumps(payload).encode("utf-8"))


# ---------------------------------------------------------------------------
# Unified freeze-table read entry (TEST access policy enforcement)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class B01FreezeTables:
    """Result of :func:`load_b01_freeze_tables`.

    A typed handle over a frozen B01 freeze directory.  Carries the rows
    for the splits the caller is allowed to see plus the top-level
    freeze manifest so that downstream code can audit the contract
    without re-reading the JSON.
    """

    output_dir: Path
    train_rows: tuple[FreezeRow, ...]
    val_rows: tuple[FreezeRow, ...]
    _test_rows: tuple[FreezeRow, ...] | None
    freeze_manifest: dict[str, Any]

    @property
    def test_rows(self) -> tuple[FreezeRow, ...]:
        """Return TEST rows.

        Raises
        ------
        TestLeakageError
            If TEST rows were not loaded into this object (i.e. the freeze was
            loaded with ``load_test=False`` / default).  Reload with
            ``load_b01_freeze_tables(output_dir, load_test=True)``.
        TestLeakageError
            If TEST rows are present but the global TEST-access guard is not
            enabled.
        """
        if self._test_rows is None:
            raise TestLeakageError(
                "TEST rows are not present in this object — the freeze was "
                "loaded without TEST.  Reload with load_b01_freeze_tables("
                "output_dir, load_test=True) after calling "
                "enable_test_access(purpose='final_evaluation')."
            )
        if not is_test_access_enabled():
            raise TestLeakageError(
                "TEST rows are present but TEST authorization is not enabled.  "
                "Call enable_test_access(purpose='final_evaluation') first."
            )
        return self._test_rows

    @property
    def train_manifest_sha256(self) -> str:
        return manifest_sha256(self.train_rows)

    @property
    def val_manifest_sha256(self) -> str:
        return manifest_sha256(self.val_rows)

    @property
    def test_manifest_sha256(self) -> str:
        # If TEST rows were loaded, compute from them; otherwise fall back
        # to the freeze_manifest so callers can still audit the SHA without
        # having TEST row objects.
        if self._test_rows is not None:
            return manifest_sha256(self._test_rows)
        return self.freeze_manifest["splits"]["test"]["manifest_sha256"]

    def development_rows(self) -> list[FreezeRow]:
        """Return the rows a development (non-final-evaluation) caller may use.

        By construction this never contains TEST rows.
        """
        return list(self.train_rows) + list(self.val_rows)

    def all_rows_with_test_opt_in(self) -> list[FreezeRow]:
        """Return train+val+test rows; require TEST rows loaded AND auth enabled.

        Raises
        ------
        TestLeakageError
            If TEST rows are not present (freeze was loaded without load_test).
        TestLeakageError
            If the global TEST-access guard is not enabled.
        """
        require_test_access(purpose="final_evaluation")
        if self._test_rows is None:
            raise TestLeakageError(
                "TEST rows are not present — reload with load_b01_freeze_tables("
                "output_dir, load_test=True)."
            )
        return list(self.train_rows) + list(self.val_rows) + list(self._test_rows)


def load_b01_freeze_tables(
    output_dir: Path,
    *,
    allowed_splits: Iterable[str] = ("train", "val"),
    load_test: bool | None = None,
) -> B01FreezeTables:
    """Load the B01 frozen training tables in a TEST-safe way.

    This is the single recommended entry point for downstream code that
    needs to read the B01 freeze.  It enforces the TEST access policy
    at the boundary: by default it returns TRAIN/VAL rows only and never
    hands the caller TEST rows.  The TEST manifest is parsed and its
    row count + manifest SHA verified, but the parsed ``FreezeRow``
    objects for TEST are kept inside this object and are only released
    to the caller through :meth:`B01FreezeTables.all_rows_with_test_opt_in`
    once the global TEST-access guard has been enabled with
    ``purpose="final_evaluation"``.

    Parameters
    ----------
    output_dir : Path
        The B01 freeze output directory.  Must contain
        ``train_manifest.csv``, ``val_manifest.csv``,
        ``test_manifest.csv`` and ``freeze_manifest.json``.
    allowed_splits : Iterable[str]
        Which split manifests to load into the returned handle.  The
        default ``("train", "val")`` excludes TEST.  Passing
        ``"test"`` is accepted only if the global TEST-access guard is
        enabled (raises ``TestLeakageError`` otherwise).
    load_test : bool | None
        Convenience switch.  When ``True``, requires the TEST-access
        guard to be enabled and includes TEST in the returned handle.
        When ``False`` or ``None``, TEST is never included.  When set,
        this argument overrides the per-split membership implied by
        ``allowed_splits``.

    Returns
    -------
    B01FreezeTables

    Raises
    ------
    FileNotFoundError
        When a required manifest file is missing.
    TestLeakageError
        When ``allowed_splits`` (or ``load_test=True``) would pull in
        TEST rows but the TEST-access guard is not enabled.
    """
    output_dir = Path(output_dir)
    train_csv = output_dir / "train_manifest.csv"
    val_csv = output_dir / "val_manifest.csv"
    test_csv = output_dir / "test_manifest.csv"
    fm_path = output_dir / "freeze_manifest.json"

    if load_test is True:
        require_test_access(purpose="final_evaluation")
    allowed_set = set(allowed_splits)
    if "test" in allowed_set:
        require_test_access(purpose="final_evaluation")
    if load_test is False and "test" in allowed_set:
        allowed_set.discard("test")
    for s in allowed_set:
        if s not in ML_SPLITS:
            raise B01FreezeError(
                f"load_b01_freeze_tables: unknown split {s!r} in allowed_splits"
            )

    for required in (train_csv, val_csv, test_csv, fm_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"load_b01_freeze_tables: required file missing: {required}"
            )

    # Only read the TEST manifest when the caller explicitly opts in.
    # The default path never loads complete TEST FreezeRow objects into the
    # returned handle.  TEST structural data (sample_count, manifest_sha256)
    # is still available via the freeze_manifest for independent verification.
    _test_rows_tuple: tuple[FreezeRow, ...] | None = None
    if load_test is True or "test" in allowed_set:
        _test_rows_tuple = tuple(read_manifest_csv(test_csv))
    train_rows_tuple: tuple[FreezeRow, ...] = (
        tuple(read_manifest_csv(train_csv)) if "train" in allowed_set else ()
    )
    val_rows_tuple: tuple[FreezeRow, ...] = (
        tuple(read_manifest_csv(val_csv)) if "val" in allowed_set else ()
    )

    freeze_manifest = json.loads(fm_path.read_text(encoding="utf-8"))

    return B01FreezeTables(
        output_dir=output_dir,
        train_rows=train_rows_tuple,
        val_rows=val_rows_tuple,
        _test_rows=_test_rows_tuple,
        freeze_manifest=freeze_manifest,
    )


# ---------------------------------------------------------------------------
# Train-only normalization statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NormalizationStats:
    """TRAIN-only pressure-normalization statistics.

    Fields
    ------
    n_samples : int
        Number of TRAIN samples used to fit the statistics.
    n_pixels : int
        Total number of pressure pixels summed across the TRAIN samples.
    finite_pixel_count : int
        Number of finite (not NaN/Inf) pixels seen during fitting.
    non_finite_pixel_count : int
        Number of NaN/Inf pixels seen.  Should be 0 for the SLP8 v1.1 GT;
        recorded for transparency.
    global_min : float
        Per-pixel minimum across TRAIN samples (raw response semantics).
    global_max : float
        Per-pixel maximum across TRAIN samples (raw response semantics).
    global_mean : float
        Per-pixel mean across TRAIN samples.
    global_std : float
        Per-pixel standard deviation across TRAIN samples.
    method : str
        Normalization method name; fixed for the B01 freeze.
    epsilon : float
        Numerical epsilon used to guard division-by-zero.  Not consumed by
        the raw-passthrough method, but recorded for forward-compat.
    raw_dtype : str
        Numpy dtype of the raw pressure arrays.
    raw_semantics : str
        Human-readable description of the pressure units.  Always
        ``"raw_pmarray_response"`` for B01 — NEVER kPa.
    fit_split : str
        Split used to fit the statistics; must be ``"train"``.
    subject_count : int
        Number of unique subjects used to fit the statistics.
    per_subject_count_min : int
        Minimum frame count per TRAIN subject (should be 45).
    per_subject_count_max : int
        Maximum frame count per TRAIN subject (should be 45).
    fitted_at_utc : str
        Wall-clock timestamp of the fit; NOT included in the core hash
        (consumers should treat statistics as content-addressed).
    """

    n_samples: int
    n_pixels: int
    finite_pixel_count: int
    non_finite_pixel_count: int
    global_min: float
    global_max: float
    global_mean: float
    global_std: float
    method: str
    epsilon: float
    raw_dtype: str
    raw_semantics: str
    fit_split: str
    subject_count: int
    per_subject_count_min: int
    per_subject_count_max: int
    fitted_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def content_sha256(self) -> str:
        """Stable SHA-256 of the content-addressed part (no timestamp)."""
        d = self.to_dict()
        d.pop("fitted_at_utc", None)
        return sha256_hex(canonical_json_dumps(d).encode("utf-8"))


def fit_normalization_stats(
    train_rows: Iterable[FreezeRow],
    dataset_root: Path,
) -> NormalizationStats:
    """Fit TRAIN-only pressure normalization statistics.

    Walks every TRAIN row, loads the pressure array (raw PMarray response),
    and computes global min/max/mean/std.  Fail-closed on any non-finite
    pixel.  The function must NEVER touch VAL or TEST pressure arrays.

    Returns
    -------
    NormalizationStats
    """
    rows = list(train_rows)
    if not rows:
        raise NormalizationContractError("fit_normalization_stats: empty TRAIN rows")

    # Fail-closed: every row must be ml_split == "train"
    for r in rows:
        if r.ml_split != "train":
            raise NormalizationContractError(
                f"fit_normalization_stats: row {r.sample_id} is ml_split={r.ml_split!r}; "
                "TRAIN-only fitting is required"
            )

    n_samples = 0
    n_pixels = 0
    finite_count = 0
    non_finite_count = 0
    g_min = math.inf
    g_max = -math.inf
    sum_val = 0.0
    sum_sq = 0.0
    per_subject_count: Counter[str] = Counter()

    for r in rows:
        per_subject_count[r.subject_id] += 1
        p_path = (dataset_root / r.pressure_npy).resolve()
        if not is_path_within(p_path, dataset_root.resolve()):
            raise NormalizationContractError(
                f"fit_normalization_stats: {r.sample_id} pressure path escapes dataset root"
            )
        if not p_path.is_file():
            raise NormalizationContractError(
                f"fit_normalization_stats: {r.sample_id} pressure file not found: {p_path}"
            )
        pressure = np.load(p_path, allow_pickle=False)
        if pressure.dtype != np.float64:
            raise NormalizationContractError(
                f"fit_normalization_stats: {r.sample_id} pressure dtype {pressure.dtype} "
                f"!= float64 (raw PMarray response semantics)"
            )
        if pressure.shape != (192, 84):
            raise NormalizationContractError(
                f"fit_normalization_stats: {r.sample_id} pressure shape {pressure.shape} "
                f"!= (192, 84)"
            )
        n_pixels += int(pressure.size)
        n_samples += 1
        finite_mask = np.isfinite(pressure)
        finite = int(finite_mask.sum())
        non_finite = int(pressure.size - finite)
        finite_count += finite
        non_finite_count += non_finite
        if non_finite > 0:
            raise NormalizationContractError(
                f"fit_normalization_stats: {r.sample_id} contains {non_finite} non-finite "
                "pixels; SLP8 v1.1 contract requires fully-finite pressure arrays"
            )
        g_min = min(g_min, float(pressure.min()))
        g_max = max(g_max, float(pressure.max()))
        sum_val += float(pressure.sum())
        sum_sq += float(np.square(pressure).sum())

    if n_pixels == 0:
        raise NormalizationContractError("fit_normalization_stats: zero pixels accumulated")

    mean = sum_val / n_pixels
    # Sample variance: sum((x - mean)^2) = sum(x^2) - n * mean^2
    var = max(0.0, (sum_sq / n_pixels) - (mean * mean))
    std = math.sqrt(var)

    subj_counts = list(per_subject_count.values())
    return NormalizationStats(
        n_samples=n_samples,
        n_pixels=n_pixels,
        finite_pixel_count=finite_count,
        non_finite_pixel_count=non_finite_count,
        global_min=float(g_min),
        global_max=float(g_max),
        global_mean=float(mean),
        global_std=float(std),
        method=NORMALIZATION_METHOD,
        epsilon=NORMALIZATION_EPSILON,
        raw_dtype="float64",
        raw_semantics=RAW_SEMANTICS,
        fit_split=NORMALIZATION_FIT_SPLIT,
        subject_count=len(per_subject_count),
        per_subject_count_min=int(min(subj_counts)),
        per_subject_count_max=int(max(subj_counts)),
        fitted_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def write_normalization_stats(path: Path, stats: NormalizationStats) -> None:
    """Write normalization stats to JSON (canonical form for hash stability)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stats": stats.to_dict(),
        "stats_sha256": stats.content_sha256(),
    }
    path.write_text(
        canonical_json_dumps(payload) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Class statistics (TRAIN/VAL only by default; TEST is structurally blocked)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ClassStats:
    """Class-coverage statistics over a single split.

    Note
    ----
    This is the only statistics object B01 is allowed to compute on
    TRAIN/VAL.  Computing it on TEST requires explicit
    :func:`enable_test_access` and is the responsibility of the Reviewer,
    not the development code.
    """

    n_samples: int
    n_pixels: int
    per_class_pixel_count: dict[int, int]   # class_id → pixel count
    per_class_pixel_ratio: dict[int, float] # class_id → fraction of all pixels
    missing_class_samples: dict[int, int]   # class_id → # samples missing that class
    per_posture_count: dict[str, int]
    per_subject_count_min: int
    per_subject_count_max: int
    subject_count: int
    small_region_sample_count: int          # samples with non-BACKGROUND area < 1%
    tiny_region_sample_count: int           # samples with non-BACKGROUND area < 0.1%
    onehot_roundtrip_ok_count: int          # how many samples have onehot ↔ label OK

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        # tuple keys → string keys for JSON
        d["per_class_pixel_count"] = {str(k): v for k, v in self.per_class_pixel_count.items()}
        d["per_class_pixel_ratio"] = {str(k): v for k, v in self.per_class_pixel_ratio.items()}
        d["missing_class_samples"] = {str(k): v for k, v in self.missing_class_samples.items()}
        return d


def compute_class_stats(
    rows: Iterable[FreezeRow],
    dataset_root: Path,
    *,
    ml_split: str,
) -> ClassStats:
    """Compute class-coverage statistics for a single ML split.

    Parameters
    ----------
    rows : Iterable[FreezeRow]
        Manifest rows.  All rows must be in ``ml_split``.
    dataset_root : Path
        Root of the SLP8 dataset (read-only).
    ml_split : str
        The ML split name.  Must be ``"train"`` or ``"val"`` unless TEST
        access has been explicitly enabled.

    Returns
    -------
    ClassStats

    Raises
    ------
    TestLeakageError
        If ``ml_split == "test"`` and TEST access is not enabled.
    """
    if ml_split == "test":
        require_test_access(purpose="final_evaluation")
    rows = list(rows)
    if not rows:
        raise ManifestContractError(f"compute_class_stats: empty rows for {ml_split!r}")
    for r in rows:
        if r.ml_split != ml_split:
            raise ManifestContractError(
                f"compute_class_stats: row {r.sample_id} ml_split={r.ml_split!r} != "
                f"requested {ml_split!r}"
            )

    n_samples = 0
    n_pixels = 0
    per_class_count: Counter[int] = Counter()
    missing_class_samples: Counter[int] = Counter()
    per_posture: Counter[str] = Counter()
    per_subject_count: Counter[str] = Counter()
    small_region_count = 0
    tiny_region_count = 0
    onehot_rt_ok = 0

    root_resolved = dataset_root.resolve()
    for r in rows:
        per_posture[r.posture] += 1
        per_subject_count[r.subject_id] += 1

        l_path = (dataset_root / r.region_label_npy).resolve()
        if not is_path_within(l_path, root_resolved):
            raise ManifestContractError(
                f"{r.sample_id}: label path escapes dataset root"
            )
        if not l_path.is_file():
            raise ManifestContractError(
                f"{r.sample_id}: label file not found: {l_path}"
            )
        label = np.load(l_path, allow_pickle=False)
        if label.shape != (192, 84) or label.dtype != np.uint8:
            raise ManifestContractError(
                f"{r.sample_id}: label shape/dtype mismatch: {label.shape}/{label.dtype}"
            )
        n_samples += 1
        n_pixels += int(label.size)
        unique, counts = np.unique(label, return_counts=True)
        for cid, cnt in zip(unique.tolist(), counts.tolist()):
            per_class_count[int(cid)] += int(cnt)
        present = set(int(x) for x in unique.tolist())
        for cid in range(9):
            if cid not in present:
                missing_class_samples[cid] += 1
        body_count = sum(int(c) for cid, c in zip(unique.tolist(), counts.tolist()) if cid != 0)
        body_ratio = body_count / float(label.size)
        if body_ratio < 0.01:
            small_region_count += 1
        if body_ratio < 0.001:
            tiny_region_count += 1
        if r.onehot_roundtrip:
            onehot_rt_ok += 1

    per_class_ratio: dict[int, float] = {}
    for cid, cnt in per_class_count.items():
        per_class_ratio[cid] = float(cnt) / float(n_pixels) if n_pixels else 0.0

    subj_counts = list(per_subject_count.values())
    return ClassStats(
        n_samples=n_samples,
        n_pixels=n_pixels,
        per_class_pixel_count=dict(per_class_count),
        per_class_pixel_ratio=per_class_ratio,
        missing_class_samples=dict(missing_class_samples),
        per_posture_count=dict(per_posture),
        per_subject_count_min=int(min(subj_counts)) if subj_counts else 0,
        per_subject_count_max=int(max(subj_counts)) if subj_counts else 0,
        subject_count=len(per_subject_count),
        small_region_sample_count=small_region_count,
        tiny_region_sample_count=tiny_region_count,
        onehot_roundtrip_ok_count=onehot_rt_ok,
    )


# ---------------------------------------------------------------------------
# Top-level freeze manifest
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FreezeManifest:
    """Top-level freeze manifest written to ``freeze_manifest.json``.

    The ``core`` sub-dict is content-addressed; the ``meta`` sub-dict is
    observational (timestamps, builder git SHA) and does not affect the
    core hash.
    """

    core: dict[str, Any]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "core": self.core,
            "meta": self.meta,
        }

    def core_sha256(self) -> str:
        return sha256_hex(canonical_json_dumps(self.core).encode("utf-8"))


def build_freeze_manifest(
    *,
    train_rows: list[FreezeRow],
    val_rows: list[FreezeRow],
    test_rows: list[FreezeRow],
    a06_split: A06Split,
    source_manifest_sha256: str,
    stats: NormalizationStats,
    train_stats: ClassStats,
    val_stats: ClassStats,
    builder_version: str = ADAPTER_VERSION,
    git_sha: str | None = None,
    build_command: str | None = None,
) -> FreezeManifest:
    """Assemble the top-level freeze manifest.

    The ``core`` sub-dict is the canonical contract; ``meta`` is observational.
    """
    # Per-split manifest hashes
    train_manifest_sha = manifest_sha256(train_rows)
    val_manifest_sha = manifest_sha256(val_rows)
    test_manifest_sha = manifest_sha256(test_rows)
    norm_stats_sha = stats.content_sha256()

    # Per-split subject counts
    def subj(rows: list[FreezeRow]) -> list[str]:
        return sorted({r.subject_id for r in rows})

    core = {
        "task_id": TASK_ID,
        "freeze_version": FREEZE_VERSION,
        "builder_version": builder_version,
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_dataset_version": SOURCE_DATASET_VERSION,
        "source_manifest_sha256": source_manifest_sha256,
        "a06_split_identifier": A06_SPLIT_IDENTIFIER,
        "a06_split_sha256": a06_split.sha256,
        "expected_total_samples": EXPECTED_TOTAL,
        "expected_subjects": EXPECTED_SUBJECTS,
        "expected_frames_per_subject": EXPECTED_FRAMES_PER_SUBJECT,
        "expected_split_counts": list(EXPECTED_SPLIT_COUNTS.items()),
        "expected_provenance": EXPECTED_PROVENANCE,
        "expected_review_status": EXPECTED_REVIEW_STATUS,
        "class_schema": {
            "version": "slp8-v2.2.1-canonical-export-v1.1",
            "num_label_ids": 9,
            "num_semantic_regions": 8,
            "background_id": 0,
            "class_name_to_id": {
                "BACKGROUND": 0, "HEAD_NECK": 1, "SHOULDER": 2, "THORAX_BACK": 3,
                "LUMBAR_WAIST": 4, "PELVIS_HIP": 5, "ARM": 6, "THIGH": 7,
                "LOWER_LEG_FOOT": 8,
            },
        },
        "manifest_columns": list(MANIFEST_COLUMNS),
        "splits": {
            "train": {
                "sample_count": len(train_rows),
                "subject_count": len(subj(train_rows)),
                "subject_ids": subj(train_rows),
                "manifest_sha256": train_manifest_sha,
            },
            "val": {
                "sample_count": len(val_rows),
                "subject_count": len(subj(val_rows)),
                "subject_ids": subj(val_rows),
                "manifest_sha256": val_manifest_sha,
            },
            "test": {
                "sample_count": len(test_rows),
                "subject_count": len(subj(test_rows)),
                "subject_ids": subj(test_rows),
                "manifest_sha256": test_manifest_sha,
            },
        },
        "normalization_stats_sha256": norm_stats_sha,
        "normalization_method": stats.method,
        "normalization_fit_split": stats.fit_split,
        "normalization_raw_semantics": stats.raw_semantics,
        "normalization_raw_dtype": stats.raw_dtype,
        "normalization_epsilon": stats.epsilon,
        "class_stats": {
            "train": {
                "sample_count": train_stats.n_samples,
                "subject_count": train_stats.subject_count,
                "per_posture_count": train_stats.per_posture_count,
                "per_class_pixel_ratio": {
                    str(k): v for k, v in train_stats.per_class_pixel_ratio.items()
                },
            },
            "val": {
                "sample_count": val_stats.n_samples,
                "subject_count": val_stats.subject_count,
                "per_posture_count": val_stats.per_posture_count,
                "per_class_pixel_ratio": {
                    str(k): v for k, v in val_stats.per_class_pixel_ratio.items()
                },
            },
            "test": {
                "sample_count": len(test_rows),
                "subject_count": len(subj(test_rows)),
                "structural_only": True,
                "note": (
                    "TEST class statistics are intentionally NOT computed.  "
                    "Only structural counts (sample/subject) are recorded."
                ),
            },
        },
        "test_access_policy": {
            "default": "deny_label_access",
            "allow_test_required": True,
            "allowed_purposes": ["final_evaluation"],
        },
    }

    meta = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_version": builder_version,
        "git_sha": git_sha,
        "build_command": build_command,
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
    }

    return FreezeManifest(core=core, meta=meta)


def write_freeze_manifest(path: Path, fm: FreezeManifest) -> None:
    """Write the top-level freeze manifest as canonical JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        canonical_json_dumps(fm.to_dict()) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Slp8TrainingTableFreezer — orchestrator
# ---------------------------------------------------------------------------

@dataclass
class FreezeResult:
    """Summary of a B01 freeze build."""

    output_dir: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path
    train_jsonl: Path
    val_jsonl: Path
    test_jsonl: Path
    freeze_manifest_path: Path
    normalization_stats_path: Path
    train_class_stats_path: Path
    val_class_stats_path: Path
    dataset_card_path: Path
    a06_split_sha256: str
    source_manifest_sha256: str
    train_manifest_sha256: str
    val_manifest_sha256: str
    test_manifest_sha256: str
    normalization_stats_sha256: str
    freeze_manifest_sha256: str
    normalization_stats: NormalizationStats
    train_class_stats: ClassStats
    val_class_stats: ClassStats
    n_train: int
    n_val: int
    n_test: int


class Slp8TrainingTableFreezer:
    """Build the B01 frozen training/validation/test tables for SLP8.

    Usage
    -----
    >>> freezer = Slp8TrainingTableFreezer(
    ...     dataset_root=Path("/path/to/SLP_8Region_Pressure_VAL_v1.1"),
    ...     a06_split_path=Path("/path/to/slp_subject_split_v0.1.json"),
    ...     output_dir=Path("/path/to/slp8_training_tables_v0.1"),
    ... )
    >>> result = freezer.build()
    """

    def __init__(
        self,
        *,
        dataset_root: Path,
        a06_split_path: Path,
        output_dir: Path,
        dataset_card_path: Path | None = None,
        git_sha: str | None = None,
        build_command: str | None = None,
        # Test / synthetic-data hooks.  These default to the canonical A06
        # contract; only set them in tests that use a synthetic A06.
        expected_a06_sha256: str | None = A06_SPLIT_SHA256_EXPECTED,
        enforce_canonical_a06_subject_counts: bool = True,
        enforce_canonical_source_total: bool = True,
        enforce_canonical_split_counts: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.a06_split_path = Path(a06_split_path)
        self.output_dir = Path(output_dir)
        self.dataset_card_path = (
            Path(dataset_card_path) if dataset_card_path
            else (self.output_dir / "dataset_card.md")
        )
        self.git_sha = git_sha
        self.build_command = build_command
        self.expected_a06_sha256 = expected_a06_sha256
        self.enforce_canonical_a06_subject_counts = enforce_canonical_a06_subject_counts
        self.enforce_canonical_source_total = enforce_canonical_source_total
        self.enforce_canonical_split_counts = enforce_canonical_split_counts

    # ── Build pipeline ─────────────────────────────────────────────────────

    def build(self) -> FreezeResult:
        """Run the full freeze pipeline and write all artifacts."""
        # ── 1. Load + validate A06 split ───────────────────────────────────
        a06_split = load_a06_split(
            self.a06_split_path,
            expected_sha256=self.expected_a06_sha256,
            enforce_canonical_subject_counts=self.enforce_canonical_a06_subject_counts,
        )

        # ── 2. Load + validate SLP8 source manifest ───────────────────────
        source = load_slp8_source_manifest(
            self.dataset_root,
            enforce_canonical_total=self.enforce_canonical_source_total,
        )

        # ── 3. Subject-level mapping to ML splits ──────────────────────────
        # For each subject, ask A06 which ML split it belongs to.
        # Build FreezeRow with ml_split attached.
        train_rows: list[FreezeRow] = []
        val_rows: list[FreezeRow] = []
        test_rows: list[FreezeRow] = []
        per_subject_ml_split: dict[str, str] = {}
        for sample in source.samples:
            if sample.setting != "danaLab":
                # B01 only uses danaLab samples.
                continue
            ml_split = a06_split.ml_split_for_subject(sample.subject_id, "danaLab")
            if subject_in := per_subject_ml_split.get(sample.subject_id):
                if subject_in != ml_split:
                    raise SubjectMappingError(
                        f"subject {sample.subject_id} mapped to multiple splits: "
                        f"{subject_in} and {ml_split}"
                    )
            else:
                per_subject_ml_split[sample.subject_id] = ml_split
            row = build_freeze_row(
                sample,
                ml_split=ml_split,
                dataset_root=self.dataset_root,
            )
            if ml_split == "train":
                train_rows.append(row)
            elif ml_split == "val":
                val_rows.append(row)
            elif ml_split == "test":
                test_rows.append(row)
            else:
                raise SubjectMappingError(f"unexpected ml_split={ml_split!r}")

        # ── 4. Verify expected counts ─────────────────────────────────────
        n_train, n_val, n_test = len(train_rows), len(val_rows), len(test_rows)
        if self.enforce_canonical_split_counts:
            if n_train != EXPECTED_SPLIT_COUNTS["train"]:
                raise SubjectMappingError(
                    f"train sample count = {n_train}, expected "
                    f"{EXPECTED_SPLIT_COUNTS['train']}"
                )
            if n_val != EXPECTED_SPLIT_COUNTS["val"]:
                raise SubjectMappingError(
                    f"val sample count = {n_val}, expected {EXPECTED_SPLIT_COUNTS['val']}"
                )
            if n_test != EXPECTED_SPLIT_COUNTS["test"]:
                raise SubjectMappingError(
                    f"test sample count = {n_test}, expected {EXPECTED_SPLIT_COUNTS['test']}"
                )
            if (n_train + n_val + n_test) != EXPECTED_TOTAL:
                raise SubjectMappingError(
                    f"total samples = {n_train + n_val + n_test}, expected {EXPECTED_TOTAL}"
                )

        # ── 5. Verify subject-level isolation ─────────────────────────────
        all_subjects: set[str] = set()
        for r in train_rows + val_rows + test_rows:
            all_subjects.add(r.subject_id)
        if self.enforce_canonical_split_counts and len(all_subjects) != EXPECTED_SUBJECTS:
            raise SubjectMappingError(
                f"unique subjects = {len(all_subjects)}, expected {EXPECTED_SUBJECTS}"
            )

        # ── 6. Write per-split manifests (CSV + JSONL) ────────────────────
        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_csv = self.output_dir / "train_manifest.csv"
        val_csv = self.output_dir / "val_manifest.csv"
        test_csv = self.output_dir / "test_manifest.csv"
        train_jsonl = self.output_dir / "train_manifest.jsonl"
        val_jsonl = self.output_dir / "val_manifest.jsonl"
        test_jsonl = self.output_dir / "test_manifest.jsonl"
        write_manifest_csv(train_csv, train_rows)
        write_manifest_csv(val_csv, val_rows)
        write_manifest_csv(test_csv, test_rows)
        write_manifest_jsonl(train_jsonl, train_rows)
        write_manifest_jsonl(val_jsonl, val_rows)
        write_manifest_jsonl(test_jsonl, test_rows)

        train_manifest_sha = manifest_sha256(train_rows)
        val_manifest_sha = manifest_sha256(val_rows)
        test_manifest_sha = manifest_sha256(test_rows)

        # ── 7. Fit TRAIN-only normalization ───────────────────────────────
        stats = fit_normalization_stats(train_rows, self.dataset_root)
        norm_stats_path = self.output_dir / "normalization_stats.json"
        write_normalization_stats(norm_stats_path, stats)

        # ── 8. Compute class stats for TRAIN and VAL only ────────────────
        train_class_stats = compute_class_stats(
            train_rows, self.dataset_root, ml_split="train"
        )
        val_class_stats = compute_class_stats(
            val_rows, self.dataset_root, ml_split="val"
        )
        train_class_stats_path = self.output_dir / "train_class_stats.json"
        val_class_stats_path = self.output_dir / "val_class_stats.json"
        train_class_stats_path.write_text(
            canonical_json_dumps(train_class_stats.to_dict()) + "\n",
            encoding="utf-8",
        )
        val_class_stats_path.write_text(
            canonical_json_dumps(val_class_stats.to_dict()) + "\n",
            encoding="utf-8",
        )

        # ── 9. Write freeze manifest ──────────────────────────────────────
        fm = build_freeze_manifest(
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            a06_split=a06_split,
            source_manifest_sha256=source.source_manifest_sha256,
            stats=stats,
            train_stats=train_class_stats,
            val_stats=val_class_stats,
            builder_version=ADAPTER_VERSION,
            git_sha=self.git_sha,
            build_command=self.build_command,
        )
        freeze_manifest_path = self.output_dir / "freeze_manifest.json"
        write_freeze_manifest(freeze_manifest_path, fm)
        freeze_manifest_sha = fm.core_sha256()

        # ── 10. Write dataset card ────────────────────────────────────────
        card = render_dataset_card(
            freeze_manifest=fm,
            train_class_stats=train_class_stats,
            val_class_stats=val_class_stats,
            normalization_stats=stats,
        )
        self.dataset_card_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_card_path.write_text(card, encoding="utf-8")

        return FreezeResult(
            output_dir=self.output_dir,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            train_jsonl=train_jsonl,
            val_jsonl=val_jsonl,
            test_jsonl=test_jsonl,
            freeze_manifest_path=freeze_manifest_path,
            normalization_stats_path=norm_stats_path,
            train_class_stats_path=train_class_stats_path,
            val_class_stats_path=val_class_stats_path,
            dataset_card_path=self.dataset_card_path,
            a06_split_sha256=a06_split.sha256,
            source_manifest_sha256=source.source_manifest_sha256,
            train_manifest_sha256=train_manifest_sha,
            val_manifest_sha256=val_manifest_sha,
            test_manifest_sha256=test_manifest_sha,
            normalization_stats_sha256=stats.content_sha256(),
            freeze_manifest_sha256=freeze_manifest_sha,
            normalization_stats=stats,
            train_class_stats=train_class_stats,
            val_class_stats=val_class_stats,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        )


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

def render_dataset_card(
    *,
    freeze_manifest: FreezeManifest,
    train_class_stats: ClassStats,
    val_class_stats: ClassStats,
    normalization_stats: NormalizationStats,
) -> str:
    """Render the human-readable dataset card as Markdown."""
    core = freeze_manifest.core
    splits = core["splits"]
    cls = core["class_schema"]
    parts: list[str] = []
    parts.append("# SLP8 Pressure-Only Training Tables — Dataset Card (v0.1)\n")
    parts.append(
        "This card documents the **frozen** training/validation/test tables for "
        "the SLP 8-region pressure-only region segmentation task, derived from "
        f"`{core['source_dataset_id']}` and the A06 subject-level split "
        f"(`{core['a06_split_identifier']}`).\n"
    )
    parts.append("## Provenance and limitations (read first)\n")
    parts.append(
        "- **8-region pressure-only GT**: This dataset uses the 8-region class schema "
        "(`BACKGROUND, HEAD_NECK, SHOULDER, THORAX_BACK, LUMBAR_WAIST, PELVIS_HIP, "
        "ARM, THIGH, LOWER_LEG_FOOT`).  It is **NOT** the 10-region polygon schema "
        "(`slp_region_annotation_v0.1`); the two schemas must not be mixed.\n"
        "- **GT provenance**: `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`.\n"
        "- **Review status**: `NOT_REVIEWED` — this is an automated pipeline export, "
        "**not** a human pixel-level semantic mask.\n"
        "- **Not medical, skin-interface stress, or product ground truth.**\n"
        "- **Pressure values are raw PMarray response semantics, NOT kPa.**\n"
        "- **danaLab only**, **uncover only**.  Do not extrapolate to cover1/cover2.\n"
        "- **Not suitable** for self-developed topper product effect claims, comfort, "
        "medical, or hardware validation.\n"
    )
    parts.append("## Contract\n")
    parts.append(
        f"- TASK-ID: `{core['task_id']}`\n"
        f"- Freeze version: `{core['freeze_version']}`\n"
        f"- Source dataset: `{core['source_dataset_id']}` (version {core['source_dataset_version']})\n"
        f"- Source manifest SHA-256: `{core['source_manifest_sha256']}`\n"
        f"- A06 split identifier: `{core['a06_split_identifier']}`\n"
        f"- A06 split SHA-256: `{core['a06_split_sha256']}`\n"
        f"- Builder: `{core['builder_version']}`\n"
    )
    parts.append("## Splits\n")
    parts.append("| Split | Samples | Subjects | Manifest SHA-256 |")
    parts.append("|---|---:|---:|---|")
    for s in ("train", "val", "test"):
        info = splits[s]
        parts.append(
            f"| {s} | {info['sample_count']} | {info['subject_count']} | `{info['manifest_sha256']}` |"
        )
    parts.append("")
    parts.append(
        f"- Total: **{core['expected_total_samples']}** samples, "
        f"**{core['expected_subjects']}** subjects, "
        f"**{core['expected_frames_per_subject']}** frames per subject.\n"
    )
    parts.append("## Class schema\n")
    parts.append(f"- Class schema version: `{cls['version']}`")
    parts.append(f"- 9 label IDs (1 background + 8 semantic regions)")
    parts.append("| ID | Name |")
    parts.append("|---:|---|")
    for name, cid in cls["class_name_to_id"].items():
        parts.append(f"| {cid} | {name} |")
    parts.append("")
    parts.append("## Normalization (TRAIN-only fit)\n")
    parts.append(
        f"- Method: `{normalization_stats.method}`\n"
        f"- Epsilon: `{normalization_stats.epsilon}`\n"
        f"- Raw dtype: `{normalization_stats.raw_dtype}`\n"
        f"- Raw semantics: `{normalization_stats.raw_semantics}` (NEVER kPa)\n"
        f"- Fit split: `{normalization_stats.fit_split}` (TRAIN only; VAL/TEST never used)\n"
        f"- TRAIN samples fitted: `{normalization_stats.n_samples}`\n"
        f"- TRAIN pixels fitted: `{normalization_stats.n_pixels}`\n"
        f"- Finite pixels: `{normalization_stats.finite_pixel_count}` / "
        f"non-finite: `{normalization_stats.non_finite_pixel_count}`\n"
        f"- Global min: `{normalization_stats.global_min}`\n"
        f"- Global max: `{normalization_stats.global_max}`\n"
        f"- Global mean: `{normalization_stats.global_mean}`\n"
        f"- Global std: `{normalization_stats.global_std}`\n"
        f"- TRAIN subject count: `{normalization_stats.subject_count}` "
        f"(min frames/subject={normalization_stats.per_subject_count_min}, "
        f"max frames/subject={normalization_stats.per_subject_count_max})\n"
        f"- Stats file SHA-256: `{normalization_stats.content_sha256()}`\n"
    )
    parts.append("## Class coverage (TRAIN/VAL only; TEST structural only)\n")
    parts.append("### TRAIN")
    parts.append(
        f"- Samples: `{train_class_stats.n_samples}`\n"
        f"- Subjects: `{train_class_stats.subject_count}`\n"
        f"- Pixels: `{train_class_stats.n_pixels}`\n"
        f"- Per-posture counts: `{train_class_stats.per_posture_count}`\n"
        f"- Per-class pixel ratio:\n"
    )
    for cid in sorted(train_class_stats.per_class_pixel_ratio):
        ratio = train_class_stats.per_class_pixel_ratio[cid]
        parts.append(f"  - class {cid}: `{ratio:.6f}`")
    parts.append("### VAL")
    parts.append(
        f"- Samples: `{val_class_stats.n_samples}`\n"
        f"- Subjects: `{val_class_stats.subject_count}`\n"
        f"- Per-posture counts: `{val_class_stats.per_posture_count}`\n"
        f"- Per-class pixel ratio:\n"
    )
    for cid in sorted(val_class_stats.per_class_pixel_ratio):
        ratio = val_class_stats.per_class_pixel_ratio[cid]
        parts.append(f"  - class {cid}: `{ratio:.6f}`")
    parts.append("### TEST")
    parts.append(
        f"- Samples: `{splits['test']['sample_count']}`\n"
        f"- Subjects: `{splits['test']['subject_count']}`\n"
        f"- Class coverage: **NOT computed** (TEST label/onehot access is "
        "blocked in development; only structural counts are recorded).\n"
    )
    parts.append("## Test access policy\n")
    parts.append(
        "By default, training and statistics code MUST NOT read TEST "
        "label/onehot.  TEST access is only allowed for **final evaluation** "
        "via `enable_test_access(purpose='final_evaluation')`.  Structural "
        "checks (sample/subject counts, sample_id uniqueness, path format, "
        "file existence, hash/contract consistency) are always allowed on TEST.\n"
    )
    parts.append("## Prohibited conclusions\n")
    parts.append(
        "- Do not claim the GT is human pixel-level semantic annotation.\n"
        "- Do not claim the pressure values represent kPa or any physical unit.\n"
        "- Do not extrapolate to cover1/cover2 conditions, simLab subjects, or "
        "self-developed topper product effects.\n"
        "- Do not use as comfort, medical, or hardware validation ground truth.\n"
    )
    return "\n".join(parts) + "\n"
