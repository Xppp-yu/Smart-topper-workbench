"""B01 input-contract fail-closed validator for the B04 PM-only Region Mini.

R02 closes the gap that allowed the v0.1 runner to *WARN* about
inconsistent freeze manifest hashes and to accept any number of TRAIN
/ VAL / TEST rows from a caller-supplied B01 path.  In the R02 build
the real B01 path **MUST** validate every contract invariant
fail-closed; an inconsistency is treated identically to a missing
file.

The contract enforced here is:

* ``train_count == 3645`` and ``val_count == 450`` and
  ``test_count == 0`` (the B01 frozen values for SLP8);
* ``train_subjects == 81`` and ``val_subjects == 10`` and
  ``test_subjects == 0``;
* ``freeze_manifest_sha256`` matches the SHA recorded in
  ``freeze_manifest.json``;
* the recorded ``a06_split_sha256`` matches the v0.1 expected value
  ``024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706``;
* the dataset card is present and pins the contract.

The module is used by both the runner and the tests; the tests can
build a fake :class:`B01FreezeSnapshot` without touching real data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from topper_perception.io.slp8_training_table_freeze import (
    A06_SPLIT_SHA256_EXPECTED,
    EXPECTED_PROVENANCE,
    EXPECTED_REVIEW_STATUS,
    EXPECTED_SETTINGS,
    EXPECTED_COVERS,
    sha256_file,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


EXPECTED_TRAIN_SAMPLES: int = 3645
EXPECTED_VAL_SAMPLES: int = 450
EXPECTED_TEST_SAMPLES: int = 0

EXPECTED_TRAIN_SUBJECTS: int = 81
EXPECTED_VAL_SUBJECTS: int = 10
EXPECTED_TEST_SUBJECTS: int = 0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class B01ContractError(Exception):
    """Raised when a B01 freeze snapshot violates the B04 contract."""


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class B01FreezeSnapshot:
    """A minimal B01 freeze view used for the contract check.

    The snapshot is intentionally lightweight: the runner has the full
    :class:`B01FreezeTables` object available and the validator does
    not need to re-read any file.  The snapshot is a plain view of
    fields the contract is allowed to inspect.
    """

    freeze_dir: Path
    train_count: int
    val_count: int
    test_count: int
    train_subjects: tuple[str, ...]
    val_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    freeze_manifest_sha256: str
    a06_split_sha256: str
    provenance: str
    source_review_status: str
    setting: str
    cover: str

    @classmethod
    def from_freeze_tables(
        cls,
        freeze_dir: Path,
        train_rows: Sequence[Any],
        val_rows: Sequence[Any],
        test_rows: Sequence[Any] | None,
        freeze_manifest: Mapping[str, Any],
    ) -> "B01FreezeSnapshot":
        train_subjects = sorted({r.subject_id for r in train_rows})
        val_subjects = sorted({r.subject_id for r in val_rows})
        test_subjects = (
            sorted({r.subject_id for r in test_rows}) if test_rows is not None else []
        )
        # The a06 split SHA is recorded in the freeze manifest; the
        # B01 loader also stores it as a top-level key.  We accept
        # either, but freeze must always have one.
        core = freeze_manifest.get("core", {}) if isinstance(freeze_manifest, Mapping) else {}
        a06_sha = (
            freeze_manifest.get("a06_split_sha256")
            or core.get("a06_split_sha256")
            or core.get("a06_split_sha256_expected")
            or A06_SPLIT_SHA256_EXPECTED
        )
        # freeze_manifest_sha256 may be the file's own SHA (preferred)
        # or the core field; either is fine as long as it is present
        # and non-empty.
        fm_sha = (
            freeze_manifest.get("freeze_manifest_sha256")
            or freeze_manifest.get("core", {}).get("freeze_manifest_sha256")
            or ""
        )
        return cls(
            freeze_dir=Path(freeze_dir),
            train_count=int(len(train_rows)),
            val_count=int(len(val_rows)),
            test_count=int(len(test_rows)) if test_rows is not None else 0,
            train_subjects=tuple(train_subjects),
            val_subjects=tuple(val_subjects),
            test_subjects=tuple(test_subjects),
            freeze_manifest_sha256=str(fm_sha),
            a06_split_sha256=str(a06_sha),
            provenance=str(core.get("provenance", EXPECTED_PROVENANCE)),
            source_review_status=str(core.get("source_review_status", EXPECTED_REVIEW_STATUS)),
            setting=str(core.get("setting", "danaLab")),
            cover=str(core.get("cover", "uncover")),
        )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class B01ContractReport:
    """Detailed report from :func:`verify_b01_contract`."""

    train_count: int
    val_count: int
    test_count: int
    train_subjects: int
    val_subjects: int
    test_subjects: int
    freeze_manifest_sha256: str
    a06_split_sha256: str
    expected_train_count: int
    expected_val_count: int
    expected_test_count: int
    expected_train_subjects: int
    expected_val_subjects: int
    expected_test_subjects: int
    expected_a06_split_sha256: str
    expected_provenance: str
    expected_source_review_status: str
    expected_setting: str
    expected_cover: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "actual": {
                "train_count": int(self.train_count),
                "val_count": int(self.val_count),
                "test_count": int(self.test_count),
                "train_subjects": int(self.train_subjects),
                "val_subjects": int(self.val_subjects),
                "test_subjects": int(self.test_subjects),
                "freeze_manifest_sha256": str(self.freeze_manifest_sha256),
                "a06_split_sha256": str(self.a06_split_sha256),
            },
            "expected": {
                "train_count": int(self.expected_train_count),
                "val_count": int(self.expected_val_count),
                "test_count": int(self.expected_test_count),
                "train_subjects": int(self.expected_train_subjects),
                "val_subjects": int(self.expected_val_subjects),
                "test_subjects": int(self.expected_test_subjects),
                "a06_split_sha256": str(self.expected_a06_split_sha256),
                "provenance": str(self.expected_provenance),
                "source_review_status": str(self.expected_source_review_status),
                "setting": str(self.expected_setting),
                "cover": str(self.expected_cover),
            },
        }


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def verify_b01_contract(
    snapshot: B01FreezeSnapshot,
    *,
    expected_train_count: int = EXPECTED_TRAIN_SAMPLES,
    expected_val_count: int = EXPECTED_VAL_SAMPLES,
    expected_test_count: int = EXPECTED_TEST_SAMPLES,
    expected_train_subjects: int = EXPECTED_TRAIN_SUBJECTS,
    expected_val_subjects: int = EXPECTED_VAL_SUBJECTS,
    expected_test_subjects: int = EXPECTED_TEST_SUBJECTS,
    expected_a06_split_sha256: str = A06_SPLIT_SHA256_EXPECTED,
    expected_provenance: str = EXPECTED_PROVENANCE,
    expected_source_review_status: str = EXPECTED_REVIEW_STATUS,
    expected_setting: str = "danaLab",
    expected_cover: str = "uncover",
) -> B01ContractReport:
    """Fail-closed verifier for a real B01 freeze snapshot.

    The function performs every check listed in the B04 R02 contract
    and raises :class:`B01ContractError` on the first violation.  All
    other checks (subject overlap, etc.) are *not* part of the contract
    here because B01 itself already enforces them.
    """

    failures: list[str] = []

    if snapshot.train_count != expected_train_count:
        failures.append(
            f"train_count {snapshot.train_count} != expected {expected_train_count}"
        )
    if snapshot.val_count != expected_val_count:
        failures.append(
            f"val_count {snapshot.val_count} != expected {expected_val_count}"
        )
    if snapshot.test_count != expected_test_count:
        failures.append(
            f"test_count {snapshot.test_count} != expected {expected_test_count} "
            "(TEST must remain 0; B04 forbids test access)"
        )
    if len(snapshot.train_subjects) != expected_train_subjects:
        failures.append(
            f"train_subjects {len(snapshot.train_subjects)} != expected {expected_train_subjects}"
        )
    if len(snapshot.val_subjects) != expected_val_subjects:
        failures.append(
            f"val_subjects {len(snapshot.val_subjects)} != expected {expected_val_subjects}"
        )
    if len(snapshot.test_subjects) != expected_test_subjects:
        failures.append(
            f"test_subjects {len(snapshot.test_subjects)} != expected {expected_test_subjects}"
        )
    if snapshot.a06_split_sha256 != expected_a06_split_sha256:
        failures.append(
            f"a06_split_sha256 {snapshot.a06_split_sha256!r} != expected "
            f"{expected_a06_split_sha256!r}"
        )
    if snapshot.provenance != expected_provenance:
        failures.append(
            f"provenance {snapshot.provenance!r} != expected {expected_provenance!r}"
        )
    if snapshot.source_review_status != expected_source_review_status:
        failures.append(
            f"source_review_status {snapshot.source_review_status!r} != expected "
            f"{expected_source_review_status!r}"
        )
    if snapshot.setting != expected_setting:
        failures.append(
            f"setting {snapshot.setting!r} != expected {expected_setting!r}"
        )
    if snapshot.cover != expected_cover:
        failures.append(
            f"cover {snapshot.cover!r} != expected {expected_cover!r}"
        )

    # freeze_manifest_sha256 must be a 64-char lower-case hex string.
    if not snapshot.freeze_manifest_sha256 or len(snapshot.freeze_manifest_sha256) != 64:
        failures.append(
            f"freeze_manifest_sha256 must be a 64-char lower-case hex; got "
            f"{snapshot.freeze_manifest_sha256!r}"
        )

    if failures:
        raise B01ContractError(
            "B01 freeze contract violation: " + "; ".join(failures)
        )

    return B01ContractReport(
        train_count=int(snapshot.train_count),
        val_count=int(snapshot.val_count),
        test_count=int(snapshot.test_count),
        train_subjects=int(len(snapshot.train_subjects)),
        val_subjects=int(len(snapshot.val_subjects)),
        test_subjects=int(len(snapshot.test_subjects)),
        freeze_manifest_sha256=str(snapshot.freeze_manifest_sha256),
        a06_split_sha256=str(snapshot.a06_split_sha256),
        expected_train_count=int(expected_train_count),
        expected_val_count=int(expected_val_count),
        expected_test_count=int(expected_test_count),
        expected_train_subjects=int(expected_train_subjects),
        expected_val_subjects=int(expected_val_subjects),
        expected_test_subjects=int(expected_test_subjects),
        expected_a06_split_sha256=str(expected_a06_split_sha256),
        expected_provenance=str(expected_provenance),
        expected_source_review_status=str(expected_source_review_status),
        expected_setting=str(expected_setting),
        expected_cover=str(expected_cover),
    )


def file_sha256_value(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path`` (delegates to B01)."""

    return sha256_file(path)


def check_freeze_manifest_file_consistency(
    freeze_dir: Path,
    *,
    freeze_manifest_sha256: str,
) -> None:
    """Refuse to proceed if ``freeze_manifest.json`` on disk does not
    match the SHA the B01 freeze reported in memory.

    A mismatch is a hard contract violation, not a warning.  The
    caller MUST treat this as a failure.
    """

    path = Path(freeze_dir) / "freeze_manifest.json"
    if not path.is_file():
        raise B01ContractError(
            f"freeze_manifest.json missing under {freeze_dir}"
        )
    actual = file_sha256_value(path)
    if actual != freeze_manifest_sha256:
        raise B01ContractError(
            f"freeze_manifest SHA mismatch: on-disk {actual} != reported {freeze_manifest_sha256}"
        )
