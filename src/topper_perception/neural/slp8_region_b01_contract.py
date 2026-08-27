"""B01 input-contract fail-closed validator for the B04 PM-only Region Mini.

R04 closes the remaining fallbacks the R02/R03 module still carried:

* Every required field on the freeze manifest ``core`` MUST be present.
  Missing ``core``, ``a06_split_sha256``, ``expected_provenance``,
  ``expected_review_status``, ``splits`` (or any of train / val /
  test under it), ``sample_count`` / ``subject_count`` /
  ``manifest_sha256`` per split — all fail-closed with
  :class:`B01ContractError`.  No default values are substituted.
* ``setting`` and ``cover`` are NOT read from the manifest.  They
  are computed as the set of unique values present in the loaded
  TRAIN/VAL rows.  The contract requires the observed set to be the
  expected singleton.
* The freeze-manifest ``core`` SHA is **computed** from the core
  sub-dict via the B01 canonical-JSON + SHA-256 algorithm.  The
  computed SHA is compared against the **frozen expected** value
  in the B04 config; it is **not** compared against the same file
  (no self-comparison).
* TRAIN/VAL manifest SHAs (computed from the actual rows) are
  compared against the ``manifest_sha256`` recorded under
  ``core.splits.{train,val}``.  This catches the case where someone
  builds a freeze with mismatched core metadata.
* The structural TEST split is recorded as ``sample_count`` /
  ``subject_count`` ONLY — taken from ``core.splits.test`` directly.
  The B04 runner MUST NOT load TEST row objects; the snapshot's
  ``structural_test_*`` fields are the sole window into TEST counts.
* The contract check is invoked from :func:`_run_real_b01` BEFORE
  the CUDA availability check, so an Operator / Reviewer can audit
  any contract failure on a CPU machine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from topper_perception.io.slp8_training_table_freeze import (
    canonical_json_dumps,
    manifest_sha256,
    sha256_hex,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class B01ContractError(Exception):
    """Raised when a B01 freeze snapshot violates the B04 contract.

    The validator NEVER returns a degraded report; every violation is
    a hard ``B01ContractError`` so a Reviewer can grep the logs for
    one consistent error class.
    """


# ---------------------------------------------------------------------------
# Expected contract (built from the B04 frozen config)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class B01ContractExpected:
    """The frozen B04-contract expected values for the real B01 freeze.

    All fields are mandatory; the contract validator has no default
    fallbacks.  The values are produced from the B04 config which
    itself is frozen, so two B04 runs on the same config see the same
    expected values.
    """

    train_count: int
    val_count: int
    test_count: int            # always 0 — B04 never loads TEST rows
    train_subjects: int
    val_subjects: int
    test_subjects: int        # always 0 — see structural_test_*
    a06_split_sha256: str
    provenance: str
    source_review_status: str
    setting: str              # singleton, always "danaLab"
    cover: str                # singleton, always "uncover"
    freeze_manifest_core_sha256: str
    structural_test_samples: int
    structural_test_subjects: int


# ---------------------------------------------------------------------------
# Structural TEST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class B01FreezeStructuralTest:
    """The TEST split as it appears structurally in the freeze manifest.

    The B04 contract NEVER reads TEST row objects or TEST labels; the
    only authorised window into TEST is the structural counts the B01
    freeze builder wrote into ``core.splits.test``.
    """

    sample_count: int
    subject_count: int
    manifest_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_count": int(self.sample_count),
            "subject_count": int(self.subject_count),
            "manifest_sha256": str(self.manifest_sha256),
        }


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class B01FreezeSnapshot:
    """A real B01 freeze view used for the contract check.

    The snapshot separates three sources of truth:

    1. **Observed from actual rows** — counts, subjects, the unique
       set of ``setting`` / ``cover`` / ``annotation_provenance`` /
       ``source_review_status`` values, and the canonical TRAIN/VAL
       manifest SHAs.
    2. **Structural TEST** — read from ``core.splits.test`` directly.
       No TEST row objects are read.
    3. **Manifest-derived** — ``a06_split_sha256`` from the core and
       the core SHA computed independently from the core sub-dict.
    """

    freeze_dir: Path

    # Observed from actual rows (TRAIN + VAL only).
    train_count: int
    val_count: int
    train_subjects: tuple[str, ...]
    val_subjects: tuple[str, ...]
    train_manifest_sha256: str
    val_manifest_sha256: str
    observed_settings: tuple[str, ...]
    observed_covers: tuple[str, ...]
    observed_provenances: tuple[str, ...]
    observed_review_statuses: tuple[str, ...]

    # Structural TEST (from core.splits.test).
    structural_test: B01FreezeStructuralTest

    # Manifest-derived.
    a06_split_sha256: str
    freeze_manifest_core_sha256: str
    # The TRAIN/VAL manifest SHAs as recorded in core.splits.{train,val}.
    # Used to compare against the SHAs computed from the actual rows.
    core_train_manifest_sha256: str
    core_val_manifest_sha256: str

    @classmethod
    def from_freeze_tables(
        cls,
        freeze_dir: Path,
        train_rows: Sequence[Any],
        val_rows: Sequence[Any],
        test_rows: Sequence[Any] | None,
        freeze_manifest: Mapping[str, Any],
    ) -> "B01FreezeSnapshot":
        """Build a snapshot from the B01 freeze tables.

        The freeze loader must pass ``test_rows=None`` (the B04 contract
        forbids loading TEST rows); this constructor refuses any other
        value.  All required ``core`` sub-dict fields are read DIRECTLY
        with no default fallback.
        """
        if test_rows is not None:
            raise B01ContractError(
                "B04 contract forbids loading TEST row objects; the freeze "
                "loader must call load_b01_freeze_tables(..., load_test=False) "
                "and pass test_rows=None.  Received test_rows of length "
                f"{len(test_rows)}."
            )
        if not isinstance(freeze_manifest, Mapping):
            raise B01ContractError(
                f"freeze_manifest must be a mapping; got {type(freeze_manifest).__name__}"
            )
        if "core" not in freeze_manifest:
            raise B01ContractError("freeze_manifest missing required 'core' key")
        core = freeze_manifest["core"]
        if not isinstance(core, Mapping):
            raise B01ContractError("freeze_manifest.core must be a mapping")

        # Required core sub-dict fields — fail-closed on every absence.
        for key in (
            "a06_split_sha256",
            "expected_provenance",
            "expected_review_status",
            "splits",
        ):
            if key not in core:
                raise B01ContractError(
                    f"freeze_manifest.core missing required field {key!r}"
                )
        splits = core["splits"]
        if not isinstance(splits, Mapping):
            raise B01ContractError("freeze_manifest.core.splits must be a mapping")
        for split_name in ("train", "val", "test"):
            if split_name not in splits:
                raise B01ContractError(
                    f"freeze_manifest.core.splits missing required key {split_name!r}"
                )
            split_block = splits[split_name]
            if not isinstance(split_block, Mapping):
                raise B01ContractError(
                    f"freeze_manifest.core.splits.{split_name} must be a mapping"
                )
            for key in ("sample_count", "subject_count", "manifest_sha256"):
                if key not in split_block:
                    raise B01ContractError(
                        f"freeze_manifest.core.splits.{split_name} missing "
                        f"required field {key!r}"
                    )

        # Manifest core SHA — computed INDEPENDENTLY from the core
        # sub-dict via the B01 canonical-JSON + SHA-256 algorithm.  We
        # compare it against the frozen expected value in the B04
        # config; we MUST NOT compare it against itself.
        core_sha = sha256_hex(canonical_json_dumps(core).encode("utf-8"))

        # Structural TEST counts — read from core.splits.test directly.
        # The B04 contract NEVER reads TEST row objects.
        structural_test = B01FreezeStructuralTest(
            sample_count=int(splits["test"]["sample_count"]),
            subject_count=int(splits["test"]["subject_count"]),
            manifest_sha256=str(splits["test"]["manifest_sha256"]),
        )

        # Observed-from-rows fields.
        train_subjects = sorted({r.subject_id for r in train_rows})
        val_subjects = sorted({r.subject_id for r in val_rows})

        def _unique(seq: Iterable[str]) -> tuple[str, ...]:
            return tuple(sorted(set(seq)))

        # Collect unique values across TRAIN + VAL rows.
        observed_settings = _unique(
            [r.setting for r in train_rows] + [r.setting for r in val_rows]
        )
        observed_covers = _unique(
            [r.cover for r in train_rows] + [r.cover for r in val_rows]
        )
        observed_provenances = _unique(
            [r.annotation_provenance for r in train_rows]
            + [r.annotation_provenance for r in val_rows]
        )
        observed_review_statuses = _unique(
            [r.source_review_status for r in train_rows]
            + [r.source_review_status for r in val_rows]
        )

        return cls(
            freeze_dir=Path(freeze_dir),
            train_count=int(len(train_rows)),
            val_count=int(len(val_rows)),
            train_subjects=tuple(train_subjects),
            val_subjects=tuple(val_subjects),
            train_manifest_sha256=str(manifest_sha256(train_rows)),
            val_manifest_sha256=str(manifest_sha256(val_rows)),
            observed_settings=observed_settings,
            observed_covers=observed_covers,
            observed_provenances=observed_provenances,
            observed_review_statuses=observed_review_statuses,
            structural_test=structural_test,
            a06_split_sha256=str(core["a06_split_sha256"]),
            freeze_manifest_core_sha256=core_sha,
            core_train_manifest_sha256=str(splits["train"]["manifest_sha256"]),
            core_val_manifest_sha256=str(splits["val"]["manifest_sha256"]),
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class B01ContractReport:
    """Detailed report from a successful :func:`verify_b01_contract`."""

    actual: dict[str, Any]
    expected: dict[str, Any]
    structural_test: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "actual": dict(self.actual),
            "expected": dict(self.expected),
            "structural_test": dict(self.structural_test),
        }


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def verify_b01_contract(
    snapshot: B01FreezeSnapshot,
    expected: B01ContractExpected,
) -> B01ContractReport:
    """Fail-closed verifier for a real B01 freeze snapshot.

    Every mismatch is collected into a single :class:`B01ContractError`
    so a Reviewer can grep the logs for one consistent error class.
    The verifier NEVER returns a degraded report.
    """

    failures: list[str] = []

    # 1. Counts (loaded rows, NOT structural TEST)
    if snapshot.train_count != expected.train_count:
        failures.append(
            f"train_count {snapshot.train_count} != expected {expected.train_count}"
        )
    if snapshot.val_count != expected.val_count:
        failures.append(
            f"val_count {snapshot.val_count} != expected {expected.val_count}"
        )
    if len(snapshot.train_subjects) != expected.train_subjects:
        failures.append(
            f"train_subjects {len(snapshot.train_subjects)} != expected "
            f"{expected.train_subjects}"
        )
    if len(snapshot.val_subjects) != expected.val_subjects:
        failures.append(
            f"val_subjects {len(snapshot.val_subjects)} != expected "
            f"{expected.val_subjects}"
        )

    # 2. Structural TEST (read from core, NOT from row objects).
    if snapshot.structural_test.sample_count != expected.structural_test_samples:
        failures.append(
            f"structural test sample_count {snapshot.structural_test.sample_count} "
            f"!= expected {expected.structural_test_samples}"
        )
    if snapshot.structural_test.subject_count != expected.structural_test_subjects:
        failures.append(
            f"structural test subject_count {snapshot.structural_test.subject_count} "
            f"!= expected {expected.structural_test_subjects}"
        )

    # 3. A06 split SHA.
    if snapshot.a06_split_sha256 != expected.a06_split_sha256:
        failures.append(
            f"a06_split_sha256 {snapshot.a06_split_sha256!r} != expected "
            f"{expected.a06_split_sha256!r}"
        )

    # 4. setting / cover — observed from real rows; must be a
    # singleton exactly equal to the expected value.
    if tuple(snapshot.observed_settings) != (expected.setting,):
        failures.append(
            f"observed_settings from TRAIN+VAL rows {snapshot.observed_settings!r} "
            f"must be the singleton {expected.setting!r}"
        )
    if tuple(snapshot.observed_covers) != (expected.cover,):
        failures.append(
            f"observed_covers from TRAIN+VAL rows {snapshot.observed_covers!r} "
            f"must be the singleton {expected.cover!r}"
        )

    # 5. provenance / review_status — sampled from real rows.  All
    # rows must carry the same expected values; we fail-closed on
    # any divergence so a poisoned row would trip the gate.
    if tuple(snapshot.observed_provenances) != (expected.provenance,):
        failures.append(
            f"observed_provenances from rows {snapshot.observed_provenances!r} "
            f"must be the singleton {expected.provenance!r}"
        )
    if tuple(snapshot.observed_review_statuses) != (expected.source_review_status,):
        failures.append(
            f"observed_review_statuses from rows "
            f"{snapshot.observed_review_statuses!r} must be the singleton "
            f"{expected.source_review_status!r}"
        )

    # 6. Freeze-manifest core SHA — independently computed, compared
    # against the frozen expected value (NOT self-comparison).
    if snapshot.freeze_manifest_core_sha256 != expected.freeze_manifest_core_sha256:
        failures.append(
            f"freeze_manifest core sha256 {snapshot.freeze_manifest_core_sha256!r} "
            f"!= expected {expected.freeze_manifest_core_sha256!r}"
        )

    # 7. TRAIN/VAL manifest SHAs computed from actual rows MUST match
    # the SHAs recorded in core.splits.{train,val}.manifest_sha256.
    if snapshot.train_manifest_sha256 != snapshot.core_train_manifest_sha256:
        failures.append(
            f"train manifest sha256 computed from rows "
            f"{snapshot.train_manifest_sha256!r} != "
            f"core.splits.train.manifest_sha256 "
            f"{snapshot.core_train_manifest_sha256!r}"
        )
    if snapshot.val_manifest_sha256 != snapshot.core_val_manifest_sha256:
        failures.append(
            f"val manifest sha256 computed from rows "
            f"{snapshot.val_manifest_sha256!r} != "
            f"core.splits.val.manifest_sha256 "
            f"{snapshot.core_val_manifest_sha256!r}"
        )

    if failures:
        raise B01ContractError(
            "B01 freeze contract violation: " + "; ".join(failures)
        )

    return B01ContractReport(
        actual={
            "train_count": int(snapshot.train_count),
            "val_count": int(snapshot.val_count),
            "train_subjects": int(len(snapshot.train_subjects)),
            "val_subjects": int(len(snapshot.val_subjects)),
            "a06_split_sha256": str(snapshot.a06_split_sha256),
            "freeze_manifest_core_sha256": str(snapshot.freeze_manifest_core_sha256),
            "train_manifest_sha256": str(snapshot.train_manifest_sha256),
            "val_manifest_sha256": str(snapshot.val_manifest_sha256),
            "observed_settings": list(snapshot.observed_settings),
            "observed_covers": list(snapshot.observed_covers),
            "observed_provenances": list(snapshot.observed_provenances),
            "observed_review_statuses": list(snapshot.observed_review_statuses),
        },
        expected={
            "train_count": int(expected.train_count),
            "val_count": int(expected.val_count),
            "train_subjects": int(expected.train_subjects),
            "val_subjects": int(expected.val_subjects),
            "test_count": int(expected.test_count),
            "test_subjects": int(expected.test_subjects),
            "a06_split_sha256": str(expected.a06_split_sha256),
            "provenance": str(expected.provenance),
            "source_review_status": str(expected.source_review_status),
            "setting": str(expected.setting),
            "cover": str(expected.cover),
            "freeze_manifest_core_sha256": str(expected.freeze_manifest_core_sha256),
            "structural_test_samples": int(expected.structural_test_samples),
            "structural_test_subjects": int(expected.structural_test_subjects),
        },
        structural_test=snapshot.structural_test.as_dict(),
    )


# ---------------------------------------------------------------------------
# Build the B01ContractExpected from the B04 frozen config
# ---------------------------------------------------------------------------


def build_b01_contract_expected(config: Mapping[str, Any]) -> B01ContractExpected:
    """Construct :class:`B01ContractExpected` from the B04 config.

    R04 freezes every expected value as an explicit top-level field
    in the B04 config so a Reviewer can grep the JSON for the
    contract terms.
    """
    if "b01_freeze_manifest_core_sha256_expected" not in config:
        raise B01ContractError(
            "B04 config missing required field "
            "'b01_freeze_manifest_core_sha256_expected'"
        )
    if "b01_structural_test" not in config:
        raise B01ContractError(
            "B04 config missing required field 'b01_structural_test'"
        )
    structural_test = config["b01_structural_test"]
    if "sample_count" not in structural_test or "subject_count" not in structural_test:
        raise B01ContractError(
            "B04 config b01_structural_test must contain sample_count and "
            "subject_count"
        )
    return B01ContractExpected(
        train_count=int(config["expected_split_counts"]["train"]),
        val_count=int(config["expected_split_counts"]["val"]),
        test_count=int(config["expected_split_counts"]["test"]),
        train_subjects=int(config["expected_subjects"]["train"]),
        val_subjects=int(config["expected_subjects"]["val"]),
        test_subjects=int(config["expected_subjects"]["test"]),
        a06_split_sha256=str(config["b01_a06_split_sha256_expected"]),
        provenance=str(config["expected_provenance"]),
        source_review_status=str(config["expected_source_review_status"]),
        setting=str(config["expected_setting"]),
        cover=str(config["expected_cover"]),
        freeze_manifest_core_sha256=str(
            config["b01_freeze_manifest_core_sha256_expected"]
        ),
        structural_test_samples=int(structural_test["sample_count"]),
        structural_test_subjects=int(structural_test["subject_count"]),
    )


# ---------------------------------------------------------------------------
# File-level check (helper kept for symmetry with R02/R03)
# ---------------------------------------------------------------------------


def check_freeze_manifest_file_consistency(
    freeze_dir: Path,
    *,
    freeze_manifest_sha256: str,
) -> None:
    """Refuse to proceed if ``freeze_manifest.json`` on disk is missing
    or its computed core SHA does not match the caller-supplied SHA.

    The caller is expected to pass the SHA reported by the B01 freeze
    handle (which is ``FreezeManifest.core_sha256()``).
    """

    path = Path(freeze_dir) / "freeze_manifest.json"
    if not path.is_file():
        raise B01ContractError(
            f"freeze_manifest.json missing under {freeze_dir}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise B01ContractError(
            f"freeze_manifest.json unreadable or invalid JSON: {exc}"
        ) from exc
    if "core" not in payload:
        raise B01ContractError("freeze_manifest.json missing 'core' key")
    actual_core_sha = sha256_hex(
        canonical_json_dumps(payload["core"]).encode("utf-8")
    )
    if actual_core_sha != freeze_manifest_sha256:
        raise B01ContractError(
            f"freeze_manifest core SHA mismatch: on-disk {actual_core_sha} != "
            f"reported {freeze_manifest_sha256}"
        )
