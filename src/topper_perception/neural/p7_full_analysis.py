"""Independent PoPu P7 Full evidence-pack re-verification and analysis.

This module performs a *fresh* re-analysis of the frozen Full P7 evidence
produced by ``EXP-P7-FULL-20260820-R02`` (and any equivalent future pack with
the same schema). It deliberately avoids re-using the ``p7_runner`` codepath
that produced the evidence so the Reviewer can compare two independent
implementations of the same metrics and detect silent drift.

The contract enforced here is the union of Reviewer requirements:

- *Pool first, then compute the metric.* Per-fold averaging is forbidden.
- *5 seeds are reported separately* as mean / std / worst; they must never be
  pooled into a single metric with ``n_seeds == 25``.
- *P6 single-checkpoint* rule uses threshold ``0.94`` (frozen).
- *P6.1 ensemble* rule uses temperature ``0.75``, threshold ``0.5``,
  ``require_unanimous=True`` (frozen).
- The four worst-subject criteria are WAR DESC, coverage ASC,
  accepted-accuracy ASC, raw-accuracy ASC.
- Every output JSON serializes non-finite values as ``null`` — the strict
  JSON loader rejects ``NaN`` / ``Infinity`` / ``-Infinity`` constants so
  downstream consumers can re-load the manifest without silent type drift.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from topper_perception.neural.data import FROZEN_LABELS, LABEL_TO_INDEX
from topper_perception.neural.metrics import compute_classification_metrics
from topper_perception.neural.p6_1 import (
    aggregate_repeat_ensemble,
    calibrated_frame,
    selective_metrics as p6_1_selective_metrics,
)
from topper_perception.neural.p6_reject import (
    PROBA_COLUMNS,
    RejectRule,
    add_uncertainty_columns,
    apply_rule,
    error_cases,
    grouped_metrics,
)
from topper_perception.neural.p7_runner import P7Condition, parse_p7_conditions

# ---------------------------------------------------------------------------
# Frozen analysis contract (Reviewer point: no free parameters)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "p7-full-analysis-v0.2"
FROZEN_P6_SINGLE_THRESHOLD = 0.94
FROZEN_P6_1_TEMPERATURE = 0.75
FROZEN_P6_1_THRESHOLD = 0.5
FROZEN_P6_1_REQUIRE_UNANIMOUS = True

# Frozen expected/actual SHA-256 from the Full pack's condition_comparison.json
# rule block. The runner MUST fail closed if the on-disk SHA drifts from the
# values below; this is the Reviewer-mandated audit anchor for the rule block.
FROZEN_P6_SOURCE_PATH = "outputs/analysis/EXP-P6-POPU-REJECT-20260820-R01/summary.json"
FROZEN_P6_EXPECTED_SHA256 = "af9ec5d74d64699c27cc2b18976d74424fa3c475915415a9afe6ed3907666929"
FROZEN_P6_ACTUAL_SHA256 = "af9ec5d74d64699c27cc2b18976d74424fa3c475915415a9afe6ed3907666929"
FROZEN_P6_THRESHOLD_POINTER = "/selected_rule/confidence_threshold"
FROZEN_P6_FALLBACK_THRESHOLD_POINTER = "/selection/threshold"
FROZEN_P6_1_SOURCE_PATH = "outputs/analysis/EXP-P6.1-POPU-CALIBRATION-20260820-R01/summary.json"
FROZEN_P6_1_EXPECTED_SHA256 = "d8b191ba2a8fefc2d4a91c654fa6c411f326f031514176eb4f92be919d744125"
FROZEN_P6_1_ACTUAL_SHA256 = "d8b191ba2a8fefc2d4a91c654fa6c411f326f031514176eb4f92be919d744125"
FROZEN_P6_1_TEMPERATURE_POINTER = "/temperature"
FROZEN_P6_1_RULE_POINTER = "/rules/1/threshold"
FROZEN_P6_1_UNANIMITY_FIELD_POINTER = "/rules/1/require_unanimous"
FROZEN_HIGH_CONFIDENCE_THRESHOLD = 0.90
FROZEN_CONDITION_NAMES: tuple[str, ...] = (
    "density_stride_2_2",
    "density_stride_4_4",
    "noise_p95_0.01",
    "noise_p95_0.05",
    "noise_p95_0.10",
    "bad_cell_0.01",
    "bad_cell_0.05",
    "bad_cell_0.10",
    "bad_rows_1",
    "bad_rows_2",
    "bad_rows_4",
    "bad_columns_1",
    "bad_columns_2",
    "bad_columns_4",
)
FROZEN_SEEDS: tuple[int, ...] = (701, 702, 703, 704, 705)
FROZEN_REPEATS: tuple[int, ...] = (0, 1, 2)
FROZEN_LOCAL_FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ArchiveIntegrityError(ValueError):
    """Raised when an evidence pack fails integrity verification."""


class StitchingError(ValueError):
    """Raised when a stitched OOF frame cannot be assembled from the pack."""


# ---------------------------------------------------------------------------
# Evidence manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceManifest:
    """Verification record for one Full P7 evidence pack."""

    evidence_root: Path
    archive_path: Path | None
    archive_sha256: str | None
    n_folds_resolved: int
    n_conditions_resolved: int
    n_seeds_resolved: int
    n_clean_records_total: int
    file_count: int
    file_sha256s: Mapping[str, str]
    parse_failures: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict with non-finite values nulled."""
        return _jsonify({
            "schema_version": self.schema_version,
            "evidence_root": str(self.evidence_root),
            "archive_path": str(self.archive_path) if self.archive_path else None,
            "archive_sha256": self.archive_sha256,
            "n_folds_resolved": self.n_folds_resolved,
            "n_conditions_resolved": self.n_conditions_resolved,
            "n_seeds_resolved": self.n_seeds_resolved,
            "n_clean_records_total": self.n_clean_records_total,
            "file_count": self.file_count,
            "file_sha256s": dict(self.file_sha256s),
            "parse_failures": list(self.parse_failures),
        })


# ---------------------------------------------------------------------------
# Strict JSON loader (NaN / Infinity rejection)
# ---------------------------------------------------------------------------


def _raise_on_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant rejected: {value!r}")


def strict_json_load(path: Path) -> dict[str, Any]:
    """Load a JSON object, refusing any NaN / Infinity / -Infinity constant."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")
    try:
        payload = json.loads(text, parse_constant=_raise_on_nonfinite)
    except ValueError as exc:
        raise ValueError(f"non-finite value in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"strict JSON loader requires an object root, got {type(payload).__name__} in {path}."
        )
    return payload


# ---------------------------------------------------------------------------
# Pinned-rule block loader (Reviewer Round 2: fail-closed rule source)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PinnedRule:
    """A pinned P6 / P6.1 rule block loaded from the Full pack with SHA verification.

    The block is loaded from ``condition_comparison.json`` and the embedded
    ``source_expected_sha256`` / ``source_actual_sha256`` pair is checked
    against the frozen module constants. Any mismatch (drift, tamper, missing
    field) raises :class:`ArchiveIntegrityError` so the analysis fails closed.
    """

    rule_kind: str
    source_path: str
    source_expected_sha256: str
    source_actual_sha256: str
    threshold: float | None
    temperature: float | None
    require_unanimous: bool | None
    threshold_pointer: str | None
    fallback_threshold_pointer: str | None
    temperature_pointer: str | None
    rule_pointer: str | None
    unanimity_field_pointer: str | None

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict with non-finite values nulled."""
        return _jsonify({
            "rule_kind": self.rule_kind,
            "source_path": self.source_path,
            "source_expected_sha256": self.source_expected_sha256,
            "source_actual_sha256": self.source_actual_sha256,
            "threshold": self.threshold,
            "temperature": self.temperature,
            "require_unanimous": self.require_unanimous,
            "threshold_pointer": self.threshold_pointer,
            "fallback_threshold_pointer": self.fallback_threshold_pointer,
            "temperature_pointer": self.temperature_pointer,
            "rule_pointer": self.rule_pointer,
            "unanimity_field_pointer": self.unanimity_field_pointer,
        })


def _require_block(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    block = payload.get(key)
    if not isinstance(block, Mapping):
        raise ArchiveIntegrityError(
            f"condition_comparison.json missing rule block {key!r}: "
            f"got {type(block).__name__}."
        )
    return block


def _require_string(block: Mapping[str, Any], key: str) -> str:
    value = block.get(key)
    if not isinstance(value, str) or not value:
        raise ArchiveIntegrityError(
            f"rule block missing string field {key!r}: got {value!r}."
        )
    return value


def _require_optional_string(block: Mapping[str, Any], key: str) -> str | None:
    value = block.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ArchiveIntegrityError(
            f"rule block field {key!r} must be string or null; got {type(value).__name__}."
        )
    return value


def _coerce_p6_threshold(block: Mapping[str, Any]) -> float:
    raw = block.get("threshold")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ArchiveIntegrityError(
            f"P6 single rule 'threshold' must be numeric; got {raw!r}."
        )
    threshold = float(raw)
    if not 0.0 < threshold <= 1.0:
        raise ArchiveIntegrityError(
            f"P6 single rule 'threshold' must be in (0, 1]; got {threshold}."
        )
    return threshold


def _coerce_p6_1_temperature(block: Mapping[str, Any]) -> float:
    raw = block.get("temperature")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ArchiveIntegrityError(
            f"P6.1 ensemble rule 'temperature' must be numeric; got {raw!r}."
        )
    temperature = float(raw)
    if not 0.0 < temperature <= 10.0:
        raise ArchiveIntegrityError(
            f"P6.1 ensemble rule 'temperature' must be in (0, 10]; got {temperature}."
        )
    return temperature


def _coerce_p6_1_threshold(block: Mapping[str, Any]) -> float:
    raw = block.get("threshold")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ArchiveIntegrityError(
            f"P6.1 ensemble rule 'threshold' must be numeric; got {raw!r}."
        )
    threshold = float(raw)
    if not 0.0 < threshold <= 1.0:
        raise ArchiveIntegrityError(
            f"P6.1 ensemble rule 'threshold' must be in (0, 1]; got {threshold}."
        )
    return threshold


def _coerce_p6_1_require_unanimous(block: Mapping[str, Any]) -> bool:
    raw = block.get("require_unanimous")
    if not isinstance(raw, bool):
        raise ArchiveIntegrityError(
            f"P6.1 ensemble rule 'require_unanimous' must be boolean; got {raw!r}."
        )
    return raw


def _verify_pin(name: str, *, expected: str, actual: str, frozen: str) -> None:
    if expected.lower() != frozen.lower():
        raise ArchiveIntegrityError(
            f"{name}: archive-embedded expected_sha256={expected} does not match "
            f"module-pinned frozen SHA={frozen}."
        )
    if actual.lower() != frozen.lower():
        raise ArchiveIntegrityError(
            f"{name}: archive-embedded actual_sha256={actual} does not match "
            f"module-pinned frozen SHA={frozen}. Archive was tampered with or "
            f"the upstream source SHA drifted."
        )


def load_p6_single_rule_from_archive(evidence_root: Path) -> PinnedRule:
    """Load the P6 single-checkpoint rule block from ``condition_comparison.json``.

    Verifies the embedded ``source_expected_sha256`` and ``source_actual_sha256``
    against the module-pinned frozen SHA. Returns a :class:`PinnedRule` whose
    ``threshold`` is the machine-readable value the analysis MUST use.
    """
    evidence_root = Path(evidence_root)
    cc_path = evidence_root / "condition_comparison.json"
    if not cc_path.is_file():
        raise ArchiveIntegrityError(
            f"condition_comparison.json not found under {evidence_root}."
        )
    payload = strict_json_load(cc_path)
    block = _require_block(payload, "p6_single_rule")
    if block.get("rule_kind") != "p6_single":
        raise ArchiveIntegrityError(
            f"p6_single_rule.rule_kind must be 'p6_single'; got {block.get('rule_kind')!r}."
        )
    expected = _require_string(block, "source_expected_sha256")
    actual = _require_string(block, "source_actual_sha256")
    _verify_pin(
        "p6_single_rule",
        expected=expected,
        actual=actual,
        frozen=FROZEN_P6_EXPECTED_SHA256,
    )
    source_path = _require_string(block, "source_path")
    if source_path != FROZEN_P6_SOURCE_PATH:
        raise ArchiveIntegrityError(
            f"p6_single_rule.source_path drift: expected {FROZEN_P6_SOURCE_PATH!r}, "
            f"got {source_path!r}."
        )
    threshold = _coerce_p6_threshold(block)
    return PinnedRule(
        rule_kind="p6_single",
        source_path=source_path,
        source_expected_sha256=expected,
        source_actual_sha256=actual,
        threshold=threshold,
        temperature=None,
        require_unanimous=None,
        threshold_pointer=_require_optional_string(block, "threshold_pointer"),
        fallback_threshold_pointer=_require_optional_string(
            block, "fallback_threshold_pointer"
        ),
        temperature_pointer=None,
        rule_pointer=None,
        unanimity_field_pointer=None,
    )


def load_p6_1_ensemble_rule_from_archive(evidence_root: Path) -> PinnedRule:
    """Load the P6.1 ensemble rule block from ``condition_comparison.json``.

    The ``rule_pointer`` MUST end with ``/rules/1/threshold`` (the unanimity
    branch) per the upstream protocol. Verifies the embedded
    ``source_expected_sha256`` and ``source_actual_sha256`` against the
    module-pinned frozen SHA. Returns a :class:`PinnedRule` whose
    ``temperature`` / ``threshold`` / ``require_unanimous`` are the
    machine-readable values the analysis MUST use.
    """
    evidence_root = Path(evidence_root)
    cc_path = evidence_root / "condition_comparison.json"
    if not cc_path.is_file():
        raise ArchiveIntegrityError(
            f"condition_comparison.json not found under {evidence_root}."
        )
    payload = strict_json_load(cc_path)
    block = _require_block(payload, "p6_1_ensemble_rule")
    if block.get("rule_kind") != "p6_1_ensemble":
        raise ArchiveIntegrityError(
            f"p6_1_ensemble_rule.rule_kind must be 'p6_1_ensemble'; "
            f"got {block.get('rule_kind')!r}."
        )
    expected = _require_string(block, "source_expected_sha256")
    actual = _require_string(block, "source_actual_sha256")
    _verify_pin(
        "p6_1_ensemble_rule",
        expected=expected,
        actual=actual,
        frozen=FROZEN_P6_1_EXPECTED_SHA256,
    )
    source_path = _require_string(block, "source_path")
    if source_path != FROZEN_P6_1_SOURCE_PATH:
        raise ArchiveIntegrityError(
            f"p6_1_ensemble_rule.source_path drift: expected "
            f"{FROZEN_P6_1_SOURCE_PATH!r}, got {source_path!r}."
        )
    rule_pointer = _require_optional_string(block, "rule_pointer")
    if rule_pointer is not None and not rule_pointer.endswith("/rules/1/threshold"):
        raise ArchiveIntegrityError(
            f"p6_1_ensemble_rule.rule_pointer must end with '/rules/1/threshold' "
            f"(the unanimity branch); got {rule_pointer!r}."
        )
    temperature = _coerce_p6_1_temperature(block)
    threshold = _coerce_p6_1_threshold(block)
    require_unanimous = _coerce_p6_1_require_unanimous(block)
    return PinnedRule(
        rule_kind="p6_1_ensemble",
        source_path=source_path,
        source_expected_sha256=expected,
        source_actual_sha256=actual,
        threshold=threshold,
        temperature=temperature,
        require_unanimous=require_unanimous,
        threshold_pointer=None,
        fallback_threshold_pointer=None,
        temperature_pointer=_require_optional_string(block, "temperature_pointer"),
        rule_pointer=rule_pointer,
        unanimity_field_pointer=_require_optional_string(
            block, "unanimity_field_pointer"
        ),
    )


def assert_rule_values_match_frozen(
    p6_rule: PinnedRule, p6_1_rule: PinnedRule
) -> None:
    """Verify the rule block values equal the module-pinned frozen values.

    This is the final consistency check: even if the SHA pair matches, if the
    rule block's numeric thresholds diverge from the module constants the
    analysis refuses to run.
    """
    if p6_rule.threshold != FROZEN_P6_SINGLE_THRESHOLD:
        raise ArchiveIntegrityError(
            f"p6_single_rule.threshold {p6_rule.threshold} != frozen "
            f"{FROZEN_P6_SINGLE_THRESHOLD}. Refusing to run with drifted value."
        )
    if p6_1_rule.temperature != FROZEN_P6_1_TEMPERATURE:
        raise ArchiveIntegrityError(
            f"p6_1_ensemble_rule.temperature {p6_1_rule.temperature} != frozen "
            f"{FROZEN_P6_1_TEMPERATURE}. Refusing to run with drifted value."
        )
    if p6_1_rule.threshold != FROZEN_P6_1_THRESHOLD:
        raise ArchiveIntegrityError(
            f"p6_1_ensemble_rule.threshold {p6_1_rule.threshold} != frozen "
            f"{FROZEN_P6_1_THRESHOLD}. Refusing to run with drifted value."
        )
    if p6_1_rule.require_unanimous != FROZEN_P6_1_REQUIRE_UNANIMOUS:
        raise ArchiveIntegrityError(
            f"p6_1_ensemble_rule.require_unanimous {p6_1_rule.require_unanimous} "
            f"!= frozen {FROZEN_P6_1_REQUIRE_UNANIMOUS}. Refusing to run with "
            f"drifted value."
        )


# ---------------------------------------------------------------------------
# Archive / directory verification
# ---------------------------------------------------------------------------


def _iter_archive_members(archive_path: Path) -> Iterable[tuple[str, bytes]]:
    """Yield (member-name, bytes) for every regular file in the tar.gz archive."""
    with tarfile.open(archive_path, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                continue
            yield member.name, extracted.read()


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as handle:
        handle.extractall(target_dir)


def _expected_layout_root(archive_members: Iterable[str]) -> str | None:
    """Return the archive's top-level directory name (e.g. ``EXP-P7-FULL-...``)."""
    for name in archive_members:
        # ``a/b/c.json`` → top-level = ``a``
        top = name.split("/", 1)[0]
        if top and top != ".." and not top.startswith("/"):
            return top
    return None


def _resolve_evidence_root(input_path: Path, archive_members: Iterable[str]) -> Path:
    """Map a tar.gz input to its extracted top-level directory."""
    if input_path.is_dir():
        return input_path
    top = _expected_layout_root(archive_members)
    if top is None:
        raise ArchiveIntegrityError(
            f"Could not determine top-level directory of archive {input_path}."
        )
    return input_path.parent / top


def _list_existing_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _sha256_hex_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_condition_comparison(payload: Mapping[str, Any]) -> None:
    """Sanity-check the top-level ``condition_comparison.json``."""
    required = {"n_folds_resolved", "clean_n_records_total", "condition_summaries"}
    missing = required - set(payload)
    if missing:
        raise ArchiveIntegrityError(
            f"condition_comparison.json missing keys: {sorted(missing)}."
        )


def verify_evidence_archive(
    input_path: Path,
    *,
    expected_sha256: str | None = None,
    extract_dir: Path | None = None,
) -> EvidenceManifest:
    """Verify the archive SHA-256 (when applicable) and the pack structure.

    When ``input_path`` points to a directory the function verifies it
    directly without extracting. When it points to a ``.tar.gz`` the
    function SHA-256-verifies the archive *and* extracts it under
    ``extract_dir`` (or a temporary directory when omitted). The original
    archive is never modified.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise ArchiveIntegrityError(f"Evidence path not found: {input_path}")

    archive_sha256: str | None = None
    archive_path: Path | None = None

    if input_path.is_dir():
        evidence_root = input_path
    elif tarfile.is_tarfile(input_path):
        archive_path = input_path
        if expected_sha256 is not None:
            actual = hashlib.sha256(input_path.read_bytes()).hexdigest()
            if actual.lower() != expected_sha256.lower():
                raise ArchiveIntegrityError(
                    f"Archive SHA-256 mismatch: pinned={expected_sha256}, "
                    f"actual={actual}."
                )
            archive_sha256 = actual
        if extract_dir is None:
            extract_dir = Path(tempfile.mkdtemp(prefix="p7_full_", suffix=".extract"))
        else:
            extract_dir.mkdir(parents=True, exist_ok=True)
        _extract_archive(input_path, extract_dir)
        evidence_root = extract_dir / _expected_layout_root(
            (name for name, _ in _iter_archive_members(input_path))
        )
        if not evidence_root.is_dir():
            raise ArchiveIntegrityError(
                f"Expected extracted directory not found: {evidence_root}."
            )
    else:
        raise ArchiveIntegrityError(
            f"Evidence path is neither a directory nor a tar.gz: {input_path}."
        )

    # Walk every file, record SHA-256, strict-parse every JSON.
    file_paths = _list_existing_files(evidence_root)
    file_sha256s: dict[str, str] = {}
    parse_failures: list[str] = []

    for file_path in file_paths:
        relative = file_path.relative_to(evidence_root).as_posix()
        data = file_path.read_bytes()
        file_sha256s[relative] = _sha256_hex_of_bytes(data)
        if file_path.suffix == ".json":
            try:
                strict_json_load(file_path)
            except ValueError as exc:
                parse_failures.append(f"{relative}: {exc}")

    if parse_failures:
        raise ArchiveIntegrityError(
            "non-finite values detected in JSON files: " + "; ".join(parse_failures)
        )

    # Layout checks.
    folds_root = evidence_root / "folds"
    if not folds_root.is_dir():
        raise ArchiveIntegrityError(f"Missing folds/ root under {evidence_root}.")
    fold_dirs = sorted(folds_root.glob("repeat_*/fold_*"))
    n_folds_resolved = len(fold_dirs)
    expected_folds = len(FROZEN_REPEATS) * len(FROZEN_LOCAL_FOLDS)
    if n_folds_resolved != expected_folds:
        raise ArchiveIntegrityError(
            f"Expected {expected_folds} folds, found {n_folds_resolved}."
        )

    # Condition + seed counts: every fold must contain the same set.
    conditions: set[str] = set()
    seeds: set[int] = set()
    clean_record_total = 0
    for fold_dir in fold_dirs:
        clean_csv = fold_dir / "clean" / "record_predictions.csv"
        if not clean_csv.is_file():
            raise ArchiveIntegrityError(f"Missing clean OOF at {clean_csv}.")
        clean_record_total += len(pd.read_csv(clean_csv))
        for condition_dir in (path for path in fold_dir.iterdir() if path.is_dir() and path.name != "clean"):
            conditions.add(condition_dir.name)
            for seed_dir in condition_dir.glob("seed_*"):
                seeds.add(int(seed_dir.name.split("_", 1)[1]))

    missing_conditions = set(FROZEN_CONDITION_NAMES) - conditions
    extra_conditions = conditions - set(FROZEN_CONDITION_NAMES)
    if missing_conditions:
        raise ArchiveIntegrityError(
            f"Missing conditions in pack: {sorted(missing_conditions)}."
        )
    if extra_conditions:
        raise ArchiveIntegrityError(
            f"Unexpected conditions in pack: {sorted(extra_conditions)}."
        )
    if seeds != set(FROZEN_SEEDS):
        raise ArchiveIntegrityError(
            f"seed set drift (expected {FROZEN_SEEDS}, got {sorted(seeds)})."
        )

    # condition_comparison.json sanity check when present.
    cc_path = evidence_root / "condition_comparison.json"
    if cc_path.is_file():
        try:
            _verify_condition_comparison(strict_json_load(cc_path))
        except ValueError as exc:
            raise ArchiveIntegrityError(f"condition_comparison.json: {exc}") from exc

    # Fail-closed P6 / P6.1 rule block load (Reviewer Round 2): the Full pack
    # must carry the pinned rule block AND its SHA pair must match the
    # module-pinned frozen SHAs. This is the only authoritative source for the
    # P6 single threshold and P6.1 ensemble parameters.
    p6_rule = load_p6_single_rule_from_archive(evidence_root)
    p6_1_rule = load_p6_1_ensemble_rule_from_archive(evidence_root)
    assert_rule_values_match_frozen(p6_rule, p6_1_rule)

    return EvidenceManifest(
        evidence_root=evidence_root,
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        n_folds_resolved=n_folds_resolved,
        n_conditions_resolved=len(conditions),
        n_seeds_resolved=len(seeds),
        n_clean_records_total=int(clean_record_total),
        file_count=len(file_paths),
        file_sha256s=file_sha256s,
        parse_failures=tuple(parse_failures),
    )


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------


_REQUIRED_RECORD_COLUMNS = {
    "model", "repeat", "outer_seed", "local_fold", "record_id", "subject_id",
    "y_true", "y_pred", "confidence", "n_snapshots", *PROBA_COLUMNS,
}


def _validate_record_frame(frame: pd.DataFrame, *, where: str) -> None:
    missing = _REQUIRED_RECORD_COLUMNS - set(frame.columns)
    if missing:
        raise StitchingError(
            f"{where}: missing required columns: {sorted(missing)}."
        )
    if frame.empty:
        raise StitchingError(f"{where}: stitched OOF is empty.")
    probabilities = frame.loc[:, list(PROBA_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise StitchingError(f"{where}: probabilities contain non-finite values.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise StitchingError(f"{where}: probabilities outside [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise StitchingError(f"{where}: probability rows do not sum to 1.")


def _stitch_directory(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.rglob("record_predictions.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 - explicit re-raise as StitchingError
            raise StitchingError(f"Could not read {path}: {exc}") from exc
        _validate_record_frame(frame, where=str(path))
        frames.append(frame)
    if not frames:
        raise StitchingError(f"No record_predictions.csv found under {root}.")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_clean_oof(evidence_root: Path) -> pd.DataFrame:
    """Load and stitch the clean OOF across all 15 folds."""
    evidence_root = Path(evidence_root)
    frames: list[pd.DataFrame] = []
    for repeat in FROZEN_REPEATS:
        for local_fold in FROZEN_LOCAL_FOLDS:
            csv_path = (
                evidence_root / "folds" / f"repeat_{repeat}" / f"fold_{local_fold}"
                / "clean" / "record_predictions.csv"
            )
            if not csv_path.is_file():
                raise StitchingError(f"Missing clean OOF at {csv_path}.")
            frame = pd.read_csv(csv_path)
            _validate_record_frame(frame, where=str(csv_path))
            frames.append(frame)
    stitched = pd.concat(frames, ignore_index=True, sort=False)
    return add_uncertainty_columns(stitched)


def load_condition_seed_oof(
    evidence_root: Path, condition: str, seed: int
) -> pd.DataFrame:
    """Load and stitch one (condition, seed) OOF across all 15 folds."""
    evidence_root = Path(evidence_root)
    if condition not in FROZEN_CONDITION_NAMES:
        raise StitchingError(
            f"Unknown condition {condition!r}; expected one of {FROZEN_CONDITION_NAMES}."
        )
    if int(seed) not in FROZEN_SEEDS:
        raise StitchingError(
            f"Unknown seed {seed!r}; expected one of {FROZEN_SEEDS}."
        )
    frames: list[pd.DataFrame] = []
    for repeat in FROZEN_REPEATS:
        for local_fold in FROZEN_LOCAL_FOLDS:
            csv_path = (
                evidence_root / "folds" / f"repeat_{repeat}" / f"fold_{local_fold}"
                / condition / f"seed_{int(seed)}" / "record_predictions.csv"
            )
            if not csv_path.is_file():
                raise StitchingError(f"Missing condition OOF at {csv_path}.")
            frame = pd.read_csv(csv_path)
            _validate_record_frame(frame, where=str(csv_path))
            frames.append(frame)
    stitched = pd.concat(frames, ignore_index=True, sort=False)
    return add_uncertainty_columns(stitched)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_stitched_classification_metrics(stitched: pd.DataFrame) -> dict[str, Any]:
    """Compute raw accuracy, balanced accuracy, macro-F1, per-class on the stitched frame."""
    if stitched.empty:
        return _empty_classification_metrics()
    true = np.asarray(
        [LABEL_TO_INDEX[str(label)] for label in stitched["y_true"]], dtype=np.int64
    )
    pred = np.asarray(
        [LABEL_TO_INDEX[str(label)] for label in stitched["y_pred"]], dtype=np.int64
    )
    metrics = compute_classification_metrics(true, pred, FROZEN_LABELS)
    return _jsonify(metrics.as_dict())


def _empty_classification_metrics() -> dict[str, Any]:
    return {
        "accuracy": None,
        "balanced_accuracy": None,
        "macro_f1": None,
        "macro_precision": None,
        "macro_recall": None,
        "per_class": [
            {"label": label, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
            for label in FROZEN_LABELS
        ],
        "confusion_matrix": [[0] * len(FROZEN_LABELS) for _ in FROZEN_LABELS],
        "n_samples": 0,
    }


def _ensure_uncertainty_columns(stitched: pd.DataFrame) -> pd.DataFrame:
    """Re-apply uncertainty columns when callers pass raw stitched frames."""
    required = {"top1_probability", "correct"}
    if required.issubset(stitched.columns):
        return stitched
    return add_uncertainty_columns(stitched.copy())


def compute_p6_single(stitched: pd.DataFrame) -> dict[str, Any]:
    """Apply the P6 single-checkpoint rule to the stitched frame.

    The threshold is the frozen ``FROZEN_P6_SINGLE_THRESHOLD = 0.94`` and must
    never be overridden by callers. The threshold is also loaded from the
    archive's pinned rule block and verified to equal this constant — see
    :func:`verify_evidence_archive`.
    """
    threshold = float(FROZEN_P6_SINGLE_THRESHOLD)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"P6 threshold must be in (0, 1], got {threshold}.")
    if stitched.empty:
        return _empty_p6_single(threshold)
    scored = apply_rule(
        _ensure_uncertainty_columns(stitched),
        RejectRule(confidence_threshold=threshold),
    )
    accepted = scored["accepted"]
    accepted_n = int(accepted.sum())
    wrong_n = int(scored["wrong_action"].sum())
    n = int(len(scored))
    return _jsonify({
        "rule_kind": "p6_single",
        "threshold": float(threshold),
        "n": n,
        "accepted_n": accepted_n,
        "coverage": float(accepted_n) / n if n else None,
        "reject_rate": 1.0 - (float(accepted_n) / n) if n else None,
        "wrong_action_n": wrong_n,
        "wrong_action_rate": float(wrong_n) / n if n else None,
        "accepted_accuracy": (
            float(scored.loc[accepted, "correct"].mean()) if accepted_n else None
        ),
        "accepted_error_rate": float(wrong_n) / accepted_n if accepted_n else None,
    })


def _empty_p6_single(threshold: float) -> dict[str, Any]:
    return _jsonify({
        "rule_kind": "p6_single",
        "threshold": float(threshold),
        "n": 0,
        "accepted_n": 0,
        "coverage": None,
        "reject_rate": None,
        "wrong_action_n": 0,
        "wrong_action_rate": None,
        "accepted_accuracy": None,
        "accepted_error_rate": None,
    })


def compute_p6_1_ensemble(stitched: pd.DataFrame) -> dict[str, Any]:
    """Apply the P6.1 calibrated ensemble rule to the stitched frame.

    The temperature / threshold / require_unanimous come from the frozen
    module constants (``FROZEN_P6_1_TEMPERATURE`` etc.) and are also verified
    against the archive's pinned rule block. They must never be overridden by
    callers — see :func:`verify_evidence_archive`.

    ``stitched`` MUST contain exactly one row per (record_id, repeat) across
    the 15 folds. The P6.1 ensemble requires three repeats per record; if the
    pack violates this contract the function emits a structured empty result
    rather than falling back to the P6 single rule.
    """
    temperature = float(FROZEN_P6_1_TEMPERATURE)
    threshold = float(FROZEN_P6_1_THRESHOLD)
    require_unanimous = bool(FROZEN_P6_1_REQUIRE_UNANIMOUS)

    empty_result = _jsonify({
        "rule_kind": "p6_1_ensemble",
        "temperature": temperature,
        "threshold": threshold,
        "require_unanimous": require_unanimous,
        "n": int(len(stitched)),
        "n_unique_records": int(stitched["record_id"].nunique()) if not stitched.empty else 0,
        "unanimous_count": 0,
        "accepted_n": 0,
        "coverage": 0.0,
        "reject_rate": 1.0,
        "wrong_action_n": 0,
        "wrong_action_rate": 0.0,
        "accepted_accuracy": None,
        "accepted_error_rate": None,
        "ensemble_error": "no records formed an ensemble (need exactly 3 repeats per record_id)",
    })

    if stitched.empty:
        return empty_result
    try:
        ensemble = aggregate_repeat_ensemble(stitched)
    except ValueError as exc:
        empty_result["ensemble_error"] = str(exc)
        return empty_result
    calibrated = calibrated_frame(ensemble, temperature=temperature)
    selective = p6_1_selective_metrics(
        calibrated, threshold=threshold, require_unanimous=require_unanimous
    )
    return _jsonify({
        "rule_kind": "p6_1_ensemble",
        "temperature": temperature,
        "threshold": threshold,
        "require_unanimous": require_unanimous,
        "unanimous_count": int(calibrated["unanimous"].sum()),
        **selective,
    })


def _empty_p6_1_ensemble() -> dict[str, Any]:
    return _jsonify({
        "rule_kind": "p6_1_ensemble",
        "temperature": FROZEN_P6_1_TEMPERATURE,
        "threshold": FROZEN_P6_1_THRESHOLD,
        "require_unanimous": FROZEN_P6_1_REQUIRE_UNANIMOUS,
        "n": 0,
        "n_unique_records": 0,
        "unanimous_count": 0,
        "accepted_n": 0,
        "coverage": None,
        "reject_rate": None,
        "wrong_action_n": 0,
        "wrong_action_rate": None,
        "accepted_accuracy": None,
        "accepted_error_rate": None,
    })


# ---------------------------------------------------------------------------
# Per-class / per-subject breakdowns
# ---------------------------------------------------------------------------


def _breakdown_by(
    stitched: pd.DataFrame, *, by: str, threshold: float
) -> pd.DataFrame:
    """Build a per-(by) breakdown with raw accuracy + P6 single-checkpoint reject metrics."""
    columns = [
        by, "n", "wrong_action_n", "wrong_action_rate", "accuracy",
        "accepted_n", "coverage", "accepted_accuracy",
        "accepted_error_rate", "p6_threshold",
    ]
    if stitched.empty:
        return pd.DataFrame(columns=columns)
    scored = apply_rule(
        _ensure_uncertainty_columns(stitched.copy()),
        RejectRule(confidence_threshold=threshold),
    )
    rows: list[dict[str, Any]] = []
    for key, group in scored.groupby(by, sort=True):
        accepted = group["accepted"]
        accepted_n = int(accepted.sum())
        wrong_n = int(group["wrong_action"].sum())
        n = int(len(group))
        rows.append({
            by: str(key),
            "n": n,
            "wrong_action_n": wrong_n,
            "wrong_action_rate": float(wrong_n) / n if n else 0.0,
            "accuracy": float(group["correct"].sum()) / n if n else 0.0,
            "accepted_n": accepted_n,
            "coverage": float(accepted_n) / n if n else 0.0,
            "accepted_accuracy": (
                float(group.loc[accepted, "correct"].mean()) if accepted_n else None
            ),
            "accepted_error_rate": float(wrong_n) / accepted_n if accepted_n else None,
            "p6_threshold": float(threshold),
        })
    return pd.DataFrame(rows, columns=columns)


def compute_per_class_breakdown(stitched: pd.DataFrame) -> pd.DataFrame:
    """Per-class breakdown using the P6 single-checkpoint threshold.

    The threshold is the frozen ``FROZEN_P6_SINGLE_THRESHOLD`` and must never
    be overridden by callers.
    """
    threshold = float(FROZEN_P6_SINGLE_THRESHOLD)
    if stitched.empty:
        # Ensure every frozen label appears in the breakdown even with no records.
        rows = []
        for label in FROZEN_LABELS:
            rows.append({
                "y_true": label, "n": 0, "wrong_action_n": 0, "wrong_action_rate": 0.0,
                "accuracy": 0.0, "accepted_n": 0, "coverage": 0.0,
                "accepted_accuracy": None, "accepted_error_rate": None,
                "p6_threshold": float(threshold),
            })
        return pd.DataFrame(rows)
    return _breakdown_by(stitched, by="y_true", threshold=threshold)


def compute_per_subject_breakdown(stitched: pd.DataFrame) -> pd.DataFrame:
    """Per-subject breakdown using the P6 single-checkpoint threshold.

    The threshold is the frozen ``FROZEN_P6_SINGLE_THRESHOLD`` and must never
    be overridden by callers.
    """
    threshold = float(FROZEN_P6_SINGLE_THRESHOLD)
    return _breakdown_by(stitched, by="subject_id", threshold=threshold)


# ---------------------------------------------------------------------------
# Worst subjects (four criteria)
# ---------------------------------------------------------------------------


def compute_worst_subjects(per_subject: pd.DataFrame) -> dict[str, dict[str, Any] | None]:
    """Return the worst subject by FOUR distinct criteria simultaneously.

    The criteria are:

    - ``by_wrong_action_rate`` — highest WAR DESC (most wrong).
    - ``by_coverage``           — lowest coverage ASC (most rejected).
    - ``by_accepted_accuracy``  — lowest accepted_accuracy ASC.
    - ``by_raw_accuracy``       — lowest accuracy ASC.

    Ties on each criterion are broken deterministically by ``subject_id`` ASC.
    ``None`` metrics are treated as ``0.0`` for sorting.
    """
    empty = {
        "by_wrong_action_rate": None,
        "by_coverage": None,
        "by_accepted_accuracy": None,
        "by_raw_accuracy": None,
    }
    if per_subject is None or per_subject.empty:
        return empty
    required = {"subject_id", "wrong_action_rate", "coverage", "accepted_accuracy", "accuracy"}
    missing = required - set(per_subject.columns)
    if missing:
        raise ValueError(
            f"per_subject breakdown missing columns for worst-subject selection: {sorted(missing)}."
        )

    def _pick(column: str, *, descending: bool) -> dict[str, Any]:
        metric_value = lambda row: float(row.get(column) or 0.0)  # noqa: E731
        return sorted(
            per_subject.to_dict(orient="records"),
            key=lambda row: (
                -metric_value(row) if descending else metric_value(row),
                str(row["subject_id"]),
            ),
        )[0]

    return _jsonify({
        "by_wrong_action_rate": _pick("wrong_action_rate", descending=True),
        "by_coverage": _pick("coverage", descending=False),
        "by_accepted_accuracy": _pick("accepted_accuracy", descending=False),
        "by_raw_accuracy": _pick("accuracy", descending=False),
    })


# ---------------------------------------------------------------------------
# High-confidence error filtering
# ---------------------------------------------------------------------------


def compute_high_confidence_errors(
    error_rows: pd.DataFrame, *, threshold: float = FROZEN_HIGH_CONFIDENCE_THRESHOLD
) -> pd.DataFrame:
    """Filter an error-cases DataFrame to those with confidence ≥ threshold.

    The output is sorted by confidence DESC so the most decisive wrong
    predictions come first. This is purely a confidence filter; the
    decision of what counts as an "error case" is the caller's
    responsibility (``compute_seed_summary`` produces error_cases by
    applying P6 single with ``threshold=0.0`` so every wrong prediction
    appears).
    """
    if error_rows is None or error_rows.empty:
        return pd.DataFrame()
    if "confidence" not in error_rows.columns:
        raise ValueError("error_rows must include the 'confidence' column.")
    filtered = error_rows[error_rows["confidence"].astype(float) >= float(threshold)].copy()
    return filtered.sort_values("confidence", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Seed / condition aggregation (5 seeds as mean / std / worst — never pooled)
# ---------------------------------------------------------------------------


def compute_seed_summary(
    *,
    condition: str,
    seed: int,
    clean_stitched: pd.DataFrame,
    perturbed_stitched: pd.DataFrame,
) -> dict[str, Any]:
    """Compute the per-(condition, seed) summary block on the already-stitched frame.

    All rule parameters come from the frozen module constants; callers cannot
    override them. The archive's pinned rule block is verified to equal these
    constants in :func:`verify_evidence_archive`.
    """
    record_metrics = compute_stitched_classification_metrics(perturbed_stitched)
    clean_metrics = compute_stitched_classification_metrics(clean_stitched)
    p6_single = compute_p6_single(perturbed_stitched)
    clean_p6_single = compute_p6_single(clean_stitched)
    p6_1 = compute_p6_1_ensemble(perturbed_stitched)
    per_class = compute_per_class_breakdown(perturbed_stitched)
    per_subject = compute_per_subject_breakdown(perturbed_stitched)
    worst_subjects = compute_worst_subjects(per_subject)
    errors = _collect_error_cases(perturbed_stitched)

    delta = _jsonify({
        "record_macro_f1": _safe_subtract(
            record_metrics.get("macro_f1"), clean_metrics.get("macro_f1")
        ),
        "record_balanced_accuracy": _safe_subtract(
            record_metrics.get("balanced_accuracy"), clean_metrics.get("balanced_accuracy")
        ),
        "record_accuracy": _safe_subtract(
            record_metrics.get("accuracy"), clean_metrics.get("accuracy")
        ),
        "p6_single_wrong_action_rate": _safe_subtract(
            p6_single.get("wrong_action_rate"), clean_p6_single.get("wrong_action_rate")
        ),
        "p6_single_coverage": _safe_subtract(
            p6_single.get("coverage"), clean_p6_single.get("coverage")
        ),
        "p6_single_accepted_accuracy": _safe_subtract(
            p6_single.get("accepted_accuracy"), clean_p6_single.get("accepted_accuracy")
        ),
    })

    return _jsonify({
        "condition": condition,
        "seed": int(seed),
        "n_records": int(len(perturbed_stitched)),
        "n_unique_records": int(perturbed_stitched["record_id"].nunique()) if not perturbed_stitched.empty else 0,
        "record_metrics": record_metrics,
        "delta_vs_clean": delta,
        "p6_single_rule": p6_single,
        "p6_1_ensemble_rule": p6_1,
        "per_class": per_class.to_dict(orient="records"),
        "per_subject": per_subject.to_dict(orient="records"),
        "worst_subjects": worst_subjects,
        "error_cases": errors,
    })


def _collect_error_cases(stitched: pd.DataFrame) -> list[dict[str, Any]]:
    """Collect every wrong prediction with confidence / uncertainty diagnostics."""
    if stitched.empty:
        return []
    frame = error_cases(
        _ensure_uncertainty_columns(stitched.copy()),
        threshold=0.0,
        high_confidence=FROZEN_HIGH_CONFIDENCE_THRESHOLD,
    )
    return _jsonify_list(frame.to_dict(orient="records"))


def _safe_subtract(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    try:
        return float(a) - float(b)
    except (TypeError, ValueError):
        return None


def compute_condition_summary(
    condition: str, seed_summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate ``seed_summaries`` as mean / std / worst across the 5 seeds.

    Per Reviewer point: 5 seeds MUST NOT be pooled into more independent
    samples. They are reported as the three statistics below and ``n_seeds``
    always equals the number of input summaries.
    """
    if len(seed_summaries) != len(FROZEN_SEEDS):
        raise ValueError(
            f"condition {condition!r}: expected {len(FROZEN_SEEDS)} seeds, "
            f"got {len(seed_summaries)}."
        )
    seeds_seen = sorted(int(summary["seed"]) for summary in seed_summaries)
    if seeds_seen != list(FROZEN_SEEDS):
        raise ValueError(
            f"condition {condition!r}: seed set drift: expected {FROZEN_SEEDS}, got {seeds_seen}."
        )

    def _stats(field: str) -> dict[str, float]:
        values = []
        for summary in seed_summaries:
            delta = summary.get("delta_vs_clean", {})
            value = delta.get(field)
            if value is None:
                continue
            values.append(float(value))
        if not values:
            return _jsonify({"mean": None, "std": None, "worst": None, "best": None})
        array = np.asarray(values, dtype=np.float64)
        return _jsonify({
            "mean": float(array.mean()),
            "std": float(array.std()),
            "worst": float(array.min()),
            "best": float(array.max()),
        })

    def _stats_direct(field: str) -> dict[str, float]:
        values = []
        for summary in seed_summaries:
            record_metrics = summary.get("record_metrics", {})
            value = record_metrics.get(field)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        if not values:
            return _jsonify({"mean": None, "std": None, "worst": None, "best": None})
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return _jsonify({"mean": None, "std": None, "worst": None, "best": None})
        return _jsonify({
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "worst": float(finite.min()),
            "best": float(finite.max()),
        })

    def _stats_p6(field: str) -> dict[str, float]:
        values = []
        for summary in seed_summaries:
            p6_block = summary.get("p6_single_rule", {})
            value = p6_block.get(field)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        if not values:
            return _jsonify({"mean": None, "std": None, "worst": None, "best": None})
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return _jsonify({"mean": None, "std": None, "worst": None, "best": None})
        return _jsonify({
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "worst": float(finite.min()),
            "best": float(finite.max()),
        })

    return _jsonify({
        "condition": condition,
        "n_seeds": len(seed_summaries),
        "seeds": seeds_seen,
        "record_metrics_stitched_means": {
            "accuracy": _stats_direct("accuracy"),
            "balanced_accuracy": _stats_direct("balanced_accuracy"),
            "macro_f1": _stats_direct("macro_f1"),
        },
        "delta_macro_f1": _stats("record_macro_f1"),
        "delta_balanced_accuracy": _stats("record_balanced_accuracy"),
        "delta_accuracy": _stats("record_accuracy"),
        "delta_vs_clean": {
            "record_macro_f1": _stats("record_macro_f1"),
            "record_balanced_accuracy": _stats("record_balanced_accuracy"),
            "record_accuracy": _stats("record_accuracy"),
        },
        "p6_single_rule_means": {
            "coverage": _stats_p6("coverage"),
            "accepted_accuracy": _stats_p6("accepted_accuracy"),
            "wrong_action_rate": _stats_p6("wrong_action_rate"),
        },
        "seed_summaries": seed_summaries,
    })


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def analyze_p7_full(
    evidence_path: Path,
    output_dir: Path,
    *,
    expected_archive_sha256: str | None = None,
) -> EvidenceManifest:
    """Run the full P7 Full re-verification pipeline and write all artifacts.

    All rule parameters (P6 threshold, P6.1 temperature / threshold /
    require_unanimous) are loaded from the archive's pinned rule block and
    verified against the module-pinned frozen constants; callers cannot
    override them via this function.

    Returns the :class:`EvidenceManifest` used for the run so callers can
    inspect the per-file SHA-256 trail.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = verify_evidence_archive(
        Path(evidence_path),
        expected_sha256=expected_archive_sha256,
    )
    p6_rule = load_p6_single_rule_from_archive(manifest.evidence_root)
    p6_1_rule = load_p6_1_ensemble_rule_from_archive(manifest.evidence_root)
    assert_rule_values_match_frozen(p6_rule, p6_1_rule)

    clean_stitched = load_clean_oof(manifest.evidence_root)
    clean_metrics = compute_stitched_classification_metrics(clean_stitched)
    clean_p6_single = compute_p6_single(clean_stitched)
    clean_p6_1 = compute_p6_1_ensemble(clean_stitched)
    clean_per_class = compute_per_class_breakdown(clean_stitched)
    clean_per_subject = compute_per_subject_breakdown(clean_stitched)
    clean_worst_subjects = compute_worst_subjects(clean_per_subject)

    condition_summaries: dict[str, dict[str, Any]] = {}
    per_class_accumulator: list[pd.DataFrame] = []
    per_subject_accumulator: list[pd.DataFrame] = []
    worst_subjects_rows: list[dict[str, Any]] = []
    error_cases_accumulator: list[dict[str, Any]] = []
    high_confidence_accumulator: list[dict[str, Any]] = []

    for condition in FROZEN_CONDITION_NAMES:
        seed_summaries: list[dict[str, Any]] = []
        for seed in FROZEN_SEEDS:
            perturbed = load_condition_seed_oof(manifest.evidence_root, condition, seed)
            seed_summary = compute_seed_summary(
                condition=condition,
                seed=seed,
                clean_stitched=clean_stitched,
                perturbed_stitched=perturbed,
            )
            seed_summaries.append(seed_summary)
            for record in seed_summary["per_class"]:
                per_class_accumulator.append({"condition": condition, "seed": seed, **record})
            for record in seed_summary["per_subject"]:
                per_subject_accumulator.append({"condition": condition, "seed": seed, **record})
            worst_subjects_rows.append({
                "condition": condition,
                "seed": seed,
                "by_wrong_action_rate_subject_id": (
                    (seed_summary["worst_subjects"]["by_wrong_action_rate"] or {}).get("subject_id")
                ),
                "by_coverage_subject_id": (
                    (seed_summary["worst_subjects"]["by_coverage"] or {}).get("subject_id")
                ),
                "by_accepted_accuracy_subject_id": (
                    (seed_summary["worst_subjects"]["by_accepted_accuracy"] or {}).get("subject_id")
                ),
                "by_raw_accuracy_subject_id": (
                    (seed_summary["worst_subjects"]["by_raw_accuracy"] or {}).get("subject_id")
                ),
            })
            for record in seed_summary["error_cases"]:
                error_cases_accumulator.append({"condition": condition, "seed": seed, **record})
                if float(record.get("top1_probability", 0.0)) >= FROZEN_HIGH_CONFIDENCE_THRESHOLD:
                    high_confidence_accumulator.append(
                        {"condition": condition, "seed": seed, **record}
                    )

        condition_summaries[condition] = compute_condition_summary(condition, seed_summaries)

    # Add clean baseline to the worst-subjects accumulator so reviewers can
    # compare the clean against the perturbed worst subjects in one CSV.
    worst_subjects_rows.insert(0, {
        "condition": "clean",
        "seed": -1,
        **{
            "by_wrong_action_rate_subject_id": (
                (clean_worst_subjects["by_wrong_action_rate"] or {}).get("subject_id")
            ),
            "by_coverage_subject_id": (
                (clean_worst_subjects["by_coverage"] or {}).get("subject_id")
            ),
            "by_accepted_accuracy_subject_id": (
                (clean_worst_subjects["by_accepted_accuracy"] or {}).get("subject_id")
            ),
            "by_raw_accuracy_subject_id": (
                (clean_worst_subjects["by_raw_accuracy"] or {}).get("subject_id")
            ),
        },
    })

    summary = _jsonify({
        "schema_version": SCHEMA_VERSION,
        "evidence_manifest": manifest.as_dict(),
        "pinned_rules": {
            "p6_single_rule": p6_rule.as_dict(),
            "p6_1_ensemble_rule": p6_1_rule.as_dict(),
        },
        "clean": {
            "n_records_total": int(len(clean_stitched)),
            "stitched_metrics": clean_metrics,
            "p6_single_rule": clean_p6_single,
            "p6_1_ensemble_rule": clean_p6_1,
            "per_class": clean_per_class.to_dict(orient="records"),
            "per_subject": clean_per_subject.to_dict(orient="records"),
            "worst_subjects": clean_worst_subjects,
        },
        "conditions": condition_summaries,
        "frozen_contract": {
            "p6_single_threshold": float(FROZEN_P6_SINGLE_THRESHOLD),
            "p6_1_temperature": float(FROZEN_P6_1_TEMPERATURE),
            "p6_1_threshold": float(FROZEN_P6_1_THRESHOLD),
            "p6_1_require_unanimous": bool(FROZEN_P6_1_REQUIRE_UNANIMOUS),
            "high_confidence_threshold": float(FROZEN_HIGH_CONFIDENCE_THRESHOLD),
            "seeds": list(FROZEN_SEEDS),
            "conditions": list(FROZEN_CONDITION_NAMES),
        },
    })

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(per_class_accumulator).to_csv(
        output_dir / "per_class_metrics.csv", index=False
    )
    pd.DataFrame(per_subject_accumulator).to_csv(
        output_dir / "per_subject_metrics.csv", index=False
    )
    pd.DataFrame(worst_subjects_rows).to_csv(
        output_dir / "worst_subjects.csv", index=False
    )
    pd.DataFrame(error_cases_accumulator).to_csv(
        output_dir / "error_cases.csv", index=False
    )
    pd.DataFrame(high_confidence_accumulator).to_csv(
        output_dir / "high_confidence_errors.csv", index=False
    )

    condition_metrics_rows: list[dict[str, Any]] = []
    for condition, summary_block in condition_summaries.items():
        condition_metrics_rows.append({
            "condition": condition,
            "n_seeds": summary_block["n_seeds"],
            "macro_f1_mean": summary_block["record_metrics_stitched_means"]["macro_f1"]["mean"],
            "macro_f1_std": summary_block["record_metrics_stitched_means"]["macro_f1"]["std"],
            "macro_f1_worst": summary_block["record_metrics_stitched_means"]["macro_f1"]["worst"],
            "balanced_accuracy_mean": summary_block["record_metrics_stitched_means"]["balanced_accuracy"]["mean"],
            "balanced_accuracy_std": summary_block["record_metrics_stitched_means"]["balanced_accuracy"]["std"],
            "accuracy_mean": summary_block["record_metrics_stitched_means"]["accuracy"]["mean"],
            "delta_macro_f1_mean": summary_block["delta_vs_clean"]["record_macro_f1"]["mean"],
            "delta_macro_f1_worst": summary_block["delta_vs_clean"]["record_macro_f1"]["worst"],
            "p6_coverage_mean": summary_block["p6_single_rule_means"]["coverage"]["mean"],
            "p6_coverage_worst": summary_block["p6_single_rule_means"]["coverage"]["worst"],
            "p6_accepted_accuracy_mean": summary_block["p6_single_rule_means"]["accepted_accuracy"]["mean"],
            "p6_wrong_action_rate_mean": summary_block["p6_single_rule_means"]["wrong_action_rate"]["mean"],
            "p6_wrong_action_rate_worst": summary_block["p6_single_rule_means"]["wrong_action_rate"]["worst"],
        })
    pd.DataFrame(condition_metrics_rows).to_csv(
        output_dir / "condition_metrics.csv", index=False
    )

    (output_dir / "evidence_manifest.json").write_text(
        json.dumps(manifest.as_dict(), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    return manifest


# ---------------------------------------------------------------------------
# JSON-safe serialization (non-finite → null)
# ---------------------------------------------------------------------------


def _jsonify(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` so the JSON output is strict."""
    if isinstance(obj, dict):
        return {key: _jsonify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(item) for item in obj]
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not np.isfinite(value) else float(value)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    return obj


def _jsonify_list(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_jsonify(item) for item in items]